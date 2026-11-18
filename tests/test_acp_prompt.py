"""ACP-сервер: session/prompt и session/cancel.

События агента переводятся в уведомления ``session/update`` (``AcpEvents``): текст модели —
``agent_message_chunk``, вызовы инструментов — ``tool_call``/``tool_call_update`` с одним
``toolCallId`` на вызов, итог хода — ``usage_update``. Все уведомления хода приходят до
ответа на ``session/prompt`` (``stopReason``). Отмена хода — уведомление ``session/cancel``.
"""

from __future__ import annotations

import os
import signal
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, closing, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, override

import pytest
from acp_client import TIMEOUT
from acp_harness import AcpHarness, ProviderFactory, acp_environment
from fakes import ScriptedProvider, text_turn, tool_turn, tools_turn
from pydantic import BaseModel

from gigacrew.agent.context_window import DEFAULT_CONTEXT_WINDOW
from gigacrew.cli.acp.rpc import INTERNAL_ERROR, INVALID_PARAMS, INVALID_REQUEST
from gigacrew.llm.base import LLMError
from gigacrew.llm.types import AssistantTurn, Message, ToolSpec, Usage
from gigacrew.tools.base import Tool, ToolContext, ToolRegistry, ToolResult, build_default_registry

# Исключение в любом потоке сервера роняет тест, а не теряется.
pytestmark = pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")

PROMPT = "session/prompt"
CANCEL = "session/cancel"
UPDATE = "session/update"
TOOL_CALL = "tool_call"
TOOL_CALL_UPDATE = "tool_call_update"
MESSAGE_CHUNK = "agent_message_chunk"
THOUGHT_CHUNK = "agent_thought_chunk"
END_TURN = "end_turn"
CANCELLED = "cancelled"
NOTES = "notes.txt"
NOTES_TEXT = "hello from notes\n"
# Сколько поддельный провайдер или инструмент ждёт отмены, прежде чем сдаться, с: дольше
# ожидания клиента — тест падает на клиенте, а не на зависшем потоке.
HOLD_LIMIT = TIMEOUT * 3
# Отрезок ожидания: между отрезками поток видит асинхронный KeyboardInterrupt.
HOLD_TICK = 0.02

type Update = dict[str, Any]


def _hold(release: threading.Event) -> None:
    """Ждёт ``release`` короткими отрезками (как «зависший» запрос или инструмент)."""
    deadline = time.monotonic() + HOLD_LIMIT
    while not release.wait(HOLD_TICK):
        if time.monotonic() > deadline:
            raise AssertionError("ход не отменён и не отпущен")


@dataclass
class _Gate:
    """Задержка внутри хода: ``entered`` — ход дошёл до неё, ``release`` — отпустить."""

    entered: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)

    def pass_through(self) -> None:
        self.entered.set()
        _hold(self.release)

    def pass_stuck(self) -> None:
        """Как :meth:`pass_through`, но одним ожиданием: асинхронное исключение не дойдёт."""
        self.entered.set()
        if not self.release.wait(HOLD_LIMIT):
            raise AssertionError("задержку не отпустили")

    def reached(self) -> None:
        assert self.entered.wait(TIMEOUT), "ход не дошёл до задержки"


def _new_gate(gates: list[_Gate]) -> _Gate:
    """Задержка, которую фикстура ``acp`` отпустит до остановки сервера."""
    created = _Gate()
    gates.append(created)
    return created


def _release_all(gates: list[_Gate]) -> None:
    for each_gate in gates:
        each_gate.release.set()


class _GatedProvider(ScriptedProvider):
    """Первое обращение хода «висит» на задержке, пока его не отпустят или не отменят."""

    def __init__(self, gate: _Gate, turns: Sequence[AssistantTurn | BaseException] = ()) -> None:
        super().__init__(turns)
        self.gate = gate
        self._holds = 1

    @override
    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> AssistantTurn:
        if tools is not None and self._holds > 0:
            self._holds -= 1
            self.gate.pass_through()
        return super().complete(messages, tools, **kwargs)


# Фабрика провайдеров с задержкой (фикстура ``gated``).
type _MakeGated = Callable[[Sequence[AssistantTurn | BaseException]], _GatedProvider]


class _StuckProvider(ScriptedProvider):
    """Первое обращение ждёт задержку без отрезков (как httpx до первого байта ответа).

    Асинхронный KeyboardInterrupt до такого потока не доходит; отпущенное обращение
    возвращает «поздний» ответ мимо сценария. Обращение, пришедшее, пока первое ещё висит
    (два хода одного агента параллельно), — ошибка теста.
    """

    def __init__(self, gate: _Gate, turns: Sequence[AssistantTurn | BaseException] = ()) -> None:
        super().__init__(turns)
        self.gate = gate
        self._stuck = True
        self._hanging = False  # зависшее обращение ещё идёт

    @override
    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> AssistantTurn:
        if self._hanging:
            raise AssertionError("обращение к модели, пока висит обращение отменённого хода")
        if tools is not None and self._stuck:
            self._stuck = False
            self._hanging = True
            try:
                self.gate.pass_stuck()
            finally:
                self._hanging = False
            return text_turn("Поздний ответ")
        return super().complete(messages, tools, **kwargs)


class _ChunkedProvider(ScriptedProvider):
    """Потоковый ответ из нескольких кусков ``pieces``."""

    def __init__(self, pieces: Sequence[str]) -> None:
        super().__init__([text_turn("".join(pieces))])
        self._pieces = list(pieces)

    @override
    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        *,
        on_delta: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> AssistantTurn:
        turn = super().stream(messages, tools, **kwargs)
        assert on_delta is not None
        for piece in self._pieces:
            on_delta(piece)
        return turn


class _NoticeProvider(ScriptedProvider):
    """Перед каждым ответом сообщает предупреждение (как пауза по лимиту запросов)."""

    @override
    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> AssistantTurn:
        kwargs["on_notice"]("пауза: лимит запросов")
        return super().complete(messages, tools, **kwargs)


class _EchoParams(BaseModel):
    text: str = ""


class _PlainTool(Tool[_EchoParams]):
    """Неизменяющий инструмент вида «прочее»: выполняет ``action``."""

    description = "тестовый инструмент"
    # Имя схемы параметров задано базовым классом Tool.
    Params = _EchoParams  # noqa: WPS115

    def __init__(self, name: str, action: Callable[[], ToolResult]) -> None:
        self.name = name
        self._action = action

    @override
    def run(self, params: _EchoParams, ctx: ToolContext) -> ToolResult:
        return self._action()


def _interrupt() -> ToolResult:
    raise KeyboardInterrupt


@pytest.fixture
def gates() -> list[_Gate]:
    """Все задержки теста: фикстура ``acp`` отпускает их до остановки сервера."""
    return []


@pytest.fixture
def gate(gates: list[_Gate]) -> _Gate:
    """Задержка инструмента ``wait_here``."""
    return _new_gate(gates)


@pytest.fixture
def gated(gates: list[_Gate]) -> _MakeGated:
    """Фабрика провайдеров с задержкой первого обращения (задержки — в ``gates``)."""

    def make(turns: Sequence[AssistantTurn | BaseException]) -> _GatedProvider:
        return _GatedProvider(_new_gate(gates), turns)

    return make


@pytest.fixture
def factory() -> ProviderFactory:
    return ProviderFactory()


@pytest.fixture
def acp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
    factory: ProviderFactory,
    gate: _Gate,
    gates: list[_Gate],
) -> Iterator[AcpHarness]:
    """Сервер со штатными инструментами и тестовыми ``echo_back``/``wait_here``/``interrupt``.

    Задержки теста отпускаются до остановки сервера: упавший тест не оставляет висящих
    потоков, которые сорвали бы чужие тесты.
    """

    def registry_factory() -> ToolRegistry:
        registry = build_default_registry()
        registry.register(_PlainTool("echo_back", lambda: ToolResult(content="эхо", summary="эхо")))
        registry.register(_PlainTool("wait_here", _wait_tool(gate)))
        registry.register(_PlainTool("interrupt", _interrupt))
        return registry

    acp_environment(monkeypatch, tmp_path_factory.mktemp("process-cwd"))
    harness = AcpHarness(factory, registry_factory=registry_factory)
    with ExitStack() as teardown:
        teardown.enter_context(closing(harness))
        teardown.callback(_release_all, gates)  # раньше closing: обратный порядок
        harness.initialize()
        yield harness


def _wait_tool(gate: _Gate) -> Callable[[], ToolResult]:
    def wait() -> ToolResult:
        gate.pass_through()
        return ToolResult(content="дождался", summary="дождался")

    return wait


@pytest.fixture
def root(project: Path) -> Path:
    """Проект сессии с файлом ``notes.txt``."""
    (project / NOTES).write_text(NOTES_TEXT, encoding="utf-8")
    return project.resolve()


def _open(acp: AcpHarness, cwd: Path, provider: ScriptedProvider | None = None) -> str:
    """Сессия в ``cwd`` (с провайдером ``provider``, если задан)."""
    if provider is not None:
        acp.factory.add(provider)
    return acp.new_session(cwd)


def _text(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _prompt_params(session_id: str, *blocks: dict[str, Any]) -> dict[str, Any]:
    return {"sessionId": session_id, "prompt": list(blocks)}


def _ask(acp: AcpHarness, session_id: str, text: str = "сделай") -> Any:
    """Ход целиком: ``stopReason`` ответа на ``session/prompt`` (ошибка JSON-RPC роняет тест)."""
    return acp.call(PROMPT, _prompt_params(session_id, _text(text)))["stopReason"]


def _begin(acp: AcpHarness, session_id: str, text: str = "сделай") -> int:
    return acp.begin(PROMPT, _prompt_params(session_id, _text(text)))


def _stop_reason(acp: AcpHarness, msg_id: int) -> Any:
    reply = acp.reply(msg_id)
    assert "error" not in reply, reply
    return reply["result"]["stopReason"]


def _updates(acp: AcpHarness, session_id: str) -> list[Update]:
    """Забирает накопленные ``session/update``; все — этой сессии."""
    notifications = acp.client.take_notifications(UPDATE)
    assert all(params["sessionId"] == session_id for params in notifications), notifications
    return [params["update"] for params in notifications]


def _chunks(updates: list[Update], kind: str = MESSAGE_CHUNK) -> list[str]:
    """Тексты кусков вида ``kind`` по порядку."""
    texts = []
    for update in updates:
        if update["sessionUpdate"] == kind:
            assert update["content"]["type"] == "text"
            texts.append(update["content"]["text"])
    return texts


def _tool_updates(updates: list[Update]) -> list[Update]:
    return [
        update for update in updates if update["sessionUpdate"] in {TOOL_CALL, TOOL_CALL_UPDATE}
    ]


def _by_call(updates: list[Update]) -> dict[str, list[Update]]:
    """Уведомления об инструментах по ``toolCallId``; первое у каждого вызова — ``tool_call``."""
    grouped: dict[str, list[Update]] = {}
    for update in _tool_updates(updates):
        grouped.setdefault(update["toolCallId"], []).append(update)
    for call_updates in grouped.values():
        assert call_updates[0]["sessionUpdate"] == TOOL_CALL
        assert all(update["sessionUpdate"] == TOOL_CALL_UPDATE for update in call_updates[1:])
    return grouped


def _single_call(updates: list[Update]) -> list[Update]:
    grouped = _by_call(updates)
    assert len(grouped) == 1, grouped
    return next(iter(grouped.values()))


def _statuses(call_updates: list[Update]) -> list[str]:
    return [update["status"] for update in call_updates if "status" in update]


def _texts_of(call_updates: list[Update]) -> str:
    """Весь текст из ``content`` уведомлений вызова (блоки ``content`` с текстом)."""
    texts = [
        part["content"]["text"]
        for update in call_updates
        for part in update.get("content") or ()
        if part["type"] == "content" and part["content"]["type"] == "text"
    ]
    return "\n".join(texts)


def _user_text(provider: ScriptedProvider) -> str:
    """Запрос пользователя, каким его получила модель в первом обращении хода."""
    messages = provider.requests[0]["messages"]
    user_messages = [message for message in messages if message.role == "user"]
    content: str = user_messages[-1].content
    return content


def _wait_in_progress(acp: AcpHarness) -> None:
    """Ждёт уведомления «вызов выполняется»; оно и прочитанное по пути остаются в ``backlog``."""
    started = acp.client.wait_for(
        lambda message: message.get("params", {}).get("update", {}).get("status") == "in_progress",
    )
    acp.client.backlog.append(started)


def _eventually(check: Callable[[], bool], what: str) -> None:
    """Короткий опрос с пределом: ``check`` станет истинным (гонка на выход процесса)."""
    deadline = time.monotonic() + TIMEOUT
    tick = threading.Event()
    while not check():
        assert time.monotonic() < deadline, what
        tick.wait(HOLD_TICK)


def _sleeper_in(cwd: Path) -> int | None:
    """Процесс ``sleep``, запущенный в каталоге ``cwd``: его pid (None — такого нет)."""
    found = [
        int(proc.name)
        for proc in Path("/proc").iterdir()
        if proc.name.isdigit() and _runs_sleep_in(proc, cwd)
    ]
    return found[0] if found else None


def _runs_sleep_in(proc: Path, cwd: Path) -> bool:
    try:
        workdir = (proc / "cwd").readlink()
        command = (proc / "cmdline").read_bytes()
    except OSError:  # процесс завершился или чужой
        return False
    return workdir == cwd and command.startswith(b"sleep")


def _gone(pid: int) -> bool:
    """Процесс завершился: его нет или он зомби (состояние Z)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError):
        return True
    # Имя команды в скобках может содержать пробелы: состояние — первое поле после него.
    return stat.rpartition(")")[2].split()[0] == "Z"


def _kill(pid: int) -> None:
    """Добивает процесс, если он ещё жив (уборка после упавшего теста)."""
    with suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)


def _unanswered(messages: Sequence[Message]) -> int:
    """Сколько вызовов инструментов в истории осталось без результата."""
    calls = sum(len(message.function_calls) for message in messages)
    return calls - sum(message.role == "function" for message in messages)


def _assert_server_alive(acp: AcpHarness) -> None:
    """Сервер отвечает на следующий запрос (``authenticate`` → пустой объект)."""
    reply = acp.call("authenticate", {"methodId": "none"})
    assert isinstance(reply, dict)
    assert not reply


def _assert_no_late_updates(acp: AcpHarness) -> None:
    """После ответа на prompt уведомлений хода больше нет (проверка — следующим запросом)."""
    _assert_server_alive(acp)
    assert not acp.client.take_notifications(UPDATE)


# --- запрос пользователя ---


def test_text_prompt_reaches_model_and_ends_turn(acp: AcpHarness, root: Path) -> None:
    provider = ScriptedProvider([text_turn("Готово")])
    session_id = _open(acp, root, provider)

    assert _ask(acp, session_id, "Привет, агент") == END_TURN

    assert "Привет, агент" in _user_text(provider)


def test_resources_are_part_of_user_request_in_order(acp: AcpHarness, root: Path) -> None:
    """Ссылка на ресурс — его адрес в тексте запроса; встроенный ресурс — адрес и текст."""
    provider = ScriptedProvider([text_turn("Готово")])
    session_id = _open(acp, root, provider)
    link = {"type": "resource_link", "uri": (root / NOTES).as_uri(), "name": NOTES}
    embedded = {
        "type": "resource",
        "resource": {
            "uri": (root / "spec.md").as_uri(),
            "text": "ВСТРОЕННЫЙ ТЕКСТ",
            "mimeType": None,
        },
    }

    reply = acp.call(PROMPT, _prompt_params(session_id, _text("Посмотри"), link, embedded))

    assert reply["stopReason"] == END_TURN
    request = _user_text(provider)
    positions = [request.find(fragment) for fragment in ("Посмотри", NOTES, "ВСТРОЕННЫЙ ТЕКСТ")]
    assert -1 not in positions, request
    assert positions == sorted(positions), request
    assert "spec.md" in request


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"prompt": [_text("x")]},
        {"sessionId": "SESSION"},
        {"sessionId": "SESSION", "prompt": []},
        {"sessionId": "SESSION", "prompt": "текст"},
        {"sessionId": "SESSION", "prompt": [{"text": "без типа"}]},
        {"sessionId": "SESSION", "prompt": [{"type": "text"}]},
        {
            "sessionId": "SESSION",
            "prompt": [{"type": "image", "data": "AA==", "mimeType": "image/png"}],
        },
        {"sessionId": "no-such-session", "prompt": [_text("x")]},
    ],
    ids=[
        "no-params",
        "no-session",
        "no-prompt",
        "empty-prompt",
        "prompt-not-array",
        "block-without-type",
        "text-without-text",
        "image-unsupported",
        "unknown-session",
    ],
)
def test_bad_prompt_is_invalid_params_and_server_survives(
    acp: AcpHarness,
    root: Path,
    params: dict[str, Any],
) -> None:
    provider = ScriptedProvider([text_turn("Готово")])
    session_id = _open(acp, root, provider)
    if params.get("sessionId") == "SESSION":
        params = {**params, "sessionId": session_id}

    assert acp.call_error(PROMPT, params)["code"] == INVALID_PARAMS

    assert not provider.requests  # до модели неверный запрос не дошёл
    assert _ask(acp, session_id) == END_TURN


# --- текст модели ---


def test_stream_deltas_are_message_chunks_in_order_before_reply(
    acp: AcpHarness,
    root: Path,
) -> None:
    pieces = ["При", "вет, ", "мир!"]
    session_id = _open(acp, root, _ChunkedProvider(pieces))

    assert _ask(acp, session_id) == END_TURN

    assert _chunks(_updates(acp, session_id)) == pieces
    _assert_no_late_updates(acp)


def test_non_streamed_answer_is_one_chunk(
    acp: AcpHarness,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GIGACREW_STREAM", "0")
    session_id = _open(acp, root, ScriptedProvider([text_turn("Ответ целиком")]))

    assert _ask(acp, session_id) == END_TURN

    assert _chunks(_updates(acp, session_id)) == ["Ответ целиком"]


def _reasoning_turn() -> AssistantTurn:
    message = Message(role="assistant", content="Ответ", reasoning="Сначала подумаю")
    return AssistantTurn(message=message, usage=Usage())


def test_reasoning_is_thought_chunk_before_answer(
    acp: AcpHarness,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GIGACREW_STREAM", "0")
    session_id = _open(acp, root, ScriptedProvider([_reasoning_turn()]))

    assert _ask(acp, session_id) == END_TURN

    updates = _updates(acp, session_id)
    texts = [
        (update["sessionUpdate"], update["content"]["text"])
        for update in updates
        if update["sessionUpdate"] in {THOUGHT_CHUNK, MESSAGE_CHUNK}
    ]
    assert texts == [(THOUGHT_CHUNK, "Сначала подумаю"), (MESSAGE_CHUNK, "Ответ")]


def test_streamed_answer_reasoning_is_one_thought_chunk(acp: AcpHarness, root: Path) -> None:
    session_id = _open(acp, root, ScriptedProvider([_reasoning_turn()]))

    assert _ask(acp, session_id) == END_TURN

    updates = _updates(acp, session_id)
    assert _chunks(updates, THOUGHT_CHUNK) == ["Сначала подумаю"]
    assert _chunks(updates) == ["Ответ"]


def test_notices_do_not_break_protocol(
    acp: AcpHarness,
    root: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Предупреждения агента не попадают в stdout процесса и не ломают сообщения."""
    provider = _NoticeProvider([text_turn("Готово")])
    session_id = _open(acp, root, provider)

    assert _ask(acp, session_id) == END_TURN

    assert "Готово" in "".join(_chunks(_updates(acp, session_id)))
    assert not capfd.readouterr().out


# --- инструменты ---


def test_read_tool_call_lifecycle(acp: AcpHarness, root: Path) -> None:
    provider = ScriptedProvider([tool_turn("read_file", {"path": NOTES}), text_turn("Прочитал")])
    session_id = _open(acp, root, provider)

    assert _ask(acp, session_id) == END_TURN

    updates = _updates(acp, session_id)
    call_updates = _single_call(updates)
    call = call_updates[0]
    assert NOTES in call["title"]
    assert call["kind"] == "read"
    assert call["status"] == "pending"
    assert call["rawInput"] == {"path": NOTES}
    assert call["locations"] == [{"path": str(root / NOTES)}]
    assert _statuses(call_updates) == ["pending", "in_progress", "completed"]
    assert "hello from notes" in _texts_of(call_updates)
    assert _chunks(updates) == ["Прочитал"]


@pytest.mark.parametrize(
    ("name", "args", "kind"),
    [
        ("read_file", {"path": NOTES}, "read"),
        ("search_content", {"pattern": "hello"}, "search"),
        ("find_files", {"pattern": "*.txt"}, "search"),
        ("find_symbol", {"query": "main"}, "search"),
        ("find_references", {"query": "main"}, "search"),
        ("edit_file", {"path": NOTES, "old_string": "hello", "new_string": "bye"}, "edit"),
        ("run_shell", {"command": "pwd"}, "execute"),
        ("git", {"subcommand": "status"}, "execute"),
        ("echo_back", {"text": "x"}, "other"),
    ],
)
def test_tool_kind_follows_tool_type(
    acp: AcpHarness,
    root: Path,
    name: str,
    args: dict[str, Any],
    kind: str,
) -> None:
    session_id = _open(acp, root, ScriptedProvider([tool_turn(name, args)]))

    assert _ask(acp, session_id) == END_TURN

    call_updates = _single_call(_updates(acp, session_id))
    assert call_updates[0]["kind"] == kind
    assert call_updates[0]["rawInput"] == args
    # Итог зависит от окружения (git-репозитория и LSP в проекте нет) — важно, что он есть.
    assert _statuses(call_updates)[-1] in {"completed", "failed"}


def test_edit_reports_diff_and_location(acp: AcpHarness, root: Path) -> None:
    edit = {"path": NOTES, "old_string": "hello", "new_string": "goodbye"}
    session_id = _open(acp, root, ScriptedProvider([tool_turn("edit_file", edit)]))

    assert _ask(acp, session_id) == END_TURN

    call_updates = _single_call(_updates(acp, session_id))
    assert call_updates[0]["locations"] == [{"path": str(root / NOTES)}]
    diffs = [
        part
        for update in call_updates
        for part in update.get("content") or ()
        if part["type"] == "diff"
    ]
    assert len(diffs) == 1, call_updates
    assert diffs[0]["path"] == str(root / NOTES)
    assert "hello" in diffs[0]["oldText"]
    assert "goodbye" in diffs[0]["newText"]
    assert "hello" not in diffs[0]["newText"]
    assert (root / NOTES).read_text(encoding="utf-8") == "goodbye from notes\n"


@pytest.mark.parametrize(
    ("name", "args", "message"),
    [
        ("read_file", {"path": "missing.txt"}, "missing.txt"),
        ("no_such_tool", {}, "no_such_tool"),
    ],
    ids=["tool-error", "unknown-tool"],
)
def test_failed_call_ends_failed_with_error_text(
    acp: AcpHarness,
    root: Path,
    name: str,
    args: dict[str, Any],
    message: str,
) -> None:
    session_id = _open(acp, root, ScriptedProvider([tool_turn(name, args)]))

    assert _ask(acp, session_id) == END_TURN

    call_updates = _single_call(_updates(acp, session_id))
    assert _statuses(call_updates)[0] == "pending"
    assert _statuses(call_updates)[-1] == "failed"
    assert message in _texts_of(call_updates)


def test_tool_call_id_is_model_call_id_or_unique_generated(acp: AcpHarness, root: Path) -> None:
    """Id вызова от модели (c1, c2) — это toolCallId; вызов без id получает уникальный."""
    read = ("read_file", {"path": NOTES})
    provider = ScriptedProvider(
        [tools_turn([read, read]), text_turn("1"), tool_turn(*read), tool_turn(*read)],
    )
    session_id = _open(acp, root, provider)

    assert _ask(acp, session_id) == END_TURN
    first_turn = _by_call(_updates(acp, session_id))
    assert _ask(acp, session_id) == END_TURN
    second_turn = _by_call(_updates(acp, session_id))

    assert list(first_turn) == ["c1", "c2"]
    assert len(second_turn) == 2
    assert not first_turn.keys() & second_turn.keys()
    for call_updates in (*first_turn.values(), *second_turn.values()):
        assert _statuses(call_updates) == ["pending", "in_progress", "completed"]


# --- конец хода ---


def test_step_limit_is_max_turn_requests(
    acp: AcpHarness,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GIGACREW_MAX_STEPS", "1")
    read = tool_turn("read_file", {"path": NOTES})
    session_id = _open(acp, root, ScriptedProvider([read, read, read]))

    assert _ask(acp, session_id) == "max_turn_requests"


def test_usage_update_ends_turn(acp: AcpHarness, root: Path) -> None:
    usage = Usage(prompt_tokens=20_000, completion_tokens=50, total_tokens=20_050)
    session_id = _open(acp, root, ScriptedProvider([text_turn("Готово", usage)]))

    assert _ask(acp, session_id) == END_TURN

    usage_update = _updates(acp, session_id)[-1]
    agent = acp.server.session(session_id).agent
    assert usage_update["sessionUpdate"] == "usage_update"
    assert usage_update["used"] == 20_050  # запрос и ответ последнего обращения
    assert usage_update["size"] == (agent.context_window or DEFAULT_CONTEXT_WINDOW)


@pytest.mark.parametrize(
    "failure",
    [LLMError("шлюз недоступен"), ValueError("непредвиденный сбой")],
    ids=["model-error", "unexpected"],
)
def test_turn_error_is_rpc_error_and_session_recovers(
    acp: AcpHarness,
    root: Path,
    failure: Exception,
) -> None:
    provider = ScriptedProvider([failure, text_turn("Снова на связи")])
    session_id = _open(acp, root, provider)

    error = acp.call_error(PROMPT, _prompt_params(session_id, _text("раз")))

    assert error["code"] == INTERNAL_ERROR
    assert str(failure) in error["message"]
    assert "Traceback" not in error["message"]
    acp.client.take_notifications(UPDATE)
    assert _ask(acp, session_id, "два") == END_TURN
    assert _chunks(_updates(acp, session_id)) == ["Снова на связи"]


# --- отмена и параллельность ---


def test_cancel_while_model_waits(
    acp: AcpHarness,
    root: Path,
    gated: _MakeGated,
) -> None:
    # Отменённое обращение не получило ответа: «Второй ход» достаётся следующему prompt.
    provider = gated([text_turn("Второй ход")])
    session_id = _open(acp, root, provider)
    msg_id = _begin(acp, session_id)
    provider.gate.reached()

    acp.client.notify(CANCEL, {"sessionId": session_id})

    assert _stop_reason(acp, msg_id) == CANCELLED
    acp.client.take_notifications(UPDATE)
    assert _ask(acp, session_id) == END_TURN
    assert _chunks(_updates(acp, session_id)) == ["Второй ход"]


def test_cancel_while_tool_runs_closes_the_call(
    acp: AcpHarness,
    root: Path,
    gate: _Gate,
) -> None:
    """Прерванный вызов не остаётся «выполняется»: последний статус — failed.

    Сервер сам закрывает открытые вызовы при отмене: failed приходит до ответа cancelled.
    """
    provider = ScriptedProvider([tool_turn("wait_here", {}), text_turn("Второй ход")])
    session_id = _open(acp, root, provider)
    msg_id = _begin(acp, session_id)
    gate.reached()

    acp.client.notify(CANCEL, {"sessionId": session_id})

    assert _stop_reason(acp, msg_id) == CANCELLED
    call_updates = _single_call(_updates(acp, session_id))
    assert _statuses(call_updates)[-1] == "failed"
    assert _ask(acp, session_id) == END_TURN
    # Прерванный вызов остался в истории с результатом: модель не получит вызов без ответа.
    messages = provider.requests[-1]["messages"]
    assert any(call.name == "wait_here" for message in messages for call in message.function_calls)
    assert _unanswered(messages) == 0


@pytest.mark.skipif(not Path("/proc/self/cwd").exists(), reason="процессы ищутся через /proc")
def test_cancel_kills_running_command(
    acp: AcpHarness,
    root: Path,
    request: pytest.FixtureRequest,
) -> None:
    """Команда, запущенная ходом, убивается отменой, а не дорабатывает до конца."""
    command = tool_turn("run_shell", {"command": "exec sleep 30"})
    session_id = _open(acp, root, ScriptedProvider([command, text_turn("Второй ход")]))
    msg_id = _begin(acp, session_id)
    _wait_in_progress(acp)
    _eventually(lambda: _sleeper_in(root) is not None, "команда не запустилась")
    pid = _sleeper_in(root)
    assert pid is not None
    # Добиваем, только если это всё ещё наша команда: pid мог достаться чужому процессу.
    request.addfinalizer(lambda: _kill(pid) if _sleeper_in(root) == pid else None)

    acp.client.notify(CANCEL, {"sessionId": session_id})

    assert _stop_reason(acp, msg_id) == CANCELLED
    _eventually(lambda: _gone(pid), f"процесс команды {pid} жив после отмены")
    acp.client.take_notifications(UPDATE)
    assert _ask(acp, session_id) == END_TURN


@pytest.mark.parametrize(
    "turns",
    [[KeyboardInterrupt()], [tool_turn("interrupt", {})]],
    ids=["in-model-call", "in-tool"],
)
def test_keyboard_interrupt_in_turn_still_answers_prompt(
    acp: AcpHarness,
    root: Path,
    turns: list[AssistantTurn | BaseException],
) -> None:
    """Защитный путь: KeyboardInterrupt в ходе без session/cancel — ответ cancelled.

    Так ведёт себя запоздавшее асинхронное исключение: prompt не остаётся без ответа,
    поток не падает, сессия жива.
    """
    session_id = _open(acp, root, ScriptedProvider([*turns, text_turn("Второй ход")]))

    assert _ask(acp, session_id) == CANCELLED

    acp.client.take_notifications(UPDATE)
    assert _ask(acp, session_id) == END_TURN


def test_repeated_cancel_is_harmless(
    acp: AcpHarness,
    root: Path,
    gated: _MakeGated,
) -> None:
    """Две отмены подряд — один cancelled, без исключений в потоках сервера."""
    provider = gated([text_turn("Второй ход")])
    session_id = _open(acp, root, provider)
    msg_id = _begin(acp, session_id)
    provider.gate.reached()

    acp.client.notify(CANCEL, {"sessionId": session_id})
    acp.client.notify(CANCEL, {"sessionId": session_id})

    assert _stop_reason(acp, msg_id) == CANCELLED
    _assert_server_alive(acp)
    acp.client.take_notifications(UPDATE)
    assert _ask(acp, session_id) == END_TURN
    assert _chunks(_updates(acp, session_id)) == ["Второй ход"]


def test_cancel_does_not_wait_for_uninterruptible_turn(
    acp: AcpHarness,
    root: Path,
    gates: list[_Gate],
) -> None:
    """Поток хода не видит асинхронного исключения (ждёт без отрезков), а отмена работает.

    Ответ cancelled приходит, пока поток ещё ждёт; поздние события отменённого хода клиенту
    не идут; следующий prompt той же сессии дожидается старого потока, а не получает ошибку.
    """
    provider = _StuckProvider(_new_gate(gates), [text_turn("Второй ход")])
    session_id = _open(acp, root, provider)
    first = _begin(acp, session_id)
    provider.gate.reached()

    acp.client.notify(CANCEL, {"sessionId": session_id})

    assert _stop_reason(acp, first) == CANCELLED  # поток хода ещё висит
    second = _begin(acp, session_id, "ещё")
    # Барьер: следующий запрос получил ответ — значит, второй prompt уже у обработчика.
    _assert_server_alive(acp)
    provider.gate.release.set()
    assert _stop_reason(acp, second) == END_TURN
    assert _chunks(_updates(acp, session_id)) == ["Второй ход"]
    _assert_no_late_updates(acp)


def test_cancel_without_turn_or_session_is_silent(acp: AcpHarness, root: Path) -> None:
    session_id = _open(acp, root, ScriptedProvider([text_turn("Готово")]))

    acp.client.notify(CANCEL, {"sessionId": session_id})
    acp.client.notify(CANCEL, {"sessionId": "no-such-session"})
    acp.client.notify(CANCEL, {})

    _assert_server_alive(acp)
    # Ни ответов, ни ошибок на отмену (уведомления о самой сессии не в счёт).
    assert all(message.get("method") == UPDATE for message in acp.client.backlog)
    assert _ask(acp, session_id) == END_TURN


def test_cancel_of_other_session_does_not_touch_turn(
    acp: AcpHarness,
    root: Path,
    gated: _MakeGated,
) -> None:
    provider = gated([text_turn("Дождался")])
    busy = _open(acp, root, provider)
    idle = _open(acp, root)
    msg_id = _begin(acp, busy)
    provider.gate.reached()

    acp.client.notify(CANCEL, {"sessionId": idle})
    provider.gate.release.set()

    assert _stop_reason(acp, msg_id) == END_TURN
    assert _chunks(_updates(acp, busy)) == ["Дождался"]


def test_second_prompt_of_same_session_rejected_during_turn(
    acp: AcpHarness,
    root: Path,
    gated: _MakeGated,
) -> None:
    provider = gated([text_turn("Первый ход")])
    session_id = _open(acp, root, provider)
    msg_id = _begin(acp, session_id)
    provider.gate.reached()

    error = acp.call_error(PROMPT, _prompt_params(session_id, _text("ещё")))
    provider.gate.release.set()

    assert error["code"] == INVALID_REQUEST
    assert _stop_reason(acp, msg_id) == END_TURN
    assert _chunks(_updates(acp, session_id)) == ["Первый ход"]
    assert len(provider.requests) == 1


def test_prompts_of_different_sessions_run_in_parallel(
    acp: AcpHarness,
    root: Path,
    gated: _MakeGated,
) -> None:
    providers = [gated([text_turn("Ответ A")]), gated([text_turn("Ответ B")])]
    sessions = [_open(acp, root, provider) for provider in providers]
    msg_ids = [_begin(acp, session_id) for session_id in sessions]

    for provider in providers:
        provider.gate.reached()  # оба хода внутри обращения к модели одновременно
    for provider in providers:
        provider.gate.release.set()

    assert [_stop_reason(acp, msg_id) for msg_id in msg_ids] == [END_TURN, END_TURN]
    by_session: dict[str, list[str]] = {}
    for params in acp.client.take_notifications(UPDATE):
        by_session.setdefault(params["sessionId"], []).extend(_chunks([params["update"]]))
    assert by_session == {sessions[0]: ["Ответ A"], sessions[1]: ["Ответ B"]}
