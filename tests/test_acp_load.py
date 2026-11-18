"""ACP-сервер: сохранение чата сессии и ``session/load`` — продолжение сохранённого чата.

Id сессии — id чата (:class:`~gigacrew.agent.chat_store.ChatStore` проекта): после каждого
хода ``session/prompt`` диалог сохраняется, как в REPL. ``session/load`` {sessionId, cwd,
mcpServers} — в новом процессе (здесь — новый сервер с тем же проектом и HOME): история
переигрывается уведомлениями ``session/update`` до ответа — ``user_message_chunk``,
``agent_message_chunk``, ``tool_call``/``tool_call_update`` с id вызовов из истории; ответ
(``LoadSessionResponse``: ``modes``, ``configOptions``) — как у ``session/new``, без
``sessionId``.

T3 собирает из переигранного только текст реплик (вызовы инструментов лишь разделяют их) и
начинает новую реплику при смене роли или ``messageId`` куска; одинаковый ``messageId``
склеивает куски в одну реплику. Поэтому у каждого куска есть ``messageId``, свой у каждой
реплики истории. Чат не найден — ошибка -32002 (Resource not found по схеме ACP); T3 на любую
ошибку ``session/load`` при запуске просто открывает новую сессию.

``session/close`` (``sessionCapabilities.close``) T3 вызывает при остановке треда: сессия
закрывается, её провайдер и MCP-серверы — тоже. Сессии, загруженные ради снимка чужого
треда, T3 не закрывает, поэтому загруженная сессия запускает MCP-серверы проекта лениво — при
первом ``session/prompt``.

T3 загружает сессию и в том же процессе (снимок другого треда), поэтому загрузка уже
открытой сессии — не ошибка: та же сессия, её история переигрывается заново.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, closing
from pathlib import Path
from typing import Any, Final, override

import pytest
from acp_harness import AcpHarness, ProviderFactory, acp_environment
from fakes import CatalogueProvider, ScriptedProvider, text_turn, tool_turn, tools_turn

from gigacrew.agent.chat_store import ChatStore
from gigacrew.agent.conversation import (
    BACKGROUND_HEADER,
    HOOK_CONTEXT_HEADER,
    Conversation,
    Summary,
)
from gigacrew.cli.acp.rpc import INTERNAL_ERROR, INVALID_PARAMS, INVALID_REQUEST, RpcError
from gigacrew.cli.mcp import mcp_trust_store
from gigacrew.llm.base import LLMProvider
from gigacrew.llm.types import AssistantTurn, FunctionCall, Message, ToolSpec
from gigacrew.project.mcp_config import load_mcp_settings, servers_fingerprint
from gigacrew.project.workspace import Workspace
from gigacrew.tools.base import ToolRegistry, build_default_registry
from gigacrew.tools.mcp import McpHub

# Исключение в любом потоке сервера роняет тест, а не теряется.
pytestmark = pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")

# Код ошибки ACP «Resource not found» (схема ACP в T3: ErrorCode).
RESOURCE_NOT_FOUND: Final = -32002

NEW = "session/new"
LOAD = "session/load"
CLOSE = "session/close"
PROMPT = "session/prompt"
UPDATE = "session/update"
USER_CHUNK = "user_message_chunk"
MESSAGE_CHUNK = "agent_message_chunk"
TOOL_CALL = "tool_call"
TOOL_CALL_UPDATE = "tool_call_update"
# Виды уведомлений, из которых T3 собирает загруженную историю треда.
HISTORY_KINDS = frozenset((
    USER_CHUNK,
    MESSAGE_CHUNK,
    "agent_thought_chunk",
    TOOL_CALL,
    TOOL_CALL_UPDATE,
))
END_TURN = "end_turn"
NOTES = "notes.txt"
NOTES_TEXT = "hello from notes\n"
CATALOGUE = ("cat-a", "cat-b")
# Инструмент поддельного MCP-сервера проекта (сервер «probe»).
PROBE_ECHO = "mcp__probe__echo"
FAKE_MCP_SERVER = Path(__file__).with_name("fake_mcp_server.py")
# Пишет свой pid в маркер sys.argv[1] и становится MCP-сервером sys.argv[2] (тот же pid).
LAUNCH_SCRIPT = (
    "import os, sys\n"
    "with open(sys.argv[1], 'w') as marker:\n"
    "    marker.write(str(os.getpid()))\n"
    "os.execv(sys.executable, [sys.executable, sys.argv[2]])\n"
)

type Json = dict[str, Any]
# Новый сервер с тем же проектом и HOME — как новый процесс gigacrew, запущенный T3.
type Restart = Callable[[], AcpHarness]


@pytest.fixture
def factory() -> ProviderFactory:
    return ProviderFactory()


@pytest.fixture
def servers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[ExitStack]:
    """Окружение инстанса и стек, который останавливает все серверы теста."""
    acp_environment(monkeypatch, tmp_path_factory.mktemp("process-cwd"))
    with ExitStack() as stack:
        yield stack


@pytest.fixture
def acp(servers: ExitStack, factory: ProviderFactory) -> AcpHarness:
    """Первый сервер (провайдеры — из ``factory``)."""
    return _start(servers, factory)


@pytest.fixture
def restart(servers: ExitStack) -> Restart:
    """Запуск нового сервера со своей фабрикой провайдеров (``harness.factory``)."""
    return lambda: _start(servers, ProviderFactory())


def _start(servers: ExitStack, factory: ProviderFactory) -> AcpHarness:
    harness = servers.enter_context(closing(AcpHarness(factory)))
    harness.initialize()
    return harness


@pytest.fixture
def root(project: Path) -> Path:
    """Проект сессии с файлом ``notes.txt``."""
    (project / NOTES).write_text(NOTES_TEXT, encoding="utf-8")
    return project.resolve()


def _store(cwd: Path) -> ChatStore:
    """Хранилище чатов проекта ``cwd`` (каталог пользователя — HOME теста)."""
    return ChatStore(Workspace(cwd))


def _save_chat(cwd: Path, conversation: Conversation, model: str = "M-default") -> str:
    """Чат, сохранённый раньше (как REPL): id чата."""
    chat_id = ChatStore.new_id()
    _store(cwd).save(chat_id, conversation, model=model)
    return chat_id


def _new(harness: AcpHarness, cwd: Path, provider: LLMProvider | None = None) -> Json:
    """Ответ ``session/new`` в ``cwd`` (с провайдером ``provider``, если задан)."""
    if provider is not None:
        harness.factory.add(provider)
    response: Json = harness.call(NEW, {"cwd": str(cwd), "mcpServers": []})
    return response


def _open(harness: AcpHarness, cwd: Path, provider: LLMProvider | None = None) -> str:
    session_id: str = _new(harness, cwd, provider)["sessionId"]
    return session_id


def _load_params(session_id: str, cwd: Path) -> Json:
    return {"sessionId": session_id, "cwd": str(cwd), "mcpServers": []}


def _load(
    harness: AcpHarness,
    session_id: str,
    cwd: Path,
    provider: LLMProvider | None = None,
) -> Json:
    """Ответ ``session/load``; переигранная история остаётся в ``client.backlog``."""
    if provider is not None:
        harness.factory.add(provider)
    response: Json = harness.call(LOAD, _load_params(session_id, cwd))
    return response


def _ask(harness: AcpHarness, session_id: str, text: str) -> Any:
    """Ход целиком: ``stopReason`` (ошибка JSON-RPC роняет тест)."""
    prompt = [{"type": "text", "text": text}]
    return harness.call(PROMPT, {"sessionId": session_id, "prompt": prompt})["stopReason"]


def _updates(harness: AcpHarness, session_id: str) -> list[Json]:
    """Забирает накопленные ``session/update``; все — этой сессии."""
    notifications = harness.client.take_notifications(UPDATE)
    assert all(params["sessionId"] == session_id for params in notifications), notifications
    return [params["update"] for params in notifications]


def _history(harness: AcpHarness, session_id: str) -> list[Json]:
    """Уведомления истории (реплики и вызовы), пришедшие до ответа на последний запрос."""
    return [
        update
        for update in _updates(harness, session_id)
        if update["sessionUpdate"] in HISTORY_KINDS
    ]


def _chunks(updates: list[Json], kind: str) -> list[str]:
    """Тексты реплик вида ``kind``, как их соберёт T3: куски склеены по ``messageId``.

    ``messageId`` обязателен у каждого куска: без него T3 склеил бы подряд идущие реплики
    одной роли (два ответа модели, два запроса пользователя).
    """
    texts: dict[str, str] = {}
    for update in updates:
        if update["sessionUpdate"] != kind:
            continue
        message_id = update.get("messageId")
        assert isinstance(message_id, str), update
        assert message_id, update
        assert update["content"]["type"] == "text", update
        texts[message_id] = texts.get(message_id, "") + update["content"]["text"]
    return list(texts.values())


def _by_call(updates: list[Json]) -> dict[str, list[Json]]:
    """Уведомления об инструментах по ``toolCallId``; первое у каждого вызова — ``tool_call``."""
    grouped: dict[str, list[Json]] = {}
    for update in updates:
        if update["sessionUpdate"] in {TOOL_CALL, TOOL_CALL_UPDATE}:
            grouped.setdefault(update["toolCallId"], []).append(update)
    for call_updates in grouped.values():
        assert call_updates[0]["sessionUpdate"] == TOOL_CALL, call_updates
        assert all(update["sessionUpdate"] == TOOL_CALL_UPDATE for update in call_updates[1:])
    return grouped


def _final_status(call_updates: list[Json]) -> str:
    statuses = [update["status"] for update in call_updates if "status" in update]
    assert statuses, call_updates
    status: str = statuses[-1]
    return status


def _texts_of(call_updates: list[Json]) -> str:
    """Весь текст из ``content`` уведомлений вызова."""
    texts = [
        part["content"]["text"]
        for update in call_updates
        for part in update.get("content") or ()
        if part["type"] == "content" and part["content"]["type"] == "text"
    ]
    return "\n".join(texts)


def _kinds(updates: list[Json]) -> list[str]:
    """Виды уведомлений по порядку; подряд идущие одинаковые — один раз."""
    kinds: list[str] = []
    for update in updates:
        if not kinds or kinds[-1] != update["sessionUpdate"]:
            kinds.append(update["sessionUpdate"])
    return kinds


def _requests_seen(provider: ScriptedProvider) -> list[str]:
    """Запросы пользователя (не служебные), какими их получила модель в последнем обращении."""
    messages = provider.requests[-1]["messages"]
    return [message.content for message in messages if message.is_request]


def _answers_seen(provider: ScriptedProvider) -> list[str]:
    """Ответы модели с текстом, которые она видела в последнем обращении."""
    messages = provider.requests[-1]["messages"]
    return [
        message.content for message in messages if message.role == "assistant" and message.content
    ]


def _without_session_id(response: Json) -> Json:
    return {key: field for key, field in response.items() if key != "sessionId"}


def _read_turn() -> ScriptedProvider:
    """Ход с двумя вызовами (c1 — прочитать заметки, c2 — несуществующий файл) и ответом."""
    calls = [("read_file", {"path": NOTES}), ("read_file", {"path": "missing.txt"})]
    return ScriptedProvider([tools_turn(calls), text_turn("Готово")])


# ------------------------------ сохранение ------------------------------ #
def test_each_prompt_saves_chat_under_session_id(acp: AcpHarness, root: Path) -> None:
    session_id = _open(acp, root, ScriptedProvider([text_turn("Здравствуй"), text_turn("Ещё")]))

    assert _ask(acp, session_id, "привет") == END_TURN
    saved = _store(root).load(session_id)
    assert saved.meta.model == "M-default"
    assert [message.content for message in saved.conversation.messages] == ["привет", "Здравствуй"]

    assert _ask(acp, session_id, "дальше") == END_TURN
    assert [chat.id for chat in _store(root).recent()] == [session_id]
    assert _store(root).load(session_id).meta.requests == 2


def test_disabled_saving_keeps_no_chat_and_load_fails(
    acp: AcpHarness,
    restart: Restart,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GIGACREW_SAVE_CHATS", "0")
    session_id = _open(acp, root, ScriptedProvider([text_turn("Здравствуй")]))

    assert _ask(acp, session_id, "привет") == END_TURN

    assert _store(root).recent() == []
    error = restart().call_error(LOAD, _load_params(session_id, root))
    assert error["code"] == RESOURCE_NOT_FOUND


# ------------------------------- загрузка ------------------------------- #
def test_load_replays_history_before_response(
    acp: AcpHarness,
    restart: Restart,
    root: Path,
) -> None:
    """Реплики и вызовы — в порядке истории, с id вызовов модели; ответ — как у session/new."""
    created = _new(acp, root, _read_turn())
    session_id = created["sessionId"]
    _ask(acp, session_id, "прочитай заметки")
    live_calls = _by_call(_updates(acp, session_id))

    reopened = restart()
    response = _load(reopened, session_id, root)

    replayed = _history(reopened, session_id)  # всё это пришло до ответа
    assert response == _without_session_id(created)
    assert _kinds(replayed)[0] == USER_CHUNK
    assert _kinds(replayed)[-1] == MESSAGE_CHUNK
    assert _chunks(replayed, USER_CHUNK) == ["прочитай заметки"]
    assert _chunks(replayed, MESSAGE_CHUNK) == ["Готово"]
    calls = _by_call(replayed)
    assert list(calls) == ["c1", "c2"]
    for call_id, call_updates in calls.items():
        assert call_updates[0]["title"] == live_calls[call_id][0]["title"]
        assert call_updates[0]["kind"] == live_calls[call_id][0]["kind"] == "read"
    assert _final_status(calls["c1"]) == "completed"
    assert NOTES_TEXT.strip() in _texts_of(calls["c1"])
    assert _final_status(calls["c2"]) == "failed"
    assert "missing.txt" in _texts_of(calls["c2"])


def test_loaded_session_continues_chat_under_same_id(
    acp: AcpHarness,
    restart: Restart,
    root: Path,
) -> None:
    session_id = _open(acp, root, _read_turn())
    _ask(acp, session_id, "прочитай заметки")
    reopened = restart()
    provider = ScriptedProvider([text_turn("Продолжаю")])
    _load(reopened, session_id, root, provider)

    assert _ask(reopened, session_id, "ещё") == END_TURN

    assert _requests_seen(provider) == ["прочитай заметки", "ещё"]
    assert _answers_seen(provider) == ["Готово"]
    assert NOTES_TEXT.strip() in "\n".join(
        message.content
        for message in provider.requests[0]["messages"]
        if message.role == "function"
    )
    assert [chat.id for chat in _store(root).recent()] == [session_id]
    saved = _store(root).load(session_id)
    assert saved.meta.requests == 2
    assert saved.conversation.messages[-1].content == "Продолжаю"


@pytest.mark.parametrize(
    ("saved_model", "provider", "expected"),
    [
        ("M-saved", ScriptedProvider, "M-saved"),
        (CATALOGUE[1], CatalogueProvider, CATALOGUE[1]),
        ("gone", CatalogueProvider, CATALOGUE[0]),
    ],
    ids=["no-catalogue", "in-catalogue", "not-in-catalogue"],
)
def test_load_restores_chat_model_if_backend_has_it(
    acp: AcpHarness,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    saved_model: str,
    provider: Callable[[], ScriptedProvider],
    expected: str,
) -> None:
    """Как продолжение чата в CLI: модели нет в каталоге — текущая модель, без ошибки."""
    monkeypatch.delenv("GIGACREW_MODEL")  # модель бэкенда по умолчанию
    conversation = Conversation([
        Message(role="user", content="вопрос"),
        Message(role="assistant", content="ответ"),
    ])
    session_id = _save_chat(root, conversation, saved_model)
    session_provider = provider()
    response = _load(acp, session_id, root, session_provider)

    _ask(acp, session_id, "ещё")

    assert session_provider.requests[0]["model"] == expected
    model_options = [option for option in response["configOptions"] if option["id"] == "model"]
    assert [option["currentValue"] for option in model_options] == [expected]


def test_service_messages_are_not_replayed_as_user_requests(acp: AcpHarness, root: Path) -> None:
    """Краткое содержание сжатия, служебные заметки и контекст хуков — не реплики пользователя."""
    hook_context = f"<system-reminder>\n{HOOK_CONTEXT_HEADER}\nконтекст хука\n</system-reminder>"
    hooked = f"второй вопрос\n\n{hook_context}"
    conversation = Conversation(
        [
            Message(role="user", content="первый вопрос"),
            Message(role="assistant", content="первый ответ"),
            Message(role="user", content="служебная заметка", meta=True),
            Message(role="assistant", content="учёл"),
            Message(role="user", content=hooked),
            Message(role="assistant", content="второй ответ"),
        ],
        summary=Summary(text="краткое содержание начала", upto=2),
    )
    session_id = _save_chat(root, conversation)

    _load(acp, session_id, root)

    replayed = _history(acp, session_id)
    assert _chunks(replayed, USER_CHUNK) == ["первый вопрос", "второй вопрос"]
    assert _chunks(replayed, MESSAGE_CHUNK) == ["первый ответ", "учёл", "второй ответ"]


def test_requests_in_a_row_replay_as_separate_messages(acp: AcpHarness, root: Path) -> None:
    """Ход упал без ответа модели, затем — новый запрос: две реплики пользователя подряд."""
    conversation = Conversation([
        Message(role="user", content="первый"),
        Message(role="user", content="второй"),
        Message(role="assistant", content="ответ"),
    ])
    session_id = _save_chat(root, conversation)

    _load(acp, session_id, root)

    replayed = _history(acp, session_id)
    assert _chunks(replayed, USER_CHUNK) == ["первый", "второй"]
    assert _chunks(replayed, MESSAGE_CHUNK) == ["ответ"]


def test_interrupted_calls_replay_as_failed_with_distinct_ids(acp: AcpHarness, root: Path) -> None:
    """Файл записан посреди вызовов (результатов нет), у второго вызова нет id от модели."""
    calls = [
        FunctionCall(name="read_file", arguments={"path": NOTES}, id="call-cut"),
        FunctionCall(name="read_file", arguments={"path": "other.txt"}),
    ]
    conversation = Conversation([
        Message(role="user", content="прочитай"),
        Message(role="assistant", function_calls=calls),
    ])
    session_id = _save_chat(root, conversation)

    _load(acp, session_id, root)

    replayed = _history(acp, session_id)
    assert _chunks(replayed, USER_CHUNK) == ["прочитай"]
    grouped = _by_call(replayed)
    assert len(grouped) == 2
    assert "call-cut" in grouped
    assert all(_final_status(call_updates) == "failed" for call_updates in grouped.values())


def test_loading_open_session_replays_it_and_keeps_it(acp: AcpHarness, root: Path) -> None:
    """T3 загружает и сессию, открытую в этом процессе: та же сессия, история — заново."""
    provider = ScriptedProvider([text_turn("ответ"), text_turn("ещё ответ")])
    session_id = _open(acp, root, provider)
    _ask(acp, session_id, "первый")
    _updates(acp, session_id)
    session = acp.server.session(session_id)

    _load(acp, session_id, root)

    replayed = _history(acp, session_id)
    assert _chunks(replayed, USER_CHUNK) == ["первый"]
    assert _chunks(replayed, MESSAGE_CHUNK) == ["ответ"]
    assert acp.server.session(session_id) is session
    assert _ask(acp, session_id, "второй") == END_TURN
    assert _requests_seen(provider) == ["первый", "второй"]
    assert len(acp.factory.issued) == 1


# -------------------------------- ошибки -------------------------------- #
def _unknown_chat(root: Path, other: Path) -> tuple[str, Path]:
    return ChatStore.new_id(), root


def _other_project(root: Path, other: Path) -> tuple[str, Path]:
    conversation = Conversation([Message(role="user", content="вопрос")])
    return _save_chat(root, conversation), other


def _malformed_id(root: Path, other: Path) -> tuple[str, Path]:
    return "../../escape", root


@pytest.mark.parametrize(
    "target",
    [_unknown_chat, _other_project, _malformed_id],
    ids=["unknown", "other-project", "malformed-id"],
)
def test_load_of_missing_chat_is_resource_not_found(
    acp: AcpHarness,
    root: Path,
    tmp_path_factory: pytest.TempPathFactory,
    target: Callable[[Path, Path], tuple[str, Path]],
) -> None:
    """Чата нет в хранилище проекта ``cwd``: T3 на эту ошибку начинает новую сессию."""
    session_id, cwd = target(root, tmp_path_factory.mktemp("other-project").resolve())

    error = acp.call_error(LOAD, _load_params(session_id, cwd))

    assert error["code"] == RESOURCE_NOT_FOUND
    _assert_server_alive(acp, root, session_id)


def test_load_of_broken_chat_file_is_error(acp: AcpHarness, root: Path) -> None:
    session_id = _save_chat(root, Conversation([Message(role="user", content="вопрос")]))
    chat_file = _store(root).directory / f"{session_id}.json"
    chat_file.write_text("не JSON", encoding="utf-8")

    error = acp.call_error(LOAD, _load_params(session_id, root))

    assert error["code"] == INTERNAL_ERROR
    assert session_id in error["message"]
    _assert_server_alive(acp, root, session_id)


def _assert_server_alive(harness: AcpHarness, root: Path, failed_id: str) -> None:
    """Неудачная загрузка не открыла сессию, а сервер работает дальше."""
    prompt = [{"type": "text", "text": "привет"}]
    error = harness.call_error(PROMPT, {"sessionId": failed_id, "prompt": prompt})
    assert error["code"] == INVALID_PARAMS
    session_id = _open(harness, root, ScriptedProvider([text_turn("на месте")]))
    assert _ask(harness, session_id, "привет") == END_TURN


# ------------------------------- закрытие ------------------------------- #
class _ClosingProvider(ScriptedProvider):
    """Провайдер, который помнит, что его закрыли."""

    def __init__(self) -> None:
        super().__init__([text_turn("ответ")])
        self.closed = False

    @override
    def close(self) -> None:
        self.closed = True


def test_close_session_closes_it(acp: AcpHarness, root: Path) -> None:
    provider = _ClosingProvider()
    session_id = _open(acp, root, provider)
    _ask(acp, session_id, "привет")

    assert acp.call(CLOSE, {"sessionId": session_id}) == {}

    assert provider.closed
    prompt = [{"type": "text", "text": "ещё"}]
    error = acp.call_error(PROMPT, {"sessionId": session_id, "prompt": prompt})
    assert error["code"] == INVALID_PARAMS
    _load(acp, session_id, root)  # чат сохранён: закрытую сессию можно загрузить снова
    assert _chunks(_history(acp, session_id), USER_CHUNK) == ["привет"]


def test_close_session_stops_background_jobs(acp: AcpHarness, root: Path) -> None:
    """Фоновая команда, запущенная в треде, не переживает закрытие его сессии.

    Процесс ``gigacrew acp`` в T3 живёт дольше треда: без остановки dev-сервер, запущенный
    агентом, работал бы до выхода всего процесса.
    """
    command = {"command": "exec sleep 60", "run_in_background": True}
    script = [tool_turn("run_shell", command), text_turn("запущено")]
    session_id = _open(acp, root, ScriptedProvider(script))
    assert _ask(acp, session_id, "запусти") == "end_turn"
    jobs = acp.server.session(session_id).agent.background
    assert jobs is not None
    assert jobs.active()

    acp.call(CLOSE, {"sessionId": session_id})

    assert not jobs.active()


class _StuckAfterFirst(ScriptedProvider):
    """Второе обращение хода ждёт без отрезков, как httpx до первого байта.

    Прерывание отмены до потока хода не доходит, пока обращение не отпустят.
    """

    def __init__(self, turns: Sequence[AssistantTurn]) -> None:
        super().__init__(turns)
        self.entered = threading.Event()
        self.release = threading.Event()

    @override
    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> AssistantTurn:
        if tools is not None and self.calls == 1:
            self.entered.set()
            self.release.wait()
        return super().complete(messages, tools, **kwargs)


def test_close_during_stuck_turn_kills_jobs_without_touching_history(
    acp: AcpHarness,
    root: Path,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Закрытие посреди хода не пишет в историю, пока поток хода её чинит.

    Поток хода, который не видит прерывания, ещё работает с историей: запись итогов
    фоновых задач из потока закрытия перемешала бы её. Процессы задач всё равно убиваются.
    """
    monkeypatch.setattr("gigacrew.cli.acp.session._CLOSE_DRAIN", 0.2)  # ход не доработает
    command = {"command": "exec sleep 60", "run_in_background": True}
    provider = _StuckAfterFirst([tool_turn("run_shell", command), text_turn("поздно")])
    request.addfinalizer(provider.release.set)
    session_id = _open(acp, root, provider)
    agent = acp.server.session(session_id).agent
    jobs = agent.background
    assert jobs is not None
    prompt = [{"type": "text", "text": "go"}]
    msg_id = acp.begin(PROMPT, {"sessionId": session_id, "prompt": prompt})
    assert provider.entered.wait(10)
    assert jobs.active()
    history = list(agent.conversation.messages)

    assert acp.call(CLOSE, {"sessionId": session_id}) == {}

    assert not jobs.active()
    assert list(agent.conversation.messages) == history
    provider.release.set()
    assert acp.reply(msg_id)["result"]["stopReason"] == "cancelled"


def test_closed_session_rejects_prompt_that_found_it_before_close(
    acp: AcpHarness,
    root: Path,
) -> None:
    """Prompt нашёл сессию до session/close и дошёл до хода после: ход не начинается."""
    provider = _ClosingProvider()
    session_id = _open(acp, root, provider)
    session = acp.server.session(session_id)

    assert acp.call(CLOSE, {"sessionId": session_id}) == {}

    with pytest.raises(RpcError) as caught:
        session.prompt("ещё")
    assert caught.value.code == INVALID_REQUEST
    assert not provider.requests


def test_close_saves_report_of_finished_background_job(acp: AcpHarness, root: Path) -> None:
    """Итог задачи, завершившейся после хода, попадает в сохранённый чат при закрытии."""
    command = {"command": "exec sleep 60", "run_in_background": True}
    script = [tool_turn("run_shell", command), text_turn("запущено")]
    session_id = _open(acp, root, ScriptedProvider(script))
    assert _ask(acp, session_id, "запусти") == "end_turn"
    jobs = acp.server.session(session_id).agent.background
    assert jobs is not None
    # Задача завершается только после хода: её итог модели ещё не доставлен.
    (job,) = jobs.active()
    jobs.stop(job, "тест")
    assert job.wait(10)
    assert not _saved_background_report(root, session_id)

    acp.call(CLOSE, {"sessionId": session_id})

    assert _saved_background_report(root, session_id)


def _saved_background_report(root: Path, session_id: str) -> bool:
    """В сохранённом чате есть итоги фоновых задач."""
    saved = _store(root).load(session_id).conversation
    return any(BACKGROUND_HEADER in (message.content or "") for message in saved.messages)


def test_close_of_unknown_session_is_quiet(acp: AcpHarness, root: Path) -> None:
    """T3 закрывает сессию при остановке треда и ошибку игнорирует: повтор — не ошибка."""
    session_id = _open(acp, root)
    acp.call(CLOSE, {"sessionId": session_id})

    assert acp.call(CLOSE, {"sessionId": session_id}) == {}
    assert acp.call(CLOSE, {"sessionId": ChatStore.new_id()}) == {}


# ------------------------- MCP-серверы проекта -------------------------- #
def _trusted_mcp_project(root: Path, scripts: Path) -> Path:
    """Подтверждённый раньше MCP-сервер ``probe`` в проекте ``root``; маркер с его pid."""
    launch = scripts / "launch_mcp.py"
    launch.write_text(LAUNCH_SCRIPT, encoding="utf-8")
    marker = scripts / "mcp-pid"
    server = {"command": sys.executable, "args": [str(launch), str(marker), str(FAKE_MCP_SERVER)]}
    (root / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"probe": server}}),
        encoding="utf-8",
    )
    workspace = Workspace(root)
    servers, _ = load_mcp_settings(workspace)
    mcp_trust_store().trust(workspace.root, servers_fingerprint(servers.project_servers))
    return marker


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_loaded_session_starts_project_mcp_at_first_prompt(
    servers: ExitStack,
    root: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Снимок чужого треда (load без prompt) не запускает MCP-серверы проекта."""
    marker = _trusted_mcp_project(root, tmp_path_factory.mktemp("scripts"))
    registries: list[ToolRegistry] = []

    def registry_factory() -> ToolRegistry:
        registries.append(build_default_registry())
        return registries[-1]

    harness = servers.enter_context(
        closing(AcpHarness(ProviderFactory(), registry_factory=registry_factory)),
    )
    harness.initialize()
    session_id = _save_chat(root, Conversation([Message(role="user", content="вопрос")]))
    _load(harness, session_id, root)

    assert not marker.exists()
    assert not any(PROBE_ECHO in registry for registry in registries)

    assert _ask(harness, session_id, "ещё") == END_TURN

    pid = int(marker.read_text(encoding="utf-8"))
    assert _alive(pid)
    assert [PROBE_ECHO in registry for registry in registries] == [True]
    assert harness.call(CLOSE, {"sessionId": session_id}) == {}
    assert not _alive(pid)  # session/close останавливает и дожидается сервер MCP


def test_loaded_session_restores_mcp_schemas_after_mcp_starts(
    servers: ExitStack,
    root: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Схема MCP-инструмента, загруженная в продолженном чате, снова загружена.

    История восстанавливается до запуска MCP-серверов (они запускаются при первом ходе):
    без повторного восстановления модель получила бы «схема не загружена».
    """
    _trusted_mcp_project(root, tmp_path_factory.mktemp("scripts"))
    harness = servers.enter_context(closing(AcpHarness(ProviderFactory())))
    harness.initialize()
    call = FunctionCall(name=PROBE_ECHO, arguments={"text": "x"}, id="c1")
    history = Conversation([
        Message(role="user", content="вопрос"),
        Message(role="assistant", content="", function_calls=[call]),
        Message(role="function", content="x", name=PROBE_ECHO, call_id="c1"),
        Message(role="assistant", content="готово"),
    ])
    session_id = _save_chat(root, history)
    _load(harness, session_id, root)

    assert _ask(harness, session_id, "ещё") == END_TURN

    # Схема — уже в первом запросе к модели загруженной сессии.
    provider = harness.factory.issued[-1]
    assert isinstance(provider, ScriptedProvider)
    first_tools = [spec.name for spec in provider.requests[0]["tools"]]
    assert PROBE_ECHO in first_tools
    assert PROBE_ECHO in harness.server.session(session_id).agent.loaded_tools


def test_cancel_while_loaded_session_starts_mcp(
    acp: AcpHarness,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """Отмена, пока первый ход загруженной сессии запускает MCP-серверы, отменяет ход."""
    entered = threading.Event()
    release = threading.Event()
    request.addfinalizer(release.set)

    def slow_connect(*_args: Any, **_kwargs: Any) -> McpHub:
        entered.set()
        assert release.wait(10), "запуск MCP не отпустили"
        return McpHub()

    session_id = _save_chat(root, Conversation([Message(role="user", content="вопрос")]))
    provider = ScriptedProvider([text_turn("поздно")])
    _load(acp, session_id, root, provider)
    monkeypatch.setattr("gigacrew.cli.acp.project_mcp.connect_trusted_mcp", slow_connect)
    prompt = [{"type": "text", "text": "ещё"}]
    msg_id = acp.begin(PROMPT, {"sessionId": session_id, "prompt": prompt})
    assert entered.wait(10)

    acp.client.notify("session/cancel", {"sessionId": session_id})
    acp.call("authenticate", {"methodId": "none"})  # отмена обработана: она раньше в потоке ввода
    release.set()

    assert acp.reply(msg_id)["result"]["stopReason"] == "cancelled"
    assert not provider.requests
