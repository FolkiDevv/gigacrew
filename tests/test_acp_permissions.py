"""ACP-сервер: подтверждения (session/request_permission) и вопросы (elicitation/create).

Подтверждение операции (``AgentEvents.confirm``) — запрос ``session/request_permission``
к клиенту: ``toolCall`` — тот же вызов (``toolCallId``), что в предшествующем ``tool_call``,
с превью операции; варианты ответа — ``optionId`` из :data:`ALLOW_ONCE`, :data:`ALLOW_TURN`,
:data:`ALLOW_ALWAYS`, :data:`REJECT_ONCE` (в этом порядке, лишние не предлагаются). Отказ
пользователя — только ``reject_once``; ответ ``cancelled``, неизвестный ``optionId``, битый
ответ и ошибка клиента — отказ без ответа пользователя (модель получает ``NO_ANSWER_NOTE``).

Кнопка «Стоп» в T3 — это ``session/cancel``; на висящий запрос подтверждения T3 затем
отвечает ``cancelled``. Сервер не полагается на этот ответ: отмена хода сама прерывает
ожидание, поздний ответ клиента ничего не разрешает.

Вопросы модели (инструмент ``ask_user``) — запрос ``elicitation/create`` в режиме ``form``,
только если клиент объявил ``clientCapabilities.elicitation.form``. Форма — в том виде,
который понимает ``parseElicitationForm`` в T3: вопрос номер N — поле ``qN`` (``title`` —
короткий заголовок, ``description`` — текст вопроса; одиночный выбор — строка с ``oneOf``,
множественный — массив с ``items: {anyOf: [...]}``), свой ответ — строковое поле
``qN__other`` (T3 сворачивает его в вопрос с выбором); ни одно поле не обязательно.

Сценарии подтверждений в режиме по умолчанию («авто»): запись в файл с секретами (не
опасна, «до конца хода» предлагается) и опасная команда ``rm -rf`` спрашиваются без
модели-рецензента. Команда, которую классификатор не знает, сначала идёт к рецензенту:
``ScriptedProvider`` отвечает ему неразборчиво, и решение переходит к пользователю — так
сценарий показывает, что «всегда» запомнит правило для команды (``ToolCallInfo.remember``).
"""

from __future__ import annotations

import json
import shlex
import sys
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from acp_harness import AcpHarness, ProviderFactory, acp_environment
from fakes import ScriptedProvider, text_turn, tool_turn

from gigacrew.agent.events import ToolCallInfo
from gigacrew.agent.loop import (
    AUTO_REMEMBER_AFTER,
    KEY_REMEMBER_HINT,
    NO_ANSWER_NOTE,
    REJECTED_NOTE,
)
from gigacrew.cli.acp.permission import options
from gigacrew.cli.acp.rpc import INTERNAL_ERROR
from gigacrew.llm.types import AssistantTurn
from gigacrew.project.settings import TrustStore, load_settings, project_fingerprint
from gigacrew.project.workspace import Workspace
from gigacrew.tools.ask_user import DECLINED_NOTE, format_answer
from gigacrew.tools.questions import Answer

# Исключение в любом потоке сервера роняет тест, а не теряется.
pytestmark = pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")

PROMPT = "session/prompt"
CANCEL = "session/cancel"
UPDATE = "session/update"
PERMISSION = "session/request_permission"
ELICITATION = "elicitation/create"
TOOL_CALL = "tool_call"
END_TURN = "end_turn"
CANCELLED = "cancelled"

# Варианты ответа на подтверждение: optionId и kind по ACP.
# allow_once обязан стоять первым: «Разрешить» в T3 выбирает первый вариант вида allow_once.
ALLOW_ONCE = "allow_once"
ALLOW_TURN = "allow_turn"  # Approval.TURN — «да» до конца хода (kind allow_once)
ALLOW_ALWAYS = "allow_always"
REJECT_ONCE = "reject_once"

# Файл с секретами: запись в него спрашивается и в режиме «авто», но не опасна.
SECRET = "credentials.json"
SECRET_TEXT = "TOKEN=needle"
DANGEROUS_COMMAND = "rm -rf build"
# Команда, которую классификатор не знает: в «авто» её проверяет рецензент; ответ
# ScriptedProvider ему неразборчив — решение переходит к пользователю.
UNKNOWN_COMMAND = "frobnicate --x"
# Начало сообщения QuestionsUnavailableError — модель узнаёт, что спросить некого.
QUESTIONS_UNAVAILABLE = "Нельзя задать вопрос"

DATABASE_QUESTION = "Какую базу взять?"
FEATURES_QUESTION = "Что включить?"
QUESTIONS = (
    {
        "question": DATABASE_QUESTION,
        "header": "База",
        "options": [
            {"label": "SQLite", "description": "файл рядом с проектом"},
            {"label": "Postgres", "description": "отдельный сервер"},
        ],
    },
    {
        "question": FEATURES_QUESTION,
        "options": [{"label": "Логи"}, {"label": "Метрики"}, {"label": "Трейсы"}],
        "multi_select": True,
    },
)

type Json = dict[str, Any]


@pytest.fixture
def client_capabilities() -> Json:
    """Возможности клиента в ``initialize``: по умолчанию — формы elicitation есть."""
    return {"elicitation": {"form": {}}}


@pytest.fixture
def acp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
    client_capabilities: Json,
) -> Iterator[AcpHarness]:
    """Сервер со штатными инструментами; клиент объявил ``client_capabilities``.

    Висящий запрос к клиенту (вопрос без ответа в упавшем тесте) завершается ошибкой при
    закрытии канала — ход заканчивается, сервер останавливается.
    """
    acp_environment(monkeypatch, tmp_path_factory.mktemp("process-cwd"))
    with closing(AcpHarness(ProviderFactory())) as harness:
        harness.initialize(client_capabilities)
        yield harness


@pytest.fixture
def root(project: Path) -> Path:
    """Проект сессии с каталогом ``build`` (цель опасной команды)."""
    (project / "build").mkdir()
    return project.resolve()


def _open(acp: AcpHarness, cwd: Path, provider: ScriptedProvider) -> str:
    acp.factory.add(provider)
    return acp.new_session(cwd)


def _prompt_params(session_id: str, text: str) -> Json:
    return {"sessionId": session_id, "prompt": [{"type": "text", "text": text}]}


def _ask(acp: AcpHarness, session_id: str, text: str = "сделай") -> Any:
    """Ход целиком: ``stopReason`` (запрос к клиенту без ответа — тест падает по таймауту)."""
    return acp.call(PROMPT, _prompt_params(session_id, text))["stopReason"]


def _begin(acp: AcpHarness, session_id: str, text: str = "сделай") -> int:
    return acp.begin(PROMPT, _prompt_params(session_id, text))


def _stop_reason(acp: AcpHarness, msg_id: int) -> Any:
    reply = acp.reply(msg_id)
    assert "error" not in reply, reply
    return reply["result"]["stopReason"]


def _write(content: str = SECRET_TEXT) -> AssistantTurn:
    return tool_turn("write_file", {"path": SECRET, "content": content})


def _shell(command: str) -> AssistantTurn:
    return tool_turn("run_shell", {"command": command})


def _ask_user() -> AssistantTurn:
    return tool_turn("ask_user", {"questions": list(QUESTIONS)})


def _wait_server_request(
    acp: AcpHarness,
    method: str,
    session_id: str,
    prompt_id: int,
) -> tuple[Any, Json]:
    """Запрос сервера ``method`` к клиенту во время хода ``prompt_id``: его id и params.

    Ход закончился, так и не спросив клиента (ответ на prompt пришёл раньше), — тест падает
    сразу с этим ответом, а не по таймауту.
    """

    def matches(message: Json) -> bool:
        if message.get("method") == method and "id" in message:
            return True
        return message.get("id") == prompt_id and "method" not in message

    request = acp.client.wait_for(matches)
    assert request.get("method") == method, f"ход закончился без запроса {method}: {request}"
    params: Json = request["params"]
    assert params["sessionId"] == session_id
    return request["id"], params


def _choose(acp: AcpHarness, msg_id: Any, option_id: str) -> None:
    acp.client.respond(msg_id, {"outcome": {"outcome": "selected", "optionId": option_id}})


def _tool_calls(acp: AcpHarness, session_id: str) -> dict[str, list[Json]]:
    """Забирает накопленные ``session/update``: уведомления об инструментах по toolCallId."""
    grouped: dict[str, list[Json]] = {}
    for params in acp.client.take_notifications(UPDATE):
        assert params["sessionId"] == session_id
        update = params["update"]
        call_id = update.get("toolCallId")
        if call_id is not None:
            grouped.setdefault(call_id, []).append(update)
    return grouped


def _single_call(acp: AcpHarness, session_id: str) -> tuple[str, list[Json]]:
    """Единственный вызов инструмента среди накопленных уведомлений: id и уведомления."""
    grouped = _tool_calls(acp, session_id)
    assert len(grouped) == 1, grouped
    call_id, call_updates = next(iter(grouped.items()))
    assert call_updates[0]["sessionUpdate"] == TOOL_CALL
    return call_id, call_updates


def _last_status(call_updates: list[Json]) -> Any:
    return [update["status"] for update in call_updates if "status" in update][-1]


def _options(params: Json) -> list[tuple[str, str]]:
    """Варианты ответа запроса подтверждения: (optionId, kind) по порядку."""
    return [(option["optionId"], option["kind"]) for option in params["options"]]


def _option_name(params: Json, option_id: str) -> str:
    names = [option["name"] for option in params["options"] if option["optionId"] == option_id]
    assert len(names) == 1, params["options"]
    name: str = names[0]
    return name


def _preview_text(tool_call: Json) -> str:
    """Текст превью в ``content`` вызова: новый текст диффа и текстовые блоки."""
    texts = []
    for part in tool_call.get("content") or ():
        if part["type"] == "diff":
            texts.append(part["newText"])
        elif part["type"] == "content" and part["content"]["type"] == "text":
            texts.append(part["content"]["text"])
    return "\n".join(texts)


def _model_results(provider: ScriptedProvider) -> str:
    """Результаты инструментов, какими их получила модель в последнем обращении."""
    messages = provider.requests[-1]["messages"]
    return "\n".join(message.content for message in messages if message.role == "function")


def _assert_server_alive(acp: AcpHarness) -> None:
    """Сервер отвечает на следующий запрос: всё, что он прислал раньше, уже прочитано."""
    reply = acp.call("authenticate", {"methodId": "none"})
    assert isinstance(reply, dict)
    assert not reply


def _assert_no_requests_to_client(acp: AcpHarness) -> None:
    """Сервер не спрашивал клиента.

    Проверка — после барьера: ответа на prompt (он приходит после всех сообщений хода) или
    :func:`_assert_server_alive`.
    """
    requests = [
        message for message in acp.client.backlog if "method" in message and "id" in message
    ]
    assert not requests, requests


# --- подтверждения: запрос ---


def test_permission_request_describes_the_pending_call(acp: AcpHarness, root: Path) -> None:
    """Запрос — о том же вызове, что в ``tool_call``, с превью; до ответа ничего не записано."""
    session_id = _open(acp, root, ScriptedProvider([_write(), text_turn("Готово")]))
    msg_id = _begin(acp, session_id)

    request_id, params = _wait_server_request(acp, PERMISSION, session_id, msg_id)

    call_id, call_updates = _single_call(acp, session_id)
    tool_call = params["toolCall"]
    assert tool_call["toolCallId"] == call_id
    assert SECRET in tool_call["title"]
    assert tool_call["kind"] == call_updates[0]["kind"]
    assert tool_call["rawInput"] == {"path": SECRET, "content": SECRET_TEXT}
    assert "needle" in _preview_text(tool_call)
    assert not (root / SECRET).exists()
    _choose(acp, request_id, REJECT_ONCE)
    assert _stop_reason(acp, msg_id) == END_TURN


@pytest.mark.parametrize(
    ("turn", "options", "name_parts"),
    [
        (
            _write(),
            [
                (ALLOW_ONCE, "allow_once"),
                (ALLOW_TURN, "allow_once"),
                (ALLOW_ALWAYS, "allow_always"),
                (REJECT_ONCE, "reject_once"),
            ],
            {ALLOW_TURN: "до конца хода"},
        ),
        (
            _shell(DANGEROUS_COMMAND),
            [(ALLOW_ONCE, "allow_once"), (REJECT_ONCE, "reject_once")],
            {},
        ),
        (
            _shell(UNKNOWN_COMMAND),
            [
                (ALLOW_ONCE, "allow_once"),
                (ALLOW_ALWAYS, "allow_always"),
                (REJECT_ONCE, "reject_once"),
            ],
            {ALLOW_ALWAYS: UNKNOWN_COMMAND},  # что запомнит «всегда» (ToolCallInfo.remember)
        ),
    ],
    ids=["secret-file", "dangerous-command", "unknown-command"],
)
def test_permission_options_follow_the_call(
    acp: AcpHarness,
    root: Path,
    turn: AssistantTurn,
    options: list[tuple[str, str]],
    name_parts: dict[str, str],
) -> None:
    """«Всегда» — не для опасного; «до конца хода» — только если агент назвал, что он покроет."""
    session_id = _open(acp, root, ScriptedProvider([turn, text_turn("Готово")]))
    msg_id = _begin(acp, session_id)

    request_id, params = _wait_server_request(acp, PERMISSION, session_id, msg_id)

    assert _options(params) == options
    assert all(_option_name(params, option_id) for option_id, _ in options)
    for option_id, part in name_parts.items():
        assert part in _option_name(params, option_id)
    _choose(acp, request_id, REJECT_ONCE)
    assert _stop_reason(acp, msg_id) == END_TURN
    assert (root / "build").is_dir()  # отклонённая опасная команда не выполнялась


def test_dangerous_command_preview_shows_the_command(acp: AcpHarness, root: Path) -> None:
    session_id = _open(acp, root, ScriptedProvider([_shell(DANGEROUS_COMMAND), text_turn("Ок")]))
    msg_id = _begin(acp, session_id)

    request_id, params = _wait_server_request(acp, PERMISSION, session_id, msg_id)

    assert DANGEROUS_COMMAND in _preview_text(params["toolCall"])
    assert params["toolCall"]["rawInput"] == {"command": DANGEROUS_COMMAND}
    _choose(acp, request_id, REJECT_ONCE)
    assert _stop_reason(acp, msg_id) == END_TURN


def test_allow_once_never_becomes_standing_permission(acp: AcpHarness, root: Path) -> None:
    """«Разрешить» в ACP — всегда один раз, даже после многих «да» подряд в режиме «авто».

    CLI после нескольких «да» запоминает команду и предупреждает об этом в вопросе; T3
    названия вариантов не показывает (кнопки у него свои), так что предупреждение не
    увидеть — запоминает только явное «всегда».
    """
    rounds = AUTO_REMEMBER_AFTER + 1
    turns = [turn for _ in range(rounds) for turn in (_shell(UNKNOWN_COMMAND), text_turn("Готово"))]
    session_id = _open(acp, root, ScriptedProvider(turns))
    names: list[str] = []
    for _ in range(rounds):
        msg_id = _begin(acp, session_id)
        request_id, params = _wait_server_request(acp, PERMISSION, session_id, msg_id)
        names.append(_option_name(params, ALLOW_ONCE))
        _choose(acp, request_id, ALLOW_ONCE)
        assert _stop_reason(acp, msg_id) == END_TURN

    assert len(names) == rounds  # спрашивали каждый раз
    assert all("запомнится" not in name for name in names)


_ASK_HOOK = (
    "import json\n"
    'print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", '
    '"permissionDecision": "ask"}}))\n'
)


def _trust_ask_hook(project: Path, script_dir: Path) -> None:
    """Подтверждённый раньше хук PreToolUse проекта, который требует спрашивать каждый раз."""
    script = script_dir / "ask_hook.py"
    script.write_text(_ASK_HOOK, encoding="utf-8")
    command = shlex.join([sys.executable, str(script)])
    hooks = {
        "PreToolUse": [
            {"matcher": "write_file", "hooks": [{"type": "command", "command": command}]},
        ],
    }
    (project / ".gigacode").mkdir(exist_ok=True)
    (project / ".gigacode" / "settings.json").write_text(
        json.dumps({"hooks": hooks}),
        encoding="utf-8",
    )
    workspace = Workspace(project)
    settings, _ = load_settings(workspace)
    TrustStore().trust(workspace.root, project_fingerprint(settings))


def test_hook_confirmation_offers_no_always(
    acp: AcpHarness,
    root: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Хук PreToolUse спрашивает каждый раз: «всегда» запомнить нельзя — и не предлагается."""
    _trust_ask_hook(root, tmp_path_factory.mktemp("hook"))
    write = tool_turn("write_file", {"path": "notes.txt", "content": "x"})
    session_id = _open(acp, root, ScriptedProvider([write, text_turn("Готово")]))
    msg_id = _begin(acp, session_id)

    request_id, params = _wait_server_request(acp, PERMISSION, session_id, msg_id)

    assert _options(params) == [(ALLOW_ONCE, "allow_once"), (REJECT_ONCE, "reject_once")]
    _choose(acp, request_id, REJECT_ONCE)
    assert _stop_reason(acp, msg_id) == END_TURN


# --- подтверждения: ответ ---


def _selected(option_id: str) -> Json:
    return {"outcome": {"outcome": "selected", "optionId": option_id}}


@pytest.mark.parametrize(
    ("response", "model_note"),
    [
        (_selected(REJECT_ONCE), REJECTED_NOTE),
        (_selected("no-such-option"), NO_ANSWER_NOTE),  # непонятный ответ — не выбор
        ({"outcome": {"outcome": CANCELLED}}, NO_ANSWER_NOTE),  # клиент отменил — ответа нет
        ({}, NO_ANSWER_NOTE),
    ],
    ids=["rejected", "unknown-option", "cancelled-outcome", "malformed"],
)
def test_answer_other_than_allow_refuses_the_call(
    acp: AcpHarness,
    root: Path,
    response: Json,
    model_note: str,
) -> None:
    provider = ScriptedProvider([_write(), text_turn("Не стал")])
    session_id = _open(acp, root, provider)
    msg_id = _begin(acp, session_id)
    request_id, _ = _wait_server_request(acp, PERMISSION, session_id, msg_id)

    acp.client.respond(request_id, response)

    assert _stop_reason(acp, msg_id) == END_TURN
    assert not (root / SECRET).exists()
    _, call_updates = _single_call(acp, session_id)
    assert _last_status(call_updates) == "failed"
    assert model_note in _model_results(provider)


def test_error_response_refuses_the_call(acp: AcpHarness, root: Path) -> None:
    provider = ScriptedProvider([_write(), text_turn("Не стал")])
    session_id = _open(acp, root, provider)
    msg_id = _begin(acp, session_id)
    request_id, _ = _wait_server_request(acp, PERMISSION, session_id, msg_id)

    acp.client.respond_error(request_id, INTERNAL_ERROR, "клиент не смог спросить")

    assert _stop_reason(acp, msg_id) == END_TURN
    assert not (root / SECRET).exists()
    _, call_updates = _single_call(acp, session_id)
    assert _last_status(call_updates) == "failed"
    assert NO_ANSWER_NOTE in _model_results(provider)


def test_allow_once_runs_the_call_and_asks_again_next_time(acp: AcpHarness, root: Path) -> None:
    """Повтор — точно такой же вызов: «один раз» не разрешает его до конца сессии."""
    provider = ScriptedProvider([_write(), text_turn("Записал"), _write(), text_turn("Ещё")])
    session_id = _open(acp, root, provider)
    msg_id = _begin(acp, session_id)
    request_id, _ = _wait_server_request(acp, PERMISSION, session_id, msg_id)

    _choose(acp, request_id, ALLOW_ONCE)

    assert _stop_reason(acp, msg_id) == END_TURN
    assert (root / SECRET).read_text(encoding="utf-8") == SECRET_TEXT
    _, call_updates = _single_call(acp, session_id)
    assert _last_status(call_updates) == "completed"
    (root / SECRET).unlink()
    second = _begin(acp, session_id, "ещё")
    request_id, _ = _wait_server_request(acp, PERMISSION, session_id, second)
    assert not (root / SECRET).exists()
    _choose(acp, request_id, ALLOW_ONCE)
    assert _stop_reason(acp, second) == END_TURN
    assert (root / SECRET).read_text(encoding="utf-8") == SECRET_TEXT


def test_allow_always_is_not_asked_again_in_the_session(acp: AcpHarness, root: Path) -> None:
    provider = ScriptedProvider([_write(), text_turn("Записал"), _write(), text_turn("Ещё раз")])
    session_id = _open(acp, root, provider)
    msg_id = _begin(acp, session_id)
    request_id, _ = _wait_server_request(acp, PERMISSION, session_id, msg_id)
    _choose(acp, request_id, ALLOW_ALWAYS)
    assert _stop_reason(acp, msg_id) == END_TURN
    (root / SECRET).unlink()
    acp.client.take_notifications(UPDATE)

    assert _ask(acp, session_id, "ещё") == END_TURN

    _assert_no_requests_to_client(acp)
    assert (root / SECRET).read_text(encoding="utf-8") == SECRET_TEXT
    _, call_updates = _single_call(acp, session_id)
    assert _last_status(call_updates) == "completed"


def test_allow_for_turn_covers_the_file_until_the_turn_ends(acp: AcpHarness, root: Path) -> None:
    provider = ScriptedProvider(
        [_write("A"), _write("B"), text_turn("Записал"), _write("C"), text_turn("Ещё")],
    )
    session_id = _open(acp, root, provider)
    msg_id = _begin(acp, session_id)
    request_id, _ = _wait_server_request(acp, PERMISSION, session_id, msg_id)

    _choose(acp, request_id, ALLOW_TURN)

    assert _stop_reason(acp, msg_id) == END_TURN  # вторая запись того же файла — без вопроса
    _assert_no_requests_to_client(acp)
    assert (root / SECRET).read_text(encoding="utf-8") == "B"
    second = _begin(acp, session_id, "ещё")
    request_id, _ = _wait_server_request(acp, PERMISSION, session_id, second)  # новый ход — снова
    _choose(acp, request_id, ALLOW_ONCE)
    assert _stop_reason(acp, second) == END_TURN
    assert (root / SECRET).read_text(encoding="utf-8") == "C"


# --- подтверждения: отмена и конец ввода ---


def test_cancel_while_waiting_for_permission(acp: AcpHarness, root: Path) -> None:
    """Отмена прерывает ожидание ответа; поздний ответ клиента ничего не разрешает.

    Поздний «всегда» не становится и правилом сессии: такой же вызов в следующем ходе
    снова спрашивается.
    """
    provider = ScriptedProvider([_write(), _write(), text_turn("Второй ход")])
    session_id = _open(acp, root, provider)
    msg_id = _begin(acp, session_id)
    request_id, _ = _wait_server_request(acp, PERMISSION, session_id, msg_id)

    acp.client.notify(CANCEL, {"sessionId": session_id})

    assert _stop_reason(acp, msg_id) == CANCELLED
    _choose(acp, request_id, ALLOW_ALWAYS)  # клиент ответил после отмены
    _assert_server_alive(acp)
    assert not (root / SECRET).exists()
    acp.client.take_notifications(UPDATE)
    second = _begin(acp, session_id, "ещё")
    request_id, _ = _wait_server_request(acp, PERMISSION, session_id, second)
    _choose(acp, request_id, REJECT_ONCE)
    assert _stop_reason(acp, second) == END_TURN
    assert not (root / SECRET).exists()


def test_end_of_input_while_waiting_for_permission_stops_server(
    acp: AcpHarness,
    root: Path,
) -> None:
    """Клиент закрыл канал, не ответив: вызов не выполняется, сервер завершается.

    Файл проверяется после конца хода (ответ на prompt — любой): иначе запись, ошибочно
    разрешённая закрытием канала, могла бы случиться уже после проверки.
    """
    session_id = _open(acp, root, ScriptedProvider([_write(), text_turn("Готово")]))
    msg_id = _begin(acp, session_id)
    _wait_server_request(acp, PERMISSION, session_id, msg_id)

    acp.stop()

    acp.reply(msg_id)
    assert not (root / SECRET).exists()


# --- вопросы ---


def _field_text(field: Json) -> str:
    title = field.get("title") or ""
    description = field.get("description") or ""
    return f"{title}\n{description}"


def test_questions_are_form_elicitation(acp: AcpHarness, root: Path) -> None:
    """Вопрос — поле выбора ``qN`` и поле своего ответа ``qN__other``; обязательных нет."""
    session_id = _open(acp, root, ScriptedProvider([_ask_user(), text_turn("Ок")]))
    msg_id = _begin(acp, session_id)

    request_id, params = _wait_server_request(acp, ELICITATION, session_id, msg_id)

    call_id, _ = _single_call(acp, session_id)
    assert params["mode"] == "form"
    assert params["toolCallId"] == call_id
    assert params["message"]
    schema = params["requestedSchema"]
    assert schema.get("type", "object") == "object"
    fields = schema["properties"]
    assert set(fields) == {"q1", "q1__other", "q2", "q2__other"}
    assert not set(schema.get("required") or ()) & set(fields)
    database = fields["q1"]
    assert database["type"] == "string"
    assert database["title"] == "База"  # заголовок вопроса в T3
    assert DATABASE_QUESTION in database["description"]  # текст вопроса в T3
    assert database["oneOf"] == [
        {"const": "SQLite", "title": "SQLite", "description": "файл рядом с проектом"},
        {"const": "Postgres", "title": "Postgres", "description": "отдельный сервер"},
    ]
    features = fields["q2"]
    assert features["type"] == "array"
    assert FEATURES_QUESTION in features["description"]
    # Иной вид items T3 не понимает и отклоняет всю форму.
    assert features["items"] == {
        "anyOf": [
            {"const": "Логи", "title": "Логи"},
            {"const": "Метрики", "title": "Метрики"},
            {"const": "Трейсы", "title": "Трейсы"},
        ],
    }
    assert fields["q1__other"]["type"] == "string"
    # Свой ответ к множественному выбору T3 не сворачивает: это отдельный вопрос — он
    # должен называть, к какому вопросу относится.
    assert fields["q2__other"]["type"] == "string"
    assert FEATURES_QUESTION in _field_text(fields["q2__other"])
    acp.client.respond(request_id, {"action": "decline"})
    assert _stop_reason(acp, msg_id) == END_TURN


@pytest.mark.parametrize(
    ("content", "answers"),
    [
        (
            {"q1__other": "MySQL", "q2": ["Логи", "Трейсы"], "q2__other": "и алерты"},
            [Answer(custom="MySQL"), Answer(("Логи", "Трейсы"), "и алерты")],
        ),
        ({"q1": "Postgres"}, [Answer(("Postgres",)), Answer()]),
        # Значение не из вариантов (клиент дал ввести своё) — свой ответ, а не выбор.
        ({"q1": "Nope", "q2": ["Логи"]}, [Answer(custom="Nope"), Answer(("Логи",))]),
    ],
    ids=["custom-and-multi", "partial", "unknown-choice-is-custom"],
)
def test_form_answers_reach_model_in_question_order(
    acp: AcpHarness,
    root: Path,
    content: Json,
    answers: list[Answer],
) -> None:
    provider = ScriptedProvider([_ask_user(), text_turn("Понял")])
    session_id = _open(acp, root, provider)
    msg_id = _begin(acp, session_id)
    request_id, _ = _wait_server_request(acp, ELICITATION, session_id, msg_id)

    acp.client.respond(request_id, {"action": "accept", "content": content})

    assert _stop_reason(acp, msg_id) == END_TURN
    seen = _model_results(provider)
    questions = (DATABASE_QUESTION, FEATURES_QUESTION)
    for number, (question, answer) in enumerate(zip(questions, answers, strict=True), 1):
        assert f"{number}. {question} → {format_answer(answer)}" in seen
    _, call_updates = _single_call(acp, session_id)
    assert _last_status(call_updates) == "completed"


@pytest.mark.parametrize(
    "response",
    [
        {"action": ["accept"]},
        {"action": {"accept": True}},
        # Одиночный выбор — строка: массив в нём — не «выбрал оба».
        {"action": "accept", "content": {"q1": ["SQLite", "Postgres"]}},
    ],
    ids=["list-action", "dict-action", "array-in-single-choice"],
)
def test_malformed_form_answer_fails_for_model(
    acp: AcpHarness,
    root: Path,
    response: Json,
) -> None:
    """Непонятный ответ на форму — ответов нет: модель решает сама, а не принимает выдумку."""
    provider = ScriptedProvider([_ask_user(), text_turn("Решу сам")])
    session_id = _open(acp, root, provider)
    msg_id = _begin(acp, session_id)
    request_id, _ = _wait_server_request(acp, ELICITATION, session_id, msg_id)

    acp.client.respond(request_id, response)

    assert _stop_reason(acp, msg_id) == END_TURN
    seen = _model_results(provider)
    assert QUESTIONS_UNAVAILABLE in seen
    assert "Postgres" not in seen
    _, call_updates = _single_call(acp, session_id)
    assert _last_status(call_updates) == "failed"


@pytest.mark.parametrize("action", ["decline", "cancel"])
def test_declined_form_is_refusal_to_answer(acp: AcpHarness, root: Path, action: str) -> None:
    provider = ScriptedProvider([_ask_user(), text_turn("Решу сам")])
    session_id = _open(acp, root, provider)
    msg_id = _begin(acp, session_id)
    request_id, _ = _wait_server_request(acp, ELICITATION, session_id, msg_id)

    acp.client.respond(request_id, {"action": action})

    assert _stop_reason(acp, msg_id) == END_TURN
    assert DECLINED_NOTE in _model_results(provider)


@pytest.mark.parametrize(
    "client_capabilities",
    [{}, {"elicitation": None}, {"elicitation": {"form": None}}, {"elicitation": {"url": {}}}],
    ids=["none", "null-elicitation", "null-form", "url-only"],
)
def test_questions_without_form_support_fail_for_model(acp: AcpHarness, root: Path) -> None:
    """Клиент не умеет формы: вопрос не задаётся, модель получает ошибку, ход продолжается."""
    provider = ScriptedProvider([_ask_user(), text_turn("Решу сам")])
    session_id = _open(acp, root, provider)

    assert _ask(acp, session_id) == END_TURN

    _assert_no_requests_to_client(acp)
    assert QUESTIONS_UNAVAILABLE in _model_results(provider)
    _, call_updates = _single_call(acp, session_id)
    assert _last_status(call_updates) == "failed"


def test_error_response_to_questions_fails_for_model(acp: AcpHarness, root: Path) -> None:
    """Клиент ответил на вопрос ошибкой: ответов нет — модель получает ошибку, а не выдумку."""
    provider = ScriptedProvider([_ask_user(), text_turn("Решу сам")])
    session_id = _open(acp, root, provider)
    msg_id = _begin(acp, session_id)
    request_id, _ = _wait_server_request(acp, ELICITATION, session_id, msg_id)

    acp.client.respond_error(request_id, INTERNAL_ERROR, "форма не открылась")

    assert _stop_reason(acp, msg_id) == END_TURN
    _, call_updates = _single_call(acp, session_id)
    assert _last_status(call_updates) == "failed"
    assert DECLINED_NOTE not in _model_results(provider)  # ошибка — не отказ пользователя


def test_cancel_while_question_is_open(acp: AcpHarness, root: Path) -> None:
    """Отмена хода во время вопроса — prompt cancelled; поздние ответы модели не достаются."""
    provider = ScriptedProvider([_ask_user(), text_turn("Второй ход")])
    session_id = _open(acp, root, provider)
    msg_id = _begin(acp, session_id)
    request_id, _ = _wait_server_request(acp, ELICITATION, session_id, msg_id)

    acp.client.notify(CANCEL, {"sessionId": session_id})

    assert _stop_reason(acp, msg_id) == CANCELLED
    acp.client.respond(request_id, {"action": "accept", "content": {"q1": "Postgres"}})
    _assert_server_alive(acp)
    acp.client.take_notifications(UPDATE)
    assert _ask(acp, session_id, "ещё") == END_TURN
    assert f"→ {format_answer(Answer(('Postgres',)))}" not in _model_results(provider)


def test_once_option_has_no_terminal_key_hint() -> None:
    """Подсказка CLI про клавишу «a» в T3 бессмысленна: в названии «да» её нет."""
    history = f"раньше: разрешали 2 — {KEY_REMEMBER_HINT}"
    call = ToolCallInfo(name="run_shell", history=history)

    names = [option["name"] for option in options(call, dangerous=False)]

    assert names[0] == "Да (раньше: разрешали 2)"
