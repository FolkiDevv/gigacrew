"""ACP-сервер: режимы, выбор модели и слэш-команды сессии.

Режимы — :class:`~gigacrew.permissions.PermissionMode`: ``session/new`` отвечает полем
``modes`` (``currentModeId`` — режим из конфигурации, ``availableModes`` — ``auto`` и
``plan``), ``session/set_mode`` меняет режим агента. Режимы не дублируются в
``configOptions`` (категория ``mode``): найдя такой параметр, T3 переключал бы режим через
``session/set_config_option``, а не ``session/set_mode``. После ``session/set_mode`` T3 сам
запоминает новый режим; уведомление ``current_mode_update`` нужно, когда режим меняет агент
(одобренный план переводит его из «плана» в «авто»). Переключатель плана в T3 ищет режим
с id ``plan``.

Модель — параметр ``configOptions`` с id ``model`` (категория ``model``, вид ``select``):
варианты — каталог бэкенда (``provider.known_models()``), без каталога — текущая модель.
T3 выбирает модель через ``session/set_config_option`` и ждёт в ответе ``configOptions``
с новым ``currentValue``; ``session/set_model`` (для клиентов без ``configOptions``) делает
то же. Без каталога модель любая — как в CLI (в T3 это «свои модели» из его настроек).
Модель, выбранная во время хода, действует со следующего обращения к модели.

Слэш-команды: после ``session/new`` сервер присылает ``available_commands_update`` —
команды без терминала: ``compact`` и скиллы (имена без ``/``; T3 шлёт их текстом
``/<имя> …``). ``/compact`` — сжатие контекста без хода модели, ``/<скилл> аргументы``
раскрывается, как в CLI; неизвестная ``/xyz`` уходит модели обычным текстом (как ``-p``).

Уведомление ``plan`` сервер не шлёт: план ядра — свободный Markdown в файле плана, без
пунктов со статусами, которые просит ACP.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, closing
from pathlib import Path
from typing import Any, override

import pytest
from acp_client import TIMEOUT
from acp_harness import AcpHarness, ProviderFactory, acp_environment
from fakes import CatalogueProvider, ScriptedProvider, text_turn, tool_turn

from gigacrew.agent.loop import PLAN_BLOCKED_NOTE, Agent
from gigacrew.cli.acp.rpc import INTERNAL_ERROR, INVALID_PARAMS
from gigacrew.llm.base import LLMError
from gigacrew.llm.types import AssistantTurn, Message, ToolSpec
from gigacrew.permissions import PermissionMode
from gigacrew.tools.plan import APPROVE

# Исключение в любом потоке сервера роняет тест, а не теряется.
pytestmark = pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")

NEW = "session/new"
PROMPT = "session/prompt"
SET_MODE = "session/set_mode"
SET_OPTION = "session/set_config_option"
SET_MODEL = "session/set_model"
UPDATE = "session/update"
ELICITATION = "elicitation/create"
MODE_UPDATE = "current_mode_update"
COMMANDS_UPDATE = "available_commands_update"
MODEL = "model"
END_TURN = "end_turn"
NOTES = "notes.txt"
NOTES_TEXT = "hello from notes\n"
CATALOGUE = ("cat-a", "cat-b")
SKILL_BODY = "Проверь изменения ветки и перечисли замечания."

type Json = dict[str, Any]


class _GatedCatalogue(CatalogueProvider):
    """Каталог ``cat-a``/``cat-b``; первое обращение хода ждёт ``release``."""

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
        if tools is not None and not self.entered.is_set():
            self.entered.set()
            assert self.release.wait(TIMEOUT * 3), "обращение к модели не отпустили"
        return super().complete(messages, tools, **kwargs)


class _LateScript(ScriptedProvider):
    """Сценарий задаётся после создания сессии (ходам нужен её id)."""

    def script(self, turns: Sequence[AssistantTurn]) -> None:
        self._turns.extend(turns)


@pytest.fixture
def factory() -> ProviderFactory:
    return ProviderFactory()


@pytest.fixture
def releases() -> list[threading.Event]:
    """Задержки провайдеров, которые фикстура ``acp`` отпустит до остановки сервера."""
    return []


@pytest.fixture
def acp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
    factory: ProviderFactory,
    releases: list[threading.Event],
) -> Iterator[AcpHarness]:
    """Сервер со штатными инструментами; клиент умеет формы (одобрение плана)."""
    acp_environment(monkeypatch, tmp_path_factory.mktemp("process-cwd"))
    harness = AcpHarness(factory)
    with ExitStack() as teardown:
        teardown.enter_context(closing(harness))
        teardown.callback(_release_all, releases)  # раньше closing: обратный порядок
        harness.initialize({"elicitation": {"form": {}}})
        yield harness


@pytest.fixture
def backend_default(acp: AcpHarness, monkeypatch: pytest.MonkeyPatch) -> None:
    """Модель не задана: сессия берёт модель бэкенда по умолчанию (первую в каталоге)."""
    monkeypatch.delenv("GIGACREW_MODEL")


def _release_all(releases: list[threading.Event]) -> None:
    for release in releases:
        release.set()


@pytest.fixture
def root(project: Path) -> Path:
    """Проект сессии с файлом ``notes.txt`` и скиллом ``review``."""
    (project / NOTES).write_text(NOTES_TEXT, encoding="utf-8")
    _add_skill(project, "review", "Ревью изменений")
    return project.resolve()


def _add_skill(project: Path, name: str, description: str) -> None:
    """Скилл проекта ``name`` с телом :data:`SKILL_BODY`."""
    skill_dir = project / ".gigacode" / "skills" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{SKILL_BODY}\n",
        encoding="utf-8",
    )


def _new(acp: AcpHarness, cwd: Path, provider: ScriptedProvider | None = None) -> Json:
    """Ответ ``session/new`` в ``cwd`` (с провайдером ``provider``, если задан)."""
    if provider is not None:
        acp.factory.add(provider)
    response: Json = acp.call(NEW, {"cwd": str(cwd), "mcpServers": []})
    return response


def _open(acp: AcpHarness, cwd: Path, provider: ScriptedProvider | None = None) -> str:
    session_id: str = _new(acp, cwd, provider)["sessionId"]
    return session_id


def _agent(acp: AcpHarness, session_id: str) -> Agent:
    return acp.server.session(session_id).agent


def _ask(acp: AcpHarness, session_id: str, text: str) -> Any:
    """Ход целиком: ``stopReason`` (ошибка JSON-RPC роняет тест)."""
    prompt = [{"type": "text", "text": text}]
    return acp.call(PROMPT, {"sessionId": session_id, "prompt": prompt})["stopReason"]


def _updates(acp: AcpHarness, session_id: str) -> list[Json]:
    """Забирает накопленные ``session/update`` сессии ``session_id``."""
    notifications = acp.client.take_notifications(UPDATE)
    return [params["update"] for params in notifications if params["sessionId"] == session_id]


def _of_kind(updates: list[Json], kind: str) -> list[Json]:
    return [update for update in updates if update["sessionUpdate"] == kind]


def _model_option(config_options: list[Json]) -> Json:
    found = [option for option in config_options if option["id"] == MODEL]
    assert len(found) == 1, config_options
    return found[0]


def _option_values(option: Json) -> list[str]:
    return [choice["value"] for choice in option["options"]]


def _set_model(acp: AcpHarness, session_id: str, model: str) -> Json:
    params = {"sessionId": session_id, "configId": MODEL, "value": model}
    response: Json = acp.call(SET_OPTION, params)
    return response


def _commands(acp: AcpHarness, session_id: str) -> list[Json]:
    """Команды из ``available_commands_update`` сессии (дожидается уведомления)."""

    def is_commands(message: Json) -> bool:
        params = message.get("params") or {}
        update = params.get("update") or {}
        return (
            message.get("method") == UPDATE
            and params.get("sessionId") == session_id
            and update.get("sessionUpdate") == COMMANDS_UPDATE
        )

    commands: list[Json] = acp.client.wait_for(is_commands)["params"]["update"]["availableCommands"]
    return commands


def _user_texts(provider: ScriptedProvider) -> list[str]:
    """Запросы пользователя, какими их получила модель в последнем обращении."""
    messages = provider.requests[-1]["messages"]
    return [message.content for message in messages if message.role == "user"]


def _tool_results(provider: ScriptedProvider) -> str:
    """Результаты инструментов, которые модель видела в последнем обращении."""
    messages = provider.requests[-1]["messages"]
    return "\n".join(message.content or "" for message in messages if message.role == "function")


# ------------------------------- режимы ------------------------------- #
def test_new_session_reports_permission_modes(acp: AcpHarness, root: Path) -> None:
    """``modes``: id режимов — значения PermissionMode с названием и описанием."""
    modes = _new(acp, root)["modes"]

    assert modes["currentModeId"] == PermissionMode.AUTO.value
    available = modes["availableModes"]
    assert [mode["id"] for mode in available] == [mode.value for mode in PermissionMode]
    for mode in available:
        assert isinstance(mode["name"], str)
        assert mode["name"].strip()
        assert isinstance(mode["description"], str)
        assert mode["description"].strip()


def test_new_session_mode_follows_configuration(
    acp: AcpHarness,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GIGACREW_MODE", PermissionMode.PLAN.value)

    response = _new(acp, root)

    assert response["modes"]["currentModeId"] == PermissionMode.PLAN.value
    assert _agent(acp, response["sessionId"]).mode is PermissionMode.PLAN


def test_set_mode_switches_agent_mode(acp: AcpHarness, root: Path) -> None:
    session_id = _open(acp, root)

    plan = {"sessionId": session_id, "modeId": PermissionMode.PLAN.value}
    response = acp.call(SET_MODE, plan)

    assert isinstance(response, dict)
    assert not response
    assert _agent(acp, session_id).mode is PermissionMode.PLAN
    acp.call(SET_MODE, {"sessionId": session_id, "modeId": PermissionMode.AUTO.value})
    assert _agent(acp, session_id).mode is PermissionMode.AUTO


@pytest.mark.parametrize(
    ("session", "mode_id"),
    [("known", "yolo"), ("known", ""), ("unknown", PermissionMode.PLAN.value)],
    ids=["unknown-mode", "empty-mode", "unknown-session"],
)
def test_set_mode_rejects_unknown_mode_or_session(
    acp: AcpHarness,
    root: Path,
    session: str,
    mode_id: str,
) -> None:
    session_id = _open(acp, root)
    target = session_id if session == "known" else "no-such-session"

    error = acp.call_error(SET_MODE, {"sessionId": target, "modeId": mode_id})

    assert error["code"] == INVALID_PARAMS
    assert _agent(acp, session_id).mode is PermissionMode.AUTO


def test_plan_mode_blocks_edits_without_asking(acp: AcpHarness, root: Path) -> None:
    """В режиме плана правка не выполняется и не спрашивается; режим ход не меняет."""
    provider = ScriptedProvider([
        tool_turn("write_file", {"path": NOTES, "content": "changed\n"}),
        text_turn("Опишу в плане"),
    ])
    session_id = _open(acp, root, provider)
    acp.call(SET_MODE, {"sessionId": session_id, "modeId": PermissionMode.PLAN.value})

    assert _ask(acp, session_id, "поправь заметки") == END_TURN

    assert (root / NOTES).read_text(encoding="utf-8") == NOTES_TEXT
    assert PLAN_BLOCKED_NOTE.split("{path}")[0] in _tool_results(provider)
    assert not _of_kind(_updates(acp, session_id), MODE_UPDATE)


def test_approved_plan_reports_mode_change(acp: AcpHarness, root: Path) -> None:
    """Одобренный план переводит агента в «авто»: ``current_mode_update`` до ответа хода.

    План одобряется формой ``elicitation/create``, как в T3.
    """
    provider = _LateScript()
    session_id = _open(acp, root, provider)
    agent = _agent(acp, session_id)
    plan_path = agent.plan_path
    assert plan_path is not None
    provider.script([
        tool_turn("write_file", {"path": str(plan_path.relative_to(root)), "content": "1. шаг\n"}),
        tool_turn("exit_plan_mode", {}),
        text_turn("Выполняю"),
    ])
    acp.call(SET_MODE, {"sessionId": session_id, "modeId": PermissionMode.PLAN.value})
    msg_id = acp.begin(
        PROMPT,
        {"sessionId": session_id, "prompt": [{"type": "text", "text": "спланируй"}]},
    )

    request = acp.wait_request(ELICITATION)
    before_approval = _mode_ids(_updates(acp, session_id))
    approval = {_approval_field(request["params"]): APPROVE}
    acp.client.respond(request["id"], {"action": "accept", "content": approval})
    reply = acp.reply(msg_id)

    assert reply["result"]["stopReason"] == END_TURN
    # Эхо режима «план» после set_mode допустимо, «авто» до одобрения — нет.
    assert PermissionMode.AUTO.value not in before_approval
    assert _mode_ids(_updates(acp, session_id))[-1:] == [PermissionMode.AUTO.value]
    assert agent.mode is PermissionMode.AUTO


def _mode_ids(updates: list[Json]) -> list[str]:
    """Режимы из ``current_mode_update`` по порядку."""
    return [update["currentModeId"] for update in _of_kind(updates, MODE_UPDATE)]


def _approval_field(params: Json) -> str:
    """Поле формы, среди вариантов которого есть одобрение плана."""
    fields = params["requestedSchema"]["properties"]
    found: list[str] = [
        name
        for name, schema in fields.items()
        if any(choice.get("const") == APPROVE for choice in schema.get("oneOf", ()))
    ]
    assert len(found) == 1, fields
    return found[0]


# ------------------------------- модель ------------------------------- #
@pytest.mark.usefixtures("backend_default")
def test_model_option_lists_backend_catalogue(acp: AcpHarness, root: Path) -> None:
    config_options = _new(acp, root, CatalogueProvider())["configOptions"]
    option = _model_option(config_options)

    assert option["category"] == MODEL
    assert option["type"] == "select"
    assert option["name"].strip()
    assert option["currentValue"] == CATALOGUE[0]
    assert _option_values(option) == list(CATALOGUE)
    assert all(choice["name"].strip() for choice in option["options"])
    # Параметр категории «mode» увёл бы переключение режима T3 мимо session/set_mode.
    assert all(each.get("category") != "mode" for each in config_options), config_options


def test_model_option_without_catalogue_is_current_model(acp: AcpHarness, root: Path) -> None:
    option = _model_option(_new(acp, root)["configOptions"])

    assert option["currentValue"] == "M-default"
    assert _option_values(option) == ["M-default"]


@pytest.mark.usefixtures("backend_default")
def test_set_model_option_applies_to_next_prompt(acp: AcpHarness, root: Path) -> None:
    provider = CatalogueProvider([text_turn("раз"), text_turn("два")])
    session_id = _open(acp, root, provider)
    _ask(acp, session_id, "первый")

    response = _set_model(acp, session_id, CATALOGUE[1])

    assert _model_option(response["configOptions"])["currentValue"] == CATALOGUE[1]
    assert _agent(acp, session_id).model == CATALOGUE[1]
    assert _ask(acp, session_id, "второй") == END_TURN
    assert [request["model"] for request in provider.requests] == list(CATALOGUE)


def test_model_without_catalogue_may_be_any(acp: AcpHarness, root: Path) -> None:
    """Каталога нет — модель любая (в T3 — «свои модели»), она становится вариантом."""
    provider = ScriptedProvider([text_turn("ок")])
    session_id = _open(acp, root, provider)

    option = _model_option(_set_model(acp, session_id, "M-other")["configOptions"])

    assert option["currentValue"] == "M-other"
    assert "M-other" in _option_values(option)
    _ask(acp, session_id, "привет")
    assert provider.requests[0]["model"] == "M-other"


@pytest.mark.usefixtures("backend_default")
@pytest.mark.parametrize(
    ("config_id", "model", "session"),
    [
        (MODEL, "cat-x", "known"),
        (MODEL, "", "known"),
        ("temperature", "0.5", "known"),
        (MODEL, CATALOGUE[1], "unknown"),
    ],
    ids=["not-in-catalogue", "empty", "unknown-option", "unknown-session"],
)
def test_set_config_option_rejects_bad_requests(
    acp: AcpHarness,
    root: Path,
    config_id: str,
    model: str,
    session: str,
) -> None:
    session_id = _open(acp, root, CatalogueProvider())
    target = session_id if session == "known" else "no-such-session"

    error = acp.call_error(
        SET_OPTION,
        {"sessionId": target, "configId": config_id, "value": model},
    )

    assert error["code"] == INVALID_PARAMS
    assert _agent(acp, session_id).model == CATALOGUE[0]


@pytest.mark.usefixtures("backend_default")
def test_set_model_method_switches_model(acp: AcpHarness, root: Path) -> None:
    """``session/set_model`` — для клиентов без ``configOptions``: та же проверка каталога."""
    session_id = _open(acp, root, CatalogueProvider())

    response = acp.call(SET_MODEL, {"sessionId": session_id, "modelId": CATALOGUE[1]})

    assert isinstance(response, dict)
    assert not response
    assert _agent(acp, session_id).model == CATALOGUE[1]
    error = acp.call_error(SET_MODEL, {"sessionId": session_id, "modelId": "cat-x"})
    assert error["code"] == INVALID_PARAMS
    assert _agent(acp, session_id).model == CATALOGUE[1]


def test_configured_catalogue_model_is_current_and_may_be_resent(
    acp: AcpHarness,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Настроенная модель — текущий вариант; повторная установка её же — не ошибка.

    T3 не покажет ``currentValue`` вне ``options`` и может прислать текущее значение
    снова. Настроенная модель вне каталога сюда не доходит: ``session/new`` с ней — ошибка,
    как запуск CLI (``select_model``, тесты шага 2).
    """
    monkeypatch.setenv("GIGACREW_MODEL", CATALOGUE[1])
    response = _new(acp, root, CatalogueProvider())
    session_id = response["sessionId"]
    option = _model_option(response["configOptions"])

    assert option["currentValue"] == CATALOGUE[1]
    assert _option_values(option) == list(CATALOGUE)
    resent = _model_option(_set_model(acp, session_id, CATALOGUE[1])["configOptions"])
    assert resent["currentValue"] == CATALOGUE[1]
    assert _agent(acp, session_id).model == CATALOGUE[1]


@pytest.mark.usefixtures("backend_default")
def test_model_chosen_during_turn_applies_to_next_request(
    acp: AcpHarness,
    root: Path,
    releases: list[threading.Event],
) -> None:
    """``set_config_option`` не блокируется занятой сессией; модель — со следующего обращения.

    Сам T3 модель внутри хода не меняет (выбор применяется перед ``session/prompt``), но
    ответ на запрос не должен ждать конца чужого хода.
    """
    provider = _GatedCatalogue([tool_turn("read_file", {"path": NOTES}), text_turn("готово")])
    releases.append(provider.release)
    session_id = _open(acp, root, provider)
    msg_id = acp.begin(
        PROMPT,
        {"sessionId": session_id, "prompt": [{"type": "text", "text": "прочитай"}]},
    )
    assert provider.entered.wait(TIMEOUT), "ход не дошёл до модели"

    response = _set_model(acp, session_id, CATALOGUE[1])
    provider.release.set()

    assert _model_option(response["configOptions"])["currentValue"] == CATALOGUE[1]
    assert acp.reply(msg_id)["result"]["stopReason"] == END_TURN
    assert [request["model"] for request in provider.requests] == list(CATALOGUE)


# --------------------------- слэш-команды ---------------------------- #
def test_new_session_advertises_compact_and_skills(acp: AcpHarness, root: Path) -> None:
    session_id = _open(acp, root)

    commands = {command["name"]: command for command in _commands(acp, session_id)}

    assert {"compact", "review"} <= set(commands)
    assert commands["review"]["description"] == "Ревью изменений"
    assert commands["compact"]["description"].strip()
    assert commands["review"]["input"]["hint"].strip()
    # Имена — без «/» (его добавляет клиент) и без «$» (так T3 помечает чужие скиллы);
    # команды терминала (выход, буфер обмена) в T3 смысла не имеют.
    assert all(name[:1] not in {"/", "$"} for name in commands), list(commands)
    assert not {"exit", "quit", "copy"} & set(commands)


def test_compact_command_summarizes_without_model_turn(acp: AcpHarness, root: Path) -> None:
    provider = ScriptedProvider([text_turn("привет")])
    session_id = _open(acp, root, provider)
    _ask(acp, session_id, "здравствуй")

    assert _ask(acp, session_id, "/compact оставь только решения") == END_TURN

    assert provider.calls == 1  # ход модели был только у первого запроса
    assert len(provider.summary_requests) == 1
    summary_messages = provider.summary_requests[0]["messages"]
    assert any("оставь только решения" in (message.content or "") for message in summary_messages)


def test_compact_with_empty_history_ends_turn_quietly(acp: AcpHarness, root: Path) -> None:
    provider = ScriptedProvider()
    session_id = _open(acp, root, provider)

    assert _ask(acp, session_id, "/compact") == END_TURN

    assert provider.calls == 0
    assert not provider.summary_requests


def _answer_text(acp: AcpHarness, session_id: str) -> str:
    """Текст ответа агента за ход (куски ``agent_message_chunk``)."""
    chunks = _of_kind(_updates(acp, session_id), "agent_message_chunk")
    return "".join(chunk["content"]["text"] for chunk in chunks)


@pytest.mark.parametrize(
    ("history", "expected"),
    [(True, "контекст сжат"), (False, "сжимать нечего")],
    ids=["compacted", "empty-history"],
)
def test_compact_command_reports_its_result(
    acp: AcpHarness,
    root: Path,
    *,
    history: bool,
    expected: str,
) -> None:
    """Пустой ответ на ``/compact`` в T3 выглядит как сбой: итог сжатия виден в ответе."""
    # Краткое содержание должно выйти короче истории — иначе ядро откажется сжимать.
    session_id = _open(acp, root, ScriptedProvider([text_turn("привет " * 200)]))
    if history:
        _ask(acp, session_id, "здравствуй " * 200)
    _updates(acp, session_id)

    assert _ask(acp, session_id, "/compact") == END_TURN

    assert expected in _answer_text(acp, session_id)


def test_compact_failure_is_rpc_error_and_session_recovers(acp: AcpHarness, root: Path) -> None:
    provider = ScriptedProvider(
        [text_turn("привет"), text_turn("снова тут")],
        summaries=[LLMError("модель недоступна")],
    )
    session_id = _open(acp, root, provider)
    _ask(acp, session_id, "здравствуй")

    error = acp.call_error(
        PROMPT,
        {"sessionId": session_id, "prompt": [{"type": "text", "text": "/compact"}]},
    )

    assert error["code"] == INTERNAL_ERROR
    assert "модель недоступна" in error["message"]
    assert _ask(acp, session_id, "ещё раз") == END_TURN
    assert any("здравствуй" in text for text in _user_texts(provider))  # история не сжата


def test_skill_command_expands_like_cli(acp: AcpHarness, root: Path) -> None:
    provider = ScriptedProvider([text_turn("сделано")])
    session_id = _open(acp, root, provider)

    assert _ask(acp, session_id, "/review только tests/") == END_TURN

    request = _user_texts(provider)[-1]
    assert "/review только tests/" in request
    assert SKILL_BODY in request
    assert "только tests/" in request.split(SKILL_BODY, 1)[1]


def test_skill_named_like_repl_command_is_a_command(acp: AcpHarness, root: Path) -> None:
    """Команд REPL (``/model``, ``/mode``…) в ACP нет: скилл с таким именем работает."""
    _add_skill(root, "model", "Модель данных проекта")
    provider = ScriptedProvider([text_turn("сделано")])
    session_id = _open(acp, root, provider)

    names = {command["name"] for command in _commands(acp, session_id)}
    assert _ask(acp, session_id, "/model users") == END_TURN

    assert "model" in names
    request = _user_texts(provider)[-1]
    assert SKILL_BODY in request
    assert "users" in request.split(SKILL_BODY, 1)[1]


def test_unknown_slash_command_is_plain_text(acp: AcpHarness, root: Path) -> None:
    provider = ScriptedProvider([text_turn("не знаю такой")])
    session_id = _open(acp, root, provider)

    assert _ask(acp, session_id, "/xyz что-то") == END_TURN

    assert "/xyz что-то" in _user_texts(provider)[-1]
    assert not provider.summary_requests
