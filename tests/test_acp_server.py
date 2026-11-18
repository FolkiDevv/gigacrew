"""ACP-сервер: initialize, authenticate, session/new и сборка агента на сессию."""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from collections.abc import Callable, Iterable, Iterator
from contextlib import closing
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, override

import pytest
from acp_harness import AcpHarness, ProviderFactory, acp_environment
from fakes import CatalogueProvider, ScriptedProvider

from gigacrew.cli.acp.rpc import INTERNAL_ERROR, INVALID_PARAMS
from gigacrew.cli.mcp import mcp_trust_store
from gigacrew.llm.base import LLMError, LLMProvider
from gigacrew.project.mcp_config import load_mcp_settings, servers_fingerprint
from gigacrew.project.settings import TrustStore, load_settings, project_fingerprint
from gigacrew.project.workspace import Workspace
from gigacrew.tools.base import ToolRegistry

# Исключение в любом потоке сервера роняет тест, а не теряется.
pytestmark = pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")

# Формат id чата (ChatStore.new_id): время создания и случайный суффикс.
CHAT_ID = re.compile(r"\d{8}-\d{6}-[0-9a-f]{8}")
# Инструмент поддельного MCP-сервера проекта (сервер «probe»).
PROBE_ECHO = "mcp__probe__echo"
FAKE_MCP_SERVER = Path(__file__).with_name("fake_mcp_server.py")
# Создаёт файл-маркер sys.argv[1]: признак того, что команду запустили.
TOUCH_SCRIPT = "import sys\nopen(sys.argv[1], 'w').close()\n"
# Пишет свой pid в маркер sys.argv[1] и становится MCP-сервером sys.argv[2] (тот же pid).
LAUNCH_SCRIPT = (
    "import os, sys\n"
    "with open(sys.argv[1], 'w') as marker:\n"
    "    marker.write(str(os.getpid()))\n"
    "os.execv(sys.executable, [sys.executable, sys.argv[2]])\n"
)

type _Setup = Callable[[pytest.MonkeyPatch, ProviderFactory, Path], None]
type _Prepare = Callable[[pytest.MonkeyPatch], LLMProvider]


@pytest.fixture
def factory() -> ProviderFactory:
    return ProviderFactory()


@pytest.fixture
def workdir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Пустой текущий каталог процесса сервера (не проект сессии)."""
    return tmp_path_factory.mktemp("process-cwd")


@pytest.fixture
def acp(
    monkeypatch: pytest.MonkeyPatch,
    factory: ProviderFactory,
    workdir: Path,
) -> Iterator[AcpHarness]:
    acp_environment(monkeypatch, workdir)
    with closing(AcpHarness(factory)) as harness:
        yield harness


class _ClosingProvider(CatalogueProvider):
    """Провайдер с каталогом, который помнит, закрыли ли его."""

    def __init__(self, *, connect_errors: tuple[BaseException, ...] = ()) -> None:
        super().__init__(connect_errors=connect_errors)
        self.closed = False

    @override
    def close(self) -> None:
        self.closed = True


class _ClosingScripted(ScriptedProvider):
    """Провайдер без каталога моделей, который помнит, закрыли ли его."""

    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    @override
    def close(self) -> None:
        self.closed = True


@dataclass(frozen=True)
class _Probe:
    """Проект с хуком SessionStart и MCP-сервером, которые оставляют маркеры."""

    project: Path
    hook_marker: Path
    mcp_marker: Path  # pid запущенного MCP-сервера


def _probe_project(tmp_path: Path) -> _Probe:
    project = (tmp_path / "project").resolve()
    (project / ".gigacode").mkdir(parents=True)
    touch = tmp_path / "touch.py"
    touch.write_text(TOUCH_SCRIPT, encoding="utf-8")
    launch = tmp_path / "launch_mcp.py"
    launch.write_text(LAUNCH_SCRIPT, encoding="utf-8")
    probe = _Probe(project, tmp_path / "hook-ran", tmp_path / "mcp-pid")
    command = shlex.join([sys.executable, str(touch), str(probe.hook_marker)])
    hooks = {"SessionStart": [{"hooks": [{"type": "command", "command": command}]}]}
    (project / ".gigacode" / "settings.json").write_text(
        json.dumps({"hooks": hooks}),
        encoding="utf-8",
    )
    server = {
        "command": sys.executable,
        "args": [str(launch), str(probe.mcp_marker), str(FAKE_MCP_SERVER)],
    }
    (project / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"probe": server}}),
        encoding="utf-8",
    )
    return probe


def _trust_project(project: Path) -> None:
    """Хуки и MCP-серверы проекта подтверждены раньше (как ответ «да» в терминале)."""
    workspace = Workspace(project)
    hooks, _ = load_settings(workspace)
    TrustStore().trust(workspace.root, project_fingerprint(hooks))
    servers, _ = load_mcp_settings(workspace)
    mcp_trust_store().trust(workspace.root, servers_fingerprint(servers.project_servers))


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _new_session_params(cwd: Path) -> dict[str, Any]:
    return {"cwd": str(cwd), "mcpServers": []}


def _assert_session_error(error: dict[str, Any], fragments: Iterable[str]) -> None:
    """Понятная ошибка сессии: внутренняя ошибка с сутью в сообщении, без трассировки."""
    assert error["code"] == INTERNAL_ERROR
    message = error["message"]
    for fragment in fragments:
        assert fragment in message
    assert "Traceback" not in message


# --- initialize и authenticate ---


@pytest.mark.parametrize("requested", [1, 7])
def test_initialize_answers_protocol_version_one(acp: AcpHarness, requested: int) -> None:
    response = acp.call("initialize", {"protocolVersion": requested})

    assert response["protocolVersion"] == 1


def test_initialize_describes_agent(acp: AcpHarness) -> None:
    response = acp.initialize()

    agent_info = response["agentInfo"]
    assert agent_info["name"] == "gigacrew"
    assert agent_info["title"]
    assert agent_info["version"] == metadata.version("gigacrew")
    capabilities = response["agentCapabilities"]
    assert capabilities["loadSession"] is False  # session/load — шаг 6
    assert capabilities["promptCapabilities"]["image"] is False
    assert capabilities["promptCapabilities"]["embeddedContext"] is True
    assert response["authMethods"] == []


@pytest.mark.parametrize(
    "params",
    [None, [], {}, {"protocolVersion": "latest"}],
    ids=["no-params", "array", "no-version", "version-not-number"],
)
def test_initialize_rejects_bad_params(acp: AcpHarness, params: Any) -> None:
    assert acp.call_error("initialize", params)["code"] == INVALID_PARAMS


def test_authenticate_has_nothing_to_check(acp: AcpHarness) -> None:
    acp.initialize()

    assert acp.call("authenticate", {"methodId": "none"}) == {}


# --- session/new: параметры ---


def _relative_cwd(root: Path) -> Any:
    # Каталог существует относительно cwd процесса: отказ — именно за относительность.
    (Path.cwd() / "relative" / "dir").mkdir(parents=True)
    return {"cwd": "relative/dir", "mcpServers": []}


def _missing_cwd(root: Path) -> Any:
    return _new_session_params(root / "missing")


def _file_cwd(root: Path) -> Any:
    path = root / "file.txt"
    path.write_text("не каталог", encoding="utf-8")
    return _new_session_params(path)


def _no_cwd(root: Path) -> Any:
    return {"mcpServers": []}


@pytest.mark.parametrize("build", [_relative_cwd, _missing_cwd, _file_cwd, _no_cwd])
def test_session_new_needs_absolute_existing_directory(
    acp: AcpHarness,
    factory: ProviderFactory,
    project: Path,
    build: Callable[[Path], Any],
) -> None:
    acp.initialize()

    assert acp.call_error("session/new", build(project))["code"] == INVALID_PARAMS
    assert not factory.configs  # до конфигурации и провайдера дело не дошло


def test_session_new_without_initialize_is_allowed(acp: AcpHarness, project: Path) -> None:
    # Решение: initialize не обязателен — сервер не хранит состояние рукопожатия.
    assert CHAT_ID.fullmatch(acp.new_session(project))


def test_client_mcp_servers_accepted_but_not_started(
    acp: AcpHarness,
    project: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    scripts = tmp_path_factory.mktemp("client-mcp")
    touch = scripts / "touch.py"
    touch.write_text(TOUCH_SCRIPT, encoding="utf-8")
    marker = scripts / "client-mcp-ran"
    servers = [
        {"name": "docs", "command": sys.executable, "args": [str(touch), str(marker)], "env": []},
        {"type": "http", "name": "web", "url": "https://mcp.test/", "headers": []},
    ]
    acp.initialize()

    # Серверы клиента пока не подключаются, но и не ошибка.
    assert CHAT_ID.fullmatch(acp.new_session(project, servers))
    assert not marker.exists()


# --- session/new: сессия и агент ---


def test_session_id_is_new_chat_id(acp: AcpHarness, project: Path) -> None:
    acp.initialize()

    first = acp.new_session(project)
    second = acp.new_session(project)

    assert CHAT_ID.fullmatch(first)
    assert CHAT_ID.fullmatch(second)
    assert first != second
    assert acp.server.session(first).agent.chat_id == first
    assert acp.server.session(second).agent.chat_id == second


def test_sessions_load_config_of_their_own_cwd(
    acp: AcpHarness,
    factory: ProviderFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GIGACREW_MODEL")  # модель — из .env каждого проекта
    roots = [tmp_path / "alpha", tmp_path / "beta"]
    for root in roots:
        root.mkdir()
        (root / ".env").write_text(f"GIGACREW_MODEL=model-{root.name}\n", encoding="utf-8")
    acp.initialize()

    sessions = [acp.server.session(acp.new_session(cwd)) for cwd in roots]

    assert [config.project_root for config in factory.configs] == [cwd.resolve() for cwd in roots]
    assert [session.agent.model for session in sessions] == ["model-alpha", "model-beta"]
    assert [session.agent.workspace.root for session in sessions] == [
        cwd.resolve() for cwd in roots
    ]


def test_env_of_process_cwd_does_not_leak_into_session(
    acp: AcpHarness,
    factory: ProviderFactory,
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # .env проекта старше .env текущего каталога, поэтому модели в проекте нет: иначе
    # утечка не видна. Сервер обязан загружать конфигурацию с cwd = каталог сессии.
    monkeypatch.delenv("GIGACREW_MODEL")
    (Path.cwd() / ".env").write_text("GIGACREW_MODEL=from-process-cwd\n", encoding="utf-8")
    factory.add(CatalogueProvider())
    acp.initialize()

    session = acp.server.session(acp.new_session(project))

    assert not factory.configs[0].model
    assert session.agent.model == "cat-a"  # модель по умолчанию из каталога бэкенда


def test_gigacode_instance_gets_gigacode_config(
    acp: AcpHarness,
    factory: ProviderFactory,
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GIGACREW_PROVIDER", "gigacode")
    monkeypatch.setenv("GIGACREW_USER_AGENT", "GigaCode-Test/1.0")
    acp.initialize()

    acp.new_session(project)

    assert factory.configs[0].provider == "gigacode"


def _no_catalogue(monkeypatch: pytest.MonkeyPatch) -> LLMProvider:
    return ScriptedProvider()


def _catalogue_default(monkeypatch: pytest.MonkeyPatch) -> LLMProvider:
    monkeypatch.delenv("GIGACREW_MODEL")
    return CatalogueProvider()


def _catalogue_chosen(monkeypatch: pytest.MonkeyPatch) -> LLMProvider:
    monkeypatch.setenv("GIGACREW_MODEL", "cat-b")
    return CatalogueProvider()


@pytest.mark.parametrize(
    ("prepare", "model"),
    [
        (_no_catalogue, "M-default"),
        (_catalogue_default, "cat-a"),
        (_catalogue_chosen, "cat-b"),
    ],
)
def test_session_model_chosen_like_cli(
    acp: AcpHarness,
    factory: ProviderFactory,
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepare: _Prepare,
    model: str,
) -> None:
    provider = prepare(monkeypatch)
    factory.add(provider)
    acp.initialize()

    session = acp.server.session(acp.new_session(project))

    assert session.agent.model == model
    if isinstance(provider, CatalogueProvider):
        assert provider.connects == 1  # каталог моделей — после подключения


def _model_not_in_catalogue(monkeypatch: pytest.MonkeyPatch) -> LLMProvider:
    monkeypatch.setenv("GIGACREW_MODEL", "nope")
    return _ClosingProvider()


def _model_not_set(monkeypatch: pytest.MonkeyPatch) -> LLMProvider:
    monkeypatch.delenv("GIGACREW_MODEL")  # и каталога нет: модель взять неоткуда
    return _ClosingScripted()


@pytest.mark.parametrize(
    ("prepare", "fragments"),
    [
        (_model_not_in_catalogue, ("nope", "cat-a", "cat-b")),
        (_model_not_set, ("GIGACREW_MODEL",)),
    ],
)
def test_unknown_model_is_session_error_and_provider_closed(
    acp: AcpHarness,
    factory: ProviderFactory,
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepare: _Prepare,
    fragments: tuple[str, ...],
) -> None:
    provider = prepare(monkeypatch)
    factory.add(provider)
    acp.initialize()

    error = acp.call_error("session/new", _new_session_params(project))

    _assert_session_error(error, fragments)
    assert isinstance(provider, _ClosingProvider | _ClosingScripted)
    assert provider.closed


# --- session/new: ошибки конфигурации и подключения ---


def _no_credentials(
    monkeypatch: pytest.MonkeyPatch,
    factory: ProviderFactory,
    project: Path,
) -> None:
    monkeypatch.delenv("GIGACREW_ACCESS_KEY")


def _broken_ca_bundle(
    monkeypatch: pytest.MonkeyPatch,
    factory: ProviderFactory,
    project: Path,
) -> None:
    monkeypatch.setenv("GIGACREW_CA_BUNDLE", "/nonexistent/ca.pem")


def _gigacode_without_user_agent(
    monkeypatch: pytest.MonkeyPatch,
    factory: ProviderFactory,
    project: Path,
) -> None:
    monkeypatch.setenv("GIGACREW_PROVIDER", "gigacode")


def _unknown_provider(
    monkeypatch: pytest.MonkeyPatch,
    factory: ProviderFactory,
    project: Path,
) -> None:
    monkeypatch.setenv("GIGACREW_PROVIDER", "foo")


def _api_url_only_in_project_env(
    monkeypatch: pytest.MonkeyPatch,
    factory: ProviderFactory,
    project: Path,
) -> None:
    # Граница безопасности: адрес API из .env проекта не берётся — его как будто нет.
    monkeypatch.delenv("GIGACREW_API_URL")
    (project / ".env").write_text(
        "GIGACREW_API_URL=https://evil.test\n",
        encoding="utf-8",
    )


def _provider_fails(
    monkeypatch: pytest.MonkeyPatch,
    factory: ProviderFactory,
    project: Path,
) -> None:
    factory.add(RuntimeError("нет токена официального CLI"))


def _token_file_missing(
    monkeypatch: pytest.MonkeyPatch,
    factory: ProviderFactory,
    project: Path,
) -> None:
    factory.add(FileNotFoundError(2, "нет файла токена", "/nonexistent/oauth_creds.json"))


@pytest.mark.parametrize(
    ("setup", "fragment", "factory_calls"),
    [
        (_no_credentials, "GIGACREW_ACCESS_KEY", 0),
        (_broken_ca_bundle, "GIGACREW_CA_BUNDLE", 0),
        (_gigacode_without_user_agent, "GIGACREW_USER_AGENT", 0),
        (_unknown_provider, "GIGACREW_PROVIDER", 0),
        (_api_url_only_in_project_env, "GIGACREW_API_URL", 0),
        (_provider_fails, "нет токена официального CLI", 1),
        (_token_file_missing, "нет файла токена", 1),
    ],
)
def test_configuration_error_is_session_error(
    acp: AcpHarness,
    factory: ProviderFactory,
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
    setup: _Setup,
    fragment: str,
    factory_calls: int,
) -> None:
    setup(monkeypatch, factory, project)
    acp.initialize()

    error = acp.call_error("session/new", _new_session_params(project))

    _assert_session_error(error, [fragment])
    # Реквизиты и адреса проверяются до создания провайдера.
    assert len(factory.configs) == factory_calls
    assert acp.call("authenticate", {"methodId": "none"}) == {}  # сервер работает дальше


def test_connect_error_is_session_error_and_server_recovers(
    acp: AcpHarness,
    factory: ProviderFactory,
    project: Path,
) -> None:
    broken = _ClosingProvider(connect_errors=(LLMError("сеть недоступна"),))
    factory.add(broken)
    acp.initialize()

    error = acp.call_error("session/new", _new_session_params(project))

    _assert_session_error(error, ["сеть недоступна"])
    assert broken.connects == 1
    assert broken.closed
    assert CHAT_ID.fullmatch(acp.new_session(project))  # следующая попытка — с новым провайдером


def test_session_providers_closed_when_input_ends(
    acp: AcpHarness,
    factory: ProviderFactory,
    project: Path,
) -> None:
    providers = [_ClosingScripted(), _ClosingScripted()]
    factory.add(*providers)
    acp.initialize()
    for _ in providers:
        acp.new_session(project)
    assert not any(provider.closed for provider in providers)

    acp.stop()

    assert all(provider.closed for provider in providers)


# --- хуки и MCP-серверы проекта: без вопросов, только подтверждённые ---


def test_untrusted_project_hooks_and_mcp_do_not_run(acp: AcpHarness, tmp_path: Path) -> None:
    probe = _probe_project(tmp_path)
    acp.initialize()

    # Спросить о доверии некого: сессия создаётся без хуков и серверов проекта.
    assert CHAT_ID.fullmatch(acp.new_session(probe.project))
    assert not probe.hook_marker.exists()
    assert not probe.mcp_marker.exists()


def test_trusted_project_hooks_and_mcp_run_and_mcp_stops_at_eof(
    monkeypatch: pytest.MonkeyPatch,
    factory: ProviderFactory,
    workdir: Path,
    tmp_path: Path,
) -> None:
    acp_environment(monkeypatch, workdir)
    probe = _probe_project(tmp_path)
    _trust_project(probe.project)
    registries: list[ToolRegistry] = []

    def registry_factory() -> ToolRegistry:
        registries.append(ToolRegistry())
        return registries[-1]

    with closing(AcpHarness(factory, registry_factory=registry_factory)) as acp:
        acp.initialize()
        acp.new_session(probe.project)

        assert probe.hook_marker.exists()  # SessionStart — при создании сессии
        pid = int(probe.mcp_marker.read_text(encoding="utf-8"))
        # Инструменты MCP — в реестре сессии из registry_factory.
        assert [PROBE_ECHO in registry for registry in registries] == [True]
        assert _alive(pid)

        acp.stop()

        assert not _alive(pid)  # сервер MCP остановлен и дождан


def test_failed_session_leaves_no_mcp_server_running(
    acp: AcpHarness,
    factory: ProviderFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _probe_project(tmp_path)
    _trust_project(probe.project)
    monkeypatch.setenv("GIGACREW_MODEL", "nope")
    factory.add(CatalogueProvider())
    acp.initialize()

    error = acp.call_error("session/new", _new_session_params(probe.project))

    _assert_session_error(error, ["nope"])
    # MCP-серверы могут подключаться и после проверки модели (тогда маркера нет), но
    # запущенный сервер несобранной сессии остановлен до ответа, а не при конце ввода.
    if probe.mcp_marker.exists():
        assert not _alive(int(probe.mcp_marker.read_text(encoding="utf-8")))
