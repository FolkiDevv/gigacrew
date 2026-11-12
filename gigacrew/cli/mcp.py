"""MCP-серверы в CLI: подтверждение, запуск и описание для ``/mcp``.

MCP-сервер — программа, которую gigacrew запускает, или сервис, которому он передаёт
данные: серверы проекта (``.mcp.json``, ``.gigacode/settings*.json``) приходят вместе с
репозиторием, а в ``~/.gigacode/settings.json`` запись может дописать и сама модель
(через ``run_shell``). Поэтому серверы работают, только если пользователь подтвердил их
(отпечатки — в ``~/.gigacode/trusted_mcp.json``; изменились — спросим снова) или передал
``--trust-mcp``. Без терминала и в ``-p`` спросить некого: неподтверждённые серверы не
запускаются, о чём выводится предупреждение.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from gigacrew.project.mcp_config import (
    HTTP,
    McpServerConfig,
    McpSettings,
    load_mcp_settings,
    mcp_files,
    servers_fingerprint,
)
from gigacrew.project.settings import PROJECT_SOURCES, SettingsSource, TrustStore
from gigacrew.project.workspace import Workspace, user_data_dir
from gigacrew.tools.base import ToolRegistry
from gigacrew.tools.mcp import McpHub, McpServer
from gigacrew.tools.mcp_schema import tool_name
from gigacrew.ui.console import Console, visible_text

__all__ = [
    "MCP_TRUST_FILE",
    "confirm_servers",
    "connect_mcp",
    "describe_hub",
    "describe_servers",
    "mcp_trust_store",
]

MCP_TRUST_FILE = "trusted_mcp.json"
_SEPARATOR = ", "
_TOOLS_LISTED = 30  # инструментов сервера в выводе /mcp


@dataclass(frozen=True)
class _Pending:
    """Неподтверждённые серверы одного владельца (пользователь или проект)."""

    owner: str  # «пользователя», «проекта»
    sources: tuple[SettingsSource, ...]
    servers: tuple[McpServerConfig, ...]
    remember: Callable[[], None]  # записать отпечаток в TrustStore (OSError)


def mcp_trust_store() -> TrustStore:
    """Подтверждённые серверы: ``~/.gigacode/trusted_mcp.json``."""
    return TrustStore(user_data_dir() / MCP_TRUST_FILE)


def _pending_servers(
    settings: McpSettings,
    workspace: Workspace,
    store: TrustStore,
) -> list[_Pending]:
    """Неподтверждённые серверы пользователя и проекта; пусто — подтверждать нечего."""
    pending: list[_Pending] = []
    user = servers_fingerprint(settings.user_servers)
    if user and not store.is_user_trusted(user):
        pending.append(
            _Pending(
                "пользователя",
                ("user",),
                settings.user_servers,
                lambda: store.trust_user(user),
            ),
        )
    project = servers_fingerprint(settings.project_servers)
    if project and not store.is_trusted(workspace.root, project):
        pending.append(
            _Pending(
                "проекта",
                PROJECT_SOURCES,
                settings.project_servers,
                lambda: store.trust(workspace.root, project),
            ),
        )
    return pending


def confirm_servers(
    settings: McpSettings,
    workspace: Workspace,
    ui: Console,
    *,
    interactive: bool,
    store: TrustStore,
) -> McpSettings:
    """Только подтверждённые серверы; новые и изменённые подтверждаются одним вопросом."""
    pending = _pending_servers(settings, workspace, store)
    if not pending:
        return settings
    owners = " и ".join(entry.owner for entry in pending)
    sources = tuple(source for entry in pending for source in entry.sources)
    servers = tuple(server for entry in pending for server in entry.servers)
    if not interactive:
        files = _listed(sorted({server.origin for server in servers}))
        ui.warn(
            f"MCP-серверы {owners} ({files}) не подтверждены и не запущены: запустите "
            "gigacrew интерактивно, чтобы подтвердить их, или передайте --trust-mcp",
        )
        return settings.without(*sources)
    ui.warn(
        f"новые или изменённые MCP-серверы {owners} — gigacrew запустит эти программы "
        "(или подключится к этим адресам) и будет передавать им данные:",
    )
    ui.print(describe_servers(servers), markup=False, highlight=False)
    if not ui.ask(f"Разрешить MCP-серверы {owners}?", dangerous=True):
        ui.info(f"MCP-серверы {owners} не запущены (спрошу снова при следующем запуске)")
        return settings.without(*sources)
    try:
        for entry in pending:
            entry.remember()
    except OSError as error:
        ui.warn(f"подтверждение не сохранено ({error}) — при следующем запуске спрошу снова")
    return settings


def connect_mcp(  # noqa: WPS211
    workspace: Workspace,
    ui: Console,
    registry: ToolRegistry,
    *,
    interactive: bool,
    trust: bool = False,
    version: str = "",
    store: TrustStore | None = None,
) -> McpHub:
    """Подтверждённые серверы из настроек — запущены, их инструменты — в ``registry``.

    ``trust`` (``--trust-mcp``) — все серверы доверены на этот запуск, без вопроса и без
    записи отпечатков. Сервер, который не запустился, — предупреждение; Ctrl+C во время
    запуска закрывает уже запущенные серверы и пробрасывается.
    """
    settings, warnings = load_mcp_settings(workspace)
    for warning in warnings:
        ui.warn(warning)
    if not trust:
        settings = confirm_servers(
            settings,
            workspace,
            ui,
            interactive=interactive,
            store=mcp_trust_store() if store is None else store,
        )
    configs = settings.effective()
    hub = McpHub([McpServer(config, workspace, version=version) for config in configs])
    if not configs:
        return hub
    names = _listed(config.name for config in configs)
    ui.system(f"MCP-серверы: подключение ({names})…")
    hub.connect()
    for server in hub.servers:
        if server.error:
            ui.warn(f"MCP-сервер {server.name} не подключён: {server.error}")
    for warning in hub.register(registry):
        ui.warn(warning)
    return hub


def _shown(text: str) -> str:
    """Текст для подтверждения: невидимые символы — как ``⟨U+202E⟩``."""
    return visible_text(text, numbered=False).plain


def _listed(names: Iterable[str]) -> str:
    return _SEPARATOR.join(names)


def _target(server: McpServerConfig) -> str:
    """Что запускается или куда подключение."""
    if server.transport == HTTP:
        return f"http {server.url}"
    return " ".join((server.command, *server.args))


def describe_servers(servers: tuple[McpServerConfig, ...]) -> str:
    """Серверы для подтверждения: полная команда или адрес, окружение, заголовки, файл.

    Пользователь подтверждает ровно то, что запустится: невидимые символы показываются
    как ``⟨U+202E⟩``. Значения заголовков не показываются (там обычно ключи), имена — да.
    """
    lines: list[str] = []
    for server in servers:
        lines.extend((
            f"  {_shown(server.name)}  ({server.origin})",
            f"    {_shown(_target(server))}",
        ))
        lines.extend(f"    env {_shown(key)}={_shown(setting)}" for key, setting in server.env)
        if server.headers:
            names = _listed(_shown(key) for key, _ in server.headers)
            lines.append(f"    заголовки: {names}")
        if server.auto_approve:
            approved = _listed(sorted(_shown(tool) for tool in server.auto_approve))
            lines.append(f"    без вопроса: {approved}")
    return "\n".join(lines)


def _server_lines(server: McpServer) -> list[str]:
    config = server.config
    head = f"  {config.name} · {config.transport} · {config.origin}"
    if not server.connected:
        reason = server.error or "не подключён"
        return [head, f"    ошибка: {reason}"]
    names = [tool_name(config.name, str(described["name"])) for described in server.tools]
    about = server.about
    title = f"{about.name} {about.version}".strip() if about else ""
    label = f" ({title})" if title else ""
    lines = [head, f"    подключён{label}: инструментов — {len(names)}"]
    if names:
        lines.append(_tools_line(names))
    if config.auto_approve:
        approved = _listed(sorted(config.auto_approve))
        lines.append(f"    без вопроса: {approved}")
    return lines


def _tools_line(names: list[str]) -> str:
    """Имена инструментов: первые :data:`_TOOLS_LISTED` и сколько ещё."""
    hidden = len(names) - _TOOLS_LISTED
    more = f" и ещё {hidden}" if hidden > 0 else ""
    shown = _listed(names[:_TOOLS_LISTED])
    return f"    {shown}{more}"


def describe_hub(hub: McpHub | None, workspace: Workspace) -> str:
    """Текст для ``/mcp``: серверы, их состояние и инструменты."""
    files = _listed(origin for _, _, origin in mcp_files(workspace))
    if hub is None or not hub.servers:
        return f"MCP-серверов нет; настройки (mcpServers): {files}"
    lines = [f"MCP-серверы (настройки: {files}):"]
    for server in hub.servers:
        lines.extend(_server_lines(server))
    return "\n".join(lines)
