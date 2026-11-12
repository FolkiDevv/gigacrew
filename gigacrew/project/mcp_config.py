"""MCP-серверы из настроек: секция ``"mcpServers"`` и файл ``.mcp.json`` проекта.

Источники по порядку (сервер с тем же именем из более позднего заменяет прежний):

* ``~/.gigacode/settings.json`` — пользователь, общие для всех проектов;
* ``<проект>/.mcp.json`` — проект, формат Claude Code (коммитится);
* ``<проект>/.gigacode/settings.json`` — проект (коммитится);
* ``<проект>/.gigacode/settings.local.json`` — личные настройки в проекте.

Формат — как у Claude Code::

    {
        "mcpServers": {
            "github": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"},
            },
            "docs": {
                "type": "http",
                "url": "https://example.com/mcp",
                "headers": {"Authorization": "Bearer ${DOCS_TOKEN}"},
                "autoApprove": ["search"],
            },
        }
    }

Здесь записи хранятся как написаны: ``${VAR}`` раскрываются при запуске сервера
(:mod:`gigacrew.tools.mcp`). Ошибки в файлах не мешают запуску: сервер с ошибкой
пропускается, вызывающий получает текст предупреждения. Сервер — программа, которую
gigacrew запускает, или сервис, которому он передаёт данные, поэтому серверы работают
только после подтверждения пользователя (как хуки, см. :mod:`gigacrew.cli.mcp`); их
отпечатки — :func:`servers_fingerprint`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final, Literal

from gigacrew.project.settings import PROJECT_SOURCES, SettingsSource
from gigacrew.project.workspace import Workspace, user_settings_file

__all__ = [
    "ANY_TOOL",
    "DEFAULT_STARTUP_TIMEOUT",
    "DEFAULT_TOOL_TIMEOUT",
    "HTTP",
    "MCP_JSON_FILE",
    "STDIO",
    "McpServerConfig",
    "McpSettings",
    "McpTransport",
    "load_mcp_settings",
    "mcp_files",
    "servers_fingerprint",
]

McpTransport = Literal["stdio", "http"]
STDIO: Final = "stdio"
HTTP: Final = "http"

MCP_JSON_FILE = ".mcp.json"
DEFAULT_TOOL_TIMEOUT = 300  # секунд на вызов инструмента
DEFAULT_STARTUP_TIMEOUT = 60  # секунд на запуск и рукопожатие (npx -y скачивает пакет)
_TOOL_TIMEOUT_RANGE = (1, 3600)
_STARTUP_TIMEOUT_RANGE = (1, 600)
ANY_TOOL: Final = "*"  # в autoApprove: все инструменты сервера

# Имя сервера входит в имя инструмента для модели: mcp__<сервер>__<инструмент>.
_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,40}")
_HTTP_TYPES = frozenset((HTTP, "streamable-http", "streamableHttp"))
# Нечитаемый файл. RecursionError — глубоко вложенный JSON из репозитория.
_READ_ERRORS = (OSError, UnicodeDecodeError, ValueError, RecursionError)


@dataclass(frozen=True)
class McpServerConfig:
    """Сервер из настроек — как записан, без раскрытия ``${VAR}``.

    ``timeout`` — секунд на вызов инструмента, ``startup_timeout`` — на запуск и
    получение списка инструментов; ``auto_approve`` — инструменты, которые выполняются
    без вопроса (``*`` — все); ``origin`` — файл, где сервер описан (для сообщений).
    """

    name: str
    transport: McpTransport
    command: str = ""
    args: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    url: str = ""
    headers: tuple[tuple[str, str], ...] = ()
    timeout: int = DEFAULT_TOOL_TIMEOUT
    startup_timeout: int = DEFAULT_STARTUP_TIMEOUT
    auto_approve: frozenset[str] = field(default_factory=frozenset)
    source: SettingsSource = "user"
    origin: str = ""

    def approves(self, tool: str) -> bool:
        """Выполняется ли инструмент ``tool`` без вопроса (``autoApprove``)."""
        return ANY_TOOL in self.auto_approve or tool in self.auto_approve

    def canonical(self) -> list[object]:
        """Всё, что меняет поведение сервера, — для отпечатка доверия."""
        return [
            self.source,
            self.origin,
            self.name,
            self.transport,
            self.command,
            list(self.args),
            [list(pair) for pair in self.env],
            self.url,
            [list(pair) for pair in self.headers],
            self.timeout,
            self.startup_timeout,
            sorted(self.auto_approve),
        ]


@dataclass(frozen=True)
class McpSettings:
    """Все записи серверов из файлов настроек, по порядку файлов."""

    servers: tuple[McpServerConfig, ...] = ()

    def effective(self) -> tuple[McpServerConfig, ...]:
        """Действующие серверы: из одноимённых — последний (личные настройки важнее)."""
        by_name: dict[str, McpServerConfig] = {}
        for server in self.servers:
            by_name[server.name] = server
        return tuple(by_name.values())

    def from_sources(self, *sources: SettingsSource) -> tuple[McpServerConfig, ...]:
        """Записи из файлов ``sources``."""
        return tuple(server for server in self.servers if server.source in sources)

    @property
    def project_servers(self) -> tuple[McpServerConfig, ...]:
        """Серверы из файлов проекта."""
        return self.from_sources(*PROJECT_SOURCES)

    @property
    def user_servers(self) -> tuple[McpServerConfig, ...]:
        """Серверы из ``~/.gigacode/settings.json``."""
        return self.from_sources("user")

    def without(self, *sources: SettingsSource) -> McpSettings:
        """Без серверов из ``sources`` (не подтверждены)."""
        kept = tuple(server for server in self.servers if server.source not in sources)
        return replace(self, servers=kept)


def mcp_files(workspace: Workspace) -> list[tuple[Path, SettingsSource, str]]:
    """Файлы с серверами по порядку: (путь, источник, как показывать)."""
    return [
        (user_settings_file(), "user", "~/.gigacode/settings.json"),
        (workspace.root / MCP_JSON_FILE, "project", MCP_JSON_FILE),
        (workspace.settings_file, "project", ".gigacode/settings.json"),
        (workspace.local_settings_file, "local", ".gigacode/settings.local.json"),
    ]


def load_mcp_settings(workspace: Workspace) -> tuple[McpSettings, list[str]]:
    """Серверы из всех файлов и предупреждения об ошибках в них.

    Нет файла — не ошибка. Нечитаемый ``.mcp.json`` — предупреждение; о нечитаемых
    файлах настроек предупреждает загрузка хуков (те же файлы), здесь они пропускаются
    молча. Файлы недоверенного проекта не читаются.
    """
    servers: list[McpServerConfig] = []
    warnings: list[str] = []
    for path, source, origin in mcp_files(workspace):
        if workspace.untrusted and source in PROJECT_SOURCES:
            continue
        reported = warnings if path.name == MCP_JSON_FILE else []
        document = _read_document(path, origin, reported)
        if document is not None:
            section = document.get("mcpServers")
            servers.extend(_parse_servers(section, source, origin, warnings))
    return McpSettings(tuple(servers)), warnings


def _read_document(path: Path, origin: str, warnings: list[str]) -> dict[str, Any] | None:
    """JSON-объект файла; None — файла нет или он пропущен (с предупреждением)."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except _READ_ERRORS as exc:
        warnings.append(f"не удалось прочитать {origin}: {exc} — файл пропущен")
        return None
    if not isinstance(document, dict):
        warnings.append(f"{origin}: ожидается JSON-объект — файл пропущен")
        return None
    return document


def _parse_servers(
    section: object,
    source: SettingsSource,
    origin: str,
    warnings: list[str],
) -> list[McpServerConfig]:
    """Секция ``mcpServers`` одного файла → серверы; ошибки — в ``warnings``."""
    if section is None:
        return []
    if not isinstance(section, dict):
        warnings.append(f"{origin}: mcpServers — не объект, серверы пропущены")
        return []
    servers: list[McpServerConfig] = []
    for name, entry in section.items():
        where = f"{origin}: mcpServers.{name}"
        if not _NAME_RE.fullmatch(name):
            warnings.append(
                f"{where}: имя — от 1 до 40 латинских букв, цифр, _ и - (входит в имена "
                "инструментов), сервер пропущен",
            )
            continue
        if not isinstance(entry, dict):
            warnings.append(f"{where}: ожидается объект — сервер пропущен")
            continue
        if entry.get("disabled") is True:
            continue
        server = _parse_server(name, entry, where, warnings)
        if server is not None:
            servers.append(replace(server, source=source, origin=origin))
    return servers


def _parse_server(
    name: str,
    entry: dict[str, Any],
    where: str,
    warnings: list[str],
) -> McpServerConfig | None:
    """Запись сервера; None — ошибка (с предупреждением)."""
    transport = _transport(entry, where, warnings)
    if transport is None:
        return None
    if transport == HTTP:
        server = _parse_http(name, entry, where, warnings)
    else:
        server = _parse_stdio(name, entry, where, warnings)
    approved = _strings(entry.get("autoApprove"), f"{where}.autoApprove", warnings)
    if server is None or approved is None:
        return None
    return replace(
        server,
        timeout=_seconds(
            entry.get("timeout", DEFAULT_TOOL_TIMEOUT),
            DEFAULT_TOOL_TIMEOUT,
            _TOOL_TIMEOUT_RANGE,
            f"{where}.timeout",
            warnings,
        ),
        startup_timeout=_seconds(
            entry.get("startupTimeout", DEFAULT_STARTUP_TIMEOUT),
            DEFAULT_STARTUP_TIMEOUT,
            _STARTUP_TIMEOUT_RANGE,
            f"{where}.startupTimeout",
            warnings,
        ),
        auto_approve=frozenset(approved),
    )


def _transport(entry: dict[str, Any], where: str, warnings: list[str]) -> McpTransport | None:
    """Вид подключения: ``type`` или, без него, по наличию ``command``/``url``."""
    kind = entry.get("type")
    if kind is None:
        kind = HTTP if "url" in entry and "command" not in entry else STDIO
    if kind == STDIO:
        return STDIO
    if kind in _HTTP_TYPES:
        return HTTP
    if kind == "sse":
        warnings.append(
            f'{where}: устаревший транспорт "sse" не поддерживается — сервер пропущен '
            '(если сервер умеет Streamable HTTP, укажите "type": "http")',
        )
        return None
    warnings.append(f'{where}: тип {kind!r} не поддерживается ("stdio" или "http") — пропущено')
    return None


def _parse_stdio(
    name: str,
    entry: dict[str, Any],
    where: str,
    warnings: list[str],
) -> McpServerConfig | None:
    command = entry.get("command")
    if not isinstance(command, str) or not command.strip() or not _clean(command):
        warnings.append(f"{where}: command — ожидается непустая строка, сервер пропущен")
        return None
    args = _strings(entry.get("args"), f"{where}.args", warnings)
    env = _pairs(entry.get("env"), f"{where}.env", warnings)
    if args is None or env is None:
        return None
    return McpServerConfig(name, STDIO, command=command, args=tuple(args), env=env)


def _parse_http(
    name: str,
    entry: dict[str, Any],
    where: str,
    warnings: list[str],
) -> McpServerConfig | None:
    url = entry.get("url")
    if not isinstance(url, str) or not url.startswith(("http://", "https://")) or not _clean(url):
        warnings.append(f"{where}: url — ожидается адрес http:// или https://, сервер пропущен")
        return None
    headers = _pairs(entry.get("headers"), f"{where}.headers", warnings)
    if headers is None:
        return None
    return McpServerConfig(name, HTTP, url=url, headers=headers)


def _strings(raw: object, where: str, warnings: list[str]) -> list[str] | None:
    """Список строк; None — ошибка (с предупреждением). Нет значения — пустой список."""
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(
        isinstance(element, str) and _clean(element) for element in raw
    ):
        warnings.append(f"{where}: ожидается список строк — сервер пропущен")
        return None
    return list(raw)


def _pairs(raw: object, where: str, warnings: list[str]) -> tuple[tuple[str, str], ...] | None:
    """Объект «строка → строка»; None — ошибка (с предупреждением)."""
    if raw is None:
        return ()
    if not isinstance(raw, dict) or not all(
        isinstance(setting, str) and _clean(key) and _clean(setting) for key, setting in raw.items()
    ):
        warnings.append(f"{where}: ожидается объект со строковыми значениями — сервер пропущен")
        return None
    return tuple((str(key), str(setting)) for key, setting in raw.items())


def _clean(text: str) -> bool:
    """Строку можно передать процессу: без NUL и одиночных суррогатов (json.loads их пропускает)."""
    if "\0" in text:
        return False
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _seconds(
    raw: object,
    default: int,
    bounds: tuple[int, int],
    where: str,
    warnings: list[str],
) -> int:
    """Число секунд, прижатое к ``bounds``; не число — ``default`` (с предупреждением).

    json.loads пропускает NaN, Infinity и 1e400 — int() на них упал бы.
    """
    if (
        isinstance(raw, bool)
        or not isinstance(raw, int | float)
        or (isinstance(raw, float) and not math.isfinite(raw))
    ):
        warnings.append(f"{where}: ожидается число секунд, взято {default}")
        return default
    low, high = bounds
    clamped = min(max(int(raw), low), high)
    if not low <= raw <= high:
        warnings.append(f"{where}: {raw} вне {low}…{high} с, взято {clamped}")
    return clamped


def servers_fingerprint(servers: tuple[McpServerConfig, ...]) -> str:
    """Отпечаток записей серверов (sha256); пустая строка — записей нет.

    Любое изменение команды, аргументов, окружения, адреса, заголовков, таймаутов,
    ``autoApprove`` или файла, где сервер описан, меняет отпечаток — и серверы снова
    требуют подтверждения.
    """
    if not servers:
        return ""
    canonical = [server.canonical() for server in servers]
    serialized = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
