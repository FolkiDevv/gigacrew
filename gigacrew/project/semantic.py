"""Точная навигация по Python-коду: ty (Astral) как LSP-сервер.

ty понимает импорты, псевдонимы и типы, поэтому находит ссылки, определения и
вызовы точнее индекса по именам. Сервер (``ty server``) запускается лениво — при
первом запросе — и живёт до конца сессии, один на корень проекта.

За файлами ty сам не следит (``workspace/didChangeWatchedFiles`` шлёт клиент):
об изменениях ему сообщает обновление индекса (:func:`add_change_listener`), а
инструменты обновляют индекс перед каждым запросом — поэтому правки, в том числе
самого агента, ty видит сразу.

Сбой (сервер упал, не ответил) — один перезапуск; повторный сбой отключает ty до
конца сессии. Вызывающий код при недоступности ty откатывается на индекс.
"""

from __future__ import annotations

import atexit
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from gigacrew.project.index import add_change_listener
from gigacrew.project.lsp import LspClient, LspError
from gigacrew.project.symbols import split_lines
from gigacrew.project.workspace import Workspace

START_TIMEOUT = 15.0  # запуск и initialize
REQUEST_TIMEOUT = 10.0
MAX_FAILURES = 2  # сбоев за сессию до отключения
LOG_NAME = "ty.log"
_MAX_LOG_BYTES = 1_000_000
PYTHON_SUFFIXES = (".py", ".pyi")
# Файлы, изменения которых ty должен узнать: исходники и его конфигурация.
_WATCHED_NAMES = frozenset({"pyproject.toml", "ty.toml"})

# Типы изменений workspace/didChangeWatchedFiles. Новый файл — только «создан»:
# «изменён» для неизвестного серверу файла ty учитывает не всегда.
_CREATED, _CHANGED, _DELETED = 1, 2, 3


class SemanticUnavailableError(Exception):
    """ty недоступен (не установлен, не запустился, отключён после сбоев)."""


def is_python(path: str) -> bool:
    return path.endswith(PYTHON_SUFFIXES)


@dataclass(frozen=True)
class Location:
    """Место в файле. ``path`` — от корня проекта (или абсолютный — вне проекта)."""

    path: str
    line: int  # с 1
    col: int  # в символах, с 0
    in_project: bool = True


@dataclass(frozen=True)
class CallItem:
    """Элемент иерархии вызовов: функция или метод и места вызовов (``calls``)."""

    name: str
    detail: str  # модуль
    location: Location  # имя определения
    calls: tuple[Location, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)


# --------------------------------------------------------------------------- #
# Позиции: у нас столбцы в символах, у LSP — в единицах кодировки сервера.
# --------------------------------------------------------------------------- #
def _to_units(text: str, col: int, encoding: str) -> int:
    prefix = text[:col]
    if encoding == "utf-8":
        return len(prefix.encode("utf-8"))
    if encoding == "utf-32":
        return len(prefix)
    return len(prefix.encode("utf-16-le")) // 2


def _from_units(text: str, units: int, encoding: str) -> int:
    if encoding == "utf-8":
        return len(text.encode("utf-8")[:units].decode("utf-8", errors="ignore"))
    if encoding == "utf-32":
        return units
    return len(text.encode("utf-16-le")[: units * 2].decode("utf-16-le", errors="ignore"))


class _Lines:
    """Строки файлов для пересчёта столбцов — читаются по файлу один раз."""

    def __init__(self) -> None:
        self._cache: dict[Path, list[str]] = {}

    def get(self, path: Path, line: int) -> str:
        if path not in self._cache:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                self._cache[path] = split_lines(text)  # нумерация строк как у LSP
            except OSError:
                self._cache[path] = []
        lines = self._cache[path]
        return lines[line - 1] if 0 < line <= len(lines) else ""


# --------------------------------------------------------------------------- #
# Сервер
# --------------------------------------------------------------------------- #
def find_binary() -> str:
    """Путь к бинарнику ty из установленного пакета."""
    try:
        from ty import find_ty_bin
    except ImportError as e:
        raise SemanticUnavailableError("пакет ty не установлен") from e
    try:
        return find_ty_bin()
    except FileNotFoundError as e:
        raise SemanticUnavailableError("бинарник ty не найден") from e


class TyServer:
    """Один процесс ``ty server`` для корня проекта."""

    def __init__(
        self,
        root: Path,
        *,
        binary: str,
        env: Mapping[str, str] | None = None,
        log_path: Path | None = None,
    ):
        self.root = root
        self._client = LspClient(
            [binary, "server"],
            cwd=root,
            env=env,
            stderr_path=log_path,
            handlers={"workspace/configuration": self._configuration},
        )
        self._encoding = "utf-16"

    @property
    def alive(self) -> bool:
        return self._client.alive

    def start(self) -> None:
        """Запуск и рукопожатие. OSError/LspError — не удалось."""
        self._client.start()
        root_uri = self.root.as_uri()
        result = self._client.request(
            "initialize",
            {
                "processId": None,
                "rootUri": root_uri,
                "workspaceFolders": [{"uri": root_uri, "name": self.root.name}],
                "capabilities": {
                    "general": {"positionEncodings": ["utf-8", "utf-16"]},
                    "workspace": {
                        "configuration": True,
                        "workspaceFolders": True,
                        "didChangeWatchedFiles": {"dynamicRegistration": True},
                    },
                    "textDocument": {
                        "definition": {"linkSupport": False},
                        "references": {},
                        "hover": {"contentFormat": ["plaintext"]},
                        "callHierarchy": {},
                    },
                },
            },
            timeout=START_TIMEOUT,
        )
        caps = (result or {}).get("capabilities") or {}
        self._encoding = caps.get("positionEncoding") or "utf-16"
        self._client.notify("initialized", {})

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def _configuration(params: Any) -> list[dict[str, Any]]:
        items = (params or {}).get("items") or [{}]
        # диагностика агенту не нужна — серверу меньше работы
        return [{"diagnosticMode": "off"} for _ in items]

    # ------------------------------ файлы ------------------------------ #
    def notify_changes(
        self,
        added: tuple[str, ...],
        updated: tuple[str, ...],
        removed: tuple[str, ...],
    ) -> None:
        changes = [
            {"uri": (self.root / path).as_uri(), "type": kind}
            for paths, kind in ((added, _CREATED), (updated, _CHANGED), (removed, _DELETED))
            for path in paths
            if is_python(path) or path.rsplit("/", 1)[-1] in _WATCHED_NAMES
        ]
        if changes:
            self._client.notify("workspace/didChangeWatchedFiles", {"changes": changes})

    # ------------------------------ запросы ------------------------------ #
    def _position(self, path: str, line: int, col: int) -> dict[str, Any]:
        abs_path = self.root / path
        text = _Lines().get(abs_path, line)
        return {
            "textDocument": {"uri": abs_path.as_uri()},
            "position": {"line": line - 1, "character": _to_units(text, col, self._encoding)},
        }

    def _location(self, uri: str, position: dict[str, Any], lines: _Lines) -> Location:
        line = int(position.get("line", 0)) + 1
        units = int(position.get("character", 0))
        parsed = urlparse(uri)
        if parsed.scheme != "file":  # встроенные заглушки ty и т.п.
            return Location(unquote(uri), line, units, in_project=False)
        # url2pathname сам раскодирует %XX; netloc — сервер UNC-пути (Windows)
        host = parsed.netloc if parsed.netloc not in {"", "localhost"} else ""
        if host and os.name != "nt":
            # С Python 3.14 url2pathname на POSIX отвергает нелокальный хост — собираем сами
            abs_path = Path(f"//{host}{unquote(parsed.path)}")
        else:
            abs_path = Path(url2pathname(f"//{host}{parsed.path}" if host else parsed.path))
        col = _from_units(lines.get(abs_path, line), units, self._encoding)
        try:
            rel = abs_path.relative_to(self.root).as_posix()
        except ValueError:
            return Location(str(abs_path), line, col, in_project=False)
        return Location(rel, line, col)

    def _request(self, method: str, params: Any) -> Any:
        return self._client.request(method, params, timeout=REQUEST_TIMEOUT)

    def _locations(self, result: Any) -> list[Location]:
        if isinstance(result, dict):
            result = [result]
        lines = _Lines()
        out = []
        for item in result or []:
            uri = item.get("uri") or item.get("targetUri")
            rng = item.get("range") or item.get("targetSelectionRange") or {}
            if uri:
                out.append(self._location(uri, rng.get("start", {}), lines))
        return out

    def references(self, path: str, line: int, col: int) -> list[Location]:
        params = self._position(path, line, col) | {"context": {"includeDeclaration": False}}
        return self._locations(self._request("textDocument/references", params))

    def definition(self, path: str, line: int, col: int) -> list[Location]:
        return self._locations(
            self._request("textDocument/definition", self._position(path, line, col)),
        )

    def hover(self, path: str, line: int, col: int) -> str:
        result = self._request("textDocument/hover", self._position(path, line, col))
        contents = (result or {}).get("contents")
        if isinstance(contents, dict):
            return str(contents.get("value", "")).strip()
        if isinstance(contents, list):
            return "\n".join(
                str(c.get("value", "")) if isinstance(c, dict) else str(c) for c in contents
            ).strip()
        return str(contents or "").strip()

    def _call_item(self, raw: dict[str, Any], lines: _Lines, calls: list[Any] = ()) -> CallItem:
        uri = raw.get("uri", "")
        start = (raw.get("selectionRange") or raw.get("range") or {}).get("start", {})
        location = self._location(uri, start, lines)
        return CallItem(
            name=str(raw.get("name", "")),
            detail=str(raw.get("detail") or ""),
            location=location,
            calls=tuple(calls),
            raw=raw,
        )

    def prepare_call_hierarchy(self, path: str, line: int, col: int) -> list[CallItem]:
        result = self._request("textDocument/prepareCallHierarchy", self._position(path, line, col))
        lines = _Lines()
        return [self._call_item(raw, lines) for raw in result or []]

    def incoming_calls(self, item: CallItem) -> list[CallItem]:
        """Кто вызывает ``item``: места вызовов — в файле вызывающего."""
        lines = _Lines()
        out = []
        for call in self._request("callHierarchy/incomingCalls", {"item": item.raw}) or []:
            caller = call.get("from") or {}
            uri = caller.get("uri", "")
            sites = [
                self._location(uri, r.get("start", {}), lines) for r in call.get("fromRanges", [])
            ]
            out.append(self._call_item(caller, lines, sites))
        return out

    def outgoing_calls(self, item: CallItem) -> list[CallItem]:
        """Что вызывает ``item``: места вызовов — в файле самого ``item``."""
        lines = _Lines()
        uri = item.raw.get("uri", "")
        out = []
        for call in self._request("callHierarchy/outgoingCalls", {"item": item.raw}) or []:
            sites = [
                self._location(uri, r.get("start", {}), lines) for r in call.get("fromRanges", [])
            ]
            out.append(self._call_item(call.get("to") or {}, lines, sites))
        return out


# --------------------------------------------------------------------------- #
# Сессии: один сервер на корень проекта на всё время работы процесса.
# --------------------------------------------------------------------------- #
@dataclass
class _Session:
    server: TyServer | None = None
    failures: int = 0
    disabled: str | None = None


_sessions: dict[Path, _Session] = {}
_lock = threading.Lock()


def server_for(workspace: Workspace, env: Mapping[str, str] | None = None) -> TyServer:
    """Работающий сервер для проекта (запускается при первом обращении).

    SemanticUnavailableError — ty не установлен, не запустился или отключён после сбоев.
    """
    root = workspace.root
    with _lock:
        session = _sessions.setdefault(root, _Session())
        if session.disabled:
            raise SemanticUnavailableError(session.disabled)
        if session.server is not None:
            if session.server.alive:
                return session.server
            session.server.close()
            session.server = None
            _count_failure(session, "сервер завершился")
            if session.disabled:
                raise SemanticUnavailableError(session.disabled)
        binary = find_binary()
        log_path = _log_path(workspace)
        server = TyServer(root, binary=binary, env=env, log_path=log_path)
        try:
            server.start()
        except (OSError, LspError) as e:
            server.close()
            _count_failure(session, f"не запустился: {e}")
            raise SemanticUnavailableError(session.disabled or f"ty не запустился: {e}") from e
        except BaseException:  # Ctrl+C посреди запуска — процесс не должен остаться
            server.close()
            raise
        session.server = server
        return server


def report_failure(workspace: Workspace, error: Exception) -> str:
    """Сбой запроса: сервер останавливается (следующий запрос перезапустит его).

    Возвращает пояснение для пользователя.
    """
    with _lock:
        session = _sessions.setdefault(workspace.root, _Session())
        if session.server is not None:
            session.server.close()
            session.server = None
        _count_failure(session, str(error))
        return session.disabled or f"ty: {error} — сервер будет перезапущен"


def status(workspace: Workspace) -> str:
    """Состояние ty для ``/index``."""
    session = _sessions.get(workspace.root)
    if session is None or (session.server is None and not session.disabled):
        return "не запущен (запустится при первом запросе)"
    if session.disabled:
        return session.disabled
    return "работает" if session.server is not None and session.server.alive else "остановлен"


def shutdown_all() -> None:
    """Останавливает все серверы и забывает сбои (при выходе и в тестах)."""
    with _lock:
        sessions = list(_sessions.values())
        _sessions.clear()
    for session in sessions:
        if session.server is not None:
            session.server.close()


def _count_failure(session: _Session, reason: str) -> None:
    session.failures += 1
    if session.failures >= MAX_FAILURES:
        session.disabled = f"ty отключён до конца сессии после сбоев ({reason})"


def _log_path(workspace: Workspace) -> Path | None:
    try:
        log = workspace.ensure_data_dir() / LOG_NAME
        if log.exists() and log.stat().st_size > _MAX_LOG_BYTES:
            log.unlink()
        return log
    except OSError:
        return None


def _on_index_changes(
    root: Path,
    added: tuple[str, ...],
    updated: tuple[str, ...],
    removed: tuple[str, ...],
) -> None:
    session = _sessions.get(root)
    server = session.server if session is not None else None
    if server is None or not server.alive:
        return
    try:
        server.notify_changes(added, updated, removed)
    except LspError:
        pass  # упавший сервер обнаружится на следующем запросе


add_change_listener(_on_index_changes)
atexit.register(shutdown_all)
