"""Минимальный синхронный клиент Model Context Protocol поверх :mod:`~gigacrew.tools.mcp_transport`.

Ровно столько, сколько нужно агенту: рукопожатие (``initialize``), список инструментов
(``tools/list``, по страницам) и их вызов (``tools/call``). Запросы сервера: ``ping`` и
``roots/list`` (корни — проект и подключённые каталоги); остальные (``sampling``,
``elicitation``) — ответ «метод не поддерживается». Уведомления сервера игнорируются.

Ответы раскладываются по ожидающим запросам, как в :mod:`gigacrew.project.lsp`:
запрос ждёт свой ответ с таймаутом; не дождался или прерван (Ctrl+C, Esc) — серверу
уходит ``notifications/cancelled``. Сервер завершился — ожидающие запросы получают
отказ сразу, а не по таймауту.
"""

from __future__ import annotations

import contextlib
import queue
import threading
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from gigacrew.tools.mcp_transport import McpError, McpTimeoutError, Transport

__all__ = [
    "PROTOCOL_VERSION",
    "SUPPORTED_VERSIONS",
    "McpClient",
    "ServerInfo",
]

# Версия, которую предлагает gigacrew; сервер может ответить любой из поддерживаемых —
# нужная агенту часть протокола (инструменты) в них одинакова.
PROTOCOL_VERSION: Final = "2025-06-18"
SUPPORTED_VERSIONS: Final = frozenset(("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"))

_METHOD_NOT_FOUND = -32601
_MAX_PAGES = 50  # страниц tools/list: дальше — сервер, вероятно, зациклился
_NOTICE_TIMEOUT = 5.0  # секунд на отправку уведомления или ответа серверу

# Место для ответа на запрос: ответ сервера или None — сигнал «сервер завершился».
_Slot = queue.Queue[dict[str, Any] | None]


@dataclass(frozen=True)
class ServerInfo:
    """Что сервер сообщил о себе при рукопожатии."""

    name: str
    version: str
    protocol: str
    instructions: str = ""  # указания модели, как пользоваться инструментами сервера
    has_tools: bool = True


class McpClient:
    """Сессия с одним сервером. Конструктор ничего не запускает — см. :meth:`connect`."""

    def __init__(
        self,
        transport: Transport,
        *,
        roots: Sequence[Path] = (),
        version: str = "",
    ) -> None:
        """``roots`` — каталоги, которые сервер может узнать запросом ``roots/list``."""
        self._transport = transport
        self._roots = tuple(roots)
        self._version = version or "0"
        self._pending: dict[int, _Slot] = {}
        self._pending_lock = threading.Lock()
        self._next_id = 0

    @property
    def alive(self) -> bool:
        """Канал к серверу открыт и сервер не завершился."""
        return self._transport.alive

    def connect(self, timeout: float) -> ServerInfo:
        """Открывает канал и проводит рукопожатие. McpError/OSError — не вышло."""
        self._transport.open(self._dispatch, self._server_gone)
        capabilities: dict[str, Any] = {}
        if self._roots:
            capabilities["roots"] = {"listChanged": False}
        reply = self.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": capabilities,
                "clientInfo": {"name": "gigacrew", "version": self._version},
            },
            timeout=timeout,
        )
        about = _server_info(reply)
        self._transport.protocol_version = about.protocol
        self.notify("notifications/initialized")
        return about

    def list_tools(self, timeout: float) -> list[dict[str, Any]]:
        """Описания инструментов сервера (все страницы); без имени — пропускаются."""
        tools: list[dict[str, Any]] = []
        cursor: object = None
        for _ in range(_MAX_PAGES):
            params = {"cursor": cursor} if isinstance(cursor, str) and cursor else None
            page = self.request("tools/list", params, timeout=timeout)
            if not isinstance(page, dict) or not isinstance(page.get("tools"), list):
                raise McpError("tools/list: ответ без списка инструментов")
            tools.extend(tool for tool in page["tools"] if _named(tool))
            cursor = page.get("nextCursor")
            if not isinstance(cursor, str) or not cursor:
                break
        return tools

    def call_tool(self, name: str, arguments: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        """Вызов инструмента; результат — объект ``CallToolResult`` как есть."""
        reply = self.request("tools/call", {"name": name, "arguments": arguments}, timeout=timeout)
        if not isinstance(reply, dict):
            raise McpError("tools/call: ответ не объект")
        return reply

    def request(self, method: str, params: Any = None, *, timeout: float) -> Any:
        """Запрос и ожидание ответа. McpError — ошибка, McpTimeoutError — нет ответа."""
        with self._pending_slot() as (msg_id, slot):
            message = _jsonrpc(id=msg_id, method=method)
            if params is not None:
                message["params"] = params
            try:
                reply = self._exchange(message, slot, timeout)
            except McpError as exc:
                if isinstance(exc, McpTimeoutError):
                    self._cancel(msg_id, "timeout")
                raise
            except BaseException:  # Ctrl+C или Esc посреди ожидания: ответ больше не нужен
                self._cancel(msg_id, "interrupted")
                raise
        return _unwrap(method, reply)

    def notify(self, method: str, params: Any = None) -> None:
        """Уведомление серверу, без ответа. McpError — не отправлено."""
        message = _jsonrpc(method=method)
        if params is not None:
            message["params"] = params
        self._transport.send(message, timeout=_NOTICE_TIMEOUT)

    def close(self) -> None:
        """Закрывает канал (stdio — завершает процесс сервера). Повторный вызов безопасен."""
        self._transport.close()
        self._server_gone()

    # ------------------------------------------------------------------ #
    def _exchange(
        self,
        message: dict[str, Any],
        slot: _Slot,
        timeout: float,
    ) -> dict[str, Any] | None:
        """Отправка и ожидание ответа (HTTP кладёт ответ в ``slot`` ещё внутри send)."""
        self._transport.send(message, timeout=timeout)
        try:
            return slot.get(timeout=timeout)
        except queue.Empty:
            waited = format(timeout, "g")
            method = message["method"]
            raise McpTimeoutError(f"{method}: сервер не ответил за {waited} с") from None

    @contextlib.contextmanager
    def _pending_slot(self) -> Generator[tuple[int, _Slot]]:
        """Номер нового запроса и место для ответа на него; по выходе место снимается."""
        with self._pending_lock:
            self._next_id += 1
            msg_id = self._next_id
            slot: _Slot = queue.Queue(maxsize=1)
            self._pending[msg_id] = slot
        try:
            yield msg_id, slot
        finally:
            with self._pending_lock:
                self._pending.pop(msg_id, None)

    def _cancel(self, msg_id: int, reason: str) -> None:
        with contextlib.suppress(McpError):
            self.notify("notifications/cancelled", {"requestId": msg_id, "reason": reason})

    def _server_gone(self) -> None:
        """Сервер недоступен: ожидающие запросы получают отказ сразу."""
        with self._pending_lock:
            waiting = list(self._pending.values())
        for slot in waiting:
            with contextlib.suppress(queue.Full):
                slot.put_nowait(None)

    def _dispatch(self, message: dict[str, Any]) -> None:
        """Входящее сообщение: ответ — ожидающему запросу, запрос сервера — ответ на него."""
        method = message.get("method")
        if method is None:
            msg_id = message.get("id")
            with self._pending_lock:
                slot = self._pending.get(msg_id) if isinstance(msg_id, int) else None
            if slot is not None:
                with contextlib.suppress(queue.Full):
                    slot.put_nowait(message)
            return
        if "id" not in message:
            return  # уведомление сервера (журнал, прогресс, смена списка инструментов)
        if method == "ping":
            reply = _jsonrpc(id=message["id"], result={})
        elif method == "roots/list" and self._roots:
            roots = [{"uri": root.as_uri(), "name": root.name} for root in self._roots]
            reply = _jsonrpc(id=message["id"], result={"roots": roots})
        else:
            error = {"code": _METHOD_NOT_FOUND, "message": f"Method not found: {method}"}
            reply = _jsonrpc(id=message["id"], error=error)
        with contextlib.suppress(McpError):
            self._transport.send(reply, timeout=_NOTICE_TIMEOUT)


def _named(tool: object) -> bool:
    """Описание инструмента с непустым именем."""
    if not isinstance(tool, dict):
        return False
    name = tool.get("name")
    return isinstance(name, str) and bool(name)


def _unwrap(method: str, reply: dict[str, Any] | None) -> Any:
    """Результат ответа на запрос ``method``; ответ-ошибка или завершение сервера — McpError."""
    if reply is None:
        raise McpError(f"{method}: сервер завершился")
    error = reply.get("error")
    if error is not None:
        raise McpError(f"{method}: {_error_text(error)}")
    return reply.get("result")


def _server_info(reply: object) -> ServerInfo:
    """Ответ на ``initialize`` → :class:`ServerInfo`; неподдерживаемая версия — McpError."""
    if not isinstance(reply, dict):
        raise McpError("initialize: ответ не объект")
    protocol = reply.get("protocolVersion")
    if protocol not in SUPPORTED_VERSIONS:
        known = ", ".join(sorted(SUPPORTED_VERSIONS))
        raise McpError(
            f"сервер предлагает версию протокола {protocol!r}, gigacrew поддерживает {known}",
        )
    about = reply.get("serverInfo")
    about = about if isinstance(about, dict) else {}
    capabilities = reply.get("capabilities")
    instructions = reply.get("instructions")
    return ServerInfo(
        name=str(about.get("name") or ""),
        version=str(about.get("version") or ""),
        protocol=str(protocol),
        instructions=instructions if isinstance(instructions, str) else "",
        has_tools=isinstance(capabilities, dict) and "tools" in capabilities,
    )


def _error_text(error: object) -> str:
    """Текст ответа-ошибки JSON-RPC."""
    if not isinstance(error, dict):
        return "ошибка сервера"
    text = error.get("message")
    code = error.get("code")
    described = text if isinstance(text, str) and text else "ошибка сервера"
    return f"{described} (код {code})" if isinstance(code, int) else described


def _jsonrpc(**fields: Any) -> dict[str, Any]:
    """Сообщение JSON-RPC 2.0: версия протокола, затем поля в порядке передачи."""
    return {"jsonrpc": "2.0", **fields}
