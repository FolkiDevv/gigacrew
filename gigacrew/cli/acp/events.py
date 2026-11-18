"""События агента → уведомления ACP ``session/update``.

Уведомления идут, только пока ход «открыт» (:meth:`AcpEvents.open_turn`): события
отменённого хода, который ещё доделывает свой поток, клиенту не отправляются.
Закрытие хода (:meth:`AcpEvents.close_turn`) сообщает о незавершённых вызовах
инструментов как о неудавшихся — у клиента не остаётся вызовов «выполняется».
"""

from __future__ import annotations

import logging
import threading
from types import MappingProxyType
from typing import Any, Final, override

from gigacrew.agent.events import AgentEvents, NoticeLevel, ToolCallInfo
from gigacrew.cli.acp.rpc import ConnectionClosedError, RpcConnection
from gigacrew.permissions import ToolKind
from gigacrew.tools.base import Display, ToolResult

__all__ = ["AcpEvents"]

UPDATE: Final = "session/update"
# Инструменты, которые читают файл целиком (вид read); прочие только смотрящие — поиск.
_READ_TOOLS: Final = frozenset(("read_file", "file_outline", "background_output"))
# Запускают программы, хотя их вид для режимов разрешений — «прочее».
_EXECUTE_TOOLS: Final = frozenset(("git",))
_TOOL_KINDS: Final = MappingProxyType({ToolKind.EDIT: "edit", ToolKind.COMMAND: "execute"})
_NOTICE_LEVELS: Final = MappingProxyType({
    "info": logging.INFO,
    "warn": logging.WARNING,
    "error": logging.ERROR,
})

type Update = dict[str, Any]

_log = logging.getLogger(__name__)


class AcpEvents(AgentEvents):
    """Получатель событий агента одной сессии ACP; id сессии задаёт :meth:`bind`."""

    def __init__(self, connection: RpcConnection) -> None:
        """События пойдут клиенту через ``connection``; до :meth:`open_turn` — не идут."""
        self._connection = connection
        self._session_id = ""
        self._lock = threading.Lock()
        self._live = False
        self._open_calls: dict[str, None] = {}  # незавершённые вызовы хода, по порядку

    def bind(self, session_id: str) -> None:
        """Id сессии, которой адресованы уведомления."""
        self._session_id = session_id

    def open_turn(self) -> None:
        """Начался ход: его события идут клиенту."""
        with self._lock:
            self._open_calls.clear()
            self._live = True

    def close_turn(self) -> None:
        """Ход закончен или отменён: незавершённые вызовы — failed, дальше — тишина."""
        with self._lock:
            for call_id in self._open_calls:
                self._send({
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": call_id,
                    "status": "failed",
                })
            self._open_calls.clear()
            self._live = False

    def report_usage(self, used: int, size: int) -> None:
        """Заполненность контекста после хода: ``used`` из ``size`` токенов."""
        self._emit({"sessionUpdate": "usage_update", "used": used, "size": size})

    @override
    def on_stream_delta(self, text: str) -> None:
        self._emit(_chunk("agent_message_chunk", text))

    @override
    def on_assistant_text(self, text: str) -> None:
        self._emit(_chunk("agent_message_chunk", text))

    @override
    def on_reasoning(self, text: str) -> None:
        self._emit(_chunk("agent_thought_chunk", text))

    @override
    def on_tool_call(self, call: ToolCallInfo) -> None:
        update: Update = {
            "sessionUpdate": "tool_call",
            "toolCallId": call.call_id,
            "title": f"{call.name} {call.summary}".strip(),
            "kind": _kind(call),
            "status": "pending",
            "rawInput": dict(call.arguments),
            "locations": [{"path": path} for path in call.paths],
        }
        with self._lock:
            if self._live:
                self._open_calls[call.call_id] = None
                self._send(update)

    @override
    def on_tool_start(self, call: ToolCallInfo) -> None:
        self._emit(_call_update(call, "in_progress"))

    @override
    def on_tool_result(
        self,
        call: ToolCallInfo,
        tool_result: ToolResult,
        *,
        previewed: bool,
    ) -> None:
        update = _call_update(call, "completed" if tool_result.ok else "failed")
        update["content"] = _result_content(call, tool_result)
        with self._lock:
            if self._live:
                self._open_calls.pop(call.call_id, None)
                self._send(update)

    @override
    def on_notice(self, text: str, *, level: NoticeLevel = "info") -> None:
        # stdout занят протоколом: предупреждения агента — в журнал.
        _log.log(_NOTICE_LEVELS[level], "%s", text)

    def _emit(self, update: Update) -> None:
        with self._lock:
            if self._live:
                self._send(update)

    def _send(self, update: Update) -> None:
        """Отправляет уведомление (вызывается под блокировкой)."""
        try:
            self._connection.notify(UPDATE, {"sessionId": self._session_id, "update": update})
        except ConnectionClosedError:
            # Клиент ушёл: ход доработает без него.
            self._live = False


def _chunk(kind: str, text: str) -> Update:
    return {"sessionUpdate": kind, "content": {"type": "text", "text": text}}


def _call_update(call: ToolCallInfo, status: str) -> Update:
    return {"sessionUpdate": "tool_call_update", "toolCallId": call.call_id, "status": status}


def _kind(call: ToolCallInfo) -> str:
    """Вид вызова ACP: правка, команда, чтение, поиск или прочее."""
    known = _TOOL_KINDS.get(call.kind)
    if known is not None:
        return known
    if call.name in _EXECUTE_TOOLS:
        return "execute"
    if call.read_only:
        return "read" if call.name in _READ_TOOLS else "search"
    return "other"


def _result_content(call: ToolCallInfo, tool_result: ToolResult) -> list[Update]:
    """Содержимое результата: дифф правки (если есть) и текст, который получила модель."""
    content: list[Update] = []
    diff = _diff(call, tool_result.display)
    if diff is not None:
        content.append(diff)
    if tool_result.content:
        content.append({
            "type": "content",
            "content": {"type": "text", "text": tool_result.content},
        })
    return content


def _diff(call: ToolCallInfo, display: Display | None) -> Update | None:
    if display is None or display.kind != "diff" or not call.paths:
        return None
    sides = _diff_sides(display.text)
    if sides is None:
        return None
    old, new = sides
    return {"type": "diff", "path": call.paths[0], "oldText": old or None, "newText": new}


def _diff_sides(diff: str) -> tuple[str, str] | None:
    """Прежний и новый текст изменённых участков unified diff (None — участков нет)."""
    old: list[str] = []
    new: list[str] = []
    hunks = 0
    for line in diff.splitlines(keepends=True):
        if line.startswith("@@"):
            hunks += 1
        elif hunks and not line.startswith("\\"):
            marker, body = line[:1], line[1:]
            if marker in {"-", " "}:
                old.append(body)
            if marker in {"+", " "}:
                new.append(body)
    if not hunks:
        return None
    return "".join(old), "".join(new)
