"""События агента → уведомления ACP ``session/update``.

Уведомления о ходе идут, только пока он «открыт» (:meth:`AcpEvents.open_turn`): события
отменённого хода, который ещё доделывает свой поток, клиенту не отправляются. Уведомления о
состоянии сессии (режим, слэш-команды) от хода не зависят.
Закрытие хода (:meth:`AcpEvents.close_turn`) сообщает о незавершённых вызовах
инструментов как о неудавшихся — у клиента не остаётся вызовов «выполняется».

Подтверждение операции — запрос ``session/request_permission``, вопросы модели — запрос
``elicitation/create`` (если клиент объявил формы). Ход ждёт ответа клиента; отмена хода
прерывает ожидание и сам ход (``KeyboardInterrupt``), поздний ответ ничего не решает.
Ошибка клиента или закрытый канал — отказ без ответа пользователя.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from types import MappingProxyType
from typing import Any, Final, override

from gigacrew.agent.events import AgentEvents, Approval, ConfirmResult, NoticeLevel, ToolCallInfo
from gigacrew.cli.acp import forms, permission
from gigacrew.cli.acp.rpc import (
    ConnectionClosedError,
    RequestCancelledError,
    RpcConnection,
    RpcError,
)
from gigacrew.permissions import PermissionMode, ToolKind
from gigacrew.tools.base import Display, ToolResult
from gigacrew.tools.questions import Answer, Question, QuestionsUnavailableError

__all__ = [
    "AcpEvents",
    "Update",
    "call_update",
    "text_chunk",
    "text_content",
    "tool_call_fields",
]

UPDATE: Final = "session/update"
PERMISSION: Final = "session/request_permission"
ELICITATION: Final = "elicitation/create"
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


class _TurnChannel:
    """Канал сессии к клиенту: уведомления и запросы открытого хода.

    ``form_questions`` — клиент объявил формы ``elicitation/create``: вопросы модели
    задаются через них, иначе спросить некого.
    """

    def __init__(self, connection: RpcConnection, *, form_questions: bool = False) -> None:
        """События пойдут клиенту через ``connection``; до :meth:`open_turn` — не идут."""
        self._connection = connection
        self._form_questions = form_questions
        self._session_id = ""
        self._lock = threading.Lock()
        self._live = False
        self._cancel = threading.Event()  # флаг отмены текущего хода
        # Незавершённые вызовы хода по порядку и их вид ACP: запрос подтверждения называет
        # тот же вид, что уведомление tool_call (агент спрашивает с видом для формулировок).
        self._open_calls: dict[str, str] = {}
        self._running_call = ""  # id запущенного вызова (к нему относятся вопросы)

    def bind(self, session_id: str) -> None:
        """Id сессии, которой адресованы уведомления."""
        self._session_id = session_id

    def open_turn(self, cancel: threading.Event) -> None:
        """Начался ход с флагом отмены ``cancel``: его события идут клиенту."""
        with self._lock:
            self._open_calls.clear()
            self._cancel = cancel
            self._live = True

    def close_turn(self) -> None:
        """Ход закончен или отменён: незавершённые вызовы — failed, дальше — тишина."""
        with self._lock:
            for call_id in self._open_calls:
                self._send(call_update(call_id, "failed"))
            self._open_calls.clear()
            self._live = False

    def report_usage(self, used: int, size: int) -> None:
        """Заполненность контекста после хода: ``used`` из ``size`` токенов."""
        self._emit(_update("usage_update", {"used": used, "size": size}))

    def report_mode(self, mode: PermissionMode) -> None:
        """Режим сменил сам агент (одобренный план): ``current_mode_update``."""
        self._notify(_update("current_mode_update", {"currentModeId": mode.value}))

    def report_commands(self, commands: list[Update]) -> None:
        """Слэш-команды сессии: ``available_commands_update``."""
        self._notify(_update("available_commands_update", {"availableCommands": commands}))

    def report_history(self, updates: list[Update]) -> None:
        """Переигранная история чата (``session/load``): идёт и вне хода."""
        with self._lock:
            for update in updates:
                self._send(update)

    def _request(self, method: str, params: Update) -> Any:
        """Запрос клиенту от имени сессии; отмена хода во время ожидания прерывает ход."""
        try:
            return self._connection.request(
                method,
                {"sessionId": self._session_id, **params},
                cancel=self._cancel,
            )
        except RequestCancelledError:
            raise KeyboardInterrupt from None

    def _emit(self, update: Update) -> None:
        with self._lock:
            if self._live:
                self._send(update)

    def _notify(self, update: Update) -> None:
        """Уведомление о состоянии сессии, а не о ходе: идёт и вне хода."""
        with self._lock:
            self._send(update)

    def _send(self, update: Update) -> None:
        """Отправляет уведомление (вызывается под блокировкой)."""
        try:
            self._connection.notify(UPDATE, {"sessionId": self._session_id, "update": update})
        except ConnectionClosedError:
            # Клиент ушёл: ход доработает без него.
            self._live = False


class AcpEvents(_TurnChannel, AgentEvents):
    """Получатель событий агента одной сессии ACP; id сессии задаёт :meth:`bind`."""

    @override
    def on_stream_delta(self, text: str) -> None:
        self._emit(text_chunk("agent_message_chunk", text))

    @override
    def on_assistant_text(self, text: str) -> None:
        self._emit(text_chunk("agent_message_chunk", text))

    @override
    def on_reasoning(self, text: str) -> None:
        self._emit(text_chunk("agent_thought_chunk", text))

    @override
    def on_tool_call(self, call: ToolCallInfo) -> None:
        update = {"sessionUpdate": "tool_call", **tool_call_fields(call)}
        with self._lock:
            if self._live:
                self._open_calls[call.call_id] = update["kind"]
                self._send(update)

    @override
    def confirm(
        self,
        call: ToolCallInfo,
        preview: Display | None,
        *,
        dangerous: bool,
    ) -> bool | Approval | ConfirmResult:
        """Запрос ``session/request_permission``; ответ не из вариантов — отказ без ответа."""
        offered = permission.options(call, dangerous=dangerous)
        tool_call = tool_call_fields(call)
        with self._lock:
            tool_call["kind"] = self._open_calls.get(call.call_id, tool_call["kind"])
        tool_call["content"] = _preview_content(call, preview)
        try:
            response = self._request(PERMISSION, {"toolCall": tool_call, "options": offered})
        except (RpcError, ConnectionClosedError):
            return ConfirmResult(Approval.NO, answered=False)
        return permission.confirm_result(response, offered)

    @override
    def can_confirm(self) -> bool:
        return True

    @override
    def ask_user(self, questions: Sequence[Question]) -> list[Answer] | None:
        """Форма ``elicitation/create``; клиент без форм — спросить некого."""
        if not self._form_questions:
            return super().ask_user(questions)
        params = forms.form_params(questions)
        if self._running_call:
            params["toolCallId"] = self._running_call
        try:
            response = self._request(ELICITATION, params)
        except (RpcError, ConnectionClosedError) as error:
            raise QuestionsUnavailableError(f"клиент не показал вопросы ({error})") from None
        return forms.answers(response, questions)

    @override
    def on_tool_start(self, call: ToolCallInfo) -> None:
        self._running_call = call.call_id
        self._emit(call_update(call.call_id, "in_progress"))

    @override
    def on_tool_result(
        self,
        call: ToolCallInfo,
        tool_result: ToolResult,
        *,
        previewed: bool,
    ) -> None:
        update = call_update(call.call_id, "completed" if tool_result.ok else "failed")
        update["content"] = _result_content(call, tool_result)
        with self._lock:
            if self._live:
                self._open_calls.pop(call.call_id, None)
                self._send(update)

    @override
    def on_notice(self, text: str, *, level: NoticeLevel = "info") -> None:
        # stdout занят протоколом: предупреждения агента — в журнал.
        _log.log(_NOTICE_LEVELS[level], "%s", text)


def text_chunk(kind: str, text: str) -> Update:
    """Текстовый кусок реплики вида ``kind`` (``agent_message_chunk`` и подобные)."""
    return _update(kind, {"content": {"type": "text", "text": text}})


def tool_call_fields(call: ToolCallInfo) -> Update:
    """Поля вызова для ``tool_call`` и запроса подтверждения."""
    return {
        "toolCallId": call.call_id,
        "title": f"{call.name} {call.summary}".strip(),
        "kind": _kind(call),
        "status": "pending",
        "rawInput": dict(call.arguments),
        "locations": [{"path": path} for path in call.paths],
    }


def _update(kind: str, fields: Update) -> Update:
    """Тело уведомления ``session/update`` вида ``kind``."""
    return {"sessionUpdate": kind, **fields}


def call_update(call_id: str, status: str) -> Update:
    """``tool_call_update`` вызова ``call_id`` с новым статусом ``status``."""
    return _update("tool_call_update", {"toolCallId": call_id, "status": status})


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
        content.append(text_content(tool_result.content))
    return content


def _preview_content(call: ToolCallInfo, preview: Display | None) -> list[Update]:
    """Превью операции для подтверждения: дифф правки или текст (команда)."""
    if preview is None:
        return []
    diff = _diff(call, preview)
    if diff is not None:
        return [diff]
    return [text_content(preview.text)]


def text_content(text: str) -> Update:
    """Текст в ``content`` вызова инструмента."""
    return {"type": "content", "content": {"type": "text", "text": text}}


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
