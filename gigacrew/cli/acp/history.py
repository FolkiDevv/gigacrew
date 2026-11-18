"""История сохранённого чата → уведомления ``session/update`` для ``session/load``.

Клиент восстанавливает тред по переигранной истории: реплики пользователя
(``user_message_chunk``) и модели (``agent_message_chunk``), вызовы инструментов
(``tool_call`` и ``tool_call_update`` с итогом) — с id вызовов из истории. Служебные
сообщения harness'а (заметки, краткое содержание сжатия, контекст хуков) репликами
пользователя не считаются. У каждого куска — ``messageId``, свой у каждой реплики: по нему
клиент отличает две реплики одной роли подряд.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from dataclasses import replace

from gigacrew.agent.conversation import INTERRUPTED_NOTE, request_text
from gigacrew.agent.events import ToolCallInfo
from gigacrew.cli.acp.events import (
    Update,
    call_update,
    text_chunk,
    text_content,
    tool_call_fields,
)
from gigacrew.llm.types import FunctionCall, Message
from gigacrew.tools.base import ERROR_PREFIX, is_error_text

__all__ = ["history_updates"]

# Вызов из истории — как его показывали во время хода (:meth:`Agent.call_info`).
type Describe = Callable[[FunctionCall], ToolCallInfo]


def history_updates(messages: Sequence[Message], describe: Describe) -> list[Update]:
    """Уведомления, которыми переигрывается история ``messages``, по порядку."""
    replay = _Replay(describe)
    for message in messages:
        replay.add(message)
    return replay.finish()


class _Replay:
    """Сборка уведомлений истории; результаты вызовов сопоставляются с вызовами."""

    def __init__(self, describe: Describe) -> None:
        self._describe = describe
        self._updates: list[Update] = []
        self._waiting: list[tuple[str, str | None]] = []  # вызовы без результата: id, id модели

    def add(self, message: Message) -> None:
        if message.role == "function":
            self._call_result(message)
        elif message.role == "assistant":
            self._answer(message)
        elif message.is_request:
            self._text("user_message_chunk", request_text(message.content))

    def finish(self) -> list[Update]:
        """Вызовы, оставшиеся без результата, — неудавшиеся."""
        self._fail_waiting()
        return self._updates

    def _answer(self, message: Message) -> None:
        self._fail_waiting()
        self._text("agent_message_chunk", message.content)
        for call in message.function_calls:
            call_id = call.id or str(uuid.uuid4())
            shown = replace(self._describe(call), call_id=call_id, arguments=call.arguments)
            self._updates.append({"sessionUpdate": "tool_call", **tool_call_fields(shown)})
            self._waiting.append((call_id, call.id))

    def _call_result(self, message: Message) -> None:
        """Результат вызова: по id вызова, без id — первому ожидающему."""
        model_id = message.call_id
        matching = [entry for entry in self._waiting if model_id and entry[1] == model_id]
        waiting = matching or self._waiting
        if not waiting:
            return
        entry = waiting[0]
        self._waiting.remove(entry)
        content = message.content
        failed = is_error_text(content) or content.startswith(INTERRUPTED_NOTE)
        update = call_update(entry[0], "failed" if failed else "completed")
        update["content"] = [text_content(content.removeprefix(ERROR_PREFIX))]
        self._updates.append(update)

    def _text(self, kind: str, text: str) -> None:
        if text:
            self._updates.append({**text_chunk(kind, text), "messageId": str(uuid.uuid4())})

    def _fail_waiting(self) -> None:
        self._updates.extend(call_update(call_id, "failed") for call_id, _ in self._waiting)
        self._waiting.clear()
