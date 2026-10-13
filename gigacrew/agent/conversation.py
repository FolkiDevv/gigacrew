"""История диалога агента.

Чистая структура данных без побочных эффектов: не читает файловую систему и
не знает о системном промпте. Сериализуется в версионированный словарь — на
этом строится сохранение и восстановление чатов. Хранит расход токенов
последнего обращения (``last_usage``).

Сжатие контекста не переписывает журнал: :class:`Summary` заменяет начало истории
кратким содержанием только в том, что видит модель (:meth:`Conversation.context_messages`).
Полный журнал остаётся для показа при продолжении чата, заголовка и счётчиков.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from gigacrew.llm.types import FunctionCall, Message, Usage

INTERRUPTED_NOTE = "Execution was interrupted by the user (Ctrl+C). No result."


@dataclass(frozen=True)
class Summary:
    """Краткое содержание начала диалога: ``messages[:upto]`` заменены текстом ``text``."""

    text: str
    upto: int


class Conversation:
    # Версия 2 — несколько вызовов инструментов в сообщении (``function_calls``, API v2).
    # Версия 1 читается: её единственный ``function_call`` приводит :class:`Message`.
    # Ключ ``summary`` необязателен: версия gigacrew без сжатия его не видит и
    # отправляет журнал целиком, как раньше.
    FORMAT_VERSION = 2
    _READABLE_VERSIONS = (1, 2)

    def __init__(
        self,
        messages: Iterable[Message] = (),
        *,
        last_usage: Usage | None = None,
        summary: Summary | None = None,
    ):
        self._messages: list[Message] = list(messages)
        self._last_usage = last_usage
        self._summary: Summary | None = None
        if summary is not None:
            self._check_summary(summary)
            self._summary = summary

    # ------------------------------------------------------------------ #
    @property
    def messages(self) -> tuple[Message, ...]:
        return tuple(self._messages)

    @property
    def last_usage(self) -> Usage | None:
        """Расход токенов последнего обращения к модели (None — ещё не было)."""
        return self._last_usage

    @property
    def summary(self) -> Summary | None:
        """Краткое содержание сжатого начала диалога (None — диалог не сжимался)."""
        return self._summary

    def __len__(self) -> int:
        return len(self._messages)

    def context_messages(self) -> list[Message]:
        """Сообщения, которые видит модель: всё после границы сжатия.

        Если после границы нет запроса пользователя (сжатие посреди длинного хода),
        спереди добавляется последний запрос до неё — текущая задача остаётся
        дословной. С первым новым запросом закрепление исчезает само.
        """
        if self._summary is None:
            return list(self._messages)
        tail = self._messages[self._summary.upto :]
        if not tail or any(m.is_request for m in tail):
            return tail
        task = next(
            (m for m in reversed(self._messages[: self._summary.upto]) if m.is_request),
            None,
        )
        return tail if task is None else [task, *tail]

    def set_summary(self, summary: Summary) -> None:
        """Заменить начало диалога кратким содержанием (сжатие контекста).

        ``last_usage`` сбрасывается: он описывал контекст до сжатия.
        """
        self._check_summary(summary)
        if self._summary is not None and summary.upto < self._summary.upto:
            raise ValueError("граница сжатия не может сдвигаться назад")
        self._summary = summary
        self._last_usage = None

    def _check_summary(self, summary: Summary) -> None:
        if not summary.text.strip():
            raise ValueError("пустое краткое содержание")
        if not 0 < summary.upto <= len(self._messages):
            raise ValueError(f"граница сжатия вне диалога: {summary.upto}")
        if summary.upto < len(self._messages) and self._messages[summary.upto].role == "function":
            # Результат без своего вызова — некорректный запрос к API.
            raise ValueError("граница сжатия не может приходиться на результат инструмента")

    # ------------------------------------------------------------------ #
    def add_user(self, text: str, *, meta: bool = False) -> None:
        """Запрос пользователя; ``meta`` — служебное сообщение harness'а (см. :class:`Message`)."""
        self._messages.append(Message(role="user", content=text, meta=meta))

    def add_assistant(self, message: Message, usage: Usage | None = None) -> None:
        self._messages.append(message)
        if usage is not None:
            self._last_usage = usage

    def add_function_result(self, name: str, content: str, call_id: str | None = None) -> None:
        """Результат вызова инструмента. Результаты идут по порядку вызовов, поэтому
        без ``call_id`` берётся id первого вызова, на который ещё нет ответа."""
        if call_id is None:
            pending = self.pending_calls()
            call_id = pending[0].id if pending else None
        self._messages.append(Message(role="function", name=name, content=content, call_id=call_id))

    # ------------------------------------------------------------------ #
    def pending_calls(self) -> list[FunctionCall]:
        """Вызовы последнего assistant-сообщения, на которые ещё нет результатов.

        Результаты идут сразу после сообщения с вызовами, по одному на вызов и по порядку.
        """
        answered = 0
        for message in reversed(self._messages):
            if message.role == "function":
                answered += 1
                continue
            if message.role == "assistant":
                return list(message.function_calls[answered:])
            break
        return []

    def repair(self, note: str = INTERRUPTED_NOTE) -> bool:
        """Закрывает «висящие» вызовы инструментов ответами-заглушками.

        Без этого следующий запрос к API содержал бы вызовы без результатов
        (API отвечает 422). Возвращает True, если история была исправлена.
        """
        pending = self.pending_calls()
        for call in pending:
            self.add_function_result(call.name, note, call.id)
        return bool(pending)

    def truncate(self, length: int) -> None:
        """Убрать сообщения после первых ``length`` (откат служебного обмена репликами).

        Граница сжатия не может оказаться за концом истории.
        """
        if not 0 <= length <= len(self._messages):
            raise ValueError(f"длина вне диалога: {length}")
        if self._summary is not None and self._summary.upto > length:
            raise ValueError("нельзя убрать сообщения, свёрнутые в краткое содержание")
        del self._messages[length:]

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "version": self.FORMAT_VERSION,
            "messages": [m.model_dump(exclude_defaults=True) for m in self._messages],
            "last_usage": self._last_usage.model_dump() if self._last_usage else None,
        }
        if self._summary is not None:
            data["summary"] = {"text": self._summary.text, "upto": self._summary.upto}
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Conversation:
        version = data.get("version")
        if version not in cls._READABLE_VERSIONS:
            raise ValueError(f"Неподдерживаемая версия формата диалога: {version!r}")
        messages = [Message.model_validate(m) for m in data.get("messages") or []]
        usage = data.get("last_usage")
        raw = data.get("summary")
        summary = None
        if raw is not None:
            if not isinstance(raw, Mapping):
                raise ValueError("поле summary — не объект")
            text, upto = raw.get("text"), raw.get("upto")
            if not isinstance(text, str) or not isinstance(upto, int) or isinstance(upto, bool):
                raise ValueError("поле summary: ожидаются text (строка) и upto (число)")
            summary = Summary(text=text, upto=upto)
        return cls(
            messages,
            last_usage=Usage.model_validate(usage) if usage else None,
            summary=summary,
        )
