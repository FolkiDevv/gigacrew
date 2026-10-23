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

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from gigacrew.llm.types import FunctionCall, Message, Usage

INTERRUPTED_NOTE = "Execution was interrupted by the user (Ctrl+C). No result."
# Первая строка блока <system-reminder>, который хуки UserPromptSubmit дописывают к запросу
# пользователя: по ней блок отрезается при показе запроса (заголовок чата, продолжение).
HOOK_CONTEXT_HEADER = "Context from the user's UserPromptSubmit hooks:"
_HOOK_CONTEXT_TAIL = f"\n\n<system-reminder>\n{HOOK_CONTEXT_HEADER}\n"
# Версии формата, которые читает :meth:`Conversation.from_dict`.
_READABLE_VERSIONS = (1, 2)


# Одиночные суррогаты: в Python так декодируются байты не в UTF-8 в именах файлов
# (архив из cp1251: «\udcee…»), а из JSON приходят экранированные «\ud800».
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def clean_text(text: str) -> str:
    """Текст без одиночных суррогатов — каждый заменяется на U+FFFD.

    Такой текст нельзя закодировать в UTF-8: одно имя файла в истории или системном
    промпте ломало бы каждый запрос к модели и сохранение чата.
    """
    return _SURROGATE_RE.sub("\ufffd", text) if _SURROGATE_RE.search(text) else text


def _clean_value(argument: Any) -> Any:
    """Аргументы вызова инструмента — без одиночных суррогатов (вложенно)."""
    if isinstance(argument, str):
        return clean_text(argument)
    if isinstance(argument, dict):
        return {clean_text(str(key)): _clean_value(nested) for key, nested in argument.items()}
    if isinstance(argument, list):
        return [_clean_value(element) for element in argument]
    return argument


def _clean_message(message: Message) -> Message:
    """Сообщение без одиночных суррогатов (то же сообщение, если их нет)."""
    calls = [
        call.model_copy(update={"arguments": _clean_value(call.arguments)})
        for call in message.function_calls
    ]
    update: dict[str, Any] = {}
    if any(
        new.arguments != old.arguments
        for new, old in zip(calls, message.function_calls, strict=True)
    ):
        update["function_calls"] = calls
    for key in ("content", "reasoning"):
        text = getattr(message, key)
        cleaned = clean_text(text)
        if cleaned is not text:
            update[key] = cleaned
    return message.model_copy(update=update) if update else message


def request_text(content: str) -> str:
    """Текст запроса пользователя для показа — без контекста хуков UserPromptSubmit."""
    head, found, _ = content.partition(_HOOK_CONTEXT_TAIL)
    return head if found else content


@dataclass(frozen=True)
class Summary:
    """Краткое содержание начала диалога: ``messages[:upto]`` заменены текстом ``text``."""

    text: str
    upto: int


def _check_summary(summary: Summary, messages: Sequence[Message]) -> None:
    """Проверить краткое содержание: текст не пуст, граница — внутри диалога ``messages``."""
    if not summary.text.strip():
        raise ValueError("пустое краткое содержание")
    if not 0 < summary.upto <= len(messages):
        raise ValueError(f"граница сжатия вне диалога: {summary.upto}")
    if summary.upto < len(messages) and messages[summary.upto].role == "function":
        # Результат без своего вызова — некорректный запрос к API.
        raise ValueError("граница сжатия не может приходиться на результат инструмента")


# Версия 2 — несколько вызовов инструментов в сообщении (``function_calls``, API v2).
# Версия 1 читается: её единственный ``function_call`` приводит :class:`Message`.
# Ключ ``summary`` необязателен: версия gigacrew без сжатия его не видит и
# отправляет журнал целиком, как раньше.
FORMAT_VERSION = 2


class Conversation:
    """История диалога: сообщения, расход токенов и граница сжатия контекста."""

    def __init__(
        self,
        messages: Iterable[Message] = (),
        *,
        last_usage: Usage | None = None,
        summary: Summary | None = None,
    ):
        """Диалог из ``messages``; ``summary`` проверяется на соответствие сообщениям."""
        self._messages: list[Message] = list(messages)
        self._last_usage = last_usage
        self._summary: Summary | None = None
        if summary is not None:
            _check_summary(summary, self._messages)
            self._summary = summary

    # ------------------------------------------------------------------ #
    @property
    def messages(self) -> tuple[Message, ...]:
        """Полный журнал диалога (включая свёрнутое в краткое содержание)."""
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
        """Число сообщений в полном журнале."""
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
        if not tail or any(message.is_request for message in tail):
            return tail
        task = next(
            (
                message
                for message in reversed(self._messages[: self._summary.upto])
                if message.is_request
            ),
            None,
        )
        return tail if task is None else [task, *tail]

    # Не сеттер свойства: проверяет границу и сбрасывает last_usage — побочный эффект явный.
    def set_summary(self, summary: Summary) -> None:  # noqa: WPS615
        """Заменить начало диалога кратким содержанием (сжатие контекста).

        ``last_usage`` сбрасывается: он описывал контекст до сжатия.
        """
        _check_summary(summary, self._messages)
        if self._summary is not None and summary.upto < self._summary.upto:
            raise ValueError("граница сжатия не может сдвигаться назад")
        self._summary = summary
        self._last_usage = None

    # ------------------------------------------------------------------ #
    # Всё, что попадает в историю, — без одиночных суррогатов (:func:`clean_text`):
    # имена файлов из вывода инструментов, запросы, ответы модели.
    def add_user(self, text: str, *, meta: bool = False) -> None:
        """Запрос пользователя; ``meta`` — служебное сообщение harness'а (см. :class:`Message`)."""
        self._messages.append(Message(role="user", content=clean_text(text), meta=meta))

    def add_assistant(self, message: Message, usage: Usage | None = None) -> None:
        """Ответ модели; ``usage`` — расход токенов этого обращения (если известен)."""
        self._messages.append(_clean_message(message))
        if usage is not None:
            self._last_usage = usage

    def add_function_result(self, name: str, content: str, call_id: str | None = None) -> None:
        """Результат вызова инструмента.

        Результаты идут по порядку вызовов, поэтому без ``call_id`` берётся id первого
        вызова, на который ещё нет ответа.
        """
        if call_id is None:
            pending = self.pending_calls()
            call_id = pending[0].id if pending else None
        self._messages.append(
            Message(role="function", name=name, content=clean_text(content), call_id=call_id),
        )

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

    def forget_reasoning(self) -> None:
        """Убрать рассуждения модели из истории: они нужны ей только в пределах хода."""
        self._messages = [
            message.model_copy(update={"reasoning": ""}) if message.reasoning else message
            for message in self._messages
        ]

    def truncate(self, length: int) -> None:
        """Убрать сообщения после первых ``length`` (откат служебного обмена репликами).

        Граница сжатия не может оказаться за концом истории.
        """
        if not 0 <= length <= len(self._messages):
            raise ValueError(f"длина вне диалога: {length}")
        if self._summary is not None and self._summary.upto > length:
            raise ValueError("нельзя убрать сообщения, свёрнутые в краткое содержание")
        self._messages = self._messages[:length]

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """Диалог версионированным словарём для сохранения чата (см. :meth:`from_dict`)."""
        serialized: dict[str, Any] = {
            "version": FORMAT_VERSION,
            "messages": [message.model_dump(exclude_defaults=True) for message in self._messages],
            "last_usage": self._last_usage.model_dump() if self._last_usage else None,
        }
        if self._summary is not None:
            serialized["summary"] = {"text": self._summary.text, "upto": self._summary.upto}
        return serialized

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Conversation:
        """Диалог из словаря :meth:`to_dict`; неверный формат — :class:`ValueError`."""
        version = payload.get("version")
        if version not in _READABLE_VERSIONS:
            raise ValueError(f"Неподдерживаемая версия формата диалога: {version!r}")
        # Файл чата — тоже вход истории: «\udcee» в JSON читается одиночным суррогатом.
        messages = [
            _clean_message(Message.model_validate(entry)) for entry in payload.get("messages") or []
        ]
        usage = payload.get("last_usage")
        raw = payload.get("summary")
        summary = None
        if raw is not None:
            if not isinstance(raw, Mapping):
                raise ValueError("поле summary — не объект")
            text, upto = raw.get("text"), raw.get("upto")
            if not isinstance(text, str) or not isinstance(upto, int) or isinstance(upto, bool):
                raise ValueError("поле summary: ожидаются text (строка) и upto (число)")
            summary = Summary(text=clean_text(text), upto=upto)
        return cls(
            messages,
            last_usage=Usage.model_validate(usage) if usage else None,
            summary=summary,
        )
