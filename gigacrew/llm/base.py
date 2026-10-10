"""Абстрактный интерфейс LLM-провайдера.

Агент работает только через этот интерфейс, поэтому добавление нового
провайдера (другой модели/вендора) сводится к реализации одного класса.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

from gigacrew.llm.types import AssistantTurn, Message, ToolSpec


class LLMError(RuntimeError):
    """Ошибка обращения к LLM (сеть, авторизация, формат ответа).

    Конкретные провайдеры наследуют свои ошибки от неё; CLI ловит только LLMError,
    не зная о провайдере.
    """


class PromptTooLong(LLMError):
    """Запрос отклонён из-за размера (замер окна контекста, :meth:`LLMProvider.measure_prompt`).

    ``status`` — HTTP-статус ответа (None — таймаут/обрыв), ``detail`` — тело ответа:
    в нём сервер нередко называет свой лимит.
    """

    def __init__(self, status: int | None, detail: str):
        self.status = status
        self.detail = detail
        code = status if status is not None else "нет ответа"
        super().__init__(f"запрос слишком велик ({code}): {detail[:300]}")


class LLMProvider(ABC):
    """Минимальный контракт провайдера для агентного цикла."""

    # Умеет ли провайдер :meth:`measure_prompt` (замер окна контекста).
    supports_measure: bool = False

    @property
    @abstractmethod
    def model(self) -> str:
        """Идентификатор модели по умолчанию."""

    def close(self) -> None:  # noqa: B027 — необязательный хук
        """Освобождает ресурсы (соединения). По умолчанию ничего не делает."""

    def list_models(self) -> list[str]:
        """Доступные чат-модели (для выбора и диагностики). По умолчанию — только текущая."""
        return [self.model]

    def measure_prompt(self, text: str, *, model: str | None = None) -> int:
        """Отправляет ``text`` одним сообщением с минимальным ответом; возвращает
        ``prompt_tokens`` по данным API — точный размер запроса для этой модели.

        Запрос, не уместившийся в окно, — :class:`PromptTooLong`; прочие ошибки —
        :class:`LLMError`. Провайдер с поддержкой замера выставляет ``supports_measure``.
        """
        raise NotImplementedError

    @abstractmethod
    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        *,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> AssistantTurn:
        """Один проход модели.

        Принимает историю диалога и (опционально) доступные инструменты,
        возвращает ход ассистента: либо текст, либо запрос на вызов функции.
        ``model`` переопределяет модель провайдера на этот вызов.
        Ошибки обращения бросаются как :class:`LLMError`.
        """

    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        *,
        model: str | None = None,
        temperature: float = 0.2,
        on_delta: Callable[[str], None] | None = None,
    ) -> AssistantTurn:
        """Потоковая версия complete().

        Базовая реализация не стримит, а вызывает complete() и отдаёт весь текст
        одним куском — провайдеры без поддержки SSE работают без изменений.
        Провайдеры с потоком переопределяют метод.
        """
        turn = self.complete(messages, tools, model=model, temperature=temperature)
        if on_delta and turn.message.content:
            on_delta(turn.message.content)
        return turn
