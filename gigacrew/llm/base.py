"""Абстрактный интерфейс LLM-провайдера.

Агент работает только через этот интерфейс, поэтому добавление нового
провайдера (другой модели/вендора) сводится к реализации одного класса.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from gigacrew.llm.types import AssistantTurn, Message, ToolChoice, ToolSpec


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


@dataclass(frozen=True)
class ModelInfo:
    """Модель из каталога провайдера; ``context_window`` — окно, если его сообщает API."""

    id: str
    context_window: int | None = None


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

    def connect(self) -> None:  # noqa: B027 — необязательный хук
        """Подготовка к запросам: вход, загрузка каталога моделей. По умолчанию ничего.

        CLI вызывает его при запуске и перед каждым запросом пользователя — вне хода,
        поэтому провайдер может попросить пользователя войти (через браузер). Ошибки —
        :class:`LLMError`, отмена входа — ``KeyboardInterrupt``.
        """

    def list_models(self) -> list[str]:
        """Доступные чат-модели (для выбора и диагностики). По умолчанию — только текущая."""
        return [self.model]

    def known_models(self) -> list[ModelInfo] | None:
        """Каталог моделей, если его даёт API: модель выбирается только из него.

        None (по умолчанию) — каталога нет, модель может быть любой.
        """
        return None

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
        tool_choice: ToolChoice = "auto",
        on_notice: Callable[[str], None] | None = None,
    ) -> AssistantTurn:
        """Один проход модели.

        Принимает историю диалога и (опционально) доступные инструменты,
        возвращает ход ассистента: либо текст, либо запрос на вызов функции.
        ``model`` переопределяет модель провайдера на этот вызов; ``tool_choice="none"``
        запрещает вызовы (схемы остаются — история с вызовами без них не принимается).
        ``on_notice`` — сообщения провайдера пользователю: пауза перед повтором (лимит
        запросов шлюза), отказ шлюза принять рассуждения модели. Ошибки обращения
        бросаются как :class:`LLMError`.
        """

    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        *,
        model: str | None = None,
        temperature: float = 0.2,
        tool_choice: ToolChoice = "auto",
        on_delta: Callable[[str], None] | None = None,
        on_notice: Callable[[str], None] | None = None,
    ) -> AssistantTurn:
        """Потоковая версия complete().

        Базовая реализация не стримит, а вызывает complete() и отдаёт весь текст
        одним куском — провайдеры без поддержки SSE работают без изменений.
        Провайдеры с потоком переопределяют метод.
        """
        turn = self.complete(
            messages,
            tools,
            model=model,
            temperature=temperature,
            tool_choice=tool_choice,
            on_notice=on_notice,
        )
        if on_delta and turn.message.content:
            on_delta(turn.message.content)
        return turn
