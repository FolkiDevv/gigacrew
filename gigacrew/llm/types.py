"""Провайдеро-независимые типы сообщений и инструментов.

Эти типы — внутреннее представление. Конкретный провайдер
сериализует их в свой формат запроса и парсит ответ обратно.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "function"]


class FunctionCall(BaseModel):
    """Запрос модели на вызов инструмента."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """Универсальное сообщение диалога.

    - system/user/assistant: обычный текст в ``content``.
    - assistant с function_call: модель просит вызвать инструмент.
    - function: результат выполнения инструмента (``name`` + ``content``).
    """

    role: Role
    content: str = ""
    name: str | None = None  # имя функции для role="function"
    function_call: FunctionCall | None = None
    # Непрозрачный идентификатор состояния функций из ответа API. Хранится в
    # истории, но в запрос не отправляется (см. gigacrew.llm.api).
    functions_state_id: str | None = None


class ToolSpec(BaseModel):
    """Описание инструмента для модели (имя, описание, JSON-schema параметров)."""

    name: str
    description: str
    parameters: dict[str, Any]


class Usage(BaseModel):
    """Расход токенов на одно обращение к модели.

    ``prompt_tokens`` — размер отправленного контекста (история + системный промпт),
    ``completion_tokens`` — ответ модели. Сумма по шагам хода — оплаченные токены;
    ``prompt_tokens + completion_tokens`` последнего шага — текущий размер контекста.
    """

    model_config = ConfigDict(extra="ignore")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    precached_prompt_tokens: int = 0  # часть промпта из кеша

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any] | None) -> Usage:
        """Терпимый разбор поля ``usage`` ответа API (мусор → нули)."""
        if not isinstance(raw, Mapping):
            return cls()
        values: dict[str, int] = {}
        for name in cls.model_fields:
            try:
                values[name] = max(int(raw.get(name) or 0), 0)
            except (TypeError, ValueError):
                values[name] = 0
        return cls(**values)


class AssistantTurn(BaseModel):
    """Результат одного обращения к модели."""

    message: Message
    finish_reason: str = "stop"
    usage: Usage = Field(default_factory=Usage)

    @property
    def wants_tool(self) -> bool:
        return self.message.function_call is not None
