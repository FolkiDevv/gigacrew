"""Провайдеро-независимые типы сообщений и инструментов.

Эти типы — внутреннее представление. Конкретный провайдер
сериализует их в свой формат запроса и парсит ответ обратно.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Role = Literal["system", "user", "assistant", "function"]


class FunctionCall(BaseModel):
    """Запрос модели на вызов инструмента."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    # Идентификатор вызова от API: результат возвращается с тем же id.
    id: str | None = None


class Message(BaseModel):
    """Универсальное сообщение диалога.

    - system/user/assistant: обычный текст в ``content``.
    - assistant с ``function_calls``: модель просит вызвать инструменты (один или
      несколько за раз); на каждый вызов по порядку идёт свой результат.
    - function: результат одного вызова (``name`` + ``content``, ``call_id`` — id вызова).
    """

    role: Role
    content: str = ""
    name: str | None = None  # имя функции для role="function"
    function_calls: list[FunctionCall] = Field(default_factory=list)
    call_id: str | None = None  # id вызова, на который отвечает role="function"
    # Непрозрачный идентификатор состояния инструментов из ответа API: возвращается
    # с assistant-сообщением и результатами его вызовов.
    state_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _legacy_fields(cls, data: Any) -> Any:
        """Чаты, сохранённые до перехода на API v2: один ``function_call`` и
        ``functions_state_id`` (состояние старого API в новом не действует)."""
        if not isinstance(data, Mapping) or (
            "function_call" not in data and "functions_state_id" not in data
        ):
            return data
        data = dict(data)
        call = data.pop("function_call", None)
        data.pop("functions_state_id", None)
        if call is not None and not data.get("function_calls"):
            data["function_calls"] = [call]
        return data


class ToolSpec(BaseModel):
    """Описание инструмента для модели (имя, описание, JSON-schema параметров)."""

    name: str
    description: str
    parameters: dict[str, Any]


# Имена полей ``usage`` в ответе API v2.
_USAGE_V2_KEYS = {"prompt_tokens": "input_tokens", "completion_tokens": "output_tokens"}


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
        """Терпимый разбор поля ``usage`` ответа API (мусор → нули).

        Понимает обе схемы: ``prompt_tokens``/``completion_tokens`` и
        ``input_tokens``/``output_tokens`` (+ ``input_tokens_details.cached_tokens``).
        """
        if not isinstance(raw, Mapping):
            return cls()
        details = raw.get("input_tokens_details")
        cached = details.get("cached_tokens") if isinstance(details, Mapping) else None
        values: dict[str, int] = {}
        for name in cls.model_fields:
            value = raw.get(name)
            if value is None and name in _USAGE_V2_KEYS:
                value = raw.get(_USAGE_V2_KEYS[name])
            if value is None and name == "precached_prompt_tokens":
                value = cached
            try:
                values[name] = max(int(value or 0), 0)
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
        return bool(self.message.function_calls)
