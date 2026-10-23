"""Провайдеро-независимые типы сообщений и инструментов.

Эти типы — внутреннее представление. Конкретный провайдер
сериализует их в свой формат запроса и парсит ответ обратно.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Role = Literal["system", "user", "assistant", "function"]
# Может ли модель вызывать инструменты в этом ответе: "none" — только текст (итог
# после остановки), хотя схемы инструментов в запросе остаются.
ToolChoice = Literal["auto", "none"]


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

    ``meta`` — служебное user-сообщение harness'а (одобрение плана, самопроверка): модель
    получает его как обычное, но это не запрос пользователя — не заголовок чата, не
    счётчик запросов, не закреплённая задача.

    ``reasoning`` — рассуждения модели перед этим ответом (``role: "reasoning"`` в API v2).
    Хранятся, только если их возвращают модели (``GIGACREW_REASONING=turn``, по
    умолчанию), и лишь до следующего запроса пользователя.
    """

    role: Role
    content: str = ""
    name: str | None = None  # имя функции для role="function"
    function_calls: list[FunctionCall] = Field(default_factory=list)
    call_id: str | None = None  # id вызова, на который отвечает role="function"
    # Непрозрачный идентификатор состояния инструментов из ответа API: возвращается
    # с assistant-сообщением и результатами его вызовов.
    state_id: str | None = None
    meta: bool = False
    reasoning: str = ""

    @property
    def is_request(self) -> bool:
        """Запрос пользователя (не служебное сообщение harness'а)."""
        return self.role == "user" and not self.meta

    @model_validator(mode="before")
    @classmethod
    def _legacy_fields(cls, raw: Any) -> Any:
        """Чаты, сохранённые до перехода на API v2.

        В них один ``function_call`` и ``functions_state_id`` (состояние старого API
        в новом не действует).
        """
        if not isinstance(raw, Mapping) or (
            "function_call" not in raw and "functions_state_id" not in raw
        ):
            return raw
        fields = dict(raw)
        call = fields.pop("function_call", None)
        fields.pop("functions_state_id", None)
        if call is not None and not fields.get("function_calls"):
            fields["function_calls"] = [call]
        return fields


class ToolSpec(BaseModel):
    """Описание инструмента для модели (имя, описание, JSON-schema параметров)."""

    name: str
    description: str
    # Имя поля — термин JSON-schema функции и часть API модуля (``spec.parameters``).
    parameters: dict[str, Any]  # noqa: WPS110


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

        Понимает обе схемы: ``prompt_tokens``/``completion_tokens`` (+
        ``prompt_tokens_details.cached_tokens`` или ``cached_tokens``) и
        ``input_tokens``/``output_tokens`` (+ ``input_tokens_details.cached_tokens``).
        """
        if not isinstance(raw, Mapping):
            return cls()
        cached = raw.get("cached_tokens")
        for key in ("input_tokens_details", "prompt_tokens_details"):
            details = raw.get(key)
            if isinstance(details, Mapping) and details.get("cached_tokens") is not None:
                cached = details["cached_tokens"]
                break
        counts: dict[str, int] = {}
        for name in cls.model_fields:
            count = raw.get(name)
            if count is None and name in _USAGE_V2_KEYS:
                count = raw.get(_USAGE_V2_KEYS[name])
            if count is None and name == "precached_prompt_tokens":
                count = cached
            try:
                counts[name] = max(int(count or 0), 0)
            except (TypeError, ValueError):
                counts[name] = 0
        return cls(**counts)


class AssistantTurn(BaseModel):
    """Результат одного обращения к модели."""

    message: Message
    finish_reason: str = "stop"
    usage: Usage = Field(default_factory=Usage)

    @property
    def wants_tool(self) -> bool:
        """Модель просит вызвать инструменты."""
        return bool(self.message.function_calls)
