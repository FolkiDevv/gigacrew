"""LLM-слой: абстрактный провайдер и HTTP-реализации — API v2 и бэкенд GigaCode."""

from gigacrew.llm.api import ApiError, ApiProvider
from gigacrew.llm.base import LLMError, LLMProvider, ModelInfo
from gigacrew.llm.gigacode import GigaCodeProvider
from gigacrew.llm.types import (
    AssistantTurn,
    FunctionCall,
    Message,
    ToolSpec,
    Usage,
)

__all__ = [
    "LLMError",
    "LLMProvider",
    "ModelInfo",
    "ApiError",
    "ApiProvider",
    "GigaCodeProvider",
    "Usage",
    "Message",
    "ToolSpec",
    "FunctionCall",
    "AssistantTurn",
]
