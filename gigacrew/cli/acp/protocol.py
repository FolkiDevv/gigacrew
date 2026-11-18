"""Сообщения ACP: разбор параметров методов (pydantic) и сборка ответов.

Неверные параметры — :class:`pydantic.ValidationError`: транспорт отвечает на него
ошибкой JSON-RPC «Invalid params».
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "PROTOCOL_VERSION",
    "InitializeParams",
    "NewSessionParams",
    "initialize_response",
]

# Версия ACP, которую понимает gigacrew (и T3 Code).
PROTOCOL_VERSION: Final = 1

type Message = dict[str, Any]


class InitializeParams(BaseModel):
    """Параметры ``initialize``: возможности клиента пока не используются."""

    protocol_version: int = Field(alias="protocolVersion", strict=True)


class NewSessionParams(BaseModel):
    """Параметры ``session/new``: каталог сессии — абсолютный путь к каталогу."""

    cwd: Path
    # MCP-серверы клиента пока не подключаются: принимаются и не запускаются.
    mcp_servers: list[Any] = Field(default_factory=list, alias="mcpServers")

    @field_validator("cwd")
    @classmethod
    def _existing_directory(cls, cwd: Path) -> Path:
        if not cwd.is_absolute():
            raise ValueError(f"cwd должен быть абсолютным путём: {cwd}")
        if not cwd.is_dir():
            raise ValueError(f"cwd: каталог не найден: {cwd}")
        return cwd


def initialize_response(version: str) -> Message:
    """Ответ на ``initialize``: версия протокола, возможности и сведения об агенте.

    Клиенту с другой версией протокола отвечаем своей: решать, работать ли с ней, — ему.
    """
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "agentCapabilities": {
            "loadSession": False,
            "promptCapabilities": {"image": False, "audio": False, "embeddedContext": True},
        },
        "agentInfo": {"name": "gigacrew", "title": "GigaCrew", "version": version},
        "authMethods": [],
    }
