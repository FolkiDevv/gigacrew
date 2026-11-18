"""Сообщения ACP: разбор параметров методов (pydantic) и сборка ответов.

Неверные параметры — :class:`pydantic.ValidationError`: транспорт отвечает на него
ошибкой JSON-RPC «Invalid params».
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, Field, field_validator

from gigacrew.agent.events import TurnStats

__all__ = [
    "CANCELLED",
    "PROTOCOL_VERSION",
    "CancelParams",
    "InitializeParams",
    "NewSessionParams",
    "PromptParams",
    "initialize_response",
    "prompt_text",
    "stop_reason",
]

# Версия ACP, которую понимает gigacrew (и T3 Code).
PROTOCOL_VERSION: Final = 1

# stopReason ответа на session/prompt, если ход отменён.
CANCELLED: Final = "cancelled"

# Почему ограничитель остановил ход → stopReason ACP; прочие остановки — end_turn.
_STOP_REASONS: Final = MappingProxyType({
    "max_steps": "max_turn_requests",
    "output_limit": "max_tokens",
})

type Message = dict[str, Any]


class InitializeParams(BaseModel):
    """Параметры ``initialize``: версия протокола и возможности клиента."""

    protocol_version: int = Field(alias="protocolVersion", strict=True)
    client_capabilities: Any = Field(default=None, alias="clientCapabilities")

    @property
    def form_questions(self) -> bool:
        """Клиент объявил формы ``elicitation/create`` (``elicitation.form`` — объект)."""
        capabilities = self.client_capabilities
        if not isinstance(capabilities, dict):
            return False
        elicitation = capabilities.get("elicitation")
        return isinstance(elicitation, dict) and isinstance(elicitation.get("form"), dict)


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


class TextBlock(BaseModel):
    """Текст запроса пользователя."""

    kind: Literal["text"] = Field(alias="type")
    text: str


class ResourceLinkBlock(BaseModel):
    """Ссылка на ресурс (файл): модель получает его адрес, а не содержимое."""

    kind: Literal["resource_link"] = Field(alias="type")
    uri: str
    name: str = ""


class EmbeddedResource(BaseModel):
    """Содержимое встроенного ресурса; двоичное (``blob``) модели не передаётся."""

    uri: str
    text: str = ""


class ResourceBlock(BaseModel):
    """Встроенный ресурс: адрес и текст."""

    kind: Literal["resource"] = Field(alias="type")
    resource: EmbeddedResource


# Блок запроса; картинок и звука агент не принимает (promptCapabilities).
type ContentBlock = Annotated[
    TextBlock | ResourceLinkBlock | ResourceBlock,
    Field(discriminator="kind"),
]


class PromptParams(BaseModel):
    """Параметры ``session/prompt``: id сессии и непустой запрос из блоков."""

    session_id: str = Field(alias="sessionId")
    prompt: list[ContentBlock] = Field(min_length=1)


class CancelParams(BaseModel):
    """Параметры уведомления ``session/cancel``."""

    session_id: str = Field(alias="sessionId")


def prompt_text(blocks: list[ContentBlock]) -> str:
    """Запрос пользователя одним текстом: блоки по порядку, через пустую строку."""
    return "\n\n".join(_block_text(block) for block in blocks)


def stop_reason(stats: TurnStats | None) -> str:
    """Причина конца хода (stopReason ответа) по его итогам."""
    if stats is None or stats.stop_reason is None:
        return "end_turn"
    return _STOP_REASONS.get(stats.stop_reason, "end_turn")


def _block_text(block: ContentBlock) -> str:
    if isinstance(block, TextBlock):
        return block.text
    if isinstance(block, ResourceLinkBlock):
        return f"@{_location(block.uri)}"
    location = _location(block.resource.uri)
    return "\n".join((f'<context ref="{location}">', block.resource.text, "</context>"))


def _location(uri: str) -> str:
    """Путь файла для ``file://``-адреса, иначе — адрес как есть."""
    try:
        return str(Path.from_uri(uri))
    except ValueError:
        return uri
