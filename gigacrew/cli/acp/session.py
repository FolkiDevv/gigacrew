"""Сессия ACP: агент над своим провайдером и MCP-серверами проекта.

Сессия собирается как запуск CLI в каталоге сессии, но без терминала: конфигурация —
из окружения процесса и ``.env`` каталога сессии, хуки и MCP-серверы — только
подтверждённые раньше, предупреждения — в журнал (stderr), а не в stdout: stdout занят
протоколом. Любая ошибка сборки — :class:`~gigacrew.cli.acp.rpc.RpcError` с понятным
сообщением; уже созданные провайдер и MCP-серверы при этом закрываются.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from gigacrew.agent.chat_store import ChatRecorder, ChatStore
from gigacrew.agent.events import AgentEvents
from gigacrew.agent.loop import Agent
from gigacrew.cli.acp.rpc import INTERNAL_ERROR, RpcError
from gigacrew.cli.assembly import (
    ModelChoiceError,
    build_agent,
    load_windows,
    select_model,
)
from gigacrew.cli.hooks import install_request_hook, trusted_hooks
from gigacrew.cli.mcp import connect_trusted_mcp
from gigacrew.config import Config, ConfigError
from gigacrew.llm.base import LLMError, LLMProvider
from gigacrew.project.workspace import Workspace
from gigacrew.tools.base import ToolRegistry
from gigacrew.tools.mcp import McpHub

__all__ = ["AcpSession", "ProviderFactory", "RegistryFactory", "open_session"]

type ProviderFactory = Callable[[Config], LLMProvider]
type RegistryFactory = Callable[[], ToolRegistry]

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AcpSession:
    """Сессия: агент, запись её чата и MCP-серверы; id сессии — id чата."""

    agent: Agent
    chats: ChatRecorder
    mcp: McpHub

    @property
    def session_id(self) -> str:
        """Id сессии — id её чата."""
        return self.chats.chat_id

    def close(self) -> None:
        """Закрывает MCP-серверы и провайдер — и тот, и другой, даже если один упал."""
        with ExitStack() as stack:
            stack.callback(self.agent.provider.close)
            stack.callback(self.mcp.close)


def open_session(
    cwd: Path,
    *,
    provider_factory: ProviderFactory,
    registry_factory: RegistryFactory,
) -> AcpSession:
    """Новая сессия в каталоге ``cwd``: конфигурация, провайдер, агент; хуки SessionStart."""
    config = _load_config(cwd)
    try:
        provider = provider_factory(config)
    except Exception as error:
        raise _session_error(f"не удалось инициализировать провайдера: {error}") from None
    with ExitStack() as cleanup:
        cleanup.callback(provider.close)
        try:
            session = _assemble(config, provider, registry_factory, cleanup)
        except Exception as error:
            raise _as_session_error(error) from None
        cleanup.pop_all()
    return session


def _warn(text: str) -> None:
    _log.warning("%s", text)


def _session_error(message: str) -> RpcError:
    return RpcError(INTERNAL_ERROR, message)


def _as_session_error(error: Exception) -> RpcError:
    """Ошибка сборки сессии для клиента: суть без трассировки."""
    if isinstance(error, RpcError):
        return error
    return _session_error(f"не удалось создать сессию: {type(error).__name__}: {error}")


def _load_config(cwd: Path) -> Config:
    """Конфигурация каталога сессии: ``.env`` текущего каталога процесса не участвует."""
    try:
        config = Config.load(project_root=cwd, cwd=cwd)
        _warn_about(config)
        config.require_credentials()
    except ConfigError as error:
        raise _session_error(f"ошибка конфигурации: {error}") from None
    except RuntimeError as error:
        raise _session_error(str(error)) from None
    return config


def _warn_about(config: Config) -> None:
    """Предупреждения запуска, как у CLI: пропущенные настройки .env проекта, TLS."""
    notices = [config.ignored_env_warning, *config.preference_warnings, config.tls_warning]
    for notice in notices:
        if notice:
            _warn(notice)


def _connect(provider: LLMProvider) -> None:
    try:
        provider.connect()  # токен и каталог моделей (бэкенд GigaCode)
    except LLMError as error:
        raise _session_error(f"не удалось подключиться: {error}") from None


def _with_model(config: Config, provider: LLMProvider, known: list[str] | None) -> Config:
    try:
        return select_model(config, provider, known)
    except ModelChoiceError as error:
        raise _session_error(str(error)) from None


def _assemble(
    config: Config,
    provider: LLMProvider,
    registry_factory: RegistryFactory,
    cleanup: ExitStack,
) -> AcpSession:
    """Хуки, подключение, модель, MCP-серверы и агент; закрыть нужное — в ``cleanup``."""
    workspace = Workspace(
        config.project_root,
        untrusted=config.untrusted_project,
        extra_dirs=config.add_dirs,
    )
    events = AgentEvents()
    hooks = trusted_hooks(workspace, _warn)
    install_request_hook(provider, hooks, events, config)
    _connect(provider)
    windows, known = load_windows(config, provider, _warn)
    config = _with_model(config, provider, known)
    registry = registry_factory()
    mcp = connect_trusted_mcp(
        workspace,
        registry,
        warn=_warn,
        version=metadata.version("gigacrew"),
    )
    cleanup.callback(mcp.close)
    agent = build_agent(
        config,
        provider,
        events,
        windows=windows,
        hooks=hooks,
        registry=registry,
        warn=_warn,
    )
    chats = ChatRecorder(ChatStore(agent.workspace), enabled=config.save_chats)
    agent.chat_id = chats.chat_id  # файл плана назван по id чата
    agent.start_session("startup")
    return AcpSession(agent, chats, mcp)
