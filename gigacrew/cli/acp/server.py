"""ACP-сервер gigacrew: обработчики методов ACP поверх соединения JSON-RPC.

Сессий в процессе может быть несколько, у каждой — свой каталог, конфигурация,
провайдер и MCP-серверы (:mod:`gigacrew.cli.acp.session`). Конец ввода закрывает все
сессии.
"""

from __future__ import annotations

import threading
from contextlib import ExitStack
from importlib import metadata
from typing import Any, BinaryIO

from gigacrew.cli.acp.protocol import InitializeParams, NewSessionParams, initialize_response
from gigacrew.cli.acp.rpc import INTERNAL_ERROR, INVALID_PARAMS, RpcConnection, RpcError
from gigacrew.cli.acp.session import AcpSession, ProviderFactory, RegistryFactory, open_session
from gigacrew.tools.base import build_default_registry

__all__ = ["AcpServer"]

type Message = dict[str, Any]


class AcpServer:
    """Агент ACP: ``initialize``, ``authenticate``, ``session/new``.

    ``provider_factory`` — провайдер LLM по конфигурации сессии; ``registry_factory`` —
    инструменты агента новой сессии.
    """

    def __init__(
        self,
        *,
        provider_factory: ProviderFactory,
        registry_factory: RegistryFactory = build_default_registry,
    ) -> None:
        """Сервер без сессий; работать начинает :meth:`serve`."""
        self._provider_factory = provider_factory
        self._registry_factory = registry_factory
        self._lock = threading.Lock()
        self._sessions: dict[str, AcpSession] = {}
        self._closed = False

    def serve(self, reader: BinaryIO, writer: BinaryIO) -> None:
        """Обслуживает клиента до конца ввода, затем закрывает все сессии."""
        connection = RpcConnection(reader, writer)
        with ExitStack() as stack:
            stack.callback(self._close_sessions)
            connection.serve(
                requests={
                    "initialize": self._initialize,
                    "authenticate": self._authenticate,
                    "session/new": self._new_session,
                },
                notifications={},
            )

    def session(self, session_id: str) -> AcpSession:
        """Открытая сессия по id; нет такой — ошибка JSON-RPC «Invalid params»."""
        with self._lock:
            found = self._sessions.get(session_id)
        if found is None:
            raise RpcError(INVALID_PARAMS, f"сессия не найдена: {session_id}")
        return found

    def _initialize(self, params: Any) -> Message:
        InitializeParams.model_validate(params)
        return initialize_response(metadata.version("gigacrew"))

    def _authenticate(self, params: Any) -> Message:
        # Реквизиты — в окружении инстанса: проверять у клиента нечего.
        return {}

    def _new_session(self, params: Any) -> Message:
        request = NewSessionParams.model_validate(params)
        session = open_session(
            request.cwd,
            provider_factory=self._provider_factory,
            registry_factory=self._registry_factory,
        )
        with self._lock:
            accepted = not self._closed
            if accepted:
                self._sessions[session.session_id] = session
        if not accepted:  # ввод кончился, пока сессия собиралась: она уже никому не нужна
            session.close()
            raise RpcError(INTERNAL_ERROR, "сервер завершает работу")
        return {"sessionId": session.session_id}

    def _close_sessions(self) -> None:
        with self._lock:
            self._closed = True
            sessions = list(self._sessions.values())
            self._sessions.clear()
        with ExitStack() as stack:
            for session in sessions:
                stack.callback(session.close)
