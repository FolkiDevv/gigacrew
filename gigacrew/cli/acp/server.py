"""ACP-сервер gigacrew: обработчики методов ACP поверх соединения JSON-RPC.

Сессий в процессе может быть несколько, у каждой — свой каталог, конфигурация,
провайдер и MCP-серверы (:mod:`gigacrew.cli.acp.session`). Конец ввода закрывает все
сессии.
"""

from __future__ import annotations

import threading
from contextlib import ExitStack
from functools import partial
from importlib import metadata
from typing import Any, BinaryIO

from gigacrew.cli.acp.options import (
    SetConfigOptionParams,
    SetModelParams,
    SetModeParams,
    choose_model,
    config_options,
    session_setup,
)
from gigacrew.cli.acp.protocol import (
    CancelParams,
    InitializeParams,
    NewSessionParams,
    PromptParams,
    initialize_response,
    prompt_text,
)
from gigacrew.cli.acp.rpc import INTERNAL_ERROR, INVALID_PARAMS, RpcConnection, RpcError
from gigacrew.cli.acp.session import AcpSession, ProviderFactory, RegistryFactory, open_session
from gigacrew.tools.base import build_default_registry

__all__ = ["AcpServer"]

type Message = dict[str, Any]


class AcpServer:
    """Агент ACP: ``initialize``, ``authenticate``, сессии, их ходы, режим и модель.

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
        self._form_questions = False  # клиент объявил формы elicitation/create

    def serve(self, reader: BinaryIO, writer: BinaryIO) -> None:
        """Обслуживает клиента до конца ввода, затем закрывает все сессии."""
        connection = RpcConnection(reader, writer)
        with ExitStack() as stack:
            stack.callback(self._close_sessions)
            connection.serve(
                requests={
                    "initialize": self._initialize,
                    "authenticate": self._authenticate,
                    "session/new": partial(self._new_session, connection),
                    "session/prompt": self._prompt,
                    "session/set_mode": self._set_mode,
                    "session/set_config_option": self._set_config_option,
                    "session/set_model": self._set_model,
                },
                notifications={"session/cancel": self._cancel},
            )

    def session(self, session_id: str) -> AcpSession:
        """Открытая сессия по id; нет такой — ошибка JSON-RPC «Invalid params»."""
        with self._lock:
            found = self._sessions.get(session_id)
        if found is None:
            raise RpcError(INVALID_PARAMS, f"сессия не найдена: {session_id}")
        return found

    def _initialize(self, params: Any) -> Message:
        request = InitializeParams.model_validate(params)
        with self._lock:
            self._form_questions = request.form_questions
        return initialize_response(metadata.version("gigacrew"))

    def _authenticate(self, params: Any) -> Message:
        # Реквизиты — в окружении инстанса: проверять у клиента нечего.
        return {}

    def _new_session(self, connection: RpcConnection, params: Any) -> Message:
        request = NewSessionParams.model_validate(params)
        with self._lock:
            form_questions = self._form_questions
        session = open_session(
            request.cwd,
            connection=connection,
            form_questions=form_questions,
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
        # Сведения о сессии клиент принимает и до ответа на session/new (T3 их копит).
        session.turns.advertise_commands()
        return {"sessionId": session.session_id, **session_setup(session.agent)}

    def _prompt(self, params: Any) -> Message:
        request = PromptParams.model_validate(params)
        session = self.session(request.session_id)
        return {"stopReason": session.turns.prompt(prompt_text(request.prompt))}

    def _set_mode(self, params: Any) -> Message:
        request = SetModeParams.model_validate(params)
        self.session(request.session_id).agent.set_mode(request.mode_id)
        return {}

    def _set_config_option(self, params: Any) -> Message:
        request = SetConfigOptionParams.model_validate(params)
        agent = self.session(request.session_id).agent
        choose_model(agent, request.choice)
        return {"configOptions": config_options(agent)}

    def _set_model(self, params: Any) -> Message:
        request = SetModelParams.model_validate(params)
        choose_model(self.session(request.session_id).agent, request.model_id)
        return {}

    def _cancel(self, params: Any) -> None:
        request = CancelParams.model_validate(params)
        with self._lock:
            found = self._sessions.get(request.session_id)
        if found is not None:  # отмена неизвестной сессии — не ошибка: ответа на неё нет
            found.turns.cancel()

    def _close_sessions(self) -> None:
        with self._lock:
            self._closed = True
            sessions = list(self._sessions.values())
            self._sessions.clear()
        with ExitStack() as stack:
            for session in sessions:
                stack.callback(session.close)
