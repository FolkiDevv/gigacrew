"""ACP-сервер gigacrew на паре каналов: запуск в фоне, вызовы методов, подмена провайдера.

Сервер (:class:`~gigacrew.cli.acp.server.AcpServer`) работает в фоновом потоке поверх
каналов :class:`acp_client.AcpClient`; провайдеры для сессий выдаёт :class:`ProviderFactory`
(по умолчанию — ``ScriptedProvider`` без ходов). Окружение инстанса, как его задаёт T3, —
:func:`acp_environment`. Каждый тестовый модуль ACP объявляет::

    pytestmark = pytest.mark.filterwarnings(
        "error::pytest.PytestUnhandledThreadExceptionWarning",
    )

— исключение в потоке сервера роняет тест, а не теряется. Типичная фикстура::

    @pytest.fixture
    def acp(monkeypatch, factory, tmp_path_factory) -> Iterator[AcpHarness]:
        acp_environment(monkeypatch, tmp_path_factory.mktemp("process-cwd"))
        with closing(AcpHarness(factory)) as harness:
            yield harness
"""

from __future__ import annotations

import itertools
import os
import threading
from collections.abc import Callable, Iterable
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from acp_client import TIMEOUT, AcpClient
from fakes import ScriptedProvider

from gigacrew.cli.acp.server import AcpServer
from gigacrew.config import Config
from gigacrew.llm.base import LLMProvider
from gigacrew.tools.base import ToolRegistry

__all__ = ["PROTOCOL_VERSION", "AcpHarness", "ProviderFactory", "acp_environment"]

# Версия ACP, которую понимают T3 и gigacrew.
PROTOCOL_VERSION = 1

type Message = dict[str, Any]


def acp_environment(monkeypatch: pytest.MonkeyPatch, workdir: Path) -> None:
    """Окружение инстанса: реквизиты API v2 и модель ``M-default``, настроек пользователя нет.

    Любые GIGACREW_*/GIGACODE_* из окружения разработчика убраны: тест от них не зависит.
    Текущий каталог процесса — пустой ``workdir``: его ``.env`` не влияет на сессии.
    """
    monkeypatch.chdir(workdir)
    user_settings = [
        env_name for env_name in os.environ if env_name.startswith(("GIGACREW_", "GIGACODE_"))
    ]
    for name in user_settings:
        monkeypatch.delenv(name)
    monkeypatch.setenv("GIGACREW_ACCESS_KEY", "dummy")
    monkeypatch.setenv("GIGACREW_AUTH_URL", "https://auth.test/oauth")
    monkeypatch.setenv("GIGACREW_API_URL", "https://api.test/v1")
    monkeypatch.setenv("GIGACREW_MODEL", "M-default")


class ProviderFactory:
    """Подменная фабрика провайдеров сервера: ``config`` → провайдер новой сессии.

    Выдаёт добавленные (:meth:`add`) провайдеры по очереди, затем — ``ScriptedProvider()``;
    добавленное исключение бросается (провайдер не создан). Конфигурации всех вызовов —
    в ``configs``, выданные провайдеры — в ``issued``.
    """

    def __init__(self) -> None:
        self._queue: list[LLMProvider | Exception] = []
        self.configs: list[Config] = []
        self.issued: list[LLMProvider] = []

    def __call__(self, config: Config) -> LLMProvider:
        """Провайдер для сессии с конфигурацией ``config``."""
        self.configs.append(config)
        upcoming = self._queue.pop(0) if self._queue else ScriptedProvider()
        if isinstance(upcoming, Exception):
            raise upcoming
        self.issued.append(upcoming)
        return upcoming

    def add(self, *providers: LLMProvider | Exception) -> None:
        """Провайдеры (или ошибки создания) для следующих сессий, по порядку."""
        self._queue.extend(providers)


class AcpHarness:
    """ACP-сервер в фоновом потоке и клиент к нему.

    ``registry_factory`` — инструменты агента каждой сессии (None — по умолчанию сервера).
    ``stop`` — конец ввода: сервер обязан завершить ``serve`` (и закрыть сессии); ``close``
    ещё и закрывает каналы.
    """

    def __init__(
        self,
        factory: ProviderFactory,
        *,
        registry_factory: Callable[[], ToolRegistry] | None = None,
    ) -> None:
        self.factory = factory
        self.client = AcpClient()
        if registry_factory is None:
            self.server = AcpServer(provider_factory=factory)
        else:
            self.server = AcpServer(provider_factory=factory, registry_factory=registry_factory)
        self._ids = itertools.count(1)
        self._thread = threading.Thread(
            target=self.server.serve,
            args=(self.client.server_input, self.client.server_output),
            daemon=True,
        )
        self._thread.start()

    def begin(self, method: str, params: Any = None) -> int:
        """Отправляет запрос, не дожидаясь ответа; id запроса (для :meth:`reply`)."""
        msg_id = next(self._ids)
        self.client.request(msg_id, method, params)
        return msg_id

    def reply(self, msg_id: int) -> Message:
        """Ответ на запрос ``msg_id``; встреченные по пути сообщения — в ``client.backlog``."""
        return self.client.response(msg_id)

    def receive(self) -> Message:
        """Следующее сообщение сервера (ответ, уведомление или запрос к клиенту)."""
        return self.client.receive()

    def send(self, method: str, params: Any = None) -> Message:
        """Запрос и ответ сервера целиком (``result`` или ``error``)."""
        return self.reply(self.begin(method, params))

    def call(self, method: str, params: Any = None) -> Any:
        """Результат запроса; ошибка JSON-RPC роняет тест."""
        reply = self.send(method, params)
        error = reply.get("error")
        if error is not None:
            pytest.fail(f"{method}: ошибка {error}")
        return reply["result"]

    def call_error(self, method: str, params: Any = None) -> Message:
        """Ошибка JSON-RPC запроса (объект ``error``); успешный ответ роняет тест."""
        reply = self.send(method, params)
        assert "error" in reply, f"{method}: ожидалась ошибка, а ответ {reply}"
        error: Message = reply["error"]
        return error

    def initialize(self, client_capabilities: Message | None = None) -> Any:
        """``initialize`` как у T3: версия протокола, возможности и сведения о клиенте."""
        return self.call(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "clientCapabilities": client_capabilities or {},
                "clientInfo": {"name": "t3-test", "version": "0.0.0"},
            },
        )

    def new_session(self, cwd: Path, mcp_servers: Iterable[Message] = ()) -> str:
        """``session/new`` в каталоге ``cwd``; id новой сессии."""
        response = self.call("session/new", {"cwd": str(cwd), "mcpServers": list(mcp_servers)})
        session_id = response["sessionId"]
        assert isinstance(session_id, str)
        return session_id

    def stop(self) -> None:
        """Конец ввода сервера; ``serve`` обязан вернуться."""
        self.client.close_input()
        self._thread.join(TIMEOUT)
        assert not self._thread.is_alive(), "сервер не завершился после конца ввода"

    def close(self) -> None:
        """Останавливает сервер и закрывает каналы (даже если сервер не остановился)."""
        with closing(self.client):
            self.stop()
