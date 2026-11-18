"""ACP-сервер на stdin/stdout процесса: подкоманда ``gigacrew acp``.

stdout процесса — только протокол: сервер пишет в дубликат дескриптора 1, а сам дескриптор 1
и ``sys.stdout`` до конца процесса ведут в stderr. Туда же уходят случайный ``print`` и вывод
дочерних процессов (хуков, команд), унаследовавших дескриптор, и журнал (``logging``).

Конец ввода — штатное завершение. SIGTERM, SIGHUP (выход :class:`SystemExit`) и SIGINT
прерывают чтение ввода: сессии закрываются, их ходы отменяются, и процессы ходов убиваются
группами. Перед выходом сервер ждёт свои рабочие потоки (ходы и обработчики запросов) — не
дольше :data:`SHUTDOWN_GRACE`, — чтобы отмена успела убить процессы.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from contextlib import ExitStack
from typing import BinaryIO, Final

from gigacrew.cli.acp.server import AcpServer
from gigacrew.cli.acp.session import ProviderFactory
from gigacrew.cli.interrupt import exit_on_signals

__all__ = ["SHUTDOWN_GRACE", "serve_stdio"]

# Сколько после остановки сервера ждать его рабочие потоки, с.
SHUTDOWN_GRACE: Final = 5.0

_STDOUT_FD: Final = 1


def serve_stdio(provider_factory: ProviderFactory) -> bool:
    """Обслуживает клиента ACP на stdin/stdout до конца ввода или сигнала завершения.

    ``provider_factory`` — провайдер LLM по конфигурации сессии. True — прервано SIGINT;
    SIGTERM и SIGHUP выходят из функции :class:`SystemExit` — уже после закрытия сессий.
    """
    protocol = _claim_stdout()
    logging.basicConfig(stream=sys.stderr, format="gigacrew: %(levelname)s: %(message)s")
    server = AcpServer(provider_factory=provider_factory)
    with ExitStack() as stack:
        stack.callback(_join_workers, SHUTDOWN_GRACE)
        stack.enter_context(exit_on_signals())
        try:
            server.serve(sys.stdin.buffer, protocol)
        except KeyboardInterrupt:
            return True
    return False


def _claim_stdout() -> BinaryIO:
    """Поток протокола — дубликат stdout; дескриптор 1 и ``sys.stdout`` — в stderr."""
    sys.stdout.flush()
    protocol = os.fdopen(os.dup(_STDOUT_FD), "wb")
    os.dup2(sys.stderr.fileno(), _STDOUT_FD)
    sys.stdout = sys.stderr
    return protocol


def _join_workers(grace: float) -> None:
    """Ждёт остальные потоки процесса, все вместе — не дольше ``grace`` секунд."""
    deadline = time.monotonic() + grace
    current = threading.current_thread()
    for thread in threading.enumerate():
        if thread is not current:
            thread.join(max(0, deadline - time.monotonic()))
