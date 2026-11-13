"""Отмена фоновой работы: флаг отмены текущего потока.

Фоновый суб-агент работает в своём потоке, и все процессы, которые он запускает (команды
``run_shell``, git, хуки), должны умирать вместе с ним, когда задачу останавливают. Агент
задаёт флаг на время работы суб-агента (:func:`cancellation`), а
:func:`~gigacrew.tools.process.run_process` берёт его сам (:func:`current_cancel`): флаг
поднят — команда убивается сразу, как по таймауту. Срок работы потока
(:func:`time_left`) урезает таймаут команды ``run_shell``: команда фонового суб-агента не
переживает его предел времени, и после него остаётся время на отчёт.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Generator
from contextlib import contextmanager

__all__ = ["cancellation", "current_cancel", "time_left"]

_local = threading.local()


@contextmanager
def cancellation(event: threading.Event, *, deadline: float | None = None) -> Generator[None]:
    """Процессы, запущенные в этом потоке внутри блока, убиваются, когда ``event`` поднят.

    ``deadline`` (по ``time.monotonic()``) — срок работы потока (:func:`time_left`).
    """
    previous = current_cancel()
    previous_deadline: float | None = getattr(_local, "deadline", None)
    _local.event = event
    _local.deadline = deadline
    try:
        yield
    finally:
        _local.event = previous
        _local.deadline = previous_deadline


def current_cancel() -> threading.Event | None:
    """Флаг отмены текущего потока (None — работа не отменяется)."""
    event: threading.Event | None = getattr(_local, "event", None)
    return event


def time_left(timeout: float) -> float:
    """``timeout``, урезанный до срока работы текущего потока (срок вышел — 0)."""
    deadline: float | None = getattr(_local, "deadline", None)
    if deadline is None:
        return timeout
    return min(timeout, max(deadline - time.monotonic(), 0))
