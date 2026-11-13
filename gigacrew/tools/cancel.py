"""Отмена фоновой работы: флаг отмены текущего потока.

Фоновый суб-агент работает в своём потоке, и все процессы, которые он запускает (команды
``run_shell``, git, хуки), должны умирать вместе с ним, когда задачу останавливают. Агент
задаёт флаг на время работы суб-агента (:func:`cancellation`), а
:func:`~gigacrew.tools.process.run_process` берёт его сам (:func:`current_cancel`): флаг
поднят — команда убивается сразу, как по таймауту.
"""

from __future__ import annotations

import threading
from collections.abc import Generator
from contextlib import contextmanager

__all__ = ["cancellation", "current_cancel"]

_local = threading.local()


@contextmanager
def cancellation(event: threading.Event) -> Generator[None]:
    """Процессы, запущенные в этом потоке внутри блока, убиваются, когда ``event`` поднят."""
    previous = current_cancel()
    _local.event = event
    try:
        yield
    finally:
        _local.event = previous


def current_cancel() -> threading.Event | None:
    """Флаг отмены текущего потока (None — работа не отменяется)."""
    event: threading.Event | None = getattr(_local, "event", None)
    return event
