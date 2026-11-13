"""Строка ввода REPL, которую прерывает завершение фоновой задачи.

Пока пользователь думает над следующим сообщением, фоновая задача агента может
завершиться — тогда агент должен продолжить работу сам, не дожидаясь Enter.
:class:`PromptReader` читает строку через prompt_toolkit, а :meth:`PromptReader.wake`
(из потока завершившейся задачи) прерывает её: вызов бросает
:class:`BackgroundWakeError` с набранным текстом, и REPL вернёт его в строку ввода после
хода агента.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Сессию строит cli.prompt; здесь она нужна только аннотациям.
    from prompt_toolkit import PromptSession

__all__ = ["BackgroundWakeError", "PromptReader"]


class BackgroundWakeError(Exception):
    """Ввод прерван: завершилась фоновая задача — агент продолжит работу сам.

    ``draft`` — набранный, но не отправленный текст: он вернётся в строку ввода.
    """

    def __init__(self, draft: str) -> None:
        """Ввод прерван с набранным текстом ``draft``."""
        super().__init__("завершилась фоновая задача")
        self.draft = draft


def _never() -> bool:
    return False


class PromptReader:
    """Строка ввода prompt_toolkit, которую можно прервать из другого потока.

    :meth:`wake` (из потока завершившейся фоновой задачи) прерывает открытую строку, если
    ``wake_due()`` — тогда вызов бросает :class:`BackgroundWakeError` с набранным текстом.
    ``wake_due`` проверяется и при каждом открытии строки: итог, пришедший между ходом и
    приглашением, не теряется.
    """

    def __init__(self, session: PromptSession[str]) -> None:
        """Строка ввода в сессии ``session``."""
        self._session = session
        # Пора ли прервать ввод: вызывается в потоке ввода (его задаёт REPL).
        self.wake_due: Callable[[], bool] = _never
        self._draft: str | None = None

    def __call__(self, default: str = "") -> str:
        """Строка; ``default`` — текст, которым заполнить ввод (набранный во время хода)."""
        self._draft = None
        text = self._session.prompt(default=default, pre_run=self._check)
        draft = self._draft
        if draft is not None:
            raise BackgroundWakeError(draft)
        return text

    def wake(self) -> None:
        """Завершилась фоновая задача (из любого потока): проверить, не прервать ли ввод."""
        app = self._session.app
        loop = app.loop
        if not app.is_running or loop is None:
            return
        with contextlib.suppress(RuntimeError):  # цикл строки ввода уже закрыт
            loop.call_soon_threadsafe(self._check)

    def _check(self) -> None:
        """В потоке ввода: прервать строку, если агенту пора продолжить работу."""
        app = self._session.app
        if self._draft is not None or not app.is_running or app.is_done:
            return
        if not self.wake_due():
            return
        # Черновик — из строки ввода, а не из текущего буфера: в поиске (Ctrl+R) тот —
        # строка поиска.
        self._draft = self._session.default_buffer.text
        app.exit(result="")
