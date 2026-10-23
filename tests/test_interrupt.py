"""Esc во время хода: разбор клавиш, прерывание через настоящий псевдотерминал."""

from __future__ import annotations

import os
import pathlib
import sys
import threading
import time
from collections.abc import Iterator
from typing import NoReturn, cast

import pytest

from gigacrew.cli.interrupt import EscInterrupt, KeyParser


def _parser(*, pending: bool = False) -> tuple[KeyParser, list[int]]:
    hits: list[int] = []
    return KeyParser(lambda: hits.append(1), lambda: pending), hits


def test_lone_escape_interrupts() -> None:
    parser, hits = _parser()
    parser.feed(b"\x1b")
    assert hits == [1]


@pytest.mark.parametrize("keys", [b"\x1b[A", b"\x1b[1;5C", b"\x1bOP", b"\x1bx", b"\x1b[200~"])
def test_escape_sequences_are_not_escape(keys: bytes) -> None:
    parser, hits = _parser()
    parser.feed(keys)
    assert hits == []
    assert not parser.take_text()


def test_sequence_split_across_reads() -> None:
    parser, hits = _parser(pending=True)  # после ESC сразу пришли ещё байты
    parser.feed(b"\x1b")
    parser.feed(b"[Bok")
    assert hits == []
    assert parser.take_text() == "ok"


def test_double_escape_still_interrupts() -> None:
    parser, hits = _parser()
    parser.feed(b"\x1b\x1b")
    assert hits == [1]


def test_typeahead_text() -> None:
    parser, _ = _parser()
    text = "следующий вопрос".encode()
    parser.feed(text[:5])  # разрыв посреди UTF-8 символа
    parser.feed(b"".join((text[5:], b"X\x7f\x01\n")))
    assert parser.take_text() == "следующий вопрос"
    assert not parser.take_text()


# ----------------------------- псевдотерминал ----------------------------- #
pty_only = pytest.mark.skipif(sys.platform == "win32", reason="нужен pty")


@pytest.fixture
def pty_pair() -> Iterator[tuple[int, int]]:
    # pty есть только в Unix: импорт — внутри, чтобы модуль собирался и на Windows.
    import pty  # noqa: PLC0415

    master, slave = pty.openpty()
    yield master, slave
    os.close(master)
    os.close(slave)


def _echo_on(fd: int) -> bool:
    # termios есть только в Unix: импорт — внутри, чтобы модуль собирался и на Windows.
    import termios  # noqa: PLC0415

    return bool(termios.tcgetattr(fd)[3] & termios.ECHO)


def _wait_for_escape(esc: EscInterrupt, master: int, slave: int) -> None:
    """Нажимает Esc во время хода и ждёт, пока он прервёт блокирующий вызов."""
    with esc:
        assert not _echo_on(slave)  # cbreak, без эха
        os.write(master, b"\x1b")
        time.sleep(3)  # блокирующий вызов прерывается
        pytest.fail("Esc не прервал ожидание")


@pty_only
def test_escape_interrupts_main_thread_and_restores_terminal(pty_pair: tuple[int, int]) -> None:
    # termios есть только в Unix: импорт — внутри, чтобы модуль собирался и на Windows.
    import termios  # noqa: PLC0415

    master, slave = pty_pair
    before = termios.tcgetattr(slave)
    esc = EscInterrupt(fd=slave, enabled=True)
    with pytest.raises(KeyboardInterrupt):
        _wait_for_escape(esc, master, slave)
    assert termios.tcgetattr(slave) == before


@pty_only
def test_interrupted_exit_still_stops_reader_and_restores_terminal(
    pty_pair: tuple[int, int],
) -> None:
    """SIGINT от Esc в самом конце хода прерывает ожидание блокировки в __exit__.

    Поток чтения всё равно останавливается, терминал восстанавливается.
    """
    # termios есть только в Unix: импорт — внутри, чтобы модуль собирался и на Windows.
    import termios  # noqa: PLC0415

    _master, slave = pty_pair
    before = termios.tcgetattr(slave)
    esc = EscInterrupt(fd=slave, enabled=True, interrupt=lambda: None)

    class Interrupted:
        def __enter__(self) -> NoReturn:
            raise KeyboardInterrupt

        def __exit__(self, *exc: object) -> None:
            """Не вызывается: вход в блокировку уже прерван."""

    # Вход и выход — по отдельности: прерванным должен оказаться именно __exit__.
    esc.__enter__()  # noqa: PLC2801
    thread = esc._thread
    esc._lock = cast("threading.Lock", Interrupted())
    with pytest.raises(KeyboardInterrupt):
        esc.__exit__(None, None, None)
    assert esc._thread is None
    assert thread is not None
    assert not thread.is_alive()
    assert termios.tcgetattr(slave) == before


@pytest.mark.parametrize("keys", [b"\x1b[A", b"abc\n"])
@pty_only
def test_other_keys_do_not_interrupt(pty_pair: tuple[int, int], keys: bytes) -> None:
    master, slave = pty_pair
    with EscInterrupt(fd=slave, enabled=True):
        os.write(master, keys)
        time.sleep(0.3)


@pty_only
def test_typeahead_and_pause(pty_pair: tuple[int, int]) -> None:
    master, slave = pty_pair
    esc = EscInterrupt(fd=slave, enabled=True)
    with esc:
        os.write(master, "дальше\n".encode())
        time.sleep(0.3)
        with esc.paused():
            assert _echo_on(slave)  # для вопроса пользователю терминал обычный
        assert not _echo_on(slave)
    assert esc.take_typeahead() == "дальше"
    assert _echo_on(slave)


def test_disabled_without_terminal(tmp_path: pathlib.Path) -> None:
    with pathlib.Path(tmp_path / "f").open("w", encoding="utf-8") as stream:
        esc = EscInterrupt(fd=stream.fileno())
        assert esc.enabled is False
        with esc as entered, esc.paused():
            assert entered is esc
        assert not esc.take_typeahead()


@pytest.mark.parametrize(
    ("keys", "text"),
    [
        (b"\x1b1hello", "hello"),  # Alt+1
        (b"\x1bxhello", "hello"),  # Alt+x
        ("\x1bжhello".encode(), "hello"),  # Alt+многобайтная буква
        (b"\x1b[1;5Cok", "ok"),  # Ctrl+стрелка с параметрами
        (b"\x1bOPok", "ok"),  # F1
        (b"\x1b[200~paste\x1b[201~", "paste"),  # bracketed paste
    ],
)
def test_text_after_sequences_is_kept(keys: bytes, text: str) -> None:
    parser, hits = _parser()
    parser.feed(keys)
    assert hits == []
    assert parser.take_text() == text


@pty_only
def test_input_typed_before_enter_is_kept(pty_pair: tuple[int, int]) -> None:
    master, slave = pty_pair
    os.write(master, "уже набрано".encode())  # до перехода в cbreak
    time.sleep(0.1)
    esc = EscInterrupt(fd=slave, enabled=True)
    with esc:
        time.sleep(0.3)
    assert esc.take_typeahead() == "уже набрано"


# --------------------------- Shift+Tab во время хода --------------------------- #
def _backtab_parser(*, pending: bool = False) -> tuple[KeyParser, list[int], list[int]]:
    escapes: list[int] = []
    backtabs: list[int] = []
    parser = KeyParser(lambda: escapes.append(1), lambda: pending, lambda: backtabs.append(1))
    return parser, escapes, backtabs


def test_shift_tab_cycles_mode() -> None:
    parser, escapes, backtabs = _backtab_parser()
    parser.feed(b"ab\x1b[Zcd\x1b[Z")
    assert backtabs == [1, 1]
    assert escapes == []
    assert parser.take_text() == "abcd"


@pytest.mark.parametrize("keys", [b"\x1b[1;2Z", b"\x1b[A", b"\x1bZ", b"\x1bOZ", b"Z"])
def test_other_keys_are_not_shift_tab(keys: bytes) -> None:
    parser, _, backtabs = _backtab_parser()
    parser.feed(keys)
    assert backtabs == []


def test_shift_tab_split_across_reads() -> None:
    parser, escapes, backtabs = _backtab_parser(pending=True)
    parser.feed(b"\x1b")
    parser.feed(b"[")
    parser.feed(b"Z")
    assert backtabs == [1]
    assert escapes == []


@pty_only
def test_shift_tab_reaches_callback_without_interrupting(pty_pair: tuple[int, int]) -> None:
    master, slave = pty_pair
    hits: list[int] = []
    esc = EscInterrupt(fd=slave, enabled=True, on_backtab=lambda: hits.append(1))
    with esc:
        os.write(master, b"\x1b[Z")
        time.sleep(0.3)
    assert hits == [1]
    assert not esc.take_typeahead()


def test_on_escape_keeps_interrupt_armed_while_subagent_takes_it() -> None:
    hits: list[int] = []
    taken = [True, True, False]  # суб-агент: итог, обрыв; затем — ход целиком
    esc = EscInterrupt(fd=-1, enabled=False, interrupt=lambda: hits.append(1))
    esc.on_escape = lambda: taken.pop(0)
    esc._armed = True
    for _ in range(4):
        esc._on_escape()
    assert hits == [1, 1, 1]  # каждый Esc прерывает текущий запрос/команду
    assert not esc._armed  # после хода целиком — как раньше, один раз


def test_without_on_escape_single_interrupt() -> None:
    hits: list[int] = []
    esc = EscInterrupt(fd=-1, enabled=False, interrupt=lambda: hits.append(1))
    esc._armed = True
    esc._on_escape()
    esc._on_escape()
    assert hits == [1]
