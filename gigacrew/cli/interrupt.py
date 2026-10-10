"""Прерывание хода агента клавишей Esc.

Пока агент работает, строка ввода неактивна и терминал находится в обычном
(построчном) режиме — одиночный Esc программа не увидит. На время хода
:class:`EscInterrupt` переводит терминал в cbreak-режим (символы приходят сразу,
без эха; Ctrl+C по-прежнему шлёт SIGINT) и читает клавиши в фоновом потоке:

* одиночный Esc — прерывание, как Ctrl+C: основному потоку доставляется SIGINT,
  поэтому прерываются и ожидание сети, и запущенная команда (её группа процессов
  убивается тем же путём, что и при Ctrl+C);
* ESC-последовательности (стрелки, Alt+клавиша) отличаются от одиночного Esc по
  тому, что за ESC сразу приходят ещё байты, и игнорируются — кроме Shift+Tab
  (``ESC [ Z``): он переключает режим разрешений (:attr:`EscInterrupt.on_backtab`);
* остальной набранный текст копится и подставляется в следующую строку ввода —
  набирать следующий запрос можно, пока агент работает.

Перед вопросом пользователю (подтверждение операции) чтение приостанавливается
(:meth:`EscInterrupt.paused`) и терминал возвращается в обычный режим.

Работает, только если stdin — терминал; иначе — ничего не делает.
"""

from __future__ import annotations

import codecs
import contextlib
import os
import select
import signal
import sys
import threading
from collections.abc import Callable, Iterator
from typing import Any

_ESC = 0x1B
_ERASE = (0x7F, 0x08)  # Backspace
_SEQUENCE_GAP_SECONDS = 0.03  # байты ESC-последовательности приходят пачкой
_POLL_SECONDS = 0.1


class KeyParser:
    """Разбор байтов, прочитанных в cbreak-режиме: Esc отдельно, текст — в буфер.

    ``more_pending`` сообщает, пришли ли следующие байты сразу после ESC в конце
    пачки (то есть это начало последовательности, а не нажатый Esc).
    ``on_backtab`` — Shift+Tab (``ESC [ Z``).
    """

    def __init__(
        self,
        on_escape: Callable[[], None],
        more_pending: Callable[[], bool],
        on_backtab: Callable[[], None] | None = None,
    ):
        self._on_escape = on_escape
        self._more_pending = more_pending
        self._on_backtab = on_backtab
        self._csi_params = False  # в текущей CSI-последовательности были параметры
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")
        self._text: list[str] = []
        # Разбор ESC-последовательности: "" — обычный текст; "esc" — после ESC;
        # "csi" — ESC [ … до финального байта; "ss3" — ESC O + один байт;
        # "alt" — хвост многобайтного символа после ESC (Alt+буква).
        self._state = ""

    def feed(self, data: bytes) -> None:
        i = 0
        while i < len(data):
            byte = data[i]
            if byte == _ESC:
                if i + 1 < len(data) or self._more_pending():
                    self._state = "esc"  # стрелка, Alt+клавиша и т.п.
                else:
                    self._state = ""
                    self._on_escape()
                i += 1
                continue
            if self._state:
                if not self._skip_sequence_byte(byte):
                    continue  # байт не относится к последовательности — разобрать заново
                i += 1
                continue
            if byte in _ERASE:
                if self._text:
                    self._text.pop()
                i += 1
                continue
            j = i
            while j < len(data) and data[j] != _ESC and data[j] not in _ERASE:
                j += 1
            self._text.extend(self._decoder.decode(data[i:j]))
            i = j

    def _skip_sequence_byte(self, byte: int) -> bool:
        """Поглощает байт ESC-последовательности. False — последовательность кончилась
        раньше, и байт нужно обработать как обычный."""
        state = self._state
        if state == "esc":
            if byte == 0x5B:  # [
                self._state = "csi"
                self._csi_params = False
            elif byte == 0x4F:  # O
                self._state = "ss3"
            else:  # Alt+символ: сам символ отбрасываем целиком
                self._state = "alt" if byte >= 0xC0 else ""
            return True
        if state == "csi":
            if 0x40 <= byte <= 0x7E:  # финальный байт; параметры — 0x20–0x3F
                self._state = ""
                if byte == 0x5A and not self._csi_params and self._on_backtab is not None:
                    self._on_backtab()  # ESC [ Z — Shift+Tab
            else:
                self._csi_params = True
            return True
        if state == "ss3":
            self._state = ""
            return True
        # "alt": байты продолжения UTF-8
        if 0x80 <= byte <= 0xBF:
            return True
        self._state = ""
        return False

    def take_text(self) -> str:
        """Набранный текст (без завершающих переводов строк); буфер очищается."""
        text = "".join(self._text).replace("\r", "\n").rstrip("\n")
        self._text.clear()
        return "".join(ch for ch in text if ch in "\n\t" or ch.isprintable())


def _interrupt_main() -> None:
    """Прерывает основной поток так же, как Ctrl+C (в т.ч. блокирующий вызов)."""
    if hasattr(signal, "pthread_kill"):
        main = threading.main_thread().ident
        if main is not None:
            signal.pthread_kill(main, signal.SIGINT)
            return
    import _thread

    _thread.interrupt_main()


class EscInterrupt:
    """Контекст «агент работает»: Esc прерывает ход. Переиспользуется между ходами."""

    def __init__(
        self,
        *,
        fd: int | None = None,
        enabled: bool | None = None,
        interrupt: Callable[[], None] = _interrupt_main,
        on_backtab: Callable[[], object] | None = None,
    ):
        if fd is None:
            try:
                fd = sys.stdin.fileno()
            except (AttributeError, OSError, ValueError):
                fd = -1
        self._fd = fd
        if enabled is None:
            enabled = fd >= 0 and os.isatty(fd) and _supported()
        self.enabled = enabled
        self._interrupt = interrupt
        # Shift+Tab во время хода (вызывается из фонового потока чтения клавиш).
        self.on_backtab = on_backtab
        # Esc во время хода, до прерывания (из фонового потока): True — Esc забран
        # (например, останавливает суб-агента), прерывание остаётся взведённым —
        # следующий Esc тоже сработает.
        self.on_escape: Callable[[], bool] | None = None
        self._lock = threading.Lock()
        self._armed = False  # можно ли ещё прервать текущий ход
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._idle = threading.Event()  # поток не читает терминал
        self._thread: threading.Thread | None = None
        self._saved_mode: Any = None
        self._parser = KeyParser(self._on_escape, self._more_pending, self._on_backtab)

    # ------------------------------------------------------------------ #
    def __enter__(self) -> EscInterrupt:
        if not self.enabled:
            return self
        if self._thread is not None:
            self._shutdown()  # прошлый выход прервали — не оставляем второй поток чтения
        self._stop.clear()
        self._pause.clear()
        self._idle.clear()
        self._enter_raw()
        with self._lock:
            self._armed = True
        self._thread = threading.Thread(target=self._run, name="esc-interrupt", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if not self.enabled or self._thread is None:
            return
        try:
            with self._lock:
                self._armed = False  # после этого прерываний не будет
        finally:
            # Esc в самом конце хода: его SIGINT может прервать ожидание блокировки,
            # но поток чтения всё равно останавливается, а терминал — восстанавливается.
            self._shutdown()

    def _shutdown(self) -> None:
        self._armed = False
        self._stop.set()
        thread, self._thread = self._thread, None
        try:
            if thread is not None:
                thread.join(timeout=1)
        finally:
            self._restore_mode()
            self._saved_mode = None

    @contextlib.contextmanager
    def paused(self) -> Iterator[None]:
        """Временно вернуть терминал (для ``input()`` с вопросом пользователю)."""
        if not self.enabled or self._thread is None or self._pause.is_set():
            yield
            return
        self._idle.clear()
        self._pause.set()
        self._idle.wait(timeout=1)
        self._restore_mode()
        try:
            yield
        finally:
            self._enter_raw()
            self._pause.clear()

    def take_typeahead(self) -> str:
        """Текст, набранный во время хода (подставляется в следующий ввод)."""
        return self._parser.take_text()

    # ------------------------------------------------------------------ #
    def _on_escape(self) -> None:
        with self._lock:
            if not self._armed:
                return
            if self.on_escape is None or not self.on_escape():
                self._armed = False
            self._interrupt()

    def _on_backtab(self) -> None:
        if self.on_backtab is not None:
            self.on_backtab()

    def _more_pending(self) -> bool:
        return self._readable(_SEQUENCE_GAP_SECONDS)

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._pause.is_set():
                self._idle.set()
                self._stop.wait(_POLL_SECONDS / 2)
                continue
            if not self._readable(_POLL_SECONDS):
                continue
            data = self._read()
            if not data:
                if os.name == "posix":
                    break  # EOF
                continue  # Windows: прочитаны только стрелки/F-клавиши
            self._parser.feed(data)
        self._idle.set()

    # --------------------------- платформа ---------------------------- #
    def _enter_raw(self) -> None:
        if os.name == "posix":
            import termios
            import tty

            if self._saved_mode is None:
                self._saved_mode = termios.tcgetattr(self._fd)
            # TCSANOW: по умолчанию TCSAFLUSH выбросил бы уже набранные клавиши.
            tty.setcbreak(self._fd, termios.TCSANOW)

    def _restore_mode(self) -> None:
        if os.name == "posix" and self._saved_mode is not None:
            import termios

            with contextlib.suppress(termios.error, OSError):
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved_mode)

    def _readable(self, timeout: float) -> bool:
        if os.name == "posix":
            ready, _, _ = select.select([self._fd], [], [], timeout)
            return bool(ready)
        import msvcrt  # type: ignore[import-not-found]

        if msvcrt.kbhit():
            return True
        self._stop.wait(timeout)
        return bool(msvcrt.kbhit())

    def _read(self) -> bytes:
        if os.name == "posix":
            try:
                return os.read(self._fd, 1024)
            except OSError:
                return b""
        import msvcrt  # type: ignore[import-not-found]

        chars = []
        while msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):  # стрелки и F-клавиши: префикс + код
                if msvcrt.getwch() == "\x0f":  # Shift+Tab — как в POSIX-терминале
                    chars.append("\x1b[Z")
                continue
            chars.append(ch)
        return "".join(chars).encode("utf-8")


def _supported() -> bool:
    if os.name == "posix":
        try:
            import termios  # noqa: F401
            import tty  # noqa: F401
        except ImportError:
            return False
        return True
    try:
        import msvcrt  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True
