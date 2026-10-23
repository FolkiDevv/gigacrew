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

# interrupt_main есть только в _thread: в threading нет способа прервать основной поток.
import _thread  # noqa: PLC2701
import codecs
import contextlib
import os
import select
import signal
import sys
import threading
from collections.abc import Callable, Generator
from typing import Any

_ESC = 0x1B
_ERASE = (0x7F, 0x8)  # Backspace
_SEQUENCE_GAP_SECONDS = 0.03  # байты ESC-последовательности приходят пачкой
_POLL_SECONDS = 0.1
_POSIX = "posix"  # os.name POSIX-систем: терминал — через termios, иначе — msvcrt
_READ_SIZE = 1024  # сколько байтов читать из терминала за раз

# Байты ESC-последовательностей.
_CSI_START = 0x5B  # «[»: ESC [ … — CSI-последовательность
_SS3_START = 0x4F  # «O»: ESC O + один байт (F1–F4 и др.)
_CSI_FINAL_MIN = 0x40  # финальный байт CSI — 0x40–0x7E; параметры — 0x20–0x3F
_CSI_FINAL_MAX = 0x7E
_BACKTAB_FINAL = 0x5A  # «Z»: ESC [ Z — Shift+Tab
_UTF8_LEAD_MIN = 0xC0  # первый байт многобайтного символа UTF-8
_UTF8_TAIL_MIN = 0x80  # байты продолжения UTF-8 — 0x80–0xBF
_UTF8_TAIL_MAX = 0xBF


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
    ) -> None:
        """Создать разбор с обработчиками Esc и Shift+Tab."""
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
        """Разобрать очередную пачку прочитанных байтов."""
        position = 0
        while position < len(data):
            position = self._feed_at(data, position)

    def take_text(self) -> str:
        """Набранный текст (без завершающих переводов строк); буфер очищается."""
        text = "".join(self._text).replace("\r", "\n").rstrip("\n")
        self._text.clear()
        return "".join(ch for ch in text if ch in "\n\t" or ch.isprintable())

    def _feed_at(self, chunk: bytes, position: int) -> int:
        """Разобрать байты пачки с позиции ``position``; вернуть позицию следующего."""
        byte = chunk[position]
        if byte == _ESC:
            if position + 1 < len(chunk) or self._more_pending():
                self._state = "esc"  # стрелка, Alt+клавиша и т.п.
            else:
                self._state = ""
                self._on_escape()
            return position + 1
        if self._state:
            if self._skip_sequence_byte(byte):
                return position + 1
            return position  # байт не относится к последовательности — разобрать заново
        if byte in _ERASE:
            if self._text:
                self._text.pop()
            return position + 1
        return self._feed_text(chunk, position)

    def _feed_text(self, chunk: bytes, start: int) -> int:
        """Добавить в буфер текст до ближайшего ESC или Backspace; вернуть его конец."""
        end = start
        while end < len(chunk) and chunk[end] != _ESC and chunk[end] not in _ERASE:
            end += 1
        self._text.extend(self._decoder.decode(chunk[start:end]))
        return end

    def _skip_sequence_byte(self, byte: int) -> bool:
        """Поглощает байт ESC-последовательности.

        False — последовательность кончилась раньше, и байт нужно обработать как обычный.
        """
        state = self._state
        if state == "esc":
            self._skip_after_escape(byte)
            return True
        if state == "csi":
            self._skip_csi_byte(byte)
            return True
        if state == "ss3":
            self._state = ""
            return True
        # "alt": байты продолжения UTF-8
        if _UTF8_TAIL_MIN <= byte <= _UTF8_TAIL_MAX:
            return True
        self._state = ""
        return False

    def _skip_after_escape(self, byte: int) -> None:
        """Первый байт после ESC: вид последовательности."""
        if byte == _CSI_START:
            self._state = "csi"
            self._csi_params = False
        elif byte == _SS3_START:
            self._state = "ss3"
        else:  # Alt+символ: сам символ отбрасываем целиком
            self._state = "alt" if byte >= _UTF8_LEAD_MIN else ""

    def _skip_csi_byte(self, byte: int) -> None:
        """Байт CSI-последовательности: параметр или финальный байт."""
        if _CSI_FINAL_MIN <= byte <= _CSI_FINAL_MAX:
            self._state = ""
            if byte == _BACKTAB_FINAL and not self._csi_params and self._on_backtab is not None:
                self._on_backtab()  # ESC [ Z — Shift+Tab
        else:
            self._csi_params = True


def _interrupt_main() -> None:
    """Прерывает основной поток так же, как Ctrl+C (в т.ч. блокирующий вызов)."""
    if hasattr(signal, "pthread_kill"):
        main = threading.main_thread().ident
        if main is not None:
            signal.pthread_kill(main, signal.SIGINT)
            return
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
    ) -> None:
        """Подготовить чтение клавиш из ``fd`` (по умолчанию — stdin).

        ``enabled`` None — включено, если ``fd`` — терминал и платформа поддерживается.
        """
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
        """Начало хода: терминал — в cbreak-режим, клавиши читает фоновый поток."""
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
        """Конец хода: прерываний больше нет, поток чтения остановлен, терминал прежний."""
        if not self.enabled or self._thread is None:
            return
        try:
            with self._lock:
                self._armed = False  # после этого прерываний не будет
        except BaseException:
            # Esc в самом конце хода: его SIGINT может прервать ожидание блокировки,
            # но поток чтения всё равно останавливается, а терминал — восстанавливается.
            self._shutdown()
            raise
        self._shutdown()

    @contextlib.contextmanager
    def paused(self) -> Generator[None]:
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
    def _shutdown(self) -> None:
        self._armed = False
        self._stop.set()
        thread = self._thread
        self._thread = None
        try:
            if thread is not None:
                thread.join(timeout=1)
        except BaseException:
            self._release_terminal()  # и при прерванном ожидании потока
            raise
        self._release_terminal()

    def _release_terminal(self) -> None:
        """Вернуть терминал в сохранённый режим и забыть его."""
        self._restore_mode()
        self._saved_mode = None

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
            chunk = self._read()
            if not chunk:
                if os.name == _POSIX:
                    break  # EOF
                continue  # Windows: прочитаны только стрелки/F-клавиши
            self._parser.feed(chunk)
        self._idle.set()

    # --------------------------- платформа ---------------------------- #
    def _enter_raw(self) -> None:
        if os.name == _POSIX:
            # termios и tty есть только в POSIX: импорт — на этой ветке.
            import termios  # noqa: PLC0415
            import tty  # noqa: PLC0415

            if self._saved_mode is None:
                self._saved_mode = termios.tcgetattr(self._fd)
            # TCSANOW: по умолчанию TCSAFLUSH выбросил бы уже набранные клавиши.
            tty.setcbreak(self._fd, termios.TCSANOW)

    def _restore_mode(self) -> None:
        if os.name == _POSIX and self._saved_mode is not None:
            # termios есть только в POSIX: импорт — на этой ветке.
            import termios  # noqa: PLC0415

            with contextlib.suppress(termios.error, OSError):
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved_mode)

    def _readable(self, timeout: float) -> bool:
        if os.name == _POSIX:
            ready, _, _ = select.select([self._fd], [], [], timeout)
            return bool(ready)
        # msvcrt есть только в Windows: импорт — на этой ветке.
        import msvcrt  # noqa: PLC0415

        if msvcrt.kbhit():  # type: ignore[attr-defined]
            return True
        self._stop.wait(timeout)
        return bool(msvcrt.kbhit())  # type: ignore[attr-defined]

    def _read(self) -> bytes:
        if os.name == _POSIX:
            try:
                return os.read(self._fd, _READ_SIZE)
            except OSError:
                return b""
        # msvcrt есть только в Windows: импорт — на этой ветке.
        import msvcrt  # noqa: PLC0415

        chars: list[str] = []
        while msvcrt.kbhit():  # type: ignore[attr-defined]
            ch = msvcrt.getwch()  # type: ignore[attr-defined]
            if ch in {"\x00", "\xe0"}:  # стрелки и F-клавиши: префикс + код
                # Shift+Tab — как в POSIX-терминале
                if msvcrt.getwch() == "\x0f":  # type: ignore[attr-defined]
                    chars.append("\x1b[Z")
                continue
            chars.append(ch)
        return "".join(chars).encode("utf-8")


def _supported() -> bool:
    if os.name == _POSIX:
        # Модули терминала есть не во всех сборках: проверяем, что импорт удаётся.
        try:
            import termios  # noqa: F401, PLC0415
            import tty  # noqa: F401, PLC0415
        except ImportError:
            return False
        return True
    # msvcrt есть только в Windows: проверяем, что импорт удаётся.
    try:
        import msvcrt  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True
