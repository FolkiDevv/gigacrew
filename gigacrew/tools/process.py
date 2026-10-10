"""Безопасный запуск внешних процессов для инструментов (run_shell, git).

Гарантии:
  * stdin — /dev/null: процесс не читает терминал пользователя и не зависает
    в ожидании ввода (``cat``, интерактивные утилиты, редактор git);
  * окружение без настроек и секретов gigacrew (``GIGACREW_*``) и без интерактива git;
  * процесс и его потомки живут в отдельной группе и убиваются целиком — по
    таймауту, по Ctrl+C и если после выхода команды фоновые потомки (``cmd &``)
    продолжают держать вывод;
  * вывод декодируется терпимо (бинарный мусор не роняет инструмент), а его
    объём в памяти ограничен.
"""

from __future__ import annotations

import codecs
import os
import signal
import subprocess
import threading
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

# Переменные с этими префиксами не передаются дочерним процессам.
SECRET_ENV_PREFIXES = ("GIGACREW_",)

# Сколько ждать закрытия вывода после выхода команды, прежде чем считать,
# что его держат фоновые потомки.
_PIPE_GRACE_SECONDS = 1.0
# Сколько символов каждого потока держим в памяти (начало и конец).
_CAPTURE_LIMIT = 200_000


def subprocess_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Окружение для дочернего процесса: без секретов и без интерактива."""
    source = os.environ if environ is None else environ
    env = {k: v for k, v in source.items() if not k.upper().startswith(SECRET_ENV_PREFIXES)}
    env["GIT_TERMINAL_PROMPT"] = "0"  # не спрашивать логин/пароль
    env["GIT_EDITOR"] = "true"  # не открывать редактор (commit без -m)
    env["GIT_PAGER"] = "cat"
    env["PAGER"] = "cat"
    return env


def truncate_middle(text: str, limit: int = 30_000, head: int = 10_000) -> str:
    """Обрезает середину длинного вывода: начало и (важнее) конец сохраняются.

    Итог тестов и ошибки компиляции обычно в конце вывода.
    """
    if len(text) <= limit:
        return text
    head = min(head, limit)
    tail = limit - head
    dropped = len(text) - head - tail
    return f"{text[:head]}\n...(пропущено {dropped} символов)...\n{text[len(text) - tail :]}"


class _Capture:
    """Накопитель вывода с ограничением памяти: первые и последние ``limit`` символов."""

    def __init__(self, limit: int = _CAPTURE_LIMIT):
        self._limit = limit
        self._head: list[str] = []
        self._head_size = 0
        self._tail: deque[str] = deque()
        self._tail_size = 0
        self._dropped = 0

    def add(self, chunk: str) -> None:
        if self._head_size < self._limit:
            take = chunk[: self._limit - self._head_size]
            self._head.append(take)
            self._head_size += len(take)
            chunk = chunk[len(take) :]
        if not chunk:
            return
        self._tail.append(chunk)
        self._tail_size += len(chunk)
        while self._tail_size - len(self._tail[0]) >= self._limit:
            self._dropped += len(self._tail[0])
            self._tail_size -= len(self._tail.popleft())

    def text(self) -> str:
        head = "".join(self._head)
        tail = "".join(self._tail)
        if self._dropped:
            return f"{head}\n...(пропущено {self._dropped} символов)...\n{tail}"
        return head + tail


@dataclass(frozen=True)
class ProcessResult:
    returncode: int | None  # None — процесс убит по таймауту
    stdout: str
    stderr: str
    timed_out: bool = False
    killed_background: bool = False  # после выхода остались фоновые потомки


def _kill_group(proc: subprocess.Popen) -> None:
    """Убивает процесс вместе со всеми потомками."""
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
            return
        # Windows: proc.kill() завершает только оболочку, потомки остались бы
        # жить. taskkill /T убивает всё дерево процессов.
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (ProcessLookupError, PermissionError, OSError, subprocess.TimeoutExpired):
        pass
    if os.name != "posix":
        try:
            proc.kill()  # на случай, если taskkill недоступен
        except OSError:
            pass


def _pump(stream, capture: _Capture) -> None:
    """Читает канал по мере поступления данных (os.read не ждёт заполнения буфера).

    Поэтому вывод сохраняется, даже если канал так и не закрылся (его держит
    потомок, ушедший из группы процессов).
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    fd = stream.fileno()
    try:
        while chunk := os.read(fd, 65536):
            capture.add(decoder.decode(chunk))
    except (OSError, ValueError):
        pass
    capture.add(decoder.decode(b"", final=True))


def run_process(
    cmd: str | Sequence[str],
    *,
    cwd: Path,
    timeout: float,
    shell: bool = False,
    environ: Mapping[str, str] | None = None,
) -> ProcessResult:
    """Запускает команду и собирает вывод. FileNotFoundError — если нет программы."""
    kwargs: dict = {}
    if os.name == "posix":
        kwargs["start_new_session"] = True  # своя группа процессов — убиваем целиком
    else:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    proc = subprocess.Popen(
        cmd,
        shell=shell,
        cwd=str(cwd),
        env=subprocess_env(environ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **kwargs,
    )
    out, err = _Capture(), _Capture()
    readers: list[threading.Thread] = []
    timed_out = killed_background = False
    # Защищённый блок начинается сразу после Popen: прерывание в любой момент
    # (в том числе при запуске читателей) не оставит команду работать.
    try:
        readers = [
            threading.Thread(target=_pump, args=(proc.stdout, out), daemon=True),
            threading.Thread(target=_pump, args=(proc.stderr, err), daemon=True),
        ]
        for t in readers:
            t.start()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_group(proc)
            proc.wait()
        # Команда завершилась, но вывод может держать фоновый потомок (`cmd &`).
        for t in readers:
            t.join(timeout=_PIPE_GRACE_SECONDS)
        if any(t.is_alive() for t in readers):
            killed_background = True
            _kill_group(proc)
            for t in readers:
                t.join(timeout=_PIPE_GRACE_SECONDS)
    except BaseException:  # Ctrl+C или сбой запуска читателей
        _kill_group(proc)
        proc.wait()
        raise
    finally:
        for i, stream in enumerate((proc.stdout, proc.stderr)):
            reader = readers[i] if i < len(readers) else None
            # Если вывод держит потомок, сбежавший из группы (setsid), поток-читатель
            # всё ещё заблокирован в read(), и close() ждал бы ту же блокировку
            # бесконечно. Такой поток не закрываем: читатель — daemon, канал
            # закроется, когда потомок завершится.
            if reader is not None and reader.is_alive():
                continue
            try:
                stream.close()  # type: ignore[union-attr]
            except OSError:
                pass

    return ProcessResult(
        returncode=None if timed_out else proc.returncode,
        stdout=out.text(),
        stderr=err.text(),
        timed_out=timed_out,
        killed_background=killed_background,
    )
