"""Безопасный запуск внешних процессов для инструментов (run_shell, git) и хуков.

Гарантии:
  * stdin — /dev/null: процесс не читает терминал пользователя и не зависает
    в ожидании ввода (``cat``, интерактивные утилиты, редактор git); хуку вместо
    него подаётся готовый текст (``stdin_text``) — из временного файла, не из канала;
  * окружение без настроек и секретов gigacrew (``GIGACREW_*``) и без интерактива git;
  * процесс и его потомки живут в отдельной группе и убиваются целиком — по
    таймауту, по Ctrl+C и если после выхода команды в группе остались фоновые
    потомки (``cmd &``), держат они вывод или нет; потомок, ушедший из группы
    (``setsid``), так не достать — это видно по ``escaped_output``;
  * вывод декодируется терпимо (бинарный мусор не роняет инструмент), а его
    объём в памяти ограничен;
  * тег заметок gigacrew (``<system-reminder>``) в выводе становится цитатой
    (:func:`quote_reminder_tags`) — вывод не выдаст себя за заметку harness'а.
"""

from __future__ import annotations

import codecs
import contextlib
import os
import re
import signal
import subprocess
import tempfile
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from gigacrew.security import subprocess_env

# Сколько ждать закрытия вывода после выхода команды, прежде чем считать,
# что его держат фоновые потомки.
_PIPE_GRACE_SECONDS = 1.0
# Сколько ждать, пока опустеет группа команды (зомби уже завершившихся потомков).
_GROUP_SETTLE_SECONDS = 0.2
# Пауза между проверками, опустела ли группа команды.
_GROUP_POLL_SECONDS = 0.01
# Сколько символов каждого потока держим в памяти (начало и конец).
_CAPTURE_LIMIT = 200_000
# Сколько байт канала читать за раз.
_READ_CHUNK_BYTES = 65536
# Состояния процесса в /proc/<pid>/stat, которые не считаются живыми: зомби и мёртвый.
_DEAD_STATES = frozenset((b"Z", b"X"))


# Тег заметок harness'а: системный промпт велит модели выполнять такие блоки и считать
# блок о хуках словами пользователя. Вывод команды (cat файла проекта, git log с
# подобранным сообщением коммита) мог бы его подделать — в выводе тег — цитата.
_REMINDER_TAG_RE = re.compile(r"<(\s*/?\s*)system[-_ ]?reminder", re.IGNORECASE)
QUOTED_REMINDER_TAG = "quoted-system-reminder"


def quote_reminder_tags(text: str) -> str:
    """``<system-reminder>`` → ``<quoted-system-reminder>`` (и закрывающий тег)."""
    return _REMINDER_TAG_RE.sub(rf"<\1{QUOTED_REMINDER_TAG}", text)


def truncate_middle(text: str, limit: int = 30_000, head: int = 10_000) -> str:
    """Обрезает середину длинного вывода: начало и (важнее) конец сохраняются.

    Итог тестов и ошибки компиляции обычно в конце вывода.
    """
    if len(text) <= limit:
        return text
    head = min(head, limit)
    tail = limit - head
    dropped = len(text) - head - tail
    head_text = text[:head]
    tail_text = text[len(text) - tail :]
    return f"{head_text}\n...(пропущено {dropped} символов)...\n{tail_text}"


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
    """Итог запуска процесса: код возврата, вывод и судьба фоновых потомков."""

    returncode: int | None  # None — процесс убит по таймауту
    stdout: str
    stderr: str
    timed_out: bool = False
    killed_background: bool = False  # после выхода остались фоновые потомки — убиты
    # Вывод держит процесс, ушедший из группы (setsid): остановить его не удалось.
    escaped_output: bool = False


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    """Убивает процесс вместе со всеми потомками."""
    with contextlib.suppress(OSError, subprocess.TimeoutExpired):
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
    if os.name != "posix":
        with contextlib.suppress(OSError):
            proc.kill()  # на случай, если taskkill недоступен


def _group_alive(proc: subprocess.Popen[bytes]) -> bool:
    """Остались ли живые процессы в группе команды (её фоновые потомки). Только POSIX.

    Сигнал 0 доходит и до зомби: уже завершившийся фоновый потомок (``(true &)``) ждёт,
    пока его уберёт init (а в контейнере init может не убирать вовсе), — поэтому, если
    есть /proc, зомби не считаются, а без него пустеющей группе даётся немного времени.
    """
    if os.name != "posix":
        return False
    deadline = time.monotonic() + _GROUP_SETTLE_SECONDS
    while True:
        try:
            os.killpg(proc.pid, 0)
        except PermissionError:
            return True
        except OSError:  # ProcessLookupError — группа пуста
            return False
        members = _live_members(proc.pid)
        if members is not None:
            return members
        if time.monotonic() >= deadline:
            return True
        time.sleep(_GROUP_POLL_SECONDS)


def _live_members(pgid: int) -> bool | None:
    """Есть ли в группе ``pgid`` процессы, кроме зомби (по /proc); None — /proc нет."""
    try:
        entries = [entry for entry in Path("/proc").iterdir() if entry.name.isdecimal()]
    except OSError:
        return None
    for entry in entries:
        try:
            stat = (entry / "stat").read_bytes()
            fields = stat.rsplit(b")", 1)[1].split()
        except (OSError, IndexError):
            continue  # процесс уже завершился
        # после имени: состояние, родитель, группа
        if len(fields) > 2 and int(fields[2]) == pgid and fields[0] not in _DEAD_STATES:
            return True
    return False


def _pump(stream: IO[bytes], capture: _Capture) -> None:
    """Читает канал по мере поступления данных (os.read не ждёт заполнения буфера).

    Поэтому вывод сохраняется, даже если канал так и не закрылся (его держит
    потомок, ушедший из группы процессов).
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    fd = stream.fileno()
    with contextlib.suppress(OSError, ValueError):
        while chunk := os.read(fd, _READ_CHUNK_BYTES):
            capture.add(decoder.decode(chunk))
    capture.add(decoder.decode(b"", final=True))


def _spawn(
    cmd: str | Sequence[str],
    *,
    cwd: Path,
    shell: bool,
    env: dict[str, str],
    stdin_text: str | None,
) -> subprocess.Popen[bytes]:
    """Запускает команду в своей группе процессов; stdin — ``stdin_text`` или /dev/null.

    Текст для stdin пишется во временный файл, который закрывается сразу после запуска:
    у процесса своя копия дескриптора.
    """
    kwargs: dict[str, Any] = {
        "shell": shell,
        "cwd": str(cwd),
        "env": env,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
    }
    if os.name == "posix":
        kwargs["start_new_session"] = True  # своя группа процессов — убиваем целиком
    else:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    if stdin_text is None:
        return subprocess.Popen(cmd, stdin=subprocess.DEVNULL, **kwargs)
    with tempfile.TemporaryFile() as stdin_file:
        stdin_file.write(stdin_text.encode("utf-8"))
        stdin_file.seek(0)
        return subprocess.Popen(cmd, stdin=stdin_file, **kwargs)


def _join_readers(readers: list[threading.Thread]) -> bool:
    """Даёт читателям дочитать вывод; True — кто-то из них всё ещё ждёт данных."""
    for reader in readers:
        reader.join(timeout=_PIPE_GRACE_SECONDS)
    return any(thread.is_alive() for thread in readers)


def _supervise(
    proc: subprocess.Popen[bytes],
    readers: list[threading.Thread],
    timeout: float,
) -> tuple[bool, bool, bool]:
    """Ждёт завершения команды и её вывода, убивает оставшихся в группе потомков.

    Возвращает флаги: таймаут, убиты фоновые потомки, вывод держит процесс вне группы.
    """
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_group(proc)
        proc.wait()
    # Команда завершилась, но вывод может держать фоновый потомок (`cmd &`).
    alive = _join_readers(readers)
    # Потомки в группе команды убиваются, даже если вывод они не держат
    # (`server >/dev/null 2>&1 &`): иначе пережили бы и команду, и gigacrew.
    killed_background = _group_alive(proc) or (alive and os.name != "posix")
    if killed_background:
        _kill_group(proc)
    escaped = False
    if alive:
        # Вывод всё ещё держит процесс вне группы (setsid) — его не достать.
        escaped = _join_readers(readers)
    return timed_out, killed_background, escaped


def _close_streams(proc: subprocess.Popen[bytes], readers: list[threading.Thread]) -> None:
    """Закрывает каналы вывода команды, кроме тех, на которых заблокирован читатель."""
    for stream_index, stream in enumerate((proc.stdout, proc.stderr)):
        reader = readers[stream_index] if stream_index < len(readers) else None
        # Если вывод держит потомок, сбежавший из группы (setsid), поток-читатель
        # всё ещё заблокирован в read(), и close() ждал бы ту же блокировку
        # бесконечно. Такой поток не закрываем: читатель — daemon, канал
        # закроется, когда потомок завершится.
        if reader is not None and reader.is_alive():
            continue
        with contextlib.suppress(OSError):
            stream.close()  # type: ignore[union-attr]


def run_process(
    cmd: str | Sequence[str],
    *,
    cwd: Path,
    timeout: float,
    shell: bool = False,
    environ: Mapping[str, str] | None = None,
    stdin_text: str | None = None,
    extra_env: Mapping[str, str] | None = None,
) -> ProcessResult:
    """Запускает команду и собирает вывод. FileNotFoundError — если нет программы.

    ``stdin_text`` — текст на stdin (UTF-8). Он пишется во временный файл, а не в канал:
    без потока-писателя, который мог бы заблокироваться на процессе, не читающем
    stdin. ``extra_env`` добавляется к окружению после фильтра секретов.
    """
    env = subprocess_env(environ)
    if extra_env:
        env.update(extra_env)
    proc = _spawn(cmd, cwd=cwd, shell=shell, env=env, stdin_text=stdin_text)
    out, err = _Capture(), _Capture()
    readers: list[threading.Thread] = []
    # Защищённый блок начинается сразу после Popen: прерывание в любой момент
    # (в том числе при запуске читателей) не оставит команду работать.
    try:
        readers = [
            threading.Thread(target=_pump, args=(proc.stdout, out), daemon=True),
            threading.Thread(target=_pump, args=(proc.stderr, err), daemon=True),
        ]
        for reader in readers:
            reader.start()
        timed_out, killed_background, escaped = _supervise(proc, readers, timeout)
    except BaseException:  # Ctrl+C или сбой запуска читателей
        _kill_group(proc)
        proc.wait()
        raise
    finally:
        _close_streams(proc, readers)

    return ProcessResult(
        returncode=None if timed_out else proc.returncode,
        stdout=quote_reminder_tags(out.text()),
        stderr=quote_reminder_tags(err.text()),
        timed_out=timed_out,
        killed_background=killed_background,
        escaped_output=escaped,
    )
