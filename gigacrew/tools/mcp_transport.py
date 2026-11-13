"""Транспорты Model Context Protocol: общий интерфейс и stdio (дочерний процесс).

Транспорт только доставляет сообщения JSON-RPC в обе стороны: входящие отдаёт функции
``receive``, которую передаёт :class:`~gigacrew.tools.mcp_client.McpClient`, а запросы,
ответы и таймауты — забота клиента.

* **stdio** — сервер запускается дочерним процессом; сообщения — строки JSON в его
  stdin и stdout. Читает и пишет отдельный поток каждый: сервер, который перестал
  читать, не вешает агента. Процесс — в своей группе: Ctrl+C в терминале его не
  убивает, а :meth:`StdioTransport.close` убивает вместе с потомками (``npx`` → ``node``).
  stderr сервера пишется в журнал.
* **Streamable HTTP** — :mod:`gigacrew.tools.mcp_http`.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import signal
import subprocess
import sys
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import IO, Any, override

__all__ = [
    "MAX_MESSAGE",
    "McpError",
    "McpTimeoutError",
    "Receiver",
    "StdioTransport",
    "Transport",
    "deliver",
    "encode_message",
    "ignore_message",
    "parse_message",
]

# Получатель входящего сообщения JSON-RPC (ответа, запроса или уведомления сервера).
Receiver = Callable[[dict[str, Any]], None]

MAX_MESSAGE = 67_108_864  # 64 МиБ — строка stdio или тело HTTP-ответа
_MAX_LOG = 1_048_576  # журнал stderr больше 1 МиБ при запуске начинается заново
_CLOSE_TIMEOUT = 2.0


class McpError(Exception):
    """Сбой обмена с MCP-сервером: ответ-ошибка, сервер завершился, нарушен протокол."""


class McpTimeoutError(McpError):
    """Сервер не ответил вовремя."""


class Transport(ABC):
    """Канал сообщений JSON-RPC к одному серверу."""

    # Согласованная версия протокола: HTTP передаёт её в заголовке каждого запроса.
    protocol_version: str = ""

    @property
    @abstractmethod
    def alive(self) -> bool:
        """Канал открыт и сервер не завершился."""

    @abstractmethod
    def open(self, receive: Receiver, gone: Callable[[], None]) -> None:
        """Открывает канал. ``gone`` — сервер завершился (stdio). Ошибка — OSError/McpError."""

    @abstractmethod
    def send(self, message: dict[str, Any], *, timeout: float) -> None:
        """Отправляет сообщение. HTTP заодно принимает ответ на него (через ``receive``)."""

    @abstractmethod
    def close(self) -> None:
        """Закрывает канал; повторный вызов безопасен."""


def encode_message(message: dict[str, Any]) -> bytes:
    """Сообщение JSON-RPC в UTF-8, одной строкой (без переводов строки внутри)."""
    return json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def deliver(payload: object, receive: Receiver) -> set[object]:
    """Сообщение или пакет сообщений — получателю; результат — id пришедших ответов."""
    messages = payload if isinstance(payload, list) else [payload]
    answered: set[object] = set()
    for message in messages:
        if not isinstance(message, dict):
            continue
        if "method" not in message:
            answered.add(message.get("id"))
        receive(message)
    return answered


def ignore_message(message: dict[str, Any]) -> None:
    """Получатель до :meth:`Transport.open`: сообщений ещё нет."""


def _nothing() -> None:
    """Обработчик завершения до :meth:`Transport.open`."""


# ---------------------------------------------------------------------------- #
# stdio


class StdioTransport(Transport):
    """Сервер — дочерний процесс; сообщения — строки JSON (см. описание модуля).

    Запуск и закрытие идут под одной блокировкой: закрытый канал не запускает процесс,
    даже если :meth:`close` пришёл из другого потока раньше :meth:`open`.
    """

    def __init__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        log_path: Path | None = None,
    ) -> None:
        """Запоминает команду запуска; процесс запускает :meth:`open`."""
        self._argv = list(argv)
        self._cwd = cwd
        self._env = dict(env)
        self._log_path = log_path
        self._proc: subprocess.Popen[bytes] | None = None
        self._outbox: queue.Queue[bytes | None] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._closed = False
        self._dead = threading.Event()
        self._receive: Receiver = ignore_message
        self._gone: Callable[[], None] = _nothing

    @property
    @override
    def alive(self) -> bool:
        """Процесс запущен и не завершился."""
        proc = self._proc
        return proc is not None and not self._dead.is_set() and proc.poll() is None

    @override
    def open(self, receive: Receiver, gone: Callable[[], None]) -> None:
        """Запускает процесс сервера и потоки чтения и записи.

        OSError или ValueError (недопустимое окружение) — не запустился; канал уже
        закрыт — McpError.
        """
        self._receive = receive
        self._gone = gone
        with self._lock:
            if self._closed:
                raise McpError("канал закрыт")
            proc = _spawn(self._argv, self._cwd, self._env, self._log_path)
            self._proc = proc
            for target, stream, name in (
                (self._read_loop, proc.stdout, "mcp-reader"),
                (self._write_loop, proc.stdin, "mcp-writer"),
            ):
                worker = threading.Thread(target=target, args=(stream,), name=name, daemon=True)
                worker.start()
                self._threads.append(worker)

    @override
    def send(self, message: dict[str, Any], *, timeout: float) -> None:
        """Ставит сообщение в очередь писателя — не блокирует, даже если сервер не читает."""
        if not self.alive:
            raise McpError("сервер завершился")
        self._outbox.put(b"".join((encode_message(message), b"\n")))

    @override
    def close(self) -> None:
        """Закрывает stdin (сигнал серверу завершиться), затем убивает группу процессов.

        Ctrl+C посреди ожидания не оставляет сервер сиротой: процесс всё равно убивается.
        """
        with self._lock:
            self._closed = True
            proc = self._proc
        if proc is None:
            return
        try:
            self._outbox.put(None)  # писатель закроет stdin
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=_CLOSE_TIMEOUT)
        except BaseException:
            _kill(proc)
            raise
        _kill(proc)
        self._release(proc)

    def _release(self, proc: subprocess.Popen[bytes]) -> None:
        """Дожидается потоков и закрывает свободный канал от убитого сервера."""
        for worker in self._threads:
            worker.join(timeout=_CLOSE_TIMEOUT)
        # Поток, всё ещё занятый каналом, держит его блокировку — такой не закрываем.
        if proc.stdout is not None and not any(thread.is_alive() for thread in self._threads):
            with contextlib.suppress(OSError):
                proc.stdout.close()
        self._dead.set()
        self._proc = None

    def _write_loop(self, stdin: IO[bytes]) -> None:
        # ValueError — запись в закрытый поток; сломанный канал — сервер завершился.
        with contextlib.suppress(OSError, ValueError):
            for chunk in iter(self._outbox.get, None):
                stdin.write(chunk)
                stdin.flush()
        with contextlib.suppress(OSError, ValueError):
            stdin.close()

    def _read_loop(self, stdout: IO[bytes]) -> None:
        # Любой сбой чтения или разбора (обрыв, строка без конца, JSON чрезмерной
        # вложенности) — как завершение сервера: ожидающие запросы получают отказ сразу.
        with contextlib.suppress(Exception):
            for line in _lines(stdout):
                deliver(parse_message(line), self._receive)
        self._dead.set()
        self._gone()


def _spawn(
    argv: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path | None,
) -> subprocess.Popen[bytes]:
    """Процесс сервера в своей группе процессов; stderr — в журнал ``log_path``."""
    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    with contextlib.ExitStack() as stack:
        # Свой файл журнала закрывается сразу после запуска: у процесса своя копия.
        stderr: IO[bytes] | int = subprocess.DEVNULL
        if log_path is not None:
            stderr = stack.enter_context(_open_log(log_path))
        return subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            **kwargs,
        )


def parse_message(line: bytes) -> object:
    """Строка stdout или данные события SSE → JSON; не JSON (отладочная печать) — None.

    JSON чрезмерной вложенности (RecursionError) — не пропускается: это нарушение
    протокола, вызывающий обрывает чтение.
    """
    try:
        return json.loads(line)
    except ValueError:
        return None


def _lines(stream: IO[bytes]) -> Iterator[bytes]:
    """Непустые строки потока; слишком длинная строка — ValueError (нарушен протокол)."""
    while True:
        line = stream.readline(MAX_MESSAGE + 1)
        if not line:
            return
        if len(line) > MAX_MESSAGE:
            raise ValueError("слишком длинное сообщение")
        if line.strip():
            yield line


def _open_log(path: Path) -> IO[bytes]:
    """Журнал stderr: дописывается, а выросший больше :data:`_MAX_LOG` — начинается заново."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        large = path.stat().st_size > _MAX_LOG
    except OSError:
        large = False
    return path.open("wb" if large else "ab")


def _kill(proc: subprocess.Popen[bytes]) -> None:
    """Убивает сервер вместе с потомками (своя группа процессов — см. open)."""
    try:
        if os.name == "posix":  # и если лидер уже вышел: потомки могли остаться
            os.killpg(proc.pid, signal.SIGKILL)
        elif proc.poll() is None:
            proc.kill()
    except OSError:
        proc.kill()
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=_CLOSE_TIMEOUT)
