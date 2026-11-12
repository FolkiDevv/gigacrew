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
    """Сервер — дочерний процесс; сообщения — строки JSON (см. описание модуля)."""

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
        """Запускает процесс сервера и потоки чтения и записи. OSError — не запустился."""
        self._receive = receive
        self._gone = gone
        kwargs: dict[str, Any] = {}
        if os.name == "posix":
            kwargs["start_new_session"] = True
        else:
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        with contextlib.ExitStack() as stack:
            # Свой файл журнала закрывается сразу после запуска: у процесса своя копия.
            stderr: IO[bytes] | int = subprocess.DEVNULL
            if self._log_path is not None:
                stderr = stack.enter_context(_open_log(self._log_path))
            self._proc = subprocess.Popen(
                self._argv,
                cwd=self._cwd,
                env=self._env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                **kwargs,
            )
        for target, name in ((self._read_loop, "mcp-reader"), (self._write_loop, "mcp-writer")):
            worker = threading.Thread(target=target, name=name, daemon=True)
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
        """Закрывает stdin (сигнал серверу завершиться), затем убивает группу процессов."""
        proc = self._proc
        if proc is None:
            return
        self._outbox.put(None)  # писатель закроет stdin
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=_CLOSE_TIMEOUT)
        _kill(proc)
        for worker in self._threads:
            worker.join(timeout=_CLOSE_TIMEOUT)
        # Поток, всё ещё занятый каналом, держит его блокировку — такой не закрываем.
        if proc.stdout is not None and not any(thread.is_alive() for thread in self._threads):
            with contextlib.suppress(OSError):
                proc.stdout.close()
        self._dead.set()
        self._proc = None

    def _write_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise AssertionError("писатель запущен без канала к серверу")
        stdin = proc.stdin
        # ValueError — запись в закрытый поток; сломанный канал — сервер завершился.
        with contextlib.suppress(OSError, ValueError):
            for chunk in iter(self._outbox.get, None):
                stdin.write(chunk)
                stdin.flush()
        with contextlib.suppress(OSError, ValueError):
            stdin.close()

    def _read_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise AssertionError("читатель запущен без канала от сервера")
        # Оборванный или испорченный поток — как завершение сервера.
        with contextlib.suppress(OSError, ValueError):
            for line in _lines(proc.stdout):
                deliver(parse_message(line), self._receive)
        self._dead.set()
        self._gone()


def parse_message(line: bytes) -> object:
    """Строка stdout или данные события SSE → JSON; не JSON (отладочная печать) — None."""
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
