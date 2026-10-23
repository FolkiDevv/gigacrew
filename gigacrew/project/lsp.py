"""Минимальный синхронный клиент Language Server Protocol (JSON-RPC поверх stdio).

Ровно столько, сколько нужно для запросов навигации к одному серверу: запрос с
таймаутом, уведомление, ответы на запросы сервера. Поток-читатель разбирает
сообщения и раскладывает ответы по ожидающим запросам; уведомления сервера
(диагностика, прогресс) игнорируются. Пишет в сервер отдельный поток: если сервер
перестал читать и канал переполнен, запрос всё равно завершится по таймауту, а
:meth:`LspClient.close` — убьёт процесс.

Сервер запускается в своей группе процессов: Ctrl+C в терминале (SIGINT всей
группе переднего плана) его не убивает — прерывается только ожидание ответа.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import signal
import subprocess
import threading
from collections.abc import Callable, Generator, Mapping, Sequence
from pathlib import Path
from typing import IO, Any

# Обработчик запроса сервера: params -> result.
ServerRequestHandler = Callable[[Any], Any]

# Место для ответа на запрос: ответ сервера или None — сигнал «сервер завершился».
_Slot = queue.Queue[dict[str, Any] | None]

_MAX_HEADER_LINE = 8192
_MAX_MESSAGE = 67_108_864  # 64 МиБ
_INTERNAL_ERROR = -32603  # код JSON-RPC «внутренняя ошибка»


class LspError(Exception):
    """Сбой обмена с сервером: ответ-ошибка, сервер завершился, нарушен протокол."""


class LspTimeoutError(LspError):
    """Сервер не ответил вовремя."""


class LspClient:
    """Клиент одного сервера. Конструктор ничего не запускает — см. :meth:`start`."""

    def __init__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None = None,
        stderr_path: Path | None = None,
        handlers: Mapping[str, ServerRequestHandler] | None = None,
    ) -> None:
        """Запоминает команду запуска сервера и обработчики его запросов."""
        self._argv = list(argv)
        self._cwd = cwd
        self._env = None if env is None else dict(env)
        self._stderr_path = stderr_path
        self._handlers = dict(handlers or {})
        self._proc: subprocess.Popen[bytes] | None = None
        self._outbox: queue.Queue[bytes | None] = queue.Queue()
        self._writer: threading.Thread | None = None
        self._reader: threading.Thread | None = None
        self._pending: dict[int, _Slot] = {}
        self._pending_lock = threading.Lock()
        self._next_id = 0
        self._dead = threading.Event()

    # ------------------------------------------------------------------ #
    @property
    def alive(self) -> bool:
        """Сервер запущен и не завершился."""
        return self._proc is not None and not self._dead.is_set() and self._proc.poll() is None

    def start(self) -> None:
        """Запускает процесс сервера. OSError — программа не запустилась."""
        kwargs: dict[str, Any] = {}
        if os.name == "posix":
            kwargs["start_new_session"] = True
        else:
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        with contextlib.ExitStack() as stack:
            # свой файл журнала закрывается сразу после запуска: у процесса своя копия дескриптора
            stderr: IO[bytes] | int = (
                stack.enter_context(Path(self._stderr_path).open("ab"))
                if self._stderr_path
                else subprocess.DEVNULL
            )
            self._proc = subprocess.Popen(
                self._argv,
                cwd=self._cwd,
                env=self._env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                **kwargs,
            )
        self._reader = threading.Thread(target=self._read_loop, name="lsp-reader", daemon=True)
        self._reader.start()
        self._writer = threading.Thread(target=self._write_loop, name="lsp-writer", daemon=True)
        self._writer.start()

    def request(self, method: str, params: Any = None, *, timeout: float) -> Any:
        """Запрос и ожидание ответа. LspError — ошибка сервера, LspTimeoutError — нет ответа."""
        with self._pending_slot() as (msg_id, slot):
            self._send(_jsonrpc(id=msg_id, method=method, params=params))
            try:
                reply = slot.get(timeout=timeout)
            except queue.Empty:
                self._cancel(msg_id)
                waited = format(timeout, "g")
                raise LspTimeoutError(f"{method}: сервер не ответил за {waited} с") from None
            except BaseException:  # Ctrl+C посреди ожидания: ответ больше не нужен
                self._cancel(msg_id)
                raise
        if reply is not None and "error" not in reply:
            return reply.get("result")
        reason = _failure_reason(reply)
        raise LspError(f"{method}: {reason}")

    def notify(self, method: str, params: Any = None) -> None:
        """Уведомление серверу, без ответа. LspError — сервер не запущен."""
        self._send(_jsonrpc(method=method, params=params))

    def close(self, timeout: float = 2.0) -> None:
        """Вежливое завершение (shutdown/exit), при неудаче — kill всей группы процессов.

        Не зависает: и при Ctrl+C посреди завершения (процесс всё равно убивается,
        прерывание пробрасывается), и когда сервер не читает или потомки держат
        канал. Повторный вызов безопасен.
        """
        proc = self._proc
        if proc is None:
            return
        try:
            if self.alive:
                self.request("shutdown", timeout=timeout)
                self.notify("exit")
            proc.wait(timeout=timeout)
        except (LspError, OSError, subprocess.TimeoutExpired):
            return  # вежливо не вышло — процесс всё равно убивается (finally)
        finally:
            self._release(proc, timeout)

    # ------------------------------------------------------------------ #
    def _release(self, proc: subprocess.Popen[bytes], timeout: float) -> None:
        """Убивает сервер, дожидается потоков и закрывает свободные каналы."""
        _kill(proc, timeout)
        # процесс мёртв: застрявшая запись получает EPIPE, чтение — конец потока
        self._outbox.put(None)
        for worker in (self._writer, self._reader):
            if worker is not None:
                worker.join(timeout=timeout)
        # поток, всё ещё занятый каналом, держит его блокировку — такой не закрываем
        for stream, owner in ((proc.stdin, self._writer), (proc.stdout, self._reader)):
            if stream is not None and (owner is None or not owner.is_alive()):
                with contextlib.suppress(OSError):
                    stream.close()
        self._dead.set()
        self._proc = None

    @contextlib.contextmanager
    def _pending_slot(self) -> Generator[tuple[int, _Slot]]:
        """Номер нового запроса и место для ответа на него; по выходе место снимается."""
        with self._pending_lock:
            self._next_id += 1
            msg_id = self._next_id
            slot: _Slot = queue.Queue(maxsize=1)
            self._pending[msg_id] = slot
        try:
            yield msg_id, slot
        finally:
            with self._pending_lock:
                self._pending.pop(msg_id, None)

    def _cancel(self, msg_id: int) -> None:
        with contextlib.suppress(LspError):
            self.notify("$/cancelRequest", {"id": msg_id})

    def _send(self, message: dict[str, Any]) -> None:
        """Ставит сообщение в очередь писателя — не блокирует, даже если сервер не читает."""
        if self._proc is None or self._dead.is_set():
            raise LspError("сервер не запущен")
        body = json.dumps(message, ensure_ascii=False).encode("utf-8")
        self._outbox.put(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)

    def _write_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise AssertionError("писатель запущен без канала к серверу")
        stdin = proc.stdin
        while True:
            chunk = self._outbox.get()
            if chunk is None:
                return
            try:
                stdin.write(chunk)
                stdin.flush()
            except (OSError, ValueError):  # ValueError — запись в закрытый поток
                self._server_gone()
                return

    def _read_message(self, stream: IO[bytes]) -> dict[str, Any] | None:
        length = None
        while True:
            line = stream.readline(_MAX_HEADER_LINE)
            if not line:
                return None
            line = line.strip()
            if not line:
                break
            header, _, header_value = line.decode("ascii", errors="replace").partition(":")
            if header.strip().lower() == "content-length":
                length = int(header_value.strip())
        if length is None or not 0 <= length <= _MAX_MESSAGE:
            raise LspError("нарушен протокол: нет или неверная длина сообщения")
        body = stream.read(length)
        if len(body) < length:
            return None
        message = json.loads(body.decode("utf-8"))
        if not isinstance(message, dict):
            raise LspError("нарушен протокол: сообщение не объект")
        return message

    def _read_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise AssertionError("читатель запущен без канала от сервера")
        stdout = proc.stdout
        try:
            while True:
                message = self._read_message(stdout)
                if message is None:
                    break
                self._dispatch(message)
        except (LspError, OSError, ValueError):
            return  # оборванный или испорченный поток — как завершение сервера
        finally:
            self._server_gone()

    def _server_gone(self) -> None:
        """Сервер недоступен: ожидающие запросы получают отказ сразу, а не по таймауту."""
        self._dead.set()
        with self._pending_lock:
            waiting = list(self._pending.values())
        for slot in waiting:
            with contextlib.suppress(queue.Full):
                slot.put_nowait(None)

    def _dispatch(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        if method is None:  # ответ на наш запрос
            msg_id = message.get("id")
            with self._pending_lock:
                slot = self._pending.get(msg_id) if isinstance(msg_id, int) else None
            if slot is not None:
                with contextlib.suppress(queue.Full):
                    slot.put_nowait(message)
            return
        if "id" not in message:
            return  # уведомление сервера
        handler = self._handlers.get(method)
        request_id = message["id"]
        try:
            outcome = handler(message.get("params")) if handler else None
        except Exception as exc:  # ошибка обработчика — ответ-ошибка, а не падение читателя
            reply = _jsonrpc(id=request_id, error={"code": _INTERNAL_ERROR, "message": str(exc)})
        else:
            reply = _jsonrpc(id=request_id, result=outcome)
        with contextlib.suppress(LspError):
            self._send(reply)


def _failure_reason(reply: dict[str, Any] | None) -> object:
    """Почему запрос не удался: сервер завершился или прислал ответ-ошибку."""
    if reply is None:
        return "сервер завершился"
    error = reply["error"]
    message = error.get("message") if isinstance(error, dict) else None
    return message or "ошибка сервера"


def _jsonrpc(**fields: Any) -> dict[str, Any]:
    """Сообщение JSON-RPC 2.0: версия протокола, затем поля в порядке передачи."""
    return {"jsonrpc": "2.0", **fields}


def _kill(proc: subprocess.Popen[bytes], timeout: float) -> None:
    """Убивает сервер вместе с его потомками (своя группа процессов — см. start)."""
    try:
        if os.name == "posix":  # и если лидер уже вышел: потомки могли остаться
            os.killpg(proc.pid, signal.SIGKILL)
        elif proc.poll() is None:
            proc.kill()
    except OSError:
        proc.kill()
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=timeout)
