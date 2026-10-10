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

import json
import os
import queue
import signal
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

# Обработчик запроса сервера: params -> result.
ServerRequestHandler = Callable[[Any], Any]

_MAX_HEADER_LINE = 8192
_MAX_MESSAGE = 64 * 1024 * 1024


class LspError(Exception):
    """Сбой обмена с сервером: ответ-ошибка, сервер завершился, нарушен протокол."""


class LspTimeout(LspError):
    """Сервер не ответил вовремя."""


_DIED = object()  # сигнал ожидающим: сервер завершился


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
    ):
        self._argv = list(argv)
        self._cwd = cwd
        self._env = dict(env) if env is not None else None
        self._stderr_path = stderr_path
        self._handlers = dict(handlers or {})
        self._proc: subprocess.Popen[bytes] | None = None
        self._outbox: queue.Queue[bytes | None] = queue.Queue()
        self._writer: threading.Thread | None = None
        self._reader: threading.Thread | None = None
        self._pending: dict[int, queue.Queue[object]] = {}
        self._pending_lock = threading.Lock()
        self._next_id = 0
        self._dead = threading.Event()

    # ------------------------------------------------------------------ #
    @property
    def alive(self) -> bool:
        return self._proc is not None and not self._dead.is_set() and self._proc.poll() is None

    def start(self) -> None:
        """Запускает процесс сервера. OSError — программа не запустилась."""
        kwargs: dict[str, Any] = {}
        if os.name == "posix":
            kwargs["start_new_session"] = True
        else:
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        stderr = open(self._stderr_path, "ab") if self._stderr_path else subprocess.DEVNULL  # noqa: SIM115
        try:
            self._proc = subprocess.Popen(
                self._argv,
                cwd=self._cwd,
                env=self._env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                **kwargs,
            )
        finally:
            if stderr is not subprocess.DEVNULL:
                stderr.close()  # у процесса своя копия дескриптора
        self._reader = threading.Thread(target=self._read_loop, name="lsp-reader", daemon=True)
        self._reader.start()
        self._writer = threading.Thread(target=self._write_loop, name="lsp-writer", daemon=True)
        self._writer.start()

    def request(self, method: str, params: Any = None, *, timeout: float) -> Any:
        """Запрос и ожидание ответа. LspError — ошибка сервера, LspTimeout — нет ответа."""
        with self._pending_lock:
            self._next_id += 1
            msg_id = self._next_id
            slot: queue.Queue[object] = queue.Queue(maxsize=1)
            self._pending[msg_id] = slot
        try:
            self._send({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params})
            try:
                reply = slot.get(timeout=timeout)
            except queue.Empty:
                self._cancel(msg_id)
                raise LspTimeout(f"{method}: сервер не ответил за {timeout:g} с") from None
            except BaseException:  # Ctrl+C посреди ожидания: ответ больше не нужен
                self._cancel(msg_id)
                raise
        finally:
            with self._pending_lock:
                self._pending.pop(msg_id, None)
        if reply is _DIED:
            raise LspError(f"{method}: сервер завершился")
        assert isinstance(reply, dict)
        if "error" in reply:
            err = reply["error"]
            message = err.get("message") if isinstance(err, dict) else None
            raise LspError(f"{method}: {message or 'ошибка сервера'}")
        return reply.get("result")

    def notify(self, method: str, params: Any = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

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
            pass
        finally:
            _kill(proc, timeout)
            # процесс мёртв: застрявшая запись получает EPIPE, чтение — конец потока
            self._outbox.put(None)
            for thread in (self._writer, self._reader):
                if thread is not None:
                    thread.join(timeout=timeout)
            # поток, всё ещё занятый каналом, держит его блокировку — такой не закрываем
            for stream, thread in ((proc.stdin, self._writer), (proc.stdout, self._reader)):
                if stream is not None and (thread is None or not thread.is_alive()):
                    try:
                        stream.close()
                    except OSError:
                        pass
            self._dead.set()
            self._proc = None

    # ------------------------------------------------------------------ #
    def _cancel(self, msg_id: int) -> None:
        try:
            self.notify("$/cancelRequest", {"id": msg_id})
        except LspError:
            pass

    def _send(self, message: dict[str, Any]) -> None:
        """Ставит сообщение в очередь писателя — не блокирует, даже если сервер не читает."""
        if self._proc is None or self._dead.is_set():
            raise LspError("сервер не запущен")
        data = json.dumps(message, ensure_ascii=False).encode("utf-8")
        self._outbox.put(f"Content-Length: {len(data)}\r\n\r\n".encode("ascii") + data)

    def _write_loop(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdin is not None
        while True:
            chunk = self._outbox.get()
            if chunk is None:
                return
            try:
                proc.stdin.write(chunk)
                proc.stdin.flush()
            except (OSError, ValueError):  # ValueError — запись в закрытый поток
                self._dead.set()
                self._wake_all()
                return

    def _read_message(self, stream) -> dict[str, Any] | None:
        length = None
        while True:
            line = stream.readline(_MAX_HEADER_LINE)
            if not line:
                return None
            line = line.strip()
            if not line:
                break
            name, _, value = line.decode("ascii", errors="replace").partition(":")
            if name.strip().lower() == "content-length":
                length = int(value.strip())
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
        assert proc is not None and proc.stdout is not None
        try:
            while True:
                message = self._read_message(proc.stdout)
                if message is None:
                    break
                self._dispatch(message)
        except (LspError, OSError, ValueError):
            pass  # оборванный или испорченный поток — как завершение сервера
        finally:
            self._dead.set()
            self._wake_all()

    def _wake_all(self) -> None:
        """Сервер недоступен: ожидающие запросы получают отказ сразу, а не по таймауту."""
        with self._pending_lock:
            waiting = list(self._pending.values())
        for slot in waiting:
            try:
                slot.put_nowait(_DIED)
            except queue.Full:
                pass

    def _dispatch(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        if method is None:  # ответ на наш запрос
            msg_id = message.get("id")
            with self._pending_lock:
                slot = self._pending.get(msg_id) if isinstance(msg_id, int) else None
            if slot is not None:
                try:
                    slot.put_nowait(message)
                except queue.Full:
                    pass
            return
        if "id" not in message:
            return  # уведомление сервера
        handler = self._handlers.get(method)
        reply: dict[str, Any] = {"jsonrpc": "2.0", "id": message["id"]}
        try:
            reply["result"] = handler(message.get("params")) if handler else None
        except Exception as e:  # ошибка обработчика — ответ-ошибка, а не падение читателя
            reply = {
                "jsonrpc": "2.0",
                "id": message["id"],
                "error": {"code": -32603, "message": str(e)},
            }
        try:
            self._send(reply)
        except LspError:
            pass


def _kill(proc: subprocess.Popen[bytes], timeout: float) -> None:
    """Убивает сервер вместе с его потомками (своя группа процессов — см. start)."""
    try:
        if os.name == "posix":  # и если лидер уже вышел: потомки могли остаться
            os.killpg(proc.pid, signal.SIGKILL)
        elif proc.poll() is None:
            proc.kill()
    except OSError:
        proc.kill()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass
