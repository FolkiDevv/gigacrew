"""Транспорт JSON-RPC 2.0 для ACP: строка JSON на сообщение поверх двоичных потоков.

Один поток читает вход (:meth:`RpcConnection.serve`). Входящие уведомления обрабатываются
сразу в нём, по порядку; каждый входящий запрос — в своём потоке, чтобы долгий обработчик
не мешал читать ответы клиента на исходящие запросы. Запись — под блокировкой, по
сообщению целиком. Исходящий запрос (:meth:`RpcConnection.request`) блокирует вызывающий
поток до ответа клиента, отмены или конца ввода.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import queue
import threading
from collections.abc import Callable, Generator, Mapping
from typing import Any, BinaryIO, Final

from pydantic import ValidationError

__all__ = [
    "INTERNAL_ERROR",
    "INVALID_PARAMS",
    "INVALID_REQUEST",
    "METHOD_NOT_FOUND",
    "PARSE_ERROR",
    "ConnectionClosedError",
    "RequestCancelledError",
    "RpcConnection",
    "RpcError",
]

# Коды ошибок JSON-RPC 2.0.
PARSE_ERROR: Final = -32700
INVALID_REQUEST: Final = -32600
METHOD_NOT_FOUND: Final = -32601
INVALID_PARAMS: Final = -32602
INTERNAL_ERROR: Final = -32603

_VERSION: Final = "2.0"
# Как часто ожидание ответа проверяет отмену, с. Ожидание с таймаутом ещё и даёт доставить
# в поток асинхронное исключение (KeyboardInterrupt отмены хода): бесконечное ожидание
# блокировки его не пропускает.
_POLL_INTERVAL: Final = 0.05

type Message = dict[str, Any]
type Handlers = Mapping[str, Callable[[Any], Any]]
# Место для ответа на исходящий запрос: ответ клиента или None — сигнал «ввод закончился».
type _Slot = queue.Queue[Message | None]


class RpcError(Exception):
    """Ошибка JSON-RPC: ответ-ошибка клиента или ошибка, которую обработчик шлёт клиенту."""

    def __init__(self, code: int, message: str, details: Any = None) -> None:
        """``details`` — поле ``data`` ошибки (None — без него)."""
        super().__init__(message)
        self.code = code
        self.message = message
        # Имя поля — термин JSON-RPC (``error.data``) и часть API исключения.
        self.data = details  # noqa: WPS110


class ConnectionClosedError(Exception):
    """Связь с клиентом оборвалась: запись не удалась или ввод закончился."""


class RequestCancelledError(Exception):
    """Исходящий запрос отменён до ответа клиента."""


class RpcConnection:
    """Соединение JSON-RPC 2.0: строка JSON на сообщение (ndjson) в обе стороны."""

    def __init__(self, reader: BinaryIO, writer: BinaryIO) -> None:
        """``reader`` — сообщения клиента (stdin агента), ``writer`` — к клиенту (stdout)."""
        self._reader = reader
        self._writer = writer
        self._write_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending: dict[int, _Slot] = {}
        self._ids = itertools.count(1)
        self._closed = False

    def serve(self, *, requests: Handlers, notifications: Handlers) -> None:
        """Читает вход до конца и раздаёт сообщения обработчикам.

        Конец ввода завершает ожидающие исходящие запросы ошибкой
        :class:`ConnectionClosedError`; запущенные обработчики запросов не дожидается.
        """
        with self._closing():
            for line in iter(self._reader.readline, b""):
                self._dispatch(line, requests, notifications)

    def notify(self, method: str, params: Any = None) -> None:
        """Отправляет уведомление; ``params=None`` — без поля params."""
        message: Message = {"jsonrpc": _VERSION, "method": method}
        if params is not None:
            message["params"] = params
        self._send(_encode(message))

    def request(
        self,
        method: str,
        params: Any = None,
        *,
        cancel: threading.Event | None = None,
    ) -> Any:
        """Отправляет запрос клиенту и ждёт ответа; возвращает его ``result``.

        Ответ-ошибка — :class:`RpcError`; поднятый ``cancel`` — :class:`RequestCancelledError`
        (поздний ответ клиента отбрасывается); обрыв связи — :class:`ConnectionClosedError`.
        """
        if cancel is not None and cancel.is_set():
            raise RequestCancelledError(method)
        with self._pending_request() as (msg_id, slot):
            message: Message = {"jsonrpc": _VERSION, "id": msg_id, "method": method}
            if params is not None:
                message["params"] = params
            self._send(_encode(message))
            reply = self._wait(slot, method, cancel)
        error = reply.get("error")
        if error is not None:
            raise _client_error(error)
        return reply.get("result")

    @contextlib.contextmanager
    def _closing(self) -> Generator[None]:
        """Как бы ни закончилось чтение, ожидающие запросы получают отказ."""
        try:
            yield
        finally:
            with self._pending_lock:
                self._closed = True
                slots = list(self._pending.values())
                self._pending.clear()
            for slot in slots:
                slot.put(None)

    @contextlib.contextmanager
    def _pending_request(self) -> Generator[tuple[int, _Slot]]:
        """Новый id и место для ответа на него; после блока поздний ответ отбрасывается."""
        slot: _Slot = queue.Queue(maxsize=1)
        with self._pending_lock:
            if self._closed:
                raise ConnectionClosedError("ввод закончился")
            msg_id = next(self._ids)
            self._pending[msg_id] = slot
        try:
            yield msg_id, slot
        finally:
            with self._pending_lock:
                self._pending.pop(msg_id, None)

    def _wait(self, slot: _Slot, method: str, cancel: threading.Event | None) -> Message:
        while True:
            try:
                reply = slot.get(timeout=_POLL_INTERVAL)
            except queue.Empty:
                if cancel is not None and cancel.is_set():
                    raise RequestCancelledError(method) from None
                continue
            if reply is None:
                raise ConnectionClosedError("ввод закончился")
            return reply

    def _send(self, line: bytes) -> None:
        with self._write_lock:
            try:
                self._writer.write(line)
                self._writer.flush()
            except (OSError, ValueError) as exc:
                # OSError — читатель ушёл (BrokenPipeError); ValueError — поток уже закрыт.
                raise ConnectionClosedError(str(exc)) from exc

    def _dispatch(self, line: bytes, requests: Handlers, notifications: Handlers) -> None:
        if not line.strip():
            return
        try:
            message = json.loads(line.decode())
        except (ValueError, RecursionError):
            self._reply_error(None, PARSE_ERROR, "Parse error")
            return
        if not isinstance(message, dict):
            self._reply_error(None, INVALID_REQUEST, "Invalid Request: ожидался объект")
            return
        if "method" not in message and ("result" in message or "error" in message):
            self._resolve(message)
        else:
            self._receive_call(message, requests, notifications)

    def _receive_call(
        self,
        message: Message,
        requests: Handlers,
        notifications: Handlers,
    ) -> None:
        raw_id = message.get("id")
        msg_id = raw_id if _is_valid_id(raw_id) else None
        problem = _call_problem(message, raw_id)
        if problem:
            self._reply_error(msg_id, INVALID_REQUEST, f"Invalid Request: {problem}")
            return
        method: str = message["method"]
        params = message.get("params")
        if "id" not in message:
            self._run_notification(notifications.get(method), params)
            return
        handler = requests.get(method)
        if handler is None:
            self._reply_error(msg_id, METHOD_NOT_FOUND, f"Method not found: {method}")
            return
        threading.Thread(
            target=self._run_request,
            args=(handler, msg_id, params),
            name=f"rpc-{method}",
            daemon=True,
        ).start()

    def _run_notification(self, handler: Callable[[Any], Any] | None, params: Any) -> None:
        if handler is None:
            return
        try:
            handler(params)
        except Exception:
            # На уведомление ответа нет, а сбой обработчика не должен останавливать чтение.
            return

    def _run_request(self, handler: Callable[[Any], Any], msg_id: Any, params: Any) -> None:
        try:
            reply = _response(msg_id, {"result": handler(params)})
        except SystemExit:
            raise
        # Любой сбой обработчика, и асинхронный KeyboardInterrupt отмены тоже, — ответ-ошибка:
        # запрос не должен остаться без ответа.
        except BaseException as exc:  # noqa: WPS424
            reply = _failure(msg_id, exc)
        try:
            self._send(reply)
        except ConnectionClosedError:
            # Клиент ушёл: ответ отдать некому, чтение входа решит само, когда закончить.
            return

    def _reply_error(self, msg_id: Any, code: int, message: str) -> None:
        try:
            self._send(_error_line(msg_id, code, message))
        except ConnectionClosedError:
            # Ответ на битый ввод отдать некому; читать вход дальше это не мешает.
            return

    def _resolve(self, message: Message) -> None:
        msg_id = message.get("id")
        if not isinstance(msg_id, int) or isinstance(msg_id, bool):
            return
        with self._pending_lock:
            slot = self._pending.pop(msg_id, None)
        if slot is not None:
            slot.put(message)


def _encode(message: Message) -> bytes:
    return f"{json.dumps(message, ensure_ascii=False)}\n".encode()


def _response(msg_id: Any, body: Message) -> bytes:
    """Ответ на запрос ``msg_id``: ``body`` — ``{"result": …}`` или ``{"error": …}``."""
    return _encode({"jsonrpc": _VERSION, "id": msg_id, **body})


def _error_line(msg_id: Any, code: int, message: str, details: Any = None) -> bytes:
    error: Message = {"code": code, "message": message}
    if details is not None:
        error["data"] = details
    return _response(msg_id, {"error": error})


def _failure(msg_id: Any, exc: BaseException) -> bytes:
    """Ответ-ошибка на сбой обработчика; трассировка клиенту не уходит."""
    if isinstance(exc, RpcError):
        return _error_line(msg_id, exc.code, exc.message, exc.data)
    if isinstance(exc, ValidationError):
        details = exc.errors(include_url=False, include_context=False, include_input=False)
        return _error_line(msg_id, INVALID_PARAMS, "Invalid params", details)
    return _error_line(msg_id, INTERNAL_ERROR, "Internal error")


def _is_valid_id(msg_id: Any) -> bool:
    return msg_id is None or (isinstance(msg_id, (int, str)) and not isinstance(msg_id, bool))


def _call_problem(message: Message, msg_id: Any) -> str:
    """Чем запрос или уведомление нарушает JSON-RPC 2.0; пустая строка — ничем."""
    if not _is_valid_id(msg_id):
        return "id должен быть строкой, числом или null"
    if message.get("jsonrpc") != _VERSION:
        return 'нужно "jsonrpc": "2.0"'
    if not isinstance(message.get("method"), str):
        return "method должен быть строкой"
    if not isinstance(message.get("params", {}), (dict, list)):
        return "params должен быть объектом или массивом"
    return ""


def _client_error(error: Any) -> RpcError:
    if not isinstance(error, dict):
        return RpcError(INTERNAL_ERROR, f"некорректная ошибка JSON-RPC: {error!r}")
    code = error.get("code")
    return RpcError(
        code if isinstance(code, int) and not isinstance(code, bool) else INTERNAL_ERROR,
        str(error.get("message", "")),
        error.get("data"),
    )
