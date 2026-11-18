"""Тестовый клиент ACP: пара каналов os.pipe к серверу и поток, читающий его сообщения.

Сервер получает ``server_input`` (его stdin) и ``server_output`` (его stdout). Клиент пишет
сообщения JSON-RPC по строке и читает ответы с таймаутом: каждая строка вывода сервера
обязана быть одним объектом JSON-RPC 2.0 — посторонний вывод роняет тест.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from collections.abc import Callable
from typing import Any, BinaryIO

import pytest

__all__ = ["TIMEOUT", "AcpClient"]

# Сколько ждать сообщения сервера, с: с запасом на медленную машину CI.
TIMEOUT = 10.0

type Message = dict[str, Any]


class AcpClient:
    """Клиентская сторона канала JSON-RPC: отправка строк и приём сообщений сервера."""

    def __init__(self) -> None:
        server_reads, client_writes = os.pipe()
        client_reads, server_writes = os.pipe()
        self.server_input: BinaryIO = os.fdopen(server_reads, "rb")
        self.server_output: BinaryIO = os.fdopen(server_writes, "wb")
        self._writer: BinaryIO = os.fdopen(client_writes, "wb")
        self._reader: BinaryIO = os.fdopen(client_reads, "rb")
        self._lines: queue.Queue[bytes] = queue.Queue()
        # Сообщения, пропущенные ``response`` по пути к нужному ответу (уведомления и т. п.).
        self.backlog: list[Message] = []
        self._pump = threading.Thread(target=self._read_lines, daemon=True)
        self._pump.start()

    def send_line(self, line: bytes) -> None:
        """Пишет серверу строку как есть (для битого ввода)."""
        self._writer.write(line)
        self._writer.flush()

    def send(self, message: Message) -> None:
        """Пишет серверу сообщение одной строкой JSON."""
        self.send_line(f"{json.dumps(message, ensure_ascii=False)}\n".encode())

    def request(self, msg_id: int | str, method: str, params: Any = None) -> None:
        """Отправляет запрос; ``params=None`` — без поля params."""
        message: Message = {"jsonrpc": "2.0", "id": msg_id, "method": method}
        if params is not None:
            message["params"] = params
        self.send(message)

    def notify(self, method: str, params: Any = None) -> None:
        """Отправляет уведомление; ``params=None`` — без поля params."""
        message: Message = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self.send(message)

    def respond(self, msg_id: int | str | None, response: Any) -> None:
        """Отвечает на запрос сервера результатом."""
        self.send({"jsonrpc": "2.0", "id": msg_id, "result": response})

    def respond_error(
        self,
        msg_id: int | str | None,
        code: int,
        message: str,
        details: Any = None,
    ) -> None:
        """Отвечает на запрос сервера ошибкой; ``details`` — поле data, если задано."""
        error: Message = {"code": code, "message": message}
        if details is not None:
            error["data"] = details
        self.send({"jsonrpc": "2.0", "id": msg_id, "error": error})

    def receive(self, timeout: float = TIMEOUT) -> Message:
        """Следующее сообщение сервера; нет за ``timeout`` секунд — тест падает."""
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            pytest.fail(f"сервер ничего не прислал за {timeout} с")
        assert line.endswith(b"\n"), f"сообщение не завершено переводом строки: {line!r}"
        message = json.loads(line)
        assert isinstance(message, dict), f"не объект JSON: {line!r}"
        assert message.get("jsonrpc") == "2.0", f"не JSON-RPC 2.0: {line!r}"
        return message

    def response(self, msg_id: int | str, timeout: float = TIMEOUT) -> Message:
        """Ответ сервера на запрос ``msg_id``; встреченные по пути сообщения — в ``backlog``.

        Ответ, уже отложенный в ``backlog`` (например, :meth:`wait_for`), забирается оттуда.
        """
        return self.wait_for(lambda message: _is_response(message, msg_id), timeout)

    def wait_for(self, matches: Callable[[Message], bool], timeout: float = TIMEOUT) -> Message:
        """Первое сообщение, для которого ``matches`` истинно; прочие — в ``backlog``.

        Сначала ищется в ``backlog`` (найденное забирается оттуда), затем — среди новых.
        """
        for index, pending in enumerate(self.backlog):
            if matches(pending):
                return self.backlog.pop(index)
        while True:
            message = self.receive(timeout)
            if matches(message):
                return message
            self.backlog.append(message)

    def take_notifications(self, method: str) -> list[Any]:
        """Забирает из ``backlog`` уведомления ``method`` по порядку и возвращает их params."""
        taken = [
            message
            for message in self.backlog
            if message.get("method") == method and "id" not in message
        ]
        self.backlog = [message for message in self.backlog if message not in taken]
        return [message.get("params") for message in taken]

    def close_input(self) -> None:
        """Закрывает вход сервера: сервер увидит конец ввода (EOF)."""
        if not self._writer.closed:
            self._writer.close()

    def close(self) -> None:
        """Закрывает все концы каналов и дожидается потока-читателя."""
        self.close_input()
        self.server_output.close()
        self._pump.join(TIMEOUT)
        self._reader.close()
        self.server_input.close()

    def _read_lines(self) -> None:
        for line in iter(self._reader.readline, b""):
            self._lines.put(line)


def _is_response(message: Message, msg_id: int | str) -> bool:
    """Ответ (результат или ошибка) на запрос ``msg_id``, а не запрос сервера."""
    return message.get("id") == msg_id and "method" not in message
