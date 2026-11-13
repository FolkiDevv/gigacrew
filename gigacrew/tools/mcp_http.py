"""Транспорт Model Context Protocol поверх Streamable HTTP.

Каждое сообщение отправляется POST-запросом; ответ на запрос — JSON или поток SSE, где
перед ответом могут прийти запросы сервера (они уходят получателю и отвечаются
отдельными POST). Сессия — заголовок ``Mcp-Session-Id``; истекла (404) — канал больше
не годится, следующий вызов откроет новую сессию. Поток уведомлений сервера (GET) не
открывается: gigacrew обходится без них (поэтому и не предлагает HTTP-серверу
``roots``). Проверка TLS — по стандартным корням.

Поток SSE ограничен: строка и событие — не больше :data:`MAX_MESSAGE`, а срок ответа
проверяется на каждом принятом куске — ни keep-alive сервера (комментарии ``:``), ни
строка без конца его не продлевают.
"""

from __future__ import annotations

import codecs
import contextlib
import json
import re
import ssl
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any, Final, override

import httpx

from gigacrew.tools.mcp_transport import (
    MAX_MESSAGE,
    McpError,
    McpTimeoutError,
    Receiver,
    Transport,
    deliver,
    ignore_message,
    parse_message,
)

__all__ = ["HttpTransport", "parse_sse", "sse_lines"]

_CONNECT_TIMEOUT = 30.0
_CLOSE_TIMEOUT = 2.0
_ERROR_BODY = 300  # символов тела HTTP-ответа с ошибкой в сообщении
_MESSAGE_EVENTS: Final = frozenset(("", "message"))  # события SSE с сообщениями JSON-RPC
# Конец строки SSE — только CRLF, LF и CR (str.splitlines режет ещё и по U+2028, U+0085…).
_LINE_END_RE = re.compile(r"\r\n|\r|\n")


def sse_lines(chunks: Iterable[bytes]) -> Iterator[str]:
    """Строки потока SSE из байтов ответа.

    httpx.iter_lines режет по str.splitlines — U+2028 или U+0085 в тексте ответа разорвали
    бы событие, поэтому строки делятся здесь, по правилам SSE. CR в конце куска ждёт
    следующего: это может быть начало CRLF. Строка длиннее :data:`MAX_MESSAGE` — McpError.
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending = ""
    for chunk in chunks:
        pending = "".join((pending, decoder.decode(chunk)))
        held = pending.endswith("\r")
        *lines, rest = _LINE_END_RE.split(pending.removesuffix("\r") if held else pending)
        pending = f"{rest}\r" if held else rest
        if len(pending) > MAX_MESSAGE:
            raise McpError("слишком длинная строка в потоке сервера")
        yield from lines
    yield from _LINE_END_RE.split("".join((pending, decoder.decode(b"", final=True))))


def parse_sse(lines: Iterable[str]) -> Iterator[str]:
    """Данные событий ``message`` потока Server-Sent Events (строки без перевода строки).

    Многострочные ``data:`` склеиваются через перевод строки; комментарии и события
    других видов пропускаются. Событие больше :data:`MAX_MESSAGE` — McpError.
    """
    event = ""
    chunks: list[str] = []
    size = 0
    for line in lines:
        if line:
            event = _sse_field(line, event, chunks)
            size += len(line)
            if size > MAX_MESSAGE:
                raise McpError("слишком большое событие в потоке сервера")
            continue
        if chunks and event in _MESSAGE_EVENTS:
            yield "\n".join(chunks)
        event = ""
        chunks = []
        size = 0
    if chunks and event in _MESSAGE_EVENTS:
        yield "\n".join(chunks)


def _sse_field(line: str, event: str, chunks: list[str]) -> str:
    """Поле события SSE: ``data`` — в ``chunks``; результат — вид события."""
    key, _, rest = line.partition(":")
    rest = rest.removeprefix(" ")
    if key == "data":
        chunks.append(rest)
    elif key == "event":
        return rest
    return event  # комментарий («:…»), id, retry


def _before[ChunkT](deadline: float, chunks: Iterable[ChunkT]) -> Iterator[ChunkT]:
    """Куски потока, пока не истёк срок ответа.

    Срок проверяется на каждом принятом куске, а не на строке: ни keep-alive, ни строка,
    которую сервер шлёт по байту без конца, его не продлевают.
    """
    for chunk in chunks:
        if time.monotonic() > deadline:
            raise McpTimeoutError("сервер не прислал ответ вовремя")
        yield chunk


def _ssl_context() -> ssl.SSLContext:
    """Проверка сервера: корни как у httpx (certifi, SSL_CERT_FILE) и системное хранилище."""
    context = httpx.create_ssl_context()
    context.load_default_certs()
    return context


def _http_error(error: Exception, timeout: float) -> McpError:
    """Ошибка httpx или разбора ответа → McpError (таймаут — McpTimeoutError)."""
    if isinstance(error, httpx.TimeoutException):
        waited = format(timeout, "g")
        return McpTimeoutError(f"сервер не ответил за {waited} с")
    if isinstance(error, httpx.HTTPError):
        return McpError(f"HTTP: {error}")
    return McpError(f"нарушен протокол: {error}")


class HttpTransport(Transport):
    """Сервер по адресу ``url``: POST на каждое сообщение (см. описание модуля)."""

    def __init__(
        self,
        url: str,
        headers: Mapping[str, str] | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Запоминает адрес и заголовки; соединение создаёт :meth:`open`.

        ``transport`` — транспорт httpx (в тестах — ``httpx.MockTransport``).
        """
        self._url = url
        self._headers = dict(headers or {})
        self._transport = transport
        self._client: httpx.Client | None = None
        self._session = ""
        self._expired = False
        self._receive: Receiver = ignore_message

    @property
    @override
    def alive(self) -> bool:
        """Канал открыт и сессия не истекла (соединение HTTP — на каждый запрос)."""
        return self._client is not None and not self._expired

    @override
    def open(self, receive: Receiver, gone: Callable[[], None]) -> None:
        """Создаёт HTTP-клиент; сеть пока не трогается."""
        self._receive = receive
        self._client = httpx.Client(
            verify=_ssl_context(),
            follow_redirects=False,
            transport=self._transport,
        )

    @override
    def send(self, message: dict[str, Any], *, timeout: float) -> None:
        """POST сообщения; для запроса — ждёт ответ и отдаёт его получателю."""
        client = self._client
        if client is None:
            raise McpError("соединение закрыто")
        expected = message.get("id") if "method" in message else None
        deadline = time.monotonic() + timeout
        limits = httpx.Timeout(timeout, connect=min(timeout, _CONNECT_TIMEOUT))
        try:
            with client.stream(
                "POST",
                self._url,
                json=message,
                headers=self._request_headers(),
                timeout=limits,
            ) as response:
                self._accept(response, expected, deadline)
        # ValueError — тело ответа не JSON, RecursionError — JSON чрезмерной вложенности.
        except (httpx.HTTPError, ValueError, RecursionError) as exc:
            raise _http_error(exc, timeout) from exc

    @override
    def close(self) -> None:
        """Завершает сессию на сервере (DELETE) и закрывает соединения."""
        client = self._client
        if client is None:
            return
        self._client = None
        if self._session:
            with contextlib.suppress(httpx.HTTPError):
                client.delete(self._url, headers=self._request_headers(), timeout=_CLOSE_TIMEOUT)
        client.close()

    def _request_headers(self) -> dict[str, str]:
        headers = dict(self._headers)
        headers["Accept"] = "application/json, text/event-stream"
        headers["Content-Type"] = "application/json"
        if self._session:
            headers["Mcp-Session-Id"] = self._session
        if self.protocol_version:
            headers["MCP-Protocol-Version"] = self.protocol_version
        return headers

    def _accept(self, response: httpx.Response, expected: object, deadline: float) -> None:
        """Разбирает ответ на POST: ошибка HTTP, сессия, JSON или поток SSE."""
        session_open = bool(self._session)
        if session_open and response.status_code == httpx.codes.NOT_FOUND:
            self._expired = True  # нужна новая сессия: канал больше не годится
        _check_status(response, session=session_open)
        session = response.headers.get("mcp-session-id")
        if session:
            self._session = session
        if expected is None:
            return  # уведомление или ответ серверу: 202 Accepted без тела
        content_type = response.headers.get("content-type", "").partition(";")[0].strip()
        if content_type.lower() == "text/event-stream":
            self._read_events(response, expected, deadline)
            return
        answered = deliver(json.loads(_read_body(response)), self._receive)
        if expected not in answered:
            raise McpError("сервер не прислал ответ на запрос")

    def _read_events(self, response: httpx.Response, expected: object, deadline: float) -> None:
        lines = sse_lines(_before(deadline, response.iter_bytes()))
        for chunk in parse_sse(lines):
            payload = parse_message(chunk.encode("utf-8"))
            if expected in deliver(payload, self._receive):
                return
        raise McpError("сервер закрыл поток, не прислав ответ")


def _check_status(response: httpx.Response, *, session: bool) -> None:
    """Код ответа — ошибка: McpError с началом тела ответа (404 в сессии — она истекла)."""
    status = response.status_code
    if status == httpx.codes.NOT_FOUND and session:
        raise McpError("сессия на сервере истекла (HTTP 404)")
    if status < httpx.codes.BAD_REQUEST:
        return
    body = _read_body(response).decode("utf-8", errors="replace").strip()
    shown = body[:_ERROR_BODY]
    detail = f": {shown}" if shown else ""
    raise McpError(f"HTTP {status}{detail}")


def _read_body(response: httpx.Response) -> bytes:
    """Тело ответа не больше :data:`MAX_MESSAGE`; больше — McpError."""
    body = bytearray()
    for chunk in response.iter_bytes():
        body.extend(chunk)
        if len(body) > MAX_MESSAGE:
            raise McpError("слишком большой ответ сервера")
    return bytes(body)
