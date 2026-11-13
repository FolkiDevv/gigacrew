"""Клиент MCP: stdio с настоящим процессом (tests/fake_mcp_server.py) и Streamable HTTP."""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

import httpx
import pytest

from gigacrew.tools.mcp_client import PROTOCOL_VERSION, McpClient
from gigacrew.tools.mcp_http import HttpTransport, parse_sse, sse_lines
from gigacrew.tools.mcp_transport import McpError, McpTimeoutError, StdioTransport

FAKE_SERVER = Path(__file__).with_name("fake_mcp_server.py")
TIMEOUT = 10.0
URL = "https://mcp.test/mcp"


def _stdio_client(tmp_path: Path, *options: str) -> McpClient:
    transport = StdioTransport(
        [sys.executable, str(FAKE_SERVER), *options],
        cwd=tmp_path,
        env=dict(os.environ),
        log_path=tmp_path / "logs" / "fake.log",
    )
    return McpClient(transport, roots=(tmp_path,), version="1.2.3")


@pytest.fixture
def client(tmp_path: Path) -> Iterator[McpClient]:
    session = _stdio_client(tmp_path)
    yield session
    session.close()


def _text(reply: dict[str, Any]) -> str:
    return str(reply["content"][0]["text"])


# ------------------------------- stdio ------------------------------- #
def test_handshake_and_paged_tool_list(client: McpClient, tmp_path: Path) -> None:
    about = client.connect(TIMEOUT)
    assert (about.name, about.version, about.protocol) == ("fake", "1.0", PROTOCOL_VERSION)
    assert about.instructions == "Сначала вызывай echo."
    assert about.has_tools
    names = [tool["name"] for tool in client.list_tools(TIMEOUT)]
    assert names == ["echo", "fail", "slow", "crash", "roots", "env", "image", "wipe"]
    assert client.alive
    log = (tmp_path / "logs" / "fake.log").read_text(encoding="utf-8")
    assert "fake server started" in log


def test_tool_call_and_server_request(client: McpClient, tmp_path: Path) -> None:
    client.connect(TIMEOUT)
    reply = client.call_tool("echo", {"text": "привет"}, timeout=TIMEOUT)
    assert json.loads(_text(reply)) == {"text": "привет"}
    roots = json.loads(_text(client.call_tool("roots", {}, timeout=TIMEOUT)))
    assert roots == {"roots": [{"uri": tmp_path.as_uri(), "name": tmp_path.name}]}
    failed = client.call_tool("fail", {}, timeout=TIMEOUT)
    assert failed["isError"] is True


def test_unknown_method_is_error(client: McpClient) -> None:
    client.connect(TIMEOUT)
    with pytest.raises(McpError, match="Method not found"):
        client.request("resources/list", timeout=TIMEOUT)


def test_timeout_raises(client: McpClient) -> None:
    client.connect(TIMEOUT)
    with pytest.raises(McpTimeoutError, match="не ответил"):
        client.call_tool("slow", {"seconds": 30}, timeout=0.3)


def test_server_exit_fails_request_immediately(client: McpClient) -> None:
    client.connect(TIMEOUT)
    started = time.monotonic()
    with pytest.raises(McpError, match="завершился"):
        client.call_tool("crash", {}, timeout=30)
    assert time.monotonic() - started < 10
    assert not client.alive


def test_unsupported_protocol_version(tmp_path: Path) -> None:
    session = _stdio_client(tmp_path, "--version", "1999-01-01")
    with closing(session), pytest.raises(McpError, match="1999-01-01"):
        session.connect(TIMEOUT)


def test_close_kills_busy_server(client: McpClient) -> None:
    client.connect(TIMEOUT)
    with pytest.raises(McpTimeoutError):
        client.call_tool("slow", {"seconds": 60}, timeout=0.2)
    started = time.monotonic()
    client.close()
    client.close()
    assert time.monotonic() - started < 10
    assert not client.alive


# -------------------------------- HTTP -------------------------------- #
class FakeHttpServer:
    """Сервер Streamable HTTP для httpx.MockTransport: записывает запросы."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.replies: list[dict[str, Any]] = []  # ответы клиента на запросы сервера
        self.status = 200
        self.stream: httpx.Response | None = None  # ответ на tools/call вместо обычного

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "DELETE":
            return httpx.Response(200)
        if self.status != 200:
            return httpx.Response(self.status, text="нет доступа")
        message = json.loads(request.content)
        if "method" not in message:
            self.replies.append(message)
            return httpx.Response(202)
        return self._answer(message)

    def _answer(self, message: dict[str, Any]) -> httpx.Response:
        method = message["method"]
        if "id" not in message:
            return httpx.Response(202)
        if method == "initialize":
            body = {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "remote", "version": "2"},
            }
            return _json(message, body, headers={"Mcp-Session-Id": "s-1"})
        if method == "tools/list":
            return _events(
                {"jsonrpc": "2.0", "method": "notifications/message", "params": {}},
                {"jsonrpc": "2.0", "id": 77, "method": "ping"},
                {"jsonrpc": "2.0", "id": message["id"], "result": {"tools": [{"name": "t"}]}},
            )
        if self.stream is not None:
            return self.stream
        return _json(message, {"content": [{"type": "text", "text": "ок"}]})


def _json(
    message: dict[str, Any],
    body: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    reply = {"jsonrpc": "2.0", "id": message["id"], "result": body}
    return httpx.Response(200, json=reply, headers=headers)


def _events(*messages: dict[str, Any]) -> httpx.Response:
    chunks = [": keep-alive\n\n"]
    chunks.extend(f"event: message\ndata: {json.dumps(message)}\n\n" for message in messages)
    stream = "".join(chunks)
    return httpx.Response(200, text=stream, headers={"Content-Type": "text/event-stream"})


def _http_client(server: FakeHttpServer) -> McpClient:
    transport = HttpTransport(
        URL,
        {"Authorization": "Bearer t"},
        transport=httpx.MockTransport(server),
    )
    return McpClient(transport)


def test_http_session_json_and_sse() -> None:
    server = FakeHttpServer()
    session = _http_client(server)
    about = session.connect(TIMEOUT)
    assert about.protocol == "2025-03-26"
    assert [tool["name"] for tool in session.list_tools(TIMEOUT)] == ["t"]
    assert server.replies == [{"jsonrpc": "2.0", "id": 77, "result": {}}]
    assert _text(session.call_tool("t", {}, timeout=TIMEOUT)) == "ок"
    session.close()
    first, *later = server.requests
    assert "mcp-session-id" not in first.headers
    assert first.headers["authorization"] == "Bearer t"
    assert all(request.headers["mcp-session-id"] == "s-1" for request in later)
    assert all(request.headers["mcp-protocol-version"] == "2025-03-26" for request in later)
    assert server.requests[-1].method == "DELETE"


def test_http_error_status() -> None:
    server = FakeHttpServer()
    server.status = 401
    with pytest.raises(McpError, match="HTTP 401: нет доступа"):
        _http_client(server).connect(TIMEOUT)


def test_http_expired_session_closes_channel() -> None:
    server = FakeHttpServer()
    session = _http_client(server)
    session.connect(TIMEOUT)
    server.status = 404
    with pytest.raises(McpError, match="сессия"):
        session.call_tool("t", {}, timeout=TIMEOUT)
    assert not session.alive


def test_parse_sse() -> None:
    lines = [
        ": comment",
        "event: endpoint",
        "data: /skip",
        "",
        "data: {",
        'data: "a": 1}',
        "",
        "id: 5",
        "data:2",
    ]
    assert list(parse_sse(lines)) == ['{\n"a": 1}', "2"]


def test_parse_sse_splits_only_on_sse_line_ends() -> None:
    chunks = [
        "data: a\u2028b\u0085c\r".encode(),
        b"\ndata: \xd0",
        b"\xb0\r\rdata: last",
    ]
    lines = list(sse_lines(chunks))
    assert lines == ["data: a\u2028b\u0085c", "data: а", "", "data: last"]
    assert list(parse_sse(lines)) == ["a\u2028b\u0085c\nа", "last"]


def test_sse_size_is_limited(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("gigacrew.tools.mcp_http.MAX_MESSAGE", 20)
    with pytest.raises(McpError, match="слишком большое событие"):
        list(parse_sse(["data: 0123456789", "data: 0123456789"]))
    with pytest.raises(McpError, match="слишком длинная строка"):
        list(sse_lines([b"data: 0123456789012345678901234567890"]))


def _keepalive_forever() -> Iterator[bytes]:
    for _ in range(10_000):  # дольше любого таймаута теста: ответа так и нет
        time.sleep(0.05)
        yield b": ping\n\n"


def test_http_keepalive_does_not_extend_timeout() -> None:
    server = FakeHttpServer()
    session = _http_client(server)
    session.connect(TIMEOUT)
    endless = httpx.Response(
        200,
        headers={"Content-Type": "text/event-stream"},
        content=_keepalive_forever(),
    )
    server.stream = endless
    started = time.monotonic()
    with pytest.raises(McpTimeoutError):
        session.call_tool("t", {}, timeout=0.5)
    assert time.monotonic() - started < 5


def test_deeply_nested_reply_fails_request_at_once(client: McpClient) -> None:
    client.connect(TIMEOUT)
    started = time.monotonic()
    with pytest.raises(McpError, match="завершился"):
        client.call_tool("deep", {}, timeout=30)
    assert time.monotonic() - started < 10


def test_closed_transport_does_not_start(tmp_path: Path) -> None:
    session = _stdio_client(tmp_path)
    session.close()
    with pytest.raises(McpError, match="закрыт"):
        session.connect(TIMEOUT)
    assert not session.alive


def test_interrupted_close_still_kills_server(tmp_path: Path) -> None:
    transport = StdioTransport(
        [sys.executable, str(FAKE_SERVER)],
        cwd=tmp_path,
        env=dict(os.environ),
    )
    session = McpClient(transport)
    session.connect(TIMEOUT)
    proc = transport._proc
    assert proc is not None
    original_wait = proc.wait
    calls: list[float | None] = []

    def wait(timeout: float | None = None) -> int:
        calls.append(timeout)
        if len(calls) == 1:
            raise KeyboardInterrupt
        return original_wait(timeout)

    proc.wait = wait  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        transport.close()
    assert proc.poll() is not None


def _endless_line() -> Iterator[bytes]:
    yield b"data: {"
    for _ in range(10_000):  # строка без конца, по байту: ответа так и нет
        time.sleep(0.05)
        yield b"x"


def test_http_unfinished_line_does_not_extend_timeout() -> None:
    server = FakeHttpServer()
    session = _http_client(server)
    session.connect(TIMEOUT)
    server.stream = httpx.Response(
        200,
        headers={"Content-Type": "text/event-stream"},
        content=_endless_line(),
    )
    started = time.monotonic()
    with pytest.raises(McpTimeoutError):
        session.call_tool("t", {}, timeout=0.5)
    assert time.monotonic() - started < 5
