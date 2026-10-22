"""Тесты минимального LSP-клиента против поддельного сервера на пайпах."""

from __future__ import annotations

import sys
import textwrap

import pytest

from gigacrew.project.lsp import LspClient, LspError, LspTimeoutError

# Поддельный сервер: echo, ошибка, зависание, запрос к клиенту, внезапный выход.
FAKE_SERVER = textwrap.dedent(
    r"""
    import json, sys, time

    def read():
        length = None
        while True:
            line = sys.stdin.buffer.readline()
            if not line:
                return None
            line = line.strip()
            if not line:
                break
            name, _, value = line.decode().partition(":")
            if name.lower() == "content-length":
                length = int(value)
        return json.loads(sys.stdin.buffer.read(length))

    def send(msg):
        data = json.dumps(msg, ensure_ascii=False).encode()
        sys.stdout.buffer.write(b"Content-Length: %d\r\n\r\n" % len(data) + data)
        sys.stdout.buffer.flush()

    while True:
        msg = read()
        if msg is None:
            break
        method, mid = msg.get("method"), msg.get("id")
        if method == "echo":
            send({"jsonrpc": "2.0", "method": "window/logMessage", "params": {"m": 1}})
            send({"jsonrpc": "2.0", "id": mid, "result": msg["params"]})
        elif method == "ask":
            send({"jsonrpc": "2.0", "id": "srv-1", "method": "workspace/configuration",
                  "params": {"items": [{}, {}]}})
            answer = read()
            send({"jsonrpc": "2.0", "id": mid, "result": answer})
        elif method == "fail":
            send({"jsonrpc": "2.0", "id": mid, "error": {"code": -1, "message": "плохо"}})
        elif method == "badfail":
            send({"jsonrpc": "2.0", "id": mid, "error": "не объект"})
        elif method == "stall":
            time.sleep(60)  # перестал читать stdin
        elif method == "hang":
            pass
        elif method == "die":
            sys.exit(3)
        elif method == "shutdown":
            send({"jsonrpc": "2.0", "id": mid, "result": None})
        elif method == "exit":
            break
    """,
)


@pytest.fixture
def client(tmp_path):
    script = tmp_path / "fake_server.py"
    script.write_text(FAKE_SERVER, encoding="utf-8")
    c = LspClient(
        [sys.executable, "-I", str(script)],
        cwd=tmp_path,
        stderr_path=tmp_path / "server.log",
        handlers={"workspace/configuration": lambda p: [{"n": i} for i in range(len(p["items"]))]},
    )
    c.start()
    yield c
    c.close()


def test_request_roundtrip_with_unicode_and_notifications(client):
    assert client.alive
    assert client.request("echo", {"текст": "ёж 😀"}, timeout=5) == {"текст": "ёж 😀"}
    assert client.request("echo", [1, 2], timeout=5) == [1, 2]  # уведомления не мешают


def test_server_request_is_answered_by_handler(client):
    answer = client.request("ask", None, timeout=5)
    assert answer == {"jsonrpc": "2.0", "id": "srv-1", "result": [{"n": 0}, {"n": 1}]}


def test_error_response_and_timeout_keep_client_usable(client):
    with pytest.raises(LspError, match="fail: плохо"):
        client.request("fail", None, timeout=5)
    with pytest.raises(LspTimeoutError, match="hang: сервер не ответил"):
        client.request("hang", None, timeout=0.2)
    assert client.request("echo", "ещё жив", timeout=5) == "ещё жив"


def test_server_death_fails_pending_and_later_requests(client):
    with pytest.raises(LspError, match="сервер завершился"):
        client.request("die", None, timeout=5)
    assert not client.alive
    with pytest.raises(LspError):
        client.request("echo", 1, timeout=1)


def test_close_is_polite_and_idempotent(client, tmp_path):
    client.close()
    assert not client.alive
    client.close()
    with pytest.raises(LspError, match="не запущен"):
        client.notify("echo")


def test_start_fails_for_missing_program(tmp_path):
    c = LspClient([str(tmp_path / "нет-такого")], cwd=tmp_path)
    with pytest.raises(OSError):
        c.start()
    c.close()


def test_malformed_error_object_is_lsp_error(client):
    with pytest.raises(LspError, match="badfail: ошибка сервера"):
        client.request("badfail", None, timeout=5)


def test_stalled_server_cannot_block_requests_or_close(client):
    import time

    with pytest.raises(LspTimeoutError):
        client.request("stall", None, timeout=0.2)
    # сервер не читает: запись в переполненный канал не должна вешать запрос
    started = time.monotonic()
    with pytest.raises(LspTimeoutError):
        client.request("echo", "x" * 2_000_000, timeout=0.5)
    client.close(timeout=1)
    assert not client.alive
    assert time.monotonic() - started < 10


def test_interrupted_close_still_kills_server(client, monkeypatch):
    proc = client._proc

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(client, "request", interrupted)
    with pytest.raises(KeyboardInterrupt):
        client.close(timeout=1)
    assert proc.poll() is not None
    assert not client.alive


@pytest.mark.skipif(sys.platform == "win32", reason="группы процессов POSIX")
def test_kill_reaches_children_after_leader_exits(tmp_path):
    import os
    import subprocess

    from gigacrew.project.lsp import _kill

    pid_file = tmp_path / "child.pid"
    proc = subprocess.Popen(
        ["sh", "-c", f"sleep 60 & echo $! > {pid_file}; exit 0"],
        start_new_session=True,
    )
    proc.wait(timeout=5)  # лидер вышел, потомок жив
    child = int(pid_file.read_text())
    _kill(proc, timeout=1)
    import time

    for _ in range(50):
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("потомок сервера пережил _kill")
