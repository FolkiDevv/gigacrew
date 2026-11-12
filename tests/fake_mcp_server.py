"""Поддельный MCP-сервер для тестов: JSON-RPC строками в stdin/stdout.

Запуск: ``python tests/fake_mcp_server.py [--version V] [--exit] [--no-tools-capability]``.

Инструменты: ``echo`` (аргументы — текстом), ``fail`` (ошибка инструмента), ``slow``
(спит ``seconds``), ``crash`` (процесс завершается), ``roots`` (спрашивает у клиента
``roots/list`` и возвращает ответ), ``env`` (значение переменной окружения ``name``),
``image`` (картинка). Список инструментов — на двух страницах.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

PAGE_ONE = (
    {
        "name": "echo",
        "description": "Повторяет аргументы",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "что повторить"},
                "count": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            },
            "required": ["text"],
        },
        "annotations": {"readOnlyHint": True},
    },
    {"name": "fail", "description": "Всегда ошибка", "inputSchema": {"type": "object"}},
    {
        "name": "slow",
        "inputSchema": {"type": "object", "properties": {"seconds": {"type": "number"}}},
    },
)
PAGE_TWO = (
    {"name": "crash", "inputSchema": {"type": "object"}},
    {"name": "roots", "inputSchema": {"type": "object"}},
    {
        "name": "env",
        "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}},
    },
    {"name": "image", "inputSchema": {"type": "object"}},
    {"name": "wipe", "inputSchema": {"type": "object"}, "annotations": {"destructiveHint": True}},
)


def send(message: dict[str, Any]) -> None:
    line = json.dumps(message, ensure_ascii=False)
    sys.stdout.write(f"{line}\n")
    sys.stdout.flush()


def read() -> dict[str, Any] | None:
    line = sys.stdin.readline()
    if not line:
        return None
    message: dict[str, Any] = json.loads(line)
    return message


def text(content: str, *, error: bool = False) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": content}], "isError": error}


def ask_roots() -> dict[str, Any]:
    send({"jsonrpc": "2.0", "id": "roots-1", "method": "roots/list"})
    while True:
        message = read()
        if message is None:
            sys.exit(0)
        if message.get("id") == "roots-1":
            return text(json.dumps(message.get("result"), ensure_ascii=False))


def call(params: dict[str, Any]) -> dict[str, Any]:
    name = params["name"]
    arguments = params.get("arguments") or {}
    if name == "echo":
        return text(json.dumps(arguments, ensure_ascii=False, sort_keys=True))
    if name == "fail":
        return text("инструмент сломан", error=True)
    if name == "slow":
        time.sleep(float(arguments.get("seconds", 5)))
        return text("проснулся")
    if name == "crash":
        sys.exit(3)
    if name == "roots":
        return ask_roots()
    if name == "env":
        return text(os.environ.get(str(arguments.get("name")), "<нет>"))
    if name == "image":
        return {"content": [{"type": "image", "data": "AAAA", "mimeType": "image/png"}]}
    return text(f"неизвестный инструмент {name}", error=True)


def answer(message: dict[str, Any], options: list[str]) -> dict[str, Any] | None:
    method = message.get("method")
    params = message.get("params") or {}
    if method == "initialize":
        version = options[options.index("--version") + 1] if "--version" in options else None
        capabilities: dict[str, Any] = {}
        if "--no-tools-capability" not in options:
            capabilities["tools"] = {}
        return {
            "protocolVersion": version or params["protocolVersion"],
            "capabilities": capabilities,
            "serverInfo": {"name": "fake", "version": "1.0"},
            "instructions": "Сначала вызывай echo.",
        }
    if method == "tools/list":
        if params.get("cursor") == "page-2":
            return {"tools": list(PAGE_TWO)}
        return {"tools": list(PAGE_ONE), "nextCursor": "page-2"}
    if method == "tools/call":
        return call(params)
    if method == "ping":
        return {}
    return None


def main() -> None:
    options = sys.argv[1:]
    sys.stderr.write("fake server started\n")
    sys.stderr.flush()
    if "--exit" in options:
        sys.stderr.write("fatal: no config\n")
        sys.exit(1)
    sys.stdout.write("not json — debug print\n")
    while True:
        message = read()
        if message is None:
            return
        if "id" not in message or "method" not in message:
            continue  # уведомление или ответ
        reply = answer(message, options)
        if reply is None:
            error = {"code": -32601, "message": "Method not found"}
            send({"jsonrpc": "2.0", "id": message["id"], "error": error})
        else:
            send({"jsonrpc": "2.0", "id": message["id"], "result": reply})


if __name__ == "__main__":
    main()
