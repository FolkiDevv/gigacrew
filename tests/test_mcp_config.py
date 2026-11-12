"""MCP-серверы в настройках: источники, разбор записей, предупреждения, отпечатки."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gigacrew.project.mcp_config import (
    DEFAULT_STARTUP_TIMEOUT,
    DEFAULT_TOOL_TIMEOUT,
    MCP_JSON_FILE,
    McpServerConfig,
    load_mcp_settings,
    servers_fingerprint,
)
from gigacrew.project.workspace import Workspace, user_settings_file


def _write(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")


def _servers(document: dict[str, object]) -> dict[str, object]:
    return {"mcpServers": document}


def _load(project: Path) -> tuple[list[McpServerConfig], list[str]]:
    settings, warnings = load_mcp_settings(Workspace(project))
    return list(settings.effective()), warnings


def test_no_files_no_servers(project: Path) -> None:
    assert _load(project) == ([], [])


def test_servers_from_all_sources_later_file_wins(project: Path) -> None:
    _write(user_settings_file(), _servers({"docs": {"url": "https://u.test/mcp"}}))
    _write(project / MCP_JSON_FILE, _servers({"gh": {"command": "npx", "args": ["-y", "gh"]}}))
    workspace = Workspace(project)
    _write(workspace.local_settings_file, _servers({"docs": {"url": "https://local.test/mcp"}}))
    servers, warnings = _load(project)
    assert warnings == []
    by_name = {server.name: server for server in servers}
    assert set(by_name) == {"docs", "gh"}
    assert by_name["docs"].url == "https://local.test/mcp"
    assert by_name["docs"].transport == "http"
    assert by_name["docs"].source == "local"
    assert by_name["docs"].origin == ".gigacode/settings.local.json"
    assert by_name["gh"].transport == "stdio"
    assert by_name["gh"].args == ("-y", "gh")
    assert by_name["gh"].source == "project"
    assert by_name["gh"].origin == MCP_JSON_FILE


def test_untrusted_project_files_are_not_read(project: Path) -> None:
    _write(project / MCP_JSON_FILE, _servers({"gh": {"command": "npx"}}))
    _write(user_settings_file(), _servers({"mine": {"command": "my-server"}}))
    settings, _ = load_mcp_settings(Workspace(project, untrusted=True))
    assert [server.name for server in settings.effective()] == ["mine"]


def test_full_stdio_entry(project: Path) -> None:
    entry = {
        "type": "stdio",
        "command": "server",
        "args": ["--flag"],
        "env": {"TOKEN": "${TOKEN}"},
        "timeout": 30,
        "startupTimeout": 10,
        "autoApprove": ["search", "read"],
    }
    _write(project / MCP_JSON_FILE, _servers({"srv": entry}))
    (server,), warnings = _load(project)
    assert warnings == []
    assert server.env == (("TOKEN", "${TOKEN}"),)
    assert (server.timeout, server.startup_timeout) == (30, 10)
    assert server.approves("search")
    assert not server.approves("write")


def test_any_tool_auto_approve(project: Path) -> None:
    _write(project / MCP_JSON_FILE, _servers({"srv": {"command": "x", "autoApprove": ["*"]}}))
    server = _load(project)[0][0]
    assert server.approves("anything")


def test_http_entry_with_headers(project: Path) -> None:
    entry = {"type": "streamable-http", "url": "https://h.test/mcp", "headers": {"X-Key": "k"}}
    _write(project / MCP_JSON_FILE, _servers({"remote": entry}))
    server = _load(project)[0][0]
    assert server.transport == "http"
    assert server.headers == (("X-Key", "k"),)
    assert (server.timeout, server.startup_timeout) == (
        DEFAULT_TOOL_TIMEOUT,
        DEFAULT_STARTUP_TIMEOUT,
    )


def test_disabled_server_skipped(project: Path) -> None:
    _write(project / MCP_JSON_FILE, _servers({"off": {"command": "x", "disabled": True}}))
    assert _load(project) == ([], [])


@pytest.mark.parametrize(
    ("name", "entry", "fragment"),
    [
        ("bad name!", {"command": "x"}, "имя"),
        ("srv", ["x"], "ожидается объект"),
        ("srv", {"type": "sse", "url": "https://s.test"}, "sse"),
        ("srv", {"type": "websocket"}, "не поддерживается"),
        ("srv", {"type": []}, "не поддерживается"),
        ("srv", {"command": "x", "env": {"A=B": "1"}}, "env"),
        ("srv", {"command": "x", "env": {"": "1"}}, "env"),
        ("srv", {"command": ""}, "command"),
        ("srv", {"command": "x\u0000y"}, "command"),
        ("srv", {"type": "http", "url": "ftp://h.test"}, "url"),
        ("srv", {"command": "x", "args": "--flag"}, "args"),
        ("srv", {"command": "x", "env": {"A": 1}}, "env"),
        ("srv", {"url": "https://h.test", "headers": ["a"]}, "headers"),
        ("srv", {"command": "x", "autoApprove": "all"}, "autoApprove"),
    ],
)
def test_invalid_entry_skipped_with_warning(
    project: Path,
    name: str,
    entry: object,
    fragment: str,
) -> None:
    _write(project / MCP_JSON_FILE, _servers({name: entry}))
    servers, warnings = _load(project)
    assert servers == []
    assert len(warnings) == 1
    assert fragment in warnings[0]
    assert MCP_JSON_FILE in warnings[0]


@pytest.mark.parametrize(
    ("raw", "expected", "warned"),
    [
        (5, 5, False),
        (0, 1, True),
        (10**6, 3600, True),
        ("30", DEFAULT_TOOL_TIMEOUT, True),
        (True, DEFAULT_TOOL_TIMEOUT, True),
    ],
)
def test_timeout_is_clamped(project: Path, raw: object, expected: int, *, warned: bool) -> None:
    _write(project / MCP_JSON_FILE, _servers({"srv": {"command": "x", "timeout": raw}}))
    (server,), warnings = _load(project)
    assert server.timeout == expected
    assert bool(warnings) is warned


def test_broken_mcp_json_warns_but_broken_settings_is_silent(project: Path) -> None:
    (project / MCP_JSON_FILE).write_text("{oops", encoding="utf-8")
    user_settings_file().parent.mkdir(parents=True, exist_ok=True)
    user_settings_file().write_text("[1, 2]", encoding="utf-8")
    servers, warnings = _load(project)
    assert servers == []
    assert len(warnings) == 1
    assert MCP_JSON_FILE in warnings[0]


def test_section_not_object_warns(project: Path) -> None:
    _write(project / MCP_JSON_FILE, {"mcpServers": ["x"]})
    assert _load(project)[1] == [f"{MCP_JSON_FILE}: mcpServers — не объект, серверы пропущены"]


def test_fingerprint_tracks_every_field() -> None:
    base = McpServerConfig("srv", "stdio", command="x", source="project", origin=MCP_JSON_FILE)
    fingerprint = servers_fingerprint((base,))
    assert fingerprint
    assert not servers_fingerprint(())
    assert servers_fingerprint((base,)) == fingerprint
    variants = (
        McpServerConfig("srv", "stdio", command="y", source="project", origin=MCP_JSON_FILE),
        McpServerConfig(
            "srv",
            "stdio",
            command="x",
            env=(("LD_PRELOAD", "evil.so"),),
            source="project",
            origin=MCP_JSON_FILE,
        ),
        McpServerConfig(
            "srv",
            "stdio",
            command="x",
            auto_approve=frozenset(("*",)),
            source="project",
            origin=MCP_JSON_FILE,
        ),
    )
    assert all(servers_fingerprint((variant,)) != fingerprint for variant in variants)


def test_disabled_entry_turns_off_earlier_server(project: Path) -> None:
    _write(user_settings_file(), _servers({"srv": {"command": "x"}, "other": {"command": "y"}}))
    _write(Workspace(project).local_settings_file, _servers({"srv": {"disabled": True}}))
    settings, warnings = load_mcp_settings(Workspace(project))
    assert warnings == []
    assert [server.name for server in settings.effective()] == ["other"]
    # Выключение ничего не запускает: подтверждать в проекте нечего, и без подтверждения
    # проекта оно всё равно действует.
    assert settings.project_servers == ()
    unconfirmed = settings.without("project", "local")
    assert [server.name for server in unconfirmed.effective()] == ["other"]
