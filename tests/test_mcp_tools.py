"""MCP-серверы в агенте: запуск и перезапуск, риск вызовов, цикл агента, CLI и /mcp."""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import closing
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, Literal, override

import httpx
import pytest
from fakes import RecordingEvents, ScriptedProvider, text_turn, tool_turn

from gigacrew.agent.guard import call_key, target_key
from gigacrew.agent.loop import Agent
from gigacrew.cli import app
from gigacrew.cli.commands import default_commands
from gigacrew.cli.mcp import connect_mcp, describe_hub, describe_servers
from gigacrew.config import Config
from gigacrew.llm.types import FunctionCall
from gigacrew.permissions import PermissionMode, ToolKind
from gigacrew.project.mcp_config import MCP_JSON_FILE, McpServerConfig
from gigacrew.project.settings import TrustStore
from gigacrew.project.workspace import Workspace, user_settings_file
from gigacrew.security import RiskLevel
from gigacrew.tools import mcp as mcp_tools
from gigacrew.tools.base import ToolContext, ToolError, ToolRegistry, build_default_registry
from gigacrew.tools.mcp import McpArguments, McpHub, McpServer, McpTool
from gigacrew.tools.mcp_client import ServerInfo
from gigacrew.tools.mcp_env import expand_env
from gigacrew.tools.mcp_http import HttpTransport
from gigacrew.tools.mcp_transport import McpError
from gigacrew.ui.console import Console

FAKE_SERVER = Path(__file__).with_name("fake_mcp_server.py")
ECHO = "mcp__fake__echo"

_Calls = list[tuple[str, dict[str, Any]]]

# Фикстура oneshot из conftest.py: ``oneshot(provider)`` подменяет провайдера для -p.
_OneShot = Callable[[ScriptedProvider], None]


def _fake(*options: str, **fields: Any) -> McpServerConfig:
    fields.setdefault("startup_timeout", 20)
    fields.setdefault("timeout", 20)
    args = (str(FAKE_SERVER), *options)
    return McpServerConfig("fake", "stdio", command=sys.executable, args=args, **fields)


@pytest.fixture
def hub(project: Path) -> Iterator[McpHub]:
    servers = McpHub([McpServer(_fake(), Workspace(project))])
    try:
        yield servers
    finally:
        servers.close()


class StubServer(McpServer):
    """Сервер без процесса: ``call`` возвращает заданный ответ и записывает вызовы."""

    def __init__(self, project: Path, reply: dict[str, Any], **fields: Any) -> None:
        config = McpServerConfig("stub", "stdio", command="x", **fields)
        super().__init__(config, Workspace(project))
        self.reply = reply
        self.calls: _Calls = []
        self.about = ServerInfo("stub", "1", "2025-06-18", instructions="Сначала search.")

    @override
    def call(
        self,
        tool: str,
        arguments: dict[str, Any],
        *,
        hints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls.append((tool, arguments))
        if not self.reply:
            raise McpError("сервер завершился")
        return self.reply


def _stub_tool(server: McpServer, name: str = "search", **described: Any) -> McpTool:
    return McpTool(server, {"name": name, **described}, f"mcp__stub__{name}")


def _text_reply(text: str, *, error: bool = False) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": error}


# ------------------------------- сервер ------------------------------- #
def test_hub_connects_and_registers_tools(hub: McpHub) -> None:
    hub.connect()
    registry = build_default_registry()
    assert not hub.register(registry)
    tool = registry.get(ECHO)
    assert isinstance(tool, McpTool)
    assert tool.spec().parameters["required"] == ["text"]
    assert tool.description == "[MCP fake] Повторяет аргументы"
    reply = tool.run(tool.parse({"text": "эхо"}), ToolContext(Workspace(Path.cwd())))
    assert json.loads(reply.content) == {"text": "эхо"}
    assert reply.ok


def test_crashed_server_restarts_on_next_call(hub: McpHub) -> None:
    hub.connect()
    server = hub.servers[0]
    with pytest.raises(McpError, match="завершился"):
        server.call("crash", {})
    assert not server.connected
    assert json.loads(server.call("echo", {"text": "снова"})["content"][0]["text"]) == {
        "text": "снова",
    }


def test_concurrent_calls_reconnect_crashed_server_once(
    hub: McpHub,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub.connect()
    server = hub.servers[0]
    with pytest.raises(McpError, match="завершился"):
        server.call("crash", {})
    connects: list[str] = []
    original = McpServer.connect

    def slow_connect(self: McpServer) -> None:
        connects.append(self.name)
        time.sleep(0.2)  # второй вызов (фоновый суб-агент) приходит, пока первый подключается
        original(self)

    monkeypatch.setattr(McpServer, "connect", slow_connect)
    replies: list[dict[str, Any]] = []

    def call_echo() -> None:
        replies.append(server.call("echo", {"text": "x"}))

    workers = [threading.Thread(target=call_echo) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(30)
    assert connects == ["fake"]  # переподключал один поток
    assert len(replies) == 2


@pytest.mark.parametrize(
    ("config", "fragment"),
    [
        (_fake("--exit"), "fatal: no config"),
        (McpServerConfig("fake", "stdio", command="no-such-mcp-program"), "программа не найдена"),
        (_fake(env=(("TOKEN", "${MCP_TEST_UNSET}"),)), "MCP_TEST_UNSET"),
        (_fake(env=(("A=B", "x"),)), "environment variable"),
    ],
)
def test_failed_start_reports_reason(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: McpServerConfig,
    fragment: str,
) -> None:
    monkeypatch.delenv("MCP_TEST_UNSET", raising=False)
    servers = McpHub([McpServer(config, Workspace(project))])
    servers.connect()
    server = servers.servers[0]
    assert not server.connected
    assert fragment in server.error
    assert servers.tools() == ([], [])
    with pytest.raises(McpError):
        server.call("echo", {})
    servers.close()


def test_env_expanded_without_gigacrew_secrets(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MCP_TEST_TOKEN", "t-1")
    monkeypatch.setenv("GIGACREW_ACCESS_KEY", "секрет")
    env = (("TOKEN", "${MCP_TEST_TOKEN}"), ("STOLEN", "${GIGACREW_ACCESS_KEY:-нет}"))
    with closing(McpServer(_fake(env=env), Workspace(project))) as server:
        server.connect()
        seen = [
            server.call("env", {"name": name})["content"][0]["text"]
            for name in ("TOKEN", "STOLEN", "GIGACREW_ACCESS_KEY")
        ]
    assert seen == ["t-1", "нет", "<нет>"]


def test_expand_env() -> None:
    env = {"A": "1", "EMPTY": ""}
    assert expand_env("${A}-${EMPTY:-d}-${B:-x}", env) == ("1-d-x", [])
    assert expand_env("${EMPTY}${B}", env) == ("", ["B"])
    assert expand_env("$A ${not-a-var}", env) == ("$A ${not-a-var}", [])


# ------------------------------ инструмент ------------------------------ #
def test_assess_risk(project: Path) -> None:
    server = StubServer(project, _text_reply("ок"), auto_approve=frozenset(("safe",)))
    plan = ToolContext(Workspace(project), get_mode=lambda: PermissionMode.PLAN)
    auto = ToolContext(Workspace(project))
    args = McpArguments()
    assert _stub_tool(server, "safe").assess(args, auto).risk is RiskLevel.SAFE
    assert _stub_tool(server).assess(args, auto).risk is RiskLevel.WRITE
    destructive = _stub_tool(server, annotations={"destructiveHint": True})
    assert destructive.assess(args, auto).risk is RiskLevel.DANGEROUS
    reader = _stub_tool(server, annotations={"readOnlyHint": True})
    assert reader.assess(args, auto).kind is None
    assert reader.assess(args, plan).kind is ToolKind.COMMAND


def test_describe_preview_and_errors(project: Path) -> None:
    server = StubServer(project, _text_reply("сломано", error=True))
    tool = _stub_tool(server)
    ctx = ToolContext(Workspace(project))
    params = tool.parse({"query": "x" * 200})
    assert len(tool.describe(params)) <= 70
    preview = tool.preview(params, ctx)
    assert preview is not None
    assert "x" * 200 in preview.text
    assert not tool.paths(params)
    failed = tool.run(params, ctx)
    assert not failed.ok
    assert failed.summary == "сломано"
    server.reply = {}
    with pytest.raises(ToolError, match="MCP-сервер stub"):
        tool.run(params, ctx)


def test_call_key_keeps_mcp_description() -> None:
    first = FunctionCall(name="mcp__jira__create", arguments={"description": "a"})
    second = FunctionCall(name="mcp__jira__create", arguments={"description": "b"})
    assert call_key(first) != call_key(second)
    shell = FunctionCall(name="run_shell", arguments={"command": "ls", "description": "a"})
    assert call_key(shell) == call_key(FunctionCall(name="run_shell", arguments={"command": "ls"}))


# ---------------------------- цикл агента ---------------------------- #
def _agent(
    project: Path,
    tool: McpTool,
    turns: list[Any],
    events: RecordingEvents,
    **cfg: Any,
) -> tuple[Agent, ScriptedProvider]:
    provider = ScriptedProvider(turns)
    registry = ToolRegistry()
    registry.register(tool)
    config = Config(access_key="x", project_root=project, stream=False, **cfg)
    return Agent(provider, registry, config, events), provider


def _function_results(provider: ScriptedProvider) -> list[str]:
    messages = provider.requests[-1]["messages"]
    return [str(message.content) for message in messages if message.role == "function"]


def test_agent_asks_before_mcp_call_and_sends_result(project: Path) -> None:
    server = StubServer(project, _text_reply("найдено"))
    events = RecordingEvents()
    turns = [tool_turn("mcp__stub__search", {"query": "q"}), text_turn("готово")]
    agent, provider = _agent(project, _stub_tool(server), turns, events)
    agent.run_turn("найди")
    assert server.calls == [("search", {"query": "q"})]
    call, preview, dangerous = events.confirms[0]
    assert call.name == "mcp__stub__search"
    assert "MCP-сервера stub" in call.reason
    assert preview is not None
    assert '"query": "q"' in preview.text
    assert not dangerous
    assert _function_results(provider) == ["найдено"]


def test_agent_refusal_does_not_call_server(project: Path) -> None:
    server = StubServer(project, _text_reply("найдено"))
    events = RecordingEvents(confirm_answer=False)
    turns = [tool_turn("mcp__stub__search", {"query": "q"}), text_turn("ладно")]
    agent, _ = _agent(project, _stub_tool(server), turns, events)
    agent.run_turn("найди")
    assert not server.calls


@pytest.mark.parametrize(
    ("mode", "hints", "runs"),
    [
        (PermissionMode.AUTO, {}, True),
        (PermissionMode.PLAN, {"readOnlyHint": True}, True),
        # autoApprove — «без вопроса», а не «ничего не меняет»: в плане не действует.
        (PermissionMode.PLAN, {}, False),
    ],
)
def test_auto_approved_tool_runs_without_question(
    project: Path,
    mode: PermissionMode,
    hints: dict[str, Any],
    *,
    runs: bool,
) -> None:
    server = StubServer(project, _text_reply("ок"), auto_approve=frozenset(("*",)))
    events = RecordingEvents()
    tool = _stub_tool(server, annotations=hints)
    turns = [tool_turn("mcp__stub__search", {}), text_turn("готово")]
    agent, _ = _agent(project, tool, turns, events, mode=mode)
    agent.run_turn("найди")
    assert not events.confirms
    assert len(server.calls) == int(runs)


def test_plan_mode_blocks_writer_but_asks_for_reader(project: Path) -> None:
    server = StubServer(project, _text_reply("ок"))
    for hints, expected in (({}, 0), ({"readOnlyHint": True}, 1)):
        events = RecordingEvents()
        tool = _stub_tool(server, annotations=hints)
        turns = [tool_turn("mcp__stub__search", {"q": 1}), text_turn("готово")]
        agent, _ = _agent(project, tool, turns, events, mode=PermissionMode.PLAN)
        agent.run_turn("план")
        assert len(events.confirms) == expected
    assert len(server.calls) == 1


def test_system_prompt_has_server_guidance_once(project: Path) -> None:
    server = StubServer(project, _text_reply("ок"))
    registry = ToolRegistry()
    registry.register(_stub_tool(server, "search"))
    registry.register(_stub_tool(server, "read"))
    config = Config(access_key="x", project_root=project, stream=False)
    prompt = Agent(ScriptedProvider(), registry, config).system_prompt()
    assert prompt.count("MCP server stub") == 1
    assert "Сначала search." in prompt
    plain = Agent(ScriptedProvider(), build_default_registry(), config).system_prompt()
    assert "MCP server" not in plain


# -------------------------------- CLI -------------------------------- #
def _settings(config: McpServerConfig) -> dict[str, object]:
    entry = {"command": config.command, "args": list(config.args)}
    return {"mcpServers": {config.name: entry}}


def _write(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _console(
    monkeypatch: pytest.MonkeyPatch,
    answers: tuple[bool, ...] = (),
) -> tuple[Console, list[str]]:
    ui = Console(no_color=True)
    said: list[str] = []
    queue = list(answers)

    def ask(question: str, *, dangerous: bool = False) -> bool:
        said.append(f"? {question}")
        return queue.pop(0)

    monkeypatch.setattr(ui, "ask", ask)
    monkeypatch.setattr(ui, "warn", lambda text: said.append(f"! {text}"))
    return ui, said


def test_unconfirmed_servers_do_not_start_without_terminal(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write(project / MCP_JSON_FILE, _settings(_fake()))
    ui, said = _console(monkeypatch)
    registry = ToolRegistry()
    servers = connect_mcp(Workspace(project), ui, registry, interactive=False)
    assert not servers.servers
    assert not list(registry)
    assert any("--trust-mcp" in line for line in said)


def test_confirmation_is_remembered(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(user_settings_file(), _settings(_fake()))
    store = TrustStore(project / "trusted.json")
    workspace = Workspace(project)
    ui, said = _console(monkeypatch, (True,))
    registry = ToolRegistry()
    connect_mcp(workspace, ui, registry, interactive=True, store=store).close()
    assert said[-1] == "? Разрешить MCP-серверы пользователя?"
    assert said[0].startswith("! новые или изменённые MCP-серверы пользователя")
    assert ECHO in registry
    ui, said = _console(monkeypatch)
    connect_mcp(workspace, ui, ToolRegistry(), interactive=True, store=store).close()
    assert not said


def test_declined_servers_are_skipped(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(project / MCP_JSON_FILE, _settings(_fake()))
    ui, _ = _console(monkeypatch, (False,))
    store = TrustStore(project / "trusted.json")
    registry = ToolRegistry()
    connect_mcp(Workspace(project), ui, registry, interactive=True, store=store).close()
    assert not list(registry)


def test_describe_hub(hub: McpHub, project: Path) -> None:
    workspace = Workspace(project)
    assert describe_hub(None, workspace).startswith("MCP-серверов нет")
    hub.connect()
    text = describe_hub(hub, workspace)
    assert "fake · stdio" in text
    assert "подключён (fake 1.0): инструментов — 8" in text
    assert ECHO in text
    assert default_commands().get("/mcp") is not None


def test_oneshot_trust_mcp_runs_tool(
    tmp_path: Path,
    oneshot: _OneShot,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write(tmp_path / MCP_JSON_FILE, _settings(_fake()))
    provider = ScriptedProvider([tool_turn(ECHO, {"text": "из CLI"}), text_turn("готово")])
    oneshot(provider)
    argv = ["-C", str(tmp_path), "-p", "эхо", "--no-color", "--no-stream", "-y", "--trust-mcp"]
    assert app.main(argv) == app.EXIT_OK
    assert _function_results(provider) == [json.dumps({"text": "из CLI"}, ensure_ascii=False)]
    assert ECHO in {spec.name for spec in provider.requests[0]["tools"]}
    assert "MCP-серверы: подключение (fake)" in capsys.readouterr().out


def test_http_server_is_not_offered_roots(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        message = json.loads(request.content)
        seen.append(message)
        if "id" not in message:
            return httpx.Response(202)
        reply = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}}
        if message["method"] == "tools/list":
            reply = {"tools": []}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": reply})

    factory = partial(HttpTransport, transport=httpx.MockTransport(handler))
    monkeypatch.setattr(mcp_tools, "HttpTransport", factory)
    config = McpServerConfig("remote", "http", url="https://h.test/mcp")
    with closing(McpServer(config, Workspace(project))) as server:
        server.connect()
        assert server.connected
    assert not seen[0]["params"]["capabilities"]


def test_confirmation_shows_argument_boundaries() -> None:
    config = McpServerConfig("srv", "stdio", command="run me", args=("a b", "--x='1'"))
    shown = describe_servers((config,))
    assert "'run me' 'a b'" in shown


@pytest.mark.parametrize("source", ["user", "project"])
def test_user_servers_do_not_start_in_project(
    project: Path,
    user_home: Path,
    source: Literal["user", "project"],
) -> None:
    config = replace(_fake(), source=source)
    with closing(McpServer(config, Workspace(project))) as server:
        server.connect()
        cwd = server.call("cwd", {})["content"][0]["text"]
    expected = project if source == "project" else user_home
    assert Path(cwd).resolve() == expected.resolve()


def test_confirmation_shows_header_values_and_newlines() -> None:
    config = McpServerConfig(
        "srv",
        "http",
        url="https://h.test/mcp",
        headers=(("X-Leak", "${AWS_SECRET_ACCESS_KEY}"),),
    )
    assert "заголовок X-Leak: ${AWS_SECRET_ACCESS_KEY}" in describe_servers((config,))
    forged = McpServerConfig("srv", "stdio", command="x", args=("ok\n  safe-looking",))
    shown = describe_servers((forged,))
    assert "⟨U+000A⟩" in shown
    assert len(shown.splitlines()) == 2


def test_preview_says_when_arguments_are_cut(project: Path) -> None:
    tool = _stub_tool(StubServer(project, _text_reply("ок")))
    preview = tool.preview(tool.parse({"text": "x" * 50_000}), ToolContext(Workspace(project)))
    assert preview is not None
    assert "показаны первые" in preview.text
    assert "отклоните" in preview.text


def test_similar_guard_tells_mcp_calls_apart() -> None:
    first = FunctionCall(name="mcp__docs__lookup", arguments={"id": 1})
    second = FunctionCall(name="mcp__docs__lookup", arguments={"id": 2})
    assert target_key(first) != target_key(second)
    builtin = FunctionCall(name="list_dir", arguments={"depth": 1})
    assert target_key(builtin) == target_key(FunctionCall(name="list_dir", arguments={}))


def test_restarted_server_with_changed_hints_refuses_call(project: Path, tmp_path: Path) -> None:
    marker = tmp_path / "destructive"
    config = _fake("--marker", str(marker))
    with closing(McpHub([McpServer(config, Workspace(project))])) as servers:
        servers.connect()
        tools, _ = servers.tools()
        echo = next(tool for tool in tools if tool.name == ECHO)
        ctx = ToolContext(Workspace(project))
        params = echo.parse({"text": "x"})
        assert echo.reads_only
        marker.touch()  # после перезапуска сервер пометит echo разрушительным
        with pytest.raises(McpError):
            servers.servers[0].call("crash", {})
        with pytest.raises(ToolError, match="изменился после перезапуска"):
            echo.run(params, ctx)
        assert echo.assess(params, ctx).risk is RiskLevel.DANGEROUS


def test_call_checks_hints_of_its_decision_when_another_call_restarted_server(
    project: Path,
    tmp_path: Path,
) -> None:
    marker = tmp_path / "destructive"
    config = _fake("--marker", str(marker))
    with closing(McpHub([McpServer(config, Workspace(project))])) as servers:
        servers.connect()
        tools, _ = servers.tools()
        echo = next(tool for tool in tools if tool.name == ECHO)
        ctx = ToolContext(Workspace(project))
        params = echo.parse({"text": "x"})
        echo.assess(params, ctx)  # разрешение — по пометке «только читает»
        marker.touch()  # после перезапуска сервер пометит echo разрушительным
        server = servers.servers[0]
        with pytest.raises(McpError):
            server.call("crash", {})
        server.call("echo", {"text": "y"})  # сервер перезапустил другой поток
        with pytest.raises(ToolError, match="изменился после перезапуска"):
            echo.run(params, ctx)
