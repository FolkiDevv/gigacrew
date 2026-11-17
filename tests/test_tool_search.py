"""Отложенные инструменты (MCP) и ``tool_search``: схемы — модели по мере надобности."""

from __future__ import annotations

from pathlib import Path
from typing import Any, override

import pytest
from fakes import RecordingEvents, ScriptedProvider, text_turn, tool_turn, tools_turn
from pydantic import BaseModel

from gigacrew.agent.conversation import Conversation
from gigacrew.agent.loop import Agent
from gigacrew.config import Config
from gigacrew.llm.types import FunctionCall, Message
from gigacrew.project.workspace import Workspace
from gigacrew.tools.base import Tool, ToolContext, ToolRegistry, ToolResult, build_default_registry
from gigacrew.tools.tool_search import TOOL_SEARCH, DeferredTools, ToolSearchTool, find_tools

ISSUES = "mcp__tracker__create_issue"
SEARCH = "mcp__tracker__search_issues"
DOCS = "mcp__docs__read_page"


class _Params(BaseModel):
    text: str


class _Remote(Tool[_Params]):
    """Отложенный инструмент «сервера»: записывает вызовы."""

    # Имя схемы параметров задано базовым классом Tool.
    Params = _Params  # noqa: WPS115
    deferred = True

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description
        self.calls: list[str] = []

    @override
    def run(self, params: _Params, ctx: ToolContext) -> ToolResult:
        self.calls.append(params.text)
        return ToolResult(f"ответ на {params.text}")


def _remotes() -> list[_Remote]:
    return [
        _Remote(ISSUES, "Create a new issue in the tracker"),
        _Remote(SEARCH, "Search issues by text"),
        _Remote(DOCS, "Read a documentation page"),
    ]


def _registry(*tools: Tool[Any]) -> ToolRegistry:
    registry = build_default_registry()
    for tool in tools:
        registry.register(tool)
    return registry


def _names(specs: list[Any]) -> set[str]:
    return {spec.name for spec in specs}


# ------------------------------- поиск ------------------------------- #
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (f"select:{DOCS}, {ISSUES},{DOCS}, нет_такого", [DOCS, ISSUES]),
        (SEARCH, [SEARCH]),
        ("issue", [ISSUES, SEARCH]),
        ("documentation", [DOCS]),
        ("+search issue", [SEARCH]),
        ("+jira issue", []),
        ("погода", []),
        ("a", []),
        ("iss", [ISSUES, SEARCH]),
    ],
)
def test_find_tools(query: str, expected: list[str]) -> None:
    found = find_tools(_remotes(), query, limit=5)
    assert [tool.name for tool in found] == expected


def test_find_tools_limit() -> None:
    assert len(find_tools(_remotes(), "tracker docs", limit=2)) == 2


# ------------------------- отложенные в реестре ------------------------- #
def test_specs_hide_deferred_until_loaded() -> None:
    deferred = DeferredTools(_registry(*_remotes()))
    names = _names(deferred.specs())
    assert TOOL_SEARCH in names
    assert "read_file" in names
    assert not names & {ISSUES, SEARCH, DOCS}
    assert deferred.pending(ISSUES)
    assert [tool.name for tool in deferred.search(f"select:{ISSUES}", 5)] == [ISSUES]
    assert ISSUES in _names(deferred.specs())
    # Загружен, но в запросе этого шага схемы не было.
    assert deferred.pending(ISSUES)
    assert ISSUES in _names(deferred.step_specs())
    assert not deferred.pending(ISSUES)
    assert not deferred.pending("read_file")


def test_nothing_deferred_hides_tool_search() -> None:
    assert TOOL_SEARCH not in _names(DeferredTools(build_default_registry()).specs())


def _assert_not_deferred(deferred: DeferredTools) -> None:
    assert {ISSUES, SEARCH, DOCS} <= _names(deferred.specs())
    assert TOOL_SEARCH not in _names(deferred.specs())
    assert not deferred.pending(ISSUES)
    assert not deferred.guidance()


def test_not_deferred_when_disabled() -> None:
    """GIGACREW_MCP_DEFER=0 — схемы всех инструментов сразу."""
    _assert_not_deferred(DeferredTools(_registry(*_remotes()), enabled=False))


def test_not_deferred_without_tool_search() -> None:
    """В реестре нет tool_search (суб-агенту он не дан) — загрузить нечем."""
    registry = ToolRegistry()
    for tool in _remotes():
        registry.register(tool)
    _assert_not_deferred(DeferredTools(registry))


def test_restore_from_history() -> None:
    deferred = DeferredTools(_registry(*_remotes()), {DOCS})
    messages = [
        Message(
            role="assistant",
            function_calls=[
                FunctionCall(name=TOOL_SEARCH, arguments={"query": f"select:{ISSUES}"}),
                FunctionCall(name=TOOL_SEARCH, arguments={"max_results": "много"}),
                FunctionCall(name=SEARCH, arguments={"text": "x"}),
                FunctionCall(name="read_file", arguments={"path": "a.py"}),
            ],
        ),
    ]
    deferred.restore(messages)
    assert deferred.loaded == {ISSUES, SEARCH}


def test_tool_search_tool_reports() -> None:
    deferred = DeferredTools(_registry(*_remotes()))
    tool = ToolSearchTool()
    ctx = ToolContext(Workspace(Path.cwd()), deferred=deferred)
    found = tool.run(tool.parse({"query": "issue"}), ctx)
    assert ISSUES in found.content
    assert "Search issues by text" in found.content
    assert deferred.loaded == {ISSUES, SEARCH}
    missing = tool.run(tool.parse({"query": "погода"}), ctx)
    assert "No deferred tool matches" in missing.content


# ------------------------------- агент ------------------------------- #
def _agent(
    project: Path,
    turns: list[Any],
    *tools: Tool[Any],
    conversation: Conversation | None = None,
    **cfg: Any,
) -> tuple[Agent, ScriptedProvider]:
    provider = ScriptedProvider(turns)
    config = Config(access_key="x", project_root=project, stream=False, **cfg)
    agent = Agent(provider, _registry(*tools), config, RecordingEvents(), conversation=conversation)
    return agent, provider


def test_agent_loads_schema_with_tool_search(project: Path) -> None:
    remote = _remotes()[0]
    turns = [
        tool_turn(TOOL_SEARCH, {"query": "create issue"}),
        tool_turn(ISSUES, {"text": "баг"}),
        text_turn("готово"),
    ]
    agent, provider = _agent(project, turns, remote)
    prompt = agent.system_prompt()
    assert "Deferred tools" in prompt
    assert ISSUES in prompt
    assert agent.run_turn("заведи задачу") == "готово"
    assert remote.calls == ["баг"]
    first, second, _ = (_names(request["tools"]) for request in provider.requests)
    assert ISSUES not in first
    assert ISSUES in second
    assert agent.loaded_tools == {ISSUES}
    # Загрузка не меняет системный промпт.
    assert agent.system_prompt() == prompt


def test_agent_refuses_call_without_schema(project: Path) -> None:
    remote = _remotes()[0]
    turns = [
        tool_turn(ISSUES, {"text": "догадка"}),
        tool_turn(ISSUES, {"text": "баг"}),
        text_turn("готово"),
    ]
    agent, provider = _agent(project, turns, remote)
    agent.run_turn("заведи задачу")
    assert remote.calls == ["баг"]
    note = next(
        message for message in provider.requests[1]["messages"] if message.role == "function"
    )
    assert "is deferred" in str(note.content)
    assert '"text"' in str(note.content)
    assert ISSUES in _names(provider.requests[1]["tools"])


@pytest.mark.parametrize(
    "first",
    [
        # tool_search и вызов одним ответом: модель писала аргументы, не видя схемы.
        (TOOL_SEARCH, {"query": f"select:{ISSUES}"}),
        # Два вызова незагруженного одним ответом: первый отклонён и загрузил инструмент.
        (ISSUES, {"text": "догадка 1"}),
    ],
)
def test_agent_refuses_call_loaded_in_same_response(
    project: Path,
    first: tuple[str, dict[str, Any]],
) -> None:
    remote = _remotes()[0]
    turns = [
        tools_turn([first, (ISSUES, {"text": "догадка"})]),
        tool_turn(ISSUES, {"text": "баг"}),
        text_turn("готово"),
    ]
    agent, _ = _agent(project, turns, remote)
    assert agent.run_turn("заведи задачу") == "готово"
    assert remote.calls == ["баг"]


def test_agent_without_defer_sends_all_schemas(project: Path) -> None:
    turns = [text_turn("готово")]
    agent, provider = _agent(project, turns, *_remotes(), mcp_defer=False)
    agent.run_turn("привет")
    names = _names(provider.requests[0]["tools"])
    assert {ISSUES, SEARCH, DOCS} <= names
    assert TOOL_SEARCH not in names
    assert "Deferred tools" not in agent.system_prompt()


def test_resumed_chat_restores_loaded(project: Path) -> None:
    conversation = Conversation()
    conversation.add_user("найди задачи")
    call = FunctionCall(name=TOOL_SEARCH, arguments={"query": f"select:{SEARCH}"}, id="1")
    conversation.add_assistant(Message(role="assistant", function_calls=[call]))
    conversation.add_function_result(TOOL_SEARCH, "Loaded", call_id="1")
    conversation.add_assistant(Message(role="assistant", content="нашёл"))
    agent, provider = _agent(project, [text_turn("ок")], *_remotes(), conversation=conversation)
    assert agent.loaded_tools == {SEARCH}
    agent.run_turn("ещё")
    assert SEARCH in _names(provider.requests[0]["tools"])
    agent.reset()
    assert not agent.loaded_tools
