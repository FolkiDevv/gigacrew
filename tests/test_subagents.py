"""Офлайн-тесты суб-агентов: инструмент task, изоляция контекста, разрешения, остановка,
ограничители. Родитель и суб-агент берут ходы из одной очереди ScriptedProvider —
по порядку обращений к модели."""

from __future__ import annotations

from pathlib import Path

import pytest
from fakes import (
    FakeTool,
    RecordingEvents,
    ScriptedProvider,
    cut_turn,
    text_turn,
    tool_turn,
    tools_turn,
)

import gigacrew.agent.loop as loop
from gigacrew.agent.events import TurnStats
from gigacrew.agent.guard import LoopGuard
from gigacrew.agent.prompts import SUBAGENT_PROMPTS
from gigacrew.agent.subagents import (
    NO_REPORT_NOTE,
    READ_ONLY_BLOCKED_NOTE,
    SUBAGENT_PLAN_BLOCKED_NOTE,
    SUBAGENT_USER_STOP_NOTE,
    SubagentTrail,
    subagent_report,
)
from gigacrew.config import Config
from gigacrew.llm.base import LLMError
from gigacrew.llm.model_windows import ModelWindows
from gigacrew.llm.types import FunctionCall, Usage
from gigacrew.permissions import PermissionMode
from gigacrew.tools.base import build_default_registry
from gigacrew.tools.task import (
    EXPLORE,
    SUBAGENT_EXCLUDED,
    SUBAGENTS,
    ExploreTool,
    allows_tool,
)


def _agent(provider, tmp_path: Path, *, events=None, registry=None, **cfg_kw) -> loop.Agent:
    cfg_kw.setdefault("auto_approve", True)
    cfg = Config(access_key="x", project_root=tmp_path, stream=False, **cfg_kw)
    return loop.Agent(
        provider, registry or build_default_registry(), cfg, events or RecordingEvents()
    )


def _task(agent: str = "explore", prompt: str = "найди функцию foo"):
    return tool_turn("task", {"agent": agent, "prompt": prompt, "description": "поиск foo"})


def _results(messages) -> list[str]:
    return [m.content for m in messages if m.role == "function"]


def _assert_well_formed(messages):
    """На каждый вызов — ровно один результат (то же имя и id), сразу после вызовов."""
    i = 0
    while i < len(messages):
        m = messages[i]
        assert m.role != "function", f"результат без вызова: {i}"
        i += 1
        calls = m.function_calls if m.role == "assistant" else []
        results = messages[i : i + len(calls)]
        assert [(r.role, r.name, r.call_id) for r in results] == [
            ("function", c.name, c.id) for c in calls
        ]
        i += len(calls)


def _kinds(events) -> list[str]:
    return [kind for kind, _ in events.events]


def _registry_with_fake(tool):
    reg = build_default_registry()
    reg.register(tool)
    return reg


class HookProvider(ScriptedProvider):
    """ScriptedProvider, вызывающий ``hooks[i]`` перед ответом на i-й запрос (с нуля)."""

    def __init__(self, turns, hooks):
        super().__init__(turns)
        self.hooks = hooks

    def complete(self, messages, tools=None, **kwargs):
        hook = self.hooks.get(len(self.requests))
        if hook is not None and tools is not None:
            self.requests.append({"messages": list(messages), "tools": tools, **kwargs})
            hook()
        return super().complete(messages, tools, **kwargs)


# ------------------------------- изоляция ------------------------------- #
def test_explore_runs_in_own_context(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
    provider = ScriptedProvider(
        [
            _task(),
            tool_turn("search_content", {"pattern": "def foo"}),
            text_turn("foo определена в a.py:1"),
            text_turn("Нашёл: a.py:1"),
        ]
    )
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events)
    assert agent.run_turn("где foo?") == "Нашёл: a.py:1"

    child = provider.requests[1]
    system, *history = child["messages"]
    assert "subagent" in system.content and "Agent explore" in system.content
    assert "ask_user" not in system.content
    assert [(m.role, m.content) for m in history] == [("user", "найди функцию foo")]
    names = {spec.name for spec in child["tools"]}
    assert names == set(EXPLORE.tools)
    assert not names & {"task", "ask_user", "exit_plan_mode", "write_file"}
    assert "run_shell" in names  # только команды, признанные «только чтение»

    # у основного агента — только отчёт, без шагов суб-агента
    messages = agent.conversation.messages
    results = _results(messages)
    assert len(results) == 1 and "foo определена в a.py:1" in results[0]
    assert all(m.name != "search_content" for m in messages)
    _assert_well_formed(messages)

    kinds = _kinds(events)
    start, end = kinds.index("subagent_start"), kinds.index("subagent_end")
    assert kinds.index("tool_start") < start < kinds.index("subagent_tool_call") < end
    assert end < kinds.index("tool_end")
    # текст суб-агента не печатается как ответ основного агента
    assert [p for k, p in events.events if k == "text"] == ["Нашёл: a.py:1"]
    assert events.results[-1][1].summary.startswith("explore · инструментов: 1")


def test_task_schema_and_registry():
    reg = build_default_registry()
    assert "task" in reg
    assert set(EXPLORE.tools) <= {tool.name for tool in reg}
    assert set(SUBAGENTS) == set(SUBAGENT_PROMPTS)
    assert SUBAGENT_EXCLUDED <= {tool.name for tool in reg}
    params = reg.get("task").spec().parameters
    assert params["required"] == ["subagent_type", "prompt"]


@pytest.mark.parametrize("key", ["subagent_type", "agent"])
def test_task_accepts_old_parameter_name(tmp_path, key):
    provider = ScriptedProvider(
        [
            tool_turn("task", {key: "Explore", "prompt": "где foo?"}),
            text_turn("в a.py"),
            text_turn("ок"),
        ]
    )
    events = RecordingEvents()
    _agent(provider, tmp_path, events=events).run_turn("x")
    info = next(p for k, p in events.events if k == "subagent_start")
    assert info.name == "explore"
    call = next(p for k, p in events.events if k == "tool_call")
    assert call.summary == "Explore · где foo?"


def test_subagent_cannot_spawn_subagents(tmp_path):
    provider = ScriptedProvider(
        [_task("general"), _task("explore"), text_turn("сам"), text_turn("ок")]
    )
    agent = _agent(provider, tmp_path)
    agent.run_turn("x")
    child_results = _results(provider.requests[2]["messages"])
    assert "не существует" in child_results[-1]


@pytest.mark.parametrize(
    ("args", "error"),
    [
        ({"agent": "nope", "prompt": "x"}, "неизвестный суб-агент"),
        ({"agent": "explore", "prompt": "  "}, "Пустая задача"),
    ],
)
def test_bad_task_arguments(tmp_path, args, error):
    provider = ScriptedProvider([tool_turn("task", args), text_turn("ок")])
    agent = _agent(provider, tmp_path)
    agent.run_turn("x")
    assert error in _results(agent.conversation.messages)[0]
    assert provider.calls == 2  # суб-агент не запускался


def test_empty_description_uses_prompt(tmp_path):
    provider = ScriptedProvider(
        [
            tool_turn("task", {"agent": "explore", "prompt": "\nпосчитай файлы\nподробно"}),
            text_turn("3"),
            text_turn("ок"),
        ]
    )
    events = RecordingEvents()
    _agent(provider, tmp_path, events=events).run_turn("x")
    info = next(p for k, p in events.events if k == "subagent_start")
    assert info.description == "посчитай файлы"
    call = next(p for k, p in events.events if k == "tool_call")
    assert call.summary == "explore · посчитай файлы"


def test_subagent_uses_current_model(tmp_path):
    provider = ScriptedProvider([_task(), text_turn("r"), text_turn("ок")])
    agent = _agent(provider, tmp_path)
    agent.set_model("Other-Model")
    agent.run_turn("x")
    assert provider.requests[1]["model"] == "Other-Model"


# ------------------------------ разрешения ------------------------------ #
def test_explore_cannot_change_even_with_yes_all(tmp_path):
    provider = ScriptedProvider(
        [
            _task(),
            tool_turn("git", {"subcommand": "commit", "args": ["-m", "x"]}),
            text_turn("не могу"),
            text_turn("ок"),
        ]
    )
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events, yes_all=True)
    agent.run_turn("x")
    assert events.confirms == []
    child_result = _results(provider.requests[2]["messages"])[-1]
    assert child_result == READ_ONLY_BLOCKED_NOTE
    blocked = [r for k, (c, r) in _sub_results(events)]
    assert blocked[0].summary == "заблокировано: агент только читает"


def test_explore_unsupported_git_subcommand_is_not_called_a_change(tmp_path):
    """git blame через инструмент git: подсказка про run_shell, а не «изменения запрещены»."""
    provider = ScriptedProvider(
        [
            _task(),
            tool_turn("git", {"subcommand": "blame", "args": ["a.py"]}),
            text_turn("итог"),
            text_turn("ок"),
        ]
    )
    agent = _agent(provider, tmp_path)
    agent.run_turn("x")
    child_result = _results(provider.requests[2]["messages"])[-1]
    assert child_result != READ_ONLY_BLOCKED_NOTE
    assert "не поддерживается" in child_result and "run_shell" in child_result


@pytest.mark.parametrize("sub", ["push", "merge"])
def test_explore_changing_git_subcommand_is_still_a_blocked_change(tmp_path, sub):
    provider = ScriptedProvider(
        [_task(), tool_turn("git", {"subcommand": sub}), text_turn("итог"), text_turn("ок")]
    )
    agent = _agent(provider, tmp_path)
    agent.run_turn("x")
    assert _results(provider.requests[2]["messages"])[-1] == READ_ONLY_BLOCKED_NOTE


def _sub_results(events):
    return [(k, p) for k, p in events.events if k == "subagent_tool_result"]


def test_general_edits_without_asking_in_auto_mode(tmp_path):
    provider = ScriptedProvider(
        [
            _task("general", "создай b.txt"),
            tool_turn("write_file", {"path": "b.txt", "content": "new\n"}),
            text_turn("создал b.txt"),
            text_turn("ок"),
        ]
    )
    events = RecordingEvents(confirm_answer=False)
    agent = _agent(provider, tmp_path, events=events, auto_approve=False)
    agent.run_turn("x")
    assert events.confirms == [] and agent.mode is PermissionMode.AUTO
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "new\n"
    report = _results(agent.conversation.messages)[0]
    assert "создал b.txt" in report and "Changed files: b.txt" in report
    assert events.results[-1][1].changed


def test_delegation_rules_only_for_main_agent():
    """Как ставить задачи суб-агентам — знает основной агент; суб-агент их не ставит."""
    from gigacrew.agent.prompts import SUBAGENT_PROMPT, SYSTEM_PROMPT

    assert "# Research and subagents" in SYSTEM_PROMPT
    assert "# Research and subagents" not in SUBAGENT_PROMPT


def test_research_goes_to_explore_first():
    """Исследование — через explore без просьбы пользователя: правило стоит до «How to
    work», с примерами; точечный поиск — самому; в режиме плана — explore первым шагом."""
    from gigacrew.agent.prompts import PLAN_MODE_PROMPT, SYSTEM_PROMPT

    research = SYSTEM_PROMPT.index("# Research and subagents")
    assert research < SYSTEM_PROMPT.index("# How to work")
    rules = SYSTEM_PROMPT[research : SYSTEM_PROMPT.index("# How to work")]
    assert "IMPORTANT: to research the code, call explore instead of searching yourself" in rules
    assert "pinpoint lookup" in rules and rules.count('\n- "') == 3  # три примера
    assert "Use it instead of searching yourself" in ExploreTool.description
    step = PLAN_MODE_PROMPT[PLAN_MODE_PROMPT.index("1. ") : PLAN_MODE_PROMPT.index("2. ")]
    assert "start with explore: 1–3 calls in one response" in step


def test_explore_tool_comes_first_and_is_not_for_subagents():
    reg = build_default_registry()
    assert next(iter(reg)).name == "explore"
    spec = reg.get("explore").spec()
    assert spec.parameters["required"] == ["question"]
    assert "explore" in SUBAGENT_EXCLUDED
    for subagent in SUBAGENTS.values():
        assert not allows_tool(subagent, "explore")


def test_explore_tool_runs_read_only_subagent(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    question = "Где определена foo\nи что она возвращает?"
    provider = ScriptedProvider(
        [
            tool_turn("explore", {"question": question, "hints": "смотри a.py"}),
            tool_turn("write_file", {"path": "b.py", "content": "x"}),  # заблокировано
            text_turn("foo в a.py:1, возвращает 1"),
            text_turn("foo возвращает 1"),
        ]
    )
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events, auto_approve=True)
    assert agent.run_turn("что возвращает foo?") == "foo возвращает 1"
    brief = provider.requests[1]["messages"][1].content
    assert brief.startswith(question) and "Already known / out of scope: смотри a.py" in brief
    assert not (tmp_path / "b.py").exists()  # суб-агент только читает
    info = next(p for k, p in events.events if k == "subagent_start")
    assert info.name == "explore" and info.description == "Где определена foo"
    call = next(p for k, p in events.events if k == "tool_call")
    assert call.summary == "Где определена foo и что она возвращает?"
    report = _results(agent.conversation.messages)[0]
    assert "foo в a.py:1" in report


def test_explore_tool_rejects_empty_question(tmp_path):
    provider = ScriptedProvider([tool_turn("explore", {"question": "  "}), text_turn("ок")])
    agent = _agent(provider, tmp_path)
    agent.run_turn("x")
    assert "Пустой вопрос" in _results(agent.conversation.messages)[0]
    assert provider.calls == 2  # суб-агент не запускался


def test_limit_stop_report_tells_not_to_rerun_the_same_task():
    stats = TurnStats(tool_calls=25, stop_reason="max_steps")
    result = subagent_report(EXPLORE, "частичный итог", stats, SubagentTrail())
    assert "the step limit is reached" in result.content
    assert "Do not rerun the same task" in result.content
    assert "частичный итог" in result.content


def test_report_without_text_is_marked_for_user():
    """Суб-агент остановлен и не дал итога: модель узнаёт об этом, пользователь — тоже."""
    stats = TurnStats(tool_calls=25, stop_reason="max_steps")
    result = subagent_report(EXPLORE, "", stats, SubagentTrail())
    assert not result.ok
    assert NO_REPORT_NOTE.format(name="explore") in result.content
    assert result.summary.endswith("исчерпан лимит шагов · без отчёта")

    result = subagent_report(EXPLORE, "итог", stats, SubagentTrail())
    assert result.ok and "без отчёта" not in result.summary


def test_rejected_operations_listed_in_report(tmp_path):
    provider = ScriptedProvider(
        [
            _task("general", "создай b.txt"),
            tool_turn("run_shell", {"command": "pip install requests"}),
            text_turn("не дали"),
            text_turn("ок"),
        ]
    )
    agent = _agent(provider, tmp_path, events=RecordingEvents(False), auto_approve=False)
    agent.run_turn("x")
    report = _results(agent.conversation.messages)[0]
    assert "Rejected by the user: run_shell pip install requests" in report


def test_parent_mode_change_reaches_running_subagent(tmp_path):
    holder: dict = {}

    def switch():
        holder["agent"].set_mode(PermissionMode.PLAN)
        from gigacrew.tools.base import ToolResult

        return ToolResult(content="ok", summary="ok")

    tool = FakeTool(run=switch)
    provider = ScriptedProvider(
        [
            _task("general", "сделай"),
            tool_turn("fake_write", {"path": "1"}),
            tool_turn("fake_write", {"path": "2"}),
            text_turn("готово"),
            text_turn("ок"),
        ]
    )
    agent = _agent(provider, tmp_path, registry=_registry_with_fake(tool))
    holder["agent"] = agent
    agent.run_turn("x")
    assert tool.runs == 1  # второй вызов заблокирован режимом плана
    last = provider.requests[3]["messages"]
    assert _results(last)[-1] == SUBAGENT_PLAN_BLOCKED_NOTE
    assert "# Plan mode" in last[0].content and "exit_plan_mode" not in last[0].content


def test_explore_gets_no_plan_mode_rules(tmp_path):
    """Explore и так только читает: правила режима плана обещали бы ему тесты с
    подтверждением, а у него они заблокированы."""
    provider = ScriptedProvider(
        [_task("explore"), text_turn("итог"), text_turn("ок"), text_turn("OK")]
    )
    agent = _agent(provider, tmp_path, mode=PermissionMode.PLAN)
    agent.run_turn("x")
    child_system = provider.requests[1]["messages"][0].content
    assert "Agent explore" in child_system and "# Plan mode" not in child_system
    assert "Plan mode is on" not in provider.requests[1]["messages"][-1].content


def test_subagent_answer_in_plan_mode_gets_no_self_check(tmp_path):
    provider = ScriptedProvider(
        [
            _task("explore", "где foo?"),
            text_turn("foo — в a.py"),  # ответ суб-агента: самопроверки нет
            text_turn("foo лежит в a.py"),  # ответ основного агента
            text_turn("OK"),  # его самопроверка
        ]
    )
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events, mode=PermissionMode.PLAN)
    assert agent.run_turn("где foo?") == "foo лежит в a.py"
    assert provider.calls == 4 and events.stats[-1].self_checks == 1
    after_subagent = provider.requests[2]["messages"]
    assert after_subagent[-1].role == "function"  # отчёт суб-агента, не самопроверка
    assert provider.requests[3]["messages"][-1].meta


class _WaitingProvider(ScriptedProvider):
    """Перед каждым ответом провайдер «ждёт» лимита запросов и сообщает об этом."""

    def complete(self, messages, tools=None, **kwargs):
        on_notice = kwargs.get("on_notice")
        if on_notice is not None:
            on_notice("лимит запросов шлюза (429) — жду 5 с и повторяю")
        return super().complete(messages, tools, **kwargs)


@pytest.mark.parametrize("stream", [False, True], ids=["complete", "stream"])
def test_rate_limit_waits_are_shown_to_the_user(tmp_path, stream):
    provider = _WaitingProvider([_task(), text_turn("нашёл"), text_turn("ок")])
    events = RecordingEvents()
    cfg = Config(access_key="x", project_root=tmp_path, stream=stream, auto_approve=True)
    loop.Agent(provider, build_default_registry(), cfg, events).run_turn("x")
    waits = [(level, text) for level, text in events.notices if "(429)" in text]
    assert [level for level, _ in waits] == ["warn"] * 3
    assert waits[1][1].startswith("explore: ")  # пауза в запросе суб-агента


# ------------------------------ токены, сбои ----------------------------- #
def test_subagent_tokens_billed_to_parent(tmp_path):
    provider = ScriptedProvider(
        [
            tool_turn(
                "task",
                {"agent": "explore", "prompt": "x"},
                Usage(prompt_tokens=7_000, completion_tokens=10),
            ),
            text_turn("r", Usage(prompt_tokens=5_000, completion_tokens=5)),
            text_turn("ок", Usage(prompt_tokens=7_200, completion_tokens=2)),
        ]
    )
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events)
    agent.run_turn("x")
    assert agent.billed_tokens == 19_217
    stats = events.stats[-1]
    assert (stats.prompt_tokens, stats.completion_tokens) == (19_200, 17)
    assert stats.context_tokens == 7_202  # контекст — только основного агента


def test_ctrl_c_in_subagent_stops_turn_and_keeps_trail(tmp_path):
    provider = ScriptedProvider(
        [
            _task("general", "создай b.txt"),
            tool_turn("write_file", {"path": "b.txt", "content": "new\n"}),
            KeyboardInterrupt(),
        ]
    )
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events)
    with pytest.raises(KeyboardInterrupt):
        agent.run_turn("x")
    messages = agent.conversation.messages
    _assert_well_formed(messages)
    assert agent.conversation.pending_calls() == []
    note = _results(messages)[-1]
    assert "interrupted by the user (Ctrl+C)" in note and "Changed files: b.txt" in note
    kinds = _kinds(events)
    assert "subagent_end" in kinds and kinds[-1] == "tool_end"


@pytest.mark.parametrize(
    "call",
    [
        tool_turn("explore", {"question": "где foo?"}),
        _task("explore", "где foo?"),
        tools_turn([("explore", {"question": "где foo?"}), ("explore", {"question": "а bar?"})]),
    ],
)
def test_ctrl_c_in_subagent_answers_the_call_that_started_it(tmp_path, call):
    # Результат — под именем и id того вызова (explore или task), что запустил суб-агента.
    (tmp_path / "a.py").write_text("def foo(): ...\n")
    provider = ScriptedProvider(
        [call, tool_turn("read_file", {"path": "a.py"}), KeyboardInterrupt()]
    )
    agent = _agent(provider, tmp_path)
    with pytest.raises(KeyboardInterrupt):
        agent.run_turn("x")
    messages = agent.conversation.messages
    _assert_well_formed(messages)
    assert agent.conversation.pending_calls() == []
    assert "interrupted by the user (Ctrl+C)" in _results(messages)[0]


def test_subagent_decisions_are_journaled_under_the_chat(tmp_path):
    import json

    from gigacrew.agent.decisions import DecisionMemory

    script = tmp_path / "build.sh"
    script.write_text("#!/bin/sh\necho built\n")
    script.chmod(0o755)
    memory, _ = DecisionMemory.load(tmp_path, base=tmp_path / "home")
    provider = ScriptedProvider(
        [
            _task("general", "собери"),
            tool_turn("run_shell", {"command": "./build.sh"}),
            text_turn("собрано"),
            text_turn("готово"),
        ]
    )
    cfg = Config(access_key="x", project_root=tmp_path, stream=False, yes_all=True)
    agent = loop.Agent(provider, build_default_registry(), cfg, RecordingEvents(), decisions=memory)
    agent.chat_id = "chat-1"
    agent.run_turn("собери проект")
    [entry] = [
        json.loads(line)
        for line in memory.journal_path.read_text().splitlines()
        if '"event": "decision"' in line
    ]
    assert (entry["agent"], entry["chat"]) == ("general", "chat-1")


def test_llm_error_in_subagent_reported_to_model(tmp_path):
    provider = ScriptedProvider([_task(), LLMError("сбой сети"), text_turn("сделаю сам")])
    agent = _agent(provider, tmp_path)
    assert agent.run_turn("x") == "сделаю сам"
    result = _results(agent.conversation.messages)[0]
    assert "Subagent explore: model error: сбой сети" in result
    _assert_well_formed(agent.conversation.messages)


def test_other_kind_changes_kept_in_trail_on_ctrl_c(tmp_path):
    # изменяющий вызов не-EDIT вида (как git commit) — тоже в журнале суб-агента
    provider = ScriptedProvider(
        [_task("general", "закоммить"), tool_turn("fake_write", {"path": "x"}), KeyboardInterrupt()]
    )
    agent = _agent(provider, tmp_path, registry=_registry_with_fake(FakeTool()))
    with pytest.raises(KeyboardInterrupt):
        agent.run_turn("x")
    assert "Changing operations: fake_write x" in _results(agent.conversation.messages)[-1]


def test_llm_error_after_changes_keeps_trail(tmp_path):
    provider = ScriptedProvider(
        [
            _task("general", "создай b.txt"),
            tool_turn("write_file", {"path": "b.txt", "content": "x\n"}),
            LLMError("сбой сети"),
            text_turn("продолжаю"),
        ]
    )
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events)
    assert agent.run_turn("x") == "продолжаю"
    result = _results(agent.conversation.messages)[0]
    assert "model error: сбой сети" in result and "Changed files: b.txt" in result
    task_result = events.results[-1][1]
    assert not task_result.ok and task_result.changed


def test_parent_repeat_guard_sees_subagent_changes(tmp_path):
    shell = tool_turn("run_shell", {"command": "echo hi"})
    provider = ScriptedProvider(
        [
            shell,
            _task("general", "создай b.txt"),
            tool_turn("write_file", {"path": "b.txt", "content": "x\n"}),
            text_turn("создал"),
            shell,
            shell,
            shell,
            text_turn("ок"),
        ]
    )
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events)
    assert agent.run_turn("x") == "ок"
    assert events.stats[-1].stop_reason is None


# ------------------------------ остановка ------------------------------- #
def _stopping_tool(holder: dict, times: int = 1) -> FakeTool:
    def stop():
        for _ in range(times):
            assert holder["agent"].request_subagent_stop()
        raise KeyboardInterrupt

    return FakeTool(run=stop)


def test_request_stop_without_subagent(tmp_path):
    assert _agent(ScriptedProvider(), tmp_path).request_subagent_stop() is False


def test_esc_soft_stop_wraps_up_and_parent_continues(tmp_path):
    holder: dict = {}
    provider = ScriptedProvider(
        [
            _task("general", "долгая задача"),
            tool_turn("fake_write", {}),
            text_turn("успел половину"),
            text_turn("основной продолжает"),
        ]
    )
    events = RecordingEvents()
    agent = _agent(
        provider, tmp_path, events=events, registry=_registry_with_fake(_stopping_tool(holder))
    )
    holder["agent"] = agent
    assert agent.run_turn("x") == "основной продолжает"
    wrap_request = provider.requests[2]["messages"]
    # просьба об итоге — заметкой к последнему сообщению, а не новым сообщением
    # пользователя: задача суб-агента остаётся закреплённой
    assert SUBAGENT_USER_STOP_NOTE in wrap_request[-1].content
    assert [m.role for m in wrap_request[1:]] == ["user", "assistant", "function"]
    assert wrap_request[1].content == "долгая задача"
    assert provider.requests[2]["tool_choice"] == "none"
    report = _results(agent.conversation.messages)[0]
    assert "stopped early by the user" in report and "успел половину" in report
    assert ("subagent_activity", ("подвожу итог", True)) in events.events
    assert events.results[-1][1].summary.endswith("остановлен пользователем")


class EscWhileStreaming(ScriptedProvider):
    """Поток ``i``-го запроса: кусок текста, затем Esc пользователя (мягкая остановка)."""

    def __init__(self, turns, *, at: int, holder: dict):
        super().__init__(turns)
        self.at = at
        self.holder = holder

    def stream(self, messages, tools=None, *, on_delta=None, **kwargs):
        if len(self.requests) == self.at:
            self.requests.append({"messages": list(messages), "tools": tools, **kwargs})
            on_delta("Пока нашёл, что foo")
            assert self.holder["agent"].request_subagent_stop()
            raise KeyboardInterrupt
        return super().stream(messages, tools, on_delta=on_delta, **kwargs)


def test_esc_while_subagent_streams_text_wraps_up_without_partial(tmp_path):
    """Esc посреди текста суб-агента: оборванный текст не сохраняется, запрос итога не
    кончается репликой ассистента и содержит задачу."""
    holder: dict = {}
    provider = EscWhileStreaming(
        [_task("explore", "найди foo"), text_turn("итог: foo в a.py"), text_turn("дальше")],
        at=1,
        holder=holder,
    )
    cfg = Config(access_key="x", project_root=tmp_path, stream=True, auto_approve=True)
    agent = loop.Agent(provider, build_default_registry(), cfg, RecordingEvents())
    holder["agent"] = agent
    assert agent.run_turn("x") == "дальше"

    wrap_request = provider.requests[2]["messages"]
    assert [m.role for m in wrap_request[1:]] == ["user"]
    assert wrap_request[-1].content.startswith("найди foo\n\n<system-reminder>")
    assert SUBAGENT_USER_STOP_NOTE in wrap_request[-1].content
    assert provider.requests[2]["tool_choice"] == "none"
    report = _results(agent.conversation.messages)[0]
    assert "stopped early by the user" in report and "итог: foo в a.py" in report


def test_double_esc_aborts_subagent_parent_continues(tmp_path):
    holder: dict = {}
    provider = ScriptedProvider(
        [_task("general", "долгая задача"), tool_turn("fake_write", {}), text_turn("дальше")]
    )
    agent = _agent(
        provider, tmp_path, registry=_registry_with_fake(_stopping_tool(holder, times=2))
    )
    holder["agent"] = agent
    assert agent.run_turn("x") == "дальше"
    result = _results(agent.conversation.messages)[0]
    assert "aborted by the user" in result
    _assert_well_formed(agent.conversation.messages)


def test_esc_during_wrap_up_aborts(tmp_path):
    holder: dict = {}

    def second_esc():
        assert holder["agent"].request_subagent_stop()
        raise KeyboardInterrupt

    provider = HookProvider(
        [_task("general", "долгая задача"), tool_turn("fake_write", {}), text_turn("дальше")],
        hooks={2: second_esc},
    )
    agent = _agent(provider, tmp_path, registry=_registry_with_fake(_stopping_tool(holder)))
    holder["agent"] = agent
    assert agent.run_turn("x") == "дальше"
    assert "aborted by the user" in _results(agent.conversation.messages)[0]


# ------------------------------ ограничители ---------------------------- #
def test_subagent_step_limit_wraps_up(tmp_path):
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    provider = ScriptedProvider(
        [
            _task(),
            tool_turn("read_file", {"path": "a.txt"}),
            tool_turn("list_dir", {}),
            text_turn("частичный итог"),
            text_turn("ок"),
        ]
    )
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events, subagent_max_steps=2)
    assert agent.run_turn("x") == "ок"
    wrap_up = provider.requests[3]
    # просьба об итоге — заметкой в хвосте запроса, вызовы запрещены
    assert "STOP: the step limit is reached" in wrap_up["messages"][-1].content
    assert wrap_up["tool_choice"] == "none"
    report = _results(agent.conversation.messages)[0]
    assert "the step limit is reached" in report and "частичный итог" in report
    assert any(level == "warn" and text.startswith("explore: ") for level, text in events.notices)
    assert events.stats[-1].stop_reason is None  # основной агент не остановлен


def _context_case(tmp_path, prompt_tokens: int, *, windows=None, **cfg_kw):
    """Суб-агент: вызов инструмента с контекстом ``prompt_tokens``, затем ещё вызов."""
    provider = ScriptedProvider(
        [
            _task(),
            tool_turn("list_dir", {}, Usage(prompt_tokens=prompt_tokens)),
            tool_turn("find_files", {"pattern": "*"}),
            text_turn("итог"),
            text_turn("ок"),
        ]
    )
    cfg = Config(access_key="x", project_root=tmp_path, stream=False, model="M", **cfg_kw)
    agent = loop.Agent(provider, build_default_registry(), cfg, RecordingEvents(), windows=windows)
    agent.run_turn("x")
    return provider, _results(agent.conversation.messages)[0]


def test_subagent_stops_at_context_share_of_unmeasured_window(tmp_path):
    # окно не замерено — 32k по умолчанию: 80% = 25 600, но бюджет запроса меньше — 21 600
    provider, report = _context_case(tmp_path, 26_000)
    assert "the context is full" in report
    assert "STOP: the context is full" in provider.requests[2]["messages"][-1].content
    assert provider.summary_requests == []  # суб-агент не сжимает историю

    _, report = _context_case(tmp_path, 20_000)
    assert "the context is full" not in report


def test_subagent_context_limit_follows_measured_window_and_share(tmp_path):
    windows = ModelWindows(entries={"M": {"context_window": 200_000}})
    _, report = _context_case(tmp_path, 26_000, windows=windows)
    assert "the context is full" not in report  # 13% окна в 200k

    _, report = _context_case(tmp_path, 26_000, windows=windows, subagent_context_share=0.1)
    assert "the context is full" in report  # предел 20 000

    windows = ModelWindows(entries={"M": {"context_window": 30_000}})
    _, report = _context_case(tmp_path, 22_000, windows=windows)  # 73% окна
    assert "the context is full" in report  # но больше бюджета запроса (19 800)


def test_subagent_context_counts_tool_results_after_last_reply(tmp_path):
    """Большой результат инструмента — контекст следующего запроса, хоть ответ был мал."""
    line = "строка с текстом, который занимает место в контексте модели"
    (tmp_path / "big.txt").write_text(f"{line}\n" * 900, encoding="utf-8")  # ~18k токенов
    provider = ScriptedProvider(
        [
            _task(),
            tool_turn("read_file", {"path": "big.txt"}),
            tool_turn("list_dir", {}),
            text_turn("итог"),
            text_turn("ок"),
        ]
    )
    windows = ModelWindows(entries={"M": {"context_window": 20_000}})  # предел 10 800
    cfg = Config(access_key="x", project_root=tmp_path, stream=False, model="M")
    agent = loop.Agent(provider, build_default_registry(), cfg, RecordingEvents(), windows=windows)
    agent.run_turn("x")
    assert "the context is full" in _results(agent.conversation.messages)[0]

    # Запрос итога: история больше бюджета и обрезается, но задача суб-агента и его
    # последние результаты в нём остаются — иначе отчёт не о чем писать.
    wrap_up = provider.requests[2]["messages"]
    assert wrap_up[1].role == "user" and wrap_up[1].content == "найди функцию foo"
    assert "строка с текстом" in wrap_up[-1].content
    assert "STOP: the context is full" in wrap_up[-1].content
    assert provider.requests[2]["tool_choice"] == "none"


def test_cut_reply_does_not_fill_subagent_context(tmp_path):
    """Оборванный длинный ответ не идёт в историю — и в счётчик контекста суб-агента тоже:
    иначе повтор ложно упёрся бы в предел контекста."""
    provider = ScriptedProvider(
        [
            _task(),
            cut_turn("длинный отчёт…", usage=Usage(prompt_tokens=1_000, completion_tokens=50_000)),
            text_turn("итог: foo в a.py"),
            text_turn("ок"),
        ]
    )
    agent = _agent(provider, tmp_path)
    assert agent.run_turn("x") == "ок"
    report = _results(agent.conversation.messages)[0]
    assert "итог: foo в a.py" in report and "the context is full" not in report


def test_subagent_time_limit(tmp_path):
    provider = ScriptedProvider([_task(), text_turn("итог"), text_turn("ок")])
    agent = _agent(provider, tmp_path, subagent_timeout=0)
    agent.run_turn("x")
    assert "the time limit is reached" in _results(agent.conversation.messages)[0]


def test_subagent_similar_calls_warned_then_stopped(tmp_path):
    (tmp_path / "a.txt").write_text("\n".join(map(str, range(50))), encoding="utf-8")
    reads = [tool_turn("read_file", {"path": "a.txt", "start_line": i}) for i in range(1, 8)]
    provider = ScriptedProvider([_task(), *reads, text_turn("итог"), text_turn("ок")])
    agent = _agent(provider, tmp_path)
    agent.run_turn("x")
    child = _results(provider.requests[-2]["messages"])
    assert len(child) == 7  # 6 выполнено, 7-й — заглушка остановки
    assert "going in circles" in child[4]
    assert "similar calls" in _results(agent.conversation.messages)[0]


def test_explore_blocked_commands_are_warned_then_stopped(tmp_path):
    """Explore перебирает отклонённые команды: предупреждение перед остановкой, затем стоп."""
    shells = [tool_turn("run_shell", {"command": f"frobnicate {i}"}) for i in range(4)]
    provider = ScriptedProvider([_task(), *shells, text_turn("итог"), text_turn("ок")])
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events)
    agent.run_turn("x")
    # запросы суб-агента после 1–3-го вызова и итог после остановки на 4-м
    child = [r["messages"][-1].content for r in provider.requests[2:6]]
    assert ["the last 3 tool calls failed" in c for c in child] == [False, False, True, False]
    assert provider.requests[5]["tool_choice"] == "none"
    assert all(c.startswith("Not executed: here run_shell runs only") for c in child[:2])
    report = _results(agent.conversation.messages)[0]
    assert "too many failed tool calls in a row" in report and "итог" in report
    notices = [text for _, text in events.notices]
    assert any("ещё одна неудача остановит ход" in text for text in notices)


def test_subagent_budget_pressure_note(tmp_path):
    provider = ScriptedProvider(
        [
            _task(),
            tool_turn("list_dir", {}),
            tool_turn("find_files", {"pattern": "*"}),
            text_turn("итог"),
            text_turn("ок"),
        ]
    )
    agent = _agent(provider, tmp_path, subagent_max_steps=4)
    agent.run_turn("x")
    assert "# Budget" not in provider.requests[1]["messages"][-1].content
    assert "# Budget" in provider.requests[3]["messages"][-1].content  # шаг 3 из 4
    assert "# Budget" not in provider.requests[3]["messages"][0].content  # не в системном
    assert "# Budget" not in provider.requests[-1]["messages"][-1].content  # основной


def test_subagent_launches_per_turn_limited(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "MAX_SUBAGENTS_PER_TURN", 1)
    provider = ScriptedProvider([_task(), text_turn("r1"), _task(prompt="другое"), text_turn("ок")])
    agent = _agent(provider, tmp_path)
    agent.run_turn("x")
    assert "лимит суб-агентов в этом ходе исчерпан" in _results(agent.conversation.messages)[1]


# ------------------------------- LoopGuard ------------------------------ #
def test_guard_new_limits_off_by_default():
    guard = LoopGuard(max_steps=10, max_failures=3)
    assert guard.before_step(10**9) is None
    assert guard.pressure(10**9) is None
    call = FunctionCall(name="read_file", arguments={"path": "a"})
    for i in range(3):
        call = FunctionCall(name="read_file", arguments={"path": "a", "start_line": i})
        assert guard.before_tool(call).warning is None


def test_guard_context_limit_and_pressure():
    guard = LoopGuard(max_steps=100, max_failures=3, max_context=1_000)
    assert guard.before_step(600) is None and guard.pressure(600) is None
    assert "context ~700 of 1000 tokens" in guard.pressure(700)
    stop = guard.before_step(1_000)
    assert stop.kind == "context_limit" and "Контекст заполнен" in stop.message


def test_guard_time_limit_and_pressure_with_fake_clock():
    now = [0.0]
    guard = LoopGuard(max_steps=100, max_failures=3, time_limit=10, clock=lambda: now[0])
    assert guard.before_step() is None and guard.pressure() is None
    now[0] = 7.5
    assert "time 7 of 10 s" in guard.pressure()
    now[0] = 10
    assert guard.before_step().kind == "time_limit"


def test_guard_similar_reset_by_own_changes():
    from gigacrew.agent.guard import ToolOutcome

    guard = LoopGuard(max_steps=100, max_failures=3, max_similar=2, max_context=10**6)
    for i in range(5):  # правки одного файла подряд — не топтание
        call = FunctionCall(name="edit_file", arguments={"path": "a", "old_string": str(i)})
        check = guard.before_tool(call)
        assert check.warning is None and check.stop is None
        guard.after_tool(call, ToolOutcome(ok=True, changed=True))
    assert guard.changes == 5
