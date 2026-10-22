"""Офлайн-тесты агентного цикла (без сети): поток, подтверждения, анти-залипание."""

from __future__ import annotations

import json
import os
import sys
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

from gigacrew.agent.context_window import estimate_text_tokens
from gigacrew.agent.conversation import INTERRUPTED_NOTE
from gigacrew.agent.loop import (
    LOOP_STOP_NOTE,
    PLAN_BLOCKED_NOTE,
    PLAN_REMINDER_NOTE,
    REJECTED_AGAIN_NOTE,
    REJECTED_NOTE,
    SELF_CHECK_NOTE,
    SKIPPED_NOTE,
    Agent,
)
from gigacrew.config import Config
from gigacrew.errors import ToolError
from gigacrew.llm.base import LLMError
from gigacrew.llm.types import FunctionCall, Message, Usage
from gigacrew.permissions import PermissionMode, ToolKind
from gigacrew.tools.base import Display, ToolRegistry, ToolResult, build_default_registry
from gigacrew.tools.plan import APPROVE
from gigacrew.tools.questions import Answer, QuestionsUnavailable
from gigacrew.tools.task import EXPLORE


def _agent(provider, tmp_path: Path, *, events=None, registry=None, **cfg_kw) -> Agent:
    cfg_kw.setdefault("auto_approve", True)
    cfg = Config(access_key="x", project_root=tmp_path, stream=False, **cfg_kw)
    return Agent(
        provider,
        registry or build_default_registry(),
        cfg,
        events or RecordingEvents(),
    )


def _registry_with(tool) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(tool)
    return reg


def test_loop_stops_after_repeated_failures(tmp_path):
    # модель упорно вызывает read_file на несуществующих файлах (разных — иначе
    # первым сработает детектор повторов)
    bad = [tool_turn("read_file", {"path": f"nope{i}.txt"}) for i in range(20)]
    provider = ScriptedProvider(bad)
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events, max_tool_failures=4)
    final = agent.run_turn("сделай что-нибудь")
    # итог модели — вызов инструмента, а не текст: остаётся сообщение ограничителя
    assert "Прервано" in final
    # должно остановиться на пороге, а не крутить 50 шагов: 4 шага + запрос итога
    assert provider.calls == 5
    # просьба об итоге — заметкой в хвосте запроса, а не в системном сообщении
    assert "# Turn stopped" in provider.requests[-1]["messages"][-1].content
    assert "# Turn stopped" not in provider.requests[-1]["messages"][0].content
    assert provider.requests[-1]["tool_choice"] == "none"  # итог — только текстом
    assert len(_function_results(agent)) == 4  # вызов из запроса итога не выполнялся
    assert events.stats[-1].stop_reason == "tool_failures"
    assert events.notices and events.notices[-1][0] == "error"


def _function_results(agent) -> list[str]:
    return [m.content for m in agent.conversation.messages if m.role == "function"]


def test_loop_stops_on_repeated_identical_calls(tmp_path):
    (tmp_path / "f.txt").write_text("hello", encoding="utf-8")
    provider = ScriptedProvider([tool_turn("read_file", {"path": "f.txt"})] * 10)
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events)
    final = agent.run_turn("прочитай")
    assert "зациклился" in final
    assert provider.calls == 5  # 4 шага + запрос итога (его вызов не выполняется)
    assert events.stats[-1].stop_reason == "tool_repeats"
    assert events.stats[-1].tool_calls == 3  # 4-й вызов не выполнялся
    assert [level for level, _ in events.notices] == ["warn", "error"]
    results = _function_results(agent)
    assert len(results) == 4
    assert ["WARNING" in r for r in results] == [False, False, True, False]
    assert "hello" in results[2]  # предупреждение дописано к результату
    assert results[3] == LOOP_STOP_NOTE
    assert agent.conversation.pending_calls() == []
    _assert_well_formed(agent.conversation.messages)


def test_loop_stops_on_repeated_identical_changes(tmp_path):
    # изменяющая операция без изменений между повторами (как pytest без правок)
    tool = FakeTool()
    provider = ScriptedProvider([tool_turn("fake_write", {})] * 10)
    events = RecordingEvents()
    _agent(provider, tmp_path, events=events, registry=_registry_with(tool)).run_turn("x")
    assert tool.runs == 3
    assert events.stats[-1].stop_reason == "tool_repeats"


def test_change_between_repeats_is_not_a_loop(tmp_path):
    (tmp_path / "f.txt").write_text("hello", encoding="utf-8")
    read = tool_turn("read_file", {"path": "f.txt"})
    write = tool_turn("write_file", {"path": "g.txt", "content": "x"})
    provider = ScriptedProvider([read, read, read, write, read, read, text_turn("ок")])
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events)
    assert agent.run_turn("x") == "ок"
    assert events.stats[-1].stop_reason is None
    assert sum("WARNING" in r for r in _function_results(agent)) == 1  # только 3-е чтение


def test_repeated_rejected_call_is_not_asked_again(tmp_path):
    tool = FakeTool()
    call = tool_turn("fake_write", {})
    provider = ScriptedProvider([call, call, text_turn("ок"), call, text_turn("ок")])
    events = RecordingEvents(confirm_answer=False)
    agent = _agent(
        provider, tmp_path, events=events, registry=_registry_with(tool), auto_approve=False
    )
    assert agent.run_turn("x") == "ок"
    assert len(events.confirms) == 1 and tool.runs == 0
    assert _function_results(agent)[-1] == REJECTED_AGAIN_NOTE
    # в следующем ходе пользователя спрашивают снова
    agent.run_turn("всё же сделай")
    assert len(events.confirms) == 2


def test_rejections_do_not_count_as_failures(tmp_path):
    tool = FakeTool()
    calls = [tool_turn("fake_write", {"path": p}) for p in "abc"]
    provider = ScriptedProvider([*calls, text_turn("ок")])
    events = RecordingEvents(confirm_answer=False)
    agent = _agent(
        provider,
        tmp_path,
        events=events,
        registry=_registry_with(tool),
        auto_approve=False,
        max_tool_failures=2,
    )
    assert agent.run_turn("x") == "ок"
    assert len(events.confirms) == 3
    assert events.stats[-1].stop_reason is None


def test_loop_completes_on_text(tmp_path):
    provider = ScriptedProvider([text_turn("Готово!")])
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events)
    assert agent.run_turn("привет") == "Готово!"
    assert ("text", "Готово!") in events.events


def test_loop_runs_tool_then_finishes(tmp_path):
    (tmp_path / "f.txt").write_text("hello", encoding="utf-8")
    provider = ScriptedProvider([tool_turn("read_file", {"path": "f.txt"}), text_turn("прочитал")])
    agent = _agent(provider, tmp_path)
    assert agent.run_turn("прочитай f.txt") == "прочитал"
    assert provider.calls == 2


def test_call_info_from_history(tmp_path):
    """Строка вызова при продолжении чата — та же, что во время хода."""
    agent = _agent(ScriptedProvider(), tmp_path)
    info = agent.call_info(FunctionCall(name="read_file", arguments={"path": "a.txt"}))
    assert info.summary == "a.txt" and info.read_only
    edit = agent.call_info(FunctionCall(name="edit_file", arguments={"path": "a.txt"}))
    assert edit.kind is ToolKind.EDIT and not edit.read_only
    assert agent.call_info(FunctionCall(name="nope", arguments={"path": "a.txt"})).summary == ""
    bad = agent.call_info(FunctionCall(name="read_file", arguments={"path": 1.5}))
    assert bad.summary == "" and bad.read_only


def test_streaming_events_wrap_each_request(tmp_path):
    provider = ScriptedProvider([text_turn("поток")])
    events = RecordingEvents()
    cfg = Config(access_key="x", project_root=tmp_path, stream=True)
    agent = Agent(provider, build_default_registry(), cfg, events)
    agent.run_turn("hi")
    kinds = [k for k, _ in events.events]
    assert kinds == ["stream_start", "delta", "stream_end"]


def test_write_outside_sandbox_does_not_crash_without_auto_approve(tmp_path):
    # Раньше SandboxError из preview() пробивал run_turn и ронял REPL.
    root = tmp_path / "proj"
    root.mkdir()
    provider = ScriptedProvider(
        [tool_turn("write_file", {"path": "../escape.txt", "content": "x"}), text_turn("не вышло")]
    )
    events = RecordingEvents()
    agent = _agent(provider, root, events=events, auto_approve=False)
    assert agent.run_turn("запиши файл") == "не вышло"
    assert not (tmp_path / "escape.txt").exists()
    assert events.confirms == []  # невыполнимую операцию не предлагаем подтверждать
    results = [m for m in agent.conversation.messages if m.role == "function"]
    assert "за пределы" in results[-1].content


@pytest.mark.parametrize("exc", [ToolError("нельзя"), ValueError("сбой превью")])
def test_preview_failure_never_runs_tool(tmp_path, exc):
    def boom():
        raise exc

    tool = FakeTool(preview=boom)
    provider = ScriptedProvider([tool_turn("fake_write", {}), text_turn("ок")])
    events = RecordingEvents()
    agent = _agent(
        provider, tmp_path, events=events, registry=_registry_with(tool), auto_approve=False
    )
    agent.run_turn("x")
    assert tool.runs == 0
    assert events.confirms == []
    assert events.results[-1][1].ok is False


def test_rejection_does_not_run_and_tells_model(tmp_path):
    tool = FakeTool()
    provider = ScriptedProvider([tool_turn("fake_write", {}), text_turn("ок")])
    events = RecordingEvents(confirm_answer=False)
    agent = _agent(
        provider, tmp_path, events=events, registry=_registry_with(tool), auto_approve=False
    )
    agent.run_turn("x")
    assert tool.runs == 0
    call, preview, dangerous = events.confirms[0]
    assert call.name == "fake_write" and preview.kind == "diff" and dangerous is False
    function_msgs = [m for m in agent.conversation.messages if m.role == "function"]
    assert function_msgs[-1].content == REJECTED_NOTE


def test_auto_approve_skips_confirmation(tmp_path):
    tool = FakeTool()
    provider = ScriptedProvider([tool_turn("fake_write", {}), text_turn("ок")])
    events = RecordingEvents(confirm_answer=False)
    agent = _agent(provider, tmp_path, events=events, registry=_registry_with(tool))
    agent.run_turn("x")
    assert tool.runs == 1
    assert events.confirms == []


def test_custom_tool_display_reaches_ui(tmp_path):
    display = Display("+new line", kind="diff", title="x")
    tool = FakeTool(run=lambda: ToolResult(content="ok", summary="готово", display=display))
    provider = ScriptedProvider([tool_turn("fake_write", {}), text_turn("ок")])
    events = RecordingEvents(confirm_answer=True)
    agent = _agent(
        provider, tmp_path, events=events, registry=_registry_with(tool), auto_approve=False
    )
    agent.run_turn("x")
    call, result, previewed = events.results[-1]
    assert result.display == display and previewed is True


def test_disallowed_git_subcommand_is_not_confirmed(tmp_path):
    provider = ScriptedProvider([tool_turn("git", {"subcommand": "push"}), text_turn("ок")])
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events, auto_approve=False)
    agent.run_turn("запушь")
    assert events.confirms == []
    assert events.results[-1][1].ok is False


def test_turn_stats_split_billed_and_context(tmp_path):
    (tmp_path / "f.txt").write_text("hello", encoding="utf-8")
    provider = ScriptedProvider(
        [
            tool_turn(
                "read_file", {"path": "f.txt"}, Usage(prompt_tokens=7_000, completion_tokens=10)
            ),
            text_turn("ок", Usage(prompt_tokens=7_030, completion_tokens=20)),
        ]
    )
    events = RecordingEvents()
    _agent(provider, tmp_path, events=events).run_turn("x")
    stats = events.stats[-1]
    assert (stats.steps, stats.tool_calls) == (2, 1)
    assert stats.billed_tokens == 14_060
    assert stats.context_tokens == 7_050


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


def test_ctrl_c_during_tool_repairs_history(tmp_path):
    def interrupted():
        raise KeyboardInterrupt

    tool = FakeTool(run=interrupted)
    provider = ScriptedProvider([tool_turn("fake_write", {}), text_turn("снова тут")])
    agent = _agent(provider, tmp_path, registry=_registry_with(tool))
    with pytest.raises(KeyboardInterrupt):
        agent.run_turn("x")
    assert agent.conversation.pending_calls() == []
    # следующий ход отправляет корректную историю
    assert agent.run_turn("продолжай") == "снова тут"
    _assert_well_formed(provider.requests[-1]["messages"])


# ------------------------- несколько вызовов за ответ ------------------------- #
def _with_fake(tool) -> ToolRegistry:
    reg = build_default_registry()
    reg.register(tool)
    return reg


def test_parallel_calls_run_in_order_and_each_gets_result(tmp_path):
    (tmp_path / "a.txt").write_text("AAA", encoding="utf-8")
    (tmp_path / "b.txt").write_text("BBB", encoding="utf-8")
    batch = tools_turn([("read_file", {"path": "a.txt"}), ("read_file", {"path": "b.txt"})])
    provider = ScriptedProvider([batch, text_turn("готово")])
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events)

    assert agent.run_turn("прочитай оба") == "готово"
    assert provider.calls == 2  # оба вызова — за один шаг
    results = [m for m in agent.conversation.messages if m.role == "function"]
    assert [r.call_id for r in results] == ["c1", "c2"]
    assert "AAA" in results[0].content and "BBB" in results[1].content
    shown = [c.summary for kind, c in events.events if kind == "tool_call"]
    assert len(shown) == 2 and "a.txt" in shown[0] and "b.txt" in shown[1]
    assert events.stats[-1].tool_calls == 2
    _assert_well_formed(provider.requests[-1]["messages"][1:])


def test_rejected_call_in_batch_does_not_block_the_rest(tmp_path):
    (tmp_path / "a.txt").write_text("AAA", encoding="utf-8")
    tool = FakeTool()
    batch = tools_turn([("fake_write", {}), ("read_file", {"path": "a.txt"})])
    provider = ScriptedProvider([batch, text_turn("ок")])
    events = RecordingEvents(confirm_answer=False)
    agent = _agent(provider, tmp_path, events=events, registry=_with_fake(tool), auto_approve=False)

    agent.run_turn("x")
    assert tool.runs == 0
    rejected, read = [m for m in agent.conversation.messages if m.role == "function"]
    assert rejected.content == REJECTED_NOTE and rejected.call_id == "c1"
    assert "AAA" in read.content and read.call_id == "c2"
    _assert_well_formed(agent.conversation.messages)


def test_guard_stop_mid_batch_closes_remaining_calls(tmp_path):
    (tmp_path / "a.txt").write_text("AAA", encoding="utf-8")
    batch = tools_turn(
        [
            ("read_file", {"path": "nope1.txt"}),
            ("read_file", {"path": "nope2.txt"}),
            ("read_file", {"path": "a.txt"}),
        ]
    )
    # первая неудача — серия у предела: пачка уходит в ответ на запрос с предупреждением
    first = tool_turn("read_file", {"path": "nope0.txt"})
    provider = ScriptedProvider([first, batch, text_turn("итог")])
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events, max_tool_failures=2)

    assert agent.run_turn("x") == "итог"
    assert events.stats[-1].stop_reason == "tool_failures"
    results = [m for m in agent.conversation.messages if m.role == "function"][1:]
    assert [r.call_id for r in results] == ["c1", "c2", "c3"]
    assert [r.content for r in results[1:]] == [SKIPPED_NOTE] * 2  # не выполнялись
    _assert_well_formed(agent.conversation.messages)
    _assert_well_formed(provider.requests[-1]["messages"][1:])


def test_guard_stop_before_a_call_mid_batch_closes_remaining_calls(tmp_path):
    (tmp_path / "a.txt").write_text("AAA", encoding="utf-8")
    (tmp_path / "b.txt").write_text("BBB", encoding="utf-8")
    batch = tools_turn([("read_file", {"path": "a.txt"})] * 4 + [("read_file", {"path": "b.txt"})])
    provider = ScriptedProvider([batch, text_turn("итог")])
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events)

    assert agent.run_turn("x") == "итог"
    assert events.stats[-1].stop_reason == "tool_repeats"
    results = [m for m in agent.conversation.messages if m.role == "function"]
    assert [r.call_id for r in results] == ["c1", "c2", "c3", "c4", "c5"]
    assert [r.content for r in results[3:]] == [LOOP_STOP_NOTE, SKIPPED_NOTE]
    _assert_well_formed(agent.conversation.messages)
    _assert_well_formed(provider.requests[-1]["messages"][1:])


def test_ctrl_c_mid_batch_closes_all_remaining_calls(tmp_path):
    def interrupted():
        raise KeyboardInterrupt

    (tmp_path / "a.txt").write_text("AAA", encoding="utf-8")
    tool = FakeTool(run=interrupted)
    batch = tools_turn([("fake_write", {}), ("read_file", {"path": "a.txt"})])
    provider = ScriptedProvider([batch, text_turn("снова тут")])
    agent = _agent(provider, tmp_path, registry=_with_fake(tool))

    with pytest.raises(KeyboardInterrupt):
        agent.run_turn("x")
    results = [m for m in agent.conversation.messages if m.role == "function"]
    assert [(r.call_id, r.content) for r in results] == [
        ("c1", INTERRUPTED_NOTE),
        ("c2", INTERRUPTED_NOTE),
    ]
    assert agent.run_turn("продолжай") == "снова тут"
    _assert_well_formed(provider.requests[-1]["messages"][1:])


def test_set_model_is_passed_to_provider(tmp_path):
    provider = ScriptedProvider([text_turn("a"), text_turn("b")])
    agent = _agent(provider, tmp_path)
    agent.run_turn("1")
    agent.set_model("Alpha-2-Max")
    agent.run_turn("2")
    assert [r["model"] for r in provider.requests] == [agent._cfg.model, "Alpha-2-Max"]
    with pytest.raises(ValueError):
        agent.set_model("  ")


def test_reset_rebuilds_system_prompt(tmp_path):
    provider = ScriptedProvider([text_turn("a"), text_turn("b")])
    agent = _agent(provider, tmp_path)
    agent.run_turn("1")
    (tmp_path / "new_file.py").write_text("x", encoding="utf-8")
    agent.reset()
    assert len(agent.conversation) == 0
    agent.run_turn("2")
    system = provider.requests[-1]["messages"][0]
    assert system.role == "system" and "new_file.py" in system.content
    assert [m.role for m in provider.requests[-1]["messages"]] == ["system", "user"]


def test_reset_with_saved_conversation_continues_it(tmp_path):
    from gigacrew.agent.conversation import Conversation

    saved = Conversation()
    saved.add_user("старый запрос")
    saved.add_assistant(
        text_turn("старый ответ").message, Usage(prompt_tokens=7_000, completion_tokens=5)
    )
    provider = ScriptedProvider([text_turn("новый ответ")])
    agent = _agent(provider, tmp_path)
    agent.reset(saved)
    assert agent.conversation is saved and agent.context_tokens == 7_005
    agent.run_turn("дальше")
    roles = [(m.role, m.content) for m in provider.requests[-1]["messages"][1:]]
    assert roles == [
        ("user", "старый запрос"),
        ("assistant", "старый ответ"),
        ("user", "дальше"),
    ]


def test_constructor_has_no_filesystem_side_effects(tmp_path, monkeypatch):
    import gigacrew.agent.loop as loop

    def boom(_ws):
        raise AssertionError("системный промпт собран в конструкторе")

    monkeypatch.setattr(loop, "build_project_context", boom)
    parent = _agent(ScriptedProvider(), tmp_path)  # не падает
    loop.Agent(  # суб-агент — тоже
        parent.provider,
        build_default_registry(),
        parent.config,
        parent=parent,
        subagent=EXPLORE,
    )


def test_long_turn_keeps_current_task_in_context(tmp_path):
    (tmp_path / "big.txt").write_text("y" * 5000, encoding="utf-8")
    steps = [tool_turn("read_file", {"path": "big.txt"}) for _ in range(6)] + [text_turn("ок")]
    provider = ScriptedProvider(steps)
    # Страховка окна истории — без сжатия контекста.
    agent = _agent(provider, tmp_path, context_budget_tokens=4000, auto_compact=False)
    agent.run_turn("ТЕКУЩАЯ ЗАДАЧА")
    for request in provider.requests:
        contents = [m.content for m in request["messages"] if m.role == "user"]
        assert contents == ["ТЕКУЩАЯ ЗАДАЧА"]
        _assert_well_formed(request["messages"])
    assert len(provider.requests[-1]["messages"]) < len(agent.conversation) + 1


def test_empty_conversation_instance_is_used(tmp_path):
    from gigacrew.agent.conversation import Conversation

    conv = Conversation()  # пустой — ложен по __len__
    cfg = Config(access_key="x", project_root=tmp_path, stream=False)
    agent = Agent(
        ScriptedProvider([text_turn("ок")]), build_default_registry(), cfg, conversation=conv
    )
    agent.run_turn("x")
    assert agent.conversation is conv and len(conv) == 2


# ------------------------- события для индикаторов ------------------------- #
def _kinds(events) -> list[str]:
    return [k for k, _ in events.events]


@pytest.mark.parametrize("error", [None, ToolError("нельзя"), ValueError("сбой")])
def test_tool_start_end_bracket_run(tmp_path, error):
    def run():
        if error is not None:
            raise error
        return ToolResult(content="ok", summary="ok")

    tool = FakeTool(run=run)
    provider = ScriptedProvider([tool_turn("fake_write", {}), text_turn("ок")])
    events = RecordingEvents()
    _agent(provider, tmp_path, events=events, registry=_registry_with(tool)).run_turn("x")
    kinds = _kinds(events)
    start = kinds.index("tool_start")
    assert kinds[start : start + 3] == ["tool_start", "tool_end", "tool_result"]


def test_tool_end_called_on_ctrl_c(tmp_path):
    def interrupted():
        raise KeyboardInterrupt

    tool = FakeTool(run=interrupted)
    events = RecordingEvents()
    agent = _agent(
        ScriptedProvider([tool_turn("fake_write", {})]),
        tmp_path,
        events=events,
        registry=_registry_with(tool),
    )
    with pytest.raises(KeyboardInterrupt):
        agent.run_turn("x")
    assert _kinds(events)[-2:] == ["tool_start", "tool_end"]


class _InterruptAfterRun(RecordingEvents):
    """Ctrl+C/Esc (асинхронный SIGINT) приходит, когда инструмент уже отработал."""

    def __init__(self, at: str):
        super().__init__()
        self.at = at

    def on_tool_end(self, call):
        super().on_tool_end(call)
        if self.at == "tool_end":
            raise KeyboardInterrupt

    def on_tool_result(self, call, result, *, previewed):
        super().on_tool_result(call, result, previewed=previewed)
        if self.at == "tool_result":
            raise KeyboardInterrupt


@pytest.mark.parametrize("at", ["tool_end", "tool_result"])
def test_interrupt_after_tool_ran_keeps_its_result(tmp_path, at):
    from gigacrew.agent.decisions import DecisionMemory

    script = _script(tmp_path)
    memory, _ = DecisionMemory.load(tmp_path, base=tmp_path / "home")
    events = _InterruptAfterRun(at)
    cfg = Config(access_key="x", project_root=tmp_path, stream=False, yes_all=True)
    provider = ScriptedProvider([tool_turn("run_shell", {"command": script})])
    agent = Agent(provider, build_default_registry(), cfg, events, decisions=memory)
    with pytest.raises(KeyboardInterrupt):
        agent.run_turn("x")
    # действие совершено: модель видит результат, а не «прервано, результата нет»
    [result] = _function_results(agent)
    assert "built" in result and result != INTERRUPTED_NOTE
    lines = [json.loads(line) for line in memory.journal_path.read_text().splitlines()]
    assert [x["event"] for x in lines] == ["decision", "outcome"]
    assert lines[1]["ok"] is True and "interrupted" not in lines[1]
    _assert_well_formed(agent.conversation.messages)


def test_interrupt_during_tool_run_is_journaled_as_interrupted(tmp_path, monkeypatch):
    from gigacrew.agent.decisions import DecisionMemory
    from gigacrew.tools.shell import RunShellTool

    def interrupted(self, params, ctx):
        raise KeyboardInterrupt

    monkeypatch.setattr(RunShellTool, "run", interrupted)
    memory, _ = DecisionMemory.load(tmp_path, base=tmp_path / "home")
    cfg = Config(access_key="x", project_root=tmp_path, stream=False, yes_all=True)
    provider = ScriptedProvider([tool_turn("run_shell", {"command": _script(tmp_path)})])
    agent = Agent(provider, build_default_registry(), cfg, RecordingEvents(), decisions=memory)
    with pytest.raises(KeyboardInterrupt):
        agent.run_turn("x")
    assert _function_results(agent) == [INTERRUPTED_NOTE]
    outcome = json.loads(memory.journal_path.read_text().splitlines()[-1])
    assert outcome["event"] == "outcome" and outcome["interrupted"] is True


def test_rejected_tool_is_not_started(tmp_path):
    tool = FakeTool()
    provider = ScriptedProvider([tool_turn("fake_write", {}), text_turn("ок")])
    events = RecordingEvents(confirm_answer=False)
    _agent(
        provider, tmp_path, events=events, registry=_registry_with(tool), auto_approve=False
    ).run_turn("x")
    assert "tool_start" not in _kinds(events)


def test_non_stream_mode_signals_waiting(tmp_path):
    events = RecordingEvents()
    _agent(ScriptedProvider([text_turn("ответ")]), tmp_path, events=events).run_turn("x")
    assert _kinds(events) == ["stream_start", "stream_end", "text"]


def test_non_stream_error_still_ends_waiting(tmp_path):
    from gigacrew.llm.base import LLMError

    events = RecordingEvents()
    agent = _agent(ScriptedProvider([LLMError("сеть")]), tmp_path, events=events)
    with pytest.raises(LLMError):
        agent.run_turn("x")
    assert _kinds(events) == ["stream_start", "stream_end"]


def test_session_counters(tmp_path):
    provider = ScriptedProvider(
        [
            text_turn("a", Usage(prompt_tokens=7_000, completion_tokens=10)),
            text_turn("b", Usage(prompt_tokens=8_000, completion_tokens=20)),
        ]
    )
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events)
    assert agent.context_tokens == 0
    agent.run_turn("1")
    assert agent.context_tokens == 7_010 and agent.billed_tokens == 7_010
    agent.reset()
    assert agent.context_tokens == 0 and agent.billed_tokens == 7_010
    agent.run_turn("2")
    assert agent.context_tokens == 8_020 and agent.billed_tokens == 15_030
    assert events.stats[-1].duration_s >= 0
    assert agent.context_budget > 0


def test_ask_user_reaches_ui_and_model(tmp_path):
    from gigacrew.tools.questions import Answer

    question = {
        "question": "Какой формат?",
        "options": [{"label": "JSON"}, {"label": "YAML", "description": "читаемее"}],
    }
    provider = ScriptedProvider(
        [tool_turn("ask_user", {"questions": [question]}), text_turn("Сделаю YAML")]
    )
    events = RecordingEvents(answers=[Answer(("YAML",))])
    agent = _agent(provider, tmp_path, events=events, auto_approve=False)
    assert agent.run_turn("сохрани конфиг") == "Сделаю YAML"
    assert events.questions[0][0].options[1].description == "читаемее"
    assert events.confirms == []  # вопрос — не изменяющая операция
    sent = provider.requests[-1]["messages"][-1]
    assert sent.role == "function" and "Какой формат? → YAML" in sent.content
    kinds = _kinds(events)
    assert kinds[kinds.index("tool_start") + 1] == "tool_end"


def test_ask_user_without_ui_tells_model(tmp_path):
    from gigacrew.agent.events import AgentEvents

    question = {"question": "Да?", "options": [{"label": "да"}, {"label": "нет"}]}
    provider = ScriptedProvider(
        [tool_turn("ask_user", {"questions": [question]}), text_turn("решу сам")]
    )
    cfg = Config(access_key="x", project_root=tmp_path, stream=False)
    agent = Agent(provider, build_default_registry(), cfg, AgentEvents())
    assert agent.run_turn("x") == "решу сам"
    assert "Нельзя задать вопрос" in provider.requests[-1]["messages"][-1].content


# ------------------------------ окно контекста ------------------------------ #
def _probe(model: str, window: int):
    from gigacrew.llm.context_probe import ProbeResult

    return ProbeResult(model, window, window + 1000, False, 5, window)


def test_context_budget_follows_model_window(tmp_path):
    from gigacrew.agent.context_window import DEFAULT_CONTEXT_WINDOW, budget_for_window
    from gigacrew.llm.model_windows import ModelWindows

    windows = ModelWindows()
    windows.record(_probe("big", 128_000))
    cfg = Config(access_key="x", project_root=tmp_path, stream=False, model="small")
    agent = Agent(ScriptedProvider(), build_default_registry(), cfg, windows=windows)
    assert agent.windows is windows
    assert agent.context_window is None
    assert agent.context_budget == budget_for_window(DEFAULT_CONTEXT_WINDOW)
    agent.set_model("big")
    assert agent.context_window == 128_000
    assert agent.context_budget == budget_for_window(128_000) < 128_000 - 8_000

    explicit = Config(access_key="x", project_root=tmp_path, context_budget_tokens=5_000)
    agent = Agent(ScriptedProvider(), build_default_registry(), explicit, windows=windows)
    agent.set_model("big")
    assert agent.context_budget == 5_000


def test_request_budget_includes_tool_schemas(tmp_path):
    from gigacrew.agent.conversation import Conversation
    from gigacrew.llm.types import Message, ToolSpec

    conv = Conversation()
    for i in range(20):
        conv.add_user(f"вопрос {i} " + "x" * 1500)
        conv.add_assistant(Message(role="assistant", content=f"ответ {i} " + "y" * 1500))
    agent = _agent(ScriptedProvider(), tmp_path, context_budget_tokens=12_000)
    agent.reset(conv)
    heavy = [ToolSpec(name="t", description="d" * 15_000, parameters={})]  # ~5000 токенов
    without = agent._build_request([])
    with_specs = agent._build_request(heavy)
    assert len(with_specs) < len(without) < len(conv) + 1
    assert with_specs[-1].content.startswith("ответ 19")


# ----------------------------- режимы разрешений ----------------------------- #
def _system_prompts(provider) -> list[str]:
    return [request["messages"][0].content for request in provider.requests]


def test_plan_mode_blocks_edits_without_asking(tmp_path):
    tool = FakeTool(kind=ToolKind.EDIT)
    provider = ScriptedProvider([tool_turn("fake_write", {}), text_turn("ок")])
    events = RecordingEvents(confirm_answer=True)
    agent = _agent(
        provider,
        tmp_path,
        events=events,
        registry=_registry_with(tool),
        auto_approve=False,
        mode=PermissionMode.PLAN,
    )
    agent.run_turn("поправь")
    assert tool.runs == 0 and events.confirms == []
    assert events.results[-1][1].summary == "заблокировано: режим планирования"
    assert _function_results(agent)[-1] == PLAN_BLOCKED_NOTE.format(path=agent._plan_display())


def test_plan_mode_blocks_even_with_auto_approve(tmp_path):
    tool = FakeTool(kind=ToolKind.EDIT)
    provider = ScriptedProvider([tool_turn("fake_write", {}), text_turn("ок")])
    agent = _agent(provider, tmp_path, registry=_registry_with(tool), mode=PermissionMode.PLAN)
    agent.run_turn("поправь")
    assert tool.runs == 0


def test_plan_mode_asks_for_commands(tmp_path):
    tool = FakeTool(kind=ToolKind.COMMAND)
    provider = ScriptedProvider([tool_turn("fake_write", {}), text_turn("ок")])
    events = RecordingEvents(confirm_answer=True)
    agent = _agent(
        provider,
        tmp_path,
        events=events,
        registry=_registry_with(tool),
        auto_approve=False,
        mode=PermissionMode.PLAN,
    )
    agent.run_turn("запусти тесты")
    assert tool.runs == 1 and len(events.confirms) == 1


def test_plan_mode_stops_model_that_keeps_editing(tmp_path):
    edits = [tool_turn("write_file", {"path": f"f{i}.txt", "content": "x"}) for i in range(10)]
    provider = ScriptedProvider(edits)
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events, mode=PermissionMode.PLAN)
    agent.run_turn("сделай")
    assert events.stats[-1].stop_reason == "tool_failures"
    assert list(tmp_path.iterdir()) == []


def test_auto_mode_applies_edits_without_asking(tmp_path):
    tool = FakeTool(kind=ToolKind.EDIT)
    provider = ScriptedProvider([tool_turn("fake_write", {}), text_turn("ок")])
    events = RecordingEvents(confirm_answer=False)
    agent = _agent(
        provider,
        tmp_path,
        events=events,
        registry=_registry_with(tool),
        auto_approve=False,
        mode=PermissionMode.AUTO,
    )
    agent.run_turn("поправь")
    assert tool.runs == 1 and events.confirms == []


@pytest.mark.parametrize("kind", [ToolKind.COMMAND, ToolKind.OTHER])
def test_auto_mode_still_asks_for_the_rest(tmp_path, kind):
    # команда без вердикта классификатора рецензенту не отдаётся
    tool = FakeTool(kind=kind)
    provider = ScriptedProvider([tool_turn("fake_write", {}), text_turn("ок")])
    events = RecordingEvents(confirm_answer=False)
    agent = _agent(
        provider,
        tmp_path,
        events=events,
        registry=_registry_with(tool),
        auto_approve=False,
        mode=PermissionMode.AUTO,
    )
    agent.run_turn("x")
    assert tool.runs == 0 and len(events.confirms) == 1


def test_plan_rules_are_sent_only_in_plan_mode(tmp_path):
    # «OK» — ответ на самопроверку: ход в режиме плана кончился текстом
    provider = ScriptedProvider([text_turn("план"), text_turn("OK"), text_turn("готово")])
    agent = _agent(provider, tmp_path, mode=PermissionMode.PLAN)
    agent.run_turn("спланируй")
    agent.set_mode(PermissionMode.AUTO)
    agent.run_turn("делай")
    planning, check, auto = _system_prompts(provider)
    assert "# Plan mode" in planning and "# Plan mode" in check
    assert "# Plan mode" not in auto


def test_mode_survives_reset_and_cycles(tmp_path):
    agent = _agent(ScriptedProvider(), tmp_path)
    assert agent.mode is PermissionMode.AUTO
    assert agent.cycle_mode() is PermissionMode.PLAN
    agent.reset()
    assert agent.mode is PermissionMode.PLAN
    assert [agent.cycle_mode(), agent.cycle_mode()] == [PermissionMode.AUTO, PermissionMode.PLAN]


PLAN_REL = ".gigacode/plans/chat1.md"


def _planning_agent(provider, tmp_path, events=None) -> Agent:
    agent = _agent(provider, tmp_path, events=events, auto_approve=False, mode=PermissionMode.PLAN)
    agent.chat_id = "chat1"
    return agent


def test_approved_plan_switches_mode_within_the_turn(tmp_path):
    provider = ScriptedProvider(
        [
            tool_turn("write_file", {"path": PLAN_REL, "content": "# План\n1. создать a.txt"}),
            tool_turn("exit_plan_mode", {}),
            tool_turn("write_file", {"path": "a.txt", "content": "x"}),
            text_turn("готово"),
        ]
    )
    events = RecordingEvents(confirm_answer=False, answers=[Answer((APPROVE,))])
    agent = _planning_agent(provider, tmp_path, events)
    assert agent.run_turn("спланируй и сделай") == "готово"
    assert agent.mode is PermissionMode.AUTO
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "x"
    assert events.confirms == []  # ни план в режиме плана, ни правка после одобрения
    (question,) = events.questions[0]
    assert question.body.startswith("# План\n1. создать a.txt")
    # правила плана (с путём файла плана) — только до одобрения
    prompts = _system_prompts(provider)
    assert ["# Plan mode" in p for p in prompts] == [True, True, False, False]
    assert PLAN_REL in prompts[0]


def test_plan_mode_reminder_goes_with_every_request(tmp_path):
    """Правила режима плана — не только в системном промпте: короткое напоминание с путём
    файла плана — в хвосте каждого запроса (модели теряют правила к концу хода)."""
    (tmp_path / "f.txt").write_text("hello", encoding="utf-8")
    provider = ScriptedProvider(
        [
            tool_turn("read_file", {"path": "f.txt"}),
            tool_turn("write_file", {"path": PLAN_REL, "content": "# План"}),
            tool_turn("exit_plan_mode", {}),
            tool_turn("write_file", {"path": "a.txt", "content": "x"}),
            text_turn("готово"),
        ]
    )
    events = RecordingEvents(confirm_answer=False, answers=[Answer((APPROVE,))])
    agent = _planning_agent(provider, tmp_path, events)
    agent.run_turn("спланируй и сделай")
    reminder = PLAN_REMINDER_NOTE.format(path=PLAN_REL)
    tails = [r["messages"][-1].content for r in provider.requests]
    assert [reminder in tail for tail in tails] == [True, True, True, False, False]
    assert tails[0].startswith("спланируй и сделай") and "<system-reminder>" in tails[0]
    assert all(reminder not in r["messages"][0].content for r in provider.requests)
    assert all(reminder not in m.content for m in agent.conversation.messages)  # не в истории


@pytest.mark.parametrize(
    "answer", [None, QuestionsUnavailable("нет терминала")], ids=["esc", "no_approver"]
)
def test_no_plan_reminder_after_the_plan_was_shown(tmp_path, answer):
    """План показан — что дальше, говорит исход показа («перескажи суть», «дождись
    указаний»); напоминание «план — не текстом ответа» ему бы противоречило."""
    provider = ScriptedProvider(
        [
            tool_turn("write_file", {"path": PLAN_REL, "content": "# План"}),
            tool_turn("exit_plan_mode", {}),
            text_turn("План в файле."),
        ]
    )
    agent = _planning_agent(provider, tmp_path, _PlanAnswers(answer))
    agent.run_turn("спланируй")
    reminder = PLAN_REMINDER_NOTE.format(path=PLAN_REL)
    assert [reminder in tail for tail in _tails(provider)] == [True, True, False]


def test_plan_mode_allows_only_the_plan_file(tmp_path):
    provider = ScriptedProvider(
        [
            tool_turn("write_file", {"path": "a.txt", "content": "x"}),
            tool_turn("write_file", {"path": ".gigacode/plans/other.md", "content": "x"}),
            tool_turn("write_file", {"path": PLAN_REL, "content": "# План"}),
            tool_turn("exit_plan_mode", {}),
            text_turn("жду"),
        ]
    )
    events = RecordingEvents(answers=None)
    agent = _planning_agent(provider, tmp_path, events)
    agent.run_turn("спланируй")
    results = _function_results(agent)
    blocked = PLAN_BLOCKED_NOTE.format(path=PLAN_REL)  # отказ называет файл плана
    assert results[0] == blocked
    assert results[1] == blocked  # чужой план — обычная правка
    assert (tmp_path / PLAN_REL).read_text(encoding="utf-8") == "# План"
    assert not (tmp_path / "a.txt").exists() and events.confirms == []


class _PlanAnswers(RecordingEvents):
    """Ответы на вопросы по очереди: ``None`` — отказ (Esc), исключение — бросается
    (``QuestionsUnavailable`` — спросить некого)."""

    def __init__(self, *answers):
        super().__init__(confirm_answer=False)
        self._queue = list(answers)

    def ask_user(self, questions):
        self.questions.append(list(questions))
        answer = self._queue.pop(0) if self._queue else None
        if isinstance(answer, BaseException):
            raise answer
        return answer


def _user_messages(agent) -> list[tuple[str, bool]]:
    return [(m.content, m.meta) for m in agent.conversation.messages if m.role == "user"]


def test_plan_left_in_text_is_shown_and_approved(tmp_path):
    provider = ScriptedProvider(
        [
            tool_turn("write_file", {"path": PLAN_REL, "content": "# План\n1. создать a.txt"}),
            text_turn("Вот план: ..."),  # забыла exit_plan_mode
            tool_turn("write_file", {"path": "a.txt", "content": "x"}),
            text_turn("готово"),
        ]
    )
    events = _PlanAnswers([Answer((APPROVE,))])
    agent = _planning_agent(provider, tmp_path, events)
    assert agent.run_turn("спланируй и сделай") == "готово"
    assert agent.mode is PermissionMode.AUTO
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "x"
    (question,) = events.questions[0]
    assert question.body.startswith("# План\n1. создать a.txt")
    # ответ модели остался в истории, за ним — служебное сообщение с решением пользователя
    request = provider.requests[2]["messages"]
    assert request[-2].role == "assistant" and request[-2].content == "Вот план: ..."
    decision = request[-1]
    assert decision.role == "user" and decision.content.startswith("<system-reminder>")
    assert "Пользователь одобрил план" in decision.content and "создать a.txt" in decision.content
    assert [meta for _, meta in _user_messages(agent)] == [False, True]
    assert len(provider.requests) == 4 and events.stats[-1].self_checks == 0
    notices = [text for _, text in events.notices]
    assert any("не отправлен на одобрение" in text for text in notices)
    assert any(text.startswith("план одобрен") for text in notices)


def test_plan_left_in_text_with_remarks_is_revised_and_shown_again(tmp_path):
    fix = {"path": PLAN_REL, "old_string": "1. код", "new_string": "1. код\n2. тесты"}
    provider = ScriptedProvider(
        [
            tool_turn("write_file", {"path": PLAN_REL, "content": "# План\n1. код"}),
            text_turn("Вот план"),
            tool_turn("edit_file", fix),
            text_turn("Исправил"),  # и снова без exit_plan_mode
        ]
    )
    events = _PlanAnswers([Answer((), custom="добавь тесты")], None)
    agent = _planning_agent(provider, tmp_path, events)
    assert agent.run_turn("спланируй") == "Исправил"
    assert agent.mode is PermissionMode.PLAN and len(events.questions) == 2
    assert "2. тесты" in events.questions[1][0].body
    remark = provider.requests[2]["messages"][-1]
    assert remark.meta and "просит доработать: «добавь тесты»" in remark.content
    # после отказа (Esc) ход кончился ответом модели, без служебного сообщения
    assert agent.conversation.messages[-1].content == "Исправил"
    assert len(provider.requests) == 4 and events.stats[-1].self_checks == 0


FAILURES_MARK = "One more failed call will stop"


def _tails(provider) -> list[str]:
    return [r["messages"][-1].content for r in provider.requests]


def test_failure_warning_survives_cut_retry(tmp_path):
    """Ответ на запрос с предупреждением оборван — повтор запроса тоже с предупреждением."""
    bad = [tool_turn("read_file", {"path": f"nope{i}.txt"}) for i in range(3)]
    provider = ScriptedProvider(
        [*bad, cut_turn("частичный"), tool_turn("read_file", {"path": "nope9.txt"})]
    )
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events, max_tool_failures=4)
    agent.run_turn("x")
    marks = [FAILURES_MARK in tail for tail in _tails(provider)]
    assert marks[:5] == [False, False, False, True, True]
    assert events.stats[-1].stop_reason == "tool_failures"
    warned = [text for _, text in events.notices if "модель предупреждена" in text]
    assert len(warned) == 1  # пользователю — один раз


def test_failure_warning_goes_with_plan_self_check(tmp_path):
    bad = [tool_turn("read_file", {"path": f"nope{i}.txt"}) for i in range(3)]
    provider = ScriptedProvider(
        [*bad, text_turn("ответ"), tool_turn("read_file", {"path": "nope9.txt"}), text_turn("x")]
    )
    events = RecordingEvents()
    agent = _planning_agent(provider, tmp_path, events)
    agent.run_turn("x")
    tails = _tails(provider)
    assert SELF_CHECK_NOTE.splitlines()[0] in tails[4] and FAILURES_MARK in tails[4]


def test_failures_crossing_the_limit_in_one_batch_are_warned_first(tmp_path):
    """Пачка вызовов перешла предел — модель сначала предупреждается, стоп — после."""

    def batch(*paths):
        return tools_turn([("read_file", {"path": p}) for p in paths])

    provider = ScriptedProvider([batch("n1", "n2"), batch("n3", "n4"), batch("n5"), text_turn("x")])
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events, max_tool_failures=4)
    agent.run_turn("x")
    marks = [FAILURES_MARK in tail for tail in _tails(provider)]
    assert marks[:3] == [False, False, True]
    assert events.stats[-1].stop_reason == "tool_failures"


@pytest.mark.skipif(sys.platform == "win32", reason="симлинки")
@pytest.mark.parametrize("variant", ["data", "dir", "file"])
def test_plan_argument_is_not_written_through_a_link(tmp_path, variant):
    """exit_plan_mode(plan=...) не пишет через ссылку на месте файла или папки плана:
    запись шла бы мимо проверки риска write_file — в файл проекта без вопроса."""
    (tmp_path / "src").mkdir()
    data = tmp_path / ".gigacode"
    if variant == "data":  # служебная папка — ссылка в каталог проекта
        os.symlink("src", data)
        target = tmp_path / "src" / "plans" / "chat1.md"
    elif variant == "dir":
        data.mkdir()
        os.symlink("../src", data / "plans")
        target = tmp_path / "src" / "chat1.md"
    else:
        (data / "plans").mkdir(parents=True)
        os.symlink("../../conftest.py", data / "plans" / "chat1.md")
        target = tmp_path / "conftest.py"
    provider = ScriptedProvider(
        [tool_turn("exit_plan_mode", {"plan": "import os  # x"}), text_turn("ok")]
    )
    events = RecordingEvents(answers=None)
    agent = _planning_agent(provider, tmp_path, events)
    agent.run_turn("план")
    assert not target.exists() and events.questions == []
    assert "символическая ссылка" in _function_results(agent)[-1]


def test_plan_edited_after_approval_does_not_hide_the_next_plan(tmp_path):
    """Отметки в плане после одобрения (режим «авто») — не новый план: в новом цикле
    планирования план из аргумента не теряется молча, старый план не показывается."""
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    provider = ScriptedProvider(
        [
            tool_turn("write_file", {"path": PLAN_REL, "content": "# План A\n- [ ] a.py"}),
            tool_turn("exit_plan_mode", {}),
            tool_turn("edit_file", {"path": "a.py", "old_string": "x = 1", "new_string": "x = 2"}),
            tool_turn(
                "edit_file", {"path": PLAN_REL, "old_string": "- [ ]", "new_string": "- [x]"}
            ),
            text_turn("готово"),
            tool_turn("exit_plan_mode", {"plan": "# План B\n- кэш"}),
            text_turn("жду"),
        ]
    )
    events = _PlanAnswers([Answer((APPROVE,))], [Answer((APPROVE,))])
    agent = _planning_agent(provider, tmp_path, events)
    agent.run_turn("спланируй и сделай A")
    # правка плана в режиме «авто» — не план, ждущий показа
    assert agent.mode is PermissionMode.AUTO and not agent._plan.dirty
    agent.set_mode(PermissionMode.PLAN)
    agent.run_turn("теперь спланируй B")
    assert len(events.questions) == 1  # план A второй раз не показан
    result = _function_results(agent)[-1]
    assert "не менялся" in result and PLAN_REL in result


def test_plan_written_in_this_cycle_survives_mode_toggle(tmp_path):
    """Shift+Tab дважды посреди хода (план → авто → план) не теряет записанный план."""
    holder: dict = {}

    class Toggle(ScriptedProvider):
        def complete(self, messages, tools=None, **kwargs):
            if tools is not None and len(self.requests) == 1:
                holder["agent"].cycle_mode()
                holder["agent"].cycle_mode()
            return super().complete(messages, tools, **kwargs)

    provider = Toggle(
        [
            tool_turn("write_file", {"path": PLAN_REL, "content": "# План C"}),
            text_turn("Вот план."),
        ]
    )
    events = _PlanAnswers(None)
    agent = _planning_agent(provider, tmp_path, events)
    holder["agent"] = agent
    agent.run_turn("спланируй")
    assert agent.mode is PermissionMode.PLAN
    assert [q[0].body for q in events.questions] == ["# План C"]  # показан сам


def test_plan_revised_many_times_is_not_a_loop(tmp_path):
    """Правка плана между показами — изменение: exit_plan_mode после неё — не повтор."""
    rounds = []
    for i in range(4):
        fix = {"path": PLAN_REL, "old_string": f"{i}.", "new_string": f"{i}.\n{i + 1}."}
        rounds += [tool_turn("exit_plan_mode", {}), tool_turn("edit_file", fix)]
    provider = ScriptedProvider(
        [
            tool_turn("write_file", {"path": PLAN_REL, "content": "# План\n0."}),
            *rounds,
            tool_turn("exit_plan_mode", {}),
            text_turn("жду"),
        ]
    )
    remarks = [Answer((), custom=f"замечание {i}") for i in range(4)]
    events = _PlanAnswers(*remarks, None)
    agent = _planning_agent(provider, tmp_path, events)
    assert agent.run_turn("спланируй") == "жду"
    assert len(events.questions) == 5 and events.stats[-1].stop_reason is None
    assert not any("WARNING" in r for r in _function_results(agent))


def test_unchanged_plan_shown_again_and_again_is_a_loop(tmp_path):
    provider = ScriptedProvider(
        [
            tool_turn("write_file", {"path": PLAN_REL, "content": "# План"}),
            *[tool_turn("exit_plan_mode", {}) for _ in range(4)],
            text_turn("итог"),
        ]
    )
    events = _PlanAnswers(QuestionsUnavailable("нет терминала"))
    agent = _planning_agent(provider, tmp_path, events)
    agent.run_turn("спланируй")
    assert events.stats[-1].stop_reason == "tool_repeats"


@pytest.mark.parametrize(
    "answer", [None, QuestionsUnavailable("нет терминала")], ids=["esc", "no_approver"]
)
def test_plan_left_in_text_ends_the_turn_when_not_approved(tmp_path, answer):
    provider = ScriptedProvider(
        [
            tool_turn("write_file", {"path": PLAN_REL, "content": "# План"}),
            text_turn("Вот план"),
            text_turn("лишний запрос"),
        ]
    )
    events = _PlanAnswers(answer)
    agent = _planning_agent(provider, tmp_path, events)
    assert agent.run_turn("спланируй") == "Вот план"
    assert len(provider.requests) == 2 and events.stats[-1].self_checks == 0
    assert agent.conversation.messages[-1].content == "Вот план"
    assert agent.mode is PermissionMode.PLAN and not agent._plan.dirty
    if answer is not None:  # одобрить некому — пользователь узнаёт, где план
        assert any(PLAN_REL in text for _, text in events.notices)


def test_plan_declined_with_esc_skips_stop_hooks(tmp_path):
    # Esc на показанном плане — пользователь остановил ход: хук Stop не возвращает к работе.
    from test_agent_hooks import FakeHooks

    from gigacrew.agent.hooks import HookResult
    from gigacrew.project.settings import HookEvent

    hooks = FakeHooks({HookEvent.STOP: HookResult(blocked=True, reason="ещё")})
    provider = ScriptedProvider(
        [
            tool_turn("write_file", {"path": PLAN_REL, "content": "# План"}),
            text_turn("Вот план"),
            text_turn("лишний запрос"),
        ]
    )
    cfg = Config(access_key="x", project_root=tmp_path, stream=False, mode=PermissionMode.PLAN)
    agent = Agent(provider, build_default_registry(), cfg, _PlanAnswers(None), hooks=hooks)
    agent.chat_id = "chat1"
    assert agent.run_turn("спланируй") == "Вот план"
    assert hooks.of(HookEvent.STOP) == [] and provider.calls == 2


def test_cut_reply_in_plan_mode_is_retried_not_self_checked(tmp_path):
    provider = ScriptedProvider(
        [
            cut_turn("Сейчас запишу план", "length"),
            tool_turn("write_file", {"path": PLAN_REL, "content": "# План"}),
            text_turn("Вот план"),
        ]
    )
    events = _PlanAnswers(None)
    agent = _planning_agent(provider, tmp_path, events)
    assert agent.run_turn("спланируй") == "Вот план"
    assert (events.stats[-1].cut_retries, events.stats[-1].self_checks) == (1, 0)
    assert "# Reply cut off" in provider.requests[1]["messages"][-1].content


def test_empty_plan_file_is_not_shown(tmp_path):
    provider = ScriptedProvider(
        [tool_turn("write_file", {"path": PLAN_REL, "content": " "}), text_turn("Вот план")]
    )
    events = _PlanAnswers()
    agent = _planning_agent(provider, tmp_path, events)
    assert agent.run_turn("спланируй") == "Вот план"
    assert events.questions == [] and not agent._plan.dirty
    assert events.notices[-1][0] == "warn"


def test_promise_to_write_the_plan_gets_self_check(tmp_path):
    (tmp_path / "f.txt").write_text("hello", encoding="utf-8")
    provider = ScriptedProvider(
        [
            tool_turn("read_file", {"path": "f.txt"}),
            text_turn("Сейчас запишу план."),  # пообещала и остановилась
            tool_turn("write_file", {"path": PLAN_REL, "content": "# План"}),
            tool_turn("exit_plan_mode", {}),
            text_turn("Жду решения."),
        ]
    )
    events = _PlanAnswers(None)
    agent = _planning_agent(provider, tmp_path, events)
    assert agent.run_turn("спланируй") == "Жду решения."
    check = provider.requests[2]["messages"]
    assert check[-2].role == "assistant" and check[-2].content == "Сейчас запишу план."
    assert check[-1].meta and SELF_CHECK_NOTE.format(path=PLAN_REL) in check[-1].content
    assert check[-1].content.startswith("<system-reminder>")
    assert len(events.questions) == 1  # план показан пользователю
    assert events.stats[-1].self_checks == 1
    # модель после самопроверки работала — служебное сообщение осталось в истории
    assert [meta for _, meta in _user_messages(agent)] == [False, True]
    _assert_well_formed(agent.conversation.messages)


def test_answer_in_plan_mode_survives_self_check(tmp_path):
    provider = ScriptedProvider(
        [
            text_turn("foo считает сумму", Usage(prompt_tokens=7_000, completion_tokens=50)),
            text_turn("OK", Usage(prompt_tokens=7_100, completion_tokens=2)),
        ]
    )
    events = RecordingEvents()
    cfg = Config(access_key="x", project_root=tmp_path, stream=True, mode=PermissionMode.PLAN)
    agent = Agent(provider, build_default_registry(), cfg, events)
    assert agent.run_turn("что делает foo?") == "foo считает сумму"
    assert len(provider.requests) == 2
    # в истории нет ни служебного сообщения, ни «OK»
    contents = [m.content for m in agent.conversation.messages]
    assert contents == ["что делает foo?", "foo считает сумму"]
    shown = [value for kind, value in events.events if kind in ("delta", "text")]
    assert shown == ["foo считает сумму"]  # ответ на самопроверку не показан
    stats = events.stats[-1]
    assert stats.self_checks == 1 and stats.billed_tokens == 14_152
    assert stats.context_tokens == agent.context_tokens == 7_050  # без служебного обмена


def test_self_check_reply_with_calls_is_shown(tmp_path):
    provider = ScriptedProvider(
        [
            text_turn("Сейчас запишу план."),
            tools_turn([("write_file", {"path": PLAN_REL, "content": "# План"})], text="Пишу."),
            tool_turn("exit_plan_mode", {}),
            text_turn("Жду."),
        ]
    )
    events = _PlanAnswers(None)
    cfg = Config(access_key="x", project_root=tmp_path, stream=True, mode=PermissionMode.PLAN)
    agent = Agent(provider, build_default_registry(), cfg, events)
    agent.chat_id = "chat1"
    assert agent.run_turn("спланируй") == "Жду."
    shown = [(kind, value) for kind, value in events.events if kind in ("delta", "text")]
    assert shown == [("delta", "Сейчас запишу план."), ("text", "Пишу."), ("delta", "Жду.")]


def test_self_check_happens_once_per_turn(tmp_path):
    (tmp_path / "f.txt").write_text("hello", encoding="utf-8")
    provider = ScriptedProvider(
        [
            text_turn("Сейчас запишу план."),
            tool_turn("read_file", {"path": "f.txt"}),
            text_turn("Теперь точно запишу."),
            text_turn("лишний запрос"),
        ]
    )
    events = RecordingEvents()
    agent = _planning_agent(provider, tmp_path, events)
    assert agent.run_turn("спланируй") == "Теперь точно запишу."
    assert provider.calls == 3 and events.stats[-1].self_checks == 1


def test_self_check_follows_answered_question(tmp_path):
    options = [{"label": "SQLite"}, {"label": "Postgres"}]
    provider = ScriptedProvider(
        [
            tool_turn("ask_user", {"questions": [{"question": "Какая БД?", "options": options}]}),
            text_turn("Понял, сейчас запишу план."),
            tool_turn("write_file", {"path": PLAN_REL, "content": "# План: SQLite"}),
            tool_turn("exit_plan_mode", {}),
            text_turn("Жду."),
        ]
    )
    events = _PlanAnswers([Answer(("SQLite",))], None)
    agent = _planning_agent(provider, tmp_path, events)
    assert agent.run_turn("спланируй") == "Жду."
    assert events.stats[-1].self_checks == 1 and len(events.questions) == 2


@pytest.mark.parametrize(
    "error", [LLMError("сбой сети"), KeyboardInterrupt()], ids=["llm_error", "interrupt"]
)
def test_failed_self_check_leaves_the_answer(tmp_path, error):
    provider = ScriptedProvider([text_turn("ответ"), error])
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events, mode=PermissionMode.PLAN)
    if isinstance(error, LLMError):
        assert agent.run_turn("вопрос") == "ответ"
        assert events.notices[-1][0] == "warn" and "самопроверка" in events.notices[-1][1]
    else:
        with pytest.raises(KeyboardInterrupt):
            agent.run_turn("вопрос")
    assert [m.content for m in agent.conversation.messages] == ["вопрос", "ответ"]


def test_cut_self_check_reply_is_retried(tmp_path):
    provider = ScriptedProvider(
        [
            text_turn("Сейчас запишу план."),
            cut_turn(reason="incomplete"),  # длинный write_file оборвался
            tool_turn("write_file", {"path": PLAN_REL, "content": "# План"}),
            tool_turn("exit_plan_mode", {}),
            text_turn("Жду."),
        ]
    )
    events = _PlanAnswers(None)
    agent = _planning_agent(provider, tmp_path, events)
    assert agent.run_turn("спланируй") == "Жду."
    retry = provider.requests[2]["messages"][-1]
    assert retry.meta and "# Turn ended in plan mode" in retry.content
    assert "# Reply cut off" in retry.content  # заметка — к служебному сообщению в хвосте
    stats = events.stats[-1]
    assert stats.self_checks == 1 and stats.cut_retries == 1 and len(events.questions) == 1


@pytest.mark.parametrize("mode", [PermissionMode.AUTO])
def test_no_self_check_outside_plan_mode(tmp_path, mode):
    provider = ScriptedProvider([text_turn("ответ на вопрос")])
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events, mode=mode)
    assert agent.run_turn("что делает foo?") == "ответ на вопрос"
    assert len(provider.requests) == 1 and events.stats[-1].self_checks == 0


def test_no_self_check_after_plan_was_shown(tmp_path):
    provider = ScriptedProvider(
        [
            tool_turn("write_file", {"path": PLAN_REL, "content": "# План"}),
            tool_turn("exit_plan_mode", {}),
            text_turn("жду"),
        ]
    )
    agent = _planning_agent(provider, tmp_path, RecordingEvents(answers=None))
    assert agent.run_turn("спланируй") == "жду"
    assert len(provider.requests) == 3


def test_chat_id_names_the_plan_file(tmp_path):
    agent = _agent(ScriptedProvider(), tmp_path)
    agent.chat_id = "a1"
    assert agent.plan_path == tmp_path.resolve() / ".gigacode" / "plans" / "a1.md"
    agent.reset(chat_id="b2")
    assert agent.chat_id == "b2" and agent.plan_path.name == "b2.md"
    agent.reset()
    assert agent.chat_id not in ("a1", "b2")


# ------------------------------ рассуждения ------------------------------ #
def _thinking(turn, reasoning: str):
    turn.message.reasoning = reasoning
    return turn


@pytest.mark.parametrize("mode", ["off", "turn"])
def test_reasoning_is_returned_only_within_the_turn(tmp_path, mode):
    (tmp_path / "f.txt").write_text("hello", encoding="utf-8")
    provider = ScriptedProvider(
        [
            _thinking(tool_turn("read_file", {"path": "f.txt"}), "прочитаю f.txt"),
            _thinking(text_turn("там hello"), "ответ готов"),
            text_turn("ещё раз: hello"),
        ]
    )
    agent = _agent(provider, tmp_path, reasoning=mode)
    agent.run_turn("что в f.txt?")
    second = provider.requests[1]["messages"]
    call = next(m for m in second if m.function_calls)
    assert call.reasoning == ("прочитаю f.txt" if mode == "turn" else "")
    agent.run_turn("повтори")
    # следующий запрос пользователя — рассуждения прошлого хода модели больше не нужны
    assert all(not m.reasoning for m in provider.requests[2]["messages"])
    assert all(not m.reasoning for m in agent.conversation.messages)


# ------------------------------ сжатие контекста ------------------------------ #
def _big_tool() -> FakeTool:
    return FakeTool(run=lambda: ToolResult(content="z" * 3000, summary="ok"))  # ~1000 токенов


def test_auto_compaction_summarizes_old_steps_and_keeps_task(tmp_path):
    from gigacrew.agent.prompts import COMPACT_PROMPT, SUMMARY_HEADER

    steps = [tool_turn("fake_write", {"path": str(i)}) for i in range(4)] + [text_turn("готово")]
    provider = ScriptedProvider(
        steps, summaries=[text_turn("СВОДКА", Usage(prompt_tokens=500, completion_tokens=50))]
    )
    events = RecordingEvents()
    agent = _agent(
        provider,
        tmp_path,
        events=events,
        registry=_registry_with(_big_tool()),
        # бюджет зависит от размера системного промпта: при его росте сжатий станет больше
        context_budget_tokens=5_000,
    )
    assert agent.run_turn("ЗАДАЧА") == "готово"

    (summary_request,) = provider.summary_requests
    system, user = summary_request["messages"]
    assert system.content == COMPACT_PROMPT and "ЗАДАЧА" in user.content
    (result,) = events.compactions
    assert result.auto and result.after_tokens < result.before_tokens
    assert ("compact_start", True) in events.events

    # После сжатия: резюме — в системном сообщении, задача — дословно, история короче.
    for request in provider.requests:
        users = [m.content for m in request["messages"] if m.role == "user"]
        assert users == ["ЗАДАЧА"]
        _assert_well_formed(request["messages"])
    last = provider.requests[-1]["messages"]
    assert SUMMARY_HEADER in last[0].content and "СВОДКА" in last[0].content
    assert len(last) < len(agent.conversation) + 1
    assert len(agent.conversation) == 10  # журнал не переписывается
    assert events.stats[-1].billed_tokens == 550  # суммаризация оплачена в этом ходе
    assert agent.billed_tokens == 550


def test_auto_compaction_failure_warns_and_turn_continues(tmp_path):
    from gigacrew.llm.base import LLMError

    steps = [tool_turn("fake_write", {"path": str(i)}) for i in range(5)] + [text_turn("готово")]
    provider = ScriptedProvider(steps, summaries=[LLMError("сеть упала")])
    events = RecordingEvents()
    agent = _agent(
        provider,
        tmp_path,
        events=events,
        registry=_registry_with(_big_tool()),
        context_budget_tokens=4_000,
    )
    assert agent.run_turn("ЗАДАЧА") == "готово"
    assert len(provider.summary_requests) == 1  # до конца хода не повторяет
    assert events.compactions == [None]
    assert any(level == "warn" and "сеть упала" in text for level, text in events.notices)
    assert agent.conversation.summary is None


def test_auto_compaction_can_be_disabled(tmp_path):
    steps = [tool_turn("fake_write", {"path": str(i)}) for i in range(4)] + [text_turn("готово")]
    provider = ScriptedProvider(steps)
    agent = _agent(
        provider,
        tmp_path,
        registry=_registry_with(_big_tool()),
        context_budget_tokens=4_000,
        auto_compact=False,
    )
    agent.run_turn("ЗАДАЧА")
    assert provider.summary_requests == [] and agent.conversation.summary is None


def test_threshold_ignores_system_prompt_and_tool_schemas(tmp_path):
    from gigacrew.agent.conversation import Conversation
    from gigacrew.llm.types import Message

    # Бюджет 4000 почти целиком занят системным промптом и схемами: доля считается
    # от остатка под историю, а не от всего бюджета — иначе сжатие на каждом шаге.
    conv = Conversation()
    for i in range(10):
        conv.add_user(f"вопрос {i}")
        conv.add_assistant(Message(role="assistant", content=f"ответ {i} " + "y" * 100))
    provider = ScriptedProvider([text_turn("ок")])
    agent = _agent(provider, tmp_path, context_budget_tokens=4_000)
    agent.reset(conv)
    agent.run_turn("ещё")
    assert provider.summary_requests == []


def test_interrupted_compaction_leaves_conversation_intact(tmp_path):
    steps = [tool_turn("fake_write", {"path": str(i)}) for i in range(4)]
    provider = ScriptedProvider(steps, summaries=[KeyboardInterrupt()])
    events = RecordingEvents()
    agent = _agent(
        provider,
        tmp_path,
        events=events,
        registry=_registry_with(_big_tool()),
        context_budget_tokens=4_000,
    )
    with pytest.raises(KeyboardInterrupt):
        agent.run_turn("ЗАДАЧА")
    assert agent.conversation.summary is None and agent.conversation.pending_calls() == []
    assert events.compactions == [None]


def test_manual_compact_folds_whole_dialog(tmp_path):
    from gigacrew.agent.prompts import SUMMARY_HEADER

    provider = ScriptedProvider(
        [
            text_turn("ответ 1", Usage(prompt_tokens=9_000, completion_tokens=10)),
            text_turn("ответ 2", Usage(prompt_tokens=9_500, completion_tokens=10)),
            text_turn("ответ 3"),
        ],
        summaries=[text_turn("СВОДКА", Usage(prompt_tokens=300, completion_tokens=20))],
    )
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events)
    assert agent.compact() is None  # пустой диалог — сжимать нечего
    agent.run_turn("вопрос 1 " + "x" * 600)
    agent.run_turn("вопрос 2 " + "x" * 600)
    billed = agent.billed_tokens

    result = agent.compact("сохрани имена файлов")
    assert result is not None and not result.auto and result.messages == 4
    assert "сохрани имена файлов" in provider.summary_requests[0]["messages"][1].content
    assert agent.conversation.summary.upto == 4 and agent.conversation.context_messages() == []
    # Оценка вместо устаревшего размера контекста (9 510) до следующего запроса.
    assert agent.context_tokens == result.after_tokens < result.before_tokens < 9_510
    assert agent.billed_tokens == billed + 320
    assert agent.compact() is None  # нового с прошлого сжатия нет

    agent.run_turn("дальше")
    sent = provider.requests[-1]["messages"]
    assert [(m.role, m.content) for m in sent[1:]] == [("user", "дальше")]
    assert SUMMARY_HEADER in sent[0].content and "СВОДКА" in sent[0].content


def test_compaction_that_does_not_shrink_is_rejected(tmp_path):
    from gigacrew.agent.compaction import CompactionError

    provider = ScriptedProvider([text_turn("да")], summaries=[text_turn("многословно " * 500)])
    agent = _agent(provider, tmp_path)
    agent.run_turn("да?")
    with pytest.raises(CompactionError):
        agent.compact()
    assert agent.conversation.summary is None


# ------------------------- инструкции подкаталогов ------------------------- #
def test_subdirectory_instructions_join_system_message(tmp_path):
    from gigacrew.agent.prompts import NESTED_HEADER

    (tmp_path / "api").mkdir()
    (tmp_path / "api" / "AGENTS.md").write_text("В api только async-код.", encoding="utf-8")
    (tmp_path / "api" / "views.py").write_text("x = 1\n", encoding="utf-8")
    provider = ScriptedProvider(
        [
            tool_turn("read_file", {"path": "api/missing.py"}),  # неудачный вызов — не в счёт
            tool_turn("read_file", {"path": "api/views.py"}),
            tool_turn("list_dir", {"path": "api"}),
            text_turn("готово"),
            text_turn("снова"),
        ]
    )
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events, auto_approve=False)
    agent.run_turn("посмотри api")

    systems = [r["messages"][0].content for r in provider.requests]
    assert [NESTED_HEADER in text for text in systems] == [False, False, True, True]
    assert "В api только async-код." in systems[-1]
    assert [text for _, text in events.notices] == [
        "подключены инструкции подкаталога: api/AGENTS.md"
    ]
    # Результат инструмента не дублирует инструкции — они в системном сообщении.
    assert all("async-код" not in m.content for m in agent.conversation.messages)

    agent.reset()  # новый диалог — инструкции подключатся заново при обращении
    agent.run_turn("ещё")
    assert NESTED_HEADER not in provider.requests[-1]["messages"][0].content


def test_resumed_compacted_chat_reports_estimated_context(tmp_path):
    from gigacrew.agent.conversation import Conversation, Summary

    saved = Conversation()
    saved.add_user("вопрос " + "x" * 3_000)
    saved.add_assistant(text_turn("ответ").message, Usage(prompt_tokens=9_000))
    saved.set_summary(Summary("СВОДКА", 2))  # чат сохранён после /compact
    agent = _agent(ScriptedProvider(), tmp_path)
    agent.reset(saved)
    estimate = agent.context_tokens
    assert 0 < estimate < 9_000
    assert agent.context_tokens == estimate  # посчитано один раз


# --------------------------- итог при остановке --------------------------- #
def test_stop_by_guard_asks_model_for_summary(tmp_path):
    bad = [tool_turn("read_file", {"path": f"nope{i}.txt"}) for i in range(4)]
    summary = "Не нашёл файлов nope*.txt. Проверьте путь — дальше могу поискать по шаблону."
    provider = ScriptedProvider([*bad, text_turn(summary)])
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events, max_tool_failures=4)
    assert agent.run_turn("прочитай") == summary
    assert events.stats[-1].stop_reason == "tool_failures"
    last = agent.conversation.messages[-1]
    assert last.role == "assistant" and last.content == summary and not last.function_calls
    assert ("error", events.notices[-1][1]) == events.notices[-1]  # причина остановки видна
    _assert_well_formed(agent.conversation.messages)


def test_max_steps_summary_and_no_tool_execution(tmp_path):
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    read = tool_turn("read_file", {"path": "f.txt"})
    provider = ScriptedProvider([read, text_turn("Прочитал f.txt, дальше — правка.")])
    agent = _agent(provider, tmp_path, max_steps=1)
    assert agent.run_turn("x") == "Прочитал f.txt, дальше — правка."
    assert agent.last_turn.stop_reason == "max_steps"
    assert len(_function_results(agent)) == 1


def test_summary_failure_keeps_guard_message(tmp_path):
    from gigacrew.llm.base import LLMError

    bad = [tool_turn("read_file", {"path": f"nope{i}.txt"}) for i in range(2)]
    provider = ScriptedProvider([*bad, LLMError("сеть")])
    agent = _agent(provider, tmp_path, max_tool_failures=2)
    assert "Прервано: 2 неудачных вызовов" in agent.run_turn("x")


# --------------------------- калибровка оценки --------------------------- #
def test_token_estimate_is_calibrated_by_usage(tmp_path):
    from gigacrew.llm.model_windows import ModelWindows

    provider = ScriptedProvider(
        # Размеры правдоподобны: реальных токенов меньше оценки в 2+ раза, затем больше.
        [text_turn("a", Usage(prompt_tokens=2_500)), text_turn("b", Usage(prompt_tokens=10**6))]
    )
    cfg = Config(access_key="x", project_root=tmp_path, stream=False, model="M")
    windows = ModelWindows(entries={"M": {"context_window": 40_000}})
    agent = Agent(provider, build_default_registry(), cfg, RecordingEvents(), windows=windows)
    specs = build_default_registry().specs()
    budget, history = agent.context_budget, agent._history_budget(specs)
    agent.run_turn("x")  # реальных токенов меньше оценки — бюджет в оценке растёт
    assert agent.token_scale == 0.5
    assert agent.context_budget == budget  # статус-строка — в реальных токенах, как раньше
    assert agent._history_budget(specs) > history * 1.5
    agent.run_turn("y")  # больше оценки — бюджет сразу сжимается
    assert agent.token_scale == 1.5


def test_explicit_budget_is_not_scaled(tmp_path):
    provider = ScriptedProvider([text_turn("a", Usage(prompt_tokens=2_500))])
    agent = _agent(provider, tmp_path, context_budget_tokens=20_000)
    agent.run_turn("x")
    assert agent._estimate_budget() == 20_000
    # Статус-строка сравнивает реальные токены с бюджетом в тех же единицах.
    assert agent.token_scale == 0.5 and agent.context_budget == 10_000


# --------------------------- счётчик контекста --------------------------- #
def test_context_counts_cached_part_of_prompt(tmp_path):
    """API не включил закешированную часть в prompt_tokens: контекст — их сумма."""
    usage = Usage(prompt_tokens=300, precached_prompt_tokens=8_000, completion_tokens=50)
    events = RecordingEvents()
    agent = _agent(ScriptedProvider([text_turn("a", usage)]), tmp_path, events=events)
    agent.run_turn("x")
    assert agent.context_tokens == events.stats[-1].context_tokens == 8_350
    assert agent.token_scale > 1  # калибровка — по полному размеру запроса
    assert agent.billed_tokens == 350  # оплата — как сообщил API


def test_context_without_prompt_usage_falls_back_to_estimate(tmp_path):
    """Usage без размера запроса не обнуляет контекст и не сдвигает калибровку."""
    provider = ScriptedProvider(
        [
            text_turn("a", Usage(prompt_tokens=7_000, completion_tokens=10)),
            text_turn("b", Usage(completion_tokens=900)),
            text_turn("c", Usage(prompt_tokens=100, completion_tokens=900)),
        ]
    )
    agent = _agent(provider, tmp_path)
    agent.run_turn("1")
    scale = agent.token_scale
    for text in ("2", "3"):
        agent.run_turn(text)
        assert agent.context_tokens > 6_000
        assert agent.token_scale == scale


def test_context_after_model_switch_is_estimated(tmp_path):
    from gigacrew.llm.model_windows import ModelWindows

    provider = ScriptedProvider([text_turn("a", Usage(prompt_tokens=20_000, completion_tokens=5))])
    cfg = Config(access_key="x", project_root=tmp_path, stream=False, model="A")
    windows = ModelWindows(entries={"A": {"context_window": 40_000}})
    agent = Agent(provider, build_default_registry(), cfg, RecordingEvents(), windows=windows)
    agent.run_turn("x")
    assert agent.context_tokens == 20_005
    agent.set_model("B")  # другой токенизатор: usage модели A не годится
    estimate = agent.context_tokens
    assert 0 < estimate < 20_000
    agent.set_model("A")
    assert agent.context_tokens == 20_005


def test_resumed_chat_usage_belongs_to_its_model(tmp_path):
    from gigacrew.agent.conversation import Conversation

    saved = Conversation()
    saved.add_user("вопрос")
    # usage заметно больше оценки (системный промпт + схемы инструментов, ~7k)
    saved.add_assistant(
        text_turn("ответ").message, Usage(prompt_tokens=10_000, completion_tokens=5)
    )
    agent = _agent(ScriptedProvider(), tmp_path, model="A")
    agent.set_model("B")  # resume_chat: модель чата — до reset()
    agent.reset(saved)
    assert agent.context_tokens == 10_005
    agent.set_model("A")
    assert agent.context_tokens < 10_000  # оценка в токенах модели A


def test_resumed_chat_with_implausible_usage_is_estimated(tmp_path):
    from gigacrew.agent.conversation import Conversation

    saved = Conversation()
    saved.add_user("вопрос")
    saved.add_assistant(text_turn("ответ").message, Usage(prompt_tokens=300, completion_tokens=5))
    agent = _agent(ScriptedProvider(), tmp_path)
    agent.reset(saved)
    assert agent.context_tokens > 1_000


# --------------------------- «мягкие» ошибки команд --------------------------- #
def test_nonzero_exit_codes_do_not_stop_the_turn(tmp_path):
    turns = [tool_turn("run_shell", {"command": f"exit {code}"}) for code in (1, 2, 1, 3, 1)]
    provider = ScriptedProvider([*turns, text_turn("ок")])
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events, max_tool_failures=4)
    assert agent.run_turn("x") == "ок"
    assert events.stats[-1].stop_reason is None
    assert all(not result.ok for _, result, _ in events.results)  # в UI — по-прежнему неуспех
    # и не пугают модель: «ещё одна неудача остановит работу» не появляется
    assert not any("tool calls failed" in tail for tail in _tails(provider))


def test_missing_commands_still_count_as_failures(tmp_path):
    turns = [tool_turn("run_shell", {"command": f"no-such-cmd-{i}"}) for i in range(4)]
    provider = ScriptedProvider([*turns, text_turn("ок")])
    events = RecordingEvents()
    _agent(provider, tmp_path, events=events, max_tool_failures=4).run_turn("x")
    assert events.stats[-1].stop_reason == "tool_failures"


# --------------------------- прерванный поток --------------------------- #
class _BrokenStream(ScriptedProvider):
    def __init__(self, error):
        super().__init__()
        self.error = error

    def stream(self, messages, tools=None, *, on_delta=None, **kwargs):
        on_delta("Начинаю: сначала прочитаю")
        raise self.error


@pytest.mark.parametrize("error", [KeyboardInterrupt(), RuntimeError("сеть")])
def test_interrupted_stream_keeps_shown_text(tmp_path, error):
    cfg = Config(access_key="x", project_root=tmp_path, stream=True)
    agent = Agent(_BrokenStream(error), build_default_registry(), cfg, RecordingEvents())
    with pytest.raises(type(error)):
        agent.run_turn("x")
    last = agent.conversation.messages[-1]
    assert last.role == "assistant" and last.content.startswith("Начинаю: сначала прочитаю")
    note = "прерван пользователем" if isinstance(error, KeyboardInterrupt) else "ошибки"
    assert note in last.content


def test_invalid_call_is_shown_with_tool_line(tmp_path):
    provider = ScriptedProvider(
        [tool_turn("read_file", {}), tool_turn("nope", {}), text_turn("ок")]
    )
    events = RecordingEvents()
    _agent(provider, tmp_path, events=events).run_turn("x")
    kinds = [(kind, getattr(value, "summary", None)) for kind, value in events.events]
    assert ("tool_call", "(неверные аргументы)") in kinds
    assert ("tool_call", "(неизвестный инструмент)") in kinds
    first_call = kinds.index(("tool_call", "(неверные аргументы)"))
    assert kinds[first_call + 1][0] == "tool_result"


# --------------------------- «всегда» и -y для опасного --------------------------- #
class _ApprovingEvents(RecordingEvents):
    def __init__(self, answers):
        super().__init__()
        self._answers = list(answers)

    def confirm(self, call, preview, *, dangerous):
        self.confirms.append((call, preview, dangerous))
        return self._answers.pop(0)


def _script(tmp_path: Path, name: str = "build.sh") -> str:
    path = tmp_path / name
    path.write_text("#!/bin/sh\necho built\n")
    path.chmod(0o755)
    return f"./{name}"


def test_always_for_command_allows_only_the_same_call(tmp_path):
    from gigacrew.agent.events import Approval

    script = _script(tmp_path)
    run = tool_turn("run_shell", {"command": f"{script} one"})
    other = tool_turn("run_shell", {"command": f"{script} two"})
    provider = ScriptedProvider([run, text_turn("ок"), run, other, text_turn("ок")])
    events = _ApprovingEvents([Approval.ALWAYS, Approval.YES])
    agent = _agent(provider, tmp_path, events=events, auto_approve=False)
    agent.run_turn("x")
    agent.run_turn("ещё")  # разрешение действует и в следующих ходах
    assert [call.summary for call, _, _ in events.confirms] == [f"{script} one", f"{script} two"]
    assert agent.mode is PermissionMode.AUTO
    assert "запомнится" in events.confirms[0][0].remember  # вопрос говорит, что запомнится
    assert any("разрешено сохранённым правилом" in text for _, text in events.notices)


def test_always_for_other_tools_lasts_the_session(tmp_path):
    from gigacrew.agent.events import Approval

    tool = FakeTool()
    same = tool_turn("fake_write", {"path": "a"})
    other = tool_turn("fake_write", {"path": "b"})
    provider = ScriptedProvider([same, text_turn("ок"), same, other, text_turn("ок")])
    events = _ApprovingEvents([Approval.ALWAYS, Approval.YES])
    agent = _agent(
        provider, tmp_path, events=events, registry=_registry_with(tool), auto_approve=False
    )
    agent.run_turn("x")
    agent.run_turn("ещё")
    assert [call.summary for call, _, _ in events.confirms] == ["a", "b"]  # другие аргументы
    assert tool.runs == 3
    assert any("больше не требует подтверждения" in text for _, text in events.notices)


def test_always_for_unverified_command_is_session_only_and_not_in_plan_mode(tmp_path):
    from gigacrew.agent.decisions import DecisionMemory
    from gigacrew.agent.events import Approval
    from gigacrew.agent.reviewer import Reviewer

    memory, _ = DecisionMemory.load(tmp_path, base=tmp_path / "home")
    run = tool_turn("run_shell", {"command": "frobnicate --all > out.txt"})
    provider = ScriptedProvider([run, text_turn("ок")] * 3)
    events = _ApprovingEvents([Approval.ALWAYS, Approval.NO])
    cfg = Config(access_key="x", project_root=tmp_path, stream=False)
    agent = Agent(
        provider,
        build_default_registry(),
        cfg,
        events,
        decisions=memory,
        reviewer=Reviewer(disabled="выключен"),
    )
    agent.run_turn("x")
    # правило по частям не сверить: «всегда» не обещает проекта и не сохраняется
    assert events.confirms[0][0].remember == "" and memory.rules == []
    agent.run_turn("ещё")
    assert len(events.confirms) == 1  # до конца сессии — без вопроса
    (tmp_path / "out.txt").unlink()
    agent.set_mode(PermissionMode.PLAN)
    agent.run_turn("в плане")
    # в плане «всегда» сессии непроверенную команду не разрешает: она пишет в файл
    assert len(events.confirms) == 2 and not (tmp_path / "out.txt").exists()


def test_read_only_commands_run_without_asking(tmp_path):
    (tmp_path / "a.txt").write_text("hello\n")
    calls = [tool_turn("run_shell", {"command": c}) for c in ("ls", "cat a.txt | wc -l")]
    provider = ScriptedProvider([*calls, text_turn("ок")])
    events = RecordingEvents(confirm_answer=False)
    agent = _agent(provider, tmp_path, events=events, auto_approve=False)
    agent.run_turn("x")
    assert events.confirms == []
    outputs = _function_results(agent)
    assert "a.txt" in outputs[0] and "1" in outputs[1]


def test_safe_commands_can_be_turned_off(tmp_path):
    provider = ScriptedProvider([tool_turn("run_shell", {"command": "ls"}), text_turn("ок")])
    events = RecordingEvents(confirm_answer=True)
    agent = _agent(provider, tmp_path, events=events, auto_approve=False, safe_commands=False)
    agent.run_turn("x")
    assert len(events.confirms) == 1


def test_confirm_shows_reason_and_feedback_reaches_model(tmp_path):
    from gigacrew.agent.events import Approval, ConfirmResult

    provider = ScriptedProvider(
        [tool_turn("run_shell", {"command": "pip install requests"}), text_turn("ок")]
    )
    events = _ApprovingEvents([ConfirmResult(Approval.NO, feedback="используй uv add")])
    agent = _agent(provider, tmp_path, events=events, auto_approve=False)
    agent.run_turn("x")
    call, _, dangerous = events.confirms[0]
    assert "сеть" in call.reason and not dangerous
    results = _function_results(agent)
    assert "REJECTED" in results[0] and "используй uv add" in results[0]


def test_plan_mode_blocks_changing_commands_even_with_yes(tmp_path):
    provider = ScriptedProvider(
        [tool_turn("run_shell", {"command": "touch new.txt"}), text_turn("ок")]
    )
    agent = _agent(provider, tmp_path, auto_approve=True, mode=PermissionMode.PLAN)
    agent.run_turn("x")
    assert not (tmp_path / "new.txt").exists()


def test_plan_mode_asks_for_unverified_commands_even_with_yes(tmp_path):
    provider = ScriptedProvider(
        [tool_turn("run_shell", {"command": "frobnicate --all"}), text_turn("ок")]
    )
    events = RecordingEvents(confirm_answer=False)
    agent = _agent(provider, tmp_path, events=events, auto_approve=True, mode=PermissionMode.PLAN)
    agent.run_turn("x")
    assert len(events.confirms) == 1


def test_plan_mode_blocks_known_changing_commands(tmp_path):
    provider = ScriptedProvider(
        [tool_turn("run_shell", {"command": "touch new.txt"}), text_turn("ок")]
    )
    events = RecordingEvents(confirm_answer=True)
    agent = _agent(provider, tmp_path, events=events, auto_approve=False, mode=PermissionMode.PLAN)
    agent.run_turn("x")
    assert events.confirms == []
    assert not (tmp_path / "new.txt").exists()


@pytest.mark.parametrize("variant", ["ask", "yes", "hook_allow"])
@pytest.mark.parametrize("command", ["rm a.txt", "git init -q"])
def test_plan_mode_blocks_changing_commands_that_are_not_edits(tmp_path, variant, command):
    # touch/mkdir — правки (kind EDIT), их блокирует decide(); rm и git init — нет: их
    # блокирует уровень WRITE в вердикте, даже с -y и при ответе хука allow.
    from test_agent_hooks import FakeHooks

    from gigacrew.agent.hooks import HookResult
    from gigacrew.project.settings import HookEvent

    (tmp_path / "a.txt").write_text("x")
    allow = FakeHooks({HookEvent.PRE_TOOL_USE: HookResult(decision="allow")})
    events = RecordingEvents(confirm_answer=True)
    cfg = Config(
        access_key="x",
        project_root=tmp_path,
        stream=False,
        mode=PermissionMode.PLAN,
        auto_approve=variant == "yes",
    )
    provider = ScriptedProvider([tool_turn("run_shell", {"command": command}), text_turn("ок")])
    hooks = {"hooks": allow} if variant == "hook_allow" else {}
    agent = Agent(provider, build_default_registry(), cfg, events, **hooks)
    agent.chat_id = "chat1"
    agent.run_turn("x")
    assert events.confirms == []
    assert _function_results(agent) == [PLAN_BLOCKED_NOTE.format(path=agent._plan_display())]
    assert (tmp_path / "a.txt").exists() and not (tmp_path / ".git").exists()


def test_plan_mode_does_not_apply_saved_rules_for_unverified_commands(tmp_path):
    # Правило сохранено в «авто»; в плане действуют только правила для того, что не меняет.
    from gigacrew.agent.decisions import DecisionMemory
    from gigacrew.safety import classify

    memory = DecisionMemory()
    assert memory.approve(classify("frobnicate --all", root=tmp_path))
    provider = ScriptedProvider(
        [tool_turn("run_shell", {"command": "frobnicate --all"}), text_turn("ок")]
    )
    events = RecordingEvents(confirm_answer=False)
    cfg = Config(access_key="x", project_root=tmp_path, stream=False, mode=PermissionMode.PLAN)
    Agent(provider, build_default_registry(), cfg, events, decisions=memory).run_turn("x")
    assert len(events.confirms) == 1


def test_ui_without_confirm_approves_nothing(tmp_path):
    from gigacrew.agent.events import AgentEvents, ToolCallInfo

    assert AgentEvents().confirm(ToolCallInfo("x"), None, dangerous=False) is False
    tool = FakeTool()
    provider = ScriptedProvider([tool_turn("fake_write", {}), text_turn("ок")])
    cfg = Config(access_key="x", project_root=tmp_path, stream=False)
    agent = Agent(provider, _registry_with(tool), cfg, AgentEvents())
    agent.run_turn("x")
    assert tool.runs == 0 and _function_results(agent) == [REJECTED_NOTE]


def test_unanswered_prompt_is_not_a_rejection(tmp_path):
    # Ctrl+C или конец ввода на вопросе: в журнале «no_answer», в истории отказов — нет.
    from gigacrew.agent.decisions import DecisionMemory, history_key
    from gigacrew.agent.events import Approval, ConfirmResult
    from gigacrew.safety import classify

    script = _script(tmp_path)
    memory, _ = DecisionMemory.load(tmp_path, base=tmp_path / "home")
    events = RecordingEvents(confirm_answer=ConfirmResult(Approval.NO, answered=False))
    provider = ScriptedProvider([tool_turn("run_shell", {"command": script}), text_turn("ок")])
    cfg = Config(access_key="x", project_root=tmp_path, stream=False)
    Agent(provider, build_default_registry(), cfg, events, decisions=memory).run_turn("x")
    key = history_key(classify(script, root=tmp_path))
    assert memory.history(key) == (0, 0) and not memory.rejected_in_session(key)
    decision = json.loads(memory.journal_path.read_text().splitlines()[0])
    assert (decision["decision"], decision["source"]) == ("no_answer", "user")


def test_ui_learns_whether_a_preview_was_shown(tmp_path):
    tool = FakeTool(preview=lambda: None)
    batch = tools_turn([("fake_write", {"path": "a"}), ("fake_write", {"path": "b"})])
    provider = ScriptedProvider([batch, text_turn("ок"), batch, text_turn("ок")])
    events = RecordingEvents(confirm_answer=True)
    agent = _agent(
        provider, tmp_path, events=events, registry=_registry_with(tool), auto_approve=False
    )
    agent.run_turn("x")
    tool._preview = lambda: Display("+x", kind="diff", title="x")
    agent.run_turn("ещё")
    assert [previewed for _, _, previewed in events.results] == [False, False, True, True]


def test_failed_changing_call_is_not_marked_changed(tmp_path):
    tool = FakeTool(run=lambda: ToolResult(content="boom", ok=False, summary="boom"))
    provider = ScriptedProvider([tool_turn("fake_write", {}), text_turn("ок")])
    events = RecordingEvents()
    _agent(provider, tmp_path, events=events, registry=_registry_with(tool)).run_turn("x")
    assert [result.changed for _, result, _ in events.results] == [False]


def test_auto_mode_runs_commands_that_only_edit_files(tmp_path):
    (tmp_path / "a.txt").write_text("old\n")
    provider = ScriptedProvider(
        [tool_turn("run_shell", {"command": "sed -i 's/old/new/' a.txt"}), text_turn("ок")]
    )
    events = RecordingEvents(confirm_answer=False)
    agent = _agent(provider, tmp_path, events=events, auto_approve=False)
    agent.run_turn("x")
    assert events.confirms == []
    assert (tmp_path / "a.txt").read_text() == "new\n"


def test_saved_approval_survives_new_session(tmp_path):
    from gigacrew.agent.decisions import DecisionMemory
    from gigacrew.agent.events import Approval

    script = _script(tmp_path)
    home = tmp_path / "home"
    memory, _ = DecisionMemory.load(tmp_path, base=home)
    run = tool_turn("run_shell", {"command": script})
    events = _ApprovingEvents([Approval.ALWAYS])
    cfg = Config(access_key="x", project_root=tmp_path, stream=False)
    Agent(
        ScriptedProvider([run, text_turn("ок")]),
        build_default_registry(),
        cfg,
        events,
        decisions=memory,
    ).run_turn("x")
    assert len(events.confirms) == 1
    # новая сессия: память читается заново с диска
    memory2, warning = DecisionMemory.load(tmp_path, base=home)
    assert warning is None and len(memory2.rules) == 1
    events2 = RecordingEvents(confirm_answer=False)
    Agent(
        ScriptedProvider([run, text_turn("ок")]),
        build_default_registry(),
        cfg,
        events2,
        decisions=memory2,
    ).run_turn("x")
    assert events2.confirms == []
    journal = (memory2.journal_path).read_text().splitlines()
    assert any('"decision": "always"' in line for line in journal)
    assert any('"event": "outcome"' in line for line in journal)


def test_yes_flag_still_asks_for_dangerous_commands(tmp_path):
    rm = tool_turn("run_shell", {"command": "rm -rf build"})
    provider = ScriptedProvider([rm, text_turn("ок")])
    events = RecordingEvents(confirm_answer=False)
    agent = _agent(provider, tmp_path, events=events)  # auto_approve=True
    agent.run_turn("почисти")
    assert [dangerous for _, _, dangerous in events.confirms] == [True]
    provider = ScriptedProvider([rm, text_turn("ок")])
    events = RecordingEvents(confirm_answer=False)
    _agent(provider, tmp_path, events=events, yes_all=True).run_turn("почисти")
    assert events.confirms == []


class _DangerTool(FakeTool):
    def risk(self, params, ctx):
        from gigacrew.security import RiskLevel

        return RiskLevel.DANGEROUS


def test_always_for_dangerous_call_is_a_one_time_yes(tmp_path):
    # Интерфейс «всегда» для опасного не предлагает; если ответ всё же пришёл — это «да»
    # один раз: вызов выполняется, а модель не слышит, что пользователь отказал.
    from gigacrew.agent.events import Approval

    tool = _DangerTool()
    call = tool_turn("fake_write", {})
    provider = ScriptedProvider([call, text_turn("ок"), call, text_turn("ок")])
    events = RecordingEvents(confirm_answer=Approval.ALWAYS)
    agent = _agent(
        provider, tmp_path, events=events, registry=_registry_with(tool), auto_approve=False
    )
    agent.run_turn("x")
    agent.run_turn("ещё")
    assert [dangerous for _, _, dangerous in events.confirms] == [True, True]
    assert tool.runs == 2 and _function_results(agent) == ["ok", "ok"]
    assert not any("подтверждения" in text for _, text in events.notices)


def test_always_for_dangerous_command_is_never_remembered(tmp_path):
    from gigacrew.agent.decisions import DecisionMemory
    from gigacrew.agent.events import Approval

    memory, _ = DecisionMemory.load(tmp_path, base=tmp_path / "home")
    rm = tool_turn("run_shell", {"command": "rm -rf build"})
    provider = ScriptedProvider([rm, text_turn("ок"), rm, text_turn("ок")])
    events = RecordingEvents(confirm_answer=Approval.ALWAYS)
    cfg = Config(access_key="x", project_root=tmp_path, stream=False)
    agent = Agent(provider, build_default_registry(), cfg, events, decisions=memory)
    for turn in ("почисти", "ещё раз"):
        (tmp_path / "build").mkdir()
        agent.run_turn(turn)
        assert not (tmp_path / "build").exists()
    assert [dangerous for _, _, dangerous in events.confirms] == [True, True]
    assert memory.rules == []
    decisions = [
        json.loads(line)["decision"]
        for line in memory.journal_path.read_text().splitlines()
        if '"event": "decision"' in line
    ]
    assert decisions == ["yes", "yes"]


# Непроверенная команда (уровень None), которую регэксп опасного не узнаёт.
UNVERIFIED_TOUCH = "for f in made.txt; do touch $f; done"


def test_yes_flag_does_not_vouch_for_unverified_commands(tmp_path):
    run = tool_turn("run_shell", {"command": UNVERIFIED_TOUCH})
    provider = ScriptedProvider([run, text_turn("ок")])
    events = RecordingEvents(confirm_answer=False)
    _agent(provider, tmp_path, events=events).run_turn("x")  # auto_approve=True
    # как без -y: сначала рецензент (его ответ не разобран), затем пользователь
    assert len(provider.summary_requests) == 1
    [(call, _, dangerous)] = events.confirms
    assert call.reason.startswith("не удалось проверить:") and not dangerous
    assert not (tmp_path / "made.txt").exists()
    # --yes-all — явное «выполнять всё»
    provider = ScriptedProvider([run, text_turn("ок")])
    events = RecordingEvents(confirm_answer=False)
    _agent(provider, tmp_path, events=events, yes_all=True).run_turn("x")
    assert events.confirms == [] and provider.summary_requests == []
    assert (tmp_path / "made.txt").exists()


def _ask_hook_later():
    """Хук PreToolUse: в первом ходе молчит, со второго требует спрашивать (как после
    ``/hooks reload``)."""
    from test_agent_hooks import FakeHooks

    from gigacrew.agent.hooks import HookResult
    from gigacrew.project.settings import HookEvent

    return FakeHooks({HookEvent.PRE_TOOL_USE: [HookResult(), HookResult(decision="ask")]})


@pytest.mark.parametrize("tool_name", ["run_shell", "fake_write"])
def test_hook_ask_beats_earlier_always(tmp_path, tool_name):
    # «всегда» сохранено правилом (команда) или на сессию (прочее) — хук ask всё равно
    # требует вопроса
    from gigacrew.agent.events import Approval

    args = {"command": _script(tmp_path)} if tool_name == "run_shell" else {"path": "a"}
    call = tool_turn(tool_name, args)
    provider = ScriptedProvider([call, text_turn("ок"), call, text_turn("ок")])
    events = _ApprovingEvents([Approval.ALWAYS, Approval.YES])
    cfg = Config(access_key="x", project_root=tmp_path, stream=False)
    registry = _with_fake(FakeTool())
    agent = Agent(provider, registry, cfg, events, hooks=_ask_hook_later())
    agent.run_turn("x")
    assert agent._decisions.rules or agent._session_allowed
    agent.run_turn("ещё")
    assert len(events.confirms) == 2
    assert "хук PreToolUse требует подтверждения" in events.confirms[1][0].reason


def test_always_under_hook_ask_is_not_remembered(tmp_path):
    from test_agent_hooks import FakeHooks

    from gigacrew.agent.decisions import DecisionMemory
    from gigacrew.agent.events import Approval
    from gigacrew.agent.hooks import HookResult
    from gigacrew.project.settings import HookEvent

    memory, _ = DecisionMemory.load(tmp_path, base=tmp_path / "home")
    hooks = FakeHooks({HookEvent.PRE_TOOL_USE: HookResult(decision="ask")})
    run = tool_turn("run_shell", {"command": _script(tmp_path)})
    events = RecordingEvents(confirm_answer=Approval.ALWAYS)
    cfg = Config(access_key="x", project_root=tmp_path, stream=False)
    agent = Agent(
        ScriptedProvider([run, text_turn("ок")]),
        build_default_registry(),
        cfg,
        events,
        decisions=memory,
        hooks=hooks,
    )
    agent.run_turn("x")
    assert "built" in _function_results(agent)[0]  # выполнено один раз
    assert memory.rules == [] and agent._session_allowed == set()
    assert not any("без вопросов" in t or "больше не требует" in t for _, t in events.notices)


def test_user_prompt_submit_json_block_stops_the_prompt(tmp_path):
    import shlex

    from gigacrew.agent.hooks import HookRunner, PromptBlocked
    from gigacrew.project.settings import HookCommand, HookEvent, HookGroup, HookSettings

    script = tmp_path / "hook.py"
    script.write_text("import json\nprint(json.dumps({'decision': 'block', 'reason': 'секрет'}))\n")
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"
    group = HookGroup(HookEvent.USER_PROMPT_SUBMIT, "", (HookCommand(command),))
    provider = ScriptedProvider([text_turn("ответ")])
    cfg = Config(access_key="x", project_root=tmp_path, stream=False)
    hooks = HookRunner(HookSettings((group,)))
    agent = Agent(provider, build_default_registry(), cfg, RecordingEvents(), hooks=hooks)
    with pytest.raises(PromptBlocked, match="секрет"):
        agent.run_turn("мой ключ sk-123")
    assert provider.calls == 0


@pytest.mark.parametrize("mode", [PermissionMode.AUTO, PermissionMode.PLAN])
def test_hook_allow_does_not_vouch_for_unverified_commands(tmp_path, mode):
    from test_agent_hooks import FakeHooks

    from gigacrew.agent.hooks import HookResult
    from gigacrew.project.settings import HookEvent

    hooks = FakeHooks({HookEvent.PRE_TOOL_USE: HookResult(decision="allow")})
    provider = ScriptedProvider([tool_turn("run_shell", {"command": UNVERIFIED_TOUCH})])
    events = RecordingEvents(confirm_answer=False)
    cfg = Config(access_key="x", project_root=tmp_path, stream=False, mode=mode)
    agent = Agent(provider, build_default_registry(), cfg, events, hooks=hooks)
    agent.run_turn("x")
    assert len(events.confirms) == 1
    assert not (tmp_path / "made.txt").exists()


def test_over_long_command_is_dangerous_even_with_yes(tmp_path):
    # Длиннее предела классификатор команду не разбирает; регэксп по ней не ручается.
    command = "touch made.txt  # " + "x" * 10_000
    provider = ScriptedProvider([tool_turn("run_shell", {"command": command}), text_turn("ок")])
    events = RecordingEvents(confirm_answer=False)
    _agent(provider, tmp_path, events=events).run_turn("x")  # auto_approve=True
    [(call, _, dangerous)] = events.confirms
    assert dangerous and call.remember == ""
    assert call.reason == "не удалось проверить: слишком длинная команда"
    assert not (tmp_path / "made.txt").exists()


# ------------------------------ оборванный ответ ---------------------------- #
@pytest.mark.parametrize("reason", ["length", "incomplete"])
def test_cut_reply_is_retried_with_note_at_the_tail(tmp_path, reason):
    """«Сейчас запишу план» с потерянным вызовом — не финальный ответ: повтор с заметкой."""
    provider = ScriptedProvider(
        [
            cut_turn("Сейчас запишу план", reason),
            tool_turn("write_file", {"path": "a.txt", "content": "x"}),
            text_turn("готово"),
        ]
    )
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events)
    assert agent.run_turn("сделай") == "готово"
    retry = provider.requests[1]["messages"]
    assert retry[-1].role == "user" and "# Reply cut off" in retry[-1].content
    assert f"finish_reason: {reason}" in retry[-1].content
    assert "# Reply cut off" not in retry[0].content
    assert "# Reply cut off" not in provider.requests[2]["messages"][-1].content
    assert all(m.content != "Сейчас запишу план" for m in agent.conversation.messages)
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "x"
    assert events.stats[-1].cut_retries == 1
    assert any(level == "warn" and reason in text for level, text in events.notices)


def test_replies_cut_again_and_again_stop_the_turn(tmp_path):
    provider = ScriptedProvider(
        [cut_turn("a"), cut_turn("b"), cut_turn("c"), text_turn("успел немного")]
    )
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events)
    assert agent.run_turn("сделай") == "успел немного"  # итог после остановки
    assert provider.calls == 4
    wrap_up = provider.requests[-1]
    assert wrap_up["tool_choice"] == "none"
    assert "your replies were cut off" in wrap_up["messages"][-1].content
    stats = events.stats[-1]
    assert (stats.stop_reason, stats.cut_retries) == ("output_limit", 2)
    assert any(level == "error" and "оборвался" in text for level, text in events.notices)


def test_stop_reply_is_final_even_if_it_announces_action(tmp_path):
    """Без разбора текста: обычный stop — финальный ответ, сколько бы он ни обещал."""
    provider = ScriptedProvider([text_turn("Сейчас запишу план")])
    agent = _agent(provider, tmp_path)
    assert agent.run_turn("сделай") == "Сейчас запишу план"
    assert provider.calls == 1


def test_note_goes_to_a_copy_of_the_last_message(tmp_path):
    agent = _agent(ScriptedProvider(), tmp_path)
    specs = build_default_registry().specs()
    agent.conversation.add_user("вопрос")
    agent._build_request(specs)
    plain = agent._last_estimate
    messages = agent._build_request(specs, note="# Заметка")
    reminder = "<system-reminder>\n# Заметка\n</system-reminder>"
    assert messages[-1].content == f"вопрос\n\n{reminder}"
    assert "# Заметка" not in messages[0].content
    assert agent.conversation.messages[-1].content == "вопрос"  # диалог не меняется
    assert agent._last_estimate - plain == estimate_text_tokens(reminder)  # учтена один раз


def test_note_falls_back_to_system_when_history_ends_with_answer(tmp_path):
    agent = _agent(ScriptedProvider(), tmp_path)
    agent.conversation.add_user("вопрос")
    agent.conversation.add_assistant(Message(role="assistant", content="ответ"))
    messages = agent._build_request(build_default_registry().specs(), note="# Заметка")
    assert "# Заметка" in messages[0].content and messages[-1].content == "ответ"
