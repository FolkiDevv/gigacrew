"""Офлайн-тесты агентного цикла (без сети): поток, подтверждения, анти-залипание."""

from __future__ import annotations

from pathlib import Path

import pytest
from fakes import (
    FakeTool,
    RecordingEvents,
    ScriptedProvider,
    text_turn,
    tool_turn,
    tools_turn,
)

from gigacrew.agent.conversation import INTERRUPTED_NOTE
from gigacrew.agent.loop import (
    LOOP_STOP_NOTE,
    PLAN_BLOCKED_NOTE,
    PLAN_NOT_SUBMITTED_NOTE,
    REJECTED_AGAIN_NOTE,
    REJECTED_NOTE,
    SKIPPED_NOTE,
    Agent,
)
from gigacrew.config import Config
from gigacrew.errors import ToolError
from gigacrew.llm.types import Usage
from gigacrew.permissions import PermissionMode, ToolKind
from gigacrew.tools.base import Display, ToolRegistry, ToolResult, build_default_registry
from gigacrew.tools.plan import APPROVE_EDITS
from gigacrew.tools.questions import Answer
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
    assert "# Turn stopped" in provider.requests[-1]["messages"][0].content
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
                "read_file", {"path": "f.txt"}, Usage(prompt_tokens=100, completion_tokens=10)
            ),
            text_turn("ок", Usage(prompt_tokens=130, completion_tokens=20)),
        ]
    )
    events = RecordingEvents()
    _agent(provider, tmp_path, events=events).run_turn("x")
    stats = events.stats[-1]
    assert (stats.steps, stats.tool_calls) == (2, 1)
    assert stats.billed_tokens == 260
    assert stats.context_tokens == 150


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
    provider = ScriptedProvider([batch, text_turn("итог")])
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events, max_tool_failures=2)

    assert agent.run_turn("x") == "итог"
    assert events.stats[-1].stop_reason == "tool_failures"
    results = [m for m in agent.conversation.messages if m.role == "function"]
    assert [r.call_id for r in results] == ["c1", "c2", "c3"]
    assert results[2].content == SKIPPED_NOTE  # третий вызов не выполнялся
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
        text_turn("старый ответ").message, Usage(prompt_tokens=50, completion_tokens=5)
    )
    provider = ScriptedProvider([text_turn("новый ответ")])
    agent = _agent(provider, tmp_path)
    agent.reset(saved)
    assert agent.conversation is saved and agent.context_tokens == 55
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
            text_turn("a", Usage(prompt_tokens=100, completion_tokens=10)),
            text_turn("b", Usage(prompt_tokens=200, completion_tokens=20)),
        ]
    )
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events)
    assert agent.context_tokens == 0
    agent.run_turn("1")
    assert agent.context_tokens == 110 and agent.billed_tokens == 110
    agent.reset()
    assert agent.context_tokens == 0 and agent.billed_tokens == 110
    agent.run_turn("2")
    assert agent.context_tokens == 220 and agent.billed_tokens == 330
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
    assert _function_results(agent)[-1] == PLAN_BLOCKED_NOTE


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


def test_accept_edits_applies_edits_without_asking(tmp_path):
    tool = FakeTool(kind=ToolKind.EDIT)
    provider = ScriptedProvider([tool_turn("fake_write", {}), text_turn("ок")])
    events = RecordingEvents(confirm_answer=False)
    agent = _agent(
        provider,
        tmp_path,
        events=events,
        registry=_registry_with(tool),
        auto_approve=False,
        mode=PermissionMode.ACCEPT_EDITS,
    )
    agent.run_turn("поправь")
    assert tool.runs == 1 and events.confirms == []


@pytest.mark.parametrize("kind", [ToolKind.COMMAND, ToolKind.OTHER])
def test_accept_edits_still_asks_for_the_rest(tmp_path, kind):
    tool = FakeTool(kind=kind)
    provider = ScriptedProvider([tool_turn("fake_write", {}), text_turn("ок")])
    events = RecordingEvents(confirm_answer=False)
    agent = _agent(
        provider,
        tmp_path,
        events=events,
        registry=_registry_with(tool),
        auto_approve=False,
        mode=PermissionMode.ACCEPT_EDITS,
    )
    agent.run_turn("x")
    assert tool.runs == 0 and len(events.confirms) == 1


def test_plan_rules_are_sent_only_in_plan_mode(tmp_path):
    provider = ScriptedProvider([text_turn("план"), text_turn("готово")])
    agent = _agent(provider, tmp_path, mode=PermissionMode.PLAN)
    agent.run_turn("спланируй")
    agent.set_mode(PermissionMode.MANUAL)
    agent.run_turn("делай")
    planning, manual = _system_prompts(provider)
    assert "# Plan mode" in planning
    assert "# Plan mode" not in manual


def test_mode_survives_reset_and_cycles(tmp_path):
    agent = _agent(ScriptedProvider(), tmp_path)
    assert agent.mode is PermissionMode.MANUAL
    assert agent.cycle_mode() is PermissionMode.ACCEPT_EDITS
    agent.reset()
    assert agent.mode is PermissionMode.ACCEPT_EDITS
    assert [agent.cycle_mode(), agent.cycle_mode()] == [PermissionMode.PLAN, PermissionMode.MANUAL]


PLAN_REL = ".gigacrew/plans/chat1.md"


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
    events = RecordingEvents(confirm_answer=False, answers=[Answer((APPROVE_EDITS,))])
    agent = _planning_agent(provider, tmp_path, events)
    assert agent.run_turn("спланируй и сделай") == "готово"
    assert agent.mode is PermissionMode.ACCEPT_EDITS
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "x"
    assert events.confirms == []  # ни план в режиме плана, ни правка после одобрения
    (question,) = events.questions[0]
    assert question.body.startswith("# План\n1. создать a.txt")
    # правила плана (с путём файла плана) — только до одобрения
    prompts = _system_prompts(provider)
    assert ["# Plan mode" in p for p in prompts] == [True, True, False, False]
    assert PLAN_REL in prompts[0]


def test_plan_mode_allows_only_the_plan_file(tmp_path):
    provider = ScriptedProvider(
        [
            tool_turn("write_file", {"path": "a.txt", "content": "x"}),
            tool_turn("write_file", {"path": ".gigacrew/plans/other.md", "content": "x"}),
            tool_turn("write_file", {"path": PLAN_REL, "content": "# План"}),
            tool_turn("exit_plan_mode", {}),
            text_turn("жду"),
        ]
    )
    events = RecordingEvents(answers=None)
    agent = _planning_agent(provider, tmp_path, events)
    agent.run_turn("спланируй")
    results = _function_results(agent)
    assert results[0] == PLAN_BLOCKED_NOTE
    assert results[1] == PLAN_BLOCKED_NOTE  # чужой план — обычная правка
    assert (tmp_path / PLAN_REL).read_text(encoding="utf-8") == "# План"
    assert not (tmp_path / "a.txt").exists() and events.confirms == []


def test_unsubmitted_plan_gets_one_reminder(tmp_path):
    provider = ScriptedProvider(
        [
            tool_turn("write_file", {"path": PLAN_REL, "content": "# План"}),
            text_turn("Вот план: ..."),  # забыла exit_plan_mode
            text_turn("План готов, см. файл"),  # и после напоминания — ответ принимается
        ]
    )
    events = RecordingEvents()
    agent = _planning_agent(provider, tmp_path, events)
    assert agent.run_turn("спланируй") == "План готов, см. файл"
    assert len(provider.requests) == 3
    reminder = provider.requests[2]["messages"]
    assert PLAN_NOT_SUBMITTED_NOTE in reminder[0].content
    assert PLAN_NOT_SUBMITTED_NOTE not in provider.requests[1]["messages"][0].content
    # текст без вызова в историю не попал: запрос не кончается репликой ассистента
    assert reminder[-1].role == "function"
    assert [m.content for m in agent.conversation.messages if m.role == "assistant"][-1] == (
        "План готов, см. файл"
    )
    assert any("не отправлен" in text for _, text in events.notices)


def test_reminder_leads_to_submission(tmp_path):
    provider = ScriptedProvider(
        [
            tool_turn("write_file", {"path": PLAN_REL, "content": "# План"}),
            text_turn("Вот план"),
            tool_turn("exit_plan_mode", {}),
            text_turn("жду указаний"),
        ]
    )
    events = RecordingEvents(answers=None)
    agent = _planning_agent(provider, tmp_path, events)
    assert agent.run_turn("спланируй") == "жду указаний"
    assert len(events.questions) == 1  # план показан пользователю
    assert PLAN_NOT_SUBMITTED_NOTE not in provider.requests[3]["messages"][0].content


@pytest.mark.parametrize("mode", [PermissionMode.PLAN, PermissionMode.MANUAL])
def test_no_reminder_without_plan_changes(tmp_path, mode):
    provider = ScriptedProvider([text_turn("ответ на вопрос")])
    agent = _agent(provider, tmp_path, mode=mode)
    assert agent.run_turn("что делает foo?") == "ответ на вопрос"
    assert len(provider.requests) == 1


def test_submitted_plan_needs_no_reminder(tmp_path):
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
    assert agent.plan_path == tmp_path.resolve() / ".gigacrew" / "plans" / "a1.md"
    agent.reset(chat_id="b2")
    assert agent.chat_id == "b2" and agent.plan_path.name == "b2.md"
    agent.reset()
    assert agent.chat_id not in ("a1", "b2")


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
        [text_turn("a", Usage(prompt_tokens=1)), text_turn("b", Usage(prompt_tokens=10**6))]
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
    provider = ScriptedProvider([text_turn("a", Usage(prompt_tokens=1))])
    agent = _agent(provider, tmp_path, context_budget_tokens=20_000)
    agent.run_turn("x")
    assert agent._estimate_budget() == 20_000


# --------------------------- «мягкие» ошибки команд --------------------------- #
def test_nonzero_exit_codes_do_not_stop_the_turn(tmp_path):
    turns = [tool_turn("run_shell", {"command": f"exit {code}"}) for code in (1, 2, 1, 3, 1)]
    provider = ScriptedProvider([*turns, text_turn("ок")])
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, events=events, max_tool_failures=4)
    assert agent.run_turn("x") == "ок"
    assert events.stats[-1].stop_reason is None
    assert all(not result.ok for _, result, _ in events.results)  # в UI — по-прежнему неуспех


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


def test_always_for_edit_switches_to_accept_edits(tmp_path):
    from gigacrew.agent.events import Approval

    writes = [tool_turn("write_file", {"path": f"f{i}.txt", "content": "x"}) for i in range(2)]
    provider = ScriptedProvider([*writes, text_turn("ок")])
    events = _ApprovingEvents([Approval.ALWAYS])
    agent = _agent(provider, tmp_path, events=events, auto_approve=False)
    agent.run_turn("x")
    assert len(events.confirms) == 1  # вторая правка — уже без вопроса
    assert agent.mode is PermissionMode.ACCEPT_EDITS
    assert (tmp_path / "f1.txt").exists()
    assert any("авто-правки" in text for _, text in events.notices)


def test_always_for_command_allows_only_the_same_call(tmp_path):
    from gigacrew.agent.events import Approval

    echo = tool_turn("run_shell", {"command": "echo 1"})
    other = tool_turn("run_shell", {"command": "echo 2"})
    provider = ScriptedProvider([echo, text_turn("ок"), echo, other, text_turn("ок")])
    events = _ApprovingEvents([Approval.ALWAYS, Approval.YES])
    agent = _agent(provider, tmp_path, events=events, auto_approve=False)
    agent.run_turn("x")
    agent.run_turn("ещё")  # разрешение действует и в следующих ходах
    assert [call.summary for call, _, _ in events.confirms] == ["echo 1", "echo 2"]
    assert agent.mode is PermissionMode.MANUAL


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
