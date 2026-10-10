"""Тесты истории диалога, окна контекста и ограничителей цикла."""

from __future__ import annotations

import json

import pytest

from gigacrew.agent.context_window import estimate_tokens, fit_history
from gigacrew.agent.conversation import Conversation, Summary
from gigacrew.agent.guard import LoopGuard, ToolOutcome, call_key
from gigacrew.llm.types import FunctionCall, Message, Usage


def _call(name="read_file", **args):
    return Message(
        role="assistant",
        function_call=FunctionCall(name=name, arguments=args),
        functions_state_id="s",
    )


def _result(name="read_file", content="ok"):
    return Message(role="function", name=name, content=content)


# ------------------------------ Conversation ------------------------------ #
def test_round_trip_preserves_everything():
    conv = Conversation()
    conv.add_user("привет")
    conv.add_assistant(_call(path="a.py"), Usage(prompt_tokens=10, completion_tokens=2))
    conv.add_function_result("read_file", "1\tx")
    conv.add_assistant(Message(role="assistant", content="готово"))

    data = json.loads(json.dumps(conv.to_dict(), ensure_ascii=False))
    restored = Conversation.from_dict(data)
    assert restored.messages == conv.messages
    assert restored.last_usage == Usage(prompt_tokens=10, completion_tokens=2)


def test_from_dict_rejects_unknown_version():
    with pytest.raises(ValueError):
        Conversation.from_dict({"version": 999, "messages": []})


def _mid_turn() -> Conversation:
    conv = Conversation()
    conv.add_user("старый запрос")
    conv.add_assistant(Message(role="assistant", content="старый ответ"))
    conv.add_user("ЗАДАЧА")
    for i in range(3):
        conv.add_assistant(_call(path=f"{i}.py"), Usage(prompt_tokens=100))
        conv.add_function_result("read_file", f"файл {i}")
    return conv


def test_summary_round_trip_and_old_files_without_it():
    conv = _mid_turn()
    conv.set_summary(Summary("сводка", 5))
    data = json.loads(json.dumps(conv.to_dict(), ensure_ascii=False))
    assert data["version"] == 1  # ключ summary необязателен — формат прежний
    restored = Conversation.from_dict(data)
    assert restored.summary == Summary("сводка", 5)
    assert restored.messages == conv.messages

    del data["summary"]
    assert Conversation.from_dict(data).summary is None


@pytest.mark.parametrize(
    "summary",
    [
        {"text": "x", "upto": 0},  # граница до начала
        {"text": "x", "upto": 99},  # за концом журнала
        {"text": "x", "upto": 4},  # на результате инструмента
        {"text": "  ", "upto": 2},  # пустое
        {"text": "x", "upto": "2"},
        {"text": "x", "upto": True},
        "сводка",
    ],
)
def test_from_dict_rejects_broken_summary(summary):
    data = _mid_turn().to_dict()
    data["summary"] = summary
    with pytest.raises(ValueError):
        Conversation.from_dict(data)


def test_context_messages_pin_current_task_mid_turn():
    conv = _mid_turn()
    assert conv.context_messages() == list(conv.messages)
    conv.set_summary(Summary("сводка", 5))  # граница посреди хода: задача до неё
    view = conv.context_messages()
    assert [m.content for m in view[:1]] == ["ЗАДАЧА"]
    assert view[1:] == list(conv.messages[5:])
    assert conv.last_usage is None  # описывал контекст до сжатия

    conv.add_assistant(Message(role="assistant", content="готово"))
    conv.add_user("новый запрос")  # закрепление исчезает с новым запросом
    assert conv.context_messages() == list(conv.messages[5:])


def test_context_messages_empty_after_full_compaction():
    conv = _mid_turn()
    conv.set_summary(Summary("сводка", len(conv)))
    assert conv.context_messages() == []
    conv.add_user("дальше")
    assert [m.content for m in conv.context_messages()] == ["дальше"]


def test_set_summary_does_not_move_back():
    conv = _mid_turn()
    conv.set_summary(Summary("сводка", 5))
    with pytest.raises(ValueError):
        conv.set_summary(Summary("сводка", 3))
    with pytest.raises(ValueError):
        conv.set_summary(Summary("сводка", 6))  # результат инструмента
    assert conv.summary == Summary("сводка", 5)


def test_repair_closes_pending_call():
    conv = Conversation()
    conv.add_user("x")
    assert conv.repair() is False
    conv.add_assistant(_call(path="a"))
    assert conv.pending_call().name == "read_file"
    assert conv.repair() is True
    assert conv.messages[-1].role == "function" and conv.messages[-1].name == "read_file"
    assert conv.pending_call() is None
    assert conv.repair() is False


# ------------------------------ fit_history ------------------------------ #
def _history():
    big = "x" * 3000  # ~1000 токенов
    return [
        Message(role="user", content="старая задача " + big),
        Message(role="assistant", content="старый ответ " + big),
        Message(role="user", content="ТЕКУЩАЯ ЗАДАЧА"),
        _call(path="1"),
        _result(content=big),
        _call(path="2"),
        _result(content=big),
        _call(path="3"),
        _result(content="последний результат"),
    ]


def test_fit_history_unchanged_under_budget():
    history = _history()
    assert fit_history(history, 10**6) == history


def test_fit_history_pins_task_and_keeps_pairs():
    history = _history()
    fitted = fit_history(history, 1500)
    assert estimate_tokens(fitted) <= 1500
    assert any(m.content == "ТЕКУЩАЯ ЗАДАЧА" for m in fitted)
    assert fitted[-1].content == "последний результат"
    assert fitted[0].role != "function"
    for i, m in enumerate(fitted):
        if m.role == "function":
            prev = fitted[i - 1]
            assert prev.role == "assistant" and prev.function_call is not None


def test_fit_history_keeps_pinned_even_over_budget():
    history = _history()
    fitted = fit_history(history, 10)
    assert [m.content for m in fitted if m.role == "user"] == ["ТЕКУЩАЯ ЗАДАЧА"]
    assert fitted[-1].content == "последний результат"


def test_fit_history_drops_leading_orphan_result():
    assert fit_history([_result(), Message(role="user", content="q")], 10**6)[0].role == "user"


# ------------------------------ LoopGuard ------------------------------ #
def test_guard_step_limit():
    guard = LoopGuard(max_steps=2, max_failures=10)
    assert guard.before_step() is None
    assert guard.before_step() is None
    stop = guard.before_step()
    assert stop is not None and stop.kind == "max_steps"


OK = ToolOutcome(ok=True)
FAIL = ToolOutcome(ok=False)
CHANGED = ToolOutcome(ok=True, changed=True)
REJECTED = ToolOutcome(ok=False, rejected=True)


def test_guard_failures_reset_on_success():
    guard = LoopGuard(max_steps=100, max_failures=3)
    call = FunctionCall(name="x")
    assert guard.after_tool(call, FAIL) is None
    assert guard.after_tool(call, FAIL) is None
    assert guard.after_tool(call, OK) is None  # успех сбрасывает серию
    assert guard.after_tool(call, FAIL) is None
    assert guard.after_tool(call, FAIL) is None
    stop = guard.after_tool(call, FAIL)
    assert stop is not None and stop.kind == "tool_failures"


def test_call_key_is_canonical():
    a = FunctionCall(name="f", arguments={"b": 1, "a": "я"})
    b = FunctionCall(name="f", arguments={"a": "я", "b": 1})
    assert call_key(a) == call_key(b)
    assert call_key(a) != call_key(FunctionCall(name="f", arguments={"a": "я", "b": 2}))
    assert call_key(a) != call_key(FunctionCall(name="g", arguments={"a": "я", "b": 1}))


def _run(guard: LoopGuard, call: FunctionCall, outcome: ToolOutcome = OK):
    check = guard.before_tool(call)
    if check.stop is None:
        guard.after_tool(call, outcome)
    return check


@pytest.mark.parametrize("outcome", [OK, CHANGED])
def test_guard_repeats_warn_then_stop(outcome):
    # собственное изменение вызова повтор не сбрасывает (pytest без правок)
    guard = LoopGuard(max_steps=100, max_failures=100, max_repeats=3)
    call = FunctionCall(name="read_file", arguments={"path": "a"})
    checks = [_run(guard, call, outcome) for _ in range(4)]
    assert [c.repeats for c in checks] == [1, 2, 3, 4]
    assert [c.warning is not None for c in checks] == [False, False, True, False]
    assert [c.stop is not None for c in checks[:3]] == [False, False, False]
    assert checks[3].stop is not None and checks[3].stop.kind == "tool_repeats"


def test_guard_change_by_other_call_resets_repeats():
    guard = LoopGuard(max_steps=100, max_failures=100, max_repeats=3)
    test = FunctionCall(name="run_shell", arguments={"command": "pytest"})
    edit = FunctionCall(name="edit_file", arguments={"path": "a.py"})
    for _ in range(5):
        assert _run(guard, test, CHANGED).repeats == 1
        _run(guard, edit, CHANGED)
    # неудачная или безвредная операция между повторами — не изменение
    guard = LoopGuard(max_steps=100, max_failures=100, max_repeats=3)
    _run(guard, test)
    _run(guard, edit, FAIL)
    assert _run(guard, test).repeats == 2


def test_guard_detects_alternating_calls():
    guard = LoopGuard(max_steps=100, max_failures=100, max_repeats=3)
    a = FunctionCall(name="read_file", arguments={"path": "a"})
    b = FunctionCall(name="list_dir", arguments={})
    checks = [_run(guard, call) for call in (a, b, a, b, a, b, a)]
    assert checks[-1].stop is not None and checks[-1].stop.kind == "tool_repeats"


def test_guard_rejection_is_neutral_and_remembered():
    guard = LoopGuard(max_steps=100, max_failures=2, max_repeats=10)
    call = FunctionCall(name="write_file", arguments={"path": "a"})
    assert guard.before_tool(call).rejected_before is False
    assert guard.after_tool(call, FAIL) is None  # серия ошибок: 1
    assert guard.after_tool(call, REJECTED) is None  # отказ не продолжает серию...
    assert guard.before_tool(call).rejected_before is True
    other = FunctionCall(name="write_file", arguments={"path": "b"})
    assert guard.before_tool(other).rejected_before is False
    stop = guard.after_tool(other, FAIL)  # ...и не сбрасывает её
    assert stop is not None and stop.kind == "tool_failures"
