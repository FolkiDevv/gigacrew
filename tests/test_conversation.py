"""Тесты истории диалога, окна контекста и ограничителей цикла."""

from __future__ import annotations

import json

import pytest

from gigacrew.agent.context_window import (
    MIN_BUDGET_TOKENS,
    budget_for_window,
    estimate_message_tokens,
    estimate_text_tokens,
    estimate_tokens,
    fit_history,
)
from gigacrew.agent.conversation import Conversation, Summary, clean_text
from gigacrew.agent.guard import CallCheck, LoopGuard, ToolOutcome, call_key, target_key
from gigacrew.llm.types import FunctionCall, Message, Usage


def _call(name: str = "read_file", **arguments: object) -> Message:
    return Message(
        role="assistant",
        function_calls=[FunctionCall(name=name, arguments=arguments)],
        state_id="s",
    )


def _tool_reply(name: str = "read_file", content: str = "ok") -> Message:
    return Message(role="function", name=name, content=content)


# ------------------------------ Conversation ------------------------------ #
def test_round_trip_preserves_everything() -> None:
    conv = Conversation()
    conv.add_user("привет")
    conv.add_assistant(_call(path="a.py"), Usage(prompt_tokens=10, completion_tokens=2))
    conv.add_function_result("read_file", "1\tx")
    conv.add_assistant(Message(role="assistant", content="готово"))

    dumped = json.loads(json.dumps(conv.to_dict(), ensure_ascii=False))
    restored = Conversation.from_dict(dumped)
    assert restored.messages == conv.messages
    assert restored.last_usage == Usage(prompt_tokens=10, completion_tokens=2)


def test_from_dict_rejects_unknown_version() -> None:
    with pytest.raises(ValueError, match="Неподдерживаемая версия формата диалога"):
        Conversation.from_dict({"version": 999, "messages": []})


def _mid_turn() -> Conversation:
    conv = Conversation()
    conv.add_user("старый запрос")
    conv.add_assistant(Message(role="assistant", content="старый ответ"))
    conv.add_user("ЗАДАЧА")
    for step in range(3):
        conv.add_assistant(_call(path=f"{step}.py"), Usage(prompt_tokens=100))
        conv.add_function_result("read_file", f"файл {step}")
    return conv


def test_summary_round_trip_and_old_files_without_it() -> None:
    conv = _mid_turn()
    conv.set_summary(Summary("сводка", 5))
    dumped = json.loads(json.dumps(conv.to_dict(), ensure_ascii=False))
    assert dumped["version"] == Conversation.FORMAT_VERSION  # summary — необязательный ключ
    restored = Conversation.from_dict(dumped)
    assert restored.summary == Summary("сводка", 5)
    assert restored.messages == conv.messages

    dumped.pop("summary")
    assert Conversation.from_dict(dumped).summary is None


@pytest.mark.parametrize(
    ("summary", "error"),
    [
        ({"text": "x", "upto": 0}, "граница сжатия вне диалога"),  # граница до начала
        ({"text": "x", "upto": 99}, "граница сжатия вне диалога"),  # за концом журнала
        # на результате инструмента
        ({"text": "x", "upto": 4}, "не может приходиться на результат инструмента"),
        ({"text": "  ", "upto": 2}, "пустое краткое содержание"),  # пустое
        ({"text": "x", "upto": "2"}, "ожидаются text"),
        ({"text": "x", "upto": True}, "ожидаются text"),
        ("сводка", "поле summary — не объект"),
    ],
)
def test_from_dict_rejects_broken_summary(summary: object, error: str) -> None:
    dumped = _mid_turn().to_dict()
    dumped["summary"] = summary
    with pytest.raises(ValueError, match=error):
        Conversation.from_dict(dumped)


def test_context_messages_pin_current_task_mid_turn() -> None:
    conv = _mid_turn()
    assert conv.context_messages() == list(conv.messages)
    conv.set_summary(Summary("сводка", 5))  # граница посреди хода: задача до неё
    view = conv.context_messages()
    assert [message.content for message in view[:1]] == ["ЗАДАЧА"]
    assert view[1:] == list(conv.messages[5:])
    assert conv.last_usage is None  # описывал контекст до сжатия

    conv.add_assistant(Message(role="assistant", content="готово"))
    conv.add_user("новый запрос")  # закрепление исчезает с новым запросом
    assert conv.context_messages() == list(conv.messages[5:])


def test_context_messages_empty_after_full_compaction() -> None:
    conv = _mid_turn()
    conv.set_summary(Summary("сводка", len(conv)))
    assert conv.context_messages() == []
    conv.add_user("дальше")
    assert [message.content for message in conv.context_messages()] == ["дальше"]


def test_meta_message_is_saved_and_is_not_a_request() -> None:
    conv = Conversation()
    conv.add_user("ЗАДАЧА")
    conv.add_assistant(Message(role="assistant", content="ответ"))
    conv.add_user("<system-reminder>\nплан одобрен\n</system-reminder>", meta=True)
    dumped = json.loads(json.dumps(conv.to_dict(), ensure_ascii=False))
    assert "meta" not in dumped["messages"][0]  # по умолчанию поле не пишется
    restored = Conversation.from_dict(dumped)
    assert [message.meta for message in restored.messages] == [False, False, True]
    assert [message.is_request for message in restored.messages] == [True, False, False]


def test_context_messages_pin_the_task_not_a_meta_message() -> None:
    conv = _mid_turn()
    conv.add_user("<system-reminder>план одобрен</system-reminder>", meta=True)
    conv.add_assistant(_call(path="3.py"))
    conv.add_function_result("read_file", "файл 3")
    conv.set_summary(Summary("сводка", 5))
    # служебное сообщение после границы — не новый запрос: задача остаётся закреплённой
    assert conv.context_messages()[0].content == "ЗАДАЧА"


def test_truncate_drops_the_tail_but_never_summarized_messages() -> None:
    conv = _mid_turn()
    size = len(conv)
    conv.add_user("служебное", meta=True)
    conv.truncate(size)
    assert len(conv) == size
    assert conv.messages[-1].role == "function"
    with pytest.raises(ValueError, match="длина вне диалога"):
        conv.truncate(size + 1)
    conv.set_summary(Summary("сводка", 5))
    with pytest.raises(ValueError, match="нельзя убрать сообщения"):
        conv.truncate(4)
    conv.truncate(5)
    assert len(conv) == 5


def test_reasoning_is_estimated_and_forgotten() -> None:
    plain = Message(role="assistant", content="ответ")
    thinking = plain.model_copy(update={"reasoning": "р" * 300})
    assert estimate_tokens([thinking]) > estimate_tokens([plain]) + 100
    conv = Conversation()
    conv.add_user("вопрос")
    conv.add_assistant(thinking)
    conv.forget_reasoning()
    assert conv.messages[-1] == plain


def test_set_summary_does_not_move_back() -> None:
    conv = _mid_turn()
    conv.set_summary(Summary("сводка", 5))
    with pytest.raises(ValueError, match="не может сдвигаться назад"):
        conv.set_summary(Summary("сводка", 3))
    with pytest.raises(ValueError, match="на результат инструмента"):
        conv.set_summary(Summary("сводка", 6))  # результат инструмента
    assert conv.summary == Summary("сводка", 5)


def test_repair_closes_pending_call() -> None:
    conv = Conversation()
    conv.add_user("x")
    assert conv.repair() is False
    conv.add_assistant(_call(path="a"))
    assert [call.name for call in conv.pending_calls()] == ["read_file"]
    assert conv.repair() is True
    assert conv.messages[-1].role == "function"
    assert conv.messages[-1].name == "read_file"
    assert conv.pending_calls() == []
    assert conv.repair() is False


def test_several_calls_get_results_in_order_with_ids() -> None:
    conv = Conversation()
    conv.add_user("x")
    calls = [
        FunctionCall(name="read_file", arguments={"path": "a"}, id="c1"),
        FunctionCall(name="list_dir", arguments={"path": "."}, id="c2"),
        FunctionCall(name="read_file", arguments={"path": "b"}, id="c3"),
    ]
    conv.add_assistant(Message(role="assistant", function_calls=calls, state_id="S"))
    assert conv.pending_calls() == calls

    conv.add_function_result("read_file", "A")  # id подставляется по порядку
    assert conv.messages[-1].call_id == "c1"
    assert conv.pending_calls() == calls[1:]

    assert conv.repair("прервано") is True  # закрывает все оставшиеся
    tail = conv.messages[-2:]
    assert [(message.name, message.call_id, message.content) for message in tail] == [
        ("list_dir", "c2", "прервано"),
        ("read_file", "c3", "прервано"),
    ]
    assert conv.pending_calls() == []


def test_pending_calls_only_for_last_assistant_message() -> None:
    conv = Conversation()
    conv.add_user("x")
    conv.add_assistant(_call(path="a"))
    conv.add_user("новый запрос")  # вызов без результата раньше — не «висящий»
    assert conv.pending_calls() == []


def test_loads_format_v1_chat_with_single_function_call() -> None:
    """Чат, сохранённый до API v2: один ``function_call``, состояние старого API."""
    stored = {
        "version": 1,
        "messages": [
            {"role": "user", "content": "x"},
            {
                "role": "assistant",
                "content": "Прочитаю.",
                "function_call": {"name": "read_file", "arguments": {"path": "a"}},
                "functions_state_id": "old-state",
            },
            {"role": "function", "name": "read_file", "content": "1\tx"},
        ],
        "last_usage": None,
    }
    conv = Conversation.from_dict(stored)
    assistant = conv.messages[1]
    assert assistant.function_calls == [FunctionCall(name="read_file", arguments={"path": "a"})]
    assert assistant.state_id is None  # состояние старого API в новом не действует
    assert conv.pending_calls() == []

    saved = conv.to_dict()
    assert saved["version"] == Conversation.FORMAT_VERSION
    assert "function_call" not in saved["messages"][1]
    assert Conversation.from_dict(saved).messages == conv.messages


# ------------------------------ fit_history ------------------------------ #
def _history() -> list[Message]:
    big = "x" * 3000  # ~1000 токенов
    return [
        Message(role="user", content=f"старая задача {big}"),
        Message(role="assistant", content=f"старый ответ {big}"),
        Message(role="user", content="ТЕКУЩАЯ ЗАДАЧА"),
        _call(path="1"),
        _tool_reply(content=big),
        _call(path="2"),
        _tool_reply(content=big),
        _call(path="3"),
        _tool_reply(content="последний результат"),
    ]


def test_fit_history_unchanged_under_budget() -> None:
    history = _history()
    assert fit_history(history, 10**6) == history


def test_fit_history_pins_task_and_keeps_pairs() -> None:
    history = _history()
    fitted = fit_history(history, 1500)
    assert estimate_tokens(fitted) <= 1500
    assert any(message.content == "ТЕКУЩАЯ ЗАДАЧА" for message in fitted)
    assert fitted[-1].content == "последний результат"
    assert fitted[0].role != "function"
    for index, message in enumerate(fitted):
        if message.role == "function":
            prev = fitted[index - 1]
            assert prev.role == "assistant"
            assert prev.function_calls
    # что именно отброшено: сначала прошлый ход, затем ранние шаги текущего — не свежие
    assert fitted == [history[2], *history[5:]]


def test_fit_history_drops_call_and_result_only_together() -> None:
    # Бюджет чуть меньше всей истории: отбросить одно сообщение (вызов) хватило бы, но
    # его результат без вызова API не примет — уходит пара целиком.
    history = [
        Message(role="user", content="ЗАДАЧА"),
        _call(path="1"),
        _tool_reply(content="x" * 3000),
        _call(path="2"),
        _tool_reply(content="ok"),
    ]
    assert fit_history(history, estimate_tokens(history) - 5) == [history[0], *history[3:]]


def test_token_estimate_counts_call_arguments_and_reasoning() -> None:
    # Большой write_file — главный расход контекста: аргументы вызова считаются.
    write = _call("write_file", path="a.py", content="y" * 3000)
    assert estimate_message_tokens(write) >= 1000
    thinking = Message(role="assistant", content="ok", reasoning="r" * 3000)
    assert estimate_message_tokens(thinking) >= 1000
    assert estimate_text_tokens("ab") == 1
    assert estimate_text_tokens("abcd") == 2


def test_budget_for_small_window_has_a_floor() -> None:
    assert budget_for_window(8_192) == budget_for_window(0) == MIN_BUDGET_TOKENS
    assert budget_for_window(128_000) > MIN_BUDGET_TOKENS


def test_fit_history_keeps_pinned_even_over_budget() -> None:
    history = _history()
    fitted = fit_history(history, 10)
    assert [message.content for message in fitted if message.role == "user"] == [
        "ТЕКУЩАЯ ЗАДАЧА",
    ]
    assert fitted[-1].content == "последний результат"


def test_fit_history_pins_the_request_not_a_meta_message() -> None:
    history = _history()
    meta = Message(
        role="user",
        content="<system-reminder>план одобрен</system-reminder>",
        meta=True,
    )
    history.insert(5, meta)  # после первого вызова текущего хода
    fitted = fit_history(history, 10)
    assert [message.content for message in fitted if message.is_request] == [
        "ТЕКУЩАЯ ЗАДАЧА",
    ]
    assert fitted[-1].content == "последний результат"


def test_fit_history_drops_leading_orphan_result() -> None:
    assert fit_history([_tool_reply(), Message(role="user", content="q")], 10**6)[0].role == "user"


# ------------------------------ LoopGuard ------------------------------ #
def test_guard_step_limit() -> None:
    guard = LoopGuard(max_steps=2, max_failures=10)
    assert guard.before_step() is None
    assert guard.before_step() is None
    stop = guard.before_step()
    assert stop is not None
    assert stop.kind == "max_steps"


OK = ToolOutcome(ok=True)
FAIL = ToolOutcome(ok=False)
CHANGED = ToolOutcome(ok=True, changed=True)
REJECTED = ToolOutcome(ok=False, rejected=True)


def test_guard_failures_reset_on_success() -> None:
    guard = LoopGuard(max_steps=100, max_failures=3)
    call = FunctionCall(name="x")
    assert guard.after_tool(call, FAIL) is None
    assert guard.after_tool(call, FAIL) is None
    assert guard.after_tool(call, OK) is None  # успех сбрасывает серию
    assert guard.after_tool(call, FAIL) is None
    assert guard.after_tool(call, FAIL) is None
    assert guard.failure_warning() is not None  # следующий запрос — с предупреждением
    stop = guard.after_tool(call, FAIL)
    assert stop is not None
    assert stop.kind == "tool_failures"


def test_guard_warns_one_failure_before_stop() -> None:
    guard = LoopGuard(max_steps=100, max_failures=4)
    call = FunctionCall(name="x")
    for _ in range(2):
        guard.after_tool(call, FAIL)
        assert guard.failure_warning() is None
    guard.after_tool(call, FAIL)
    warning = guard.failure_warning()
    assert warning is not None
    assert "the last 3 tool calls failed" in warning.model
    assert "ещё одна неудача остановит ход" in warning.notice
    # заметка — в каждый запрос, пока серия у предела (ответ мог быть отброшен), строка
    # пользователю — один раз на серию
    again = guard.failure_warning()
    assert again is not None
    assert again.model == warning.model
    assert not again.notice
    guard.after_tool(call, ToolOutcome(ok=False, soft=True))  # мягкая неудача серию не меняет
    assert guard.failure_warning() is not None
    guard.after_tool(call, OK)  # серия прервалась — новая предупреждается снова
    for _ in range(3):
        guard.after_tool(call, FAIL)
    renewed = guard.failure_warning()
    assert renewed is not None
    assert renewed.notice
    stop = guard.after_tool(call, FAIL)
    assert stop is not None
    assert stop.kind == "tool_failures"
    assert LoopGuard(max_steps=1, max_failures=1).failure_warning() is None


def test_guard_stops_on_failures_only_after_the_warning_was_sent() -> None:
    """Пачка вызовов одного ответа перешла предел — стоп только после предупреждения."""
    guard = LoopGuard(max_steps=100, max_failures=4)
    call = FunctionCall(name="x")
    guard.after_tool(call, FAIL)
    assert guard.failure_warning() is None
    for _ in range(4):  # один ответ: четыре неудачи подряд
        assert guard.after_tool(call, FAIL) is None
    warning = guard.failure_warning()  # следующий запрос
    assert warning is not None
    assert "the last 5 tool calls failed" in warning.model
    stop = guard.after_tool(call, FAIL)
    assert stop is not None
    assert stop.kind == "tool_failures"
    # с пределом 1 предупреждать некогда — стоп сразу
    assert LoopGuard(max_steps=1, max_failures=1).after_tool(call, FAIL) is not None


@pytest.mark.parametrize(
    "outcome",
    [REJECTED, ToolOutcome(ok=False, soft=True)],
    ids=["rejected", "soft"],
)
def test_guard_over_the_limit_stops_only_on_a_failure(outcome: ToolOutcome) -> None:
    """Отказ пользователя и мягкая неудача за пределом серии ход не останавливают.

    Серия за пределом (её перешла пачка) и предупреждение ушло: отказ пользователя или
    мягкая неудача — не неудача, ход не останавливают.
    """
    guard = LoopGuard(max_steps=100, max_failures=2)
    call = FunctionCall(name="x")
    for _ in range(3):
        guard.after_tool(call, FAIL)
    assert guard.failure_warning() is not None
    assert guard.after_tool(call, outcome) is None
    assert guard.after_tool(call, OK) is None  # успех сбрасывает серию


def test_call_key_is_canonical() -> None:
    original = FunctionCall(name="f", arguments={"b": 1, "a": "я"})
    reordered = FunctionCall(name="f", arguments={"a": "я", "b": 1})
    assert call_key(original) == call_key(reordered)
    other_value = FunctionCall(name="f", arguments={"a": "я", "b": 2})
    assert call_key(original) != call_key(other_value)
    other_name = FunctionCall(name="g", arguments={"a": "я", "b": 1})
    assert call_key(original) != call_key(other_name)


def test_call_key_ignores_description() -> None:
    """Та же команда с перефразированным пояснением — тот же вызов."""
    described = FunctionCall(name="run_shell", arguments={"command": "ls", "description": "файлы"})
    rephrased = FunctionCall(name="run_shell", arguments={"command": "ls", "description": "список"})
    assert call_key(described) == call_key(rephrased)
    # null — не значение по умолчанию: у str-поля это ошибка валидации, другой вызов
    null_cwd = FunctionCall(name="run_shell", arguments={"command": "ls", "cwd": None})
    assert call_key(null_cwd) != call_key(described)
    null_description = FunctionCall(
        name="run_shell",
        arguments={"command": "ls", "description": None},
    )
    assert call_key(null_description) != call_key(described)
    guard = LoopGuard(max_steps=100, max_failures=100, max_repeats=3)
    checks = [_run(guard, call) for call in (described, rephrased, described, rephrased)]
    assert checks[2].warning is not None
    assert checks[3].stop is not None


def test_target_key_normalizes_paths() -> None:
    def key(path: str) -> str:
        return target_key(FunctionCall(name="read_file", arguments={"path": path}))

    assert key("a.py") == key("./a.py") == key(" a.py ")
    assert key("src/") == key("src")
    assert key(r"src\a.py") == key("src/a.py")
    assert key("a.py") != key("b.py")
    assert key("a b.txt") != key("a  b.txt")  # разные файлы


def _run(guard: LoopGuard, call: FunctionCall, outcome: ToolOutcome = OK) -> CallCheck:
    check = guard.before_tool(call)
    if check.stop is None:
        guard.after_tool(call, outcome)
    return check


@pytest.mark.parametrize("outcome", [OK, CHANGED])
def test_guard_repeats_warn_then_stop(outcome: ToolOutcome) -> None:
    # собственное изменение вызова повтор не сбрасывает (pytest без правок)
    guard = LoopGuard(max_steps=100, max_failures=100, max_repeats=3)
    call = FunctionCall(name="read_file", arguments={"path": "a"})
    checks = [_run(guard, call, outcome) for _ in range(4)]
    assert [check.repeats for check in checks] == [1, 2, 3, 4]
    assert [check.warning is not None for check in checks] == [False, False, True, False]
    assert [check.stop is not None for check in checks[:3]] == [False, False, False]
    assert checks[3].stop is not None
    assert checks[3].stop.kind == "tool_repeats"
    assert checks[2].notice is not None
    assert "3-й раз повторяет вызов" in checks[2].notice


def test_guard_similar_notice_names_the_similar_count() -> None:
    """Предупреждение о похожих вызовах — не «1-й раз повторяет»: счёт похожих и предел."""
    guard = LoopGuard(max_steps=100, max_failures=100, max_similar=5, max_context=10**6)
    calls = [
        FunctionCall(name="read_file", arguments={"path": "a", "start_line": line})
        for line in range(7)
    ]
    checks = [_run(guard, call) for call in calls]
    assert [check.notice is not None for check in checks] == [
        False,
        False,
        False,
        False,
        True,
        True,
        False,
    ]
    assert checks[4].notice == (
        "Модель 5-й раз подряд вызывает read_file с той же целью — предупреждена; "
        "ход остановится на 7-м таком вызове."
    )
    assert checks[6].stop is not None
    assert checks[6].stop.kind == "similar_calls"


def test_guard_change_by_other_call_resets_repeats() -> None:
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


def test_guard_detects_alternating_calls() -> None:
    guard = LoopGuard(max_steps=100, max_failures=100, max_repeats=3)
    read = FunctionCall(name="read_file", arguments={"path": "a"})
    listing = FunctionCall(name="list_dir", arguments={})
    checks = [_run(guard, call) for call in (read, listing, read, listing, read, listing, read)]
    assert checks[-1].stop is not None
    assert checks[-1].stop.kind == "tool_repeats"


def test_guard_rejection_is_neutral_and_remembered() -> None:
    guard = LoopGuard(max_steps=100, max_failures=2, max_repeats=10)
    call = FunctionCall(name="write_file", arguments={"path": "a"})
    assert guard.before_tool(call).rejected_before is False
    assert guard.after_tool(call, FAIL) is None  # серия ошибок: 1
    assert guard.after_tool(call, REJECTED) is None  # отказ не продолжает серию...
    assert guard.before_tool(call).rejected_before is True
    other = FunctionCall(name="write_file", arguments={"path": "b"})
    assert guard.before_tool(other).rejected_before is False
    assert guard.failure_warning() is not None  # серия у предела — запрос с предупреждением
    stop = guard.after_tool(other, FAIL)  # ...и не сбрасывает её
    assert stop is not None
    assert stop.kind == "tool_failures"


def test_lone_surrogates_are_replaced_on_the_way_into_history() -> None:
    # Имя файла с байтами не в UTF-8 Python декодирует в суррогаты (\udcee); такой текст
    # нельзя закодировать в UTF-8 — ни запрос к модели, ни сохранение чата.
    name = b"\xee\xf2\xf7.txt".decode("utf-8", "surrogateescape")
    assert clean_text(name) == "���.txt"
    assert clean_text("обычный текст") == "обычный текст"
    conv = Conversation()
    conv.add_user(f"открой {name}")
    conv.add_assistant(
        Message(
            role="assistant",
            content="смотрю \ud800",
            reasoning="думаю \udfff",
            function_calls=[FunctionCall(name="read_file", arguments={"path": [name]})],
        ),
    )
    conv.add_function_result("list_dir", f"{name}\nb.txt")
    for message in conv.messages:
        message.model_dump_json().encode("utf-8")  # не UnicodeEncodeError
    assert conv.messages[0].content == "открой ���.txt"
    assert conv.messages[1].content == "смотрю �"
    assistant_call = conv.messages[1].function_calls[0]
    assert assistant_call.arguments == {"path": ["���.txt"]}
    assert conv.messages[2].content == "���.txt\nb.txt"
