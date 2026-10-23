"""Офлайн-тесты кодека OpenAI Chat Completions: тело запроса и разбор ответа/потока."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any

import httpx
import pytest

from gigacrew.llm import openai_chat
from gigacrew.llm.transport import ApiError
from gigacrew.llm.types import AssistantTurn, FunctionCall, Message, ToolSpec

TOOL = ToolSpec(
    name="read_file",
    description="Читает файл",
    parameters={"type": "object", "properties": {"path": {"type": "string"}}},
)


def _sse(*chunks: dict[str, Any] | str) -> list[str]:
    """Строки SSE-потока: куски и ``[DONE]``."""
    lines: list[str] = []
    for chunk in chunks:
        payload = chunk if isinstance(chunk, str) else json.dumps(chunk, ensure_ascii=False)
        lines += [f"data: {payload}", ""]
    return lines


def _delta(finish: str | None = None, **delta: Any) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def _call(
    index: int | None = None,
    *,
    call_id: str | None = None,
    name: str | None = None,
    args: str | dict[str, Any] | None = None,
) -> dict[str, Any]:
    function: dict[str, Any] = {}
    raw: dict[str, Any] = {"type": "function", "function": function}
    if index is not None:
        raw["index"] = index
    if call_id is not None:
        raw["id"] = call_id
    if name is not None:
        function["name"] = name
    if args is not None:
        function["arguments"] = args
    return raw


def _consume(
    lines: Iterable[str],
    content_type: str = "text/event-stream",
) -> tuple[AssistantTurn, list[str]]:
    shown: list[str] = []
    turn = openai_chat.consume_stream(lines, shown.append, content_type=content_type)
    return turn, shown


# ------------------------------- запрос ------------------------------- #
def test_history_layout_with_calls_results_and_reasoning() -> None:
    history = [
        Message(role="system", content="sys"),
        Message(role="user", content="что в a и b?"),
        Message(
            role="assistant",
            content="",
            reasoning="прочитаю оба",
            function_calls=[
                FunctionCall(name="read_file", arguments={"path": "a"}, id="c1"),
                FunctionCall(name="read_file", arguments={"path": "б"}, id="c2"),
            ],
        ),
        Message(role="function", name="read_file", content="A", call_id="c1"),
        Message(role="function", name="read_file", content="B", call_id="c2"),
        Message(role="assistant", content="в a — A, в b — B"),
    ]
    sent = openai_chat.messages_payload(history)
    assert sent[:2] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "что в a и b?"},
    ]
    assert sent[2] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "a"}'},
            },
            {
                "id": "c2",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "б"}'},
            },
        ],
        "reasoning_content": "прочитаю оба",
    }
    assert sent[3:] == [
        {"role": "tool", "tool_call_id": "c1", "content": "A"},
        {"role": "tool", "tool_call_id": "c2", "content": "B"},
        {"role": "assistant", "content": "в a — A, в b — B"},
    ]
    assert openai_chat.has_reasoning({"messages": sent})
    without = openai_chat.messages_payload(history, reasoning=False)
    assert "reasoning_content" not in without[2]
    assert not openai_chat.has_reasoning({"messages": without})


def test_results_follow_their_calls_by_position_and_by_id() -> None:
    calls = [FunctionCall(name="f", arguments={"n": num}, id=f"c{num}") for num in (1, 2)]
    history = [
        Message(role="user", content="x"),
        Message(role="assistant", function_calls=calls),
        Message(role="function", name="f", content="второй", call_id="c2"),  # по id
        Message(role="function", name="f", content="первый"),  # без id — по порядку
    ]
    tool = [msg for msg in openai_chat.messages_payload(history) if msg["role"] == "tool"]
    pairs = [(msg["tool_call_id"], msg["content"]) for msg in tool]
    assert pairs == [("c2", "второй"), ("c1", "первый")]


def test_calls_without_ids_get_stable_ids_shared_with_results() -> None:
    """Чаты, сохранённые без id вызовов: id одинаков у вызова и результата и не меняется."""
    same_call = FunctionCall(name="f", arguments={"a": 1})
    history = [
        Message(role="user", content="x"),
        Message(role="assistant", function_calls=[same_call, same_call]),
        Message(role="function", name="f", content="1"),
        Message(role="function", name="f", content="2"),
    ]
    first, second = openai_chat.messages_payload(history), openai_chat.messages_payload(history)
    assert first == second
    ids = [call["id"] for call in first[1]["tool_calls"]]
    assert len(set(ids)) == 2
    assert all(call_id.startswith("call_") for call_id in ids)
    assert [msg["tool_call_id"] for msg in first[2:]] == ids


def test_missing_results_are_stubbed_and_orphan_results_become_user_text() -> None:
    history = [
        Message(role="function", name="grep", content="старый результат"),  # без вызова
        Message(role="user", content="дальше"),
        Message(role="assistant", function_calls=[FunctionCall(name="f", id="c1")]),
        Message(role="user", content="почему молчишь?"),  # результата c1 нет
    ]
    sent = openai_chat.messages_payload(history)
    assert sent[0] == {"role": "user", "content": "Result of grep:\nстарый результат\n\nдальше"}
    assert sent[2] == {"role": "tool", "tool_call_id": "c1", "content": openai_chat.MISSING_RESULT}
    assert sent[3] == {"role": "user", "content": "почему молчишь?"}
    # хвост без результата тоже закрывается
    tail = openai_chat.messages_payload(history[2:3])
    assert tail[-1] == {"role": "tool", "tool_call_id": "c1", "content": openai_chat.MISSING_RESULT}


def test_consecutive_user_messages_are_merged() -> None:
    history = [
        Message(role="user", content="первый"),
        Message(role="user", content="<system-reminder>заметка</system-reminder>", meta=True),
    ]
    assert openai_chat.messages_payload(history) == [
        {"role": "user", "content": "первый\n\n<system-reminder>заметка</system-reminder>"},
    ]


def test_empty_assistant_text_without_calls_is_empty_string() -> None:
    assert openai_chat.messages_payload([Message(role="assistant", content="")]) == [
        {"role": "assistant", "content": ""},
    ]


def test_chat_payload_options_tools_and_stream() -> None:
    msgs = [Message(role="user", content="привет")]
    body = openai_chat.chat_payload(
        msgs,
        [TOOL],
        model="M",
        temperature=0.2,
        stream=True,
        options={"top_p": 0.9, "temperature": 1.0, "model": "чужая"},
    )
    assert body["model"] == "M"
    assert body["temperature"] == pytest.approx(0.2)
    assert body["top_p"] == pytest.approx(0.9)
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["tool_choice"] == "auto"
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Читает файл",
                "parameters": TOOL.parameters,
            },
        },
    ]
    assert "parallel_tool_calls" not in body
    assert "max_tokens" not in body
    plain = openai_chat.chat_payload(
        msgs,
        None,
        model="M",
        temperature=0.2,
        stream=False,
        tool_choice="none",
    )
    assert plain["stream"] is False
    assert not {"tools", "tool_choice", "stream_options"} & plain.keys()
    none = openai_chat.chat_payload(
        msgs,
        [TOOL],
        model="M",
        temperature=0,
        stream=False,
        tool_choice="none",
    )
    assert none["tool_choice"] == "none"


def test_probe_payload_asks_for_one_token() -> None:
    body = openai_chat.probe_payload("текст", "M", stream=True)
    assert body["max_tokens"] == 1
    assert body["messages"] == [{"role": "user", "content": "текст"}]
    assert body["stream_options"] == {"include_usage": True}
    assert "stream_options" not in openai_chat.probe_payload("т", "M", stream=False)


# ------------------------------- ответ ------------------------------- #
@pytest.mark.parametrize(
    ("finish", "calls", "done", "expected", "kept"),
    [
        ("stop", False, True, "stop", 0),
        ("stop", True, True, "function_call", 1),
        ("tool_calls", True, True, "function_call", 1),
        ("tool_calls", False, True, "incomplete", 0),
        ("function_call", True, True, "function_call", 1),
        ("length", True, True, "length", 0),
        ("max_tokens", False, True, "length", 0),
        ("content_filter", True, True, "content_filter", 0),
        ("eos", False, True, "eos", 0),
        ("eos", True, True, "function_call", 1),
        (None, True, True, "function_call", 1),  # [DONE] без причины
        (None, False, True, "stop", 0),
        (None, True, False, "incomplete", 0),  # поток оборван
        (None, False, False, "incomplete", 0),
    ],
)
def test_finish_reason_table(
    finish: str | None,
    *,
    calls: bool,
    done: bool,
    expected: str,
    kept: int,
) -> None:
    chunks: list[dict[str, Any] | str] = [_delta(content="Текст")]
    if calls:
        chunks.append(_delta(tool_calls=[_call(0, call_id="c1", name="f", args='{"a": 1}')]))
    if finish:
        chunks.append(_delta(finish))
    if done:
        chunks.append("[DONE]")
    turn, shown = _consume(_sse(*chunks))
    assert turn.finish_reason == expected
    assert len(turn.message.function_calls) == kept
    assert turn.message.content == "Текст"
    assert shown == ["Текст"]


def test_usage_chunk_ends_the_answer_and_counts_cache() -> None:
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 5,
        "total_tokens": 105,
        "prompt_tokens_details": {"cached_tokens": 64},
    }
    turn, _ = _consume(_sse(_delta(content="ok"), {"choices": [], "usage": usage}))
    assert turn.finish_reason == "stop"
    assert (turn.usage.prompt_tokens, turn.usage.precached_prompt_tokens) == (100, 64)
    flat = openai_chat.ChunkBuilder()
    flat.add({"choices": [], "usage": {"prompt_tokens": 7, "cached_tokens": 3}})
    assert flat.turn().usage.precached_prompt_tokens == 3


def test_calls_streamed_in_pieces_by_index() -> None:
    turn, shown = _consume(
        _sse(
            _delta(tool_calls=[_call(0, call_id="c1", name="read_file", args="")]),
            _delta(tool_calls=[_call(0, args='{"pa')]),
            _delta(tool_calls=[_call(1, call_id="c2", name="grep", args='{"q": "x"}')]),
            _delta(tool_calls=[_call(0, args='th": "a.txt"}')]),
            _delta("tool_calls"),
            "[DONE]",
        ),
    )
    assert turn.finish_reason == "function_call"
    assert not shown
    assert [(call.id, call.name, call.arguments) for call in turn.message.function_calls] == [
        ("c1", "read_file", {"path": "a.txt"}),
        ("c2", "grep", {"q": "x"}),
    ]
    assert not turn.message.content


def test_calls_without_index_follow_id_changes() -> None:
    turn, _ = _consume(
        _sse(
            _delta(tool_calls=[_call(name="f", args='{"a"')]),  # id придёт позже
            _delta(tool_calls=[_call(call_id="c1", args=": 1}")]),
            _delta(tool_calls=[_call(call_id="c2", name="g", args="{}")]),
            _delta(tool_calls=[_call(args="")]),  # без id — к последнему
            _delta("tool_calls"),
        ),
    )
    assert [(call.id, call.name, call.arguments) for call in turn.message.function_calls] == [
        ("c1", "f", {"a": 1}),
        ("c2", "g", {}),
    ]


def test_repeated_name_and_repeated_arguments_are_not_glued() -> None:
    turn, _ = _consume(
        _sse(
            _delta(tool_calls=[_call(0, call_id="c1", name="write_file", args='{"path": "a",')]),
            _delta(tool_calls=[_call(0, name="write_file", args=' "content": "x\ny"}')]),
            # сервер повторил аргументы целиком в последнем куске
            _delta(tool_calls=[_call(0, args=r'{"path": "a", "content": "x\ny"}')]),
            _delta("tool_calls"),
        ),
    )
    calls = turn.message.function_calls
    assert len(calls) == 1
    call = calls[0]
    assert call.name == "write_file"  # не «write_filewrite_file»
    assert call.arguments == {"path": "a", "content": "x\ny"}


def test_new_call_on_a_reused_index_is_not_merged_into_the_old_one() -> None:
    """Сервер даёт каждому параллельному вызову index 0: новый id с именем — новый вызов.

    Иначе второй вызов молча пропал бы (его аргументы приклеились бы к первому).
    """
    turn, _ = _consume(
        _sse(
            _delta(
                tool_calls=[_call(0, call_id="call_A", name="read_file", args='{"path": "a.py"}')],
            ),
            _delta(tool_calls=[_call(0, call_id="call_B", name="run_shell", args='{"command": ')]),
            _delta(tool_calls=[_call(0, args='"pytest -q"}')]),  # продолжение — к последнему
            _delta("tool_calls"),
        ),
    )
    assert [(call.id, call.name, call.arguments) for call in turn.message.function_calls] == [
        ("call_A", "read_file", {"path": "a.py"}),
        ("call_B", "run_shell", {"command": "pytest -q"}),
    ]


def test_fresh_id_without_a_name_continues_the_call() -> None:
    """Сервер, который шлёт новый id в каждом куске одного вызова, не дробит его."""
    turn, _ = _consume(
        _sse(
            _delta(tool_calls=[_call(0, call_id="c1", name="read_file", args='{"path": ')]),
            _delta(tool_calls=[_call(0, call_id="c2", args='"a.py"}')]),
            _delta("tool_calls"),
        ),
    )
    assert [(call.id, call.name, call.arguments) for call in turn.message.function_calls] == [
        ("c1", "read_file", {"path": "a.py"}),
    ]


def test_whole_call_in_one_chunk_and_object_arguments() -> None:
    turn, _ = _consume(
        _sse(
            _delta(
                "tool_calls",
                tool_calls=[
                    _call(0, name="f", args={"a": 1}),  # объект вместо строки
                    _call(1, name="g", args="[1, 2]"),  # не объект — пустые аргументы
                    _call(2, args="{}"),  # без имени — отбрасывается
                ],
            ),
        ),
    )
    calls = turn.message.function_calls
    assert [(call.name, call.arguments) for call in calls] == [("f", {"a": 1}), ("g", {})]
    assert all(call.id and call.id.startswith("call_") for call in calls)  # id — свои


def test_parse_arguments_is_tolerant() -> None:
    assert not openai_chat.parse_arguments("")
    assert openai_chat.parse_arguments('{"a": "x\ny"}') == {"a": "x\ny"}
    assert openai_chat.parse_arguments('{"a": 1}{"a": 1}') == {"a": 1}
    assert not openai_chat.parse_arguments("не json")
    assert not openai_chat.parse_arguments("42")


def test_reasoning_is_collected_but_not_shown() -> None:
    turn, shown = _consume(
        _sse(
            _delta(reasoning_content="думаю "),
            _delta(reasoning="ещё "),
            _delta(thinking="и всё"),
            _delta(content=[{"type": "text", "text": "Ответ"}]),
            _delta("stop"),
        ),
    )
    assert shown == ["Ответ"]
    assert turn.message.content == "Ответ"
    assert turn.message.reasoning == "думаю ещё и всё"


@pytest.mark.parametrize(
    ("pieces", "content"),
    [
        (["</th", "ink>\n\nОтвет"], "Ответ"),
        (["</think>", "\n", "Ответ"], "Ответ"),
        (["<", "b>жирный</b>"], "<b>жирный</b>"),
    ],
)
def test_leaked_reasoning_tag_is_dropped(pieces: list[str], content: str) -> None:
    turn, shown = _consume(_sse(*[_delta(content=piece) for piece in pieces], _delta("stop")))
    assert turn.message.content == content
    assert "".join(shown) == content


def test_only_first_choice_counts() -> None:
    chunk = {
        "choices": [
            {"index": 1, "delta": {"content": "чужой"}},
            {"index": 0, "delta": {"content": "свой"}},
        ],
    }
    turn, _ = _consume(_sse(chunk, _delta("stop")))
    assert turn.message.content == "свой"


@pytest.mark.parametrize(
    "content_type",
    ["application/x-ndjson", "application/stream+json", "", "application/json"],
)
def test_ndjson_and_mislabelled_streams(content_type: str) -> None:
    lines = [json.dumps(_delta(content="При")), json.dumps(_delta("stop", content="вет"))]
    if content_type == "application/json":
        lines = [f"data: {line}" for line in lines]  # SSE с чужим content-type
    turn, shown = _consume(lines, content_type)
    assert shown == ["При", "вет"]
    assert turn.finish_reason == "stop"


def test_whole_json_response_is_parsed_like_a_stream() -> None:
    whole = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            },
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
    }
    pretty = json.dumps(whole, indent=2).splitlines()
    for lines, content_type in ((pretty, "application/json"), (pretty, "")):
        turn, _ = _consume(lines, content_type)
        assert turn.finish_reason == "function_call"
        assert turn.usage.prompt_tokens == 3
    assert openai_chat.parse_response(whole).message.function_calls[0].id == "c1"


def test_whole_json_without_finish_reason_is_complete() -> None:
    whole = {"choices": [{"index": 0, "message": {"content": "ok"}}]}
    assert openai_chat.parse_response(whole).finish_reason == "stop"
    assert _consume([json.dumps(whole)], "application/json")[0].finish_reason == "stop"


@pytest.mark.parametrize(
    "lines",
    [
        _sse({"error": {"message": "overloaded"}}),
        _sse({"object": "error", "message": "bad"}),
        ["event: error", 'data: {"message": "boom"}', ""],
    ],
)
def test_error_frames_raise(lines: list[str]) -> None:
    with pytest.raises(ApiError, match="ошибку в потоке"):
        _consume(lines)


def test_bad_whole_responses_raise() -> None:
    with pytest.raises(ApiError, match="не-JSON"):
        _consume(["<html>oops</html>"], "application/json")
    with pytest.raises(ApiError, match="Неожиданный формат"):
        _consume(['{"result": 1}'], "application/json")
    with pytest.raises(ApiError, match="ошибку"):
        openai_chat.parse_response({"error": {"message": "x"}})
    with pytest.raises(ApiError, match="Неожиданный формат"):
        openai_chat.parse_response({"messages": []})


def test_sse_events_without_blank_lines_between_them() -> None:
    lines = [f"data: {json.dumps(_delta(content=text))}" for text in ("При", "вет")] + [
        "data: [DONE]",
    ]
    turn, shown = _consume(lines)
    assert shown == ["При", "вет"]
    assert turn.finish_reason == "stop"


def test_garbage_lines_in_stream_are_skipped() -> None:
    turn, shown = _consume([": ping", "", "data: не json", "", *_sse(_delta("stop", content="ok"))])
    assert shown == ["ok"]
    assert turn.finish_reason == "stop"


def test_nothing_after_done_is_read() -> None:
    """После ``[DONE]`` поток не читается: мусор нестандартного сервера не показывается."""

    def lines() -> Iterator[str]:
        yield from _sse(_delta(content="ok"), "[DONE]", _delta(content="мусор"))
        raise AssertionError("поток читается после [DONE]")

    turn, shown = _consume(lines())
    assert shown == ["ok"]
    assert turn.message.content == "ok"
    assert turn.finish_reason == "stop"


def _cut(lines: list[str]) -> Iterator[str]:
    """Строки потока, после которых соединение рвётся."""
    yield from lines
    raise httpx.RemoteProtocolError("peer closed connection without sending complete body")


def test_drop_after_finish_reason_keeps_the_answer() -> None:
    """Ответ закончен (finish_reason), осталось только usage и ``[DONE]``: обрыв — не ошибка."""
    call = _call(0, call_id="c1", name="read_file", args='{"path": "a.py"}')
    turn, shown = _consume(
        _cut(_sse(_delta(content="Читаю"), _delta(tool_calls=[call]), _delta("tool_calls"))),
    )
    assert shown == ["Читаю"]
    assert turn.finish_reason == "function_call"
    assert turn.message.function_calls == [
        FunctionCall(name="read_file", arguments={"path": "a.py"}, id="c1"),
    ]


def test_drop_before_the_end_of_the_answer_is_an_error() -> None:
    with pytest.raises(httpx.RemoteProtocolError):
        _consume(_cut(_sse(_delta(content="Читаю"))))
