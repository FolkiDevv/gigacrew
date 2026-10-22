"""Офлайн-тесты сериализации/разбора HTTP-провайдера (без сети), формат API v2."""

from __future__ import annotations

import json

import pytest

from gigacrew.config import Config
from gigacrew.llm.api import ApiError, ApiProvider, chat_url
from gigacrew.llm.transport import iter_sse_events
from gigacrew.llm.types import FunctionCall, Message, ToolSpec


def _provider() -> ApiProvider:
    # Не делает сетевых запросов до вызова complete(); ключ нужен лишь для конструктора.
    cfg = Config(
        access_key="dummy",
        auth_url="https://auth.test/oauth",
        base_url="https://api.test/v1",
    )
    return ApiProvider(cfg)


TOOL = ToolSpec(
    name="read_file",
    description="Читает файл",
    parameters={"type": "object", "properties": {"path": {"type": "string"}}},
)


def _two_calls() -> list[Message]:
    """Ход с двумя вызовами и результатами — как в чате «расскажи об этом проекте»."""
    return [
        Message(role="system", content="sys"),
        Message(role="user", content="расскажи об этом проекте"),
        Message(
            role="assistant",
            content="Изучу проект.",
            function_calls=[
                FunctionCall(name="file_outline", arguments={"path": "src"}, id="c1"),
                FunctionCall(name="read_file", arguments={"path": "README.md"}, id="c2"),
            ],
            state_id="S",
        ),
        Message(role="function", name="file_outline", content="src: файлов 100", call_id="c1"),
        Message(role="function", name="read_file", content='{"text": "# demo"}', call_id="c2"),
    ]


# ------------------------------ адрес ------------------------------ #
@pytest.mark.parametrize(
    ("base", "url"),
    [
        ("https://h/api/v1", "https://h/api/v2/chat/completions"),
        ("https://h/v1/", "https://h/v2/chat/completions"),
        ("https://h/v2", "https://h/v2/chat/completions"),
        ("https://h/api", "https://h/api/chat/completions"),  # без /vN — как есть
    ],
)
def test_chat_url_switches_version_to_v2(base, url):
    assert chat_url(base) == url


# ------------------------------ запрос ------------------------------ #
def test_payload_v2_layout_with_tools():
    p = _provider()
    payload = p._payload([Message(role="user", content="q")], [TOOL], model="m", temperature=0.3)
    assert payload["model"] == "m"
    assert payload["messages"] == [{"role": "user", "content": [{"text": "q"}]}]
    assert payload["model_options"] == {"temperature": 0.3, "parallel_tool_calls": True}
    assert payload["tools"] == [
        {
            "functions": {
                "specifications": [
                    {
                        "name": "read_file",
                        "description": "Читает файл",
                        "parameters": TOOL.parameters,
                    },
                ],
            },
        },
    ]
    assert "stream" not in payload
    assert "functions" not in payload
    p.close()


def test_payload_tool_choice_none_forbids_calls():
    """Итог после остановки: схемы остаются (история с вызовами), вызывать нельзя."""
    p = _provider()
    payload = p._payload(
        [Message(role="user", content="q")],
        [TOOL],
        model="m",
        temperature=0,
        tool_choice="none",
    )
    assert payload["tool_config"] == {"mode": "none"}
    assert payload["tools"]
    assert "parallel_tool_calls" not in payload["model_options"]
    p.close()


def test_payload_without_tools_has_no_tool_options():
    p = _provider()
    payload = p._payload([Message(role="user", content="q")], None, model="m", temperature=0)
    assert "tools" not in payload
    assert payload["model_options"] == {"temperature": 0}
    assert p._payload([], None, model="m", temperature=0, stream=True)["stream"] is True
    p.close()


def test_reasoning_goes_before_its_answer():
    """Рассуждения ответа — сообщение role=reasoning перед ним (формат проверен пробой)."""
    call = FunctionCall(name="read_file", arguments={"path": "a"}, id="c1")
    history = [
        Message(role="user", content="?"),
        Message(role="assistant", function_calls=[call], state_id="S", reasoning="надо прочитать"),
        Message(role="function", name="read_file", content="x", call_id="c1"),
        Message(role="assistant", content="готово"),  # без рассуждений
    ]
    sent = ApiProvider._messages_payload(history)
    assert [m["role"] for m in sent] == ["user", "reasoning", "assistant", "tool", "assistant"]
    assert sent[1] == {"role": "reasoning", "content": [{"text": "надо прочитать"}]}
    assert sent[3]["tools_state_id"] == "S"


def test_several_calls_and_results_serialized_as_one_tool_message():
    """Все результаты вызовов одного ответа — одно сообщение role=tool с тем же
    состоянием и id вызовов (иначе 422 «Function calls count does not match…»).
    """
    sent = ApiProvider._messages_payload(_two_calls())
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "tool"]
    assert sent[2] == {
        "role": "assistant",
        "content": [
            {"text": "Изучу проект."},
            {"function_call": {"name": "file_outline", "arguments": {"path": "src"}, "id": "c1"}},
            {
                "function_call": {
                    "name": "read_file",
                    "arguments": {"path": "README.md"},
                    "id": "c2",
                },
            },
        ],
        "tools_state_id": "S",
    }
    assert sent[3] == {
        "role": "tool",
        "content": [
            {
                "function_result": {
                    "id": "c1",
                    "name": "file_outline",
                    "result": {"result": "src: файлов 100"},
                },
            },
            {"function_result": {"id": "c2", "name": "read_file", "result": {"text": "# demo"}}},
        ],
        "tools_state_id": "S",
    }


def test_results_of_different_steps_are_separate_tool_messages():
    history = _two_calls()
    history += [
        Message(
            role="assistant",
            function_calls=[FunctionCall(name="list_dir", id="c3")],
            state_id="S2",
        ),
        Message(role="function", name="list_dir", content="a.py", call_id="c3"),
        Message(role="assistant", content="Готово."),
    ]
    sent = ApiProvider._messages_payload(history)
    assert [m["role"] for m in sent] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert sent[5]["tools_state_id"] == "S2"
    assert len(sent[5]["content"]) == 1
    assert sent[6] == {"role": "assistant", "content": [{"text": "Готово."}]}


def test_history_without_ids_and_state():
    """Чат формата 1 (до v2): у вызова нет id, состояния нет — поля не отправляются."""
    sent = ApiProvider._messages_payload(
        [
            Message(role="user", content="x"),
            Message(role="assistant", function_calls=[FunctionCall(name="read_file")]),
            Message(role="function", name="read_file", content="1\tx"),
        ],
    )
    assert sent[1] == {
        "role": "assistant",
        "content": [{"function_call": {"name": "read_file", "arguments": {}}}],
    }
    assert sent[2] == {
        "role": "tool",
        "content": [{"function_result": {"name": "read_file", "result": {"result": "1\tx"}}}],
    }


def test_empty_assistant_message_has_text_part():
    sent = ApiProvider._messages_payload([Message(role="assistant", content="")])
    assert sent == [{"role": "assistant", "content": [{"text": ""}]}]


@pytest.mark.parametrize("content", ["42", "true", "null", "[]", '"строка"', "1", "текст"])
def test_non_object_results_are_wrapped(content):
    """Не объект — значит текст: API ждёт в result JSON-объект."""
    assert ApiProvider._result_object(content) == {"result": content}


def test_object_result_passed_as_is():
    assert ApiProvider._result_object('{"a": 1}') == {"a": 1}


@pytest.mark.parametrize(
    "content",
    [
        '{"loss": NaN}',
        '{"max": Infinity, "min": -Infinity}',
        '{"big": 1e400}',
        '{"nested": {"items": [1, NaN]}}',
    ],
)
def test_result_with_non_finite_numbers_is_text(content):
    """json.loads принимает NaN/Infinity, а тело запроса — строгий JSON (httpx отверг бы его
    ValueError, и сообщение сломало бы каждый следующий запрос чата): такой результат — текст.
    """
    result = ApiProvider._result_object(content)
    assert result == {"result": content}
    json.dumps(result, allow_nan=False)  # тело запроса собирается


def test_arguments_with_non_finite_numbers_are_dropped():
    """Вызов из ответа уходит обратно в истории: NaN в аргументах сломал бы каждый запрос."""
    data = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"function_call": {"id": "c1", "name": "f", "arguments": '{"x": NaN}'}},
                    {"function_call": {"id": "c2", "name": "g", "arguments": {"y": float("inf")}}},
                    {"function_call": {"id": "c3", "name": "h", "arguments": {"z": 1.5}}},
                ],
            },
        ],
    }
    calls = ApiProvider._parse_response(data).message.function_calls
    assert [(c.name, c.arguments) for c in calls] == [("f", {}), ("g", {}), ("h", {"z": 1.5})]


# ------------------------------ ответ ------------------------------ #
def test_parse_response_with_reasoning_text_and_parallel_calls():
    """Ответ с живого API (curl): рассуждения, текст и два вызова."""
    data = {
        "model": "MiniMax-M3:latest",
        "messages": [
            {"role": "reasoning", "content": [{"text": "Сделаю оба вызова сразу."}]},
            {
                "role": "assistant",
                "tool_state_id": "01a12ea5",
                "content": [
                    {"text": "Вызываю get_weather для обоих городов:"},
                    {
                        "function_call": {
                            "id": "ba12",
                            "name": "get_weather",
                            "arguments": {"city": "Москва"},
                        },
                    },
                    {
                        "function_call": {
                            "id": "4552",
                            "name": "get_weather",
                            "arguments": {"city": "Казань"},
                        },
                    },
                ],
            },
        ],
        "finish_reason": "function_call",
        "usage": {
            "input_tokens": 445,
            "input_tokens_details": {"prompt_tokens": 445, "cached_tokens": 5},
            "output_tokens": 121,
            "total_tokens": 566,
        },
    }
    turn = ApiProvider._parse_response(data)
    assert turn.wants_tool
    assert turn.finish_reason == "function_call"
    assert turn.message.content == "Вызываю get_weather для обоих городов:"  # без рассуждений
    assert turn.message.reasoning == "Сделаю оба вызова сразу."  # отдельно от текста
    assert turn.message.function_calls == [
        FunctionCall(name="get_weather", arguments={"city": "Москва"}, id="ba12"),
        FunctionCall(name="get_weather", arguments={"city": "Казань"}, id="4552"),
    ]
    assert turn.message.state_id == "01a12ea5"
    u = turn.usage
    assert (u.prompt_tokens, u.completion_tokens, u.total_tokens) == (445, 121, 566)
    assert u.precached_prompt_tokens == 5


def test_parse_text_answer_drops_state_id():
    data = {
        "messages": [{"role": "assistant", "tool_state_id": "S", "content": [{"text": "Готово!"}]}],
        "finish_reason": "stop",
    }
    turn = ApiProvider._parse_response(data)
    assert not turn.wants_tool
    assert turn.message.content == "Готово!"
    assert turn.message.state_id is None  # состояние нужно только результатам вызовов
    assert turn.finish_reason == "stop"


def test_parse_response_arguments_as_string():
    """Подстраховка: если arguments прилетит строкой, парсим её."""
    data = {
        "messages": [
            {
                "role": "assistant",
                "content": [{"function_call": {"name": "f", "arguments": '{"k": "v"}'}}],
            },
        ],
    }
    turn = ApiProvider._parse_response(data)
    assert turn.message.function_calls[0].arguments == {"k": "v"}
    assert turn.finish_reason == "function_call"


def test_parse_response_unexpected_format():
    with pytest.raises(ApiError):
        ApiProvider._parse_response({"choices": []})


# ------------------------------ поток ------------------------------ #
class _FakeSSE:
    """Минимальный двойник httpx.Response для проверки разбора SSE."""

    def __init__(self, text: str):
        self._lines = text.split("\n")

    def iter_lines(self):
        return iter(self._lines)


# Поток с живого API (curl): рассуждения, текст, затем оба вызова в done.
PARALLEL_STREAM = """\
event: response.message.delta
data: {"model":"M","messages":[{"role":"reasoning","content":[{"text":"Оба вызова сразу."}]}]}

event: response.message.delta
data: {"model":"M","messages":[{"role":"assistant","content":[{"text":"Получаю погоду "}]}]}

event: response.message.delta
data: {"model":"M","messages":[{"role":"assistant","content":[{"text":"для обоих городов:"}]}]}

event: response.message.done
data: {"model":"M","messages":[{"role":"assistant","tool_state_id":"01a12ea7",\
"content":[{"function_call":{"id":"8879","name":"get_weather","arguments":{"city":"Москва"}}},\
{"function_call":{"id":"bd7a","name":"get_weather","arguments":{"city":"Казань"}}}]}],\
"finish_reason":"function_call","usage":{"input_tokens":445,"output_tokens":110,"total_tokens":555}}

"""


def test_consume_sse_parallel_calls():
    deltas: list[str] = []
    turn = ApiProvider._consume_sse(_FakeSSE(PARALLEL_STREAM), deltas.append)
    assert deltas == ["Получаю погоду ", "для обоих городов:"]  # рассуждения не показываются
    assert turn.message.content == "Получаю погоду для обоих городов:"
    assert turn.message.reasoning == "Оба вызова сразу."
    assert [c.id for c in turn.message.function_calls] == ["8879", "bd7a"]
    assert turn.message.function_calls[1].arguments == {"city": "Казань"}
    assert turn.message.state_id == "01a12ea7"
    assert turn.finish_reason == "function_call"
    assert turn.usage.prompt_tokens == 445
    assert turn.usage.completion_tokens == 110


def test_consume_sse_text_stream():
    text = (
        'event: response.message.delta\ndata: {"messages":[{"role":"assistant",'
        '"content":[{"text":"При"}]}]}\n\n'
        'event: response.message.delta\ndata: {"messages":[{"role":"assistant",'
        '"content":[{"text":"вет"}]}]}\n\n'
        'event: response.message.done\ndata: {"messages":[{"role":"assistant",'
        '"tool_state_id":"S"}],"finish_reason":"stop","usage":{"total_tokens":5}}\n\n'
        "data: [DONE]\n\n"
    )
    deltas: list[str] = []
    turn = ApiProvider._consume_sse(_FakeSSE(text), deltas.append)
    assert deltas == ["При", "вет"]
    assert turn.message.content == "Привет"
    assert turn.finish_reason == "stop"
    assert turn.usage.total_tokens == 5
    assert not turn.wants_tool


_PREAMBLE = (
    'event: response.message.delta\ndata: {"messages":[{"role":"assistant",'
    '"content":[{"text":"Сейчас запишу план"}]}]}\n\n'
)


@pytest.mark.parametrize(
    ("tail", "finish"),
    [
        ("", "incomplete"),  # шлюз закрыл поток посреди ответа: вызов мог потеряться
        ('event: response.message.done\ndata: {"messages":[]}\n\n', "stop"),
        ("data: [DONE]\n\n", "stop"),
        ('data: {"messages":[],"usage":{"total_tokens":5}}\n\n', "stop"),
        (
            'event: response.message.done\ndata: {"messages":[],"finish_reason":"length"}\n\n',
            "length",
        ),
    ],
)
def test_consume_sse_stream_without_end_is_incomplete(tail, finish):
    turn = ApiProvider._consume_sse(_FakeSSE(_PREAMBLE + tail), lambda _s: None)
    assert turn.message.content == "Сейчас запишу план"
    assert turn.finish_reason == finish
    assert not turn.wants_tool


def _delta(text: str) -> str:
    chunk = {"messages": [{"role": "assistant", "content": [{"text": text}]}]}
    return f"event: response.message.delta\ndata: {json.dumps(chunk, ensure_ascii=False)}\n\n"


_DONE = 'event: response.message.done\ndata: {"messages":[],"finish_reason":"stop"}\n\n'


@pytest.mark.parametrize(
    ("pieces", "shown", "content"),
    [
        # шлюз убрал рассуждения, а закрывающий тег оставил в начале ответа (MiniMax)
        (["</mm:", "think>", "\n\nОтвет"], ["Ответ"], "Ответ"),
        (["</mm:think>План записан"], ["План записан"], "План записан"),
        (["\n</think>\n", "Готово"], ["Готово"], "Готово"),
        # не тег — текст как есть, с исходными пробелами
        (["  При", "вет"], ["  При", "вет"], "  Привет"),
        (["<b>жирный</b>"], ["<b>жирный</b>"], "<b>жирный</b>"),
        (["</", "mm:", "thinking"], ["</mm:thinking"], "</mm:thinking"),
        (["Ответ </think>"], ["Ответ </think>"], "Ответ </think>"),
        # ответ кончился, так и не став тегом, — в истории он целиком
        (["</mm"], [], "</mm"),
    ],
)
def test_leaked_reasoning_tag_is_dropped_from_the_start(pieces, shown, content):
    stream = "".join(_delta(piece) for piece in pieces) + _DONE
    deltas: list[str] = []
    turn = ApiProvider._consume_sse(_FakeSSE(stream), deltas.append)
    assert deltas == shown
    assert turn.message.content == content


def test_leaked_reasoning_tag_is_dropped_from_whole_response():
    data = {"messages": [{"role": "assistant", "content": [{"text": "</mm:think>Ответ"}]}]}
    assert ApiProvider._parse_response(data).message.content == "Ответ"


def test_parse_response_without_finish_reason_is_complete():
    data = {"messages": [{"role": "assistant", "content": [{"text": "ok"}]}]}
    assert ApiProvider._parse_response(data).finish_reason == "stop"


def test_consume_sse_repeated_call_id_is_not_duplicated():
    call = '{"function_call":{"id":"c1","name":"f","arguments":{"a":1}}}'
    text = (
        f'event: response.message.delta\ndata: {{"messages":[{{"role":"assistant",'
        f'"content":[{call}]}}]}}\n\n'
        f'event: response.message.done\ndata: {{"messages":[{{"role":"assistant",'
        f'"content":[{call}]}}],"finish_reason":"function_call"}}\n\n'
    )
    turn = ApiProvider._consume_sse(_FakeSSE(text), lambda _s: None)
    assert turn.message.function_calls == [FunctionCall(name="f", arguments={"a": 1}, id="c1")]


def test_consume_sse_multiline_data_and_comments():
    text = (
        ": keep-alive\n"
        'data: {"messages":[{"role":"assistant",\n'
        'data: "content":[{"text":"ok"}]}]}\n\n'
    )
    turn = ApiProvider._consume_sse(_FakeSSE(text), lambda _s: None)
    assert turn.message.content == "ok"


@pytest.mark.parametrize(
    "event",
    [
        'event: error\ndata: {"message":"boom"}\n\n',
        'data: {"status":422,"message":"Function calls count does not match"}\n\n',
        'data: {"status":400,"message":"bad request"}\n\n',  # граница: 400 — уже ошибка
        'data: {"error":{"message":"overloaded"}}\n\n',
    ],
)
def test_consume_sse_error_event_raises(event):
    with pytest.raises(ApiError, match="ошибку в потоке"):
        ApiProvider._consume_sse(_FakeSSE(event), lambda _s: None)


def test_consume_sse_status_below_400_is_not_an_error():
    stream = 'data: {"status":200}\n\n' + _delta("ok") + _DONE
    assert ApiProvider._consume_sse(_FakeSSE(stream), lambda _s: None).message.content == "ok"


def test_sse_event_without_blank_separator_is_not_merged():
    """Строка ``event:`` без пустой строки перед ней начинает новое событие: склеенный
    JSON не разобрался бы и молча пропал бы — вместе с текстом, вызовами или окончанием.
    """
    assert list(iter_sse_events(["event: a", "data: 1", "event: b", "data: 2", ""])) == [
        ("a", "1"),
        ("b", "2"),
    ]
    stream = _delta("ok").rstrip("\n") + "\n" + _DONE
    deltas: list[str] = []
    turn = ApiProvider._consume_sse(_FakeSSE(stream), deltas.append)
    assert deltas == ["ok"]
    assert turn.finish_reason == "stop"


def test_parallel_calls_without_ids_are_all_kept():
    data = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"function_call": {"name": "f", "arguments": {"a": 1}}},
                    {"function_call": {"name": "g", "arguments": {"b": 2}}},
                ],
            },
        ],
    }
    assert ApiProvider._parse_response(data).message.function_calls == [
        FunctionCall(name="f", arguments={"a": 1}),
        FunctionCall(name="g", arguments={"b": 2}),
    ]


@pytest.mark.parametrize("key", ["tool_state_id", "tools_state_id"])
def test_state_id_under_either_spelling(key):
    data = {
        "messages": [
            {
                "role": "assistant",
                key: "S",
                "content": [{"function_call": {"id": "c1", "name": "f", "arguments": {}}}],
            },
        ],
    }
    assert ApiProvider._parse_response(data).message.state_id == "S"


def test_malformed_parts_are_skipped():
    """Мусор в ответе — пропускается, а не роняет разбор; аргументы-не-JSON — {}."""
    data = {
        "messages": [
            "junk",
            {"role": "user", "content": [{"text": "чужое"}]},
            {
                "role": "assistant",
                "content": ["x", {"function_call": {"name": "f", "arguments": "{bad"}}],
            },
        ],
    }
    turn = ApiProvider._parse_response(data)
    assert turn.message.function_calls == [FunctionCall(name="f", arguments={})]
    assert turn.message.content == ""


def test_provider_without_catalogue_defaults():
    """Провайдер без каталога: подключаться не к чему, модель — любая."""
    p = _provider()
    p.connect()  # ничего не делает и не ходит в сеть
    assert p.known_models() is None
    p.close()
