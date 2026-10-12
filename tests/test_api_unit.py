"""Офлайн-тесты сериализации/разбора HTTP-провайдера (без сети), формат API v2."""

from __future__ import annotations

import pytest

from gigacrew.config import Config
from gigacrew.llm.api import ApiError, ApiProvider, chat_url
from gigacrew.llm.types import FunctionCall, Message, ToolSpec


def _provider() -> ApiProvider:
    # Не делает сетевых запросов до вызова complete(); ключ нужен лишь для конструктора.
    cfg = Config(
        access_key="dummy", auth_url="https://auth.test/oauth", base_url="https://api.test/v1"
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
                    }
                ]
            }
        }
    ]
    assert "stream" not in payload and "functions" not in payload
    p.close()


def test_payload_tool_choice_none_forbids_calls():
    """Итог после остановки: схемы остаются (история с вызовами), вызывать нельзя."""
    p = _provider()
    payload = p._payload(
        [Message(role="user", content="q")], [TOOL], model="m", temperature=0, tool_choice="none"
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


def test_several_calls_and_results_serialized_as_one_tool_message():
    """Все результаты вызовов одного ответа — одно сообщение role=tool с тем же
    состоянием и id вызовов (иначе 422 «Function calls count does not match…»)."""
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
                }
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
                }
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
        ]
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
                        }
                    },
                    {
                        "function_call": {
                            "id": "4552",
                            "name": "get_weather",
                            "arguments": {"city": "Казань"},
                        }
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
            }
        ]
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
    assert [c.id for c in turn.message.function_calls] == ["8879", "bd7a"]
    assert turn.message.function_calls[1].arguments == {"city": "Казань"}
    assert turn.message.state_id == "01a12ea7"
    assert turn.finish_reason == "function_call"
    assert turn.usage.prompt_tokens == 445 and turn.usage.completion_tokens == 110


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
    ],
)
def test_consume_sse_error_event_raises(event):
    with pytest.raises(ApiError, match="ошибку в потоке"):
        ApiProvider._consume_sse(_FakeSSE(event), lambda _s: None)
