"""Live-тесты реального API модели.

Проверяют все типы обращений к LLM на живом API (v2):
  1. простой запрос-ответ;
  2. вызов функции (tool call), в том числе несколько вызовов за ответ;
  3. многошаговый диалог с возвратом результатов инструментов в модель.

Помечены маркером ``live`` и пропускаются, если нет доступного ключа.
Запуск только live-тестов:  pytest -m live
"""

from __future__ import annotations

import pytest

from gigacrew.llm.api import ApiProvider
from gigacrew.llm.types import FunctionCall, Message, ToolSpec

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def provider(live_config):
    p = ApiProvider(live_config)
    yield p
    p.close()


def test_simple_completion(provider):
    """Тип 1: простой запрос-ответ."""
    turn = provider.complete(
        [Message(role="user", content="Ответь ровно одним словом: столица России?")]
    )
    assert turn.finish_reason == "stop"
    assert "Москва" in turn.message.content
    assert turn.usage.total_tokens > 0


def test_models_listing(provider):
    models = provider.list_models()
    assert models


def test_streaming_text(provider):
    """Потоковая генерация: дельты приходят и собираются в полный ответ."""
    deltas = []
    turn = provider.stream(
        [Message(role="user", content="Перечисли через запятую цвета радуги.")],
        on_delta=deltas.append,
    )
    assert turn.finish_reason == "stop"
    assert deltas, "не пришло ни одной дельты"
    # собранный из дельт текст совпадает с финальным сообщением
    assert "".join(deltas) == turn.message.content
    assert turn.message.content.strip()
    # завершающее событие не повторяет уже присланный текст
    assert turn.message.content.lower().count("фиолет") == 1


def test_streaming_tool_call(provider):
    """Потоковый режим корректно отдаёт вызов функции."""
    turn = provider.stream(
        [Message(role="user", content="Погода в Сочи? Вызови get_weather.")],
        tools=[WEATHER_TOOL],
    )
    assert turn.finish_reason == "function_call"
    assert turn.wants_tool
    assert turn.message.function_calls[0].name == "get_weather"
    assert "city" in turn.message.function_calls[0].arguments


WEATHER_TOOL = ToolSpec(
    name="get_weather",
    description="Возвращает текущую погоду в указанном городе",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string", "description": "город"}},
        "required": ["city"],
    },
)


def test_single_tool_call(provider):
    """Тип 2: модель запрашивает вызов функции."""
    turn = provider.complete(
        [
            Message(
                role="user",
                content="Какая погода в Москве? Обязательно вызови функцию get_weather.",
            )
        ],
        tools=[WEATHER_TOOL],
    )
    assert turn.finish_reason == "function_call"
    assert turn.wants_tool
    fc = turn.message.function_calls[0]
    assert fc.name == "get_weather"
    assert isinstance(fc.arguments, dict)
    assert "city" in fc.arguments
    # API возвращает id вызова и непрозрачный идентификатор состояния инструментов
    assert fc.id
    assert turn.message.state_id


def test_multistep_with_tool_result(provider):
    """Тип 3: возврат результата инструмента обратно в модель."""
    messages = [
        Message(
            role="user",
            content="Какая температура в Москве? Вызови get_weather, потом ответь.",
        )
    ]
    turn1 = provider.complete(messages, tools=[WEATHER_TOOL])
    assert turn1.wants_tool, "ожидался вызов функции на первом шаге"

    # кладём ход ассистента и результаты всех его вызовов
    messages.append(turn1.message)
    for call in turn1.message.function_calls:
        messages.append(
            Message(
                role="function",
                name=call.name,
                call_id=call.id,
                content='{"city": "Москва", "temperature_c": 17, "condition": "ясно"}',
            )
        )
    turn2 = provider.complete(messages, tools=[WEATHER_TOOL])
    assert turn2.finish_reason == "stop"
    # модель должна использовать переданное значение температуры
    assert "17" in turn2.message.content


def test_parallel_calls_round_trip(provider):
    """Несколько вызовов за ответ: результаты всех возвращаются одним шагом."""
    messages = [
        Message(
            role="user",
            content=(
                "Узнай погоду в Москве и в Казани. Вызови get_weather для обоих городов "
                "сразу, в одном ответе, потом сравни."
            ),
        )
    ]
    turn1 = provider.stream(messages, tools=[WEATHER_TOOL])
    calls = turn1.message.function_calls
    if len(calls) < 2:
        pytest.skip(f"модель сделала {len(calls)} вызов(а) — параллельность не проверена")
    assert len({c.id for c in calls}) == len(calls), "у вызовов должны быть разные id"
    messages.append(turn1.message)
    for i, call in enumerate(calls):
        city = call.arguments.get("city", "?")
        messages.append(
            Message(
                role="function",
                name=call.name,
                call_id=call.id,
                content=f'{{"city": "{city}", "temperature_c": {10 + i}}}',
            )
        )
    turn2 = provider.stream(messages, tools=[WEATHER_TOOL])
    assert turn2.finish_reason == "stop"
    assert turn2.message.content.strip()


def test_history_without_call_ids_continues(provider):
    """Чат, сохранённый до перехода на v2: у вызова нет id, состояния нет."""
    messages = [
        Message(role="user", content="Какая погода в Москве? Вызови get_weather."),
        Message(
            role="assistant",
            function_calls=[FunctionCall(name="get_weather", arguments={"city": "Москва"})],
        ),
        Message(role="function", name="get_weather", content='{"temperature_c": 17}'),
        Message(role="user", content="Сколько градусов? Ответь числом."),
    ]
    turn = provider.complete(messages, tools=[WEATHER_TOOL])
    assert turn.finish_reason == "stop"
    assert "17" in turn.message.content
