"""Live-тесты реального API модели.

Проверяют все типы обращений к LLM на живом API (v2):
  1. простой запрос-ответ;
  2. вызов функции (tool call), в том числе несколько вызовов за ответ;
  3. многошаговый диалог с возвратом результатов инструментов в модель.

Помечены маркером ``live`` и пропускаются, если нет доступного ключа.
Запуск только live-тестов:  pytest -m live
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from gigacrew.config import Config
from gigacrew.llm.api import ApiProvider
from gigacrew.llm.types import FunctionCall, Message, ToolSpec

pytestmark = pytest.mark.live

# Температура в ответе на первый вызов; у следующих — на градус выше.
_FIRST_TEMPERATURE = 10


@pytest.fixture(scope="module")
def provider(live_config: Config) -> Iterator[ApiProvider]:
    api = ApiProvider(live_config)
    yield api
    api.close()


def test_simple_completion(provider: ApiProvider) -> None:
    """Тип 1: простой запрос-ответ."""
    turn = provider.complete(
        [Message(role="user", content="Ответь ровно одним словом: столица России?")],
    )
    assert turn.finish_reason == "stop"
    assert "Москва" in turn.message.content
    assert turn.usage.total_tokens > 0


def test_models_listing(provider: ApiProvider) -> None:
    """Список — чат-модели шлюза, и модель тестов среди них."""
    models = provider.list_models()
    assert provider.model in models, f"{provider.model} нет в списке моделей: {models}"
    assert all(isinstance(model, str) for model in models)
    assert all(models)
    assert len(set(models)) == len(models)


def test_streaming_text(provider: ApiProvider) -> None:
    """Потоковая генерация: дельты приходят и собираются в полный ответ."""
    deltas: list[str] = []
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


def test_streaming_tool_call(provider: ApiProvider) -> None:
    """Потоковый режим корректно отдаёт вызов функции."""
    turn = provider.stream(
        [Message(role="user", content="Погода в Сочи? Вызови get_weather.")],
        tools=[WEATHER_TOOL],
    )
    assert turn.finish_reason == "function_call"
    assert turn.wants_tool
    fc = turn.message.function_calls[0]
    assert fc.name == "get_weather"
    assert "сочи" in str(fc.arguments.get("city", "")).lower()
    # вызов из потока можно вернуть в историю: id и состояние пришли и здесь
    assert fc.id
    assert turn.message.state_id


WEATHER_TOOL = ToolSpec(
    name="get_weather",
    description="Возвращает текущую погоду в указанном городе",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string", "description": "город"}},
        "required": ["city"],
    },
)


def test_single_tool_call(provider: ApiProvider) -> None:
    """Тип 2: модель запрашивает вызов функции."""
    turn = provider.complete(
        [
            Message(
                role="user",
                content="Какая погода в Москве? Обязательно вызови функцию get_weather.",
            ),
        ],
        tools=[WEATHER_TOOL],
    )
    assert turn.finish_reason == "function_call"
    assert turn.wants_tool
    fc = turn.message.function_calls[0]
    assert fc.name == "get_weather"
    assert isinstance(fc.arguments, dict)
    assert "москв" in str(fc.arguments.get("city", "")).lower()
    # API возвращает id вызова и непрозрачный идентификатор состояния инструментов
    assert fc.id
    assert turn.message.state_id


def test_multistep_with_tool_result(provider: ApiProvider) -> None:
    """Тип 3: возврат результата инструмента обратно в модель."""
    messages = [
        Message(
            role="user",
            content="Какая температура в Москве? Вызови get_weather, потом ответь.",
        ),
    ]
    turn1 = provider.complete(messages, tools=[WEATHER_TOOL])
    assert turn1.wants_tool, "ожидался вызов функции на первом шаге"

    # кладём ход ассистента и результаты всех его вызовов
    messages.append(turn1.message)
    messages.extend(
        Message(
            role="function",
            name=call.name,
            call_id=call.id,
            content='{"city": "Москва", "temperature_c": 17, "condition": "ясно"}',
        )
        for call in turn1.message.function_calls
    )
    turn2 = provider.complete(messages, tools=[WEATHER_TOOL])
    assert turn2.finish_reason == "stop"
    assert not turn2.wants_tool
    # модель должна использовать переданное значение температуры
    assert "17" in turn2.message.content


def _weather_replies(calls: list[FunctionCall]) -> list[Message]:
    """Результаты вызовов get_weather: у каждого следующего температура на градус выше."""
    replies: list[Message] = []
    for index, call in enumerate(calls):
        city = call.arguments.get("city", "?")
        temperature = _FIRST_TEMPERATURE + index
        replies.append(
            Message(
                role="function",
                name=call.name,
                call_id=call.id,
                content=f'{{"city": "{city}", "temperature_c": {temperature}}}',
            ),
        )
    return replies


def test_parallel_calls_round_trip(provider: ApiProvider) -> None:
    """Несколько вызовов за ответ: результаты всех возвращаются одним шагом."""
    messages = [
        Message(
            role="user",
            content=(
                "Узнай погоду в Москве и в Казани. Вызови get_weather для обоих городов "
                "сразу, в одном ответе, потом сравни."
            ),
        ),
    ]
    turn1 = provider.stream(messages, tools=[WEATHER_TOOL])
    calls = turn1.message.function_calls
    if len(calls) < 2:
        pytest.skip(f"модель сделала {len(calls)} вызов(а) — параллельность не проверена")
    assert len({call.id for call in calls}) == len(calls), "у вызовов должны быть разные id"
    cities = " ".join(str(call.arguments.get("city", "")) for call in calls).lower()
    assert "москв" in cities, cities
    assert "казан" in cities, cities
    messages.append(turn1.message)
    messages.extend(_weather_replies(calls))
    turn2 = provider.stream(messages, tools=[WEATHER_TOOL])
    assert turn2.finish_reason == "stop"
    assert not turn2.wants_tool
    # результаты обоих вызовов дошли до модели: в сравнении — обе температуры
    temperatures = [str(_FIRST_TEMPERATURE + shift) for shift in range(len(calls))]
    answer = turn2.message.content
    assert all(temperature in answer for temperature in temperatures), answer


def test_tool_choice_none_answers_with_text(provider: ApiProvider) -> None:
    """Итог после остановки: инструменты в запросе, но модель отвечает только текстом."""
    turn = provider.complete(
        [Message(role="user", content="Какая погода в Москве? Вызови get_weather.")],
        tools=[WEATHER_TOOL],
        tool_choice="none",
    )
    assert not turn.wants_tool
    assert turn.message.function_calls == []
    assert turn.finish_reason == "stop"
    assert turn.message.content.strip()


def test_history_without_call_ids_continues(provider: ApiProvider) -> None:
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
