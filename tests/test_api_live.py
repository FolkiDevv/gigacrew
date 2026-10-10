"""Live-тесты реального API модели.

Проверяют все типы обращений к LLM на живом API:
  1. простой запрос-ответ;
  2. одиночный вызов функции (tool call);
  3. многошаговый диалог с возвратом результата инструмента в модель.

Помечены маркером ``live`` и пропускаются, если нет доступного ключа.
Запуск только live-тестов:  pytest -m live
"""

from __future__ import annotations

import pytest

from gigacrew.llm.api import ApiProvider
from gigacrew.llm.types import Message, ToolSpec

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


def test_streaming_tool_call(provider):
    """Потоковый режим корректно отдаёт вызов функции."""
    turn = provider.stream(
        [Message(role="user", content="Погода в Сочи? Вызови get_weather.")],
        tools=[WEATHER_TOOL],
    )
    assert turn.finish_reason == "function_call"
    assert turn.wants_tool
    assert turn.message.function_call.name == "get_weather"
    assert "city" in turn.message.function_call.arguments


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
    fc = turn.message.function_call
    assert fc is not None
    assert fc.name == "get_weather"
    assert isinstance(fc.arguments, dict)
    assert "city" in fc.arguments
    # API возвращает непрозрачный идентификатор состояния функций
    assert turn.message.functions_state_id


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

    # кладём ход ассистента и результат функции
    messages.append(turn1.message)
    messages.append(
        Message(
            role="function",
            name="get_weather",
            content='{"city": "Москва", "temperature_c": 17, "condition": "ясно"}',
        )
    )
    turn2 = provider.complete(messages, tools=[WEATHER_TOOL])
    assert turn2.finish_reason == "stop"
    # модель должна использовать переданное значение температуры
    assert "17" in turn2.message.content
