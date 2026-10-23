"""Окно контекста: оценка размера истории и отбор сообщений под бюджет.

Это «скользящее окно»: старые сообщения отбрасываются целиком. Основной
механизм — сжатие (:mod:`gigacrew.agent.compaction`): начало диалога заменяется
кратким содержанием раньше, чем история упрётся в бюджет. Окно остаётся
страховкой — на случай, если сжатие выключено, не удалось или один шаг слишком велик.

Инварианты :func:`fit_history`:
  * последний запрос пользователя (текущая задача) никогда не отбрасывается;
  * вызов инструмента и его результат отбрасываются только вместе;
  * история никогда не начинается с результата инструмента.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence

from gigacrew.llm.types import Message, ToolSpec

# Грубая оценка для русского/кода: ~3 символа на токен (с запасом).
CHARS_PER_TOKEN = 3
MESSAGE_OVERHEAD_TOKENS = 4

# Окно, если провайдер не знает модель: консервативно, как у самых старых моделей.
DEFAULT_CONTEXT_WINDOW = 32_000
# Место под ответ модели: max_tokens в запросе не задаётся.
RESPONSE_RESERVE_TOKENS = 8_000
# Доля окна, отдаваемая под запрос, — запас на погрешность оценки токенов.
ESTIMATE_SAFETY = 0.9
MIN_BUDGET_TOKENS = 4_000
# Минимум под историю, даже если системный промпт и схемы съели весь бюджет.
MIN_HISTORY_TOKENS = 1_000


def budget_for_window(window: int) -> int:
    """Бюджет запроса (по оценке) для модели с окном ``window`` токенов."""
    return max(int((window - RESPONSE_RESERVE_TOKENS) * ESTIMATE_SAFETY), MIN_BUDGET_TOKENS)


def estimate_text_tokens(text: str) -> int:
    """Оценка числа токенов текста (с округлением вверх)."""
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def estimate_message_tokens(message: Message) -> int:
    """Оценка размера сообщения: текст, вызовы инструментов, рассуждения и обвязка."""
    size = estimate_text_tokens(message.content)
    for call in message.function_calls:
        args = json.dumps(call.arguments, ensure_ascii=False)
        size += estimate_text_tokens(call.name + args)
    if message.reasoning:  # уходит отдельным сообщением перед ответом
        size += estimate_text_tokens(message.reasoning) + MESSAGE_OVERHEAD_TOKENS
    return size + MESSAGE_OVERHEAD_TOKENS


def estimate_tokens(messages: Iterable[Message]) -> int:
    """Оценка размера сообщений в токенах (сумма по сообщениям)."""
    return sum(estimate_message_tokens(message) for message in messages)


def estimate_specs_tokens(specs: Iterable[ToolSpec]) -> int:
    """Схемы инструментов уходят в каждом запросе и тоже занимают окно."""
    return sum(
        estimate_text_tokens(json.dumps(spec.model_dump(), ensure_ascii=False)) for spec in specs
    )


def _blocks(history: Sequence[Message]) -> list[list[Message]]:
    """Делит историю на неделимые блоки: сообщение + следующие за ним результаты."""
    blocks: list[list[Message]] = []
    for message in history:
        if message.role == "function" and blocks:
            blocks[-1].append(message)
        else:
            blocks.append([message])
    return blocks


def _pinned_blocks(blocks: Sequence[Sequence[Message]]) -> set[int]:
    """Блоки, которые не отбрасываются: самый свежий и с последним запросом пользователя."""
    pinned_user = max(
        (index for index, block in enumerate(blocks) if block[0].is_request),
        default=None,
    )
    pinned = {len(blocks) - 1}
    if pinned_user is not None:
        pinned.add(pinned_user)
    return pinned


def _dropped_blocks(sizes: Sequence[int], pinned: set[int], budget_tokens: int) -> set[int]:
    """Блоки, отбрасываемые от самых старых, пока история не уложится в ``budget_tokens``."""
    total = sum(sizes)
    dropped: set[int] = set()
    for position, size in enumerate(sizes):  # от старых к новым
        if total <= budget_tokens:
            break
        if position in pinned:
            continue
        dropped.add(position)
        total -= size
    return dropped


def fit_history(history: Sequence[Message], budget_tokens: int) -> list[Message]:
    """Сообщения истории, укладывающиеся в ``budget_tokens`` (по оценке).

    Отбрасывает блоки от самых старых: сначала предыдущие ходы, затем ранние
    шаги текущего хода. Последний запрос пользователя и самый свежий блок
    сохраняются всегда, даже если вместе они превышают бюджет.
    """
    if estimate_tokens(history) <= budget_tokens:
        return _strip_leading_results(list(history))

    blocks = _blocks(history)
    sizes = [estimate_tokens(block) for block in blocks]
    dropped = _dropped_blocks(sizes, _pinned_blocks(blocks), budget_tokens)
    kept = [
        message for index, block in enumerate(blocks) if index not in dropped for message in block
    ]
    return _strip_leading_results(kept)


def _strip_leading_results(messages: list[Message]) -> list[Message]:
    while messages and messages[0].role == "function":
        messages.pop(0)
    return messages
