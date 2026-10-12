"""Сжатие контекста: начало диалога заменяется кратким содержанием.

Модель пересказывает старую часть истории (вызов без инструментов), резюме
становится :class:`~gigacrew.agent.conversation.Summary` диалога и уходит в
системном сообщении вместо самих сообщений. Журнал при этом не переписывается.

* :func:`plan_compaction` — что сворачивать: граница проходит по блокам «сообщение +
  результаты его вызова», свежие блоки в пределах ``keep_tokens`` остаются как есть;
* :func:`render_transcript` / :func:`chunk_entries` — переписка плоским текстом,
  нарезанная так, чтобы каждый запрос суммаризации уместился в окно;
* :func:`summarize` — итеративное резюме: прежнее краткое содержание + очередной
  кусок → новое краткое содержание.

Порог автозапуска и объёмы выбирает агент (:mod:`gigacrew.agent.loop`).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from gigacrew.agent.context_window import CHARS_PER_TOKEN, estimate_text_tokens, estimate_tokens
from gigacrew.agent.conversation import Conversation
from gigacrew.agent.prompts import COMPACT_PROMPT
from gigacrew.llm.base import LLMError, LLMProvider
from gigacrew.llm.types import Message, Usage

# Сколько текста каждого сообщения попадает к суммаризатору (символов).
USER_CLIP = 6_000
ASSISTANT_CLIP = 4_000
RESULT_CLIP = 2_000
ARGS_CLIP = 300
# Суммаризатор читает не больше стольких кусков (самых свежих): более старое модель
# всё равно уже не видела — его отбрасывало окно истории.
MAX_INPUT_CHUNKS = 3
MIN_CHUNK_TOKENS = 500
REQUEST_OVERHEAD_TOKENS = 200  # обвязка запроса суммаризации
MAX_INSTRUCTIONS_CHARS = 2_000  # пожелания пользователя (/compact …)


class CompactionError(LLMError):
    """Сжать не удалось: модель не дала пригодного краткого содержания."""


@dataclass(frozen=True)
class CompactionPlan:
    """Что сворачивать: сообщения вида до новой границы ``upto`` (индекс в журнале)."""

    upto: int
    messages: list[Message]


def plan_compaction(
    conversation: Conversation, *, keep_tokens: int, min_tokens: int = 1
) -> CompactionPlan | None:
    """План сжатия или None, если сворачивать нечего.

    Блоки после прежней границы перебираются с конца: остаются свежие блоки в
    пределах ``keep_tokens`` (при ``keep_tokens > 0`` последний — всегда). ``0`` —
    свернуть всё (``/compact``). Если сворачивать меньше ``min_tokens`` (по оценке),
    сжатие не стоит запроса к модели.
    """
    messages = conversation.messages
    start = conversation.summary.upto if conversation.summary is not None else 0
    starts = [
        i for i in range(start, len(messages)) if i == start or messages[i].role != "function"
    ]

    upto = len(messages)
    if keep_tokens > 0:
        kept = 0
        for k in range(len(starts) - 1, -1, -1):
            end = starts[k + 1] if k + 1 < len(starts) else len(messages)
            size = estimate_tokens(messages[starts[k] : end])
            if upto < len(messages) and kept + size > keep_tokens:
                break
            kept += size
            upto = starts[k]
    if upto <= start:
        return None

    view = conversation.context_messages()
    # Вид = [закреплённый запрос] + messages[start:]; сворачивается всё до upto.
    head = view[: len(view) - (len(messages) - upto)]
    if not head or estimate_tokens(head) < min_tokens:
        return None
    return CompactionPlan(upto=upto, messages=head)


def clip_middle(text: str, limit: int) -> str:
    """Обрезка с вырезом середины: в начале — суть, в конце — итог (вывод тестов)."""
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    cut = len(text) - head - tail
    return f"{text[:head]}\n…[{cut} chars cut]…\n{text[len(text) - tail :]}"


def render_message(message: Message) -> str:
    """Сообщение диалога плоским текстом (вызовы инструментов — тоже текстом)."""
    if message.role == "user":
        return f"## User\n{clip_middle(message.content, USER_CLIP)}"
    if message.role == "function":
        return f"## Result of {message.name or ''}\n{clip_middle(message.content, RESULT_CLIP)}"
    parts = []
    if message.content.strip():
        parts.append(clip_middle(message.content, ASSISTANT_CLIP))
    for call in message.function_calls:
        args = json.dumps(call.arguments, ensure_ascii=False, default=str)
        parts.append(f"→ call {call.name}({clip_middle(args, ARGS_CLIP)})")
    title = "Assistant" if message.role == "assistant" else message.role
    return f"## {title}\n" + "\n".join(parts)


def render_transcript(messages: Sequence[Message]) -> list[str]:
    return [render_message(m) for m in messages]


def chunk_entries(entries: Sequence[str], budget_tokens: int) -> list[list[str]]:
    """Делит записи на куски не больше ``budget_tokens`` (по оценке).

    В каждом куске минимум одна запись; запись крупнее бюджета обрезается.
    """
    budget_tokens = max(budget_tokens, MIN_CHUNK_TOKENS)
    chunks: list[list[str]] = []
    current: list[str] = []
    size = 0
    for entry in entries:
        tokens = estimate_text_tokens(entry)
        if tokens > budget_tokens:
            entry = clip_middle(entry, budget_tokens * CHARS_PER_TOKEN - 100)
            tokens = estimate_text_tokens(entry)
        if current and size + tokens > budget_tokens:
            chunks.append(current)
            current, size = [], 0
        current.append(entry)
        size += tokens
    if current:
        chunks.append(current)
    return chunks


def _recent_chunks(messages: Sequence[Message], budget_tokens: int) -> list[list[str]]:
    """Последние :data:`MAX_INPUT_CHUNKS` кусков переписки для суммаризатора.

    Последний запрос пользователя (текущая задача) попадает на вход всегда, даже
    если его кусок старше последних: окно истории его тоже никогда не отбрасывало.
    """
    entries = render_transcript(messages)
    chunks = chunk_entries(entries, budget_tokens)
    kept = chunks[-MAX_INPUT_CHUNKS:]
    last_user = max((i for i, m in enumerate(messages) if m.role == "user"), default=None)
    if len(chunks) <= MAX_INPUT_CHUNKS or last_user is None:
        return kept
    first_kept = len(entries) - sum(len(c) for c in kept)
    if last_user >= first_kept:
        return kept
    # Запрос — в отброшенном куске: он идёт первым, переписка после него — следом,
    # в пределах тех же кусков.
    budget_tokens = max(budget_tokens, MIN_CHUNK_TOKENS)
    task = clip_middle(entries[last_user], budget_tokens * CHARS_PER_TOKEN // 2)
    after = entries[last_user + 1 :]
    kept = chunk_entries(after, budget_tokens - estimate_text_tokens(task))[-MAX_INPUT_CHUNKS:]
    kept[0] = [task, *kept[0]]
    return kept


def _request_text(
    previous: str | None, chunk: str, instructions: str, omitted: bool, max_chars: int
) -> str:
    parts = []
    if previous:
        parts.append(
            "Summary of the earlier part of the conversation (update it without losing "
            f"anything important):\n{previous}"
        )
    if omitted:
        parts.append("(The earliest part of the conversation did not fit and is omitted.)")
    parts.append(f"Conversation fragment to compress:\n{chunk}")
    if instructions.strip():
        parts.append(f"The user's wishes for the summary: {instructions.strip()}")
    parts.append(
        f"Write the summary following the rules. Size — at most {max_chars} characters "
        f"(~{max(max_chars // CHARS_PER_WORD, 1)} words)."
    )
    return "\n\n".join(parts)


CHARS_PER_WORD = 7  # русский текст с пробелами — для подсказки модели об объёме
CLIPPED_MARK = "\n…(краткое содержание обрезано)"


def _clip_summary(text: str, max_chars: int) -> str:
    """Резюме не длиннее ``max_chars``: режется с конца, по границе строки.

    Модель кладёт важное первым (текущее состояние, следующий шаг), поэтому
    теряется хвост, а не начало, и строка не обрывается на полуслове.
    """
    if len(text) <= max_chars:
        return text
    head = text[:max_chars]
    cut = head.rfind("\n")
    if cut >= max_chars // 2:
        head = head[:cut]
    return head.rstrip() + CLIPPED_MARK


def summarize(
    provider: LLMProvider,
    messages: Sequence[Message],
    *,
    model: str,
    temperature: float,
    input_budget: int,
    max_chars: int,
    previous: str | None = None,
    instructions: str = "",
    on_usage: Callable[[Usage], None] | None = None,
) -> str:
    """Краткое содержание ``messages`` (с учётом прежнего ``previous``).

    ``input_budget`` — бюджет одного запроса (оценка в токенах); ``max_chars`` —
    предел длины резюме. ``on_usage`` получает расход каждого обращения — и при
    сбое на следующем куске (оплаченное не теряется). Ошибки — :class:`LLMError`.

    Пожелания и прежнее резюме обрезаются, чтобы запрос с ними и куском переписки
    укладывался в ``input_budget``.
    """
    instructions = clip_middle(instructions.strip(), MAX_INSTRUCTIONS_CHARS)
    if previous is not None:
        previous = _clip_summary(previous, max_chars)  # могло остаться от модели с большим окном
    reserve = (
        estimate_text_tokens(COMPACT_PROMPT)
        + max_chars // CHARS_PER_TOKEN  # прежнее резюме
        + estimate_text_tokens(instructions)
        + REQUEST_OVERHEAD_TOKENS
    )
    chunks = _recent_chunks(messages, input_budget - reserve)
    omitted = sum(len(c) for c in chunks) < len(messages)

    summary = previous
    for i, chunk in enumerate(chunks):
        request = [
            Message(role="system", content=COMPACT_PROMPT),
            Message(
                role="user",
                content=_request_text(
                    summary, "\n\n".join(chunk), instructions, omitted and i == 0, max_chars
                ),
            ),
        ]
        turn = provider.complete(request, tools=None, model=model, temperature=temperature)
        if on_usage is not None:
            on_usage(turn.usage)
        if turn.finish_reason != "stop":
            raise CompactionError(
                f"модель не закончила краткое содержание (finish_reason={turn.finish_reason})"
            )
        text = turn.message.content.strip()
        if not text:
            raise CompactionError("модель вернула пустое краткое содержание")
        summary = _clip_summary(text, max_chars)
    if summary is None:
        raise CompactionError("нечего сжимать")
    return summary
