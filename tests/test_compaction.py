"""Сжатие контекста: план, транскрипт, итеративное краткое содержание (без сети)."""

from __future__ import annotations

import pytest
from fakes import ScriptedProvider, text_turn

from gigacrew.agent.compaction import (
    MAX_INPUT_CHUNKS,
    CompactionError,
    chunk_entries,
    clip_middle,
    plan_compaction,
    render_message,
    summarize,
)
from gigacrew.agent.context_window import estimate_tokens
from gigacrew.agent.conversation import Conversation, Summary
from gigacrew.agent.prompts import COMPACT_PROMPT
from gigacrew.llm.base import LLMError
from gigacrew.llm.types import AssistantTurn, FunctionCall, Message, Usage

# Температура передаётся в запрос как есть, без вычислений: сравнение точное.
_TEMPERATURE = 0.1


def _turn(
    conv: Conversation,
    request: str,
    tool_outputs: list[str],
    answer: str = "готово",
) -> None:
    conv.add_user(request)
    for index, content in enumerate(tool_outputs):
        call = FunctionCall(name="read_file", arguments={"path": f"{index}.py"})
        conv.add_assistant(Message(role="assistant", function_calls=[call]))
        conv.add_function_result("read_file", content)
    if answer:
        conv.add_assistant(Message(role="assistant", content=answer))


# ------------------------------- план ------------------------------- #
def test_manual_plan_takes_everything() -> None:
    conv = Conversation()
    _turn(conv, "первый", ["a"])
    _turn(conv, "второй", ["b"])
    plan = plan_compaction(conv, keep_tokens=0)
    assert plan is not None
    assert plan.upto == len(conv)
    assert plan.messages == list(conv.messages)

    conv.set_summary(Summary("сводка", plan.upto))
    assert plan_compaction(conv, keep_tokens=0) is None  # нового ничего
    _turn(conv, "третий", [])
    plan = plan_compaction(conv, keep_tokens=0)
    assert plan is not None
    assert [message.content for message in plan.messages] == ["третий", "готово"]


def test_plan_keeps_recent_blocks_and_never_splits_call_and_result() -> None:
    conv = Conversation()
    _turn(conv, "ЗАДАЧА", ["x" * 3000 for _ in range(4)], answer="")
    last_block = estimate_tokens(conv.messages[-2:])
    plan = plan_compaction(conv, keep_tokens=last_block + 10)
    assert plan is not None
    assert plan.upto == len(conv) - 2  # остался последний блок «вызов + результат»
    assert conv.messages[plan.upto].function_calls
    assert plan.messages[0].content == "ЗАДАЧА"

    # Последний блок остаётся, даже если он больше keep_tokens.
    oversized = plan_compaction(conv, keep_tokens=1)
    assert oversized is not None
    assert oversized.upto == len(conv) - 2


def test_plan_after_mid_turn_compaction_includes_pinned_task() -> None:
    conv = Conversation()
    _turn(conv, "ЗАДАЧА", ["x" * 300 for _ in range(5)], answer="")
    conv.set_summary(Summary("сводка", 5))  # посреди хода: задача закреплена
    plan = plan_compaction(conv, keep_tokens=1)
    assert plan is not None
    assert plan.upto == len(conv) - 2
    roles = [message.role for message in plan.messages]
    assert roles == ["user", "assistant", "function", "assistant", "function"]
    assert plan.messages[0].content == "ЗАДАЧА"


def test_plan_skips_when_too_little_to_gain() -> None:
    conv = Conversation()
    _turn(conv, "ЗАДАЧА", ["коротко" for _ in range(3)], answer="")
    assert plan_compaction(conv, keep_tokens=1, min_tokens=10_000) is None
    assert plan_compaction(Conversation(), keep_tokens=0) is None
    single = Conversation()
    single.add_user("один запрос")  # единственный блок — оставить его нечем заменить
    assert plan_compaction(single, keep_tokens=1) is None


# ---------------------------- транскрипт ---------------------------- #
def test_render_message_clips_results_and_shows_calls() -> None:
    call = Message(
        role="assistant",
        content="читаю",
        function_calls=[FunctionCall(name="read_file", arguments={"path": "a.py"})],
    )
    assert render_message(call) == '## Assistant\nчитаю\n→ call read_file({"path": "a.py"})'
    rendered = render_message(Message(role="function", name="run_shell", content="A" * 10_000))
    assert rendered.startswith("## Result of run_shell\nAAA")
    assert "chars cut" in rendered
    assert len(rendered) < 2_200
    assert render_message(Message(role="user", content="привет")) == "## User\nпривет"
    note = Message(role="user", content="план одобрен", meta=True)
    assert render_message(note) == "## gigacrew (harness note)\nплан одобрен"


def test_clip_middle_keeps_head_and_tail() -> None:
    filler = "x" * 1000
    text = f"начало {filler} конец"
    clipped = clip_middle(text, 100)
    assert clipped.startswith("начало")
    assert clipped.endswith("конец")
    assert clip_middle("коротко", 100) == "коротко"


def test_chunk_entries_split_by_budget() -> None:
    entries = ["a" * 1500, "b" * 1500, "c" * 1500]  # ~500 токенов каждая
    assert chunk_entries(entries, 1_100) == [entries[:2], entries[2:]]
    chunks = chunk_entries(["z" * 30_000], 600)  # больше бюджета — обрезается
    assert len(chunks) == 1
    huge = chunks[0]
    assert len(huge) == 1
    assert len(huge[0]) < 1_900


# ------------------------------ резюме ------------------------------ #
def _messages(count: int, size: int = 1500) -> list[Message]:
    filler = "x" * size
    return [Message(role="user", content=f"запрос {index} {filler}") for index in range(count)]


def test_summarize_single_request_without_tools() -> None:
    provider = ScriptedProvider(summaries=[text_turn("  итог  ")])
    text = summarize(
        provider,
        _messages(2, 10),
        model="M",
        temperature=_TEMPERATURE,
        input_budget=20_000,
        max_chars=1_000,
        instructions="сохрани имена файлов",
    )
    assert text == "итог"
    assert len(provider.summary_requests) == 1
    request = provider.summary_requests[0]
    assert request["model"] == "M"
    assert request["temperature"] == _TEMPERATURE
    system, user = request["messages"]
    assert system.content == COMPACT_PROMPT
    assert "запрос 0" in user.content
    assert "сохрани имена файлов" in user.content
    assert provider.requests == []  # запросы суммаризации — без инструментов


def test_summarize_chains_chunks_and_keeps_only_recent() -> None:
    count = MAX_INPUT_CHUNKS + 2
    provider = ScriptedProvider(
        summaries=[text_turn(f"сводка {index}") for index in range(count)],
    )
    usages: list[Usage] = []
    text = summarize(
        provider,
        _messages(count, 4_000),  # ~1350 токенов каждое
        model="M",
        temperature=0.2,
        input_budget=2_500,  # за вычетом промпта и прежнего резюме — одно сообщение на кусок
        max_chars=600,
        previous="старая сводка",
        on_usage=usages.append,
    )
    assert len(provider.summary_requests) == MAX_INPUT_CHUNKS
    first, second = (request["messages"][1].content for request in provider.summary_requests[:2])
    assert "старая сводка" in first
    assert "omitted" in first
    assert "запрос 2" in first
    assert "запрос 0" not in first
    assert "сводка 0" in second  # резюме обновляется итеративно
    last_index = MAX_INPUT_CHUNKS - 1
    assert text == f"сводка {last_index}"
    assert len(usages) == MAX_INPUT_CHUNKS


def test_summarize_clips_long_summary() -> None:
    provider = ScriptedProvider(summaries=[text_turn("я" * 5_000)])
    text = summarize(
        provider,
        _messages(1),
        model="M",
        temperature=0.2,
        input_budget=20_000,
        max_chars=1_000,
    )
    assert len(text) < 1_100
    assert text.endswith("(краткое содержание обрезано)")


def test_summarize_tells_model_the_size_limit() -> None:
    provider = ScriptedProvider(summaries=[text_turn("итог")])
    summarize(
        provider,
        _messages(1),
        model="M",
        temperature=0.2,
        input_budget=20_000,
        max_chars=1_400,
    )
    request = provider.summary_requests[0]
    user = request["messages"][1].content
    assert "at most 1400 characters (~200 words)" in user


def test_summarize_clips_by_line_keeping_current_state() -> None:
    details = "\n".join(f"- сделано {index}: подробности подробности" for index in range(100))
    reply = f"Текущее состояние: правлю loop.py, дальше — тесты.\n{details}"
    provider = ScriptedProvider(summaries=[text_turn(reply)])
    text = summarize(
        provider,
        _messages(1),
        model="M",
        temperature=0.2,
        input_budget=20_000,
        max_chars=500,
    )
    body, mark = text.rsplit("\n", 1)
    assert text.startswith("Текущее состояние: правлю loop.py, дальше — тесты.")
    assert len(body) <= 500
    assert mark == "…(краткое содержание обрезано)"
    assert body.split("\n")[-1].endswith("подробности")  # строка не оборвана на полуслове


@pytest.mark.parametrize(
    "reply",
    [
        text_turn("   "),
        AssistantTurn(message=Message(role="assistant", content="обры"), finish_reason="length"),
        AssistantTurn(message=Message(role="assistant", content="—"), finish_reason="blacklist"),
    ],
)
def test_summarize_rejects_unusable_reply(reply: AssistantTurn) -> None:
    provider = ScriptedProvider(summaries=[reply])
    with pytest.raises(CompactionError):
        summarize(
            provider,
            _messages(1),
            model="M",
            temperature=0.2,
            input_budget=20_000,
            max_chars=900,
        )


def test_summarize_reports_paid_usage_before_failure() -> None:
    provider = ScriptedProvider(
        summaries=[
            text_turn("сводка", Usage(prompt_tokens=700, completion_tokens=30)),
            LLMError("x"),
        ],
    )
    usages: list[Usage] = []
    with pytest.raises(LLMError):
        summarize(
            provider,
            _messages(2, 4_000),
            model="M",
            temperature=0.2,
            input_budget=2_500,
            max_chars=600,
            on_usage=usages.append,
        )
    assert usages == [Usage(prompt_tokens=700, completion_tokens=30)]


def test_summarize_request_fits_budget_with_huge_instructions_and_old_summary() -> None:
    provider = ScriptedProvider(summaries=[text_turn("сводка")])
    summarize(
        provider,
        _messages(3, 4_000),
        model="M",
        temperature=0.2,
        input_budget=4_000,
        max_chars=1_000,
        previous="с" * 12_000,  # резюме от модели с большим окном
        instructions="п" * 50_000,  # вставили огромный текст в /compact
    )
    for request in provider.summary_requests:
        assert estimate_tokens(request["messages"]) <= 4_000


@pytest.mark.parametrize("meta", [False, True], ids=["plain", "with_meta_message"])
def test_summarize_keeps_latest_request_when_old_chunks_dropped(*, meta: bool) -> None:
    # Длинный последний ход: запрос, затем много больших результатов — запрос
    # оказывается старше трёх последних кусков, но на вход суммаризатора попадает.
    messages = _messages(2, 4_000)
    messages.append(Message(role="user", content="ТЕКУЩАЯ ЗАДАЧА: перенеси API на async"))
    filler = "x" * 4_000
    # ~2 результата на кусок (обрезаются до 2000 символов)
    numbers = [format(index, "02") for index in range(12)]
    messages += [
        Message(role="function", name="read_file", content=f"файл {number} {filler}")
        for number in numbers
    ]
    if meta:  # служебное сообщение посреди хода — не запрос: закрепляется задача
        messages.insert(5, Message(role="user", content="план одобрен", meta=True))
    provider = ScriptedProvider(summaries=[text_turn(f"сводка {index}") for index in range(5)])
    summarize(provider, messages, model="M", temperature=0.2, input_budget=2_500, max_chars=600)
    requests = provider.summary_requests
    assert len(requests) == MAX_INPUT_CHUNKS
    first = requests[0]["messages"][1].content
    assert "ТЕКУЩАЯ ЗАДАЧА" in first
    assert "omitted" in first
    assert "запрос 0" not in first
    assert "файл 05" not in first
    assert "файл 06" in first
    last = requests[-1]["messages"][1].content
    assert "файл 11" in last  # свежая переписка — на месте
    for request in requests:
        assert estimate_tokens(request["messages"]) <= 2_500


@pytest.mark.parametrize("finish_reason", ["length", "blacklist"])
def test_summarize_bills_a_paid_reply_it_rejects(finish_reason: str) -> None:
    # Ответ оплачен, даже если он непригоден: on_usage получает расход каждого вызова.
    usage = Usage(prompt_tokens=700, completion_tokens=30)
    reply = AssistantTurn(
        message=Message(role="assistant", content="обры"),
        finish_reason=finish_reason,
        usage=usage,
    )
    provider = ScriptedProvider(summaries=[reply])
    usages: list[Usage] = []
    with pytest.raises(CompactionError):
        summarize(
            provider,
            _messages(1),
            model="M",
            temperature=0.2,
            input_budget=20_000,
            max_chars=900,
            on_usage=usages.append,
        )
    assert usages == [usage]


def test_summarize_without_anything_to_fold_makes_no_request() -> None:
    provider = ScriptedProvider()
    with pytest.raises(CompactionError, match="нечего"):
        summarize(provider, [], model="M", temperature=0.2, input_budget=20_000, max_chars=900)
    assert provider.summary_requests == []
