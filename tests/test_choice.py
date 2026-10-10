"""Меню ответа на вопрос агента: стрелки, цифры, мультивыбор, свой ответ, отказ."""

from __future__ import annotations

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from gigacrew.tools.questions import Answer, Question, QuestionOption
from gigacrew.ui.choice import ChoiceState, choose, render

DOWN, UP, ESC = "\x1b[B", "\x1b[A", "\x1b"

DB = Question(
    "Какую БД использовать?",
    (
        QuestionOption("PostgreSQL", "Надёжная реляционная БД, нужен отдельный сервер"),
        QuestionOption("SQLite", "Файл в проекте, без сервера"),
    ),
    header="База данных",
)
PARTS = Question(
    "Что добавить?",
    (QuestionOption("Логи"), QuestionOption("Метрики"), QuestionOption("Трейсы")),
    multi_select=True,
)


def _choose(question: Question, *chunks: str) -> Answer | None:
    with create_pipe_input() as pipe:
        for chunk in chunks:
            pipe.send_text(chunk)
        return choose(question, 1, 2, input=pipe, output=DummyOutput())


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        (("\r",), Answer(("PostgreSQL",))),
        ((DOWN, "\r"), Answer(("SQLite",))),
        ((DOWN, DOWN, DOWN, "\r"), Answer(("PostgreSQL",))),  # по кругу
        ((UP, UP, "\r"), Answer(("SQLite",))),
        (("2",), Answer(("SQLite",))),  # цифра — сразу
    ],
)
def test_single_choice(keys, expected):
    assert _choose(DB, *keys) == expected


def test_custom_answer():
    assert _choose(DB, "3", "  Mongo 2\x7fDB \r") == Answer(custom="Mongo DB")
    assert _choose(DB, UP, "x", "\x15", "Redis\r") == Answer(custom="Redis")  # Ctrl+U


def test_empty_custom_answer_is_not_accepted():
    assert _choose(DB, "3", "\r", DOWN, "\r") == Answer(("PostgreSQL",))


def test_escape_declines():
    assert _choose(DB, ESC) is None
    assert _choose(DB, "\x03") is None  # Ctrl+C


def test_multi_select():
    assert _choose(PARTS, " ", DOWN, DOWN, " ", "\r") == Answer(("Логи", "Трейсы"))
    assert _choose(PARTS, "2", "3", "2", "\r") == Answer(("Трейсы",))  # цифры переключают
    assert _choose(PARTS, DOWN, "\r") == Answer(("Метрики",))  # ничего не отмечено — текущий
    assert _choose(PARTS, "1", "4", "свой 1\r") == Answer(("Логи",), custom="свой 1")
    assert _choose(PARTS, "4", "\r", UP, "\r") == Answer(("Трейсы",))


def _text(state: ChoiceState, width: int = 80) -> str:
    return "".join(text for _, text in render(state, width))


def test_render_shows_counter_options_descriptions_and_custom():
    text = _text(ChoiceState(DB, 1, 2))
    assert "Вопрос 1 из 2 · База данных" in text
    assert "❯ 1. PostgreSQL" in text and "  2. SQLite" in text
    assert "Файл в проекте, без сервера" in text
    assert "3. Свой ответ…" in text and "Начните печатать" in text
    assert "Esc — отказаться" in text


def test_render_custom_text():
    text = _text(ChoiceState(DB, 1, 1, cursor=2, custom="MongoDB"))
    assert "❯ 3. Свой ответ: MongoDB" in text and "Начните печатать" not in text


def test_render_multi_select_and_wrapping():
    state = ChoiceState(PARTS, 2, 2, cursor=1, checked={0})
    text = _text(state)
    assert "[x] Логи" in text and "❯ 2. [ ] Метрики" in text
    assert "можно выбрать несколько" in text and "Пробел" in text
    long = Question("Вопрос?", (QuestionOption("A", "слово " * 40), QuestionOption("B")))
    lines = _text(ChoiceState(long, 1, 1), width=40).splitlines()
    assert all(len(line) <= 40 for line in lines)
    assert sum("слово" in line for line in lines) > 1  # описание перенесено с отступом
