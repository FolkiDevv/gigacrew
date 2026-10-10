"""Инструмент ask_user: проверка вопросов, формат ответов модели, отказ, нет UI."""

from __future__ import annotations

import pytest

from gigacrew.project.workspace import Workspace
from gigacrew.tools.ask_user import AskUserTool
from gigacrew.tools.base import ToolContext, ToolError
from gigacrew.tools.questions import Answer, QuestionsUnavailable


def _q(text="Какую БД?", labels=("PostgreSQL", "SQLite"), **kw):
    return {
        "question": text,
        "options": [{"label": label, "description": f"про {label}"} for label in labels],
        **kw,
    }


def _run(tmp_path, questions, answers=None, *, ui=True):
    asked = []

    def ask(qs):
        asked.append(list(qs))
        return answers

    tool = AskUserTool()
    ctx = ToolContext(Workspace(tmp_path), ask_user=ask if ui else None)
    return tool.run(tool.parse({"questions": questions}), ctx), asked


def test_answers_are_formatted_for_model(tmp_path):
    result, asked = _run(
        tmp_path,
        [_q(header="БД"), _q("Что включить?", ("Логи", "Метрики", "Трейсы"), multi_select=True)],
        [Answer(("SQLite",)), Answer(("Логи", "Трейсы"), custom="и алерты")],
    )
    assert result.ok and result.summary == "получены ответы: 2"
    assert result.content.splitlines() == [
        "Ответы пользователя:",
        "1. Какую БД? → SQLite",
        "2. Что включить? → Логи, Трейсы, свой ответ: «и алерты»",
    ]
    first, second = asked[0]
    assert first.header == "БД" and first.options[1].description == "про SQLite"
    assert second.multi_select is True and len(second.options) == 3


def test_declined_is_not_a_failure(tmp_path):
    result, _ = _run(tmp_path, [_q()], None)
    assert result.ok and "отказался" in result.content


def test_no_user_to_ask(tmp_path):
    with pytest.raises(QuestionsUnavailable, match="нет интерактивного пользователя"):
        _run(tmp_path, [_q()], ui=False)


@pytest.mark.parametrize(
    ("questions", "error"),
    [
        ([], "от 1 до 4"),
        ([_q()] * 5, "от 1 до 4"),
        ([_q(labels=("один",))], "2–4"),
        ([_q(labels=("a", "b", "c", "d", "e"))], "2–4"),
        ([_q(labels=("a", " "))], "пустая метка"),
        ([_q(labels=("Да", "да"))], "повторяются"),
        ([_q(text="  ")], "пустой текст"),
    ],
)
def test_limits_are_checked(tmp_path, questions, error):
    with pytest.raises(ToolError, match=error):
        _run(tmp_path, questions, [Answer()])


def test_describe():
    tool = AskUserTool()
    assert tool.describe(tool.parse({"questions": [_q()]})) == "Какую БД?"
    assert tool.describe(tool.parse({"questions": [_q(), _q()]})) == "2 вопроса"
