"""Инструмент ask_user: проверка вопросов, формат ответов модели, отказ, нет UI."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from gigacrew.project.workspace import Workspace
from gigacrew.tools.ask_user import AskUserTool
from gigacrew.tools.base import ToolContext, ToolError, ToolResult
from gigacrew.tools.questions import Answer, Question, QuestionsUnavailableError

# Вопросы, показанные пользователю: по списку на каждый вызов колбэка.
type _Asked = list[list[Question]]


def _question(
    text: str = "Какую БД?",
    labels: tuple[str, ...] = ("PostgreSQL", "SQLite"),
    **extra: Any,
) -> dict[str, Any]:
    return {
        "question": text,
        "options": [{"label": label, "description": f"про {label}"} for label in labels],
        **extra,
    }


def _run(
    tmp_path: Path,
    questions: list[dict[str, Any]],
    answers: list[Answer] | None = None,
    *,
    ui: bool = True,
) -> tuple[ToolResult, _Asked]:
    asked: _Asked = []

    def ask(shown: Sequence[Question]) -> list[Answer] | None:
        asked.append(list(shown))
        return answers

    tool = AskUserTool()
    ctx = ToolContext(Workspace(tmp_path), ask_user=ask if ui else None)
    return tool.run(tool.parse({"questions": questions}), ctx), asked


def test_answers_are_formatted_for_model(tmp_path: Path) -> None:
    tool_result, asked = _run(
        tmp_path,
        [
            _question(header="БД"),
            _question("Что включить?", ("Логи", "Метрики", "Трейсы"), multi_select=True),
        ],
        [Answer(("SQLite",)), Answer(("Логи", "Трейсы"), custom="и алерты")],
    )
    assert tool_result.ok
    assert tool_result.summary == "получены ответы: 2"
    assert tool_result.content.splitlines() == [
        "Ответы пользователя:",
        "1. Какую БД? → SQLite",
        "2. Что включить? → Логи, Трейсы, свой ответ: «и алерты»",
    ]
    first, second = asked[0]
    assert first.header == "БД"
    assert first.options[1].description == "про SQLite"
    assert second.multi_select is True
    assert len(second.options) == 3


def test_declined_is_not_a_failure(tmp_path: Path) -> None:
    tool_result, _ = _run(tmp_path, [_question()], None)
    assert tool_result.ok
    assert "отказался" in tool_result.content


def test_no_user_to_ask(tmp_path: Path) -> None:
    with pytest.raises(QuestionsUnavailableError, match="нет интерактивного пользователя"):
        _run(tmp_path, [_question()], ui=False)


@pytest.mark.parametrize(
    ("questions", "error"),
    [
        ([], "от 1 до 4"),
        ([_question() for _ in range(5)], "от 1 до 4"),
        ([_question(labels=("один",))], "2–4"),
        ([_question(labels=("a", "b", "c", "d", "e"))], "2–4"),
        ([_question(labels=("a", " "))], "пустая метка"),
        ([_question(labels=("Да", "да"))], "повторяются"),
        ([_question(text="  ")], "пустой текст"),
    ],
)
def test_limits_are_checked(tmp_path: Path, questions: list[dict[str, Any]], error: str) -> None:
    with pytest.raises(ToolError, match=error):
        _run(tmp_path, questions, [Answer()])


def test_describe() -> None:
    tool = AskUserTool()
    assert tool.describe(tool.parse({"questions": [_question()]})) == "Какую БД?"
    assert tool.describe(tool.parse({"questions": [_question(), _question()]})) == "2 вопроса"
