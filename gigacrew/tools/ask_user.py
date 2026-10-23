"""Инструмент ``ask_user``: вопросы пользователю с вариантами ответа посреди хода."""

from __future__ import annotations

from itertools import starmap
from typing import override

from pydantic import BaseModel, Field

from gigacrew.tools.base import Tool, ToolContext, ToolError, ToolResult
from gigacrew.tools.questions import Answer, Question, QuestionOption, QuestionsUnavailableError

MAX_QUESTIONS = 4
MIN_OPTIONS, MAX_OPTIONS = 2, 4
_DESCRIBE_LIMIT = 70  # длина описания вызова в UI

DECLINED_NOTE = (
    "Пользователь отказался отвечать на вопросы. Не задавай их повторно: прими "
    "разумное решение сам и перечисли сделанные допущения в ответе."
)


# Без docstring: он ушёл бы в JSON-схему параметров, которую видит модель.
class OptionParams(BaseModel):  # noqa: D101
    label: str = Field(description="The answer option, short (1–5 words).")
    description: str = Field(
        "",
        description="What this option means: the gist, pros and cons, consequences.",
    )


# Без docstring: он ушёл бы в JSON-схему параметров, которую видит модель.
class QuestionParams(BaseModel):  # noqa: D101
    question: str = Field(description="The full text of the question, ending with '?'.")
    header: str = Field("", description="A short topic label, up to 20 characters ('Database').")
    options: list[OptionParams] = Field(
        description=(
            f"{MIN_OPTIONS}–{MAX_OPTIONS} answer options. An option for a custom answer is "
            "added automatically — do not add it."
        ),
    )
    multi_select: bool = Field(
        default=False,
        description="true — several options can be chosen at once.",
    )


# Без docstring: он ушёл бы в JSON-схему параметров, которую видит модель.
class AskUserParams(BaseModel):  # noqa: D101
    questions: list[QuestionParams] = Field(
        description=f"1–{MAX_QUESTIONS} questions, asked one after another.",
    )


class AskUserTool(Tool[AskUserParams]):
    """Инструмент ask_user: вопросы пользователю с вариантами ответа."""

    name = "ask_user"
    description = (
        "Asks the user questions with answer options and waits for the answers. Use it "
        "when the task is ambiguous and the answer changes the solution: a choice between "
        "approaches, unclear requirements, preferences. Do not ask about what you can find "
        "out yourself from the code and tools. Each option has a short label and a "
        "detailed description; the user can pick an option or write their own answer. "
        "Write the questions and options in the user's language."
    )
    # Модель параметров, а не константа: имя задаёт базовый класс Tool.
    Params = AskUserParams  # noqa: WPS115

    @override
    def describe(self, params: AskUserParams) -> str:
        """Текст единственного вопроса или число вопросов."""
        if len(params.questions) == 1:
            return params.questions[0].question[:_DESCRIBE_LIMIT]
        return f"{len(params.questions)} вопроса"

    @override
    def run(self, params: AskUserParams, ctx: ToolContext) -> ToolResult:
        """Задать вопросы пользователю и вернуть модели его ответы."""
        questions = _to_questions(params)
        if ctx.ask_user is None:
            raise QuestionsUnavailableError
        answers = ctx.ask_user(questions)
        if answers is None:
            return ToolResult(content=DECLINED_NOTE, summary="пользователь не ответил")
        lines = ["Ответы пользователя:"]
        for number, (question, answer) in enumerate(zip(questions, answers, strict=False), 1):
            lines.append(f"{number}. {question.text} → {format_answer(answer)}")
        return ToolResult(content="\n".join(lines), summary=f"получены ответы: {len(answers)}")


def format_answer(answer: Answer) -> str:
    """Ответ одной строкой: выбранные варианты и свой ответ пользователя."""
    parts = list(answer.selected)
    if answer.custom:
        parts.append(f"свой ответ: «{answer.custom}»")
    return ", ".join(parts) if parts else "(без ответа)"


def _to_questions(params: AskUserParams) -> list[Question]:
    """Проверка ограничений (не в схеме — см. CONTRIBUTING) и перевод в типы UI."""
    if not 1 <= len(params.questions) <= MAX_QUESTIONS:
        raise ToolError(
            f"Нужно от 1 до {MAX_QUESTIONS} вопросов, передано {len(params.questions)}.",
        )
    return list(starmap(_to_question, enumerate(params.questions, 1)))


def _to_question(number: int, question_params: QuestionParams) -> Question:
    """Проверка вопроса номер ``number`` и перевод в тип UI."""
    text = question_params.question.strip()
    if not text:
        raise ToolError(f"Вопрос {number}: пустой текст вопроса.")
    labels = _option_labels(number, question_params)
    options = tuple(
        QuestionOption(label, option.description.strip())
        for label, option in zip(labels, question_params.options, strict=True)
    )
    return Question(text, options, question_params.header.strip(), question_params.multi_select)


def _option_labels(number: int, question_params: QuestionParams) -> list[str]:
    """Метки вариантов вопроса номер ``number``: их число, непустота и уникальность."""
    labels = [option.label.strip() for option in question_params.options]
    if not MIN_OPTIONS <= len(labels) <= MAX_OPTIONS:
        raise ToolError(
            f"Вопрос {number}: нужно {MIN_OPTIONS}–{MAX_OPTIONS} варианта, передано {len(labels)}.",
        )
    if not all(labels):
        raise ToolError(f"Вопрос {number}: у варианта пустая метка.")
    if len({label.lower() for label in labels}) != len(labels):
        raise ToolError(f"Вопрос {number}: метки вариантов повторяются.")
    return labels
