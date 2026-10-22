"""Инструмент ``ask_user``: вопросы пользователю с вариантами ответа посреди хода."""

from __future__ import annotations

from pydantic import BaseModel, Field

from gigacrew.tools.base import Tool, ToolContext, ToolError, ToolResult
from gigacrew.tools.questions import Answer, Question, QuestionOption, QuestionsUnavailableError

MAX_QUESTIONS = 4
MIN_OPTIONS, MAX_OPTIONS = 2, 4

DECLINED_NOTE = (
    "Пользователь отказался отвечать на вопросы. Не задавай их повторно: прими "
    "разумное решение сам и перечисли сделанные допущения в ответе."
)


class OptionParams(BaseModel):
    label: str = Field(description="The answer option, short (1–5 words).")
    description: str = Field(
        "",
        description="What this option means: the gist, pros and cons, consequences.",
    )


class QuestionParams(BaseModel):
    question: str = Field(description="The full text of the question, ending with '?'.")
    header: str = Field("", description="A short topic label, up to 20 characters ('Database').")
    options: list[OptionParams] = Field(
        description=(
            f"{MIN_OPTIONS}–{MAX_OPTIONS} answer options. An option for a custom answer is "
            "added automatically — do not add it."
        ),
    )
    multi_select: bool = Field(False, description="true — several options can be chosen at once.")


class AskUserParams(BaseModel):
    questions: list[QuestionParams] = Field(
        description=f"1–{MAX_QUESTIONS} questions, asked one after another.",
    )


class AskUserTool(Tool[AskUserParams]):
    name = "ask_user"
    description = (
        "Asks the user questions with answer options and waits for the answers. Use it "
        "when the task is ambiguous and the answer changes the solution: a choice between "
        "approaches, unclear requirements, preferences. Do not ask about what you can find "
        "out yourself from the code and tools. Each option has a short label and a "
        "detailed description; the user can pick an option or write their own answer. "
        "Write the questions and options in the user's language."
    )
    Params = AskUserParams

    def describe(self, params: AskUserParams) -> str:
        if len(params.questions) == 1:
            return params.questions[0].question[:70]
        return f"{len(params.questions)} вопроса"

    def run(self, params: AskUserParams, ctx: ToolContext) -> ToolResult:
        questions = _to_questions(params)
        if ctx.ask_user is None:
            raise QuestionsUnavailableError
        answers = ctx.ask_user(questions)
        if answers is None:
            return ToolResult(content=DECLINED_NOTE, summary="пользователь не ответил")
        lines = ["Ответы пользователя:"]
        for i, (question, answer) in enumerate(zip(questions, answers, strict=False), 1):
            lines.append(f"{i}. {question.text} → {format_answer(answer)}")
        return ToolResult(content="\n".join(lines), summary=f"получены ответы: {len(answers)}")


def format_answer(answer: Answer) -> str:
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
    result = []
    for i, q in enumerate(params.questions, 1):
        text = q.question.strip()
        if not text:
            raise ToolError(f"Вопрос {i}: пустой текст вопроса.")
        labels = [o.label.strip() for o in q.options]
        if not MIN_OPTIONS <= len(labels) <= MAX_OPTIONS:
            raise ToolError(
                f"Вопрос {i}: нужно {MIN_OPTIONS}–{MAX_OPTIONS} варианта, передано {len(labels)}.",
            )
        if not all(labels):
            raise ToolError(f"Вопрос {i}: у варианта пустая метка.")
        if len({label.lower() for label in labels}) != len(labels):
            raise ToolError(f"Вопрос {i}: метки вариантов повторяются.")
        options = tuple(
            QuestionOption(label, o.description.strip())
            for label, o in zip(labels, q.options, strict=True)
        )
        result.append(Question(text, options, q.header.strip(), q.multi_select))
    return result
