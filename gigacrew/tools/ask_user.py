"""Инструмент ``ask_user``: вопросы пользователю с вариантами ответа посреди хода."""

from __future__ import annotations

from pydantic import BaseModel, Field

from gigacrew.tools.base import Tool, ToolContext, ToolError, ToolResult
from gigacrew.tools.questions import Answer, Question, QuestionOption, QuestionsUnavailable

MAX_QUESTIONS = 4
MIN_OPTIONS, MAX_OPTIONS = 2, 4

DECLINED_NOTE = (
    "Пользователь отказался отвечать на вопросы. Не задавай их повторно: прими "
    "разумное решение сам и перечисли сделанные допущения в ответе."
)


class OptionParams(BaseModel):
    label: str = Field(description="Вариант ответа, коротко (1–5 слов).")
    description: str = Field(
        "", description="Что означает этот вариант: суть, плюсы и минусы, последствия."
    )


class QuestionParams(BaseModel):
    question: str = Field(description="Полный текст вопроса, заканчивается знаком «?».")
    header: str = Field("", description="Короткая метка темы, до 20 символов («База данных»).")
    options: list[OptionParams] = Field(
        description=(
            f"{MIN_OPTIONS}–{MAX_OPTIONS} варианта ответа. Вариант «свой ответ» "
            "добавляется автоматически — не добавляй его."
        )
    )
    multi_select: bool = Field(False, description="true — можно выбрать несколько вариантов сразу.")


class AskUserParams(BaseModel):
    questions: list[QuestionParams] = Field(
        description=f"1–{MAX_QUESTIONS} вопроса, задаются по очереди."
    )


class AskUserTool(Tool):
    name = "ask_user"
    description = (
        "Задаёт пользователю вопросы с вариантами ответа и ждёт ответа. Используй, когда "
        "задача неоднозначна и от ответа зависит решение: выбор между подходами, "
        "неясные требования, предпочтения. Не спрашивай о том, что можно выяснить "
        "самому из кода и инструментов. У каждого варианта — короткая метка и "
        "подробное описание; пользователь может выбрать вариант или написать свой ответ."
    )
    Params = AskUserParams

    def describe(self, params: AskUserParams) -> str:
        if len(params.questions) == 1:
            return params.questions[0].question[:70]
        return f"{len(params.questions)} вопроса"

    def run(self, params: AskUserParams, ctx: ToolContext) -> ToolResult:
        questions = _to_questions(params)
        if ctx.ask_user is None:
            raise QuestionsUnavailable()
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
            f"Нужно от 1 до {MAX_QUESTIONS} вопросов, передано {len(params.questions)}."
        )
    result = []
    for i, q in enumerate(params.questions, 1):
        text = q.question.strip()
        if not text:
            raise ToolError(f"Вопрос {i}: пустой текст вопроса.")
        labels = [o.label.strip() for o in q.options]
        if not MIN_OPTIONS <= len(labels) <= MAX_OPTIONS:
            raise ToolError(
                f"Вопрос {i}: нужно {MIN_OPTIONS}–{MAX_OPTIONS} варианта, передано {len(labels)}."
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
