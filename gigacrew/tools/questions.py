"""Вопросы пользователю: типы, общие для инструмента ``ask_user``, ядра и UI.

Модуль без зависимостей: инструмент описывает вопросы, агент передаёт их в UI через
колбэк :data:`AskUser` из :class:`~gigacrew.tools.base.ToolContext`, UI возвращает
:class:`Answer` на каждый вопрос.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from gigacrew.errors import ToolError


@dataclass(frozen=True)
class QuestionOption:
    label: str  # короткий вариант
    description: str = ""  # что означает выбор, последствия


@dataclass(frozen=True)
class Question:
    text: str
    options: tuple[QuestionOption, ...]
    header: str = ""  # короткая метка темы («База данных»)
    multi_select: bool = False
    body: str = ""  # Markdown над вариантами ответа (например, план на одобрение)


@dataclass(frozen=True)
class Answer:
    """Ответ на один вопрос: выбранные варианты и/или свой текст."""

    selected: tuple[str, ...] = ()
    custom: str = ""


# Задать вопросы пользователю. Ответ на каждый вопрос по порядку; None — пользователь
# отказался отвечать. Нет интерактивного пользователя — QuestionsUnavailable.
AskUser = Callable[[Sequence[Question]], "list[Answer] | None"]


class QuestionsUnavailable(ToolError):
    """Спросить некого: одноразовый режим без терминала, ввод не с клавиатуры."""

    def __init__(self, reason: str = "нет интерактивного пользователя"):
        super().__init__(
            f"Нельзя задать вопрос: {reason}. Прими разумное решение сам и перечисли "
            "сделанные допущения в ответе."
        )
