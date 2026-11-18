"""Вопросы модели через ACP: форма ``elicitation/create`` и разбор ответа на неё.

Форма — в том виде, который понимает ``parseElicitationForm`` в T3. Вопрос номер N — поле
``qN``: ``title`` — короткий заголовок, ``description`` — текст вопроса; одиночный выбор —
строка с ``oneOf``, множественный — массив с ``items: {anyOf: [...]}``. Свой ответ —
строковое поле ``qN__other``: к одиночному выбору T3 его сворачивает, к множественному
показывает отдельным вопросом, поэтому оно называет свой вопрос. Обязательных полей нет.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

from gigacrew.tools.questions import Answer, Question, QuestionOption, QuestionsUnavailableError

__all__ = ["answers", "form_params"]

_TITLE: Final = "title"
_DESCRIPTION: Final = "description"
_TYPE: Final = "type"
_STRING: Final = "string"
_OTHER_SUFFIX: Final = "__other"
_MESSAGE: Final = "Агент задаёт вопросы"
_DECLINED: Final = frozenset(("decline", "cancel"))

type Json = dict[str, Any]


def form_params(questions: Sequence[Question]) -> Json:
    """Параметры ``elicitation/create`` (без ``sessionId`` и ``toolCallId``)."""
    properties: Json = {}
    for number, question in enumerate(questions, 1):
        key = _key(number)
        properties[key] = _choice_field(number, question)
        properties[f"{key}{_OTHER_SUFFIX}"] = _other_field(number, question)
    message = questions[0].text if len(questions) == 1 else _MESSAGE
    return {
        "mode": "form",
        "message": message,
        "requestedSchema": {_TYPE: "object", "properties": properties, "required": []},
    }


def answers(response: Any, questions: Sequence[Question]) -> list[Answer] | None:
    """Ответы по порядку вопросов; None — пользователь отказался отвечать.

    Непонятный ответ клиента — :class:`QuestionsUnavailableError`: ответов нет, модель
    решит сама, а не примет выдуманный ответ.
    """
    action = response.get("action") if isinstance(response, dict) else None
    if action in _DECLINED:
        return None
    content = response.get("content") if action == "accept" else None
    if not isinstance(content, dict):
        raise QuestionsUnavailableError("клиент прислал непонятный ответ на вопросы")
    return [
        _answer(content, _key(number), question) for number, question in enumerate(questions, 1)
    ]


def _key(number: int) -> str:
    return f"q{number}"


def _title(number: int, question: Question) -> str:
    return question.header or f"Вопрос {number}"


def _choice_field(number: int, question: Question) -> Json:
    description = question.text
    if question.body:
        description = f"{description}\n\n{question.body}"
    choices = [_choice(option) for option in question.options]
    field: Json = {_TITLE: _title(number, question), _DESCRIPTION: description}
    if question.multi_select:
        field[_TYPE] = "array"
        field["items"] = {"anyOf": choices}
    else:
        field[_TYPE] = _STRING
        field["oneOf"] = choices
    return field


def _choice(option: QuestionOption) -> Json:
    choice: Json = {"const": option.label, _TITLE: option.label}
    if option.description:
        choice[_DESCRIPTION] = option.description
    return choice


def _other_field(number: int, question: Question) -> Json:
    label = question.custom_label or "Свой ответ"
    hint = f"\n{question.custom_hint}" if question.custom_hint else ""
    return {
        _TYPE: _STRING,
        _TITLE: f"{_title(number, question)}: {label.lower()}",
        _DESCRIPTION: f"{label} на вопрос «{question.text}»{hint}",
    }


def _answer(content: Json, key: str, question: Question) -> Answer:
    """Ответ на вопрос: известные варианты — выбор, прочее (и ``qN__other``) — свой ответ."""
    chosen = content.get(key)
    picked = chosen if isinstance(chosen, list) else [chosen]
    labels = {option.label for option in question.options}
    texts = [entry.strip() for entry in picked if isinstance(entry, str) and entry.strip()]
    selected = tuple(dict.fromkeys(text for text in texts if text in labels))
    custom = [text for text in texts if text not in labels]
    other = content.get(f"{key}{_OTHER_SUFFIX}")
    if isinstance(other, str) and other.strip():
        custom.append(other.strip())
    return Answer(selected, "; ".join(custom))
