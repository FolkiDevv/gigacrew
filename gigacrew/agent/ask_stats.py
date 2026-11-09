"""Сводка журнала решений: как часто и почему gigacrew спрашивал пользователя.

Журнал (:mod:`gigacrew.agent.decisions`) хранит каждое решение о команде и каждый вопрос
пользователю. Сводка отвечает, какой шлюз зовёт человека чаще всего: по ней видно, что
стоит научить классификатор проверять и где вопросы вызваны сбоями, а не существом дела.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

__all__ = ("AskSummary", "summarize")

_UNKNOWN_CAUSE = "unknown"


@dataclass(frozen=True)
class AskSummary:
    """Кто решал и почему спрашивали пользователя.

    ``answers`` — ответы пользователя (``yes``, ``always``, ``no``, ``no_answer``),
    ``causes`` — почему спросили (поле ``asked_because``; ``unknown`` — запись прежней
    версии), ``programs`` — о каких командах (ключ истории), ``automatic`` — кто решил без
    пользователя (источник решения), ``denied`` — отказы рецензента, вернувшиеся модели.
    """

    answers: Counter[str] = field(default_factory=Counter)
    causes: Counter[str] = field(default_factory=Counter)
    programs: Counter[str] = field(default_factory=Counter)
    automatic: Counter[str] = field(default_factory=Counter)
    denied: int = 0

    @property
    def asked(self) -> int:
        """Сколько раз спросили пользователя."""
        return self.answers.total()

    @property
    def decided(self) -> int:
        """Сколько всего решений в сводке."""
        return self.asked + self.automatic.total() + self.denied


def summarize(journal: Iterable[Mapping[str, Any]]) -> AskSummary:
    """Сводка записей журнала решений (:func:`gigacrew.agent.decisions.read_journal`)."""
    totals = AskSummary()
    denied = 0
    for entry in journal:
        if entry.get("event") != "decision":
            continue
        decision = str(entry.get("decision"))
        source = str(entry.get("source"))
        if source == "user":
            totals.answers[decision] += 1
            totals.causes[str(entry.get("asked_because") or _UNKNOWN_CAUSE)] += 1
            if entry.get("key"):
                totals.programs[str(entry["key"])] += 1
        elif decision == "denied":
            denied += 1
        elif decision == "auto":
            # «rule:<id>» — одно сохранённое правило; в сводке все правила вместе.
            totals.automatic[source.split(":", maxsplit=1)[0]] += 1
    return replace(totals, denied=denied)
