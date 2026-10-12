"""Ограничители агентного цикла.

Решают, когда принудительно остановить ход:

* исчерпан лимит шагов;
* серия неудачных вызовов инструментов (модель застряла на ошибках);
* зацикливание — модель повторяет один и тот же вызов (имя + аргументы), хотя
  между повторами ничего не менялось, и результат заведомо будет тем же. На
  ``max_repeats``-м повторе модель получает предупреждение вместе с результатом,
  следующий повтор не выполняется, ход останавливается.

«Ничего не менялось» — между двумя одинаковыми вызовами не было успешной
изменяющей операции *другого* вызова. Собственное изменение повтор не
сбрасывает: ``pytest`` без правок между запусками — зацикливание, правка →
``pytest`` — нет.

Отказ пользователя не считается ошибкой инструмента, но запоминается:
повтор отклонённого в этом ходе вызова отклоняется без нового вопроса.

Для суб-агентов включаются дополнительные ограничители (у основного агента они
выключены — значения по умолчанию ``None``):

* заполнение контекста и время на запуск (``max_context``, ``time_limit``): контекст
  суб-агента растёт с каждым результатом инструмента, у предела он подводит итог;
* похожие вызовы — тот же инструмент с той же целью (файл, шаблон, запрос) при
  разных прочих аргументах (диапазоны строк, флаги): суб-агент «топчется» на одном
  месте. На ``max_similar``-м — предупреждение, через два — остановка;
* давление бюджета (:meth:`LoopGuard.pressure`): ближе к исчерпанию любого лимита
  модель получает напоминание сворачивать работу и не начинать новых направлений.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from gigacrew.llm.types import FunctionCall

StopKind = Literal[
    "max_steps",
    "tool_failures",
    "tool_repeats",
    "similar_calls",
    "context_limit",
    "time_limit",
    "user_stop",
]

# Через сколько похожих вызовов после предупреждения ход останавливается.
SIMILAR_GRACE = 2
# С какой доли любого лимита модель получает напоминание о бюджете.
PRESSURE_SHARE = 0.7
# Поля аргументов, задающие «цель» вызова, — по убыванию приоритета.
_TARGET_FIELDS = ("pattern", "query", "path", "command", "subcommand", "focus")


@dataclass(frozen=True)
class StopReason:
    kind: StopKind
    message: str


@dataclass(frozen=True)
class ToolOutcome:
    """Исход вызова инструмента для ограничителей."""

    ok: bool
    changed: bool = False  # успешная изменяющая операция (риск ≥ WRITE)
    rejected: bool = False  # отклонено пользователем
    # Неуспех, который не говорит о застревании: команда отработала с ненулевым кодом
    # (grep без совпадений, упавшие тесты). Серию ошибок не меняет.
    soft: bool = False


@dataclass(frozen=True)
class CallCheck:
    """Решение ограничителя перед выполнением вызова."""

    repeats: int  # который раз подряд (без изменений между ними) делается этот вызов
    warning: str | None = None  # предупреждение для модели — дописывается к результату
    stop: StopReason | None = None  # вызов не выполнять, ход остановить
    rejected_before: bool = False  # пользователь уже отклонил этот вызов в этом ходе


def call_key(call: FunctionCall) -> str:
    """Канонический ключ вызова: имя + JSON аргументов с упорядоченными ключами."""
    args = json.dumps(
        call.arguments, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    )
    return f"{call.name}:{args}"


def target_key(call: FunctionCall) -> str:
    """Ключ «похожести»: имя + главная цель вызова, без диапазонов строк и флагов.

    ``read_file`` одного файла с разными диапазонами, ``search_content`` одного
    шаблона в разных каталогах — похожие вызовы. Без цели — ключ по имени.
    """
    for field in _TARGET_FIELDS:
        value = call.arguments.get(field)
        if value not in (None, "", []):
            text = value if isinstance(value, str) else json.dumps(value, default=str)
            return f"{call.name}:{field}={text.strip()}"
    return call.name


class LoopGuard:
    """Состояние ограничителей на один ход агента."""

    def __init__(
        self,
        *,
        max_steps: int,
        max_failures: int,
        max_repeats: int = 3,
        max_similar: int | None = None,
        max_context: int | None = None,
        time_limit: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._max_steps = max_steps
        self._max_failures = max_failures
        self._max_repeats = max_repeats
        self._max_similar = max_similar
        self._max_context = max_context
        self._time_limit = time_limit
        self._clock = clock
        self._started = clock()
        self._similar_seen: dict[str, int] = {}  # ключ похожести → _changes после вызова
        self._similar: dict[str, int] = {}  # ключ похожести → похожих подряд без изменений
        self._steps = 0
        self._failures = 0
        self._changes = 0  # успешных изменяющих операций за ход
        self._seen: dict[str, int] = {}  # ключ → значение _changes сразу после вызова
        self._repeats: dict[str, int] = {}  # ключ → повторов подряд без изменений
        self._rejected: set[str] = set()

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def changes(self) -> int:
        """Успешных изменяющих операций за ход."""
        return self._changes

    def _elapsed(self) -> float:
        return self._clock() - self._started

    def before_step(self, context_tokens: int = 0) -> StopReason | None:
        """Вызывается перед обращением к модели; учитывает шаг.

        ``context_tokens`` — размер контекста, который уйдёт в запросе (для ``max_context``).
        """
        if self._steps >= self._max_steps:
            return StopReason(
                "max_steps",
                f"Достигнут лимит шагов агента ({self._max_steps}). "
                "Задача может быть не завершена.",
            )
        if self._max_context is not None and context_tokens >= self._max_context:
            return StopReason(
                "context_limit",
                f"Контекст заполнен (~{context_tokens} из допустимых {self._max_context} "
                "токенов). Задача может быть не завершена.",
            )
        if self._time_limit is not None and self._elapsed() >= self._time_limit:
            return StopReason(
                "time_limit",
                f"Исчерпан лимит времени ({int(self._time_limit)} с). "
                "Задача может быть не завершена.",
            )
        self._steps += 1
        return None

    def pressure(self, context_tokens: int = 0) -> str | None:
        """Напоминание о бюджете, если израсходовано ≥70% любого лимита (иначе None).

        Только когда заданы лимиты суб-агента (у основного агента — None).
        """
        if self._max_context is None and self._time_limit is None:
            return None
        shares = [self._steps / self._max_steps]
        parts = [f"шагов {self._steps} из {self._max_steps}"]
        if self._max_context is not None:
            shares.append(context_tokens / self._max_context)
            parts.append(f"контекст ~{context_tokens} из {self._max_context} токенов")
        if self._time_limit is not None:
            elapsed = self._elapsed()
            shares.append(elapsed / self._time_limit)
            parts.append(f"времени {int(elapsed)} из {int(self._time_limit)} с")
        if max(shares) < PRESSURE_SHARE:
            return None
        return (
            "=== БЮДЖЕТ ===\n"
            f"Израсходовано: {', '.join(parts)}. Новых направлений не начинай: заверши "
            "текущую проверку и переходи к итоговому отчёту. На исчерпании любого лимита "
            "работа будет остановлена."
        )

    def before_tool(self, call: FunctionCall) -> CallCheck:
        """Вызывается перед выполнением инструмента; учитывает повтор."""
        key = call_key(call)
        unchanged = self._seen.get(key) == self._changes
        repeats = self._repeats.get(key, 0) + 1 if unchanged else 1
        self._repeats[key] = repeats
        rejected = key in self._rejected
        similar = self._count_similar(call)

        if repeats > self._max_repeats:
            return CallCheck(
                repeats,
                stop=StopReason(
                    "tool_repeats",
                    f"Прервано: агент зациклился — {repeats}-й одинаковый вызов "
                    f"{call.name} подряд без изменений между ними. "
                    "Уточните задачу или попробуйте другую модель.",
                ),
                rejected_before=rejected,
            )
        if self._max_similar is not None and similar >= self._max_similar + SIMILAR_GRACE:
            return CallCheck(
                repeats,
                stop=StopReason(
                    "similar_calls",
                    f"Прервано: агент топчется на месте — {similar}-й похожий вызов "
                    f"{call.name} подряд (та же цель) без изменений между ними.",
                ),
                rejected_before=rejected,
            )
        warning = None
        if repeats == self._max_repeats:
            warning = (
                f"ВНИМАНИЕ: ты уже {repeats}-й раз вызываешь {call.name} с теми же "
                "аргументами, а между вызовами ничего не менялось — результат тот же. "
                "Не повторяй этот вызов: используй уже полученный результат, смени "
                "подход или ответь пользователю. Следующий такой повтор остановит работу."
            )
        elif self._max_similar is not None and similar >= self._max_similar:
            warning = (
                f"ВНИМАНИЕ: ты уже {similar}-й раз подряд вызываешь {call.name} для той же "
                "цели. Ты топчешься на месте: используй уже полученные результаты, смени "
                "подход или переходи к отчёту. Ещё несколько таких вызовов остановят работу."
            )
        return CallCheck(repeats, warning=warning, rejected_before=rejected)

    def _count_similar(self, call: FunctionCall) -> int:
        """Учесть вызов как похожий на предыдущие (та же цель, без изменений между)."""
        if self._max_similar is None:
            return 0
        key = target_key(call)
        unchanged = self._similar_seen.get(key) == self._changes
        count = self._similar.get(key, 0) + 1 if unchanged else 1
        self._similar[key] = count
        return count

    def after_tool(self, call: FunctionCall, outcome: ToolOutcome) -> StopReason | None:
        """Вызывается после каждого выполненного (или отклонённого) вызова."""
        key = call_key(call)
        if outcome.rejected:
            self._rejected.add(key)  # серия ошибок не меняется: модель не застряла
        elif not outcome.soft:
            self._failures = 0 if outcome.ok else self._failures + 1
        if outcome.changed:
            self._changes += 1
        self._seen[key] = self._changes
        if self._max_similar is not None:
            similar = target_key(call)
            self._similar_seen[similar] = self._changes
            if outcome.changed:
                # Правки одного файла подряд — работа, а не топтание на месте.
                self._similar.pop(similar, None)
        if self._failures >= self._max_failures:
            return StopReason(
                "tool_failures",
                f"Прервано: {self._failures} неудачных вызовов инструментов подряд. "
                "Похоже, агент застрял — уточните задачу или попробуйте другую модель.",
            )
        return None
