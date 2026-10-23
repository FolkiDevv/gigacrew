"""Ограничители агентного цикла.

Решают, когда принудительно остановить ход:

* исчерпан лимит шагов;
* серия неудачных вызовов инструментов (модель застряла на ошибках): за один
  неудачный вызов до предела модель получает предупреждение (:meth:`LoopGuard.failure_warning`),
  и ход останавливается, только когда предупреждение уже ушло ей в запросе;
* зацикливание — модель повторяет один и тот же вызов (имя + аргументы), хотя
  между повторами ничего не менялось, и результат заведомо будет тем же. На
  ``max_repeats``-м повторе модель получает предупреждение вместе с результатом,
  следующий повтор не выполняется, ход останавливается.

«Ничего не менялось» — между двумя одинаковыми вызовами не было успешной
изменяющей операции *другого* вызова. Собственное изменение повтор не
сбрасывает: ``pytest`` без правок между запусками — зацикливание, правка →
``pytest`` — нет. Одинаковые вызовы — с точностью до пояснения для пользователя
(``description``): модель часто перефразирует его, повторяя ту же команду.

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
import posixpath
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
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
    "output_limit",
]

# Через сколько похожих вызовов после предупреждения ход останавливается.
SIMILAR_GRACE = 2
# Причина остановки — для модели (запрос итога, отчёт суб-агента); пользователю
# показывается ``StopReason.message``.
MODEL_STOP_REASONS: Mapping[str, str] = MappingProxyType({
    "max_steps": "the step limit is reached",
    "tool_failures": "too many failed tool calls in a row",
    "tool_repeats": "the same call was repeated without any changes (a loop)",
    "similar_calls": "similar calls for the same target kept repeating (going in circles)",
    "context_limit": "the context is full",
    "time_limit": "the time limit is reached",
    "user_stop": "the user stopped the work early",
    "output_limit": "your replies were cut off before they finished, several times in a row",
})

# С какой доли любого лимита модель получает напоминание о бюджете.
PRESSURE_SHARE = 0.7
# Поля аргументов, задающие «цель» вызова, — по убыванию приоритета.
_TARGET_FIELDS = ("pattern", "query", "path", "command", "subcommand", "focus")
# Аргументы, от которых результат не зависит (пояснение для пользователя у run_shell и task).
_COSMETIC_FIELDS = frozenset(("description",))

FAILURES_WARNING = (
    "WARNING: the last {count} tool calls failed or were not executed. Do not try "
    "variations of the same: read the errors above and change the approach (other tools "
    "or arguments), or answer with what you already have. One more failed call will stop "
    "the work."
)


@dataclass(frozen=True)
class StopReason:
    """Причина остановки хода: ``kind`` — вид ограничителя, ``message`` — для пользователя."""

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
    notice: str | None = None  # то же предупреждение — строкой для пользователя


@dataclass(frozen=True)
class GuardWarning:
    """Предупреждение ограничителя: ``model`` — заметка модели, ``notice`` — пользователю."""

    model: str
    notice: str


def call_key(call: FunctionCall) -> str:
    """Канонический ключ вызова: имя + JSON аргументов с упорядоченными ключами.

    Без пояснения для пользователя (``description``): от него результат не зависит. Не
    строка (``null``) — не пояснение, а ошибка аргументов: такой вызов — другой.
    """
    arguments = {
        name: argument
        for name, argument in call.arguments.items()
        if not (name in _COSMETIC_FIELDS and isinstance(argument, str))
    }
    args = json.dumps(
        arguments,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return f"{call.name}:{args}"


def target_key(call: FunctionCall) -> str:
    """Ключ «похожести»: имя + главная цель вызова, без диапазонов строк и флагов.

    ``read_file`` одного файла с разными диапазонами, ``search_content`` одного
    шаблона в разных каталогах — похожие вызовы. Без цели — ключ по имени.
    """
    for field in _TARGET_FIELDS:
        target = call.arguments.get(field)
        if target not in (None, "", []):
            text = target if isinstance(target, str) else json.dumps(target, default=str)
            # ./a.py и a.py, src/ и src — одна цель; пробелы внутри имени — его часть
            if field == "path":
                text = posixpath.normpath(text.strip().replace("\\", "/"))
            else:
                text = " ".join(text.split())
            return f"{call.name}:{field}={text}"
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
        """Ограничители хода; ``None`` у лимитов суб-агента — ограничитель выключен."""
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
        self._failures_warned = False  # предупреждение о серии неудач ушло модели в запросе
        self._failures_noticed = False  # ...и показано пользователю

    @property
    def steps(self) -> int:
        """Шагов (обращений к модели) за ход."""
        return self._steps

    @property
    def changes(self) -> int:
        """Успешных изменяющих операций за ход."""
        return self._changes

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
        parts = [f"steps {self._steps} of {self._max_steps}"]
        if self._max_context is not None:
            shares.append(context_tokens / self._max_context)
            parts.append(f"context ~{context_tokens} of {self._max_context} tokens")
        if self._time_limit is not None:
            elapsed = self._elapsed()
            shares.append(elapsed / self._time_limit)
            parts.append(f"time {int(elapsed)} of {int(self._time_limit)} s")
        if max(shares) < PRESSURE_SHARE:
            return None
        used = ", ".join(parts)
        return (
            "# Budget\n"
            f"Used: {used}. Do not start new directions: finish the current "
            "check and move on to the final report. When any limit is reached, the work "
            "will be stopped and you will be asked for the report."
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
        stop_similar = None if self._max_similar is None else self._max_similar + SIMILAR_GRACE
        if stop_similar is not None and similar >= stop_similar:
            return CallCheck(
                repeats,
                stop=StopReason(
                    "similar_calls",
                    f"Прервано: агент топчется на месте — {similar}-й похожий вызов "
                    f"{call.name} подряд (та же цель) без изменений между ними.",
                ),
                rejected_before=rejected,
            )
        warning: str | None = None
        notice: str | None = None
        if repeats == self._max_repeats:
            warning = (
                f"WARNING: this is call #{repeats} of {call.name} with the same arguments, "
                "and nothing changed in between — the result is the same. Do not repeat "
                "this call: use the result you already have, change the approach or answer. "
                "The next such repeat will stop the work."
            )
            notice = (
                f"Модель {repeats}-й раз повторяет вызов {call.name} без изменений — "
                "предупреждена; следующий повтор остановит ход."
            )
        elif self._max_similar is not None and similar >= self._max_similar:
            warning = (
                f"WARNING: this is call #{similar} in a row of {call.name} for the same "
                "target. You are going in circles: use the results you already have, change "
                "the approach or move on to the report. A few more such calls will stop the "
                "work."
            )
            notice = (
                f"Модель {similar}-й раз подряд вызывает {call.name} с той же целью — "
                f"предупреждена; ход остановится на {stop_similar}-м таком вызове."
            )
        return CallCheck(repeats, warning=warning, rejected_before=rejected, notice=notice)

    def failure_warning(self) -> GuardWarning | None:
        """Предупреждение о серии неудач для следующего запроса.

        До предела — одна неудача (или серия уже на пределе: её дотянула пачка вызовов
        одного ответа).

        Заметка модели (``model``) — в каждый такой запрос: ответ может быть отброшен
        (оборван, самопроверка), и предупреждение не должно пропасть вместе с ним. Строка
        пользователю (``notice``) — один раз на серию, дальше пустая. Вызов означает, что
        заметка уйдёт в запросе: после него следующая неудача останавливает ход.
        """
        if self._max_failures < 2 or self._failures < self._max_failures - 1:
            return None
        self._failures_warned = True
        notice = ""
        if not self._failures_noticed:
            self._failures_noticed = True
            notice = (
                f"Неудачных вызовов инструментов подряд: {self._failures} — модель "
                "предупреждена; ещё одна неудача остановит ход."
            )
        return GuardWarning(FAILURES_WARNING.format(count=self._failures), notice)

    def after_tool(self, call: FunctionCall, outcome: ToolOutcome) -> StopReason | None:
        """Вызывается после каждого выполненного (или отклонённого) вызова."""
        key = call_key(call)
        failed = not (outcome.ok or outcome.rejected or outcome.soft)
        if outcome.rejected:
            self._rejected.add(key)  # серия ошибок не меняется: модель не застряла
        elif not outcome.soft:
            self._failures = 0 if outcome.ok else self._failures + 1
            if self._failures < self._max_failures - 1:
                # Серия прервалась — новая предупреждается снова.
                self._failures_warned = False
                self._failures_noticed = False
        if outcome.changed:
            self._changes += 1
        self._seen[key] = self._changes
        if self._max_similar is not None:
            similar = target_key(call)
            self._similar_seen[similar] = self._changes
            if outcome.changed:
                # Правки одного файла подряд — работа, а не топтание на месте.
                self._similar.pop(similar, None)
        # Останавливает неудача, и только после предупреждения: предел могла перейти пачка
        # вызовов одного ответа — тогда модель получит предупреждение в следующем запросе.
        if (
            failed
            and self._failures >= self._max_failures
            and (self._failures_warned or self._max_failures < 2)
        ):
            return StopReason(
                "tool_failures",
                f"Прервано: {self._failures} неудачных вызовов инструментов подряд. "
                "Похоже, агент застрял — уточните задачу или попробуйте другую модель.",
            )
        return None

    def _elapsed(self) -> float:
        return self._clock() - self._started

    def _count_similar(self, call: FunctionCall) -> int:
        """Учесть вызов как похожий на предыдущие (та же цель, без изменений между)."""
        if self._max_similar is None:
            return 0
        key = target_key(call)
        unchanged = self._similar_seen.get(key) == self._changes
        count = self._similar.get(key, 0) + 1 if unchanged else 1
        self._similar[key] = count
        return count
