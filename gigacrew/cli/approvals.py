"""Команда ``/permissions``: сохранённые разрешения, журнал решений и его сводка.

* ``/permissions`` — разрешения «всегда» и где они хранятся;
* ``/permissions forget N`` / ``clear`` — удалить одно или все;
* ``/permissions log`` — последние решения пользователя и рецензента;
* ``/permissions stats`` — как часто и почему gigacrew спрашивал пользователя.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from gigacrew.agent.ask_stats import summarize
from gigacrew.agent.decisions import read_journal, redact
from gigacrew.agent.loop import Agent
from gigacrew.ui.console import Console

__all__ = ("PERMISSIONS_USAGE", "permissions_command")

PERMISSIONS_USAGE = "/permissions [forget N | clear | log | stats]"


def permissions_command(agent: Agent, ui: Console, arg: str) -> bool:
    """``/permissions``: сохранённые разрешения «всегда», последние решения и их сводка."""
    words = arg.split()
    if not words:
        _show_permissions(agent, ui)
        return True
    action = words[0].lower()
    if action in {"forget", "rm", "удалить"}:
        _forget_permission(agent, ui, words)
    elif action in {"clear", "очистить"}:
        _clear_permissions(agent, ui)
    elif action in {"log", "журнал"}:
        _show_decisions(agent, ui)
    elif action in {"stats", "статистика"}:
        _show_ask_stats(agent, ui)
    else:
        ui.error(f"использование: {PERMISSIONS_USAGE}")
    return True


def _forget_permission(agent: Agent, ui: Console, words: list[str]) -> None:
    """``/permissions forget N``: удалить разрешение с номером N из списка."""
    # isdecimal, а не isdigit: «²» — цифра, но int() её не разберёт
    if len(words) != 2 or not words[1].isdecimal():
        ui.error(f"использование: {PERMISSIONS_USAGE}")
        return
    memory = agent.decisions
    try:
        rule = memory.forget(int(words[1]))
    except IndexError:
        ui.error(f"нет разрешения №{words[1]} (список — /permissions)")
    except OSError as error:
        ui.error(f"не удалось сохранить: {error}")
    else:
        ui.info(f"разрешение удалено: {redact(rule.describe())}")


def _clear_permissions(agent: Agent, ui: Console) -> None:
    """``/permissions clear``: удалить все сохранённые разрешения."""
    memory = agent.decisions
    try:
        removed = memory.clear()
    except OSError as error:
        ui.error(f"не удалось сохранить: {error}")
    else:
        ui.info(f"удалено разрешений: {removed}")


def _show_decisions(agent: Agent, ui: Console) -> None:
    """``/permissions log``: последние решения из журнала."""
    memory = agent.decisions
    decisions = memory.recent()
    if not decisions:
        reason = "" if memory.log else " (не ведётся: --no-save)"
        ui.info(f"журнал решений пуст{reason}")
        return
    lines = [_describe_decision(decision) for decision in decisions]
    listing = "\n".join(lines)
    ui.info(f"последние решения:\n{listing}")


_ANSWER_LABELS = MappingProxyType({
    "yes": "да",
    "always": "всегда",
    "no": "нет",
    "no_answer": "без ответа",
})
_CAUSE_LABELS = MappingProxyType({
    "dangerous": "опасная операция",
    "unverified": "команда не проверена классификатором",
    "hook": "хук требует вопроса",
    "plan": "режим плана",
    "not_command": "не команда (правка, другой инструмент)",
    "reviewer_failed": "сбой рецензента",
    "reviewer_disabled": "рецензент отключён",
    "reviewer_repeat": "агент повторил команду после отказа рецензента",
    "reviewer_limit": "много отказов рецензента",
    "rejected_before": "такую команду уже отклоняли",
    "too_long": "команда слишком длинная для рецензента",
    "other": "решение режима",
    "unknown": "причина не записана (прежняя версия)",
})
_SOURCE_LABELS = MappingProxyType({
    "read": "только чтение",
    "reviewer": "разрешил рецензент",
    "rule": "сохранённое разрешение",
    "session": "«всегда» до конца сессии",
    "turn": "«да» до конца хода",
    "yes_flag": "флаг -y",
    "hook": "разрешил хук",
    "edits_mode": "правки в режиме «авто»",
    "auto": "режим",
})
_TOP_PROGRAMS = 10


def _counts(counter: Mapping[str, int], labels: Mapping[str, str]) -> str:
    """Счётчик одной строкой, по убыванию: «подпись число · …»."""
    labelled = [(labels.get(name, name), count) for name, count in counter.items()]
    ordered = sorted(labelled, key=lambda pair: (-pair[1], pair[0]))
    return " · ".join(f"{label} {count}" for label, count in ordered)


def _show_ask_stats(agent: Agent, ui: Console) -> None:
    """``/permissions stats``: как часто и почему gigacrew спрашивал пользователя."""
    memory = agent.decisions
    summary = summarize(read_journal(memory.journal_path))
    if not summary.decided:
        reason = "" if memory.log else " (не ведётся: --no-save)"
        ui.info(f"журнал решений пуст{reason}")
        return
    share = round(100 * summary.asked / summary.decided)
    lines = [f"вопросов вам: {summary.asked} из {summary.decided} решений ({share} %)"]
    if summary.asked:
        lines += [
            f"  ответы: {_counts(summary.answers, _ANSWER_LABELS)}",
            f"  почему спросили: {_counts(summary.causes, _CAUSE_LABELS)}",
        ]
    top = summary.programs.most_common(_TOP_PROGRAMS)
    if top:
        lines.append("  чаще всего спрашивали о командах:")
        lines += [f"    • {program} — {count}" for program, count in top]
    if summary.automatic:
        lines.append(f"без вопроса: {_counts(summary.automatic, _SOURCE_LABELS)}")
    if summary.denied:
        lines.append(f"рецензент не разрешил (агент искал другой путь): {summary.denied}")
    # Ключи команд в журнале прежних версий вырезались слабее — вырезаем ещё раз.
    ui.info(redact("\n".join(lines)))


def _describe_decision(entry: Mapping[str, object]) -> str:
    """Строка журнала решений для ``/permissions log`` (секреты вырезаны)."""
    decision = str(entry.get("decision"))
    mark = {"yes": "да", "always": "всегда", "no": "нет", "no_answer": "без ответа"}.get(
        decision,
        decision,
    )
    raw_review = entry.get("review")
    review = raw_review if isinstance(raw_review, dict) else {}
    if entry.get("source") == "reviewer":
        mark = "не разрешил рецензент" if decision == "denied" else "разрешил рецензент"
    cwd = entry.get("cwd")
    where = "" if cwd in {None, "."} else f" [{cwd}]"
    text = f"  • {mark}{where}: {entry.get('command', '')}"
    if entry.get("feedback"):
        text = f"{text} — «{entry['feedback']}»"
    elif entry.get("source") == "reviewer" and review.get("reason"):
        text = f"{text} — {review['reason']}"
    # Журнал прежних версий вырезал меньше форм секретов, пояснение рецензента
    # не вырезается вовсе — на экран не выводим ни то, ни другое.
    return redact(text)


def _show_permissions(agent: Agent, ui: Console) -> None:
    """``/permissions``: сохранённые разрешения и где они хранятся."""
    memory = agent.decisions
    if memory.rules:
        lines = ["разрешено без вопросов:"]
        for number, rule in enumerate(memory.rules, 1):
            used = f", применялось {rule.hits}" if rule.hits else ""
            # Правило — команда как есть: прежние версии сохраняли и команды с паролем.
            lines.append(f"  {number}. {redact(rule.describe())}{used}")
        lines.append("удалить — /permissions forget N, все — /permissions clear")
    else:
        lines = ["сохранённых разрешений нет: ответ «a» на вопрос о команде запомнит её"]
    if memory.approvals_path is not None and memory.remember:
        lines.append(f"файл: {memory.approvals_path}")
    elif memory.directory is None or not memory.remember:
        lines.append("разрешения хранятся до конца сессии (GIGACREW_REMEMBER_APPROVALS=0)")
    ui.info("\n".join(lines))
