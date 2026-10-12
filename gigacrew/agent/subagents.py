"""Суб-агенты: события, реестр и отчёт дочернего агента.

Суб-агент — обычный :class:`~gigacrew.agent.loop.Agent` со своей историей, урезанным
реестром инструментов и своим системным промптом; запускает его основной агент
(``Agent._run_subagent``) из инструмента ``task``. Здесь — то, что нужно для запуска и
не зависит от агентного цикла:

* :class:`SubagentEvents` — события суб-агента в событиях основного: текст суб-агента
  пользователю не печатается, его инструменты и ожидание видны строками и индикатором
  под вызовом ``task``, подтверждения идут пользователю как обычно;
* :func:`subagent_registry` — инструменты, доступные суб-агенту;
* :func:`subagent_report` — что вернётся основному агенту.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gigacrew.agent.events import (
    AgentEvents,
    Approval,
    CompactResult,
    NoticeLevel,
    SubagentInfo,
    ToolCallInfo,
    TurnStats,
)
from gigacrew.permissions import ToolKind
from gigacrew.tools.base import Display, ToolRegistry, ToolResult
from gigacrew.tools.task import SubagentSpec, allows_tool

# Сколько суб-агентов основной агент может запустить за один ход.
MAX_SUBAGENTS_PER_TURN = 8
# Предел текста отчёта: окно по умолчанию — 32K, а при сжатии истории результаты
# инструментов всё равно режутся до 2K символов.
MAX_REPORT_CHARS = 8_000
REJECTED_MARK = "отклонено пользователем"

SUBAGENT_STOP_NOTE = (
    "=== РАБОТА ОСТАНОВЛЕНА ===\n{reason}\nИнструменты больше не вызывай. Дай итоговый "
    "отчёт для основного агента: что уже выяснил или сделал (с путями и строками), что "
    "не успел и что предлагаешь проверить дальше."
)
SUBAGENT_USER_STOP_NOTE = (
    "Пользователь остановил тебя досрочно. Инструменты больше не вызывай. Сразу дай "
    "итоговый отчёт для основного агента по уже сделанному: что выяснил или сделал (с "
    "путями и строками), что не успел."
)
READ_ONLY_BLOCKED_NOTE = (
    "Не выполнено: ты суб-агент только для исследования — любые изменения запрещены. "
    "Не пытайся их вносить: опиши нужные изменения в итоговом отчёте."
)
SUBAGENT_PLAN_BLOCKED_NOTE = (
    "Не выполнено: включён режим планирования, изменения запрещены. Не пытайся вносить "
    "их: опиши нужные изменения в итоговом отчёте."
)
USER_STOPPED_NOTE = (
    "Суб-агент {name} остановлен пользователем досрочно — ниже его итог по сделанному, "
    "он может быть неполным. Не запускай эту задачу снова, если пользователь не попросит."
)
LIMIT_STOPPED_NOTE = "Суб-агент {name} остановлен: {reason} — отчёт может быть неполным."
ABORTED_NOTE = (
    "Суб-агент {name} оборван пользователем, отчёта нет. Не запускай эту задачу снова, "
    "если пользователь не попросит; продолжай сам с тем, что известно."
)
INTERRUPTED_NOTE = "Суб-агент {name} прерван пользователем (Ctrl+C). Результата нет."
NO_REPORT_NOTE = "Суб-агент {name} не вернул отчёта."

_STOP_REASONS = {
    "max_steps": "исчерпан лимит шагов",
    "context_limit": "контекст заполнен",
    "time_limit": "исчерпан лимит времени",
    "tool_failures": "серия неудачных вызовов инструментов",
    "tool_repeats": "зацикливание (повтор одного вызова)",
    "similar_calls": "зацикливание (похожие вызовы с той же целью)",
}


@dataclass
class SubagentTrail:
    """Что суб-агент сделал с проектом — для отчёта (не со слов модели, а по фактам)."""

    changed_files: list[str] = field(default_factory=list)
    # Прочие изменяющие операции: команды, изменяющий git.
    operations: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.changed_files or self.operations)

    def lines(self) -> list[str]:
        out = []
        if self.changed_files:
            out.append(f"Изменённые файлы: {', '.join(dict.fromkeys(self.changed_files))}")
        if self.operations:
            out.append(f"Изменяющие операции: {'; '.join(dict.fromkeys(self.operations))}")
        if self.rejected:
            out.append(f"Отклонено пользователем: {'; '.join(dict.fromkeys(self.rejected))}")
        return out


class SubagentEvents(AgentEvents):
    """События суб-агента → события основного агента (``parent``)."""

    def __init__(self, parent: AgentEvents, info: SubagentInfo):
        self._parent = parent
        self._info = info
        self._wrapping = False
        self.trail = SubagentTrail()

    def _activity(self, activity: str) -> None:
        self._parent.on_subagent_activity(self._info, activity, wrapping_up=self._wrapping)

    # Текст суб-агента не печатается: отчёт получит основной агент.
    def on_stream_start(self) -> None:
        self._activity("подвожу итог" if self._wrapping else "думаю")

    def on_tool_call(self, call: ToolCallInfo) -> None:
        self._parent.on_subagent_tool_call(self._info, call)

    def confirm(
        self, call: ToolCallInfo, preview: Display | None, *, dangerous: bool
    ) -> bool | Approval:
        return self._parent.confirm(call, preview, dangerous=dangerous)

    def on_tool_start(self, call: ToolCallInfo) -> None:
        self._activity(call.name)

    def on_tool_result(self, call: ToolCallInfo, result: ToolResult, *, previewed: bool) -> None:
        if result.ok and result.changed:
            if call.kind is ToolKind.EDIT and call.summary:
                self.trail.changed_files.append(call.summary)
            else:
                self.trail.operations.append(f"{call.name} {call.summary}".strip())
        elif not result.ok and REJECTED_MARK in result.content:
            self.trail.rejected.append(f"{call.name} {call.summary}".strip())
        self._parent.on_subagent_tool_result(self._info, call, result, previewed=previewed)

    def on_compact_start(self, *, auto: bool) -> None:
        self._activity("сжимаю контекст")

    def on_compact_end(self, result: CompactResult | None) -> None:
        """Не передаётся: интерфейс остановил бы индикатор суб-агента."""

    def on_wrap_up(self, reason: str) -> None:
        self._wrapping = True
        self._activity("подвожу итог")

    def on_notice(self, text: str, *, level: NoticeLevel = "info") -> None:
        # Остановка суб-агента — не остановка хода: основной агент продолжает.
        self._parent.on_notice(
            f"{self._info.name}: {text}", level="warn" if level == "error" else level
        )

    def on_turn_end(self, stats: TurnStats) -> None:
        """Итоги суб-агента показываются строкой результата ``task``."""


def subagent_registry(registry: ToolRegistry, spec: SubagentSpec) -> ToolRegistry:
    """Инструменты основного агента, доступные суб-агенту ``spec``."""
    return registry.subset(lambda tool: allows_tool(spec, tool.name))


def _clip(text: str) -> str:
    text = text.strip()
    if len(text) <= MAX_REPORT_CHARS:
        return text
    return text[:MAX_REPORT_CHARS].rstrip() + "\n…(отчёт обрезан)"


def _summary(spec: SubagentSpec, stats: TurnStats | None, tail: str = "") -> str:
    parts = [spec.name]
    if stats is not None:
        parts.append(f"инструментов: {stats.tool_calls}")
        if stats.duration_s:
            parts.append(f"{stats.duration_s:.0f} с")
    if tail:
        parts.append(tail)
    return " · ".join(parts)


def subagent_report(
    spec: SubagentSpec, text: str, stats: TurnStats | None, trail: SubagentTrail
) -> ToolResult:
    """Результат ``task`` для основного агента: отчёт суб-агента и факты о сделанном."""
    stop = stats.stop_reason if stats is not None else None
    parts = []
    tail = ""
    if stop == "user_stop":
        parts.append(USER_STOPPED_NOTE.format(name=spec.name))
        tail = "остановлен пользователем"
    elif stop is not None:
        reason = _STOP_REASONS.get(stop, stop)
        parts.append(LIMIT_STOPPED_NOTE.format(name=spec.name, reason=reason))
        tail = reason
    if text.strip():
        parts.append(_clip(text))
    else:
        parts.append(NO_REPORT_NOTE.format(name=spec.name))
    facts = trail.lines()
    if facts:
        parts.append("\n".join(facts))
    return ToolResult(
        content="\n\n".join(parts),
        ok=bool(text.strip()),
        summary=_summary(spec, stats, tail),
        changed=stats is not None and stats.changes > 0,
    )


def aborted_report(spec: SubagentSpec, trail: SubagentTrail) -> ToolResult:
    """Суб-агент оборван пользователем без отчёта (повторная остановка)."""
    parts = [ABORTED_NOTE.format(name=spec.name), *trail.lines()]
    return ToolResult(
        content="\n\n".join(parts),
        ok=False,
        summary=_summary(spec, None, "оборван пользователем"),
        changed=trail.changed,
    )


def model_error_report(spec: SubagentSpec, error: Exception, trail: SubagentTrail) -> ToolResult:
    """Суб-агент упал на ошибке модели: ошибка + что он успел сделать до неё."""
    parts = [f"суб-агент {spec.name}: ошибка модели: {error}", *trail.lines()]
    return ToolResult(
        content="\n\n".join(parts),
        ok=False,
        summary=_summary(spec, None, "ошибка модели"),
        changed=trail.changed,
    )


def interrupted_note(spec: SubagentSpec, trail: SubagentTrail) -> str:
    """Результат ``task`` в истории основного агента после Ctrl+C (ход остановлен)."""
    return "\n\n".join([INTERRUPTED_NOTE.format(name=spec.name), *trail.lines()])
