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
from types import MappingProxyType
from typing import TYPE_CHECKING, override

from gigacrew.agent.events import AgentEvents
from gigacrew.agent.guard import MODEL_STOP_REASONS
from gigacrew.permissions import ToolKind
from gigacrew.tools.base import Display, ToolRegistry, ToolResult
from gigacrew.tools.process import quote_reminder_tags
from gigacrew.tools.task import SubagentSpec, allows_tool

if TYPE_CHECKING:
    # Типы событий нужны только аннотациям.
    from gigacrew.agent.events import (
        Approval,
        CompactResult,
        ConfirmResult,
        NoticeLevel,
        Review,
        SubagentInfo,
        ToolCallInfo,
        TurnStats,
    )

# Сколько суб-агентов основной агент может запустить за один ход.
MAX_SUBAGENTS_PER_TURN = 8
# Предел текста отчёта: окно по умолчанию — 32K, а при сжатии истории результаты
# инструментов всё равно режутся до 2K символов.
MAX_REPORT_CHARS = 8_000
REJECTED_MARK = "отклонено пользователем"

# Заметки для моделей — на английском, как и промпты (пользователь их не видит).
SUBAGENT_STOP_NOTE = (
    "STOP: {reason}. Do not call tools anymore. Write the final report for the main "
    "agent now: what you found out or did (with paths and lines), what you did not get "
    "to and what you suggest checking next."
)
SUBAGENT_USER_STOP_NOTE = (
    "The user stopped you early. Do not call tools anymore. Write the final report for "
    "the main agent right away, based on what is already done: what you found out or did "
    "(with paths and lines), what you did not get to."
)
READ_ONLY_BLOCKED_NOTE = (
    "Not executed: you are a research-only subagent, any changes are forbidden. Do not "
    "try to make them: describe the needed changes in the final report."
)
READ_ONLY_SHELL_NOTE = (
    "Not executed: here run_shell runs only commands verified as read-only inside the "
    "project (ls, cat, grep, rg, git status/log/diff, Python one-liners that only read); "
    "this one is not: {reason}. Shell loops and if/case blocks, the project's tests and "
    "builds and file paths computed at run time ($(…), $VAR) are not verified. Do not just "
    "rephrase the command: use read_file, search_content, find_files and git instead, or "
    "put the command in the final report for the main agent."
)
SUBAGENT_PLAN_BLOCKED_NOTE = (
    "Not executed: plan mode is on, changes are forbidden. Do not try to make them: "
    "describe the needed changes in the final report."
)
USER_STOPPED_NOTE = (
    "Subagent {name} was stopped early by the user — below is its summary of what was "
    "done, it may be incomplete. Do not run this task again unless the user asks."
)
LIMIT_STOPPED_NOTE = (
    "Subagent {name} was stopped: {reason} — the report below may be incomplete. Do not "
    "rerun the same task: use what it found; if something is still missing, finish it "
    "yourself or delegate a narrower follow-up that says what is already known and asks "
    "only for what remains."
)
ABORTED_NOTE = (
    "Subagent {name} was aborted by the user, there is no report. Do not run this task "
    "again unless the user asks; continue yourself with what is known."
)
INTERRUPTED_NOTE = "Subagent {name} was interrupted by the user (Ctrl+C). No result."
NO_REPORT_NOTE = "Subagent {name} returned no report."

# Причины остановки — для строки результата task в интерфейсе.
_STOP_REASONS = MappingProxyType({
    "max_steps": "исчерпан лимит шагов",
    "context_limit": "контекст заполнен",
    "time_limit": "исчерпан лимит времени",
    "tool_failures": "серия неудачных вызовов инструментов",
    "tool_repeats": "зацикливание (повтор одного вызова)",
    "similar_calls": "зацикливание (похожие вызовы с той же целью)",
    "output_limit": "ответы модели обрывались",
})


@dataclass
class SubagentTrail:
    """Что суб-агент сделал с проектом — для отчёта (не со слов модели, а по фактам)."""

    changed_files: list[str] = field(default_factory=list)
    # Прочие изменяющие операции: команды, изменяющий git.
    operations: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        """Суб-агент что-то изменил: файлы или изменяющие операции."""
        return bool(self.changed_files or self.operations)

    def lines(self) -> list[str]:
        """Строки фактов для отчёта (повторы убраны, порядок сохранён)."""
        out = []
        if self.changed_files:
            files = ", ".join(dict.fromkeys(self.changed_files))
            out.append(f"Changed files: {files}")
        if self.operations:
            operations = "; ".join(dict.fromkeys(self.operations))
            out.append(f"Changing operations: {operations}")
        if self.rejected:
            rejected = "; ".join(dict.fromkeys(self.rejected))
            out.append(f"Rejected by the user: {rejected}")
        return out


class SubagentEvents(AgentEvents):
    """События суб-агента → события основного агента (``parent``)."""

    def __init__(self, parent: AgentEvents, subagent: SubagentInfo):
        """События суб-агента ``subagent`` передаются в события ``parent``."""
        self._parent = parent
        self._subagent = subagent
        self._wrapping = False
        self.trail = SubagentTrail()

    # Текст суб-агента не печатается: отчёт получит основной агент.
    @override
    def on_stream_start(self) -> None:
        self._activity("подвожу итог" if self._wrapping else "думаю")

    @override
    def on_tool_call(self, call: ToolCallInfo) -> None:
        self._parent.on_subagent_tool_call(self._subagent, call)

    @override
    def confirm(
        self,
        call: ToolCallInfo,
        preview: Display | None,
        *,
        dangerous: bool,
    ) -> bool | Approval | ConfirmResult:
        return self._parent.confirm(call, preview, dangerous=dangerous)

    @override
    def on_review_start(self, call: ToolCallInfo) -> None:
        self._activity("рецензент проверяет команду")

    @override
    def on_review_end(self, call: ToolCallInfo, review: Review | None) -> None:
        # Не передаётся как есть: интерфейс остановил бы индикатор суб-агента.
        if review is not None and review.allowed:
            name = self._subagent.name
            self._parent.on_notice(f"{name}: {review.approval}")

    @override
    def on_tool_start(self, call: ToolCallInfo) -> None:
        self._activity(call.name)

    @override
    def on_tool_result(self, call: ToolCallInfo, result: ToolResult, *, previewed: bool) -> None:
        if result.ok and result.changed:
            if call.kind is ToolKind.EDIT and call.summary:
                self.trail.changed_files.append(call.summary)
            else:
                self.trail.operations.append(f"{call.name} {call.summary}".strip())
        elif not result.ok and REJECTED_MARK in result.content:
            self.trail.rejected.append(f"{call.name} {call.summary}".strip())
        self._parent.on_subagent_tool_result(self._subagent, call, result, previewed=previewed)

    @override
    def on_compact_start(self, *, auto: bool) -> None:
        self._activity("сжимаю контекст")

    @override
    def on_compact_end(self, result: CompactResult | None) -> None:
        """Не передаётся: интерфейс остановил бы индикатор суб-агента."""

    @override
    def on_hook_start(self, event: str, command: str) -> None:
        self._activity(f"хук {event}")

    @override
    def on_wrap_up(self, reason: str) -> None:
        self._wrapping = True
        self._activity("подвожу итог")

    @override
    def on_notice(self, text: str, *, level: NoticeLevel = "info") -> None:
        # Остановка суб-агента — не остановка хода: основной агент продолжает.
        name = self._subagent.name
        self._parent.on_notice(
            f"{name}: {text}",
            level="warn" if level == "error" else level,
        )

    @override
    def on_turn_end(self, stats: TurnStats) -> None:
        """Итоги суб-агента показываются строкой результата ``task``."""

    def _activity(self, activity: str) -> None:
        self._parent.on_subagent_activity(self._subagent, activity, wrapping_up=self._wrapping)


def subagent_registry(registry: ToolRegistry, spec: SubagentSpec) -> ToolRegistry:
    """Инструменты основного агента, доступные суб-агенту ``spec``."""
    return registry.subset(lambda tool: allows_tool(spec, tool.name))


def _clip(text: str) -> str:
    text = text.strip()
    if len(text) <= MAX_REPORT_CHARS:
        return text
    head = text[:MAX_REPORT_CHARS].rstrip()
    return f"{head}\n…(отчёт обрезан)"


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
    spec: SubagentSpec,
    text: str,
    stats: TurnStats | None,
    trail: SubagentTrail,
) -> ToolResult:
    """Результат ``task`` для основного агента: отчёт суб-агента и факты о сделанном."""
    stop = None if stats is None else stats.stop_reason
    parts = []
    tail = ""
    if stop == "user_stop":
        parts.append(USER_STOPPED_NOTE.format(name=spec.name))
        tail = "остановлен пользователем"
    elif stop is not None:
        reason = MODEL_STOP_REASONS.get(stop, stop)
        parts.append(LIMIT_STOPPED_NOTE.format(name=spec.name, reason=reason))
        tail = _STOP_REASONS.get(stop, stop)
    if text.strip():
        # Отчёт пересказывает файлы и вывод команд — тег заметок gigacrew в нём цитата.
        parts.append(_clip(quote_reminder_tags(text)))
    else:
        parts.append(NO_REPORT_NOTE.format(name=spec.name))
        tail = f"{tail} · без отчёта" if tail else "без отчёта"
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
    parts = [f"Subagent {spec.name}: model error: {error}", *trail.lines()]
    return ToolResult(
        content="\n\n".join(parts),
        ok=False,
        summary=_summary(spec, None, "ошибка модели"),
        changed=trail.changed,
    )


def interrupted_note(spec: SubagentSpec, trail: SubagentTrail) -> str:
    """Результат ``task`` в истории основного агента после Ctrl+C (ход остановлен)."""
    return "\n\n".join([INTERRUPTED_NOTE.format(name=spec.name), *trail.lines()])
