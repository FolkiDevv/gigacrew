"""Фоновые суб-агенты и заметки о фоновых задачах для модели.

Фоновый суб-агент — тот же :class:`~gigacrew.agent.loop.Agent`, что и у ``task``, но он
работает в своём потоке (:class:`SubagentJob`), пока основной агент продолжает ход или
ждёт ввода. Отличия от обычного суб-агента:

* его события не идут в интерфейс — интерфейс занят основным агентом; они копятся в
  задаче (:class:`BackgroundSubagentEvents`): чем занят, сколько вызовов, что изменил;
* спросить пользователя он не может: подтверждение посреди чужого хода некому показать,
  поэтому всё, что требует подтверждения, ему отказано (как при запуске без клавиатуры), а
  отчёт перечисляет, что осталось сделать с подтверждением;
* отмена — флагом (:attr:`SubagentJob.cancel_event`), который суб-агент проверяет перед
  каждым шагом и каждым вызовом инструмента; если он завис внутри шага, задачу
  остановит предел :class:`~gigacrew.tools.jobs.BackgroundJobs`.

Здесь же — заметки для модели: итоги завершившихся задач (:func:`finished_note`) и
напоминание о незавершённых (:func:`active_note`).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, override

from gigacrew.agent.conversation import BACKGROUND_HEADER
from gigacrew.agent.events import AgentEvents, Approval, ConfirmResult
from gigacrew.agent.subagents import SubagentTrail
from gigacrew.tools.jobs import (
    DEFAULT_LIMITS,
    SUBAGENT_POOL,
    Job,
    JobCancelledError,
    JobReport,
    JobState,
    PoolLimits,
    job_heading,
)

if TYPE_CHECKING:
    # Типы нужны только аннотациям.
    from gigacrew.agent.events import ToolCallInfo
    from gigacrew.tools.base import Display, ToolResult
    from gigacrew.tools.task import SubagentSpec

__all__ = [
    "BackgroundSubagentEvents",
    "SubagentJob",
    "active_note",
    "finished_note",
    "job_limits",
    "started_note",
]

# Заметки для модели — на английском, как и промпты (пользователь их не видит).
STARTED_NOTE = (
    "Subagent {name} started in the background as job {job_id}. Continue with other work: "
    "its report will be delivered to you automatically when it finishes — do not poll it "
    "and do not start the same task again. If you have nothing else to do until then, end "
    "your turn: you will be woken with the report. background_stop stops it."
)
QUEUED_NOTE = (
    "Subagent {name} is queued as job {job_id}: the limit of background subagents running "
    "at once is reached. It starts by itself when one of them finishes, and its report "
    "will be delivered to you automatically. Do not start the same task again; "
    "background_stop removes it from the queue."
)
STOPPED_NOTE = "Subagent {name} was stopped before it finished ({reason}); there is no report."
FAILED_NOTE = "Subagent {name} failed: {error}"
FINISHED_NOTE = (
    "Background jobs you started have finished; their results are below. Take them into "
    "account and continue the user's task."
)
ACTIVE_NOTE = (
    "# Background jobs not finished yet\n{jobs}\nTheir results will be delivered to you "
    "automatically — do not poll them and do not start them again. In your final answer "
    "say which of them are still running. If the rest of the task depends on them, end the "
    "turn: you will be woken with their results."
)

# Запуск суб-агента фоновой задачи: задача → результат его отчёта.
SubagentRun = Callable[["SubagentJob"], "ToolResult"]


class BackgroundSubagentEvents(AgentEvents):
    """События фонового суб-агента: копятся в задаче, в интерфейс не идут.

    Подтверждения некому показать — ответ «нет ответа» (как при запуске без клавиатуры).
    Пишет поток суб-агента, читает основной (для ``/tasks`` и ``background_output``).
    """

    def __init__(self) -> None:
        """События нового суб-агента."""
        self.trail = SubagentTrail()
        self.activity = "запускается"
        self.tool_calls = 0

    @override
    def on_stream_start(self) -> None:
        self.activity = "думает"

    @override
    def confirm(
        self,
        call: ToolCallInfo,
        preview: Display | None,
        *,
        dangerous: bool,
    ) -> bool | Approval | ConfirmResult:
        return ConfirmResult(Approval.NO, answered=False)

    @override
    def can_confirm(self) -> bool:
        return False

    @override
    def on_review_start(self, call: ToolCallInfo) -> None:
        self.activity = "рецензент проверяет команду"

    @override
    def on_tool_start(self, call: ToolCallInfo) -> None:
        self.activity = f"{call.name} {call.summary}".strip()

    @override
    def on_tool_result(
        self,
        call: ToolCallInfo,
        tool_result: ToolResult,
        *,
        previewed: bool,
    ) -> None:
        self.tool_calls += 1
        self.trail.record(call, tool_result)

    @override
    def on_compact_start(self, *, auto: bool) -> None:
        self.activity = "сжимает контекст"

    @override
    def on_wrap_up(self, reason: str) -> None:
        self.activity = "подводит итог"


class SubagentJob(Job):
    """Суб-агент фоновой задачей: работает в своём потоке, его отчёт — итог задачи."""

    pool = SUBAGENT_POOL

    def __init__(
        self,
        spec: SubagentSpec,
        description: str,
        *,
        timeout: float,
        run: SubagentRun,
    ) -> None:
        """Суб-агент ``spec`` с задачей ``description``; ``run`` запускает его и ждёт отчёта."""
        super().__init__(spec.name, description, timeout=timeout)
        self.events = BackgroundSubagentEvents()
        self.cancel_event = threading.Event()
        self._run = run

    @override
    def start(self) -> None:
        threading.Thread(target=self._work, name=f"gigacrew-{self.id}", daemon=True).start()

    @override
    def cancel(self) -> None:
        self.cancel_event.set()

    @override
    def stopped_report(self, state: JobState, reason: str) -> JobReport:
        parts = [STOPPED_NOTE.format(name=self.kind, reason=reason), *self.events.trail.lines()]
        return JobReport(state, "\n\n".join(parts), f"{self.kind} · остановлен")

    @override
    def progress(self) -> str:
        events = self.events
        parts = [
            f"Current activity: {events.activity}; tool calls so far: {events.tool_calls}.",
            *events.trail.lines(),
        ]
        return "\n".join(parts)

    @override
    def activity(self) -> str:
        events = self.events
        return f"{events.activity} · инструментов: {events.tool_calls}"

    def _work(self) -> None:
        """Тело потока: суб-агент до отчёта; отмена и сбой — тоже итог задачи."""
        try:
            outcome = self._run(self)
        except JobCancelledError:
            state, reason = self.stop_request or (JobState.STOPPED, "")
            self.finish(self.stopped_report(state, reason))
        except Exception as exc:  # сбой внутри суб-агента — итог задачи, а не тишина
            error = f"{type(exc).__name__}: {exc}"
            parts = [FAILED_NOTE.format(name=self.kind, error=error), *self.events.trail.lines()]
            self.finish(JobReport(JobState.FAILED, "\n\n".join(parts), f"{self.kind} · сбой"))
        else:
            state = JobState.DONE if outcome.ok else JobState.FAILED
            self.finish(JobReport(state, outcome.content, outcome.summary))


def job_limits(subagents: int, queued: int) -> dict[str, PoolLimits]:
    """Пределы фоновых задач: ``subagents`` суб-агентов одновременно и ``queued`` в очереди."""
    return {**DEFAULT_LIMITS, SUBAGENT_POOL: PoolLimits(subagents, queued)}


def started_note(job: Job) -> str:
    """Результат вызова ``task`` с ``run_in_background`` для модели."""
    template = QUEUED_NOTE if job.queued else STARTED_NOTE
    return template.format(name=job.kind, job_id=job.id)


def finished_note(jobs: Sequence[Job]) -> str:
    """Итоги завершившихся задач для модели (заголовок — :data:`BACKGROUND_HEADER`)."""
    parts = [f"{BACKGROUND_HEADER}\n{FINISHED_NOTE}"]
    for job in jobs:
        report = job.report
        content = "" if report is None else report.content
        parts.append(f"## {job_heading(job)}\n{content}")
    return "\n\n".join(parts)


def active_note(jobs: Sequence[Job]) -> str:
    """Напоминание модели о незавершённых задачах (работают и ждут в очереди)."""
    listed = "\n".join(f"- {job_heading(job)}" for job in jobs)
    return ACTIVE_NOTE.format(jobs=listed)
