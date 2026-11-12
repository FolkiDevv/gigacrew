"""Инструменты фоновых задач: ``background_output`` и ``background_stop``.

Итог завершившейся фоновой задачи агентный цикл доставляет модели сам — опрашивать
задачу не нужно. Эти инструменты — чтобы посмотреть промежуточный вывод, дождаться
задачи, когда больше делать нечего, и остановить ненужную.
"""

from __future__ import annotations

import time
from typing import override

from pydantic import BaseModel, Field

from gigacrew.tools.base import Tool, ToolContext, ToolError, ToolResult
from gigacrew.tools.jobs import STATE_LABELS, BackgroundJobs, Job, job_heading

# Дольше одного вызова модель не ждёт: дальше её разбудит итог задачи.
MAX_WAIT = 300
# Ожидание — короткими отрезками: Esc/Ctrl+C прерывает его сразу.
_WAIT_SLICE = 0.2


# Без docstring: он ушёл бы в JSON-схему параметров, которую видит модель.
class BackgroundOutputParams(BaseModel):  # noqa: D101
    job_id: str = Field(
        "",
        description="Job id from run_in_background (bg1, …). Empty — list all background jobs.",
    )
    wait: int = Field(
        0,
        description=(
            f"Seconds to wait for the job to finish before answering (0–{MAX_WAIT}); "
            "0 — answer at once."
        ),
    )


# Без docstring: он ушёл бы в JSON-схему параметров, которую видит модель.
class BackgroundStopParams(BaseModel):  # noqa: D101
    job_id: str = Field(description="Job id from run_in_background (bg1, …).")


def _jobs(ctx: ToolContext) -> BackgroundJobs:
    if ctx.background is None:
        raise ToolError("фоновых задач здесь нет.")
    return ctx.background


def _job(jobs: BackgroundJobs, job_id: str) -> Job:
    job = jobs.get(job_id)
    if job is not None:
        return job
    known = ", ".join(known_job.id for known_job in jobs.jobs()) or "нет ни одной"
    raise ToolError(f"фоновой задачи {job_id.strip()!r} нет; задачи этого чата: {known}.")


def _wait(job: Job, seconds: int) -> None:
    deadline = time.monotonic() + seconds
    while not job.finished:
        left = deadline - time.monotonic()
        if left <= 0:
            return
        job.wait(min(left, _WAIT_SLICE))


def _job_text(job: Job, prefix: str = "") -> str:
    """Итог задачи для модели: заголовок и её отчёт."""
    report = job.report
    content = "" if report is None else report.content
    return f"{prefix}{job_heading(job)}\n{content}"


def _summary(job: Job) -> str:
    report = job.report
    label = report.summary if report is not None and report.summary else STATE_LABELS[job.state]
    return f"{job.id} · {label}"


class BackgroundOutputTool(Tool[BackgroundOutputParams]):
    """Инструмент background_output: состояние и новый вывод фоновой задачи."""

    name = "background_output"
    description = (
        "Shows a background job started with run_in_background (run_shell or task): its "
        "state and what is new — command output since the last check, a subagent's current "
        "activity; for a finished job — its full result. A finished job's result is "
        "delivered to you automatically, so do not poll: use wait (seconds) only when you "
        "have nothing else to do until the job finishes. Without job_id — lists all "
        "background jobs."
    )
    # Модель параметров, а не константа: имя задаёт базовый класс Tool.
    Params = BackgroundOutputParams  # noqa: WPS115
    read_only = True

    @override
    def describe(self, params: BackgroundOutputParams) -> str:
        """Id задачи (и ожидание) для UI."""
        job_id = params.job_id.strip() or "все задачи"
        return f"{job_id} · жду до {params.wait} с" if params.wait > 0 else job_id

    @override
    def run(self, params: BackgroundOutputParams, ctx: ToolContext) -> ToolResult:
        """Состояние задачи (после ожидания, если оно задано) или список задач."""
        jobs = _jobs(ctx)
        if not params.job_id.strip():
            return _listing(jobs)
        job = _job(jobs, params.job_id)
        _wait(job, min(max(params.wait, 0), MAX_WAIT))
        if not job.finished:
            return ToolResult(
                content=(
                    f"{job_heading(job)}\n{job.progress()}\nIts result will be delivered to "
                    "you automatically when it finishes."
                ),
                summary=f"{job.id} · работает {int(job.elapsed)} с",
            )
        jobs.mark_delivered(job)
        return ToolResult(content=_job_text(job), summary=_summary(job))


class BackgroundStopTool(Tool[BackgroundStopParams]):
    """Инструмент background_stop: остановить фоновую задачу."""

    name = "background_stop"
    description = (
        "Stops a background job started with run_in_background (a command — together with "
        "its child processes, a subagent — at its next step) and returns what it produced so "
        "far. Use it for a job that is no longer needed or went wrong."
    )
    # Модель параметров, а не константа: имя задаёт базовый класс Tool.
    Params = BackgroundStopParams  # noqa: WPS115

    @override
    def describe(self, params: BackgroundStopParams) -> str:
        """Id задачи для UI."""
        return params.job_id.strip()

    @override
    def run(self, params: BackgroundStopParams, ctx: ToolContext) -> ToolResult:
        """Остановить задачу и вернуть её итог."""
        jobs = _jobs(ctx)
        job = _job(jobs, params.job_id)
        stopped = jobs.stop(job, "stopped by the agent (background_stop)")
        jobs.mark_delivered(job)
        prefix = "" if stopped else "The job had already finished before the stop. "
        return ToolResult(content=_job_text(job, prefix), summary=_summary(job))


def _listing(jobs: BackgroundJobs) -> ToolResult:
    known = jobs.jobs()
    if not known:
        return ToolResult(content="No background jobs in this chat.", summary="задач нет")
    lines = "\n".join(f"- {job_heading(job)}" for job in known)
    active = sum(not job.finished for job in known)
    return ToolResult(
        content=f"Background jobs of this chat:\n{lines}",
        summary=f"задач: {len(known)}, не завершены: {active}",
    )
