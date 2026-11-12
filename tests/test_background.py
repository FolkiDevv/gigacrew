"""Фоновые задачи: очередь и пределы, фоновые команды, суб-агенты, доставка итогов и пробуждение."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, override

import pytest
from fakes import FakeTool, RecordingEvents, ScriptedProvider, text_turn, tool_turn, tools_turn
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from gigacrew.agent import loop
from gigacrew.agent.conversation import BACKGROUND_HEADER, request_text
from gigacrew.agent.reviewer import recent_requests
from gigacrew.cli import repl
from gigacrew.cli.app import EXIT_OK, run_oneshot
from gigacrew.cli.commands import CommandRegistry, default_commands
from gigacrew.cli.prompt import StatusInfo, create_prompt_session
from gigacrew.cli.reader import BackgroundWakeError, PromptReader
from gigacrew.config import Config
from gigacrew.llm.types import AssistantTurn, Message, ToolSpec
from gigacrew.project.workspace import Workspace
from gigacrew.tools import jobs as jobs_module
from gigacrew.tools.background import (
    BackgroundOutputParams,
    BackgroundOutputTool,
    BackgroundStopParams,
    BackgroundStopTool,
)
from gigacrew.tools.base import (
    ToolContext,
    ToolError,
    ToolRegistry,
    ToolResult,
    build_default_registry,
)
from gigacrew.tools.jobs import BackgroundJobs, Job, JobInfo, JobReport, JobState, PoolLimits
from gigacrew.tools.process import OutputLog
from gigacrew.tools.shell import RunShellParams, RunShellTool
from gigacrew.ui.console import Console

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="команды /bin/sh")


class FakeJob(Job):
    """Задача, которую завершает тест; ``obeys=False`` — работа не слушает отмену (зависла)."""

    pool = jobs_module.SUBAGENT_POOL

    def __init__(self, title: str = "задача", *, timeout: float = 60, obeys: bool = True) -> None:
        super().__init__("fake", title, timeout=timeout)
        self.started = False
        self.cancelled = False
        self._obeys = obeys

    @override
    def start(self) -> None:
        self.started = True

    @override
    def cancel(self) -> None:
        self.cancelled = True
        if self._obeys:
            state, reason = self.stop_request or (JobState.STOPPED, "")
            self.finish(JobReport(state, f"stopped: {reason}"))

    @override
    def stopped_report(self, state: JobState, reason: str) -> JobReport:
        return JobReport(state, f"forced: {reason}")

    @override
    def progress(self) -> str:
        return "в процессе"

    def done(self, content: str = "ok") -> None:
        self.finish(JobReport(JobState.DONE, content, "готово"))


class BackgroundProvider(ScriptedProvider):
    """Основной поток берёт ходы из сценария, фоновые суб-агенты — из ``background``.

    ``release`` — фоновые запросы ждут его: тест решает, когда суб-агент ответит.
    """

    def __init__(
        self,
        turns: Iterable[AssistantTurn],
        *,
        background: Iterable[AssistantTurn] = (),
        release: threading.Event | None = None,
    ) -> None:
        super().__init__(turns)
        self.background = list(background)
        self.background_requests: list[list[Message]] = []
        self._release = release

    @override
    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> AssistantTurn:
        if threading.current_thread() is threading.main_thread():
            return super().complete(messages, tools, **kwargs)
        if self._release is not None:
            self._release.wait(10)
        self.background_requests.append(list(messages))
        return self.background.pop(0) if self.background else text_turn("отчёт")


class HookedProvider(ScriptedProvider):
    """Перед каждым запросом вызывает ``on_request(номер запроса)`` — событие посреди хода."""

    def __init__(self, turns: Iterable[AssistantTurn], on_request: Callable[[int], None]) -> None:
        super().__init__(turns)
        self._on_request = on_request

    @override
    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> AssistantTurn:
        self._on_request(len(self.requests))
        return super().complete(messages, tools, **kwargs)


class BackgroundEvents(RecordingEvents):
    """Записывает и итоги фоновых задач, переданные модели."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.background: list[JobInfo] = []

    @override
    def on_background_end(self, job: JobInfo) -> None:
        self.background.append(job)


def _agent(
    provider: ScriptedProvider,
    tmp_path: Path,
    *,
    events: RecordingEvents | None = None,
    registry: ToolRegistry | None = None,
    **cfg_kw: Any,
) -> loop.Agent:
    cfg_kw.setdefault("auto_approve", True)
    cfg = Config(access_key="x", project_root=tmp_path, stream=False, **cfg_kw)
    return loop.Agent(
        provider,
        registry or build_default_registry(),
        cfg,
        events or RecordingEvents(),
    )


def _jobs(agent: loop.Agent) -> BackgroundJobs:
    jobs = agent.background
    assert jobs is not None
    return jobs


def _job(jobs: BackgroundJobs, job_id: str = "bg1") -> Job:
    job = jobs.get(job_id)
    assert job is not None
    return job


def _report(job: Job) -> JobReport:
    report = job.report
    assert report is not None
    return report


def _state(job: Job) -> JobState:
    """Состояние задачи сейчас (вызов — чтобы mypy не сужал тип по прошлой проверке)."""
    return job.state


def _sent(provider: ScriptedProvider, index: int = -1) -> Message:
    """Последнее сообщение запроса ``index`` к модели."""
    messages: list[Message] = provider.requests[index]["messages"]
    return messages[-1]


def _ctx(tmp_path: Path, jobs: BackgroundJobs | None = None) -> ToolContext:
    return ToolContext(workspace=Workspace(tmp_path), background=jobs)


def _bg_args(prompt: str, subagent: str = "explore") -> dict[str, Any]:
    return {
        "subagent_type": subagent,
        "prompt": prompt,
        "description": "в фоне",
        "run_in_background": True,
    }


def _bg_task(prompt: str, subagent: str = "explore") -> AssistantTurn:
    return tool_turn("task", _bg_args(prompt, subagent))


def _join_job_thread(job_id: str) -> None:
    for thread in threading.enumerate():
        if thread.name == f"gigacrew-{job_id}":
            thread.join(10)


# ------------------------------- реестр задач ------------------------------- #
def test_pool_runs_limited_jobs_and_queues_the_rest() -> None:
    jobs = BackgroundJobs({jobs_module.SUBAGENT_POOL: PoolLimits(running=2, queued=1)})
    first, second, third = FakeJob("1"), FakeJob("2"), FakeJob("3")
    assert [jobs.start(job) for job in (first, second, third)] == ["bg1", "bg2", "bg3"]
    assert first.started
    assert second.started
    assert not third.started
    assert third.state is JobState.QUEUED
    assert third.elapsed == 0
    with pytest.raises(ToolError, match="работают 2 из 2, ждут в очереди 1 из 1"):
        jobs.start(FakeJob("4"))
    first.done()
    assert _state(third) is JobState.RUNNING  # место освободилось — из очереди
    assert [job.id for job in jobs.active()] == ["bg2", "bg3"]


def test_finished_job_is_delivered_once_and_quiet_one_does_not_wake() -> None:
    jobs = BackgroundJobs()
    loud, quiet = FakeJob(), FakeJob()
    jobs.start(loud)
    jobs.start(quiet)
    woken: list[str] = []
    jobs.subscribe(lambda: woken.append("x"))
    assert jobs.stop(quiet, "пользователь", quiet=True)
    assert jobs.pending()
    assert not jobs.pending(waking=True)
    loud.done()
    assert jobs.pending(waking=True)
    assert len(woken) == 2
    assert {job.id for job in jobs.take_finished()} == {loud.id, quiet.id}
    assert not jobs.pending()
    assert jobs.take_finished() == []


def test_stop_removes_queued_job_without_starting_it() -> None:
    jobs = BackgroundJobs({jobs_module.SUBAGENT_POOL: PoolLimits(running=1, queued=1)})
    running, waiting = FakeJob(), FakeJob()
    jobs.start(running)
    jobs.start(waiting)
    assert jobs.stop(waiting, "не нужна")
    assert waiting.state is JobState.STOPPED
    assert not waiting.started
    running.done()
    assert not waiting.started  # снятая с очереди задача не запускается


def test_stuck_job_gets_forced_report_after_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jobs_module, "STOP_GRACE", 0.05)
    jobs = BackgroundJobs()
    stuck = FakeJob(obeys=False)
    jobs.start(stuck)
    assert jobs.stop(stuck, "надоела")
    assert stuck.cancelled
    assert stuck.report == JobReport(JobState.STOPPED, "forced: надоела")


def test_hard_limit_stops_job_that_never_finishes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jobs_module, "HARD_STOP_GRACE", 0)
    monkeypatch.setattr(jobs_module, "STOP_GRACE", 0.05)
    jobs = BackgroundJobs()
    stuck = FakeJob(timeout=0.05, obeys=False)
    jobs.start(stuck)
    assert stuck.wait(5)
    assert stuck.state is JobState.TIMED_OUT
    assert jobs.pending(waking=True)  # итог «превышен лимит» будит агента


def test_queued_job_limit_counts_from_its_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jobs_module, "HARD_STOP_GRACE", 0)
    jobs = BackgroundJobs({jobs_module.SUBAGENT_POOL: PoolLimits(running=1, queued=1)})
    running, waiting = FakeJob(), FakeJob(timeout=0.2)
    jobs.start(running)
    jobs.start(waiting)
    time.sleep(0.3)  # в очереди дольше своего предела — но предел ещё не начался
    assert waiting.state is JobState.QUEUED
    running.done()
    assert _state(waiting) is JobState.RUNNING
    assert waiting.wait(5)
    assert _state(waiting) is JobState.TIMED_OUT


def test_stop_all_forget_clears_chat_jobs() -> None:
    jobs = BackgroundJobs()
    job = FakeJob()
    jobs.start(job)
    assert [stopped.id for stopped in jobs.stop_all("новый чат", forget=True)] == ["bg1"]
    assert job.state is JobState.STOPPED
    assert jobs.jobs() == []
    assert not jobs.pending()


def test_output_log_reads_from_position_and_marks_dropped() -> None:
    log = OutputLog(limit=5)
    log.add("abc")
    assert log.read() == ("abc", 3)
    log.add("defgh")
    log.add("ijklm")
    assert log.read(3) == ("...(пропущено 5 символов)...\nijklm", 13)
    assert log.read(13) == ("", 13)


# ------------------------------ фоновые команды ----------------------------- #
@posix_only
def test_background_command_reports_exit_code_and_output(tmp_path: Path) -> None:
    jobs = BackgroundJobs()
    params = RunShellParams(command="echo one; echo two; exit 3", run_in_background=True)
    started = RunShellTool().run(params, _ctx(tmp_path, jobs))
    assert "bg1" in started.content
    job = _job(jobs)
    assert job.wait(10)
    report = _report(job)
    assert report.state is JobState.DONE
    assert report.content.startswith("exit code: 3")
    assert "one\ntwo" in report.content


@posix_only
def test_background_output_returns_new_output_and_stop_kills_command(tmp_path: Path) -> None:
    jobs = BackgroundJobs()
    ctx = _ctx(tmp_path, jobs)
    RunShellTool().run(RunShellParams(command="echo first; sleep 30", run_in_background=True), ctx)
    tool = BackgroundOutputTool()
    first = ToolResult(content="")
    for _ in range(100):
        first = tool.run(BackgroundOutputParams(job_id="bg1"), ctx)
        if "first" in first.content:
            break
        time.sleep(0.05)
    assert "running" in first.content
    assert "first" in first.content
    assert "No new output" in tool.run(BackgroundOutputParams(job_id="bg1"), ctx).content
    started = time.monotonic()
    stopped = BackgroundStopTool().run(BackgroundStopParams(job_id="bg1"), ctx)
    assert time.monotonic() - started < 5  # команда убита, а не дождалась sleep 30
    assert "stopped" in stopped.content
    assert _job(jobs).state is JobState.STOPPED
    assert not jobs.pending()  # итог отдан вызову — второй раз не доставляется


@posix_only
def test_background_command_is_killed_by_its_timeout(tmp_path: Path) -> None:
    jobs = BackgroundJobs()
    started = time.monotonic()
    params = RunShellParams(command="sleep 30", timeout=1, run_in_background=True)
    RunShellTool().run(params, _ctx(tmp_path, jobs))
    job = _job(jobs)
    assert job.wait(10)
    assert job.state is JobState.TIMED_OUT
    assert time.monotonic() - started < 10


def test_run_in_background_needs_background_jobs(tmp_path: Path) -> None:
    params = RunShellParams(command="true", run_in_background=True)
    with pytest.raises(ToolError, match="фоновые команды здесь недоступны"):
        RunShellTool().run(params, _ctx(tmp_path))


def test_background_output_lists_jobs_and_rejects_unknown_id(tmp_path: Path) -> None:
    jobs = BackgroundJobs()
    ctx = _ctx(tmp_path, jobs)
    tool = BackgroundOutputTool()
    assert "No background jobs" in tool.run(BackgroundOutputParams(), ctx).content
    job = FakeJob("проверка")
    jobs.start(job)
    assert "bg1 (fake: проверка)" in tool.run(BackgroundOutputParams(), ctx).content
    with pytest.raises(ToolError, match="фоновой задачи 'bg7' нет"):
        tool.run(BackgroundOutputParams(job_id="bg7"), ctx)
    job.done("итог проверки")
    answer = tool.run(BackgroundOutputParams(job_id="1", wait=5), ctx)
    assert "итог проверки" in answer.content
    assert not jobs.pending()  # итог отдан вызову


# ---------------------------------- агент ----------------------------------- #
def test_finished_job_rides_along_with_next_user_request(tmp_path: Path) -> None:
    provider = ScriptedProvider([text_turn("учёл")])
    events = BackgroundEvents()
    agent = _agent(provider, tmp_path, events=events)
    job = FakeJob("тесты")
    _jobs(agent).start(job)
    job.done("all green")
    agent.run_turn("что там?")
    request = _sent(provider, 0)
    assert BACKGROUND_HEADER in request.content
    assert "all green" in request.content
    assert request_text(request.content) == "что там?"
    assert recent_requests(agent.conversation.messages) == ["что там?"]  # рецензенту
    assert [seen.id for seen in events.background] == ["bg1"]
    assert not _jobs(agent).pending()


def test_job_finishing_mid_turn_is_appended_to_tool_result(tmp_path: Path) -> None:
    job = FakeJob("сборка")

    def build() -> ToolResult:
        job.done("собрано")  # фоновая задача завершилась, пока шёл другой вызов
        return ToolResult(content="ok")

    fake = FakeTool(run=build)
    registry = build_default_registry()
    registry.register(fake)
    provider = ScriptedProvider([tool_turn("fake_write", {}), text_turn("готово")])
    agent = _agent(provider, tmp_path, registry=registry)
    _jobs(agent).start(job)
    assert agent.run_turn("собери") == "готово"
    first, second = provider.requests
    assert "# Background jobs not finished yet" in first["messages"][-1].content
    tool_message = second["messages"][-1]
    assert tool_message.role == "function"
    assert tool_message.content.startswith("ok")
    assert "собрано" in tool_message.content
    # Напоминание о незавершённых — только в запросе, не в истории.
    assert all("not finished yet" not in message.content for message in agent.conversation.messages)


def test_wake_without_finished_jobs_does_nothing(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    agent = _agent(provider, tmp_path)
    _jobs(agent).start(FakeJob())
    assert agent.wake() is None
    assert provider.calls == 0


def test_reset_stops_and_forgets_background_jobs(tmp_path: Path) -> None:
    events = RecordingEvents()
    agent = _agent(ScriptedProvider(), tmp_path, events=events)
    job = FakeJob()
    _jobs(agent).start(job)
    agent.reset()
    assert job.state is JobState.STOPPED
    assert _jobs(agent).jobs() == []
    assert any("bg1" in str(notice) for notice in events.notices)


def test_background_subagent_reports_back_through_wake(tmp_path: Path) -> None:
    release = threading.Event()
    provider = BackgroundProvider(
        [_bg_task("где foo?"), text_turn("жду отчёт"), text_turn("учёл отчёт")],
        background=[text_turn("foo — в a.py:3")],
        release=release,
    )
    events = BackgroundEvents()
    agent = _agent(provider, tmp_path, events=events)
    assert agent.run_turn("исследуй") == "жду отчёт"
    task_result = agent.conversation.messages[2]
    assert task_result.role == "function"
    assert "started in the background as job bg1" in (task_result.content)
    release.set()
    job = _job(_jobs(agent))
    assert job.wait(10)
    assert job.state is JobState.DONE
    assert agent.wake() == "учёл отчёт"
    wake_message = _sent(provider, -1)
    assert wake_message.meta
    assert BACKGROUND_HEADER in wake_message.content
    assert "foo — в a.py:3" in wake_message.content
    assert [seen.id for seen in events.background] == ["bg1"]
    assert agent.wake() is None  # итог доставлен один раз
    # Суб-агент видит свою задачу, а не диалог основного агента.
    assert provider.background_requests[0][-1].content == "где foo?"


def test_background_subagent_cannot_ask_user(tmp_path: Path) -> None:
    fake = FakeTool()
    registry = build_default_registry()
    registry.register(fake)
    provider = BackgroundProvider(
        [_bg_task("почини", "general"), text_turn("ок")],
        background=[tool_turn("fake_write", {"path": "a.py"}), text_turn("не смог")],
    )
    events = RecordingEvents()  # на вопрос ответил бы «да»
    agent = _agent(provider, tmp_path, events=events, registry=registry, auto_approve=False)
    agent.run_turn("почини в фоне")
    job = _job(_jobs(agent))
    assert job.wait(10)
    assert fake.runs == 0
    assert events.confirms == []
    assert "Not run, needs the user's confirmation: fake_write a.py" in _report(job).content


def test_stopped_background_subagent_runs_no_more_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(jobs_module, "STOP_GRACE", 0.05)
    fake = FakeTool()
    registry = build_default_registry()
    registry.register(fake)
    release = threading.Event()
    provider = BackgroundProvider(
        [
            _bg_task("почини", "general"),
            tool_turn("background_stop", {"job_id": "bg1"}),
            text_turn("остановил"),
        ],
        background=[tool_turn("fake_write", {})],
        release=release,
    )
    agent = _agent(provider, tmp_path, registry=registry)
    assert agent.run_turn("x") == "остановил"
    job = _job(_jobs(agent))
    assert job.state is JobState.STOPPED  # итог записан без зависшего суб-агента
    release.set()  # суб-агент дождался ответа модели, но вызов уже не выполнит
    _join_job_thread("bg1")
    assert fake.runs == 0
    assert "stopped before it finished" in _report(job).content
    assert not _jobs(agent).pending()  # итог получил вызов background_stop


def test_background_subagents_wait_in_queue(tmp_path: Path) -> None:
    release = threading.Event()
    provider = BackgroundProvider(
        [
            tools_turn([("task", _bg_args(question)) for question in ("один", "два", "три")]),
            text_turn("жду"),
        ],
        release=release,
    )
    agent = _agent(provider, tmp_path, background_subagents=2, background_queue=1)
    agent.run_turn("три вопроса")
    states = [job.state for job in _jobs(agent).jobs()]
    assert states == [JobState.RUNNING, JobState.RUNNING, JobState.QUEUED]
    third_result = agent.conversation.messages[4].content
    assert "is queued as job bg3" in third_result
    release.set()
    for job in _jobs(agent).jobs():
        assert job.wait(10)
    assert all(job.state is JobState.DONE for job in _jobs(agent).jobs())


# ----------------------------------- REPL ----------------------------------- #
def _repl_env(
    tmp_path: Path,
    provider: ScriptedProvider,
    registry: ToolRegistry | None = None,
) -> tuple[loop.Agent, Console]:
    cfg = Config(access_key="x", project_root=tmp_path, stream=False, auto_approve=True)
    ui = Console(no_color=True)
    return loop.Agent(provider, registry or build_default_registry(), cfg, ui), ui


def _reader(*answers: str | BaseException) -> Callable[..., str]:
    queue = list(answers)

    def read(default: str = "") -> str:
        answer = queue.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    return read


def _run_repl(agent: loop.Agent, ui: Console, read: Callable[..., str]) -> int:
    commands: CommandRegistry = default_commands()
    return repl.run_repl(agent, ui, commands, read_input=read, index_on_start=False)


def test_repl_wakes_agent_when_background_job_finishes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    provider = ScriptedProvider([text_turn("тесты прошли — готово")])
    agent, ui = _repl_env(tmp_path, provider)
    job = FakeJob("pytest")
    _jobs(agent).start(job)
    job.done("exit code: 0")
    assert _run_repl(agent, ui, _reader("/exit")) == 0
    assert provider.calls == 1
    message = _sent(provider, 0)
    assert message.meta
    assert "exit code: 0" in message.content
    out = capsys.readouterr().out
    assert "фоновая задача bg1 (fake): готово" in out
    assert "тесты прошли" in out


def test_repl_wake_from_prompt_keeps_draft(tmp_path: Path) -> None:
    provider = ScriptedProvider([text_turn("учёл")])
    agent, ui = _repl_env(tmp_path, provider)
    job = FakeJob()
    _jobs(agent).start(job)
    defaults: list[str] = []

    def read(default: str = "") -> str:
        defaults.append(default)
        if len(defaults) == 1:
            job.done("готово")  # задача завершилась, пока открыта строка ввода
            raise BackgroundWakeError("недописанный запрос")
        return "/exit"

    assert _run_repl(agent, ui, read) == 0
    assert defaults == ["", "недописанный запрос"]
    assert provider.calls == 1


def test_repl_limits_wakes_in_a_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(repl, "MAX_AUTO_WAKES", 1)
    first, second = FakeJob("1"), FakeJob("2")

    def finish_second(requests: int) -> None:
        if requests == 0:
            second.done("второй итог")  # завершилась во время хода по готовности первой

    provider = HookedProvider([text_turn("учёл первый"), text_turn("учёл второй")], finish_second)
    agent, ui = _repl_env(tmp_path, provider)
    _jobs(agent).start(first)
    _jobs(agent).start(second)
    first.done("первый итог")
    assert _run_repl(agent, ui, _reader("а второй?", "/exit")) == 0
    assert provider.calls == 2  # второй ход — по сообщению пользователя, а не сам
    user_request = _sent(provider, 1)
    assert request_text(user_request.content) == "а второй?"
    assert "второй итог" in user_request.content
    assert "получит с вашим следующим сообщением" in capsys.readouterr().out


def test_repl_exit_asks_before_stopping_running_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent, ui = _repl_env(tmp_path, ScriptedProvider())
    job = FakeJob()
    _jobs(agent).start(job)
    answers = [False, True]
    questions: list[str] = []

    def ask(question: str, **_: Any) -> bool:
        if "фоновые задачи" not in question:
            return False
        questions.append(question)
        return answers.pop(0)

    monkeypatch.setattr(ui, "interactive", lambda: True)
    monkeypatch.setattr(ui, "ask", ask)
    assert _run_repl(agent, ui, _reader(EOFError(), "/exit")) == 0
    assert len(questions) == 2
    assert "bg1" in questions[0]
    assert job.state is JobState.STOPPED  # остановлена при выходе


def test_repl_without_terminal_waits_for_jobs_before_exit(tmp_path: Path) -> None:
    provider = ScriptedProvider([text_turn("итог учтён")])
    agent, ui = _repl_env(tmp_path, provider)
    job = FakeJob()
    _jobs(agent).start(job)
    threading.Timer(0.2, job.done, args=("готово",)).start()
    assert _run_repl(agent, ui, _reader(EOFError())) == 0
    assert provider.calls == 1
    assert "готово" in _sent(provider, 0).content


def test_oneshot_waits_for_background_job_started_in_turn(tmp_path: Path) -> None:
    job = FakeJob("сборка")

    def finish_job(requests: int) -> None:
        if requests == 1:
            job.done("собрано")  # завершилась, когда ход уже не доставит итог сам

    provider = HookedProvider(
        [tool_turn("fake_write", {}), text_turn("жду сборку"), text_turn("сборка готова")],
        finish_job,
    )
    registry = build_default_registry()
    agent, ui = _repl_env(tmp_path, provider, registry)
    registry.register(FakeTool(run=lambda: ToolResult(content=_jobs(agent).start(job))))
    assert run_oneshot(agent, ui, "собери") == EXIT_OK
    assert provider.calls == 3
    assert "собрано" in _sent(provider, 2).content


def test_prompt_reader_wakes_open_prompt_and_returns_draft(tmp_path: Path) -> None:
    with create_pipe_input() as pipe:
        session = create_prompt_session(
            commands=default_commands(),
            workspace=Workspace(tmp_path),
            status=lambda: StatusInfo(model="m"),
            input=pipe,
            output=DummyOutput(),
        )
        reader = PromptReader(session)
        due = threading.Event()
        reader.wake_due = due.is_set

        def job_finished() -> None:
            due.set()
            deadline = time.monotonic() + 5
            while not session.app.is_running and time.monotonic() < deadline:
                time.sleep(0.01)
            reader.wake()

        threading.Timer(0.1, job_finished).start()
        with pytest.raises(BackgroundWakeError) as caught:
            reader(default="черновик")
        assert caught.value.draft == "черновик"
        due.clear()
        pipe.send_text("привет\r")
        assert reader() == "привет"


def test_status_line_shows_unfinished_background_jobs(tmp_path: Path) -> None:
    agent, _ = _repl_env(tmp_path, ScriptedProvider())
    _jobs(agent).start(FakeJob())
    assert repl.status_of(agent).background == 1
