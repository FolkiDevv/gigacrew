"""Выполнение shell-команд с захватом вывода."""

from __future__ import annotations

import contextlib
import os
import threading
from pathlib import Path
from typing import override

from pydantic import BaseModel, Field, PrivateAttr

from gigacrew.permissions import ToolKind
from gigacrew.safety import MAX_COMMAND, CommandLevel, Part, Verdict, classify
from gigacrew.security import (
    RiskLevel,
    classify_shell_command,
    subprocess_env,
)
from gigacrew.tools.base import Assessment, Display, Tool, ToolContext, ToolError, ToolResult
from gigacrew.tools.grep_search import GrepSearch, as_grep_search
from gigacrew.tools.jobs import SHELL_POOL, BackgroundJobs, Job, JobReport, JobState
from gigacrew.tools.process import (
    BackgroundProcess,
    ProcessResult,
    quote_reminder_tags,
    run_process,
    truncate_middle,
)
from gigacrew.tools.search import search_paths

_MAX_OUTPUT = 30_000
MIN_TIMEOUT = 1
MAX_TIMEOUT = 600
_DEFAULT_TIMEOUT = 120
# Предел фоновой команды (run_in_background): сборка, долгие тесты.
MAX_BACKGROUND_TIMEOUT = 3600
# Сколько вывода фоновой команды модель получает в итоге (остальное — пропуск в середине).
_REPORT_OUTPUT = 8_000
# Сколько последних символов вывода смотреть, чтобы показать его последнюю строку.
_ACTIVITY_CHARS = 200
# Коды оболочки: команда не запускается (126) или не найдена (127).
_SHELL_FAILURES = (126, 127)
# Сколько символов вызова показывать в UI (describe) и в кратком итоге команды.
_DESCRIBE_LIMIT = 70
_SUMMARY_COMMAND_LIMIT = 60
# Пометка для модели к результату grep -r, выполненного встроенным поиском.
_GREP_SEARCH_NOTE = (
    "[grep -r здесь не выполнялся: по этому каталогу его не проверить (файлы с секретами, "
    "ссылки за пределы проекта или слишком много файлов). Вместо него выполнен встроенный "
    "поиск — {call}: файлы с секретами, из .gitignore, служебные каталоги и бинарные файлы "
    "пропущены. Для поиска по содержимому сразу вызывайте search_content.]"
)


# Без докстринга: pydantic отдал бы его в JSON-схему параметров, которая уходит модели.
class RunShellParams(BaseModel):  # noqa: D101
    command: str = Field(description="Command to run with /bin/sh")
    timeout: int = Field(
        default=_DEFAULT_TIMEOUT,
        description=(
            f"Timeout in seconds ({MIN_TIMEOUT}–{MAX_TIMEOUT}; with run_in_background up to "
            f"{MAX_BACKGROUND_TIMEOUT})"
        ),
    )
    cwd: str = Field(
        default="",
        description=(
            "Working directory relative to the project root (default: the root) or an "
            "absolute path in an additional directory. Prefer it over `cd dir && …`"
        ),
    )
    description: str = Field(
        default="",
        description=(
            "What the command is for, in a few words in the user's language (shown when "
            "the user is asked to confirm it)"
        ),
    )
    run_in_background: bool = Field(
        default=False,
        description=(
            "Run without waiting: returns a job id at once, the command keeps running (up "
            "to timeout) while you continue; its exit code and output are delivered to you "
            "automatically when it finishes. For long builds and test runs when you have "
            "other work meanwhile — not when your next step needs the result"
        ),
    )
    _planned: GrepSearch | None = PrivateAttr(None)

    @property
    def planned(self) -> GrepSearch | None:
        """Встроенный поиск, которым оценка (assess) решила выполнить вызов.

        Разрешённый поиск run выполняет поиском, даже если к запуску перевод команды уже
        не удаётся. None — оценки не было или она решила выполнить команду: тогда run
        проверяет команду заново (дерево поиска могло измениться).
        """
        return self._planned

    @planned.setter
    def planned(self, search: GrepSearch | None) -> None:
        # Приватный атрибут pydantic: в классе — его объявление, а не значение.
        self._planned = search  # noqa: WPS601


def command_verdict(params: RunShellParams, ctx: ToolContext) -> Verdict:
    """Вердикт классификатора для команды в её рабочем каталоге."""
    cwd = _relative_cwd(params, ctx)
    return classify(
        params.command,
        root=ctx.root,
        cwd=cwd,
        environ=subprocess_env(),
        extra=ctx.workspace.extra_dirs,
    )


def _relative_cwd(params: RunShellParams, ctx: ToolContext) -> str:
    if not params.cwd.strip() or params.cwd.strip() == ".":
        return "."
    target = ctx.resolve(params.cwd.strip())
    if not target.is_dir():
        raise ToolError(f"Каталог '{params.cwd}' не найден в проекте.")
    rel = os.path.relpath(target, ctx.root)
    return "." if rel == "." else rel.replace(os.sep, "/")


def _grep_substitute(
    params: RunShellParams,
    ctx: ToolContext,
    verdict: Verdict | None = None,
) -> GrepSearch | None:
    """Рекурсивный grep, который run_shell выполнит встроенным поиском; None — команду.

    Подменяется только grep, который классификатор не проверил (в дереве секреты, ссылки
    наружу или слишком много файлов): проверенный выполняется как есть. ``verdict`` —
    уже готовый вердикт команды (иначе он вычисляется, если команда — такой grep).
    """
    if os.name != "posix":  # команды выполняет не /bin/sh: разбор bash к ним не относится
        return None
    search = as_grep_search(params.command, ctx, _relative_cwd(params, ctx))
    if search is None:
        return None
    if verdict is None:
        verdict = command_verdict(params, ctx)
    return search if verdict.level is None else None


def _search_verdict(verdict: Verdict, search: GrepSearch) -> Verdict:
    """Вердикт grep, выполняемого встроенным поиском: чтение без секретов."""
    reason = f"выполняется встроенным поиском: {search.call}"
    part = Part(CommandLevel.READ, reason, text=verdict.command)
    return Verdict(
        CommandLevel.READ,
        reason,
        verdict.cwd,
        verdict.command,
        parts=(part,),
        extra_dirs=verdict.extra_dirs,
    )


def _search_result(command: str, search: GrepSearch, ctx: ToolContext) -> ToolResult:
    """Результат grep -r, выполненного встроенным поиском, с пометкой для модели."""
    # Пути проверены при оценке; ссылка, подменённая с тех пор, не уведёт поиск наружу.
    bases = tuple(ctx.resolve(base) for base in search.bases)
    found = search_paths(
        ctx,
        bases,
        search.regex,
        limit=search.limit,
        search_filter=search.search_filter,
        files_only=search.files_only,
    )
    note = _GREP_SEARCH_NOTE.format(call=search.call)
    output = truncate_middle(found.content, _MAX_OUTPUT)
    command_head = command[:_SUMMARY_COMMAND_LIMIT]
    return ToolResult(
        content=f"{note}\n{output}",
        summary=f"$ {command_head} → встроенный поиск, {found.summary}",
        display=Display(output, title=f"$ {command_head}"),
    )


def _risk(verdict: Verdict, command: str, *, classify_commands: bool) -> RiskLevel:
    level = verdict.level
    if level is None:
        if len(command) > MAX_COMMAND:
            # Классификатор такую команду не разбирал вовсе (даже невидимые символы не
            # искал), а регэксп в ней найдёт не всё опасное: решает пользователь, -y не
            # ручается.
            return RiskLevel.DANGEROUS
        return classify_shell_command(command)  # «не знаю» — как раньше: регэксп опасного
    if level is CommandLevel.READ:
        return RiskLevel.SAFE if classify_commands else RiskLevel.WRITE
    if level is CommandLevel.DANGER:
        return RiskLevel.DANGEROUS
    return RiskLevel.WRITE


def _combined_output(res: ProcessResult) -> str:
    """Вывод команды одним текстом: stdout, затем stderr под меткой; середина обрезается."""
    combined = res.stdout
    if res.stderr:
        separator = "\n" if combined else ""
        combined = f"{combined}{separator}[stderr]\n{res.stderr}"
    return truncate_middle(combined, _MAX_OUTPUT)


def _notes(res: ProcessResult | BackgroundProcess, timeout: int) -> list[str]:
    """Пояснения к итогу команды: остановленные фоновые процессы, таймаут."""
    notes: list[str] = []
    if res.killed_background:
        notes.append(
            "фоновые процессы, оставшиеся после команды, остановлены "
            "(долгоживущие процессы через run_shell не поддерживаются)",
        )
    if res.timed_out:
        notes.append(f"команда превысила таймаут ({timeout}с) и была остановлена")
    if res.escaped_output:
        # Потомок ушёл из группы команды (setsid) и держит вывод: killpg его не достал.
        notes.append(
            "фоновый процесс, ушедший из группы команды (setsid), остановить не удалось: "
            "он продолжает работать, его вывод сюда больше не попадёт",
        )
    return notes


def _exit_state(process: BackgroundProcess) -> tuple[JobState, str]:
    """Состояние итога команды, завершившейся без запроса остановки, и его причина."""
    if process.error:
        return JobState.FAILED, f"supervising the command failed ({process.error}); it was killed"
    if process.timed_out:
        return JobState.TIMED_OUT, ""
    return JobState.DONE, ""


class ShellJob(Job):
    """Команда ``run_shell`` в фоне: вывод копится, модель читает его частями."""

    pool = SHELL_POOL

    def __init__(self, command: str, *, cwd: Path, timeout: int) -> None:
        """Команда ``command`` в каталоге ``cwd`` с пределом ``timeout`` секунд."""
        super().__init__("shell", command[:_SUMMARY_COMMAND_LIMIT], timeout=timeout)
        self._command = command
        self._cwd = cwd
        self._process: BackgroundProcess | None = None
        self._read_lock = threading.Lock()
        self._read_to = 0  # до какой позиции вывод уже отдан модели

    @override
    def start(self) -> None:
        # Оболочка — суть инструмента: команду до запуска проверили классификатор и права.
        self._process = BackgroundProcess(  # noqa: S604
            self._command,
            cwd=self._cwd,
            timeout=self.timeout,
            shell=True,
            on_exit=self._exited,
        )

    @override
    def cancel(self) -> None:
        if self._process is not None:
            self._process.kill()

    @override
    def stopped_report(self, state: JobState, reason: str) -> JobReport:
        return self._build_report(state, reason)

    @override
    def progress(self) -> str:
        output = self._unread(_MAX_OUTPUT)
        return f"New output since the last check:\n{output}" if output else "No new output."

    @override
    def activity(self) -> str:
        if self._process is None:
            return ""
        output = self._process.output
        tail, _ = output.read(max(output.total - _ACTIVITY_CHARS, 0))
        lines = [line.strip() for line in tail.splitlines() if line.strip()]
        return lines[-1] if lines else ""

    def _exited(self, process: BackgroundProcess) -> None:
        """Команда завершилась (вызывает надзиратель процесса, из своего потока).

        Процесс — аргументом: быстрая команда завершается раньше, чем :meth:`start`
        сохранит его, а итог должен содержать её код и вывод.
        """
        self._process = process
        self.finish(self._build_report(*(self.stop_request or _exit_state(process))))

    def _unread(self, limit: int) -> str:
        """Вывод, который модель ещё не видела (середина длинного — пропущена)."""
        if self._process is None:
            return ""
        with self._read_lock:
            text, end = self._process.output.read(self._read_to)
            self._read_to = end
        return truncate_middle(quote_reminder_tags(text), limit, head=limit // 4)

    def _build_report(self, state: JobState, reason: str) -> JobReport:
        process = self._process
        code = None if process is None else process.returncode
        if state is JobState.DONE:
            head = f"exit code: {code}"
            summary = f"$ {self.title} → код {code}"
        elif state is JobState.TIMED_OUT:
            head = f"exit code: (timeout: stopped after {self.timeout} s)"
            summary = f"$ {self.title} → таймаут"
        elif state is JobState.FAILED:
            head = f"failed: {reason}"
            summary = f"$ {self.title} → сбой"
        else:
            head = f"stopped before it finished: {reason}"
            summary = f"$ {self.title} → остановлена"
        notes = [] if process is None else _notes(process, int(self.timeout))
        note_text = "".join(f"\n[{note}]" for note in notes)
        output = self._unread(_REPORT_OUTPUT) or "(no new output)"
        return JobReport(state, f"{head}{note_text}\n{output}", summary)


class RunShellTool(Tool[RunShellParams]):
    """Инструмент run_shell: команда оболочки в каталоге проекта."""

    name = "run_shell"
    description = (
        "Runs a shell command and returns stdout/stderr and the exit code. Use it to build, "
        "run tests and utilities — not to read or search files (read_file, search_content "
        "and the navigation tools are faster and safer). Search file contents with "
        "search_content, not grep -r or rg: a recursive grep over a directory with secrets "
        "(.env) or a virtualenv cannot be verified, so it is run as search_content instead. "
        "Commands that only read the project (ls, cat, wc, git status/log/diff, Python "
        "one-liners that only read) run without asking the user; others are checked by a "
        "reviewer model or confirmed by the user. There is no keyboard "
        "input (stdin is empty) — use non-interactive flags. Network commands (curl/wget) "
        "are forbidden by the security policy. Put temporary files in .gigacode/tmp/ (it "
        "exists), not /tmp: writing outside the project always needs the user's "
        "confirmation; output saved there with > or tee does not change how the command "
        "is checked."
    )
    # Модель параметров (класс pydantic), а не константа: имя — часть API инструментов.
    Params = RunShellParams  # noqa: WPS115
    kind = ToolKind.COMMAND

    @override
    def assess(self, params: RunShellParams, ctx: ToolContext) -> Assessment:
        """Оценка команды классификатором: риск, причина, вид вызова, вердикт."""
        verdict = command_verdict(params, ctx)
        search = _grep_substitute(params, ctx, verdict)
        params.planned = search
        if search is not None:
            verdict = _search_verdict(verdict, search)
        risk = _risk(verdict, params.command, classify_commands=ctx.classify_commands)
        reason = "" if verdict.level is CommandLevel.READ else verdict.reason
        if verdict.level is None:
            reason = f"не удалось проверить: {verdict.reason}" if verdict.reason else ""
        kind = ToolKind.EDIT if verdict.edits_only else None
        return Assessment(
            risk,
            reason=reason,
            kind=kind,
            verdict=verdict,
            purpose=params.description.strip(),
        )

    @override
    def risk(self, params: RunShellParams, ctx: ToolContext) -> RiskLevel:
        """Уровень риска команды (из :meth:`assess`)."""
        return self.assess(params, ctx).risk

    @override
    def describe(self, params: RunShellParams) -> str:
        """Команда (с рабочим каталогом, если он не корень) для UI."""
        cwd = params.cwd.strip()
        prefix = "" if cwd in {"", "."} else f"[{cwd}] "
        if params.run_in_background:
            prefix = f"(в фоне) {prefix}"
        return f"{prefix}{params.command}"[:_DESCRIBE_LIMIT]

    @override
    def paths(self, params: RunShellParams) -> tuple[str, ...]:
        """Рабочий каталог команды, если он задан и не корень проекта."""
        cwd = params.cwd.strip()
        return (cwd,) if cwd and cwd != "." else ()

    @override
    def preview(self, params: RunShellParams, ctx: ToolContext) -> Display:
        """Команда (и её рабочий каталог) для вопроса пользователю."""
        cwd = _relative_cwd(params, ctx)
        title = "команда" if cwd == "." else f"команда · {cwd}"
        return Display(f"$ {params.command}", title=title)

    @override
    def run(self, params: RunShellParams, ctx: ToolContext) -> ToolResult:
        """Выполняет команду через /bin/sh и возвращает код возврата и вывод.

        Непроверенный рекурсивный grep выполняется встроенным поиском
        (:func:`_grep_substitute`). Поиск, разрешённый при оценке
        (:attr:`RunShellParams.planned`), не превращается в команду. Команда проверяется
        заново: если с оценки в дереве появились секреты, вместо неё выполняется поиск.
        С ``run_in_background`` команда запускается фоновой задачей и вызов сразу
        возвращает её id; встроенный поиск быстрый — он выполняется сразу.
        """
        if params.run_in_background and ctx.background is None:
            raise ToolError(
                "фоновые команды здесь недоступны — запусти команду без run_in_background.",
            )
        search = params.planned or _grep_substitute(params, ctx)
        if search is not None:
            return _search_result(params.command, search, ctx)
        return _execute(params, ctx)


def _execute(params: RunShellParams, ctx: ToolContext) -> ToolResult:
    """Выполняет команду через /bin/sh: код возврата и вывод для модели (или фоновая задача)."""
    cwd = (ctx.root / _relative_cwd(params, ctx)).resolve()
    # Папка временных файлов модели есть к запуску команды: не нужен лишний шаг с mkdir.
    # Не вышло (симлинк, нет прав) — команда всё равно выполняется.
    with contextlib.suppress(OSError):
        ctx.workspace.ensure_temp_dir()
    if params.run_in_background and ctx.background is not None:
        return _start_background(params, ctx.background, cwd)
    timeout = min(max(params.timeout, MIN_TIMEOUT), MAX_TIMEOUT)
    # Оболочка — суть инструмента: команду до запуска проверяют классификатор и права.
    res = run_process(params.command, cwd=cwd, timeout=timeout, shell=True)  # noqa: S604

    combined = _combined_output(res)
    command_head = params.command[:_SUMMARY_COMMAND_LIMIT]
    if res.timed_out:
        head = "exit code: (таймаут)"
        summary = "таймаут команды"
    else:
        head = f"exit code: {res.returncode}"
        summary = f"$ {command_head} → код {res.returncode}"
    if res.escaped_output:
        summary = f"{summary} · фоновый процесс остановить не удалось"
    note_text = "".join(f"\n[{note}]" for note in _notes(res, timeout))
    output = combined or "(пустой вывод)"
    ok = res.returncode == 0 and not res.timed_out
    return ToolResult(
        content=f"{head}{note_text}\n{output}",
        ok=ok,
        summary=summary,
        # Ненулевой код (grep без совпадений, упавшие тесты) — результат, а не
        # застревание; таймаут и «команда не найдена/не запускается» — сбой.
        soft=not ok and not res.timed_out and res.returncode not in _SHELL_FAILURES,
        display=Display(combined, title=f"$ {command_head}") if combined else None,
        exit_code=None if res.timed_out else res.returncode,
    )


def _start_background(params: RunShellParams, jobs: BackgroundJobs, cwd: Path) -> ToolResult:
    """Запустить команду фоновой задачей; результат вызова — её id."""
    timeout = min(max(params.timeout, MIN_TIMEOUT), MAX_BACKGROUND_TIMEOUT)
    job = ShellJob(params.command, cwd=cwd, timeout=timeout)
    job_id = jobs.start(job)
    return ToolResult(
        content=(
            f"Started in the background as job {job_id} (timeout {timeout} s). Continue with "
            "other work: its exit code and output will be delivered to you automatically when "
            "it finishes — do not poll it. If you have nothing else to do until then, end your "
            "turn: you will be woken with the result. background_output shows its output so "
            "far, background_stop stops it."
        ),
        summary=f"в фоне {job_id}: $ {job.title}",
    )
