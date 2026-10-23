"""Выполнение shell-команд с захватом вывода."""

from __future__ import annotations

import os
from typing import override

from pydantic import BaseModel, Field

from gigacrew.permissions import ToolKind
from gigacrew.safety import MAX_COMMAND, CommandLevel, Verdict, classify
from gigacrew.security import (
    RiskLevel,
    classify_shell_command,
    resolve_in_root,
    subprocess_env,
)
from gigacrew.tools.base import Assessment, Display, Tool, ToolContext, ToolError, ToolResult
from gigacrew.tools.process import ProcessResult, run_process, truncate_middle

_MAX_OUTPUT = 30_000
MIN_TIMEOUT = 1
MAX_TIMEOUT = 600
_DEFAULT_TIMEOUT = 120
# Коды оболочки: команда не запускается (126) или не найдена (127).
_SHELL_FAILURES = (126, 127)
# Сколько символов вызова показывать в UI (describe) и в кратком итоге команды.
_DESCRIBE_LIMIT = 70
_SUMMARY_COMMAND_LIMIT = 60


# Без докстринга: pydantic отдал бы его в JSON-схему параметров, которая уходит модели.
class RunShellParams(BaseModel):  # noqa: D101
    command: str = Field(description="Command to run with /bin/sh")
    timeout: int = Field(
        default=_DEFAULT_TIMEOUT,
        description=f"Timeout in seconds ({MIN_TIMEOUT}–{MAX_TIMEOUT})",
    )
    cwd: str = Field(
        default="",
        description=(
            "Working directory relative to the project root (default: the root). Prefer it "
            "over `cd dir && …`"
        ),
    )
    description: str = Field(
        default="",
        description=(
            "What the command is for, in a few words in the user's language (shown when "
            "the user is asked to confirm it)"
        ),
    )


def command_verdict(params: RunShellParams, ctx: ToolContext) -> Verdict:
    """Вердикт классификатора для команды в её рабочем каталоге."""
    cwd = _relative_cwd(params, ctx)
    return classify(params.command, root=ctx.root, cwd=cwd, environ=subprocess_env())


def _relative_cwd(params: RunShellParams, ctx: ToolContext) -> str:
    if not params.cwd.strip() or params.cwd.strip() == ".":
        return "."
    target = resolve_in_root(ctx.root, params.cwd.strip())
    if not target.is_dir():
        raise ToolError(f"Каталог '{params.cwd}' не найден в проекте.")
    rel = os.path.relpath(target, ctx.root)
    return "." if rel == "." else rel.replace(os.sep, "/")


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


def _notes(res: ProcessResult, timeout: int) -> list[str]:
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


class RunShellTool(Tool[RunShellParams]):
    """Инструмент run_shell: команда оболочки в каталоге проекта."""

    name = "run_shell"
    description = (
        "Runs a shell command and returns stdout/stderr and the exit code. Use it to build, "
        "run tests and utilities — not to read or search files (read_file, search_content "
        "and the navigation tools are faster and safer). Commands that only read the "
        "project (ls, cat, grep, git status/log/diff, Python one-liners that only read) run "
        "without asking the user; others are checked by a reviewer model or confirmed by "
        "the user. There is no keyboard "
        "input (stdin is empty) — use non-interactive flags. Network commands (curl/wget) "
        "are forbidden by the security policy."
    )
    # Модель параметров (класс pydantic), а не константа: имя — часть API инструментов.
    Params = RunShellParams  # noqa: WPS115
    kind = ToolKind.COMMAND

    @override
    def assess(self, params: RunShellParams, ctx: ToolContext) -> Assessment:
        """Оценка команды классификатором: риск, причина, вид вызова, вердикт."""
        verdict = command_verdict(params, ctx)
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
        """Выполняет команду через /bin/sh и возвращает код возврата и вывод."""
        timeout = min(max(params.timeout, MIN_TIMEOUT), MAX_TIMEOUT)
        cwd = ctx.root / _relative_cwd(params, ctx)
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
