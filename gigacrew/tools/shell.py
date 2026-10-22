"""Выполнение shell-команд с захватом вывода."""

from __future__ import annotations

import os

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
from gigacrew.tools.process import run_process, truncate_middle

_MAX_OUTPUT = 30_000
MIN_TIMEOUT = 1
MAX_TIMEOUT = 600
# Коды оболочки: команда не запускается (126) или не найдена (127).
_SHELL_FAILURES = (126, 127)


class RunShellParams(BaseModel):
    command: str = Field(description="Command to run with /bin/sh")
    timeout: int = Field(
        default=120,
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


class RunShellTool(Tool[RunShellParams]):
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
    Params = RunShellParams
    kind = ToolKind.COMMAND

    def assess(self, params: RunShellParams, ctx: ToolContext) -> Assessment:
        verdict = command_verdict(params, ctx)
        risk = _risk(verdict, params.command, classify_commands=ctx.classify_commands)
        reason = verdict.reason if verdict.level is not CommandLevel.READ else ""
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

    def risk(self, params: RunShellParams, ctx: ToolContext) -> RiskLevel:
        return self.assess(params, ctx).risk

    def describe(self, params: RunShellParams) -> str:
        prefix = f"[{params.cwd.strip()}] " if params.cwd.strip() not in {"", "."} else ""
        return (prefix + params.command)[:70]

    def paths(self, params: RunShellParams) -> tuple[str, ...]:
        cwd = params.cwd.strip()
        return (cwd,) if cwd and cwd != "." else ()

    def preview(self, params: RunShellParams, ctx: ToolContext) -> Display:
        cwd = _relative_cwd(params, ctx)
        title = "команда" + (f" · {cwd}" if cwd != "." else "")
        return Display(f"$ {params.command}", title=title)

    def run(self, params: RunShellParams, ctx: ToolContext) -> ToolResult:
        timeout = min(max(params.timeout, MIN_TIMEOUT), MAX_TIMEOUT)
        cwd = ctx.root / _relative_cwd(params, ctx)
        res = run_process(params.command, cwd=cwd, timeout=timeout, shell=True)

        combined = res.stdout
        if res.stderr:
            combined += ("\n" if combined else "") + "[stderr]\n" + res.stderr
        combined = truncate_middle(combined, _MAX_OUTPUT)

        notes = []
        if res.killed_background:
            notes.append(
                "фоновые процессы, оставшиеся после команды, остановлены "
                "(долгоживущие процессы через run_shell не поддерживаются)",
            )
        if res.timed_out:
            notes.append(f"команда превысила таймаут ({timeout}с) и была остановлена")
            head = "exit code: (таймаут)"
            summary = "таймаут команды"
        else:
            head = f"exit code: {res.returncode}"
            summary = f"$ {params.command[:60]} → код {res.returncode}"
        if res.escaped_output:
            # Потомок ушёл из группы команды (setsid) и держит вывод: killpg его не достал.
            notes.append(
                "фоновый процесс, ушедший из группы команды (setsid), остановить не удалось: "
                "он продолжает работать, его вывод сюда больше не попадёт",
            )
            summary += " · фоновый процесс остановить не удалось"
        note_text = "".join(f"\n[{n}]" for n in notes)
        content = f"{head}{note_text}\n{combined or '(пустой вывод)'}"
        ok = res.returncode == 0 and not res.timed_out
        return ToolResult(
            content=content,
            ok=ok,
            summary=summary,
            # Ненулевой код (grep без совпадений, упавшие тесты) — результат, а не
            # застревание; таймаут и «команда не найдена/не запускается» — сбой.
            soft=not ok and not res.timed_out and res.returncode not in _SHELL_FAILURES,
            display=Display(combined, title=f"$ {params.command[:60]}") if combined else None,
            exit_code=None if res.timed_out else res.returncode,
        )
