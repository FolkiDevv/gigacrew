"""Выполнение shell-команд с захватом вывода."""

from __future__ import annotations

from pydantic import BaseModel, Field

from gigacrew.permissions import ToolKind
from gigacrew.security import RiskLevel, classify_shell_command
from gigacrew.tools.base import Display, Tool, ToolContext, ToolResult
from gigacrew.tools.process import run_process, truncate_middle

_MAX_OUTPUT = 30_000
MIN_TIMEOUT = 1
MAX_TIMEOUT = 600
# Коды оболочки: команда не запускается (126) или не найдена (127).
_SHELL_FAILURES = (126, 127)


class RunShellParams(BaseModel):
    command: str = Field(description="Команда для выполнения через /bin/sh")
    timeout: int = Field(default=120, description="Таймаут в секундах (1–600)")


class RunShellTool(Tool):
    name = "run_shell"
    description = (
        "Выполняет shell-команду в корне проекта и возвращает stdout/stderr и код "
        "возврата. Используйте для сборки, запуска тестов, утилит. Ввод с клавиатуры "
        "недоступен (stdin пуст) — используйте неинтерактивные флаги. "
        "Сетевые команды (curl/wget) запрещены политикой безопасности."
    )
    Params = RunShellParams
    kind = ToolKind.COMMAND

    def risk(self, params: RunShellParams, ctx: ToolContext) -> RiskLevel:
        return classify_shell_command(params.command)

    def describe(self, params: RunShellParams) -> str:
        return params.command[:70]

    def preview(self, params: RunShellParams, ctx: ToolContext) -> Display:
        return Display(f"$ {params.command}", title="команда")

    def run(self, params: RunShellParams, ctx: ToolContext) -> ToolResult:
        timeout = min(max(params.timeout, MIN_TIMEOUT), MAX_TIMEOUT)
        res = run_process(params.command, cwd=ctx.root, timeout=timeout, shell=True)

        combined = res.stdout
        if res.stderr:
            combined += ("\n" if combined else "") + "[stderr]\n" + res.stderr
        combined = truncate_middle(combined, _MAX_OUTPUT)

        notes = []
        if res.killed_background:
            notes.append(
                "фоновые процессы, оставшиеся после команды, остановлены "
                "(долгоживущие процессы через run_shell не поддерживаются)"
            )
        if res.timed_out:
            notes.append(f"команда превысила таймаут ({timeout}с) и была остановлена")
            head = "exit code: (таймаут)"
            summary = "таймаут команды"
        else:
            head = f"exit code: {res.returncode}"
            summary = f"$ {params.command[:60]} → код {res.returncode}"
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
        )
