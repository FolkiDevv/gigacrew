"""Выполнение shell-команд с захватом вывода."""

from __future__ import annotations

import contextlib
import os
from typing import Literal, override

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
from gigacrew.tools.process import ProcessResult, run_process, truncate_middle
from gigacrew.tools.search import search_paths

_MAX_OUTPUT = 30_000
MIN_TIMEOUT = 1
MAX_TIMEOUT = 600
_DEFAULT_TIMEOUT = 120
# Коды оболочки: команда не запускается (126) или не найдена (127).
_SHELL_FAILURES = (126, 127)
# Сколько символов вызова показывать в UI (describe) и в кратком итоге команды.
_DESCRIBE_LIMIT = 70
_SUMMARY_COMMAND_LIMIT = 60
_COMMAND: Literal["command"] = "command"  # вызов выполняется командой оболочки
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
        description=f"Timeout in seconds ({MIN_TIMEOUT}–{MAX_TIMEOUT})",
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
    _planned: GrepSearch | Literal["command"] | None = PrivateAttr(None)

    @property
    def planned(self) -> GrepSearch | Literal["command"] | None:
        """Как выполнить вызов по его оценке: встроенным поиском или командой.

        None — оценки не было. run выполняет вызов так, как его оценили и разрешили: если
        к запуску поиск стал невозможен, команда всё равно не выполняется.
        """
        return self._planned

    @planned.setter
    def planned(self, plan: GrepSearch | Literal["command"]) -> None:
        # Приватный атрибут pydantic: в классе — его объявление, а не значение.
        self._planned = plan  # noqa: WPS601


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
        params.planned = _COMMAND if search is None else search
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
        (:func:`_grep_substitute`) — так, как решила оценка
        (:attr:`RunShellParams.planned`): разрешённый поиск не превращается в команду.
        Без оценки это решается здесь.
        """
        planned = params.planned
        if planned is None:
            planned = _grep_substitute(params, ctx) or _COMMAND
        if isinstance(planned, GrepSearch):
            return _search_result(params.command, planned, ctx)
        return _execute(params, ctx)


def _execute(params: RunShellParams, ctx: ToolContext) -> ToolResult:
    """Выполняет команду через /bin/sh: код возврата и вывод для модели."""
    timeout = min(max(params.timeout, MIN_TIMEOUT), MAX_TIMEOUT)
    cwd = (ctx.root / _relative_cwd(params, ctx)).resolve()
    # Папка временных файлов модели есть к запуску команды: не нужен лишний шаг с mkdir.
    # Не вышло (симлинк, нет прав) — команда всё равно выполняется.
    with contextlib.suppress(OSError):
        ctx.workspace.ensure_temp_dir()
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
