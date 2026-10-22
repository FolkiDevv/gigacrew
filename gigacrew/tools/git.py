"""Инструмент для работы с git (безопасное подмножество операций)."""

from __future__ import annotations

import shlex

from pydantic import BaseModel, Field

from gigacrew.safety.gitconfig import repo_config_runs_programs
from gigacrew.security import RiskLevel
from gigacrew.tools.base import Display, Tool, ToolContext, ToolError, ToolResult
from gigacrew.tools.process import run_process, truncate_middle

# Разрешённые подкоманды. Деструктивные (reset --hard, clean, push --force)
# намеренно не входят — их при необходимости вызывают через run_shell с
# подтверждением.
_READ_ONLY = {"status", "diff", "log", "show"}
_ALLOWED = _READ_ONLY | {
    "branch",
    "stash",
    "add",
    "commit",
    "checkout",
    "switch",
    "restore",
}

# Подкоманды, которых нет в инструменте, а в исследовании они обычно читают (у reflog есть и
# изменяющие формы): для них — подсказка про run_shell, там их проверяет классификатор.
_READ_ELSEWHERE = frozenset(
    {
        "blame",
        "grep",
        "ls-files",
        "ls-tree",
        "rev-parse",
        "rev-list",
        "shortlog",
        "describe",
        "reflog",
        "merge-base",
        "cat-file",
        "show-ref",
        "name-rev",
        "for-each-ref",
        "whatchanged",
    }
)

# Опции, позволяющие писать/читать файлы вне проекта или запускать внешние
# программы. Запрещены для всех подкоманд (включая сокращения: git принимает
# уникальные префиксы длинных опций).
_FORBIDDEN_OPTIONS = (
    "--output",  # diff/log/show --output=<file>: запись произвольного файла
    "--no-index",  # diff --no-index /abs/path: чтение вне репозитория
    "--ext-diff",  # запуск внешней программы сравнения
    "--pathspec-from-file",  # чтение произвольного файла
)

# Флаги `git branch`, которые только показывают ветки.
_BRANCH_LIST_FLAGS = {
    "-a",
    "-r",
    "-v",
    "-vv",
    "-l",
    "--list",
    "--all",
    "--remotes",
    "--verbose",
    "--show-current",
    "--no-color",
    "--color",
}

_MAX_OUTPUT = 30_000
_TIMEOUT = 60

# Защита в глубину: даже если настройки репозитория подменены, git не запускает
# внешние программы на просмотре. fsmonitor-хук выполняется на `status`, внешний
# diff — на `diff` (и без --ext-diff); неявный bare-репозиторий (HEAD, config,
# objects/ обычными файлами в каталоге проекта) не распознаётся.
_SAFE_CONFIG = ("-c", "core.fsmonitor=false", "-c", "safe.bareRepository=explicit")
_NO_EXT_DIFF = {"diff", "log", "show"}


def _is_forbidden(arg: str) -> bool:
    if not arg.startswith("--") or arg == "--":
        return False
    name = arg.split("=", 1)[0]
    return any(
        name == opt or (len(name) >= 3 and opt.startswith(name)) for opt in _FORBIDDEN_OPTIONS
    )


def _branch_is_listing(args: list[str]) -> bool:
    flags = [a for a in args if a.startswith("-")]
    positional = [a for a in args if not a.startswith("-")]
    if any(f.split("=", 1)[0] not in _BRANCH_LIST_FLAGS for f in flags):
        return False
    # `git branch foo` создаёт ветку; позиционные аргументы — только шаблоны для --list
    return not positional or bool({"-l", "--list"} & set(flags))


class GitParams(BaseModel):
    subcommand: str = Field(
        description=(
            "git subcommand: status, diff, log, show, branch, add, commit, checkout, "
            "switch, restore, stash"
        )
    )
    args: list[str] = Field(
        default_factory=list,
        description="Additional arguments, e.g. ['-m', 'message'] for commit",
    )


class GitTool(Tool):
    name = "git"
    description = (
        "Runs git operations in the project repository. Supported: status, diff, log, "
        "show, branch (without arguments — list branches), stash list/show (read-only); "
        "add, commit, checkout, switch, restore, stash (changing). Destructive operations "
        "and the --output/--no-index options are not available. No editor is opened — "
        "pass -m to commit."
    )
    Params = GitParams

    def risk(self, params: GitParams, ctx: ToolContext) -> RiskLevel:
        # Читающая подкоманда, которой нет в инструменте, — ошибка с подсказкой до решения по
        # режиму: иначе в режиме плана и у explore `git blame` отклонялся бы как изменение.
        # Изменяющие (push, merge…) по-прежнему блокирует режим.
        if params.subcommand in _READ_ELSEWHERE:
            self._check(params)
        sub, args = params.subcommand, params.args
        reading = (
            sub in _READ_ONLY
            or (sub == "branch" and _branch_is_listing(args))
            or (sub == "stash" and bool(args) and args[0] in ("list", "show"))
        )
        if not reading:
            return RiskLevel.WRITE
        # diff/log/show запускают textconv и clean-фильтры из настроек репозитория:
        # если такие есть, чтение — уже не только чтение.
        if repo_config_runs_programs(ctx.root) is not False:
            return RiskLevel.WRITE
        return RiskLevel.SAFE

    def _check(self, params: GitParams) -> None:
        sub = params.subcommand
        if sub not in _ALLOWED:
            hint = (
                f" Команды git вроде {sub}, которых нет в этом инструменте, запускай через "
                "run_shell."
                if sub in _READ_ELSEWHERE
                else ""
            )
            raise ToolError(
                f"Подкоманда git '{sub}' не поддерживается этим инструментом. Доступно: "
                f"{sorted(_ALLOWED)}.{hint}"
            )
        # Опции проверяются и после `--`: `stash list` отбрасывает `--` и передаёт следующие
        # слова в `git log` как опции. Путь, названный как запрещённая опция, не пройдёт.
        for arg in params.args:
            if _is_forbidden(arg):
                raise ToolError(f"Опция '{arg}' запрещена политикой безопасности gigacrew.")

    @staticmethod
    def _command(params: GitParams) -> list[str]:
        sub, args = params.subcommand, list(params.args)
        if sub == "log" and not args:
            args = ["--oneline", "-n", "20"]  # компактный лог по умолчанию
        if sub in _NO_EXT_DIFF:
            args = ["--no-ext-diff", *args]
        elif sub == "stash" and args[:1] in (["show"], ["list"]):
            args = [args[0], "--no-ext-diff", *args[1:]]  # list передаёт опции git log
        return ["git", *_SAFE_CONFIG, sub, *args]

    def describe(self, params: GitParams) -> str:
        # с кавычками, как в shell: `commit -m 'fix bug'`, а не `commit -m fix bug`
        return f"{params.subcommand} {shlex.join(params.args)}".strip()

    def preview(self, params: GitParams, ctx: ToolContext) -> Display | None:
        # Сначала валидация: недопустимую команду не предлагаем подтверждать.
        self._check(params)
        if self.risk(params, ctx) >= RiskLevel.WRITE:
            return Display(f"$ git {self.describe(params)}", title="команда")
        return None

    def run(self, params: GitParams, ctx: ToolContext) -> ToolResult:
        self._check(params)
        cmd = self._command(params)
        try:
            res = run_process(cmd, cwd=ctx.root, timeout=_TIMEOUT)
        except FileNotFoundError as e:
            raise ToolError("git не установлен или недоступен в PATH.") from e
        if res.timed_out:
            return ToolResult(content="git: таймаут", ok=False, summary="git таймаут")

        out = res.stdout + (("\n[stderr]\n" + res.stderr) if res.stderr else "")
        out = truncate_middle(out.strip(), _MAX_OUTPUT) or "(нет вывода)"
        return ToolResult(
            content=f"exit code: {res.returncode}\n{out}",
            ok=res.returncode == 0,
            summary=f"git {params.subcommand} → код {res.returncode}",
            display=Display(out, title=f"git {params.subcommand}"),
        )
