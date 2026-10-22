"""Инструмент для работы с git (безопасное подмножество операций)."""

from __future__ import annotations

import shlex
from pathlib import Path

from pydantic import BaseModel, Field

from gigacrew.errors import SandboxError
from gigacrew.safety.gitconfig import repo_config_runs_programs, repo_toplevel
from gigacrew.security import RiskLevel, resolve_in_root
from gigacrew.tools.base import Assessment, Display, Tool, ToolContext, ToolError, ToolResult
from gigacrew.tools.process import run_process, truncate_middle

# Разрешённые подкоманды. Деструктивные (reset --hard, clean, push --force)
# намеренно не входят — их при необходимости вызывают через run_shell с
# подтверждением. Формы разрешённых, теряющие работу (restore, checkout -- <пути>,
# branch -D, stash drop/clear), — опасные: о них спрашивают и с -y.
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
_READ_ELSEWHERE = frozenset({
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
})

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


def _options(args: list[str], valued: frozenset[str] = frozenset()) -> tuple[set[str], list[str]]:
    """(опции, позиционные до ``--``): ``--name`` без значения, буквы ``-abc`` по отдельности.
    Слово после опции из ``valued`` — её значение, а не позиционный аргумент.
    """
    flags: set[str] = set()
    positional: list[str] = []
    skip = False
    for arg in args:
        if arg == "--":
            break
        if skip:
            skip = False
        elif arg.startswith("--"):
            flags.add(arg.split("=", 1)[0])
            skip = arg in valued
        elif arg.startswith("-") and arg != "-":
            flags.add(arg)
            if arg[1:].isalpha():
                flags.update("-" + ch for ch in arg[1:])
            skip = arg in valued
        else:
            positional.append(arg)
    return flags, positional


def _has(flags: set[str], *names: str) -> bool:
    """Есть ли опция; длинную git принимает и сокращённой (`--del` — это `--delete`)."""
    return any(
        f == n or (f.startswith("--") and len(f) >= 3 and n.startswith(f))
        for f in flags
        for n in names
    )


def _discards_work(sub: str, args: list[str], root: Path) -> str:
    """Почему вызов теряет работу (правки, ветки, отложенное) — как у классификатора
    run_shell; "" — не теряет. Такие вызовы спрашиваются и с ``-y``.
    """
    flags, positional = _options(args, frozenset({"-b", "-B", "--orphan"}))
    lost = "незакоммиченные правки будут потеряны"
    if sub == "branch" and (
        "-D" in flags or (_has(flags, "-d", "--delete") and _has(flags, "-f", "--force"))
    ):
        return "git branch -D: удаление ветки без проверки слияния"
    overwrite = "ветка с этим именем будет перезаписана"
    if sub == "branch" and _has(flags, "-f", "--force", "-M", "-C"):
        return f"git branch -f: {overwrite}"
    if sub == "stash" and positional[:1] in (["drop"], ["clear"]):
        return f"git stash {positional[0]}: удаление отложенных правок"
    if sub == "checkout":
        if _has(flags, "-f", "--force"):
            return f"git checkout -f: {lost}"
        if "-B" in flags:
            return f"git checkout -B: {overwrite}"
        paths = args[args.index("--") + 1 :] if "--" in args else []
        # `checkout <ревизия> <пути>`, `checkout <файл>` или шаблон путей — это не ветка
        if (
            paths
            or len(positional) > 1
            or any(
                p.startswith(":") or set(p) & set("*?[") or (root / p).exists() for p in positional
            )
        ):
            return f"git checkout <файл>: {lost}"
    if sub == "restore" and (not _has(flags, "-S", "--staged") or _has(flags, "-W", "--worktree")):
        return f"git restore: {lost}"
    if sub == "switch" and _has(flags, "--discard-changes", "-f", "--force"):
        return f"git switch --discard-changes: {lost}"
    if sub == "switch" and _has(flags, "-C", "--force-create"):
        return f"git switch -C: {overwrite}"
    return ""


def _reading(sub: str, args: list[str]) -> bool:
    return (
        sub in _READ_ONLY
        or (sub == "branch" and _branch_is_listing(args))
        or (sub == "stash" and bool(args) and args[0] in {"list", "show"})
    )


def _commit_files(args: list[str]) -> list[str]:
    """Файлы, которые читает `git commit`: -F/--file (сообщение), -t/--template."""
    out: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--":
            break
        if arg.startswith("--"):
            name, eq, value = arg.partition("=")
            if len(name) >= 3 and any(o.startswith(name) for o in ("--file", "--template")):
                if not eq and i + 1 < len(args):
                    value, i = args[i + 1], i + 1
                out.append(value)
        elif arg.startswith("-"):
            # сгруппированные короткие: значение — остаток слова или следующее слово
            for j, ch in enumerate(arg[1:], 1):
                if ch in "mFtcC":
                    value = arg[j + 1 :]
                    if not value and i + 1 < len(args):
                        value, i = args[i + 1], i + 1
                    if ch in "Ft":
                        out.append(value)
                    break
                if ch in "Su":  # необязательное значение — только слитно
                    break
        i += 1
    return out


def _outside_path(sub: str, args: list[str], root: Path) -> str | None:
    """Аргумент, который ведёт за пределы проекта: путь у читающих подкоманд (с ним git
    сам переходит в --no-index или читает чужой репозиторий) или файл `commit -F/-t`.
    """
    if _reading(sub, args) and sub != "branch":
        candidates = [a for a in args if not a.startswith("-")]  # ревизии и пути
    elif sub == "commit":
        candidates = _commit_files(args)
    else:
        return None
    for value in candidates:
        # `<ревизия>:./путь` и `<ревизия>:../путь` — относительно рабочего каталога
        _, colon, path = value.partition(":")
        relative = colon and path.startswith(("./", "../"))
        for candidate in (value, path) if relative else (value,):
            try:
                resolve_in_root(root, candidate)
            except (SandboxError, OSError, ValueError):
                return value
    return None


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
        ),
    )
    args: list[str] = Field(
        default_factory=list,
        description="Additional arguments, e.g. ['-m', 'message'] for commit",
    )


class GitTool(Tool[GitParams]):
    name = "git"
    description = (
        "Runs git operations in the project repository. Supported: status, diff, log, "
        "show, branch (without arguments — list branches), stash list/show (read-only); "
        "add, commit, checkout, switch, restore, stash (changing). Destructive subcommands "
        "(reset, clean, push) and the --output/--no-index options are not available; forms "
        "that discard work (restore, checkout <paths>, branch -D, stash drop/clear) always "
        "need the user's confirmation. No editor is opened — pass -m to commit."
    )
    Params = GitParams

    def assess(self, params: GitParams, ctx: ToolContext) -> Assessment:
        # Читающая подкоманда, которой нет в инструменте, — ошибка с подсказкой до решения по
        # режиму: иначе в режиме плана и у explore `git blame` отклонялся бы как изменение.
        # Изменяющие (push, merge…) по-прежнему блокирует режим.
        if params.subcommand in _READ_ELSEWHERE:
            self._check(params, ctx)
        sub, args = params.subcommand, params.args
        if not _reading(sub, args):
            reason = _discards_work(sub, args, ctx.root) if sub in _ALLOWED else ""
            return Assessment(RiskLevel.DANGEROUS if reason else RiskLevel.WRITE, reason=reason)
        # Чтение — только своего репозитория: вне репозитория `git diff a b` сравнивает любые
        # файлы, а в репозитории над проектом git покажет и то, что лежит вне проекта.
        top = repo_toplevel(ctx.root)
        if top is None:
            return Assessment(RiskLevel.WRITE, reason="git: каталог проекта — не репозиторий git")
        if top != ctx.root.resolve():
            return Assessment(
                RiskLevel.WRITE,
                reason=f"git: проект — подкаталог репозитория {top}, чтение выйдет за проект",
            )
        # diff/log/show запускают textconv и clean-фильтры из настроек репозитория:
        # если такие есть, чтение — уже не только чтение.
        if repo_config_runs_programs(ctx.root) is not False:
            return Assessment(RiskLevel.WRITE)
        return Assessment(RiskLevel.SAFE)

    def risk(self, params: GitParams, ctx: ToolContext) -> RiskLevel:
        return self.assess(params, ctx).risk

    def _check(self, params: GitParams, ctx: ToolContext) -> None:
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
                f"{sorted(_ALLOWED)}.{hint}",
            )
        # Опции проверяются и после `--`: `stash list` отбрасывает `--` и передаёт следующие
        # слова в `git log` как опции. Путь, названный как запрещённая опция, не пройдёт.
        for arg in params.args:
            if _is_forbidden(arg):
                raise ToolError(f"Опция '{arg}' запрещена политикой безопасности gigacrew.")
        outside = _outside_path(sub, params.args, ctx.root)
        if outside is not None:
            raise ToolError(
                f"Путь '{outside}' — вне проекта: инструмент git работает только с файлами проекта.",
            )

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
        self._check(params, ctx)
        if self.risk(params, ctx) >= RiskLevel.WRITE:
            return Display(f"$ git {self.describe(params)}", title="команда")
        return None

    def run(self, params: GitParams, ctx: ToolContext) -> ToolResult:
        self._check(params, ctx)
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
