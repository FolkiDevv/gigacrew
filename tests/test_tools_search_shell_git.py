"""Тесты инструментов: поиск по содержимому, shell, git."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from fakes import RecordingEvents, ScriptedProvider, text_turn, tool_turn

from gigacrew.agent.loop import Agent
from gigacrew.config import Config
from gigacrew.permissions import Decision, PermissionMode, ToolKind, decide
from gigacrew.project.workspace import Workspace
from gigacrew.safety.gitconfig import repo_config_runs_programs
from gigacrew.security import RiskLevel
from gigacrew.tools import git as git_module
from gigacrew.tools.base import ToolContext, ToolError, build_default_registry
from gigacrew.tools.fs import WriteFileTool
from gigacrew.tools.git import GitTool
from gigacrew.tools.search import SearchContentParams, SearchContentTool
from gigacrew.tools.shell import RunShellTool

# Автор коммитов и разрешение file:// — для подмодуля из локального каталога.
_SUBMODULE_GIT_OPTIONS = (
    "-c",
    "user.email=t@t.t",
    "-c",
    "user.name=t",
    "-c",
    "protocol.file.allow=always",
)

# Вызов git: подкоманда и её аргументы.
type _GitCall = tuple[str, list[str]]


def _write(ctx: ToolContext, path: str, content: str) -> None:
    writer = WriteFileTool()
    writer.run(writer.parse({"path": path, "content": content}), ctx)


# ------------------------------- search ------------------------------- #
def test_search_content_finds_matches(ctx: ToolContext) -> None:
    _write(ctx, "a.py", "def foo():\n    return 42\n")
    _write(ctx, "b.py", "x = foo()\n")
    search = SearchContentTool()
    out = search.run(search.parse({"pattern": r"foo"}), ctx)
    assert "a.py:1:" in out.content
    assert "b.py:1:" in out.content


def test_search_glob_filter(ctx: ToolContext) -> None:
    _write(ctx, "a.py", "needle")
    _write(ctx, "a.txt", "needle")
    search = SearchContentTool()
    out = search.run(search.parse({"pattern": "needle", "glob": "*.py"}), ctx)
    assert "a.py" in out.content
    assert "a.txt" not in out.content


def test_search_skips_ignored_binary_large_and_secrets(ctx: ToolContext) -> None:
    _write(ctx, "src/a.py", "needle")
    _write(ctx, "node_modules/m.js", "needle")
    _write(ctx, ".env", "needle=secret")
    _write(ctx, ".env.example", "needle=")
    (ctx.root / "bin.dat").write_bytes(b"needle\0\1\2")
    (ctx.root / "big.txt").write_text("needle\n" * 200_000, encoding="utf-8")
    search = SearchContentTool()
    out = search.run(search.parse({"pattern": "needle"}), ctx).content
    hits = [line for line in out.splitlines() if not line.startswith("…")]
    files = sorted({line.split(":", 1)[0] for line in hits})
    assert files == [".env.example", "src/a.py"]
    assert "пропущено больших файлов (>1 МБ): 1" in out  # модель знает, где не искали


def test_search_glob_with_path(ctx: ToolContext) -> None:
    _write(ctx, "src/x/a.py", "needle")
    _write(ctx, "lib/b.py", "needle")
    search = SearchContentTool()
    out = search.run(search.parse({"pattern": "needle", "glob": "src/**/*.py"}), ctx).content
    assert "src/x/a.py" in out
    assert "lib/b.py" not in out


def test_search_max_results_is_clamped(ctx: ToolContext) -> None:
    _write(ctx, "a.txt", "hit\n" * 2000)
    search = SearchContentTool()
    out = search.run(search.parse({"pattern": "hit", "max_results": 10**6}), ctx).content
    assert len([ln for ln in out.splitlines() if ln.startswith("a.txt:")]) == 500


@pytest.mark.skipif(sys.platform == "win32", reason="симлинки")
def test_search_does_not_follow_symlinks_outside(
    ctx: ToolContext,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    outside = tmp_path_factory.mktemp("outside") / "secret.txt"
    outside.write_text("needle", encoding="utf-8")
    (ctx.root / "link.txt").symlink_to(outside)
    search = SearchContentTool()
    out = search.run(search.parse({"pattern": "needle"}), ctx).content
    assert "link.txt" not in out


def test_search_invalid_regex(ctx: ToolContext) -> None:
    search = SearchContentTool()
    with pytest.raises(ToolError):
        search.run(search.parse({"pattern": "([unclosed"}), ctx)


# ------------------------------- shell -------------------------------- #
def test_shell_echo(ctx: ToolContext) -> None:
    sh = RunShellTool()
    res = sh.run(sh.parse({"command": "echo hello"}), ctx)
    assert res.ok
    assert "hello" in res.content
    assert "exit code: 0" in res.content


def test_shell_nonzero_exit(ctx: ToolContext) -> None:
    sh = RunShellTool()
    res = sh.run(sh.parse({"command": "exit 3"}), ctx)
    assert not res.ok
    assert "exit code: 3" in res.content


def test_shell_runs_in_project_root(ctx: ToolContext) -> None:
    sh = RunShellTool()
    res = sh.run(sh.parse({"command": "pwd"}), ctx)
    assert str(ctx.root) in res.content


def test_shell_dangerous_classification(ctx: ToolContext) -> None:
    sh = RunShellTool()
    safe = sh.parse({"command": "ls -la"})
    write = sh.parse({"command": "touch x"})
    danger = sh.parse({"command": "rm -rf /tmp/x"})
    assert sh.risk(safe, ctx) == RiskLevel.SAFE  # только чтение — без вопроса
    assert sh.risk(write, ctx) == RiskLevel.WRITE
    assert sh.risk(danger, ctx) == RiskLevel.DANGEROUS


def test_shell_scratch_file_needs_no_confirmation(tmp_path: Path) -> None:
    """Временный файл в .gigacode/tmp/ — правка: в режиме «авто» без вопроса и рецензента.

    Папку run_shell создаёт сам (со своим .gitignore): модели не нужен лишний шаг с mkdir.
    """
    command = "cat > .gigacode/tmp/out.txt <<'EOF'\nhello\nEOF"
    provider = ScriptedProvider([tool_turn("run_shell", {"command": command}), text_turn("ок")])
    events = RecordingEvents()
    cfg = Config(access_key="x", project_root=tmp_path, stream=False)
    Agent(provider, build_default_registry(), cfg, events).run_turn("сохрани вывод")
    assert not events.confirms
    temp_dir = Workspace(tmp_path).data_dir / "tmp"
    assert (temp_dir / "out.txt").read_text(encoding="utf-8") == "hello\n"
    assert (temp_dir / ".gitignore").is_file()


def test_shell_runs_in_cwd_and_rejects_outside(ctx: ToolContext) -> None:
    (ctx.root / "sub").mkdir()
    sh = RunShellTool()
    res = sh.run(sh.parse({"command": "pwd", "cwd": "sub"}), ctx)
    assert res.content.rstrip().endswith("/sub")
    assert sh.paths(sh.parse({"command": "ls", "cwd": "sub"})) == ("sub",)
    with pytest.raises(ToolError):
        sh.run(sh.parse({"command": "pwd", "cwd": "../.."}), ctx)
    verdict = sh.assess(sh.parse({"command": "cd sub && ls"}), ctx).verdict
    assert verdict is not None
    assert verdict.cwd == "sub"
    assert verdict.command == "ls"


# -------------------------------- git --------------------------------- #
@pytest.fixture
def git_repo(ctx: ToolContext) -> ToolContext:
    root = ctx.root
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)
    return ctx


def test_git_status(git_repo: ToolContext) -> None:
    ctx = git_repo
    _write(ctx, "f.txt", "hi")
    git = GitTool()
    res = git.run(git.parse({"subcommand": "status"}), ctx)
    assert res.ok
    assert "f.txt" in res.content


def test_git_add_and_commit(git_repo: ToolContext) -> None:
    ctx = git_repo
    _write(ctx, "f.txt", "hi")
    git = GitTool()
    git.run(git.parse({"subcommand": "add", "args": ["f.txt"]}), ctx)
    res = git.run(git.parse({"subcommand": "commit", "args": ["-m", "init"]}), ctx)
    assert res.ok
    log = git.run(git.parse({"subcommand": "log"}), ctx)
    assert "init" in log.content


def test_git_disallowed_subcommand(git_repo: ToolContext) -> None:
    git = GitTool()
    with pytest.raises(ToolError):
        git.run(git.parse({"subcommand": "push"}), git_repo)


def _git(ctx: ToolContext, *args: str) -> None:
    subprocess.run(["git", *args], cwd=ctx.root, check=True, capture_output=True)


@pytest.fixture
def stashed_repo(git_repo: ToolContext) -> ToolContext:
    """Репозиторий с коммитом и одной отложенной правкой (refs/stash есть)."""
    _write(git_repo, "f.txt", "one\n")
    _git(git_repo, "add", "f.txt")
    _git(git_repo, "commit", "-qm", "init")
    _write(git_repo, "f.txt", "two\n")
    _git(git_repo, "stash", "-q")
    return git_repo


@pytest.mark.parametrize("dashdash", [False, True])
@pytest.mark.parametrize(
    ("sub", "args"),
    [
        ("stash", ["list"]),
        ("stash", ["show"]),
        ("diff", []),
        ("log", []),
        ("show", []),
    ],
)
def test_git_forbidden_options_rejected_after_double_dash(
    stashed_repo: ToolContext,
    sub: str,
    args: list[str],
    *,
    dashdash: bool,
) -> None:
    """`stash list` передаёт слова после `--` в `git log` как опции: --output записал бы файл."""
    target = stashed_repo.root / "written.txt"
    extra = ["--"] if dashdash else []
    params_args = [*args, *extra, "--pretty=format:CHOSEN", f"--output={target}"]
    git = GitTool()
    params = git.parse({"subcommand": sub, "args": params_args})
    with pytest.raises(ToolError, match="запрещена"):
        git.preview(params, stashed_repo)
    with pytest.raises(ToolError, match="запрещена"):
        git.run(params, stashed_repo)
    assert not target.exists()


@pytest.mark.parametrize(
    ("sub", "args"),
    [
        ("add", ["--pathspec-from-file=/etc/hostname"]),
        ("add", ["--pathspec-from=x"]),  # сокращение опции
        ("commit", ["--", "--pathspec-from-file=x"]),
        ("stash", ["list", "--ext-diff"]),
        ("show", ["--no-index"]),
    ],
)
def test_git_forbidden_options_for_every_subcommand(
    stashed_repo: ToolContext,
    sub: str,
    args: list[str],
) -> None:
    git = GitTool()
    params = git.parse({"subcommand": sub, "args": args})
    with pytest.raises(ToolError, match="запрещена"):
        git.preview(params, stashed_repo)
    with pytest.raises(ToolError, match="запрещена"):
        git.run(params, stashed_repo)


@pytest.mark.parametrize(
    ("sub", "args", "reason"),
    [
        ("branch", ["-D", "feature"], "git branch -D: удаление ветки без проверки слияния"),
        ("branch", ["--delete", "--force", "feature"], "git branch -D"),
        ("branch", ["-df", "feature"], "git branch -D"),
        ("branch", ["--del", "--forc", "feature"], "git branch -D"),  # сокращения опций
        ("stash", ["drop"], "git stash drop: удаление отложенных правок"),
        ("stash", ["clear"], "git stash clear: удаление отложенных правок"),
        ("checkout", ["--", "f.txt"], "git checkout <файл>: незакоммиченные правки"),
        ("checkout", ["HEAD", "f.txt"], "git checkout <файл>"),
        ("checkout", ["f.txt"], "git checkout <файл>"),
        ("checkout", ["."], "git checkout <файл>"),
        ("checkout", ["*.txt"], "git checkout <файл>"),
        ("checkout", ["-f"], "git checkout -f: незакоммиченные правки будут потеряны"),
        ("checkout", ["--force", "HEAD"], "git checkout -f"),
        ("restore", ["f.txt"], "git restore: незакоммиченные правки будут потеряны"),
        ("restore", ["."], "git restore"),
        ("restore", ["--staged", "--worktree", "f.txt"], "git restore"),
        ("restore", ["-SW", "f.txt"], "git restore"),
        ("switch", ["--discard-changes", "x"], "git switch --discard-changes: незакоммиченные"),
        ("switch", ["-f", "x"], "git switch --discard-changes"),
        ("switch", ["--force", "x"], "git switch --discard-changes"),
        # перезапись существующей ветки — DANGER и у классификатора run_shell
        ("branch", ["-f", "main", "HEAD~1"], "git branch -f: ветка с этим именем будет"),
        ("branch", ["--force", "main"], "git branch -f"),
        ("branch", ["--forc", "main"], "git branch -f"),
        ("branch", ["-M", "old", "main"], "git branch -f"),
        ("branch", ["-C", "old", "main"], "git branch -f"),
        ("checkout", ["-B", "main"], "git checkout -B: ветка с этим именем будет перезаписана"),
        ("checkout", ["-qB", "main", "HEAD~1"], "git checkout -B"),
        ("switch", ["-C", "main"], "git switch -C: ветка с этим именем будет перезаписана"),
        ("switch", ["--force-create", "main"], "git switch -C"),
    ],
)
def test_git_destructive_operations_are_dangerous(
    stashed_repo: ToolContext,
    sub: str,
    args: list[str],
    reason: str,
) -> None:
    """Как и через run_shell: -y и хук `allow` не запускают их без вопроса."""
    git = GitTool()
    assessment = git.assess(git.parse({"subcommand": sub, "args": args}), stashed_repo)
    assert assessment.risk is RiskLevel.DANGEROUS
    assert assessment.reason.startswith(reason)
    decision = decide(PermissionMode.AUTO, assessment.risk, git.kind, auto_approve=True)
    assert decision is Decision.ASK
    hooked = decide(
        PermissionMode.AUTO,
        assessment.risk,
        git.kind,
        auto_approve=False,
        hook="allow",
    )
    assert hooked is Decision.ASK


@pytest.mark.parametrize(
    ("sub", "args"),
    [
        ("branch", ["-d", "feature"]),
        ("branch", ["-m", "new"]),
        ("branch", ["-c", "copy"]),
        ("branch", ["new", "HEAD"]),
        ("stash", []),
        ("stash", ["pop"]),
        ("stash", ["push", "-m", "drop"]),
        ("checkout", ["other"]),  # ветка: файла с таким именем нет
        ("checkout", ["-b", "f.txt"]),  # новая ветка f.txt, а не файл
        ("checkout", ["other", "--"]),  # `--` без путей — явно ветка
        ("restore", ["--staged", "f.txt"]),
        ("switch", ["other"]),
        ("switch", ["-c", "new"]),
        ("add", ["."]),
        ("commit", ["-m", "drop -f -- ."]),
        ("commit", ["-m", "../../outside"]),  # сообщение, а не путь
        ("commit", ["-qm", "/etc/passwd", "--trailer", "x"]),
    ],
)
def test_git_changes_that_keep_work_are_write(
    stashed_repo: ToolContext,
    sub: str,
    args: list[str],
) -> None:
    git = GitTool()
    params = git.parse({"subcommand": sub, "args": args})
    assessment = git.assess(params, stashed_repo)
    assert assessment.risk is RiskLevel.WRITE
    assert not assessment.reason
    preview = git.preview(params, stashed_repo)
    assert preview is not None
    assert preview.text == f"$ git {git.describe(params)}"


def test_git_restore_with_yes_flag_still_asks(stashed_repo: ToolContext) -> None:
    """Агент с -y: `restore .` спрашивает (как опасное), а правка остаётся при отказе."""
    root = stashed_repo.root
    _write(stashed_repo, "f.txt", "three\n")
    provider = ScriptedProvider(
        [tool_turn("git", {"subcommand": "restore", "args": ["."]}), text_turn("ок")],
    )
    events = RecordingEvents(confirm_answer=False)
    cfg = Config(access_key="x", project_root=root, stream=False, auto_approve=True)
    Agent(provider, build_default_registry(), cfg, events).run_turn("откати правки")
    assert len(events.confirms) == 1
    assert events.confirms[0][2] is True
    assert (root / "f.txt").read_text(encoding="utf-8") == "three\n"


@pytest.fixture
def outside_secret(tmp_path_factory: pytest.TempPathFactory) -> Path:
    secret = tmp_path_factory.mktemp("outside") / "secret.txt"
    secret.write_text("SECRET-LINE\n", encoding="utf-8")
    return secret


@pytest.mark.parametrize(
    ("sub", "args"),
    [
        ("diff", ["{abs}", "f.txt"]),  # путь вне дерева — git сам включает --no-index
        ("diff", ["--", "{abs}", "/dev/null"]),
        ("diff", ["f.txt", "/dev/null"]),
        ("diff", ["{rel}", "f.txt"]),
        ("log", ["-p", "--", "{abs}"]),
        ("show", ["HEAD:../{rel}"]),
        ("stash", ["show", "--", "{abs}"]),
        ("commit", ["-F", "{abs}"]),  # текст файла попал бы в сообщение коммита
        ("commit", ["--file={abs}"]),
        ("commit", ["--templ", "{abs}"]),  # сокращение --template
        ("commit", ["-qF", "{rel}"]),
        ("commit", ["-t{abs}"]),
    ],
)
def test_git_paths_outside_project_are_refused(
    stashed_repo: ToolContext,
    outside_secret: Path,
    sub: str,
    args: list[str],
) -> None:
    outside_dir = outside_secret.parent.name
    rel = f"../{outside_dir}/{outside_secret.name}"
    resolved_args = [arg.format(abs=outside_secret, rel=rel) for arg in args]
    git = GitTool()
    params = git.parse({"subcommand": sub, "args": resolved_args})
    with pytest.raises(ToolError, match="вне проекта"):
        git.preview(params, stashed_repo)
    with pytest.raises(ToolError, match="вне проекта"):
        git.run(params, stashed_repo)


def test_git_paths_outside_project_are_refused_outside_a_repository(
    ctx: ToolContext,
    outside_secret: Path,
) -> None:
    """Вне репозитория `git diff a b` — это `--no-index`: содержимое любого файла."""
    _write(ctx, "f.txt", "x\n")
    git = GitTool()
    params = git.parse({"subcommand": "diff", "args": [str(outside_secret), "/dev/null"]})
    with pytest.raises(ToolError, match="вне проекта"):
        git.run(params, ctx)


@pytest.mark.parametrize(
    ("sub", "args"),
    [
        ("diff", ["f.txt"]),
        ("diff", ["HEAD", "--", "f.txt"]),
        ("log", ["--oneline", "HEAD~0..HEAD", "--", "."]),
        ("show", ["HEAD:f.txt"]),
        ("stash", ["show", "-p"]),
    ],
)
def test_git_reads_inside_project_stay_safe(
    stashed_repo: ToolContext,
    sub: str,
    args: list[str],
) -> None:
    git = GitTool()
    params = git.parse({"subcommand": sub, "args": args})
    assert git.assess(params, stashed_repo).risk is RiskLevel.SAFE
    assert git.run(params, stashed_repo).ok is True


@pytest.mark.parametrize(("sub", "args"), [("status", []), ("diff", []), ("log", [])])
def test_git_reads_outside_a_repository_are_not_safe(
    ctx: ToolContext,
    sub: str,
    args: list[str],
) -> None:
    git = GitTool()
    assessment = git.assess(git.parse({"subcommand": sub, "args": args}), ctx)
    assert assessment.risk is RiskLevel.WRITE
    assert assessment.reason == "git: каталог проекта — не репозиторий git"


def test_git_reads_in_enclosing_repository_are_not_safe(tmp_path: Path) -> None:
    """Проект — подкаталог чужого репозитория: git показал бы и файлы вне проекта."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".netrc").write_text("password hunter2\n", encoding="utf-8")
    project = tmp_path / "proj"
    project.mkdir()
    (project / "a.txt").write_text("a\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    commit = ["git", "-c", "user.email=t@t.t", "-c", "user.name=t", "commit", "-qm", "i"]
    subprocess.run(commit, cwd=tmp_path, check=True)
    ctx = ToolContext(workspace=Workspace(project))
    git = GitTool()
    reads: tuple[_GitCall, ...] = (
        ("status", []),
        ("log", ["-p"]),
        ("show", ["HEAD:./../.netrc"]),
    )
    for sub, args in reads:
        assessment = git.assess(git.parse({"subcommand": sub, "args": args}), ctx)
        assert assessment.risk is RiskLevel.WRITE
        assert assessment.reason.startswith("git: проект — подкаталог репозитория ")
        assert decide(PermissionMode.PLAN, assessment.risk, git.kind, auto_approve=False) is (
            Decision.BLOCK
        )


def test_git_command_disables_external_programs() -> None:
    git = GitTool()
    cmd = git._command(git.parse({"subcommand": "diff", "args": ["--", "a.py"]}))
    assert cmd[:5] == ["git", "-c", "core.fsmonitor=false", "-c", "safe.bareRepository=explicit"]
    assert cmd[5:] == ["diff", "--no-ext-diff", "--", "a.py"]
    assert git._command(git.parse({"subcommand": "log"}))[5:] == [
        "log",
        "--no-ext-diff",
        "--oneline",
        "-n",
        "20",
    ]
    stash = git._command(git.parse({"subcommand": "stash", "args": ["show", "-p"]}))
    assert stash[5:] == ["stash", "show", "--no-ext-diff", "-p"]
    # stash list передаёт опции git log: -p показал бы дифф внешней программой
    stash = git._command(git.parse({"subcommand": "stash", "args": ["list", "-p"]}))
    assert stash[5:] == ["stash", "list", "--no-ext-diff", "-p"]
    assert git._command(git.parse({"subcommand": "status"}))[5:] == ["status"]


def _marker_script(root: Path, name: str) -> tuple[Path, Path]:
    script = root / f"{name}.sh"
    marker = root / f"{name}.marker"
    script.write_text(f'#!/bin/sh\ntouch "{marker}"\nexit 0\n', encoding="utf-8")
    script.chmod(0o755)
    return script, marker


@pytest.mark.skipif(sys.platform == "win32", reason="sh-скрипты")
def test_git_status_does_not_run_fsmonitor_hook(git_repo: ToolContext) -> None:
    """Подменённый .git/config не запускает команду на `git status` (SAFE, без вопроса)."""
    root = git_repo.root
    hook, marker = _marker_script(root, "fsmonitor")
    subprocess.run(["git", "config", "core.fsmonitor", str(hook)], cwd=root, check=True)
    subprocess.run(["git", "status"], cwd=root, capture_output=True, check=False)
    if not marker.exists():
        pytest.skip("эта версия git не запускает fsmonitor-хук")
    marker.unlink()
    git = GitTool()
    git.run(git.parse({"subcommand": "status"}), git_repo)
    assert not marker.exists()


@pytest.mark.parametrize(
    ("sub", "args"),
    [("diff", []), ("log", ["-p"]), ("show", []), ("status", []), ("stash", ["list"])],
)
def test_git_reads_with_textconv_in_repo_config_are_not_safe(
    stashed_repo: ToolContext,
    sub: str,
    args: list[str],
) -> None:
    """Textconv из .git/config запускается на чтении: оно уже не SAFE (и в репозитории)."""
    git = GitTool()
    params = git.parse({"subcommand": sub, "args": args})
    assert git.risk(params, stashed_repo) is RiskLevel.SAFE
    _git(stashed_repo, "config", "diff.evil.textconv", "sh -c id")
    assert git.risk(params, stashed_repo) is RiskLevel.WRITE


def test_git_reads_are_not_safe_when_repo_config_is_unknown(
    stashed_repo: ToolContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(git_module, "repo_config_runs_programs", lambda root: None)
    git = GitTool()
    assert git.risk(git.parse({"subcommand": "diff"}), stashed_repo) is RiskLevel.WRITE


@pytest.mark.parametrize(
    ("args", "risk"),
    [
        (["--set-upstream-to=origin/main"], RiskLevel.WRITE),
        (["--unset-upstream"], RiskLevel.WRITE),
        (["--edit-description"], RiskLevel.WRITE),
        (["-u", "origin/main"], RiskLevel.WRITE),
        (["--list", "-d", "x"], RiskLevel.WRITE),
        (["-D", "--list", "x"], RiskLevel.DANGEROUS),
        (["-r", "--no-color"], RiskLevel.SAFE),
        (["-l", "feat*"], RiskLevel.SAFE),
    ],
)
def test_git_branch_only_listing_flags_are_safe(
    stashed_repo: ToolContext,
    args: list[str],
    risk: RiskLevel,
) -> None:
    """Флаги без позиционных аргументов: изменяющие ветки — не список, даже без имени."""
    git = GitTool()
    assert git.risk(git.parse({"subcommand": "branch", "args": args}), stashed_repo) is risk


def test_git_tool_is_not_an_edit() -> None:
    """Правки файлов в «авто» идут без вопроса; git commit/restore — не правка файла."""
    assert GitTool.kind is ToolKind.OTHER
    assert decide(PermissionMode.AUTO, RiskLevel.WRITE, GitTool.kind, auto_approve=False) is (
        Decision.ASK
    )


def test_git_commit_in_auto_mode_is_confirmed(stashed_repo: ToolContext) -> None:
    root = stashed_repo.root
    _git(stashed_repo, "add", "f.txt")
    rev_parse = ["git", "rev-parse", "HEAD"]
    head = subprocess.run(rev_parse, cwd=root, capture_output=True, check=False).stdout
    provider = ScriptedProvider(
        [tool_turn("git", {"subcommand": "commit", "args": ["-m", "x"]}), text_turn("ок")],
    )
    events = RecordingEvents(confirm_answer=False)
    cfg = Config(access_key="x", project_root=root, stream=False, auto_approve=False)
    Agent(provider, build_default_registry(), cfg, events).run_turn("закоммить")
    assert len(events.confirms) == 1
    assert events.confirms[0][2] is False
    after = subprocess.run(rev_parse, cwd=root, capture_output=True, check=False).stdout
    assert after == head


@pytest.mark.skipif(sys.platform == "win32", reason="sh-скрипты")
def test_git_diff_does_not_run_external_diff(git_repo: ToolContext) -> None:
    root = git_repo.root
    _write(git_repo, "f.txt", "one\n")
    subprocess.run(["git", "add", "f.txt"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=root, check=True)
    _write(git_repo, "f.txt", "two\n")
    ext, marker = _marker_script(root, "extdiff")
    subprocess.run(["git", "config", "diff.external", str(ext)], cwd=root, check=True)
    subprocess.run(["git", "diff"], cwd=root, capture_output=True, check=False)
    assert marker.exists()  # голый git внешнюю программу запускает
    marker.unlink()
    git = GitTool()
    res = git.run(git.parse({"subcommand": "diff"}), git_repo)
    assert not marker.exists()
    assert "-one" in res.content
    assert "+two" in res.content


def _git_in(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", *_SUBMODULE_GIT_OPTIONS, *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def _library_repo(base: Path) -> Path:
    """Репозиторий lib с одним коммитом: data.txt и фильтр x для *.txt в .gitattributes."""
    lib = base / "lib"
    lib.mkdir()
    _git_in(lib, "init", "-q")
    (lib / "data.txt").write_text("v1\n", encoding="utf-8")
    (lib / ".gitattributes").write_text("*.txt filter=x\n", encoding="utf-8")
    _git_in(lib, "add", ".")
    _git_in(lib, "commit", "-qm", "i")
    return lib


def _project_with_submodule(base: Path, *, filter_program: Path | None = None) -> ToolContext:
    """Проект с подмодулем libs/lib; программа фильтра — только в настройках подмодуля."""
    lib = _library_repo(base)
    root = base / "proj"
    root.mkdir()
    _git_in(root, "init", "-q")
    _git_in(root, "submodule", "add", "-q", str(lib), "libs/lib")
    _git_in(root, "commit", "-qm", "s")
    if filter_program is not None:
        _git_in(root / "libs" / "lib", "config", "filter.x.clean", str(filter_program))
    (root / "libs" / "lib" / "data.txt").write_text("v2\n", encoding="utf-8")
    return ToolContext(workspace=Workspace(root))


@pytest.mark.skipif(sys.platform == "win32", reason="sh-скрипты")
def test_git_status_with_program_in_submodule_config_is_not_safe(tmp_path: Path) -> None:
    """`git status` заходит в подмодули и запускает фильтры из .git/modules/<имя>/config."""
    script, marker = _marker_script(tmp_path, "clean")
    ctx = _project_with_submodule(tmp_path, filter_program=script)
    subprocess.run(["git", "status"], cwd=ctx.root, check=True, capture_output=True)
    assert marker.exists()  # голый git программу из настроек подмодуля запускает
    assert repo_config_runs_programs(ctx.root) is True
    assert repo_config_runs_programs(ctx.root / "libs") is True  # и из подкаталога
    git = GitTool()
    assert git.risk(git.parse({"subcommand": "status"}), ctx) is RiskLevel.WRITE
    sh = RunShellTool()  # тот же признак у классификатора run_shell
    assert sh.risk(sh.parse({"command": "git status"}), ctx) is not RiskLevel.SAFE


def test_git_status_with_clean_submodule_stays_safe(tmp_path: Path) -> None:
    ctx = _project_with_submodule(tmp_path)
    assert repo_config_runs_programs(ctx.root) is False
    git = GitTool()
    assert git.risk(git.parse({"subcommand": "status"}), ctx) is RiskLevel.SAFE


@pytest.mark.skipif(sys.platform == "win32", reason="sh-скрипты")
def test_git_ignores_implicit_bare_repository(ctx: ToolContext) -> None:
    """HEAD/config/objects обычными файлами в проекте не делают его bare-репозиторием."""
    if subprocess.run(["git", "--version"], capture_output=True, check=False).returncode:
        pytest.skip("нет git")
    bare = ctx.root / "planted"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    hook, marker = _marker_script(ctx.root, "planted")
    subprocess.run(["git", "config", "core.fsmonitor", str(hook)], cwd=bare, check=True)
    res = subprocess.run(
        ["git", "-c", "safe.bareRepository=explicit", "status"],
        cwd=bare,
        capture_output=True,
        check=False,
    )
    if res.returncode == 0:
        pytest.skip("эта версия git не знает safe.bareRepository")
    git = GitTool()
    out = git.run(git.parse({"subcommand": "log"}), ToolContext(workspace=Workspace(bare)))
    assert not out.ok
    assert "safe.bareRepository" in out.content
    assert not marker.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="FIFO")
def test_search_skips_fifo(ctx: ToolContext) -> None:
    os.mkfifo(ctx.root / "pipe")
    _write(ctx, "a.txt", "needle")
    search = SearchContentTool()
    out = search.run(search.parse({"pattern": "needle"}), ctx).content
    assert "a.txt" in out


def test_search_skips_symlinks_to_secrets(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("симлинки")
    (tmp_path / ".env").write_text("TOKEN=s3cr3t\n", encoding="utf-8")
    (tmp_path / "config.txt").symlink_to(tmp_path / ".env")
    ctx = ToolContext(workspace=Workspace(tmp_path))
    found = SearchContentTool().run(SearchContentParams(pattern="s3cr3t"), ctx)
    assert "s3cr3t" not in found.content


def test_git_describe_quotes_arguments(ctx: ToolContext) -> None:
    git = GitTool()
    params = git.parse({"subcommand": "commit", "args": ["-m", "fix bug"]})
    assert git.describe(params) == "commit -m 'fix bug'"
    preview = git.preview(params, ctx)
    assert preview is not None
    assert preview.text == "$ git commit -m 'fix bug'"


def test_search_limit_note_only_when_more_matches(ctx: ToolContext) -> None:
    _write(ctx, "a.txt", "hit\n" * 3)
    search = SearchContentTool()
    exact = search.run(search.parse({"pattern": "hit", "max_results": 3}), ctx).content
    assert "показаны первые" not in exact
    assert exact.count("a.txt:") == 3
    more = search.run(search.parse({"pattern": "hit", "max_results": 2}), ctx).content
    assert more.count("a.txt:") == 2
    assert "показаны первые 2 совпадений" in more


def test_search_line_numbers_match_read_file(ctx: ToolContext) -> None:
    _write(ctx, "f.txt", "a\x0cb\nneedle\n")
    search = SearchContentTool()
    assert search.run(search.parse({"pattern": "needle"}), ctx).content == "f.txt:2:needle"
