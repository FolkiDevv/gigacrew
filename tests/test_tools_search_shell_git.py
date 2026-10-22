"""Тесты инструментов: поиск по содержимому, shell, git."""

from __future__ import annotations

import subprocess
import sys

import pytest

from gigacrew.permissions import Decision, PermissionMode, decide
from gigacrew.security import RiskLevel
from gigacrew.tools.base import ToolError
from gigacrew.tools.fs import WriteFileTool
from gigacrew.tools.git import GitTool
from gigacrew.tools.search import SearchContentTool
from gigacrew.tools.shell import RunShellTool


def _write(ctx, path, content):
    w = WriteFileTool()
    w.run(w.parse({"path": path, "content": content}), ctx)


# ------------------------------- search ------------------------------- #
def test_search_content_finds_matches(ctx):
    _write(ctx, "a.py", "def foo():\n    return 42\n")
    _write(ctx, "b.py", "x = foo()\n")
    s = SearchContentTool()
    out = s.run(s.parse({"pattern": r"foo"}), ctx)
    assert "a.py:1:" in out.content
    assert "b.py:1:" in out.content


def test_search_glob_filter(ctx):
    _write(ctx, "a.py", "needle")
    _write(ctx, "a.txt", "needle")
    s = SearchContentTool()
    out = s.run(s.parse({"pattern": "needle", "glob": "*.py"}), ctx)
    assert "a.py" in out.content and "a.txt" not in out.content


def test_search_skips_ignored_binary_large_and_secrets(ctx):
    _write(ctx, "src/a.py", "needle")
    _write(ctx, "node_modules/m.js", "needle")
    _write(ctx, ".env", "needle=secret")
    _write(ctx, ".env.example", "needle=")
    (ctx.root / "bin.dat").write_bytes(b"needle\0\1\2")
    (ctx.root / "big.txt").write_text("needle\n" * 200_000, encoding="utf-8")
    s = SearchContentTool()
    out = s.run(s.parse({"pattern": "needle"}), ctx).content
    files = sorted({ln.split(":", 1)[0] for ln in out.splitlines() if not ln.startswith("…")})
    assert files == [".env.example", "src/a.py"]
    assert "пропущено больших файлов (>1 МБ): 1" in out  # модель знает, где не искали


def test_search_glob_with_path(ctx):
    _write(ctx, "src/x/a.py", "needle")
    _write(ctx, "lib/b.py", "needle")
    s = SearchContentTool()
    out = s.run(s.parse({"pattern": "needle", "glob": "src/**/*.py"}), ctx).content
    assert "src/x/a.py" in out and "lib/b.py" not in out


def test_search_max_results_is_clamped(ctx):
    _write(ctx, "a.txt", "hit\n" * 2000)
    s = SearchContentTool()
    out = s.run(s.parse({"pattern": "hit", "max_results": 10**6}), ctx).content
    assert len([ln for ln in out.splitlines() if ln.startswith("a.txt:")]) == 500


@pytest.mark.skipif(sys.platform == "win32", reason="симлинки")
def test_search_does_not_follow_symlinks_outside(ctx, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "secret.txt"
    outside.write_text("needle", encoding="utf-8")
    (ctx.root / "link.txt").symlink_to(outside)
    s = SearchContentTool()
    out = s.run(s.parse({"pattern": "needle"}), ctx).content
    assert "link.txt" not in out


def test_search_invalid_regex(ctx):
    s = SearchContentTool()
    with pytest.raises(ToolError):
        s.run(s.parse({"pattern": "([unclosed"}), ctx)


# ------------------------------- shell -------------------------------- #
def test_shell_echo(ctx):
    sh = RunShellTool()
    res = sh.run(sh.parse({"command": "echo hello"}), ctx)
    assert res.ok
    assert "hello" in res.content
    assert "exit code: 0" in res.content


def test_shell_nonzero_exit(ctx):
    sh = RunShellTool()
    res = sh.run(sh.parse({"command": "exit 3"}), ctx)
    assert not res.ok
    assert "exit code: 3" in res.content


def test_shell_runs_in_project_root(ctx):
    sh = RunShellTool()
    res = sh.run(sh.parse({"command": "pwd"}), ctx)
    assert str(ctx.root) in res.content


def test_shell_dangerous_classification(ctx):
    from gigacrew.security import RiskLevel

    sh = RunShellTool()
    safe = sh.parse({"command": "ls -la"})
    write = sh.parse({"command": "touch x"})
    danger = sh.parse({"command": "rm -rf /tmp/x"})
    assert sh.risk(safe, ctx) == RiskLevel.SAFE  # только чтение — без вопроса
    assert sh.risk(write, ctx) == RiskLevel.WRITE
    assert sh.risk(danger, ctx) == RiskLevel.DANGEROUS


def test_shell_runs_in_cwd_and_rejects_outside(ctx):
    (ctx.root / "sub").mkdir()
    sh = RunShellTool()
    res = sh.run(sh.parse({"command": "pwd", "cwd": "sub"}), ctx)
    assert res.content.rstrip().endswith("/sub")
    assert sh.paths(sh.parse({"command": "ls", "cwd": "sub"})) == ("sub",)
    with pytest.raises(ToolError):
        sh.run(sh.parse({"command": "pwd", "cwd": "../.."}), ctx)
    verdict = sh.assess(sh.parse({"command": "cd sub && ls"}), ctx).verdict
    assert verdict is not None and verdict.cwd == "sub" and verdict.command == "ls"


# -------------------------------- git --------------------------------- #
@pytest.fixture
def git_repo(ctx):
    root = ctx.root
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)
    return ctx


def test_git_status(git_repo):
    ctx = git_repo
    _write(ctx, "f.txt", "hi")
    g = GitTool()
    res = g.run(g.parse({"subcommand": "status"}), ctx)
    assert res.ok
    assert "f.txt" in res.content


def test_git_add_and_commit(git_repo):
    ctx = git_repo
    _write(ctx, "f.txt", "hi")
    g = GitTool()
    g.run(g.parse({"subcommand": "add", "args": ["f.txt"]}), ctx)
    res = g.run(g.parse({"subcommand": "commit", "args": ["-m", "init"]}), ctx)
    assert res.ok
    log = g.run(g.parse({"subcommand": "log"}), ctx)
    assert "init" in log.content


def test_git_disallowed_subcommand(git_repo):
    g = GitTool()
    with pytest.raises(ToolError):
        g.run(g.parse({"subcommand": "push"}), git_repo)


def _git(ctx, *args):
    subprocess.run(["git", *args], cwd=ctx.root, check=True, capture_output=True)


@pytest.fixture
def stashed_repo(git_repo):
    """Репозиторий с коммитом и одной отложенной правкой (refs/stash есть)."""
    _write(git_repo, "f.txt", "one\n")
    _git(git_repo, "add", "f.txt")
    _git(git_repo, "commit", "-qm", "init")
    _write(git_repo, "f.txt", "two\n")
    _git(git_repo, "stash", "-q")
    return git_repo


@pytest.mark.parametrize("dashdash", [False, True])
@pytest.mark.parametrize(
    "sub, args",
    [
        ("stash", ["list"]),
        ("stash", ["show"]),
        ("diff", []),
        ("log", []),
        ("show", []),
    ],
)
def test_git_forbidden_options_rejected_after_double_dash(stashed_repo, sub, args, dashdash):
    """`stash list` передаёт слова после `--` в `git log` как опции: --output записал бы файл."""
    target = stashed_repo.root / "written.txt"
    extra = ["--"] if dashdash else []
    params_args = [*args, *extra, "--pretty=format:CHOSEN", f"--output={target}"]
    g = GitTool()
    params = g.parse({"subcommand": sub, "args": params_args})
    with pytest.raises(ToolError, match="запрещена"):
        g.preview(params, stashed_repo)
    with pytest.raises(ToolError, match="запрещена"):
        g.run(params, stashed_repo)
    assert not target.exists()


@pytest.mark.parametrize(
    "sub, args",
    [
        ("add", ["--pathspec-from-file=/etc/hostname"]),
        ("add", ["--pathspec-from=x"]),  # сокращение опции
        ("commit", ["--", "--pathspec-from-file=x"]),
        ("stash", ["list", "--ext-diff"]),
        ("show", ["--no-index"]),
    ],
)
def test_git_forbidden_options_for_every_subcommand(stashed_repo, sub, args):
    g = GitTool()
    params = g.parse({"subcommand": sub, "args": args})
    with pytest.raises(ToolError, match="запрещена"):
        g.preview(params, stashed_repo)
    with pytest.raises(ToolError, match="запрещена"):
        g.run(params, stashed_repo)


@pytest.mark.parametrize(
    "sub, args, reason",
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
def test_git_destructive_operations_are_dangerous(stashed_repo, sub, args, reason):
    """Как и через run_shell: -y и хук `allow` не запускают их без вопроса."""
    g = GitTool()
    assessment = g.assess(g.parse({"subcommand": sub, "args": args}), stashed_repo)
    assert assessment.risk is RiskLevel.DANGEROUS
    assert assessment.reason.startswith(reason)
    decision = decide(PermissionMode.AUTO, assessment.risk, g.kind, auto_approve=True)
    assert decision is Decision.ASK
    hooked = decide(PermissionMode.AUTO, assessment.risk, g.kind, auto_approve=False, hook="allow")
    assert hooked is Decision.ASK


@pytest.mark.parametrize(
    "sub, args",
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
def test_git_changes_that_keep_work_are_write(stashed_repo, sub, args):
    g = GitTool()
    params = g.parse({"subcommand": sub, "args": args})
    assessment = g.assess(params, stashed_repo)
    assert assessment.risk is RiskLevel.WRITE
    assert assessment.reason == ""
    assert g.preview(params, stashed_repo).text == f"$ git {g.describe(params)}"


def test_git_restore_with_yes_flag_still_asks(stashed_repo):
    """Агент с -y: `restore .` спрашивает (как опасное), а правка остаётся при отказе."""
    from fakes import RecordingEvents, ScriptedProvider, text_turn, tool_turn

    from gigacrew.agent.loop import Agent
    from gigacrew.config import Config
    from gigacrew.tools.base import build_default_registry

    root = stashed_repo.root
    _write(stashed_repo, "f.txt", "three\n")
    provider = ScriptedProvider(
        [tool_turn("git", {"subcommand": "restore", "args": ["."]}), text_turn("ок")]
    )
    events = RecordingEvents(confirm_answer=False)
    cfg = Config(access_key="x", project_root=root, stream=False, auto_approve=True)
    Agent(provider, build_default_registry(), cfg, events).run_turn("откати правки")
    assert len(events.confirms) == 1 and events.confirms[0][2] is True  # dangerous
    assert (root / "f.txt").read_text(encoding="utf-8") == "three\n"


@pytest.fixture
def outside_secret(tmp_path_factory):
    secret = tmp_path_factory.mktemp("outside") / "secret.txt"
    secret.write_text("SECRET-LINE\n", encoding="utf-8")
    return secret


@pytest.mark.parametrize(
    "sub, args",
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
def test_git_paths_outside_project_are_refused(stashed_repo, outside_secret, sub, args):
    rel = f"../{outside_secret.parent.name}/{outside_secret.name}"
    args = [a.format(abs=outside_secret, rel=rel) for a in args]
    g = GitTool()
    params = g.parse({"subcommand": sub, "args": args})
    with pytest.raises(ToolError, match="вне проекта"):
        g.preview(params, stashed_repo)
    with pytest.raises(ToolError, match="вне проекта"):
        g.run(params, stashed_repo)


def test_git_paths_outside_project_are_refused_outside_a_repository(ctx, outside_secret):
    """Вне репозитория `git diff a b` — это `--no-index`: содержимое любого файла."""
    _write(ctx, "f.txt", "x\n")
    g = GitTool()
    params = g.parse({"subcommand": "diff", "args": [str(outside_secret), "/dev/null"]})
    with pytest.raises(ToolError, match="вне проекта"):
        g.run(params, ctx)


@pytest.mark.parametrize(
    "sub, args",
    [
        ("diff", ["f.txt"]),
        ("diff", ["HEAD", "--", "f.txt"]),
        ("log", ["--oneline", "HEAD~0..HEAD", "--", "."]),
        ("show", ["HEAD:f.txt"]),
        ("stash", ["show", "-p"]),
    ],
)
def test_git_reads_inside_project_stay_safe(stashed_repo, sub, args):
    g = GitTool()
    params = g.parse({"subcommand": sub, "args": args})
    assert g.assess(params, stashed_repo).risk is RiskLevel.SAFE
    assert g.run(params, stashed_repo).ok is True


@pytest.mark.parametrize("sub, args", [("status", []), ("diff", []), ("log", [])])
def test_git_reads_outside_a_repository_are_not_safe(ctx, sub, args):
    g = GitTool()
    assessment = g.assess(g.parse({"subcommand": sub, "args": args}), ctx)
    assert assessment.risk is RiskLevel.WRITE
    assert assessment.reason == "git: каталог проекта — не репозиторий git"


def test_git_reads_in_enclosing_repository_are_not_safe(tmp_path):
    """Проект — подкаталог чужого репозитория: git показал бы и файлы вне проекта."""
    from gigacrew.project.workspace import Workspace
    from gigacrew.tools.base import ToolContext

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".netrc").write_text("password hunter2\n", encoding="utf-8")
    project = tmp_path / "proj"
    project.mkdir()
    (project / "a.txt").write_text("a\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    commit = ["git", "-c", "user.email=t@t.t", "-c", "user.name=t", "commit", "-qm", "i"]
    subprocess.run(commit, cwd=tmp_path, check=True)
    ctx = ToolContext(workspace=Workspace(project))
    g = GitTool()
    for sub, args in [("status", []), ("log", ["-p"]), ("show", ["HEAD:./../.netrc"])]:
        assessment = g.assess(g.parse({"subcommand": sub, "args": args}), ctx)
        assert assessment.risk is RiskLevel.WRITE
        assert assessment.reason.startswith("git: проект — подкаталог репозитория ")
        assert decide(PermissionMode.PLAN, assessment.risk, g.kind, auto_approve=False) is (
            Decision.BLOCK
        )


def test_git_command_disables_external_programs():
    g = GitTool()
    cmd = g._command(g.parse({"subcommand": "diff", "args": ["--", "a.py"]}))
    assert cmd[:5] == ["git", "-c", "core.fsmonitor=false", "-c", "safe.bareRepository=explicit"]
    assert cmd[5:] == ["diff", "--no-ext-diff", "--", "a.py"]
    assert g._command(g.parse({"subcommand": "log"}))[5:] == [
        "log",
        "--no-ext-diff",
        "--oneline",
        "-n",
        "20",
    ]
    stash = g._command(g.parse({"subcommand": "stash", "args": ["show", "-p"]}))
    assert stash[5:] == ["stash", "show", "--no-ext-diff", "-p"]
    # stash list передаёт опции git log: -p показал бы дифф внешней программой
    stash = g._command(g.parse({"subcommand": "stash", "args": ["list", "-p"]}))
    assert stash[5:] == ["stash", "list", "--no-ext-diff", "-p"]
    assert g._command(g.parse({"subcommand": "status"}))[5:] == ["status"]


def _marker_script(root, name):
    script = root / f"{name}.sh"
    marker = root / f"{name}.marker"
    script.write_text(f'#!/bin/sh\ntouch "{marker}"\nexit 0\n', encoding="utf-8")
    script.chmod(0o755)
    return script, marker


@pytest.mark.skipif(sys.platform == "win32", reason="sh-скрипты")
def test_git_status_does_not_run_fsmonitor_hook(git_repo):
    """Подменённый .git/config не запускает команду на `git status` (SAFE, без вопроса)."""
    root = git_repo.root
    hook, marker = _marker_script(root, "fsmonitor")
    subprocess.run(["git", "config", "core.fsmonitor", str(hook)], cwd=root, check=True)
    subprocess.run(["git", "status"], cwd=root, capture_output=True)
    if not marker.exists():
        pytest.skip("эта версия git не запускает fsmonitor-хук")
    marker.unlink()
    g = GitTool()
    g.run(g.parse({"subcommand": "status"}), git_repo)
    assert not marker.exists()


@pytest.mark.parametrize(
    "sub, args", [("diff", []), ("log", ["-p"]), ("show", []), ("status", []), ("stash", ["list"])]
)
def test_git_reads_with_textconv_in_repo_config_are_not_safe(stashed_repo, sub, args):
    """textconv из .git/config запускается на чтении: оно уже не SAFE (и в репозитории)."""
    g = GitTool()
    params = g.parse({"subcommand": sub, "args": args})
    assert g.risk(params, stashed_repo) is RiskLevel.SAFE
    _git(stashed_repo, "config", "diff.evil.textconv", "sh -c id")
    assert g.risk(params, stashed_repo) is RiskLevel.WRITE


def test_git_reads_are_not_safe_when_repo_config_is_unknown(stashed_repo, monkeypatch):
    import gigacrew.tools.git as git_module

    monkeypatch.setattr(git_module, "repo_config_runs_programs", lambda root: None)
    g = GitTool()
    assert g.risk(g.parse({"subcommand": "diff"}), stashed_repo) is RiskLevel.WRITE


@pytest.mark.parametrize(
    "args, risk",
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
def test_git_branch_only_listing_flags_are_safe(stashed_repo, args, risk):
    """Флаги без позиционных аргументов: изменяющие ветки — не список, даже без имени."""
    g = GitTool()
    assert g.risk(g.parse({"subcommand": "branch", "args": args}), stashed_repo) is risk


def test_git_tool_is_not_an_edit():
    """Правки файлов в «авто» идут без вопроса; git commit/restore — не правка файла."""
    from gigacrew.permissions import ToolKind

    assert GitTool.kind is ToolKind.OTHER
    assert decide(PermissionMode.AUTO, RiskLevel.WRITE, GitTool.kind, auto_approve=False) is (
        Decision.ASK
    )


def test_git_commit_in_auto_mode_is_confirmed(stashed_repo):
    from fakes import RecordingEvents, ScriptedProvider, text_turn, tool_turn

    from gigacrew.agent.loop import Agent
    from gigacrew.config import Config
    from gigacrew.tools.base import build_default_registry

    root = stashed_repo.root
    _git(stashed_repo, "add", "f.txt")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True).stdout
    provider = ScriptedProvider(
        [tool_turn("git", {"subcommand": "commit", "args": ["-m", "x"]}), text_turn("ок")]
    )
    events = RecordingEvents(confirm_answer=False)
    cfg = Config(access_key="x", project_root=root, stream=False, auto_approve=False)
    Agent(provider, build_default_registry(), cfg, events).run_turn("закоммить")
    assert len(events.confirms) == 1 and events.confirms[0][2] is False
    after = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True).stdout
    assert after == head


@pytest.mark.skipif(sys.platform == "win32", reason="sh-скрипты")
def test_git_diff_does_not_run_external_diff(git_repo):
    root = git_repo.root
    _write(git_repo, "f.txt", "one\n")
    subprocess.run(["git", "add", "f.txt"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=root, check=True)
    _write(git_repo, "f.txt", "two\n")
    ext, marker = _marker_script(root, "extdiff")
    subprocess.run(["git", "config", "diff.external", str(ext)], cwd=root, check=True)
    subprocess.run(["git", "diff"], cwd=root, capture_output=True)
    assert marker.exists()  # голый git внешнюю программу запускает
    marker.unlink()
    g = GitTool()
    res = g.run(g.parse({"subcommand": "diff"}), git_repo)
    assert not marker.exists()
    assert "-one" in res.content and "+two" in res.content


def _project_with_submodule(base, *, filter_program=None):
    """Проект с подмодулем libs/lib; программа фильтра — только в настройках подмодуля."""
    from gigacrew.project.workspace import Workspace
    from gigacrew.tools.base import ToolContext

    def git(cwd, *args):
        ident = ["-c", "user.email=t@t.t", "-c", "user.name=t", "-c", "protocol.file.allow=always"]
        subprocess.run(["git", *ident, *args], cwd=cwd, check=True, capture_output=True)

    lib = base / "lib"
    lib.mkdir()
    git(lib, "init", "-q")
    (lib / "data.txt").write_text("v1\n", encoding="utf-8")
    (lib / ".gitattributes").write_text("*.txt filter=x\n", encoding="utf-8")
    git(lib, "add", ".")
    git(lib, "commit", "-qm", "i")
    root = base / "proj"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "submodule", "add", "-q", str(lib), "libs/lib")
    git(root, "commit", "-qm", "s")
    if filter_program is not None:
        git(root / "libs" / "lib", "config", "filter.x.clean", str(filter_program))
    (root / "libs" / "lib" / "data.txt").write_text("v2\n", encoding="utf-8")
    return ToolContext(workspace=Workspace(root))


@pytest.mark.skipif(sys.platform == "win32", reason="sh-скрипты")
def test_git_status_with_program_in_submodule_config_is_not_safe(tmp_path):
    """`git status` заходит в подмодули и запускает фильтры из .git/modules/<имя>/config."""
    from gigacrew.safety.gitconfig import repo_config_runs_programs

    script, marker = _marker_script(tmp_path, "clean")
    ctx = _project_with_submodule(tmp_path, filter_program=script)
    subprocess.run(["git", "status"], cwd=ctx.root, check=True, capture_output=True)
    assert marker.exists()  # голый git программу из настроек подмодуля запускает
    assert repo_config_runs_programs(ctx.root) is True
    assert repo_config_runs_programs(ctx.root / "libs") is True  # и из подкаталога
    g = GitTool()
    assert g.risk(g.parse({"subcommand": "status"}), ctx) is RiskLevel.WRITE
    sh = RunShellTool()  # тот же признак у классификатора run_shell
    assert sh.risk(sh.parse({"command": "git status"}), ctx) is not RiskLevel.SAFE


def test_git_status_with_clean_submodule_stays_safe(tmp_path):
    from gigacrew.safety.gitconfig import repo_config_runs_programs

    ctx = _project_with_submodule(tmp_path)
    assert repo_config_runs_programs(ctx.root) is False
    g = GitTool()
    assert g.risk(g.parse({"subcommand": "status"}), ctx) is RiskLevel.SAFE


@pytest.mark.skipif(sys.platform == "win32", reason="sh-скрипты")
def test_git_ignores_implicit_bare_repository(ctx):
    """HEAD/config/objects обычными файлами в проекте не делают его bare-репозиторием."""
    if subprocess.run(["git", "--version"], capture_output=True).returncode:
        pytest.skip("нет git")
    bare = ctx.root / "planted"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    hook, marker = _marker_script(ctx.root, "planted")
    subprocess.run(["git", "config", "core.fsmonitor", str(hook)], cwd=bare, check=True)
    res = subprocess.run(
        ["git", "-c", "safe.bareRepository=explicit", "status"], cwd=bare, capture_output=True
    )
    if res.returncode == 0:
        pytest.skip("эта версия git не знает safe.bareRepository")
    from gigacrew.project.workspace import Workspace
    from gigacrew.tools.base import ToolContext

    g = GitTool()
    out = g.run(g.parse({"subcommand": "log"}), ToolContext(workspace=Workspace(bare)))
    assert not out.ok and "safe.bareRepository" in out.content and not marker.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="FIFO")
def test_search_skips_fifo(ctx):
    import os

    os.mkfifo(ctx.root / "pipe")
    _write(ctx, "a.txt", "needle")
    s = SearchContentTool()
    out = s.run(s.parse({"pattern": "needle"}), ctx).content
    assert "a.txt" in out


def test_search_skips_symlinks_to_secrets(tmp_path):
    import os

    import pytest as _pytest

    from gigacrew.project.workspace import Workspace
    from gigacrew.tools.base import ToolContext
    from gigacrew.tools.search import SearchContentParams, SearchContentTool

    if os.name == "nt":
        _pytest.skip("симлинки")
    (tmp_path / ".env").write_text("TOKEN=s3cr3t\n", encoding="utf-8")
    (tmp_path / "config.txt").symlink_to(tmp_path / ".env")
    ctx = ToolContext(workspace=Workspace(tmp_path))
    result = SearchContentTool().run(SearchContentParams(pattern="s3cr3t"), ctx)
    assert "s3cr3t" not in result.content


def test_git_describe_quotes_arguments(ctx):
    g = GitTool()
    params = g.parse({"subcommand": "commit", "args": ["-m", "fix bug"]})
    assert g.describe(params) == "commit -m 'fix bug'"
    assert g.preview(params, ctx).text == "$ git commit -m 'fix bug'"


def test_search_limit_note_only_when_more_matches(ctx):
    _write(ctx, "a.txt", "hit\n" * 3)
    s = SearchContentTool()
    exact = s.run(s.parse({"pattern": "hit", "max_results": 3}), ctx).content
    assert "показаны первые" not in exact and exact.count("a.txt:") == 3
    more = s.run(s.parse({"pattern": "hit", "max_results": 2}), ctx).content
    assert more.count("a.txt:") == 2 and "показаны первые 2 совпадений" in more


def test_search_line_numbers_match_read_file(ctx):
    _write(ctx, "f.txt", "a\x0cb\nneedle\n")
    s = SearchContentTool()
    assert s.run(s.parse({"pattern": "needle"}), ctx).content == "f.txt:2:needle"
