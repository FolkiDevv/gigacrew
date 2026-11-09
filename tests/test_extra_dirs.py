"""Подключённые каталоги (``--add-dir``, ``/add-dir``): инструменты, классификатор, агент."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from fakes import RecordingEvents, ScriptedProvider, text_turn, tool_turn

from gigacrew.agent.loop import Agent
from gigacrew.agent.prompts import build_project_context
from gigacrew.agent.reviewer import review_request
from gigacrew.cli.app import main
from gigacrew.cli.commands import CommandContext, default_commands
from gigacrew.config import Config
from gigacrew.errors import ToolError
from gigacrew.permissions import PermissionMode
from gigacrew.project.extra_dirs import check_extra_dir
from gigacrew.project.workspace import Workspace
from gigacrew.safety import CommandLevel, classify
from gigacrew.security import SandboxError, resolve_in_roots
from gigacrew.tools import build_default_registry
from gigacrew.tools.base import ToolContext
from gigacrew.tools.fs import EditFileTool, FindFilesTool, ListDirTool, ReadFileTool, WriteFileTool
from gigacrew.tools.search import SearchContentTool
from gigacrew.tools.shell import RunShellTool, command_verdict
from gigacrew.ui.console import Console

READ, WRITE, DANGER = CommandLevel.READ, CommandLevel.WRITE, CommandLevel.DANGER


@pytest.fixture
def project(tmp_path: Path) -> Path:
    folder = tmp_path / "proj"
    (folder / "src").mkdir(parents=True)
    (folder / "src" / "main.py").write_text("print('proj')\n", encoding="utf-8")
    return folder


@pytest.fixture
def other(tmp_path: Path) -> Path:
    """Другой проект — с решением, которое нужно повторить."""
    folder = tmp_path / "other"
    (folder / "src").mkdir(parents=True)
    (folder / "src" / "solution.py").write_text("def solve():\n    return 42\n", "utf-8")
    (folder / ".gitignore").write_text("build/\n", encoding="utf-8")
    (folder / "build").mkdir()
    (folder / "build" / "solution.py").write_text("def solve(): ...\n", encoding="utf-8")
    return folder


@pytest.fixture
def extra_ctx(project: Path, other: Path) -> ToolContext:
    return ToolContext(workspace=Workspace(project, extra_dirs=(other,)))


def _git_init(folder: Path) -> None:
    subprocess.run(["git", "init", "-q", str(folder)], check=True)


# ------------------------------ модель данных ------------------------------ #
def test_workspace_drops_nested_and_duplicate_dirs(project: Path, other: Path) -> None:
    workspace = Workspace(project, extra_dirs=(other / "src", other, project / "src", other))
    assert workspace.extra_dirs == (other.resolve(),)


def test_resolve_in_roots(project: Path, other: Path) -> None:
    roots = (project, other)
    assert resolve_in_roots(roots, "src/main.py") == (project / "src/main.py").resolve()
    assert resolve_in_roots(roots, "../other/src") == (other / "src").resolve()
    assert resolve_in_roots(roots, other / "src") == (other / "src").resolve()
    with pytest.raises(SandboxError, match="подключённых"):
        resolve_in_roots(roots, "../elsewhere")
    (other / "escape").symlink_to(project.parent)
    with pytest.raises(SandboxError):
        resolve_in_roots(roots, other / "escape" / "x")


def test_check_extra_dir(project: Path, other: Path, tmp_path: Path) -> None:
    assert check_extra_dir(other, project) == other.resolve()
    with pytest.raises(ValueError, match="не найден"):
        check_extra_dir(tmp_path / "missing", project)
    with pytest.raises(ValueError, match="уже в проекте"):
        check_extra_dir(project / "src", project)
    with pytest.raises(ValueError, match="слишком широко"):
        check_extra_dir(tmp_path, project, home=other)
    secrets = tmp_path / ".ssh"
    secrets.mkdir()
    with pytest.raises(ValueError, match="секретами"):
        check_extra_dir(secrets, project)


# ------------------------------ инструменты ------------------------------ #
def test_read_list_find_search_in_extra_dir(extra_ctx: ToolContext, other: Path) -> None:
    reader, lister, finder, searcher = (
        ReadFileTool(),
        ListDirTool(),
        FindFilesTool(),
        SearchContentTool(),
    )
    solution = str(other / "src" / "solution.py")
    assert "return 42" in reader.run(reader.parse({"path": solution}), extra_ctx).content
    relative = reader.parse({"path": "../other/src/solution.py"})
    assert "return 42" in reader.run(relative, extra_ctx).content
    listing = lister.run(lister.parse({"path": str(other / "src")}), extra_ctx)
    assert "solution.py" in listing.content
    found = finder.run(finder.parse({"pattern": "src/*.py", "path": str(other)}), extra_ctx)
    assert found.content == solution  # build/ — в .gitignore другого проекта
    hits = searcher.run(searcher.parse({"pattern": "def solve", "path": str(other)}), extra_ctx)
    assert solution in hits.content
    assert "build" not in hits.content
    # проект — как раньше: пути от корня
    assert finder.run(finder.parse({"pattern": "*.py"}), extra_ctx).content == "src/main.py"


def test_extra_dir_refuses_secrets_and_service_dirs(extra_ctx: ToolContext, other: Path) -> None:
    (other / ".env").write_text("KEY=1\n", encoding="utf-8")
    reader, writer = ReadFileTool(), WriteFileTool()
    with pytest.raises(ToolError, match="секретами"):
        reader.run(reader.parse({"path": str(other / ".env")}), extra_ctx)
    _git_init(other)
    for service in (".git/hooks/pre-commit", ".gigacode/settings.json"):
        with pytest.raises(ToolError, match="недоступн"):
            writer.run(writer.parse({"path": str(other / service), "content": "x"}), extra_ctx)
    with pytest.raises(SandboxError):
        reader.run(reader.parse({"path": str(other.parent / "elsewhere.txt")}), extra_ctx)


def test_write_and_edit_in_extra_dir(extra_ctx: ToolContext, other: Path) -> None:
    writer, editor = WriteFileTool(), EditFileTool()
    target = other / "src" / "new.py"
    writer.run(writer.parse({"path": str(target), "content": "x = 1\n"}), extra_ctx)
    editor.run(
        editor.parse({"path": str(target), "old_string": "x = 1", "new_string": "x = 2"}),
        extra_ctx,
    )
    assert target.read_text(encoding="utf-8") == "x = 2\n"


def test_shell_cwd_in_extra_dir(extra_ctx: ToolContext, other: Path) -> None:
    shell = RunShellTool()
    params = shell.parse({"command": "ls", "cwd": str(other / "src")})
    assert command_verdict(params, extra_ctx).level is READ
    assert "solution.py" in shell.run(params, extra_ctx).content


# ------------------------------ классификатор ------------------------------ #
@pytest.mark.parametrize(
    ("command", "cwd", "expected"),
    [
        ("cat ../other/src/solution.py", ".", READ),
        ("grep -rn solve ../other/src", ".", READ),
        ("ls ../other", ".", READ),
        ("python -c 'print(open(\"../other/src/solution.py\").read())'", ".", READ),
        ("cd ../other && cat src/solution.py", ".", READ),
        ("cat src/solution.py", "../other", READ),
        ("cat ../proj/src/main.py", "../other", READ),
        ("echo x > ../other/src/new.py", ".", WRITE),
        ("echo x > ../other/.git/hooks/pre-commit", ".", DANGER),
        ("echo x > ../other/.gigacode/settings.json", ".", DANGER),
        ("echo x > ../elsewhere.txt", ".", DANGER),
        ("cat ../other/.env", ".", None),
        ("python ../other/src/solution.py", ".", None),
        ("cat ../elsewhere.txt", ".", None),
    ],
)
def test_classifier_with_extra_dir(
    project: Path,
    other: Path,
    command: str,
    cwd: str,
    expected: CommandLevel | None,
) -> None:
    (other / ".env").write_text("KEY=1\n", encoding="utf-8")
    (other / "src" / ".gitkeep").write_text("", encoding="utf-8")
    assert classify(command, root=project, cwd=cwd, extra=(other,)).level is expected


def test_classifier_without_extra_dir_is_unchanged(project: Path, other: Path) -> None:
    assert classify("cat ../other/src/solution.py", root=project).level is None
    assert classify("echo x > ../other/src/new.py", root=project).level is DANGER


def test_git_in_extra_dir_checks_its_config(project: Path, other: Path) -> None:
    _git_init(other)
    extra = (other,)
    assert classify("git -C ../other log", root=project, extra=extra).level is READ
    assert classify("cd ../other && git status", root=project, extra=extra).level is READ
    subprocess.run(
        ["git", "-C", str(other), "config", "core.fsmonitor", "evil"],
        check=True,
    )
    assert classify("git -C ../other status", root=project, extra=extra).level is None
    assert classify("cd ../other && git status", root=project, extra=extra).level is None


def test_reviewer_sees_extra_dirs(project: Path, other: Path) -> None:
    verdict = classify("touch ../other/x", root=project, extra=(other,))
    text = review_request("touch ../other/x", verdict, ["повтори решение"])
    assert str(other.resolve()) in text
    assert "Additional directories" not in review_request(
        "touch x",
        classify("touch x", root=project),
        [],
    )


# ------------------------------ агент и CLI ------------------------------ #
def test_plan_mode_reads_extra_dir_without_asking(project: Path, other: Path) -> None:
    solution = str(other / "src" / "solution.py")
    provider = ScriptedProvider([
        tool_turn("read_file", {"path": solution}),
        tool_turn("run_shell", {"command": f"cat {solution}"}),
        text_turn("готово"),
    ])
    events = RecordingEvents(confirm_answer=False)
    cfg = Config(
        access_key="x",
        project_root=project,
        stream=False,
        mode=PermissionMode.PLAN,
        add_dirs=(other,),
    )
    agent = Agent(provider, build_default_registry(), cfg, events)
    agent.run_turn("повтори решение из другого проекта")
    assert not events.confirms
    outputs = [tool_result for _call, tool_result, _shown in events.tool_results]
    assert len(outputs) == 2
    assert all("return 42" in tool_result.content for tool_result in outputs)


def test_add_dir_command(project: Path, other: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = Config(access_key="x", project_root=project, stream=False)
    ui = Console(no_color=True)
    agent = Agent(ScriptedProvider(), build_default_registry(), cfg, ui)
    ctx = CommandContext(agent=agent, ui=ui, commands=default_commands())
    assert "Additional directories" not in agent.project_context()
    ctx.commands.dispatch("/add-dir", ctx)
    assert "нет" in capsys.readouterr().out
    ctx.commands.dispatch("/add-dir ../other", ctx)
    assert agent.extra_dirs == (other.resolve(),)
    assert str(other.resolve()) in agent.project_context()
    ctx.commands.dispatch("/add-dir ../missing", ctx)
    assert "не найден" in capsys.readouterr().out
    assert agent.extra_dirs == (other.resolve(),)


def test_project_context_lists_extra_dirs(project: Path, other: Path) -> None:
    text = build_project_context(Workspace(project, extra_dirs=(other,)))
    assert f"- {other.resolve()}" in text


def test_cli_rejects_bad_add_dir(
    project: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing"
    code = main(["-C", str(project), "--add-dir", str(missing), "-p", "привет"])
    assert code != 0
    assert "--add-dir: каталог не найден" in capsys.readouterr().out
