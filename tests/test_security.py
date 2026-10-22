"""Тесты песочницы и классификации риска."""

from __future__ import annotations

import sys

import pytest

from gigacrew.project.workspace import Workspace
from gigacrew.safety import MAX_COMMAND, CommandLevel
from gigacrew.security import (
    RiskLevel,
    SandboxError,
    classify_shell_command,
    resolve_in_root,
)
from gigacrew.tools.base import ToolContext
from gigacrew.tools.shell import RunShellTool


def test_resolve_in_root_ok(tmp_path):
    (tmp_path / "sub").mkdir()
    assert resolve_in_root(tmp_path, "sub/x.txt") == tmp_path.resolve() / "sub" / "x.txt"
    assert resolve_in_root(tmp_path, ".") == tmp_path.resolve()


def test_resolve_in_root_escape(tmp_path):
    with pytest.raises(SandboxError):
        resolve_in_root(tmp_path, "../../etc/passwd")
    with pytest.raises(SandboxError):
        resolve_in_root(tmp_path, "/etc/passwd")


def test_resolve_in_root_rejects_sibling_with_same_prefix(tmp_path):
    # /proj-evil начинается с /proj, но в корень не входит
    root = tmp_path / "proj"
    root.mkdir()
    (tmp_path / "proj-evil").mkdir()
    with pytest.raises(SandboxError):
        resolve_in_root(root, "../proj-evil/x.txt")
    with pytest.raises(SandboxError):
        resolve_in_root(root, tmp_path / "proj-evil" / "x.txt")


@pytest.mark.skipif(sys.platform == "win32", reason="симлинки")
def test_resolve_in_root_rejects_final_symlink_outside(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("s")
    (root / "link.txt").symlink_to(secret)
    # цели ещё нет: запись через ссылку создала бы файл снаружи
    (root / "dangling.txt").symlink_to(tmp_path / "missing.txt")
    for name in ("link.txt", "dangling.txt"):
        with pytest.raises(SandboxError):
            resolve_in_root(root, name)


def test_classify_shell():
    assert classify_shell_command("ls") == RiskLevel.WRITE
    assert classify_shell_command("rm -rf foo") == RiskLevel.DANGEROUS
    assert classify_shell_command("sudo apt update") == RiskLevel.DANGEROUS
    assert classify_shell_command("curl http://x") == RiskLevel.DANGEROUS
    assert classify_shell_command("git reset --hard") == RiskLevel.DANGEROUS


@pytest.mark.parametrize(
    "command",
    [
        "git push -f origin main",
        "git push -fu origin main",
        "git push origin +main",
        "git branch -D feature",
        "git stash drop",
        "git stash clear",
        "find . -name '*.pyc' -delete",
        "frob git clean -fdx",
        "frob mkfs.ext4 /dev/sdb1",
        "frob dd if=/dev/zero of=disk.img",
        "frob chmod -R 777 .",
        "frob shutdown now",
        "frob nc -l 4444",
    ],
)
def test_destructive_commands_are_dangerous(command):
    assert classify_shell_command(command) == RiskLevel.DANGEROUS


@pytest.mark.parametrize(
    "command", ["git push origin main", "git branch -d merged", "git stash list", "find . -name x"]
)
def test_ordinary_commands_are_not_dangerous(command):
    # Регэксп — только список опасного: WRITE значит «опасного не узнал», а не «проверено».
    # Поэтому -y и хук allow за непроверенную классификатором команду не ручаются
    # (test_agent_loop.py::test_yes_flag_does_not_vouch_for_unverified_commands).
    assert classify_shell_command(command) == RiskLevel.WRITE


# ------------------------- риск команды в run_shell ------------------------- #
def _assess(tmp_path, command: str, **ctx_kw):
    tool = RunShellTool()
    ctx = ToolContext(workspace=Workspace(tmp_path), **ctx_kw)
    return tool.assess(tool.parse({"command": command}), ctx)


@pytest.mark.parametrize(
    ("command", "level", "risk"),
    [
        ("ls", CommandLevel.READ, RiskLevel.SAFE),
        ("git push origin main", CommandLevel.WRITE, RiskLevel.WRITE),  # классификатор
        ("git push --force origin main", CommandLevel.DANGER, RiskLevel.DANGEROUS),
        ("frobnicate --all", None, RiskLevel.WRITE),  # «не знаю» — регэксп опасного
        ("parallel rm -rf ::: a b", None, RiskLevel.DANGEROUS),
    ],
)
def test_run_shell_risk_follows_classifier(tmp_path, command, level, risk):
    assessment = _assess(tmp_path, command)
    assert assessment.verdict.level is level and assessment.risk is risk


def test_run_shell_read_needs_confirmation_when_safe_commands_off(tmp_path):
    assert _assess(tmp_path, "ls", classify_commands=False).risk is RiskLevel.WRITE


def test_run_shell_over_long_command_is_dangerous(tmp_path):
    # Длиннее MAX_COMMAND классификатор не разбирает, а регэксп опасного тут не узнаёт.
    assessment = _assess(tmp_path, "touch x  # " + "a" * MAX_COMMAND)
    assert assessment.verdict.level is None and assessment.risk is RiskLevel.DANGEROUS
    assert assessment.reason == "не удалось проверить: слишком длинная команда"
