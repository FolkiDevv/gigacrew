"""Тесты песочницы и классификации риска."""

from __future__ import annotations

import pytest

from gigacrew.security import (
    RiskLevel,
    SandboxError,
    classify_shell_command,
    resolve_in_root,
)


def test_resolve_in_root_ok(tmp_path):
    (tmp_path / "sub").mkdir()
    p = resolve_in_root(tmp_path, "sub/x.txt")
    assert str(p).startswith(str(tmp_path))


def test_resolve_in_root_escape(tmp_path):
    with pytest.raises(SandboxError):
        resolve_in_root(tmp_path, "../../etc/passwd")
    with pytest.raises(SandboxError):
        resolve_in_root(tmp_path, "/etc/passwd")


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
    ],
)
def test_destructive_commands_are_dangerous(command):
    assert classify_shell_command(command) == RiskLevel.DANGEROUS


@pytest.mark.parametrize(
    "command", ["git push origin main", "git branch -d merged", "git stash list", "find . -name x"]
)
def test_ordinary_commands_are_not_dangerous(command):
    assert classify_shell_command(command) == RiskLevel.WRITE
