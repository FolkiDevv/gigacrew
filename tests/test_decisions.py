"""Память решений о командах: разрешения «всегда», журнал, вырезание секретов."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from gigacrew.agent.decisions import DecisionMemory, project_key, redact
from gigacrew.safety import classify


@pytest.fixture
def root(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    (project / "frontend").mkdir(parents=True)
    (project / "package.json").write_text('{"scripts": {"test": "jest", "build": "tsc"}}')
    (project / "frontend" / "package.json").write_text('{"scripts": {"test": "jest"}}')
    script = project / "build.sh"
    script.write_text("#!/bin/sh\necho ok\n")
    script.chmod(0o755)
    return project


@pytest.fixture
def home(tmp_path: Path) -> Path:
    return tmp_path / "home"


def memory(root: Path, home: Path, **kw) -> DecisionMemory:
    loaded, warning = DecisionMemory.load(root, base=home, **kw)
    assert warning is None
    return loaded


def verdict(command: str, root: Path, cwd: str = "."):
    return classify(command, root=root, cwd=cwd)


def test_program_rule_matches_wrapped_variants(root, home):
    mem = memory(root, home)
    assert "pytest" in mem.offer(verdict("pytest -q", root))
    [rule] = mem.approve(verdict("pytest -q", root))
    assert rule.kind == "program" and rule.value == "pytest"
    for command in (
        "pytest",
        "pytest -x tests",
        "pytest -q 2>&1 | tail -30",
        "python -m pytest -k smoke",
        "uv run pytest -q",
        "pytest; echo done",
    ):
        assert mem.match(verdict(command, root)) is not None, command
    for command in (
        "pytest -p evil",  # флаг вне проверенного набора — только точная команда
        "pytest > log.txt",  # запись в файл
        "pytest && rm -rf build",  # опасная часть
        "pytest && npm test",  # другая программа
    ):
        assert mem.match(verdict(command, root)) is None, command


def test_rule_is_bound_to_working_directory(root, home):
    mem = memory(root, home)
    mem.approve(verdict("npm test", root, cwd="frontend"))
    assert mem.match(verdict("cd frontend && npm test", root)) is not None
    assert mem.match(verdict("npm test", root, cwd="frontend")) is not None
    assert mem.match(verdict("npm test", root)) is None  # корень — другой каталог


def test_exact_rules_and_stdin_identity(root, home):
    mem = memory(root, home)
    mem.approve(verdict("./build.sh --release", root))
    assert mem.match(verdict("./build.sh --release", root)) is not None
    assert mem.match(verdict("./build.sh --debug", root)) is None
    heredoc = "python3 - <<'EOF'\nimport json\nprint(json.loads(input()))\nEOF"
    mem.approve(verdict(heredoc, root))
    assert mem.match(verdict(heredoc, root)) is not None
    other = "python3 - <<'EOF'\nimport json\nprint(json.loads(open(input()).read()))\nEOF"
    assert mem.match(verdict(other, root)) is None  # другой код — другая команда


def test_dangerous_is_never_remembered_or_matched(root, home):
    mem = memory(root, home)
    danger = verdict("rm -rf build", root)
    assert mem.offer(danger) == "" and mem.approve(danger) == []
    mem.rules.append(mem._candidates(verdict("touch x", root))[0])
    assert mem.match(verdict("touch x && rm -rf build", root)) is None


def test_plan_mode_applies_only_non_changing_rules(root, home):
    mem = memory(root, home)
    mem.approve(verdict("pytest", root))
    mem.approve(verdict("touch x", root))
    assert mem.match(verdict("pytest", root), plan=True) is not None
    assert mem.match(verdict("touch x", root), plan=True) is None
    assert mem.match(verdict("touch x", root)) is not None


def test_rules_persist_and_merge_between_windows(root, home):
    first = memory(root, home)
    second = memory(root, home)
    first.approve(verdict("pytest", root))
    second.approve(verdict("npm test", root))  # другое окно — перечитывает перед записью
    third = memory(root, home)
    assert {r.value for r in third.rules} == {"pytest", "npm test"}
    path = third.approvals_path
    assert path is not None and stat.S_IMODE(path.stat().st_mode) == 0o600
    third.forget(1)
    assert len(memory(root, home).rules) == 1
    assert third.clear() == 1 and memory(root, home).rules == []


def test_remember_off_keeps_rules_for_session_only(root, home):
    mem = memory(root, home, remember=False)
    mem.approve(verdict("pytest", root))
    assert mem.match(verdict("pytest", root)) is not None
    assert memory(root, home).rules == []
    assert "до конца сессии" in mem.offer(verdict("npm test", root))


def test_rules_of_another_project_are_ignored(root, home, tmp_path):
    mem = memory(root, home)
    mem.approve(verdict("pytest", root))
    data = json.loads(mem.approvals_path.read_text())
    data["root"] = "/elsewhere"
    mem.approvals_path.write_text(json.dumps(data))
    loaded, warning = DecisionMemory.load(root, base=home)
    assert loaded.rules == [] and warning is not None


def test_journal_records_choice_context_and_outcome(root, home):
    mem = memory(root, home)
    v = verdict("pytest -q", root)
    entry = mem.record(
        {
            "tool": "run_shell",
            "command": "GITHUB_TOKEN=abc pytest -q",
            "cwd": v.cwd,
            "key": "pytest",
            "decision": "no",
            "source": "user",
            "feedback": "используй uv run",
        }
    )
    mem.outcome(entry, ok=False, interrupted=True)
    mem.record(
        {"command": "pytest", "cwd": ".", "key": "pytest", "decision": "yes", "source": "user"}
    )
    lines = [json.loads(x) for x in mem.journal_path.read_text().splitlines()]
    assert lines[0]["command"] == "GITHUB_TOKEN=*** pytest -q"
    assert lines[0]["feedback"] == "используй uv run" and lines[0]["rules_version"] >= 1
    assert lines[1] == {**lines[1], "id": entry, "event": "outcome", "interrupted": True}
    assert stat.S_IMODE(os.stat(mem.journal_path).st_mode) == 0o600
    assert mem.history((".", "pytest")) == (1, 1)
    assert memory(root, home).history((".", "pytest")) == (1, 1)  # из журнала
    assert [x["decision"] for x in mem.recent()] == ["no", "yes"]


def test_journal_off(root, home):
    mem = memory(root, home, log=False)
    mem.record({"command": "ls", "cwd": ".", "key": "ls", "decision": "yes", "source": "user"})
    assert not mem.journal_path.exists()


def test_session_rejections_and_reviewer_entries(root, home):
    mem = memory(root, home)
    entry = {"command": "make", "cwd": ".", "key": "make"}
    mem.record({**entry, "decision": "auto", "source": "reviewer", "review": {"reason": "ок"}})
    assert not mem.rejected_in_session((".", "make"))
    mem.record({**entry, "decision": "no", "source": "user"})
    assert mem.rejected_in_session((".", "make")) and not mem.rejected_in_session(("src", "make"))
    assert [x["source"] for x in mem.recent()] == ["reviewer", "user"]
    # в новой сессии отказ — уже история, а не «отклонено в этой сессии»
    again = memory(root, home)
    assert not again.rejected_in_session((".", "make")) and again.history((".", "make")) == (0, 1)


def test_journal_key_is_redacted(root, home):
    mem = memory(root, home)
    command = "API_KEY=secret123 make deploy"
    mem.record({"command": command, "cwd": ".", "key": command, "decision": "no", "source": "user"})
    assert "secret123" not in mem.journal_path.read_text()
    mem.record(
        {
            "command": "x",
            "key": "x",
            "verdict": {"reason": "API_KEY=secret456 в аргументах"},
            "review": {"reason": "передаёт API_KEY=secret789"},
        }
    )
    text = mem.journal_path.read_text()
    assert "secret456" not in text and "secret789" not in text


def test_in_memory_only():
    mem = DecisionMemory()
    assert mem.approvals_path is None and mem.journal_path is None
    mem.record({"command": "ls", "decision": "yes", "source": "user", "cwd": ".", "key": "ls"})


@pytest.mark.parametrize(
    "text, expected",
    [
        ("API_KEY=secret123 make deploy", "API_KEY=*** make deploy"),
        ("TOKEN=abc PASSWORD=x ./run", "TOKEN=*** PASSWORD=*** ./run"),
        ("git clone https://user:pa55@host/repo", "git clone https://user:***@host/repo"),
        ("curl -H 'Authorization: Bearer abc.def' x", "curl -H 'Authorization: Bearer ***' x"),
        ("tool --password hunter2 --x", "tool --password ***** --x".replace("*****", "***")),
        ("ls -la", "ls -la"),
    ],
)
def test_redact(text, expected):
    assert redact(text) == expected


def test_project_key_follows_worktree_to_main_checkout(tmp_path):
    main = tmp_path / "main"
    (main / ".git" / "worktrees" / "wt").mkdir(parents=True)
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {main / '.git' / 'worktrees' / 'wt'}\n")
    assert project_key(wt) == project_key(main)
    assert project_key(tmp_path / "other") != project_key(main)
