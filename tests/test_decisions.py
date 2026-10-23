"""Память решений о командах: разрешения «всегда», журнал, вырезание секретов."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from gigacrew.agent.decisions import DecisionMemory, Rule, history_key, project_key, redact
from gigacrew.safety import CommandLevel, Verdict, classify

# Тело поддельного токена GitHub собирается при запуске, чтобы в исходнике не было
# строки, похожей на настоящий токен.
_FAKE_TOKEN_BODY = "a1" * 18


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


def memory(root: Path, home: Path, **kw: bool) -> DecisionMemory:
    loaded, warning = DecisionMemory.load(root, base=home, **kw)
    assert warning is None
    return loaded


def verdict(command: str, root: Path, cwd: str = ".") -> Verdict:
    return classify(command, root=root, cwd=cwd)


def _approvals(mem: DecisionMemory) -> Path:
    path = mem.approvals_path
    assert path is not None
    return path


def _journal(mem: DecisionMemory) -> Path:
    path = mem.journal_path
    assert path is not None
    return path


def test_program_rule_matches_wrapped_variants(root: Path, home: Path) -> None:
    mem = memory(root, home)
    assert "pytest" in mem.offer(verdict("pytest -q", root))
    approved = mem.approve(verdict("pytest -q", root))
    assert len(approved) == 1
    rule = approved[0]
    assert rule.kind == "program"
    assert rule.value == "pytest"
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


def test_rule_is_bound_to_working_directory(root: Path, home: Path) -> None:
    mem = memory(root, home)
    mem.approve(verdict("npm test", root, cwd="frontend"))
    assert mem.match(verdict("cd frontend && npm test", root)) is not None
    assert mem.match(verdict("npm test", root, cwd="frontend")) is not None
    assert mem.match(verdict("npm test", root)) is None  # корень — другой каталог


def test_exact_rules_and_stdin_identity(root: Path, home: Path) -> None:
    mem = memory(root, home)
    mem.approve(verdict("./build.sh --release", root))
    assert mem.match(verdict("./build.sh --release", root)) is not None
    assert mem.match(verdict("./build.sh --debug", root)) is None
    heredoc = "python3 - <<'EOF'\nimport json\nprint(json.loads(input()))\nEOF"
    mem.approve(verdict(heredoc, root))
    assert mem.match(verdict(heredoc, root)) is not None
    other = "python3 - <<'EOF'\nimport json\nprint(json.loads(open(input()).read()))\nEOF"
    assert mem.match(verdict(other, root)) is None  # другой код — другая команда


def test_dangerous_is_never_remembered_or_matched(root: Path, home: Path) -> None:
    mem = memory(root, home)
    danger = verdict("rm -rf build", root)
    assert not mem.offer(danger)
    assert not mem.approve(danger)
    mem.rules.append(mem._candidates(verdict("touch x", root))[0])
    assert mem.match(verdict("touch x && rm -rf build", root)) is None
    # целиком, не по частям (перенаправление): защита — у всей команды
    whole = verdict("rm -rf build > log.txt", root)
    assert whole.level is CommandLevel.DANGER
    assert not whole.composable
    assert not mem.offer(whole)
    assert not mem.approve(whole)
    # устаревшее (прежняя версия правил) или вписанное руками правило не действует
    mem.rules.append(Rule("old", ".", "command", whole.command))
    assert mem.match(whole) is None


def test_plan_mode_ignores_whole_command_rules(root: Path, home: Path) -> None:
    mem = memory(root, home)
    command_verdict = verdict("pytest > log.txt", root)  # пишет в файл
    mem.approve(command_verdict)
    assert mem.match(command_verdict) is not None
    assert mem.match(command_verdict, plan=True) is None


def test_venv_activation_needs_an_approved_program(root: Path, home: Path) -> None:
    _venv(root)
    mem = memory(root, home)
    mem.approve(verdict("touch x", root))
    command_verdict = verdict("source .venv/bin/activate && touch x", root)
    assert command_verdict.parts[0].prelude
    assert mem.match(command_verdict) is None  # скрипт активации — код проекта: только с программой
    mem.approve(verdict("pytest", root))
    assert mem.match(verdict("source .venv/bin/activate && pytest", root)) is not None


def test_always_counts_as_allowed(root: Path, home: Path) -> None:
    mem = memory(root, home)
    mem.record(
        {"command": "make", "cwd": ".", "key": "make", "source": "user", "decision": "always"},
    )
    assert mem.history((".", "make")) == (1, 0)
    assert memory(root, home).history((".", "make")) == (1, 0)  # и из журнала


def test_history_key_is_the_program_or_whole_command(root: Path) -> None:
    assert history_key(verdict("pytest -q", root)) == (".", "pytest")
    assert history_key(verdict("pytest && make", root)) == (".", "pytest && make")
    assert history_key(verdict("npm test", root, cwd="frontend")) == ("frontend", "npm test")


def test_forget_rejects_numbers_outside_the_list(root: Path, home: Path) -> None:
    mem = memory(root, home)
    mem.approve(verdict("pytest", root))
    for index in (0, -1, 2):
        with pytest.raises(IndexError):
            mem.forget(index)
    assert len(mem.rules) == 1


def _venv(root: Path) -> None:
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / ".venv" / "bin" / "activate").write_text("")
    (root / ".venv" / "pyvenv.cfg").write_text("home = /usr/bin\n")


@pytest.mark.parametrize(
    "command",
    [
        "frobnicate > out.txt",  # непроверенная и по частям не сверить
        "if pytest; then echo ok; fi",
        "source .venv/bin/activate && touch x",  # активация venv — только с программой
        ". .venv/bin/activate && ./build.sh --release",
        " && ".join(f"touch {name}" for name in "abcdef"),  # частей больше предела
    ],
)
def test_always_does_not_save_rules_that_never_match(root: Path, home: Path, command: str) -> None:
    # Такое правило match() не применит: «всегда» — только до конца сессии (в агенте).
    _venv(root)
    mem = memory(root, home)
    command_verdict = verdict(command, root)
    assert not mem.offer(command_verdict)
    assert not mem.approve(command_verdict)
    assert not mem.rules
    assert mem.approvals_path is not None
    assert not _approvals(mem).exists()


@pytest.mark.parametrize(
    "command",
    [
        "pytest -q",
        "pytest > log.txt",
        "./build.sh --release",
        "frobnicate --all",
        "source .venv/bin/activate && pytest -q",
        "touch a && touch b",
        "LANG=C ./build.sh",
    ],
)
def test_offered_rule_matches_the_same_command(root: Path, home: Path, command: str) -> None:
    _venv(root)
    mem = memory(root, home)
    command_verdict = verdict(command, root)
    assert mem.offer(command_verdict)
    assert mem.approve(command_verdict)
    assert mem.match(command_verdict) is not None
    assert memory(root, home).match(command_verdict) is not None  # и после перечитывания с диска


def test_rules_are_bound_to_working_directory_for_every_kind(root: Path, home: Path) -> None:
    script = root / "frontend" / "build.sh"  # другой скрипт с тем же именем
    script.write_text("#!/bin/sh\necho frontend\n")
    script.chmod(0o755)
    mem = memory(root, home)
    for command in ("./build.sh --release", "pytest > log.txt"):  # точная и целиком
        mem.approve(verdict(command, root))
        assert mem.match(verdict(command, root)) is not None, command
        assert mem.match(verdict(command, root, cwd="frontend")) is None, command
        assert mem.match(verdict(f"cd frontend && {command}", root)) is None, command


def test_plan_mode_applies_only_non_changing_rules(root: Path, home: Path) -> None:
    mem = memory(root, home)
    mem.approve(verdict("pytest", root))
    mem.approve(verdict("touch x", root))
    assert mem.match(verdict("pytest", root), plan=True) is not None
    assert mem.match(verdict("touch x", root), plan=True) is None
    assert mem.match(verdict("touch x", root)) is not None


def test_rules_persist_and_merge_between_windows(root: Path, home: Path) -> None:
    first = memory(root, home)
    second = memory(root, home)
    first.approve(verdict("pytest", root))
    second.approve(verdict("npm test", root))  # другое окно — перечитывает перед записью
    third = memory(root, home)
    assert {rule.value for rule in third.rules} == {"pytest", "npm test"}
    path = third.approvals_path
    assert path is not None
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    third.forget(1)
    assert len(memory(root, home).rules) == 1
    assert third.clear() == 1
    assert not memory(root, home).rules


def test_remember_off_keeps_rules_for_session_only(root: Path, home: Path) -> None:
    mem = memory(root, home, remember=False)
    mem.approve(verdict("pytest", root))
    assert mem.match(verdict("pytest", root)) is not None
    assert not memory(root, home).rules
    assert "до конца сессии" in mem.offer(verdict("npm test", root))


def test_rules_of_another_project_are_ignored(root: Path, home: Path, tmp_path: Path) -> None:
    mem = memory(root, home)
    mem.approve(verdict("pytest", root))
    stored = json.loads(_approvals(mem).read_text())
    stored["root"] = "/elsewhere"
    _approvals(mem).write_text(json.dumps(stored))
    loaded, warning = DecisionMemory.load(root, base=home)
    assert not loaded.rules
    assert warning is not None


def test_journal_records_choice_context_and_outcome(root: Path, home: Path) -> None:
    mem = memory(root, home)
    command_verdict = verdict("pytest -q", root)
    entry = mem.record(
        {
            "tool": "run_shell",
            "command": "GITHUB_TOKEN=abc pytest -q",
            "cwd": command_verdict.cwd,
            "key": "pytest",
            "decision": "no",
            "source": "user",
            "feedback": "используй uv run",
        },
    )
    mem.outcome(entry, ok=False, interrupted=True)
    mem.record(
        {"command": "pytest", "cwd": ".", "key": "pytest", "decision": "yes", "source": "user"},
    )
    lines = [json.loads(raw) for raw in _journal(mem).read_text().splitlines()]
    assert lines[0]["command"] == "GITHUB_TOKEN=*** pytest -q"
    assert lines[0]["feedback"] == "используй uv run"
    assert lines[0]["rules_version"] >= 1
    assert lines[1] == {**lines[1], "id": entry, "event": "outcome", "interrupted": True}
    assert stat.S_IMODE(_journal(mem).stat().st_mode) == 0o600
    assert mem.history((".", "pytest")) == (1, 1)
    assert memory(root, home).history((".", "pytest")) == (1, 1)  # из журнала
    assert [logged["decision"] for logged in mem.recent()] == ["no", "yes"]


def test_journal_off(root: Path, home: Path) -> None:
    mem = memory(root, home, log=False)
    mem.record({"command": "ls", "cwd": ".", "key": "ls", "decision": "yes", "source": "user"})
    assert not _journal(mem).exists()


def test_session_rejections_and_reviewer_entries(root: Path, home: Path) -> None:
    mem = memory(root, home)
    entry = {"command": "make", "cwd": ".", "key": "make"}
    mem.record({**entry, "decision": "auto", "source": "reviewer", "review": {"reason": "ок"}})
    assert not mem.rejected_in_session((".", "make"))
    mem.record({**entry, "decision": "no", "source": "user"})
    assert mem.rejected_in_session((".", "make"))
    assert not mem.rejected_in_session(("src", "make"))
    assert [logged["source"] for logged in mem.recent()] == ["reviewer", "user"]
    # в новой сессии отказ — уже история, а не «отклонено в этой сессии»
    again = memory(root, home)
    assert not again.rejected_in_session((".", "make"))
    assert again.history((".", "make")) == (0, 1)


def test_journal_key_is_redacted(root: Path, home: Path) -> None:
    mem = memory(root, home)
    command = "API_KEY=secret123 make deploy"
    mem.record({"command": command, "cwd": ".", "key": command, "decision": "no", "source": "user"})
    assert "secret123" not in _journal(mem).read_text()
    mem.record(
        {
            "command": "x",
            "key": "x",
            "verdict": {"reason": "API_KEY=secret456 в аргументах"},
            "review": {"reason": "передаёт API_KEY=secret789"},
        },
    )
    text = _journal(mem).read_text()
    assert "secret456" not in text
    assert "secret789" not in text


def test_in_memory_only() -> None:
    mem = DecisionMemory()
    assert mem.approvals_path is None
    assert mem.journal_path is None
    mem.record({"command": "ls", "decision": "yes", "source": "user", "cwd": ".", "key": "ls"})


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("API_KEY=secret123 make deploy", "API_KEY=*** make deploy"),
        ("TOKEN=abc PASSWORD=x ./run", "TOKEN=*** PASSWORD=*** ./run"),
        ("AWS_ACCESS_KEY_ID=AKIA123 aws s3 ls", "AWS_ACCESS_KEY_ID=*** aws s3 ls"),
        ("git clone https://user:pa55@host/repo", "git clone https://user:***@host/repo"),
        ("curl -H 'Authorization: Bearer abc.def' x", "curl -H 'Authorization: Bearer ***' x"),
        ("tool --password hunter2 --x", "tool --password ***** --x".replace("*****", "***")),
        ("tool --password=hunter2 x", "tool --password=*** x"),
        ("gh --access-token abc x", "gh --access-token *** x"),
        ("x --auth-token=abc --client-secret s", "x --auth-token=*** --client-secret ***"),
        ("mysql -u root -phunter2 -e 'select 1'", "mysql -u root -p*** -e 'select 1'"),
        ("mysqldump -uroot -p'pa ss' db", "mysqldump -uroot -p*** db"),
        ("curl -u bob:hunter2 http://x", "curl -u bob:*** http://x"),
        ("curl --user 'bob:hunter2' http://x", "curl --user 'bob:***' http://x"),
        ("sshpass -p hunter2 ssh host", "sshpass -p *** ssh host"),
        ("curl -H 'X-Api-Key: k123' x", "curl -H 'X-Api-Key: ***' x"),
        ('curl -H "X-Auth-Token:t123" x', 'curl -H "X-Auth-Token:***" x'),
        (
            'curl -d \'{"api_key": "k123", "n": 1}\' x',
            'curl -d \'{"api_key": "***", "n": 1}\' x',
        ),
        ("http POST x token:=abc", "http POST x token:=***"),
        (
            f"echo ghp_{_FAKE_TOKEN_BODY} | gh auth login --with-token",
            "echo *** | gh auth login --with-token",
        ),
        ("ls -la", "ls -la"),
        ("git push -u origin main", "git push -u origin main"),
        ("sort -u keys.txt", "sort -u keys.txt"),
        ("pytest -p no:cacheprovider", "pytest -p no:cacheprovider"),
    ],
)
def test_redact(text: str, expected: str) -> None:
    assert redact(text) == expected


def test_outcome_error_is_redacted(root: Path, home: Path) -> None:
    mem = memory(root, home)
    mem.outcome("id1", ok=False, error="ошибка: TOKEN=abc123 не подошёл")
    assert "abc123" not in _journal(mem).read_text()


def test_feedback_and_description_are_redacted(root: Path, home: Path) -> None:
    mem = memory(root, home)
    mem.record(
        {
            "command": "x",
            "cwd": ".",
            "key": "x",
            "decision": "no",
            "source": "user",
            "feedback": "используй TOKEN=fb_secret",
            "description": "деплой с API_KEY=desc_secret",
        },
    )
    text = _journal(mem).read_text()
    assert "fb_secret" not in text
    assert "desc_secret" not in text
    assert "TOKEN=***" in text
    assert "API_KEY=***" in text


@pytest.mark.parametrize(
    "command",
    ["TOKEN=hunter2 ./build.sh", "./build.sh --password hunter2 > log.txt"],
)
def test_command_with_secret_is_not_saved_as_rule(root: Path, home: Path, command: str) -> None:
    # approvals.json не журнал: секрет из правила не вырезать — правило не сохраняется.
    mem = memory(root, home)
    command_verdict = verdict(command, root)
    assert not mem.offer(command_verdict)
    assert not mem.approve(command_verdict)
    assert not _approvals(mem).exists()


def _worktree(main: Path, wt: Path, name: str = "wt") -> None:
    """Связанное рабочее дерево, как его делает ``git worktree add``."""
    gitdir = main / ".git" / "worktrees" / name
    gitdir.mkdir(parents=True)
    wt.mkdir(parents=True)
    (wt / ".git").write_text(f"gitdir: {gitdir}\n")
    back_link = wt / ".git"
    (gitdir / "gitdir").write_text(f"{back_link}\n")  # обратная ссылка git


def test_project_key_follows_worktree_to_main_checkout(tmp_path: Path) -> None:
    main = tmp_path / "main"
    _worktree(main, tmp_path / "wt")
    assert project_key(tmp_path / "wt") == project_key(main)
    assert project_key(tmp_path / "other") != project_key(main)


def test_worktrees_share_approvals(tmp_path: Path, home: Path) -> None:
    main, wt = tmp_path / "main", tmp_path / "wt"
    _worktree(main, wt)
    memory(main, home).approve(verdict("pytest", main))
    in_wt = memory(wt, home)  # без предупреждения «файл другого проекта»
    assert [rule.value for rule in in_wt.rules] == ["pytest"]
    assert in_wt.match(verdict("pytest", wt)) is not None
    in_wt.approve(verdict("npm test", wt))
    assert {rule.value for rule in memory(main, home).rules} == {"pytest", "npm test"}
    # файл, записанный прежней версией из рабочего дерева (корень — его путь), — тоже свой
    stored = json.loads(_approvals(in_wt).read_text())
    _approvals(in_wt).write_text(json.dumps({**stored, "root": str(wt.resolve())}))
    assert len(memory(main, home).rules) == 2


@pytest.mark.parametrize("variant", ["no_back_link", "back_link_elsewhere", "missing"])
def test_forged_git_file_does_not_borrow_another_projects_memory(
    tmp_path: Path,
    variant: str,
) -> None:
    # Скачанный (не клонированный) проект с файлом .git, указывающим на чужой репозиторий.
    trusted, other = tmp_path / "trusted", tmp_path / "other"
    gitdir = trusted / ".git" / "worktrees" / "x"
    if variant != "missing":
        gitdir.mkdir(parents=True)
    if variant == "back_link_elsewhere":
        elsewhere = tmp_path / "real-wt" / ".git"
        (gitdir / "gitdir").write_text(f"{elsewhere}\n")
    other.mkdir()
    (other / ".git").write_text("gitdir: ../trusted/.git/worktrees/x\n")
    assert project_key(other) != project_key(trusted)
