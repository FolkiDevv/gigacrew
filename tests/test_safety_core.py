"""Ядро классификатора: разбор оболочки, перенаправления, пути и секреты, вердикт.

Каждая проверка — граница, на которой классификатор должен отказать (не «только
чтение»): то, что в нём не понято до конца, выполняется только после вопроса.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gigacrew.safety import MAX_COMMAND, CommandLevel, classify

READ, EXEC, WRITE, DANGER = (
    CommandLevel.READ,
    CommandLevel.EXEC,
    CommandLevel.WRITE,
    CommandLevel.DANGER,
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    for d in ("src", "tests"):
        (tmp_path / d).mkdir()
    for f in ("README.md", "a.txt", "b.txt", "src/main.py", "tests/test_a.py"):
        (tmp_path / f).write_text("hello\n")
    return tmp_path


def level(command: str, root: Path, **kw) -> CommandLevel | None:
    return classify(command, root=root, **kw).level


# ------------------------- невидимые символы ------------------------- #
@pytest.mark.parametrize("command", ["touch a\u202eb", "git checkout -- .\u200b", "ls\x1b[2K"])
def test_hidden_characters_in_long_command_are_danger(root, command):
    """Длинная команда не разбирается, но невидимые символы в ней — всё равно опасно:
    иначе хвост из 10 000 символов прятал бы подмену текста."""
    padded = command + "  # " + "x" * MAX_COMMAND
    verdict = classify(padded, root=root)
    assert verdict.level is DANGER, verdict.reason
    assert level("echo " + "x" * MAX_COMMAND, root) is None
