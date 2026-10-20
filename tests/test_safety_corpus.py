"""Классификатор на размеченных корпусах: ни одна изменяющая команда — не «только чтение».

Метки — по смыслу команды (tests/safety_corpus.py). Главное свойство — безопасность:
команда с меткой выше «только чтение» не выполняется без вопроса. Полнота (сколько
читающих команд проходит без вопроса) проверяется порогом — чтобы правка правил не
вернула «спрашивать обо всём».
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from safety_corpus import CORPUS, HOLDOUT, load

from gigacrew.safety import CommandLevel, classify

NAMES = ["R", "X", "W", "D"]


@pytest.fixture(scope="module")
def project(tmp_path_factory) -> Path:
    """Проект с файлами, о которых говорят команды корпуса."""
    root = tmp_path_factory.mktemp("corpus")
    for d in ("src", "tests", "gigacrew", "docs", "build", "logs", "scripts"):
        (root / d).mkdir()
    for f in (
        "README.md",
        "pyproject.toml",
        "data.csv",
        "a.txt",
        "b.txt",
        "src/main.py",
        "gigacrew/__init__.py",
        "gigacrew/config.py",
        "tests/test_security.py",
        "logs/app.log",
        "setup.cfg",
        "Makefile",
        "data.tsv",
        "tools.py",
    ):
        (root / f).write_text("x = 1\n")
    (root / "main.py").write_text("import gigacrew\nprint(gigacrew)\n")  # код проекта
    (root / "package.json").write_text(
        json.dumps(
            {
                "scripts": {
                    "test": "jest",
                    "build": "tsc",
                    "lint": "eslint .",
                    "deploy": "x",
                    "format": "prettier --write .",
                    "lint:fix": "eslint --fix .",
                }
            }
        )
    )
    return root


@pytest.mark.parametrize("name, raw", [("corpus", CORPUS), ("holdout", HOLDOUT)])
def test_changing_commands_are_never_read_only(project, name, raw):
    wrong = []
    for command, label in load(raw):
        verdict = classify(command, root=project)
        if label > 0 and verdict.level is CommandLevel.READ:
            wrong.append((NAMES[label], command, verdict.reason))
    assert wrong == []


@pytest.mark.parametrize("name, raw", [("corpus", CORPUS), ("holdout", HOLDOUT)])
def test_dangerous_commands_are_not_exec(project, name, raw):
    """Опасное не попадает в «выполняет код проекта» (его разрешают «всегда»)."""
    wrong = []
    for command, label in load(raw):
        verdict = classify(command, root=project)
        if label == 3 and verdict.level in (CommandLevel.READ, CommandLevel.EXEC):
            wrong.append((command, verdict.reason))
    assert wrong == []


@pytest.mark.parametrize(
    "name, raw, minimum", [("corpus", CORPUS, 0.85), ("holdout", HOLDOUT, 0.6)]
)
def test_read_only_recall(project, name, raw, minimum):
    """Читающие команды в основном проходят без вопроса (иначе классификатор бесполезен)."""
    reads = [c for c, label in load(raw) if label == 0]
    passed = [c for c in reads if classify(c, root=project).level is CommandLevel.READ]
    assert len(passed) / len(reads) >= minimum
