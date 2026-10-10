"""Архитектурные ограничения: ядро агента не зависит от UI."""

from __future__ import annotations

import ast
from pathlib import Path

import gigacrew

PKG = Path(gigacrew.__file__).parent
FORBIDDEN_FOR_CORE = ("gigacrew.ui", "gigacrew.cli", "rich", "prompt_toolkit")


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_core_does_not_import_ui():
    core = ["agent", "llm", "tools", "project"]
    offenders = []
    for sub in core:
        for path in (PKG / sub).rglob("*.py"):
            for name in _imports(path):
                if name.startswith(FORBIDDEN_FOR_CORE):
                    offenders.append(f"{path.relative_to(PKG)}: {name}")
    assert offenders == []
