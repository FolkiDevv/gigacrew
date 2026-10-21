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
    core = ["agent", "llm", "tools", "project", "safety"]
    offenders = []
    for sub in core:
        for path in (PKG / sub).rglob("*.py"):
            for name in _imports(path):
                if name.startswith(FORBIDDEN_FOR_CORE):
                    offenders.append(f"{path.relative_to(PKG)}: {name}")
    assert offenders == []


def test_tree_sitter_is_imported_only_by_treesitter_module():
    """API tree-sitter ломается между версиями — он загружается в одном месте."""
    offenders = []
    for path in PKG.rglob("*.py"):
        if path.relative_to(PKG).as_posix() == "project/treesitter.py":
            continue
        for name in _imports(path):
            if name.split(".")[0] in ("tree_sitter", "tree_sitter_bash"):
                offenders.append(f"{path.relative_to(PKG)}: {name}")
    assert offenders == []
