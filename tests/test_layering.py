"""Архитектурные ограничения: ядро агента не зависит от UI."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import gigacrew

PKG = Path(gigacrew.__file__).parent
FORBIDDEN_FOR_CORE = ("gigacrew.ui", "gigacrew.cli", "rich", "prompt_toolkit")
# Ядро: подпакеты и модули верхнего уровня, которые они импортируют.
CORE_PACKAGES = ("agent", "llm", "tools", "project", "safety")
CORE_MODULES = ("__init__.py", "config.py", "errors.py", "permissions.py", "security.py")


def _imports(path: Path) -> set[str]:
    """Импортируемые модули; ``from X import y`` — и ``X``, и ``X.y`` (y может быть
    модулем: ``from gigacrew import ui``)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{a.name}" for a in node.names)
    return names


def _core_files() -> list[Path]:
    files = [PKG / name for name in CORE_MODULES]
    for sub in CORE_PACKAGES:
        files += sorted((PKG / sub).rglob("*.py"))
    return files


def _forbidden(name: str) -> bool:
    return any(name == f or name.startswith(f + ".") for f in FORBIDDEN_FOR_CORE)


def test_imports_sees_every_form(tmp_path):
    source = tmp_path / "m.py"
    source.write_text(
        "import rich.console\nfrom gigacrew import ui\nfrom gigacrew.cli.app import main\n"
        "def f():\n    from prompt_toolkit import prompt\n",
        encoding="utf-8",
    )
    names = _imports(source)
    assert {"rich.console", "gigacrew.ui", "gigacrew.cli.app", "prompt_toolkit"} <= names
    assert all(_forbidden(n) for n in ("rich", "gigacrew.ui", "prompt_toolkit.shortcuts"))
    assert not _forbidden("richtext") and not _forbidden("gigacrew.uix")


def test_core_does_not_import_ui():
    offenders = []
    for path in _core_files():
        for name in _imports(path):
            if _forbidden(name):
                offenders.append(f"{path.relative_to(PKG)}: {name}")
    assert offenders == []


def test_core_does_not_load_ui_transitively():
    """Импорт всего ядра не тянет UI и через сторонние модули."""
    modules = [f"gigacrew.{name.removesuffix('.py')}" for name in CORE_MODULES[1:]]
    for sub in CORE_PACKAGES:
        for path in sorted((PKG / sub).rglob("*.py")):
            rel = path.relative_to(PKG.parent).with_suffix("")
            parts = rel.parts[:-1] if rel.name == "__init__" else rel.parts
            modules.append(".".join(parts))
    code = (
        "import importlib, json, sys\n"
        f"for name in {modules!r}:\n"
        "    importlib.import_module(name)\n"
        f"bad = sorted(m for m in sys.modules if m.split('.')[0] in ('rich', 'prompt_toolkit')"
        " or m.startswith(('gigacrew.ui', 'gigacrew.cli')))\n"
        "print(json.dumps(bad))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(PKG.parent)}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )
    assert json.loads(out.stdout) == []


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
