"""Тесты инструментов find_symbol и file_outline."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, NoReturn

import pytest

from gigacrew.project import index as index_mod
from gigacrew.project.workspace import Workspace
from gigacrew.tools import index as tools_index
from gigacrew.tools.base import ToolContext, ToolError, ToolResult, build_default_registry
from gigacrew.tools.index import (
    FileOutlineParams,
    FileOutlineTool,
    FindSymbolParams,
    FindSymbolTool,
)
from gigacrew.tools.navigation import FindReferencesParams, FindReferencesTool


def _make(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    _make(
        tmp_path,
        "app/agent.py",
        "class Agent:\n    def run_turn(self, text):\n        return text\n\n\n"
        "def helper():\n    pass\n",
    )
    _make(tmp_path, "docs/guide.md", "# Гайд\n## Установка\n")
    return ToolContext(workspace=Workspace(tmp_path))


def _find(ctx: ToolContext, **kw: Any) -> ToolResult:
    return FindSymbolTool().run(FindSymbolParams(**kw), ctx)


def _outline(ctx: ToolContext, path: str) -> ToolResult:
    return FileOutlineTool().run(FileOutlineParams(path=path), ctx)


def test_registered_in_default_registry() -> None:
    reg = build_default_registry()
    assert "find_symbol" in reg
    assert "file_outline" in reg


def test_find_symbol_lists_definitions(ctx: ToolContext) -> None:
    found = _find(ctx, query="run_turn")
    assert found.ok
    assert found.content == ("app/agent.py:2-3 method Agent.run_turn — def run_turn(self, text):")
    assert found.summary.startswith("найдено определений: 1; индекс обновлён (файлов: 2)")
    # второй вызов: индекс уже актуален
    assert _find(ctx, query="run_turn").summary == "найдено определений: 1"


def test_find_symbol_sees_fresh_edits(ctx: ToolContext) -> None:
    _find(ctx, query="Agent")
    _make(ctx.root, "app/new.py", "def brand_new():\n    pass\n")
    assert "app/new.py:1-2 function brand_new" in _find(ctx, query="brand").content


def test_find_symbol_filters_and_limit(ctx: ToolContext) -> None:
    assert "helper" in _find(ctx, query="e", kind="function", glob="app/*.py").content
    assert "Гайд" not in _find(ctx, query="е", kind="function").content
    limited = _find(ctx, query="e", max_results=1)
    assert "… показано 1 из" in limited.content


def test_find_symbol_nothing_found_and_empty_query(ctx: ToolContext) -> None:
    assert "search_content" in _find(ctx, query="nope_nothing").content
    with pytest.raises(ToolError):
        _find(ctx, query="   ")


def test_find_symbol_describe() -> None:
    tool = FindSymbolTool()
    assert tool.describe(FindSymbolParams(query="Agent")) == "Agent"
    assert tool.describe(FindSymbolParams(query="Agent", kind="class")) == "Agent (class)"


def test_file_outline_of_file(ctx: ToolContext) -> None:
    outline = _outline(ctx, "app/agent.py")
    assert outline.content.splitlines() == [
        "app/agent.py (python, строк: 7, определений: 3)",
        "1-3  class Agent:",
        "  2-3  def run_turn(self, text):",
        "6-7  def helper():",
    ]
    md = _outline(ctx, "docs/guide.md").content.splitlines()
    assert md[1:] == ["1  # Гайд", "  2  ## Установка"]


def test_file_outline_of_directory(ctx: ToolContext) -> None:
    outline = _outline(ctx, ".")
    assert outline.content.splitlines() == [
        "app/agent.py (python, строк: 7, определений: 3)",
        "docs/guide.md (markdown, строк: 2, определений: 2)",
    ]
    assert outline.summary.startswith(".: файлов 2, определений 5")
    assert _outline(ctx, "app").content.startswith("agent.py (python")


def test_file_outline_groups_large_directories(
    ctx: ToolContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tools_index, "MAX_OUTLINE_FILES", 1)
    _make(ctx.root, "top.py", "def top():\n    pass\n")
    lines = _outline(ctx, ".").content.splitlines()
    assert lines[:3] == [
        "app/ (файлов: 1, определений: 3)",
        "docs/ (файлов: 1, определений: 2)",
        "top.py (python, строк: 2, определений: 1)",
    ]
    assert "всего файлов: 3" in lines[-1]


def test_file_outline_errors(ctx: ToolContext) -> None:
    _make(ctx.root, ".env", "KEY=1\n")
    for rel in ("credentials.json", "keys/id_rsa", ".ssh/config.py"):
        _make(ctx.root, rel, "KEY = 1\n")
    _make(ctx.root, ".gitignore", "build.py\n")
    _make(ctx.root, "build.py", "def x():\n    pass\n")
    (ctx.root / "blob.bin").write_bytes(b"\0\1\2")
    for path, needle in (
        ("missing.py", "не найден"),
        (".env", "секрет"),
        ("credentials.json", "секрет"),
        ("keys/id_rsa", "секрет"),
        (".ssh/config.py", "секрет"),
        ("build.py", "исключён"),
        ("blob.bin", "Бинарный"),
        ("../outside.py", "за пределы"),
    ):
        with pytest.raises(ToolError, match=needle):
            _outline(ctx, path)


def test_large_file_outline_suggests_ranges(
    ctx: ToolContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(index_mod, "MAX_INDEX_FILE_BYTES", 10)
    with pytest.raises(ToolError, match="диапазоном строк"):
        _outline(ctx, "app/agent.py")


def test_broken_index_becomes_tool_error(ctx: ToolContext, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(self: index_mod.ProjectIndex) -> NoReturn:
        raise OSError("диск недоступен")

    monkeypatch.setattr(index_mod.ProjectIndex, "open", boom)
    with pytest.raises(ToolError, match="Индекс проекта недоступен"):
        _find(ctx, query="Agent")


def test_file_outline_reports_parse_errors(
    ctx: ToolContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(text: str, language: str | None, path: str) -> NoReturn:
        raise RuntimeError("сбой разбора")

    monkeypatch.setattr(index_mod, "extract", boom)
    with pytest.raises(ToolError, match="Не удалось разобрать"):
        _outline(ctx, "app/agent.py")


# ------------------------------ импорты и докстринги ------------------------------ #
def _cli(ctx: ToolContext) -> None:
    _make(
        ctx.root,
        "app/cli.py",
        "from app.agent import Agent, helper\n\n\n"
        "def main(text):\n"
        "    Agent().run_turn(text)\n"
        "    helper()\n",
    )
    _make(ctx.root, "scripts/x.py", "def other(bot):\n    bot.run_turn()\n")


def test_file_outline_shows_project_imports(ctx: ToolContext) -> None:
    _cli(ctx)
    cli = _outline(ctx, "app/cli.py").content.splitlines()
    assert cli[:2] == [
        "app/cli.py (python, строк: 6, определений: 1)",
        "импортирует из проекта: app/agent.py",
    ]
    agent = _outline(ctx, "app/agent.py").content.splitlines()
    assert agent[1] == "импортируется в (1): app/cli.py"


def test_find_symbol_and_outline_show_docstring(ctx: ToolContext) -> None:
    _make(ctx.root, "app/doc.py", 'def documented():\n    """Делает дело.\n\n    Подробно."""\n')
    assert _find(ctx, query="documented").content.endswith("  # Делает дело.")
    assert _outline(ctx, "app/doc.py").content.splitlines()[1] == (
        "1-4  def documented():  # Делает дело."
    )


def test_corrupted_index_is_recreated_by_the_next_call(tmp_path: Path) -> None:
    # Повреждены страницы данных (не заголовок): без пересоздания каждый вызов
    # навигации падал бы до /index rebuild, которую модель выполнить не может.
    for module_num in range(150):
        body = "\n".join(
            f"def func_{module_num}_{func_num}(x):\n    return helper_{func_num}(x)\n"
            for func_num in range(20)
        )
        (tmp_path / f"m{module_num}.py").write_text(body, encoding="utf-8")
    ctx = ToolContext(workspace=Workspace(tmp_path), semantic=False)
    tool = FindSymbolTool()
    assert "func_1_1" in tool.run(FindSymbolParams(query="func_1_1"), ctx).content
    db = ctx.workspace.index_dir / index_mod.INDEX_FILE_NAME
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    size = db.stat().st_size
    with Path(db).open("r+b") as db_file:  # страницы 1–16 (схема, meta) целы, дальше — мусор
        db_file.seek(16 * 4096)
        db_file.write(bytes(range(256)) * ((size - 16 * 4096) // 256))
    with index_mod.ProjectIndex(ctx.workspace) as ix:
        assert ix.is_complete()
    with pytest.raises(ToolError, match="повреждён"):
        tool.run(FindSymbolParams(query="func_1_1"), ctx)
    assert "m1.py" in tool.run(FindSymbolParams(query="func_1_1"), ctx).content
    refs = FindReferencesTool().run(FindReferencesParams(query="helper_1"), ctx)
    assert "m1.py" in refs.content
