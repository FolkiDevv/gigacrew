"""Тесты инструментов навигации: find_references, goto_definition, call_hierarchy.

Режим «по индексу» — ``ToolContext(semantic=False)``; режим ty — настоящий
``ty server`` (пакет ty — обязательная зависимость), он запускается на время теста.
"""

from __future__ import annotations

import os

import pytest

from gigacrew.project import semantic
from gigacrew.project.index import RESOLVED_IMPORT, ProjectIndex, RefHit
from gigacrew.project.lsp import LspError
from gigacrew.project.semantic import Location, SemanticUnavailable
from gigacrew.project.workspace import Workspace
from gigacrew.tools.base import ToolContext, ToolError, build_default_registry
from gigacrew.tools.navigation import (
    RESOLVED_TY,
    CallHierarchyParams,
    CallHierarchyTool,
    FindReferencesParams,
    FindReferencesTool,
    GotoDefinitionParams,
    GotoDefinitionTool,
    external_path,
)

AGENT = (
    "class Agent:\n"
    "    def run_turn(self, text: str) -> str:\n"
    '        """Ход агента."""\n'
    "        return helper(text)\n"
    "\n"
    "\n"
    "def helper(value: str) -> str:\n"
    "    return value\n"
)
CLI = (
    "from app.agent import Agent, helper as hp\n"
    "\n"
    "\n"
    "def main(text: str) -> None:\n"
    "    Agent().run_turn(text)\n"
    "    hp(text)\n"
    "    print(len(text))\n"
)


def _make(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


@pytest.fixture(autouse=True)
def _stop_servers():
    yield
    semantic.shutdown_all()


@pytest.fixture
def root(tmp_path):
    _make(tmp_path, "app/__init__.py", "")
    _make(tmp_path, "app/agent.py", AGENT)
    _make(tmp_path, "app/cli.py", CLI)
    _make(tmp_path, "scripts/x.py", "def other(bot):\n    bot.run_turn()\n")
    return tmp_path


@pytest.fixture
def plain(root):
    """Только индекс: ty выключен."""
    return ToolContext(workspace=Workspace(root), semantic=False)


@pytest.fixture
def typed(root):
    return ToolContext(workspace=Workspace(root))


def _refs(ctx, **kw):
    return FindReferencesTool().run(FindReferencesParams(**kw), ctx)


def _goto(ctx, path, line, name):
    return GotoDefinitionTool().run(GotoDefinitionParams(path=path, line=line, name=name), ctx)


def _calls(ctx, query, direction="incoming", depth=1):
    params = CallHierarchyParams(query=query, direction=direction, depth=depth)
    return CallHierarchyTool().run(params, ctx)


def test_registered_in_default_registry():
    reg = build_default_registry()
    assert all(name in reg for name in ("find_references", "goto_definition", "call_hierarchy"))


# ------------------------------ по индексу ------------------------------ #
def test_find_references_by_index(plain):
    result = _refs(plain, query="Agent.run_turn")
    assert result.content.splitlines() == [
        "определение: app/agent.py:2-4 method Agent.run_turn — "
        "def run_turn(self, text: str) -> str:  # Ход агента.",
        "использования (2):",
        "app/cli.py — импортирует модуль с определением:",
        "  5 [вызов] в main — Agent().run_turn(text)",
        "scripts/x.py — совпадение только по имени — может быть другое определение:",
        "  2 [вызов] в other — bot.run_turn()",
    ]
    assert result.summary.startswith("найдено использований: 2; индекс обновлён")


def test_find_references_kind_limit_and_errors(plain):
    imports = _refs(plain, query="helper", kind="IMPORT").content.splitlines()
    assert imports[1:] == [
        "использования (1):",
        "app/cli.py — импортирует модуль с определением:",
        "  1 [импорт] — from app.agent import Agent, helper as hp",
    ]
    limited = _refs(plain, query="run_turn", max_results=1).content
    assert "… показано 1 из 2; уточните запрос, kind или glob" in limited
    assert _refs(plain, query="nowhere").content.splitlines() == [
        "(определение «nowhere» в индексе не найдено — ищу использования по имени)",
        "(использований не найдено)",
    ]
    with pytest.raises(ToolError, match="допустимые: call, attr, name, import"):
        _refs(plain, query="helper", kind="usage")
    with pytest.raises(ToolError, match="Пустой запрос"):
        _refs(plain, query=" . ")


def test_goto_definition_by_index_follows_import_alias(plain):
    result = _goto(plain, "app/cli.py", 6, "hp")
    assert result.content.splitlines() == [
        "app/agent.py:7-8 function helper — def helper(value: str) -> str:",
        "(по индексу — по имени и импортам файла)",
    ]
    assert result.summary.startswith("определений: 1 (по индексу)")
    assert _goto(plain, "app/agent.py", 4, "helper").content.startswith("app/agent.py:7-8")
    assert _goto(plain, "app/cli.py", 7, "len").content.startswith("(определение не найдено)")


def test_goto_definition_errors(plain):
    with pytest.raises(ToolError, match="Файл не найден"):
        _goto(plain, "app/none.py", 1, "x")
    with pytest.raises(ToolError, match="нет строки 99"):
        _goto(plain, "app/cli.py", 99, "x")
    with pytest.raises(ToolError, match="нет имени «missing»"):
        _goto(plain, "app/cli.py", 5, "missing")
    _make(plain.root, ".env", "KEY=1\n")
    with pytest.raises(ToolError, match="секретами"):
        _goto(plain, ".env", 1, "KEY")
    # То же правило, что у read_file и классификатора команд: не только .env*.
    for rel in ("deploy/credentials.json", "certs/server.pem", ".aws/config.py"):
        _make(plain.root, rel, "KEY = 1\n")
        with pytest.raises(ToolError, match="секретами"):
            _goto(plain, rel, 1, "KEY")


def test_call_hierarchy_by_index(plain):
    incoming = _calls(plain, "helper", depth=2).content.splitlines()
    assert incoming == [
        "кто вызывает (глубина 2, по индексу):",
        "helper — app/agent.py:7",
        "  ← main — app/cli.py (строка 6)",
        "  ← Agent.run_turn — app/agent.py (строка 4)",
        "    ← main — app/cli.py (строка 5)",
        "    ← other — scripts/x.py (строка 2) (по имени)",
    ]
    outgoing = _calls(plain, "main", "outgoing", depth=2).content.splitlines()
    assert outgoing == [
        "что вызывает (глубина 2, по индексу):",
        "main — app/cli.py:4",
        "  → Agent (строка 5)",
        "  → run_turn (строка 5)",
        "  → hp (строка 6)",
        "  (глубже 1 уровня исходящие вызовы — только через ty)",
    ]
    with pytest.raises(ToolError, match="не найдено в индексе"):
        _calls(plain, "nowhere")
    with pytest.raises(ToolError, match="direction"):
        _calls(plain, "helper", "sideways")


# ------------------------------ через ty ------------------------------ #
def test_find_references_with_ty_marks_confirmed(typed):
    result = _refs(typed, query="Agent.run_turn")
    assert result.content.splitlines()[1:] == [
        "использования (2):",
        "app/cli.py — точно (ty):",
        "  5 [вызов] в main — Agent().run_turn(text)",
        "scripts/x.py — совпадение только по имени, ty не подтвердил — "
        "вероятно, другое определение:",
        "  2 [вызов] в other — bot.run_turn()",
    ]
    assert "подтверждено ty: 1" in result.summary
    # псевдоним импорта ty не находит — остаётся найденное индексом
    helper = _refs(typed, query="helper").content.splitlines()
    assert helper[1:] == [
        "использования (3):",
        "app/agent.py — точно (ty):",
        "  4 [вызов] в Agent.run_turn — return helper(text)",
        "app/cli.py — импортирует модуль с определением:",
        "  1 [импорт] — from app.agent import Agent, helper as hp",
        "  6 [вызов] в main — hp(text)",
    ]


def test_ty_sees_edits_made_after_start(typed):
    _refs(typed, query="helper")  # сервер запущен
    _make(typed.root, "app/new.py", "from app.agent import helper\n\nhelper('x')\n")
    _make(typed.root, "app/cli.py", CLI.replace("Agent().run_turn(text)", "pass"))
    content = _refs(typed, query="Agent.run_turn").content
    assert "app/cli.py" not in content  # изменённый файл перечитан
    new = _refs(typed, query="helper").content.splitlines()
    assert "app/new.py — точно (ty):" in new  # новый файл замечен


def test_goto_definition_with_ty(typed):
    result = _goto(typed, "app/cli.py", 5, "Agent().run_turn")
    assert result.content.splitlines() == [
        "app/agent.py:2-4 method Agent.run_turn — "
        "def run_turn(self, text: str) -> str:  # Ход агента.",
        "тип: bound method Agent.run_turn(text: str) -> str",
    ]
    assert result.summary.startswith("определений: 1 (ty)")
    builtin = _goto(typed, "app/cli.py", 7, "len").content.splitlines()
    assert builtin[0].startswith("typeshed/stdlib/builtins.pyi:") and "(вне проекта)" in builtin[0]
    assert builtin[1] == "тип: def len(obj: Sized, /) -> int"


def test_call_hierarchy_with_ty(typed):
    incoming = _calls(typed, "helper", depth=2).content.splitlines()
    assert incoming == [
        "кто вызывает (глубина 2, ty):",
        "helper — app/agent.py:7",
        "  ← Agent.run_turn — app/agent.py:2 (строка 4)",
        "    ← main — app/cli.py:4 (строка 5)",
        "  ← main — app/cli.py:4 (строка 6)",  # через псевдоним hp
    ]
    outgoing = _calls(typed, "main", "outgoing").content.splitlines()
    assert outgoing[:5] == [
        "что вызывает (глубина 1, ty):",
        "main — app/cli.py:4",
        "  → Agent — app/agent.py:1 (строка 5)",
        "  → Agent.run_turn — app/agent.py:2 (строка 5)",
        "  → helper — app/agent.py:7 (строка 6)",
    ]
    assert "  → print — вне проекта: typeshed/stdlib/builtins.pyi (строка 7)" in outgoing


# --------------------------- сбои и откат на индекс --------------------------- #
def test_missing_ty_falls_back_to_index(typed, monkeypatch):
    def missing():
        raise SemanticUnavailable("пакет ty не установлен")

    monkeypatch.setattr(semantic, "find_binary", missing)
    lines = _refs(typed, query="Agent.run_turn").content.splitlines()
    assert "app/cli.py — импортирует модуль с определением:" in lines
    assert lines[-1] == "(ty недоступен: пакет ty не установлен; ответ по индексу — по именам)"


def test_failures_restart_once_then_disable_ty(typed, monkeypatch):
    def broken(self, path, line, col):
        raise LspError("textDocument/references: сбой")

    monkeypatch.setattr(semantic.TyServer, "references", broken)
    first = _refs(typed, query="helper").content.splitlines()[-1]
    assert first == (
        "(ty: textDocument/references: сбой — сервер будет перезапущен; "
        "ответ по индексу — по именам)"
    )
    second = _refs(typed, query="helper").content.splitlines()[-1]
    assert "ty отключён до конца сессии после сбоев" in second
    assert "ty отключён" in semantic.status(typed.workspace)
    # отключённый ty больше не запускается
    monkeypatch.setattr(semantic, "find_binary", lambda: pytest.fail("перезапуск"))
    assert "ty отключён" in _refs(typed, query="helper").content


def test_merge_adds_ty_only_hits_with_kind_and_scope(root):
    ws = Workspace(root)
    with ProjectIndex(ws) as index:
        index.refresh()
        definitions = index.definitions("helper")
        hits = index.find_refs("helper").hits
        extra = [
            Location("app/cli.py", 7, 4),  # строка без использования helper — область по индексу
            Location("app/agent.py", 7, 4),  # строка определения — не использование
            Location("/usr/lib/x.py", 1, 0, in_project=False),
        ]
        merged = FindReferencesTool._merge(index, hits, extra, definitions, None, None)
        added = [h for h in merged if h not in hits]
        assert added == [RefHit("app/cli.py", 7, 4, "name", "main", RESOLVED_TY)]
        assert FindReferencesTool._merge(index, hits, extra, definitions, "call", None) == hits
        assert FindReferencesTool._merge(index, hits, extra, definitions, None, "scripts/**") == [
            h for h in hits
        ]
        assert all(h.resolution == RESOLVED_IMPORT for h in hits if h.path == "app/cli.py")


@pytest.mark.parametrize(
    "path, expected",
    [
        (
            "/root/.cache/ty/vendored/typeshed/b932/stdlib/os/__init__.pyi",
            "typeshed/stdlib/os/__init__.pyi",
        ),
        ("/venv/lib/python3.13/site-packages/httpx/_client.py", "site-packages/httpx/_client.py"),
        ("/opt/other.py", "/opt/other.py"),
    ],
)
def test_external_path(path, expected):
    assert external_path(path) == expected


@pytest.mark.skipif(os.name == "nt", reason="группы процессов POSIX")
def test_ty_server_runs_in_own_process_group(typed):
    _refs(typed, query="helper")
    server = semantic.server_for(typed.workspace)
    pid = server._client._proc.pid  # noqa: SLF001
    assert os.getpgid(pid) != os.getpgid(0)  # Ctrl+C терминала серверу не достаётся
    assert semantic.status(typed.workspace) == "работает"


def test_merge_matches_by_position_not_line(root):
    line = "    a.run_turn('x'); b.run_turn('y')"
    _make(root, "app/two.py", f"from app.agent import Agent\n\n\ndef both(a: Agent, b):\n{line}\n")
    first, second = line.index("run_turn"), line.rindex("run_turn")
    with ProjectIndex(Workspace(root)) as index:
        index.refresh()
        definitions = index.definitions("Agent.run_turn")
        hits = [h for h in index.find_refs("Agent.run_turn").hits if h.path == "app/two.py"]
        assert [h.col for h in hits] == [first, second]
        # ty подтвердил только b.run_turn: a.run_turn не помечается
        merged = FindReferencesTool._merge(
            index, hits, [Location("app/two.py", 5, second)], definitions, None, None
        )
        assert [(h.col, h.resolution) for h in merged] == [
            (first, RESOLVED_IMPORT),
            (second, RESOLVED_TY),
        ]


def test_index_hierarchy_has_one_shared_budget(plain, monkeypatch):
    from gigacrew.tools import navigation

    monkeypatch.setattr(navigation, "MAX_HIERARCHY_NODES", 2)
    lines = _calls(plain, "helper", depth=3).content.splitlines()
    assert sum(line.lstrip().startswith("←") for line in lines) == 2
    assert lines[-1] == "… дерево обрезано (2 узлов); уменьшите depth"


def test_call_hierarchy_with_ty_keeps_other_language_definitions(typed):
    _make(typed.root, "web/helper.js", "function helper() {}\nfunction use() { helper(); }\n")
    lines = _calls(typed, "helper").content.splitlines()
    assert lines[0] == "кто вызывает (глубина 1, ty + индекс):"
    assert "helper — web/helper.js:1" in lines and "  ← use — web/helper.js (строка 2)" in lines


@pytest.mark.parametrize("path", ["../outside.py", "{outside}"])
def test_goto_definition_rejects_paths_outside_project(tmp_path, path):
    # Инструмент «только чтение» не должен открывать файлы вне проекта: его сообщение
    # «В строке N нет имени…» процитировало бы их строку.
    root = tmp_path / "proj"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("API_TOKEN = 'hunter2'\n", encoding="utf-8")
    ctx = ToolContext(workspace=Workspace(root), semantic=False)
    params = GotoDefinitionParams(path=path.format(outside=outside), line=1, name="nothing")
    with pytest.raises(ToolError, match="за пределы") as error:
        GotoDefinitionTool().run(params, ctx)
    assert "hunter2" not in str(error.value)


def test_find_references_orders_files_by_resolution_not_by_path(tmp_path):
    # Алфавитный порядок путей здесь обратен порядку разрешения: сортировка видна.
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "core.py").write_text("def helper(): pass\n", encoding="utf-8")
    (tmp_path / "z_user.py").write_text("from pkg.core import helper\nhelper()\n", encoding="utf-8")
    (tmp_path / "a_other.py").write_text("def f(x):\n    x.helper()\n", encoding="utf-8")
    ctx = ToolContext(workspace=Workspace(tmp_path), semantic=False)
    out = FindReferencesTool().run(FindReferencesParams(query="helper"), ctx).content
    heads = [line for line in out.splitlines() if line.endswith(":") and " — " in line]
    assert heads[0].startswith("z_user.py — импортирует")
    assert heads[-1].startswith("a_other.py — совпадение только по имени")


def test_call_hierarchy_depth_is_clamped(tmp_path):
    (tmp_path / "m.py").write_text(
        "def helper(): pass\ndef main():\n    helper()\n", encoding="utf-8"
    )
    ctx = ToolContext(workspace=Workspace(tmp_path), semantic=False)
    out = CallHierarchyTool().run(CallHierarchyParams(query="helper", depth=10), ctx)
    assert out.content.splitlines()[0] == "кто вызывает (глубина 3, по индексу):"
