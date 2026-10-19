"""Тесты ранжированной карты проекта и инструмента repo_map."""

from __future__ import annotations

import pytest

from gigacrew.agent.prompts import build_system_prompt
from gigacrew.project.index import ProjectIndex
from gigacrew.project.repomap import build_repo_map, pagerank, rank_definitions
from gigacrew.project.workspace import Workspace
from gigacrew.tools.base import ToolContext, build_default_registry
from gigacrew.tools.index import RepoMapParams, RepoMapTool


def _make(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


# ------------------------------ PageRank ------------------------------ #
def test_pagerank_sums_to_one_and_prefers_cited_nodes():
    edges = {"a": {"c": 1.0}, "b": {"c": 1.0}, "c": {"a": 1.0}}
    ranks = pagerank(edges, ["a", "b", "c", "d"])
    assert sum(ranks.values()) == pytest.approx(1.0)
    assert ranks["c"] > ranks["a"] > ranks["b"]
    assert ranks["b"] == pytest.approx(ranks["d"])  # на них никто не ссылается


def test_pagerank_personalization_and_weights():
    weighted = pagerank({"x": {"a": 1.0, "b": 9.0}}, ["x", "a", "b"])
    assert weighted["b"] > weighted["a"]  # ранг делится по весам рёбер
    edges = {"x": {"b": 1.0}, "y": {"a": 1.0}}
    focused = pagerank(edges, ["x", "y", "a", "b"], {"y": 1.0})
    assert focused["a"] > focused["b"] and focused["x"] == pytest.approx(0.0)
    assert pagerank({}, []) == {}


# ------------------------------ карта ------------------------------ #
@pytest.fixture
def project(tmp_path):
    _make(
        tmp_path,
        "core/engine.py",
        "class Engine:\n"
        "    def start(self):\n"
        "        pass\n\n"
        "    def get(self):\n"
        "        pass\n\n\n"
        "def _private_helper():\n"
        "    pass\n",
    )
    _make(tmp_path, "core/util.py", "def rarely_used_function():\n    pass\n")
    for i in range(3):
        _make(
            tmp_path,
            f"app/user{i}.py",
            "from core.engine import Engine\n\n\n"
            f"def run{i}(cache):\n"
            "    Engine().start()\n"
            "    cache.get('k')\n",
        )
    _make(tmp_path, "README.md", "# Проект\n")
    with ProjectIndex(Workspace(tmp_path)) as ix:
        ix.refresh()
        yield ix


def test_rank_definitions_follow_references(project):
    ranked = [(d.path, d.name) for d in rank_definitions(project)]
    assert ranked[:2] == [("core/engine.py", "Engine"), ("core/engine.py", "start")]
    # `get` совпадает с dict.get — вызов cache.get() почти ничего не добавляет
    assert ranked.index(("core/engine.py", "get")) > ranked.index(("core/engine.py", "start"))
    assert ("README.md", "Проект") not in ranked  # заголовки Markdown — не определения
    # неиспользуемые — после используемых, по рангу файла (а не наверху за счёт файла)
    assert ranked[3:] == [
        ("core/engine.py", "_private_helper"),
        ("app/user0.py", "run0"),
        ("app/user1.py", "run1"),
        ("app/user2.py", "run2"),
        ("core/util.py", "rarely_used_function"),
    ]


def test_focus_names_and_files_change_order(project):
    rare = ("core/util.py", "rarely_used_function")
    plain = [(d.path, d.name) for d in rank_definitions(project)]
    focused = [
        (d.path, d.name) for d in rank_definitions(project, focus_names=["rarely_used_function"])
    ]
    assert focused.index(rare) <= plain.index(rare)
    by_file = [(d.path, d.name) for d in rank_definitions(project, focus_files=["app/user1.py"])]
    assert by_file[0] == ("core/engine.py", "Engine")
    assert ("app/user1.py", "run1") in by_file


def test_build_repo_map_respects_budget_and_shows_context(project):
    full = build_repo_map(project, max_chars=10_000)
    assert full.total == full.definitions
    lines = full.text.splitlines()
    assert lines[:3] == ["core/engine.py:", "     1  class Engine:", "     2    def start(self):"]
    small = build_repo_map(project, max_chars=60)
    assert len(small.text) <= 60 and small.definitions < full.definitions
    # метод показывается вместе с классом, пропущенные — «⋮ ещё N»
    only_start = build_repo_map(project, focus_names=["start"], max_chars=90)
    assert "class Engine:" in only_start.text
    assert build_repo_map(project, max_chars=0).text == ""


def test_empty_index_has_no_map(tmp_path):
    with ProjectIndex(Workspace(tmp_path)) as ix:
        ix.refresh()
        assert build_repo_map(ix).text == ""


def test_unbuilt_index_gives_no_map_in_prompt(tmp_path):
    for i in range(250):
        _make(tmp_path, f"pkg/m{i:03}.py", "def f():\n    pass\n")
    prompt = build_system_prompt(Workspace(tmp_path))
    assert "the tree is incomplete" in prompt and "Project map" not in prompt
    assert not (tmp_path / ".gigacode").exists()  # промпт индекс не строит


# ------------------------------ инструмент ------------------------------ #
def test_repo_map_tool(tmp_path, project):
    ctx = ToolContext(workspace=Workspace(tmp_path), semantic=False)
    assert "repo_map" in build_default_registry()
    result = RepoMapTool().run(RepoMapParams(), ctx)
    assert result.content.startswith("карта проекта: определений ")
    assert "core/engine.py:" in result.content
    focused = RepoMapTool().run(
        RepoMapParams(focus=["app", "core/util.py", "Engine.start"], max_tokens=50), ctx
    )
    assert focused.summary.startswith("карта: определений ")
    assert RepoMapTool().describe(RepoMapParams(focus=["app"])) == "app"


def test_system_prompt_has_repo_map_only_for_large_indexed_projects(tmp_path):
    _make(tmp_path, "core/engine.py", "class Engine:\n    pass\n")
    _make(tmp_path, "app/main.py", "from core.engine import Engine\n\nEngine()\n")
    ws = Workspace(tmp_path)
    with ProjectIndex(ws) as ix:
        ix.refresh()
    assert "Project map" not in build_system_prompt(ws)  # дерево целиком — карта не нужна
    for i in range(250):
        _make(tmp_path, f"pkg/m{i:03}.py", "from core.engine import Engine\n")
    prompt = build_system_prompt(ws)
    assert "Project map" in prompt and "class Engine:" in prompt


def test_repo_map_tool_explains_too_small_budget(tmp_path, project, monkeypatch):
    from gigacrew.project.repomap import RepoMap
    from gigacrew.tools import index as tools_index

    monkeypatch.setattr(tools_index, "build_repo_map", lambda *a, **kw: RepoMap("", 0, 0, 7))
    ctx = ToolContext(workspace=Workspace(tmp_path), semantic=False)
    content = RepoMapTool().run(RepoMapParams(max_tokens=200), ctx).content
    assert content == "(определения не поместились в объём карты — увеличьте max_tokens)"
