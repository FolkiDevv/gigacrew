"""Тесты индекса проекта (SQLite в .gigacode/index)."""

from __future__ import annotations

import os
import sqlite3

import pytest

from gigacrew.project import index as index_mod
from gigacrew.project.index import INDEX_FILE_NAME, ProjectIndex
from gigacrew.project.workspace import Workspace


def _make(root, rel, text="x = 1\n"):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _bump(path, text):
    """Перезаписать файл так, чтобы изменились размер и mtime."""
    st = path.stat()
    path.write_text(text, encoding="utf-8")
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))


@pytest.fixture
def project(tmp_path):
    _make(tmp_path, "app/agent.py", "class Agent:\n    def run_turn(self):\n        pass\n")
    _make(
        tmp_path, "app/events.py", "class AgentEvents:\n    def on_turn_end(self):\n        pass\n"
    )
    _make(tmp_path, "web/store.ts", "export class Store {}\nexport function runTurn() {}\n")
    _make(tmp_path, "README.md", "# Проект\n## Установка\n")
    return tmp_path


@pytest.fixture
def index(project):
    with ProjectIndex(Workspace(project)) as ix:
        yield ix


def _paths(ix):
    return [e.path for e in ix.files_under("")]


# ------------------------------ построение ------------------------------ #
def test_constructor_has_no_side_effects(project):
    ix = ProjectIndex(Workspace(project))
    assert not (project / ".gigacode").exists()
    assert ix.is_complete() is False
    assert not (project / ".gigacode").exists()


def test_build_creates_data_dir_and_marks_complete(index, project):
    stats = index.refresh()
    assert (stats.scanned, stats.added, stats.updated, stats.removed) == (4, 4, 0, 0)
    assert (project / ".gigacode" / ".gitignore").is_file()
    assert (project / ".gigacode" / "index" / INDEX_FILE_NAME).is_file()
    assert index.is_complete()
    info = index.stats()
    assert info.files == 4 and info.symbols == 8 and info.complete
    assert dict(info.languages) == {"python": 2, "typescript": 1, "markdown": 1}
    assert info.updated_at is not None and info.db_bytes > 0


def test_incremental_refresh_reparses_only_changes(index, project, monkeypatch):
    index.refresh()
    parsed = []
    real = index_mod.extract
    monkeypatch.setattr(
        index_mod, "extract", lambda text, lang, path: parsed.append(lang) or real(text, lang, path)
    )
    assert index.refresh().changed == 0 and parsed == []

    _bump(project / "app/agent.py", "class Agent:\n    def renamed(self):\n        pass\n")
    _make(project, "app/new.py", "def fresh():\n    pass\n")
    (project / "web/store.ts").unlink()
    stats = index.refresh()
    assert (stats.added, stats.updated, stats.removed) == (1, 1, 1)
    assert parsed == ["python", "python"]
    assert "web/store.ts" not in _paths(index)
    assert [s.qualname for s in index.outline("app/agent.py")] == ["Agent", "Agent.renamed"]
    assert index.find_symbols("runTurn")[1] == 0


def test_refresh_subtree_leaves_rest_untouched(index, project):
    index.refresh()
    (project / "web/store.ts").unlink()
    _make(project, "app/more.py", "def more():\n    pass\n")
    stats = index.refresh(project / "app")
    assert (stats.added, stats.removed) == (1, 0)
    assert "web/store.ts" in _paths(index)  # вне поддерева — не трогаем
    assert index.is_complete()  # частичное обновление не снимает признак полноты


def test_refresh_single_file(index, project):
    index.refresh(project / "app" / "agent.py")
    assert _paths(index) == ["app/agent.py"]
    assert not index.is_complete()


def test_secrets_binary_large_and_ignored_files(index, project, monkeypatch):
    monkeypatch.setattr(index_mod, "MAX_INDEX_FILE_BYTES", 100)
    _make(project, ".env", "GIGACREW_ACCESS_KEY=secret\n")
    _make(project, ".gitignore", "generated/\n")
    _make(project, "generated/big.py", "def hidden(): pass\n")
    (project / "logo.png").write_bytes(b"\x89PNG\0\0\0")
    _make(project, "huge.py", "def huge(): pass\n" * 20)
    index.refresh()
    paths = _paths(index)
    assert ".env" not in paths and "generated/big.py" not in paths
    assert index.file_entry("logo.png").status == "binary"
    big = index.file_entry("huge.py")
    assert big.status == "large" and big.symbols == 0
    # явное обновление исключённого каталога не протаскивает его в индекс
    index.refresh(project / "generated")
    assert "generated/big.py" not in _paths(index)


def test_interrupted_build_keeps_committed_batches(project, monkeypatch):
    monkeypatch.setattr(index_mod, "_BATCH", 2)
    for i in range(5):
        _make(project, f"many/m{i}.py", f"def f{i}(): pass\n")

    def stop_at(n):
        if n == 6:
            raise KeyboardInterrupt

    ws = Workspace(project)
    with ProjectIndex(ws) as ix, pytest.raises(KeyboardInterrupt):
        ix.refresh(progress=stop_at)
    with ProjectIndex(ws) as ix:
        assert not ix.is_complete()
        assert len(_paths(ix)) == 4  # две зафиксированные пачки по 2 файла
        stats = ix.refresh()
        assert stats.added == 5 and ix.is_complete()


def test_rebuild(index, project):
    index.refresh()
    stats = index.rebuild()
    assert stats.added == 4 and index.is_complete()


# ------------------------------ совместимость ------------------------------ #
def test_old_schema_is_recreated(project):
    ws = Workspace(project)
    with ProjectIndex(ws) as ix:
        ix.refresh()
    conn = sqlite3.connect(ws.index_dir / INDEX_FILE_NAME)
    with conn:
        conn.execute("UPDATE meta SET value = '0' WHERE key = 'schema_version'")
    conn.close()
    with ProjectIndex(ws) as ix:
        assert not ix.is_complete() and ix.stats().files == 0


def test_corrupted_database_is_recreated(project):
    ws = Workspace(project)
    ws.ensure_data_dir()
    ws.index_dir.mkdir()
    (ws.index_dir / INDEX_FILE_NAME).write_bytes(b"not a database at all" * 100)
    with ProjectIndex(ws) as ix:
        assert ix.refresh().added == 4


# ------------------------------ поиск ------------------------------ #
def test_find_symbols_ranking(index):
    index.refresh()
    hits, total = index.find_symbols("agent")
    assert total == 2
    assert [h.symbol.name for h in hits] == ["Agent", "AgentEvents"]  # точное — первым


def test_find_symbols_qualified_kind_glob_and_limit(index):
    index.refresh()
    hits, _ = index.find_symbols("Agent.run_turn")
    assert [(h.path, h.symbol.qualname) for h in hits] == [("app/agent.py", "Agent.run_turn")]
    # регистр не важен, 'function' включает методы
    hits, total = index.find_symbols("RUNTURN", kind="function")
    assert total == 1 and hits[0].path == "web/store.ts"
    hits, total = index.find_symbols("run", kind="function", path_glob="app/**")
    assert [h.symbol.qualname for h in hits] == ["Agent.run_turn"]
    hits, total = index.find_symbols("e", limit=1)
    assert len(hits) == 1 and total > 1
    assert index.find_symbols("100%_")[1] == 0  # спецсимволы LIKE экранируются
    with pytest.raises(ValueError):
        index.find_symbols("  ")


@pytest.mark.skipif(os.name == "nt", reason="регистр путей")
def test_subtree_is_case_sensitive(index, project):
    _make(project, "Src/upper.py", "def upper(): pass\n")
    _make(project, "src/lower.py", "def lower(): pass\n")
    index.refresh()
    index.refresh(project / "src")
    assert "Src/upper.py" in _paths(index)  # соседний каталог в другом регистре не тронут
    assert [e.path for e in index.files_under("src")] == ["src/lower.py"]


def test_unreadable_file_is_retried_when_unchanged(index, project, monkeypatch):
    real = index_mod.Path.read_bytes
    broken = {"app/agent.py"}

    def read_bytes(self):
        if self.relative_to(project.resolve()).as_posix() in broken:
            raise PermissionError("нет доступа")
        return real(self)

    monkeypatch.setattr(index_mod.Path, "read_bytes", read_bytes)
    index.refresh()
    assert index.file_entry("app/agent.py").status == "error"
    broken.clear()  # права починили, файл не менялся
    assert index.refresh().updated == 1
    assert index.file_entry("app/agent.py").status == "indexed"
    assert index.refresh().changed == 0


@pytest.mark.skipif(os.name == "nt", reason="симлинки")
def test_symlinks_to_secrets_and_excluded_files_are_not_indexed(index, project):
    _make(project, ".env", 'SECRET_KEY = "s3cr3t"\n')
    (project / "settings.py").symlink_to(project / ".env")
    _make(project, ".gitignore", "vendor/\n")
    _make(project, "vendor/lib.py", "def vendored(): pass\n")
    (project / "lib_link.py").symlink_to(project / "vendor" / "lib.py")
    index.refresh()
    paths = _paths(index)
    assert "settings.py" not in paths and "lib_link.py" not in paths
    assert index.find_symbols("SECRET")[1] == 0
    assert index.find_symbols("vendored")[1] == 0


# ------------------------- использования и импорты ------------------------- #
@pytest.fixture
def refs_project(tmp_path):
    _make(tmp_path, "pkg/__init__.py", "")
    _make(
        tmp_path,
        "pkg/core.py",
        "def helper():\n    pass\n\n\ndef use():\n    return helper()\n",
    )
    _make(
        tmp_path,
        "pkg/user.py",
        "from pkg.core import helper as hp\n\n\ndef go():\n    return hp()\n",
    )
    _make(tmp_path, "pkg/rel.py", "from .core import helper\n\n\ndef go():\n    helper()\n")
    _make(tmp_path, "other.py", "def run(x):\n    return x.helper()\n")
    with ProjectIndex(Workspace(tmp_path)) as ix:
        ix.refresh()
        yield ix


def test_find_refs_ranks_by_resolution(refs_project):
    result = refs_project.find_refs("HELPER")  # определение — без учёта регистра
    assert [(d.path, d.symbol.name) for d in result.definitions] == [("pkg/core.py", "helper")]
    got = [(h.path, h.line, h.kind, h.resolution) for h in result.hits]
    assert got == [
        ("pkg/rel.py", 1, "import", "import"),
        ("pkg/rel.py", 5, "call", "import"),
        ("pkg/user.py", 1, "import", "import"),
        ("pkg/user.py", 5, "call", "import"),  # через псевдоним hp
        ("pkg/core.py", 6, "call", "same_file"),
        ("other.py", 2, "call", "name_only"),
    ]
    assert result.total == 6
    assert refs_project.find_refs("helper", kind="import").total == 2
    calls = refs_project.find_refs("helper", kind="call", path_glob="pkg/**", limit=1)
    assert calls.total == 3 and len(calls.hits) == 1
    assert refs_project.find_refs("pkg.rel.go").hits == []  # без использований
    with pytest.raises(ValueError):
        refs_project.find_refs(" . ")


def test_find_refs_without_definition_searches_by_name(refs_project):
    result = refs_project.find_refs("hp")
    assert result.definitions == []
    assert [(h.path, h.line, h.resolution) for h in result.hits] == [
        ("pkg/user.py", 5, "name_only")
    ]


def test_imports_resolve_to_project_files(tmp_path):
    _make(tmp_path, "src/pkg/__init__.py", "")
    _make(tmp_path, "src/pkg/b.py", "B = 1\n")
    _make(tmp_path, "tests/fake/pkg/b.py", "B = 2\n")  # тот же суффикс глубже — не он
    _make(
        tmp_path,
        "src/pkg/a.py",
        "import os\nfrom pkg import b\nfrom . import b as bb\nimport pkg.b\nfrom pkg.b import B\n",
    )
    with ProjectIndex(Workspace(tmp_path)) as ix:
        ix.refresh()
        assert ix.resolve_module("pkg.b") == "src/pkg/b.py"
        assert ix.resolve_module("pkg") == "src/pkg/__init__.py"
        assert ix.resolve_module("os") is None and ix.resolve_module("..x") is None
        entries = ix.imports_of("src/pkg/a.py")
        assert [(e.module, e.name, e.target) for e in entries] == [
            ("os", "", None),
            ("pkg", "b", "src/pkg/b.py"),  # подмодуль, а не атрибут пакета
            ("src.pkg", "b", "src/pkg/b.py"),
            ("pkg.b", "", "src/pkg/b.py"),
            ("pkg.b", "B", "src/pkg/b.py"),
        ]
        assert ix.imported_by("src/pkg/b.py") == [("src/pkg/a.py", 2)]
        assert ix.imported_by("src/pkg/__init__.py") == []
        assert ix.imported_by("README.md") == []


def test_refs_and_imports_follow_incremental_refresh(index, project):
    stats = index.refresh()
    assert set(stats.added_paths) == {"app/agent.py", "app/events.py", "web/store.ts", "README.md"}
    _make(project, "app/main.py", "from app.agent import Agent\n\nAgent().run_turn()\n")
    stats = index.refresh()
    assert (stats.added_paths, stats.updated_paths, stats.removed_paths) == (
        ("app/main.py",),
        (),
        (),
    )
    _bump(project / "app/main.py", "from app.agent import Agent\n\nAgent().run_turn(1)\n")
    assert index.refresh().updated_paths == ("app/main.py",)
    assert index.find_refs("Agent.run_turn").total == 1
    assert index.stats().refs == 2

    (project / "app/main.py").unlink()
    stats = index.refresh()
    assert stats.removed_paths == ("app/main.py",)
    assert index.find_refs("Agent").total == 0
    assert index.imported_by("app/agent.py") == []
    assert index.stats().refs == 0


def test_stub_only_package_resolves(tmp_path):
    _make(tmp_path, "stubs/pkg/__init__.pyi", "def f() -> int: ...\n")
    _make(tmp_path, "app.py", "from pkg import f\n\nf()\n")
    with ProjectIndex(Workspace(tmp_path)) as ix:
        ix.refresh()
        assert ix.resolve_module("pkg") == "stubs/pkg/__init__.pyi"
        assert ix.imported_by("stubs/pkg/__init__.pyi") == [("app.py", 1)]


def test_recursive_call_on_definition_line_is_a_reference(tmp_path):
    _make(tmp_path, "m.py", "def fact(n): return 1 if n < 2 else n * fact(n - 1)\n")
    with ProjectIndex(Workspace(tmp_path)) as ix:
        ix.refresh()
        hits = ix.find_refs("fact").hits
        assert [(h.line, h.kind, h.scope) for h in hits] == [(1, "call", "fact")]


def _corrupt_pages(ws: Workspace) -> None:
    """Порча страниц данных при целых заголовке и meta (сбой диска, запись наполовину)."""
    import sqlite3

    db = ws.index_dir / INDEX_FILE_NAME
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")  # страницы — в основной файл
    conn.close()
    size = db.stat().st_size
    with open(db, "r+b") as f:  # страницы 1–16 (схема, meta) целы, дальше — мусор
        f.seek(16 * 4096)
        f.write(bytes(range(256)) * ((size - 16 * 4096) // 256))


def _big_project(root) -> Workspace:
    for i in range(150):
        body = "\n".join(f"def func_{i}_{j}(x):\n    return helper_{j}(x)\n" for j in range(20))
        (root / f"m{i}.py").write_text(body, encoding="utf-8")
    ws = Workspace(root)
    with ProjectIndex(ws) as ix:
        ix.refresh()
    return ws


def test_corrupted_data_pages_are_detected_and_index_recreated(tmp_path):
    import sqlite3

    from gigacrew.project.index import is_corruption

    ws = _big_project(tmp_path)
    _corrupt_pages(ws)
    with ProjectIndex(ws) as ix:
        assert ix.is_complete()  # заголовок и meta целы — повреждение не видно сразу
        with pytest.raises(sqlite3.DatabaseError) as error:
            ix.find_symbols("func_1_1")
        assert is_corruption(error.value)
        ix.discard()
        assert not ix.is_complete()
        assert ix.refresh().added == 150
        assert ix.find_symbols("func_1_19")[1] == 1
    assert not is_corruption(sqlite3.OperationalError("database is locked"))


def test_index_command_rebuilds_corrupted_index(tmp_path, capsys):
    from gigacrew.cli.indexing import run_indexing
    from gigacrew.ui.console import Console

    ws = _big_project(tmp_path)
    _corrupt_pages(ws)
    result = run_indexing(ProjectIndex(ws), Console(no_color=True))
    assert result is not None and result[1].files == 150
    assert "повреждён" in capsys.readouterr().out
    with ProjectIndex(ws) as ix:
        assert ix.find_symbols("func_1_19")[1] == 1
