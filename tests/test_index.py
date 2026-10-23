"""Тесты индекса проекта (SQLite в .gigacode/index)."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from gigacrew.cli.indexing import run_indexing
from gigacrew.project import index as index_mod
from gigacrew.project.index import INDEX_FILE_NAME, ProjectIndex, is_corruption
from gigacrew.project.symbols import FileFacts, extract
from gigacrew.project.workspace import Workspace
from gigacrew.ui.console import Console


def _make(root: Path, rel: str, text: str = "x = 1\n") -> Path:
    file_path = root / rel
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(text, encoding="utf-8")
    return file_path


def _bump(path: Path, text: str) -> None:
    """Перезаписать файл так, чтобы изменились размер и mtime."""
    st = path.stat()
    path.write_text(text, encoding="utf-8")
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))


@pytest.fixture
def project(tmp_path: Path) -> Path:
    _make(tmp_path, "app/agent.py", "class Agent:\n    def run_turn(self):\n        pass\n")
    _make(
        tmp_path,
        "app/events.py",
        "class AgentEvents:\n    def on_turn_end(self):\n        pass\n",
    )
    _make(tmp_path, "web/store.ts", "export class Store {}\nexport function runTurn() {}\n")
    _make(tmp_path, "README.md", "# Проект\n## Установка\n")
    return tmp_path


@pytest.fixture
def index(project: Path) -> Iterator[ProjectIndex]:
    with ProjectIndex(Workspace(project)) as ix:
        yield ix


def _paths(ix: ProjectIndex) -> list[str]:
    return [entry.path for entry in ix.files_under("")]


def _status(ix: ProjectIndex, rel_path: str) -> str:
    entry = ix.file_entry(rel_path)
    assert entry is not None
    return entry.status


# ------------------------------ построение ------------------------------ #
def test_constructor_has_no_side_effects(project: Path) -> None:
    ix = ProjectIndex(Workspace(project))
    assert not (project / ".gigacode").exists()
    assert ix.is_complete() is False
    assert not (project / ".gigacode").exists()


def test_build_creates_data_dir_and_marks_complete(index: ProjectIndex, project: Path) -> None:
    stats = index.refresh()
    assert (stats.scanned, stats.added, stats.updated, stats.removed) == (4, 4, 0, 0)
    assert (project / ".gigacode" / ".gitignore").is_file()
    assert (project / ".gigacode" / "index" / INDEX_FILE_NAME).is_file()
    assert index.is_complete()
    summary = index.stats()
    assert summary.files == 4
    assert summary.symbols == 8
    assert summary.complete
    assert dict(summary.languages) == {"python": 2, "typescript": 1, "markdown": 1}
    assert summary.updated_at is not None
    assert summary.db_bytes > 0


def test_incremental_refresh_reparses_only_changes(
    index: ProjectIndex,
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index.refresh()
    parsed: list[str | None] = []

    def counting_extract(text: str, lang: str | None, path: str) -> FileFacts:
        parsed.append(lang)
        return extract(text, lang, path)

    monkeypatch.setattr(index_mod, "extract", counting_extract)
    assert index.refresh().changed == 0
    assert parsed == []

    _bump(project / "app/agent.py", "class Agent:\n    def renamed(self):\n        pass\n")
    _make(project, "app/new.py", "def fresh():\n    pass\n")
    (project / "web/store.ts").unlink()
    stats = index.refresh()
    assert (stats.added, stats.updated, stats.removed) == (1, 1, 1)
    assert parsed == ["python", "python"]
    assert "web/store.ts" not in _paths(index)
    assert [symbol.qualname for symbol in index.outline("app/agent.py")] == [
        "Agent",
        "Agent.renamed",
    ]
    assert index.find_symbols("runTurn")[1] == 0


def test_refresh_subtree_leaves_rest_untouched(index: ProjectIndex, project: Path) -> None:
    index.refresh()
    (project / "web/store.ts").unlink()
    _make(project, "app/more.py", "def more():\n    pass\n")
    stats = index.refresh(project / "app")
    assert (stats.added, stats.removed) == (1, 0)
    assert "web/store.ts" in _paths(index)  # вне поддерева — не трогаем
    assert index.is_complete()  # частичное обновление не снимает признак полноты


def test_refresh_single_file(index: ProjectIndex, project: Path) -> None:
    index.refresh(project / "app" / "agent.py")
    assert _paths(index) == ["app/agent.py"]
    assert not index.is_complete()


def test_secrets_binary_large_and_ignored_files(
    index: ProjectIndex,
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(index_mod, "MAX_INDEX_FILE_BYTES", 100)
    _make(project, ".env", "GIGACREW_ACCESS_KEY=secret\n")
    _make(project, ".gitignore", "generated/\n")
    _make(project, "generated/big.py", "def hidden(): pass\n")
    (project / "logo.png").write_bytes(b"\x89PNG\0\0\0")
    _make(project, "huge.py", "def huge(): pass\n" * 20)
    index.refresh()
    paths = _paths(index)
    assert ".env" not in paths
    assert "generated/big.py" not in paths
    assert _status(index, "logo.png") == "binary"
    big = index.file_entry("huge.py")
    assert big is not None
    assert big.status == "large"
    assert big.symbols == 0
    # явное обновление исключённого каталога не протаскивает его в индекс
    index.refresh(project / "generated")
    assert "generated/big.py" not in _paths(index)


def test_interrupted_build_keeps_committed_batches(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(index_mod, "_BATCH", 2)
    for num in range(5):
        _make(project, f"many/m{num}.py", f"def f{num}(): pass\n")

    def stop_at(scanned: int) -> None:
        if scanned == 6:
            raise KeyboardInterrupt

    ws = Workspace(project)
    with ProjectIndex(ws) as ix, pytest.raises(KeyboardInterrupt):
        ix.refresh(progress=stop_at)
    with ProjectIndex(ws) as ix:
        assert not ix.is_complete()
        assert len(_paths(ix)) == 4  # две зафиксированные пачки по 2 файла
        stats = ix.refresh()
        assert stats.added == 5
        assert ix.is_complete()


def test_rebuild(index: ProjectIndex, project: Path) -> None:
    index.refresh()
    stats = index.rebuild()
    assert stats.added == 4
    assert index.is_complete()


# ------------------------------ совместимость ------------------------------ #
def test_old_schema_is_recreated(project: Path) -> None:
    ws = Workspace(project)
    with ProjectIndex(ws) as ix:
        ix.refresh()
    conn = sqlite3.connect(ws.index_dir / INDEX_FILE_NAME)
    with conn:
        conn.execute("UPDATE meta SET value = '0' WHERE key = 'schema_version'")
    conn.close()
    with ProjectIndex(ws) as ix:
        assert not ix.is_complete()
        assert ix.stats().files == 0


def test_corrupted_database_is_recreated(project: Path) -> None:
    ws = Workspace(project)
    ws.ensure_data_dir()
    ws.index_dir.mkdir()
    (ws.index_dir / INDEX_FILE_NAME).write_bytes(b"not a database at all" * 100)
    with ProjectIndex(ws) as ix:
        assert ix.refresh().added == 4


# ------------------------------ поиск ------------------------------ #
def test_find_symbols_ranking(index: ProjectIndex) -> None:
    index.refresh()
    hits, total = index.find_symbols("agent")
    assert total == 2
    assert [hit.symbol.name for hit in hits] == ["Agent", "AgentEvents"]  # точное — первым


def test_find_symbols_qualified_kind_glob_and_limit(index: ProjectIndex) -> None:
    index.refresh()
    hits, _ = index.find_symbols("Agent.run_turn")
    assert [(hit.path, hit.symbol.qualname) for hit in hits] == [
        ("app/agent.py", "Agent.run_turn"),
    ]
    # регистр не важен, 'function' включает методы
    hits, total = index.find_symbols("RUNTURN", kind="function")
    assert total == 1
    assert hits[0].path == "web/store.ts"
    hits, total = index.find_symbols("run", kind="function", path_glob="app/**")
    assert [hit.symbol.qualname for hit in hits] == ["Agent.run_turn"]
    hits, total = index.find_symbols("e", limit=1)
    assert len(hits) == 1
    assert total > 1
    assert index.find_symbols("100%_")[1] == 0  # спецсимволы LIKE экранируются
    with pytest.raises(ValueError, match="пустой запрос"):
        index.find_symbols("  ")


@pytest.mark.skipif(os.name == "nt", reason="регистр путей")
def test_subtree_is_case_sensitive(index: ProjectIndex, project: Path) -> None:
    _make(project, "Src/upper.py", "def upper(): pass\n")
    _make(project, "src/lower.py", "def lower(): pass\n")
    index.refresh()
    index.refresh(project / "src")
    assert "Src/upper.py" in _paths(index)  # соседний каталог в другом регистре не тронут
    assert [entry.path for entry in index.files_under("src")] == ["src/lower.py"]


def test_unreadable_file_is_retried_when_unchanged(
    index: ProjectIndex,
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = Path.read_bytes
    broken = {"app/agent.py"}

    def read_bytes(self: Path) -> bytes:
        if self.relative_to(project.resolve()).as_posix() in broken:
            raise PermissionError("нет доступа")
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    index.refresh()
    assert _status(index, "app/agent.py") == "error"
    broken.clear()  # права починили, файл не менялся
    assert index.refresh().updated == 1
    assert _status(index, "app/agent.py") == "indexed"
    assert index.refresh().changed == 0


@pytest.mark.skipif(os.name == "nt", reason="симлинки")
def test_symlinks_to_secrets_and_excluded_files_are_not_indexed(
    index: ProjectIndex,
    project: Path,
) -> None:
    _make(project, ".env", 'SECRET_KEY = "s3cr3t"\n')
    (project / "settings.py").symlink_to(project / ".env")
    _make(project, ".gitignore", "vendor/\n")
    _make(project, "vendor/lib.py", "def vendored(): pass\n")
    (project / "lib_link.py").symlink_to(project / "vendor" / "lib.py")
    index.refresh()
    paths = _paths(index)
    assert "settings.py" not in paths
    assert "lib_link.py" not in paths
    assert index.find_symbols("SECRET")[1] == 0
    assert index.find_symbols("vendored")[1] == 0


def test_secret_files_are_not_indexed_by_the_shell_rule(
    index: ProjectIndex,
    project: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    # Секреты — по правилу read_file и классификатора команд, а не только .env*.
    _make(project, "deploy/credentials.json", '{"TOKEN_A": 1}\n')
    _make(project, "config/secrets.yaml", "TOKEN_B: 1\n")
    _make(project, ".aws/settings.py", "TOKEN_C = 1\n")
    _make(project, "certs/server.key", "TOKEN_D = 1\n")
    outside = tmp_path_factory.mktemp("home") / ".aws"
    _make(outside, "conf.py", "TOKEN_E = 1\n")
    if os.name != "nt":
        (project / "conf.py").symlink_to(outside / "conf.py")
    index.refresh()
    assert index.find_symbols("TOKEN")[1] == 0
    assert _paths(index) == ["README.md", "app/agent.py", "app/events.py", "web/store.ts"]


# ------------------------- использования и импорты ------------------------- #
@pytest.fixture
def refs_project(tmp_path: Path) -> Iterator[ProjectIndex]:
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


def test_find_refs_ranks_by_resolution(refs_project: ProjectIndex) -> None:
    refs = refs_project.find_refs("HELPER")  # определение — без учёта регистра
    assert [(definition.path, definition.symbol.name) for definition in refs.definitions] == [
        ("pkg/core.py", "helper"),
    ]
    got = [(hit.path, hit.line, hit.kind, hit.resolution) for hit in refs.hits]
    assert got == [
        ("pkg/rel.py", 1, "import", "import"),
        ("pkg/rel.py", 5, "call", "import"),
        ("pkg/user.py", 1, "import", "import"),
        ("pkg/user.py", 5, "call", "import"),  # через псевдоним hp
        ("pkg/core.py", 6, "call", "same_file"),
        ("other.py", 2, "call", "name_only"),
    ]
    assert refs.total == 6
    assert refs_project.find_refs("helper", kind="import").total == 2
    calls = refs_project.find_refs("helper", kind="call", path_glob="pkg/**", limit=1)
    assert calls.total == 3
    assert len(calls.hits) == 1
    assert refs_project.find_refs("pkg.rel.go").hits == []  # без использований
    with pytest.raises(ValueError, match="пустой запрос"):
        refs_project.find_refs(" . ")


def test_find_refs_without_definition_searches_by_name(refs_project: ProjectIndex) -> None:
    refs = refs_project.find_refs("hp")
    assert refs.definitions == []
    assert [(hit.path, hit.line, hit.resolution) for hit in refs.hits] == [
        ("pkg/user.py", 5, "name_only"),
    ]


def test_imports_resolve_to_project_files(tmp_path: Path) -> None:
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
        assert ix.resolve_module("os") is None
        assert ix.resolve_module("..x") is None
        entries = ix.imports_of("src/pkg/a.py")
        assert [(entry.module, entry.name, entry.target) for entry in entries] == [
            ("os", "", None),
            ("pkg", "b", "src/pkg/b.py"),  # подмодуль, а не атрибут пакета
            ("src.pkg", "b", "src/pkg/b.py"),
            ("pkg.b", "", "src/pkg/b.py"),
            ("pkg.b", "B", "src/pkg/b.py"),
        ]
        assert ix.imported_by("src/pkg/b.py") == [("src/pkg/a.py", 2)]
        assert ix.imported_by("src/pkg/__init__.py") == []
        assert ix.imported_by("README.md") == []


def test_refs_and_imports_follow_incremental_refresh(
    index: ProjectIndex,
    project: Path,
) -> None:
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


def test_stub_only_package_resolves(tmp_path: Path) -> None:
    _make(tmp_path, "stubs/pkg/__init__.pyi", "def f() -> int: ...\n")
    _make(tmp_path, "app.py", "from pkg import f\n\nf()\n")
    with ProjectIndex(Workspace(tmp_path)) as ix:
        ix.refresh()
        assert ix.resolve_module("pkg") == "stubs/pkg/__init__.pyi"
        assert ix.imported_by("stubs/pkg/__init__.pyi") == [("app.py", 1)]


def test_recursive_call_on_definition_line_is_a_reference(tmp_path: Path) -> None:
    _make(tmp_path, "m.py", "def fact(n): return 1 if n < 2 else n * fact(n - 1)\n")
    with ProjectIndex(Workspace(tmp_path)) as ix:
        ix.refresh()
        hits = ix.find_refs("fact").hits
        assert [(hit.line, hit.kind, hit.scope) for hit in hits] == [(1, "call", "fact")]


def _corrupt_pages(ws: Workspace) -> None:
    """Порча страниц данных при целых заголовке и meta (сбой диска, запись наполовину)."""
    db = ws.index_dir / INDEX_FILE_NAME
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")  # страницы — в основной файл
    conn.close()
    size = db.stat().st_size
    # страницы 1–16 (схема, meta) целы, дальше — мусор
    with Path(db).open("r+b") as db_file:
        db_file.seek(16 * 4096)
        db_file.write(bytes(range(256)) * ((size - 16 * 4096) // 256))


def _big_project(root: Path) -> Workspace:
    for module_no in range(150):
        body = "\n".join(
            f"def func_{module_no}_{func_no}(x):\n    return helper_{func_no}(x)\n"
            for func_no in range(20)
        )
        (root / f"m{module_no}.py").write_text(body, encoding="utf-8")
    ws = Workspace(root)
    with ProjectIndex(ws) as ix:
        ix.refresh()
    return ws


def test_corrupted_data_pages_are_detected_and_index_recreated(tmp_path: Path) -> None:
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


def test_index_command_rebuilds_corrupted_index(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ws = _big_project(tmp_path)
    _corrupt_pages(ws)
    outcome = run_indexing(ProjectIndex(ws), Console(no_color=True))
    assert outcome is not None
    assert outcome[1].files == 150
    assert "повреждён" in capsys.readouterr().out
    with ProjectIndex(ws) as ix:
        assert ix.find_symbols("func_1_19")[1] == 1


# ---------------- запросы, где верное и ошибочное поведение различаются ---------------- #
def test_subtree_refresh_does_not_touch_sibling_with_same_prefix(tmp_path: Path) -> None:
    # Соседний каталог с тем же началом имени (app2 рядом с app) — не часть поддерева.
    _make(tmp_path, "app/a.py", "def a(): pass\n")
    _make(tmp_path, "app2/b.py", "def b(): pass\n")
    with ProjectIndex(Workspace(tmp_path)) as ix:
        ix.refresh()
        stats = ix.refresh(tmp_path / "app")
        assert stats.removed == 0
        assert _paths(ix) == ["app/a.py", "app2/b.py"]
        assert [entry.path for entry in ix.files_under("app")] == ["app/a.py"]


def test_edit_that_keeps_the_size_is_reindexed(tmp_path: Path) -> None:
    # Переименование foo → bar размер не меняет: изменение видно только по mtime.
    module = _make(tmp_path, "m.py", "def foo(): pass\n")
    with ProjectIndex(Workspace(tmp_path)) as ix:
        ix.refresh()
        st = module.stat()
        module.write_text("def bar(): pass\n", encoding="utf-8")
        os.utime(module, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        assert module.stat().st_size == st.st_size
        assert ix.refresh().updated == 1
        assert ix.find_symbols("bar")[1] == 1
        assert ix.find_symbols("foo")[1] == 0


def test_qualified_query_filters_by_owner(tmp_path: Path) -> None:
    _make(
        tmp_path,
        "m.py",
        "class A:\n    def run(self): pass\nclass B:\n    def run(self): pass\n",
    )
    _make(tmp_path, "n.py", "class Outer:\n    class Inner:\n        def run(self): pass\n")
    with ProjectIndex(Workspace(tmp_path)) as ix:
        ix.refresh()
        hits, total = ix.find_symbols("A.run")
        assert total == 1
        assert hits[0].symbol.qualname == "A.run"
        assert [hit.symbol.qualname for hit in ix.definitions("B.run")] == ["B.run"]
        # владелец — и по концу имени вложенного класса
        assert [hit.symbol.qualname for hit in ix.find_symbols("Inner.run")[0]] == [
            "Outer.Inner.run",
        ]


def test_definitions_are_exact_names_not_substrings(tmp_path: Path) -> None:
    _make(tmp_path, "m.py", "def run(): pass\ndef run_all(): pass\n\nrun_all()\n")
    with ProjectIndex(Workspace(tmp_path)) as ix:
        ix.refresh()
        assert [hit.symbol.name for hit in ix.definitions("run")] == ["run"]
        assert [hit.symbol.name for hit in ix.definitions("RUN")] == ["run"]
        assert ix.find_refs("run").total == 0  # run_all() — не использование run
        assert ix.find_refs("run_all").total == 1


def test_like_wildcards_in_query_are_literal(tmp_path: Path) -> None:
    _make(tmp_path, "m.py", "def get_x(): pass\ndef getax(): pass\ndef p100x(): pass\n")
    with ProjectIndex(Workspace(tmp_path)) as ix:
        ix.refresh()
        hits, total = ix.find_symbols("get_x")
        assert total == 1
        assert hits[0].symbol.name == "get_x"
        assert ix.find_symbols("100%")[1] == 0
        assert ix.find_symbols("p1%x")[1] == 0


def test_prefix_match_ranks_before_substring(tmp_path: Path) -> None:
    _make(tmp_path, "m.py", "def aget(): pass\ndef get_item(): pass\n")
    with ProjectIndex(Workspace(tmp_path)) as ix:
        ix.refresh()
        assert [hit.symbol.name for hit in ix.find_symbols("get")[0]] == ["get_item", "aget"]
