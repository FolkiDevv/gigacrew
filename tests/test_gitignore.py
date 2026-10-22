"""Тесты учёта .gitignore: разбор правил, стек по каталогам, обход и дерево."""

from __future__ import annotations

import os

import pytest

from gigacrew.project.files import build_file_tree, walk_files
from gigacrew.project.gitignore import IgnoreStack, parse_rule, parse_rules, stack_for
from gigacrew.project.workspace import Workspace
from gigacrew.tools.base import ToolContext
from gigacrew.tools.fs import FindFilesParams, FindFilesTool
from gigacrew.tools.search import SearchContentParams, SearchContentTool


def _make(root, rel, text="x"):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _rels(root, base=None):
    return [p.relative_to(root.resolve()).as_posix() for p in walk_files(root, base)]


def _ignored(rules: str, path: str, is_dir: bool = False) -> bool:
    return IgnoreStack().push("", parse_rules(rules)).is_ignored(path, is_dir)


# ------------------------------ правила ------------------------------ #
@pytest.mark.parametrize("line", ["", "   ", "# комментарий", "/", "!"])
def test_empty_and_comment_lines_give_no_rule(line):
    assert parse_rule(line) is None


@pytest.mark.parametrize(
    ("rules", "path", "is_dir", "expected"),
    [
        ("*.log", "a.log", False, True),
        ("*.log", "deep/x/a.log", False, True),  # без '/' — на любой глубине
        ("*.log", "a.txt", False, False),
        ("build/", "build", True, True),
        ("build/", "build", False, False),  # хвостовой '/' — только каталоги
        ("build/", "src/build", True, True),
        ("/build", "build", True, True),
        ("/build", "src/build", True, False),  # ведущий '/' — только от корня
        ("doc/*.txt", "doc/a.txt", False, True),
        ("doc/*.txt", "x/doc/a.txt", False, False),  # '/' в середине — якорь
        ("doc/*.txt", "doc/sub/a.txt", False, False),
        ("doc/**/*.txt", "doc/sub/a.txt", False, True),
        ("**/cache", "a/b/cache", True, True),
        ("logs/**", "logs/x/y.txt", False, True),
        ("*.log\n!keep.log", "keep.log", False, False),  # последнее совпадение побеждает
        ("!keep.log\n*.log", "keep.log", False, True),
        ("\\#name", "#name", False, True),
        ("\\!bang", "!bang", False, True),
        ("trail\\ ", "trail ", False, True),
        ("trail   ", "trail", False, True),  # неэкранированные хвостовые пробелы
        ("a[0-9].txt", "a5.txt", False, True),
    ],
)
def test_rule_semantics(rules, path, is_dir, expected):
    assert _ignored(rules, path, is_dir) is expected


def test_nested_rules_apply_to_subdir_and_override_parent(tmp_path):
    _make(tmp_path, ".gitignore", "*.gen\n")
    _make(tmp_path, "pkg/.gitignore", "!keep.gen\nlocal/\n")
    stack = stack_for(tmp_path, "pkg")
    assert stack.is_ignored("pkg/x.gen", False)
    assert not stack.is_ignored("pkg/keep.gen", False)
    assert stack.is_ignored("pkg/local", True)
    # правила подкаталога не действуют на соседей
    assert not stack_for(tmp_path, "").is_ignored("local", True)


def test_info_exclude_is_weaker_than_gitignore(tmp_path):
    _make(tmp_path, ".git/info/exclude", "*.tmp\nsecret/\n")
    _make(tmp_path, ".gitignore", "!keep.tmp\n")
    stack = stack_for(tmp_path, "")
    assert stack.is_ignored("a.tmp", False)
    assert stack.is_ignored("secret", True)
    assert not stack.is_ignored("keep.tmp", False)


# ------------------------------ обход ------------------------------ #
def test_walk_files_respects_gitignore(tmp_path):
    _make(tmp_path, ".gitignore", "*.log\nout/\n/top.txt\n")
    for rel in ("a.py", "app.log", "out/x.py", "src/top.txt", "top.txt", "src/b.log"):
        _make(tmp_path, rel)
    assert _rels(tmp_path) == [".gitignore", "a.py", "src/top.txt"]
    # без учёта .gitignore видно всё (кроме служебных каталогов)
    all_rels = [p.relative_to(tmp_path).as_posix() for p in walk_files(tmp_path, gitignore=False)]
    assert "app.log" in all_rels
    assert "out/x.py" in all_rels


def test_walk_files_from_subdir_keeps_parent_rules(tmp_path):
    _make(tmp_path, ".gitignore", "*.gen\n")
    _make(tmp_path, "pkg/.gitignore", "tmp/\n")
    for rel in ("pkg/a.py", "pkg/b.gen", "pkg/tmp/c.py", "pkg/sub/d.gen", "pkg/sub/e.py"):
        _make(tmp_path, rel)
    assert _rels(tmp_path, (tmp_path / "pkg").resolve()) == [
        "pkg/.gitignore",
        "pkg/a.py",
        "pkg/sub/e.py",
    ]


def test_negation_cannot_reinclude_file_in_ignored_dir(tmp_path):
    _make(tmp_path, ".gitignore", "vendor/\n!vendor/keep.py\n")
    _make(tmp_path, "vendor/keep.py")
    assert _rels(tmp_path) == [".gitignore"]


def test_unreadable_gitignore_is_skipped(tmp_path):
    (tmp_path / ".gitignore").mkdir()  # каталог вместо файла
    _make(tmp_path, "a.py")
    assert _rels(tmp_path) == ["a.py"]


def test_file_tree_respects_gitignore(tmp_path):
    _make(tmp_path, ".gitignore", "generated/\n*.bak\n")
    _make(tmp_path, "src/main.py")
    _make(tmp_path, "src/main.py.bak")
    _make(tmp_path, "generated/huge.py")
    tree = build_file_tree(tmp_path)
    assert "main.py" in tree
    assert "generated" not in tree
    assert ".bak" not in tree


def test_find_and_search_tools_respect_gitignore(tmp_path):
    _make(tmp_path, ".gitignore", "dist-*/\n")
    _make(tmp_path, "src/a.py", "needle = 1\n")
    _make(tmp_path, "dist-web/a.py", "needle = 2\n")
    ctx = ToolContext(workspace=Workspace(tmp_path))
    found = FindFilesTool().run(FindFilesParams(pattern="*.py"), ctx)
    assert found.content == "src/a.py"
    hits = SearchContentTool().run(SearchContentParams(pattern="needle"), ctx)
    assert "src/a.py:1" in hits.content
    assert "dist-web" not in hits.content


def test_is_excluded_matches_walk_decision(tmp_path):
    from gigacrew.project.files import is_excluded

    _make(tmp_path, ".gitignore", "out/\n*.log\n")
    _make(tmp_path, "pkg/.gitignore", "local.py\n")
    for rel in ("a.py", "out/x.py", "pkg/local.py", "pkg/ok.py", "node_modules/m.js", "b.log"):
        _make(tmp_path, rel)
    walked = set(_rels(tmp_path))
    for rel in ("a.py", "out/x.py", "pkg/local.py", "pkg/ok.py", "node_modules/m.js", "b.log"):
        assert is_excluded(tmp_path, tmp_path.resolve() / rel) is (rel not in walked), rel
    assert is_excluded(tmp_path, tmp_path.resolve() / "out")
    assert not is_excluded(tmp_path, tmp_path.resolve())
    assert is_excluded(tmp_path, tmp_path.parent)


def test_walk_files_accepts_unresolved_base(tmp_path):
    _make(tmp_path, ".gitignore", "*.gen\n")
    _make(tmp_path, "pkg/a.py")
    _make(tmp_path, "pkg/b.gen")
    base = tmp_path / "pkg" / ".." / "pkg"  # не раскрытый путь внутри корня
    assert _rels(tmp_path, base) == ["pkg/a.py"]


def test_malformed_pattern_does_not_break_traversal(tmp_path):
    """Битый класс символов ([z-a]) не роняет разбор и обход."""
    _make(tmp_path, ".gitignore", "[z-a]\n*.log\n")
    _make(tmp_path, "a.py")
    _make(tmp_path, "b.log")
    assert _rels(tmp_path) == [".gitignore", "a.py"]
    assert not _ignored("[z-a]", "a.py")  # правило просто ничего не исключает


@pytest.mark.parametrize(
    ("rules", "path", "expected"),
    [
        ("/foo**bar", "foobar", True),
        ("/foo**bar", "fooXbar", True),
        ("/foo**bar", "foo/x/bar", False),  # '**' не целым сегментом — это '*'
        ("a/**/b", "a/x/y/b", True),
        ("**/b", "x/y/b", True),
        ("a/**", "a/x/y", True),
    ],
)
def test_double_star_crosses_slash_only_as_whole_segment(rules, path, expected):
    assert _ignored(rules, path) is expected


@pytest.mark.skipif(os.name == "nt", reason="симлинки")
def test_symlinked_gitignore_is_not_read(tmp_path):
    """Как git: .gitignore-симлинк в рабочем дереве не действует."""
    _make(tmp_path, "rules.txt", "*.py\n")
    (tmp_path / ".gitignore").symlink_to(tmp_path / "rules.txt")
    _make(tmp_path, "a.py")
    assert "a.py" in _rels(tmp_path)
