"""Тесты перевода рекурсивного grep из run_shell во встроенный поиск."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from gigacrew.project.workspace import Workspace
from gigacrew.tools.base import ToolContext
from gigacrew.tools.grep_search import DEFAULT_LIMIT, GrepSearch, as_grep_search
from gigacrew.tools.search import SearchFilter


@pytest.fixture
def tree(tmp_path: Path) -> ToolContext:
    """Проект с каталогом src и файлом с секретами в корне."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("foo = 1\n", encoding="utf-8")
    (tmp_path / ".env").write_text("TOKEN=foo\n", encoding="utf-8")
    return ToolContext(workspace=Workspace(tmp_path))


def _search(command: str, ctx: ToolContext) -> GrepSearch:
    search = as_grep_search(command, ctx)
    assert search is not None, command
    return search


@pytest.mark.parametrize(
    ("command", "pattern"),
    [
        ("grep -rn foo .", "foo"),
        ("grep -rnI foo", "foo"),
        ("grep foo -rn .", "foo"),
        ("grep -rn --color=never -- -foo .", "-foo"),
        (r"grep -r 'a\|b' .", "a|b"),
        ("egrep -r 'a|b' .", "a|b"),
        ("grep -rE 'a|b' .", "a|b"),
        ("grep -r 'f(x)' .", r"f\(x\)"),
        (r"grep -r 'a\{2\}' .", "a{2}"),
        ("grep -r '*.py' .", r"\*.py"),
        ("grep -rn '$HOME' .", r"\$HOME"),
        ("grep -rn 'end$' .", "end$"),
        (r"grep -r '\<foo\>' .", r"\bfoo\b"),
        ("grep -r '[]a]' .", "[]a]"),
        ("grep -rF 'a.b' .", r"a\.b"),
        ("fgrep -r 'a.b' .", r"a\.b"),
        ("grep -rw foo .", r"(?<!\w)(?:foo)(?!\w)"),
        ("grep -rx foo .", "^(?:foo)$"),
        ("grep -r -e a -e b .", "(?:a)|(?:b)"),
        ("grep -r -efoo .", "foo"),
        ("grep -r --regexp=foo .", "foo"),
    ],
)
def test_pattern_translation(command: str, pattern: str, tree: ToolContext) -> None:
    assert _search(command, tree).regex.pattern == pattern


def test_ignore_case_and_files_only(tree: ToolContext) -> None:
    search = _search("grep -rli Foo .", tree)
    assert search.regex.flags & re.IGNORECASE
    assert search.files_only
    assert search.regex.search("FOO = 1")


def test_paths_relative_to_command_directory(tree: ToolContext) -> None:
    src = tree.root / "src"
    assert _search("grep -rn foo src", tree).bases == (src,)
    assert _search("cd src && grep -rn foo .", tree).bases == (src,)
    assert as_grep_search("grep -rn foo .", tree, cwd="src") == _search("grep -rn foo src", tree)
    assert _search("grep -rn foo", tree).bases == (tree.root,)
    assert _search("grep -rn foo src .", tree).bases == (src, tree.root)


@pytest.mark.parametrize(
    ("command", "limit"),
    [
        ("grep -rn foo .", DEFAULT_LIMIT),
        ("grep -rn foo . | head", 10),
        ("grep -rn foo . | head -20", 20),
        ("grep -rn foo . | head -n 5", 5),
        ("grep -rn foo . | head -n5", 5),
        ("grep -rn foo . | head --lines=7", 7),
        ("grep -rn foo . 2>/dev/null | head -3", 3),
        ("grep -rn foo . 2>&1", DEFAULT_LIMIT),
    ],
)
def test_output_limit(command: str, limit: int, tree: ToolContext) -> None:
    assert _search(command, tree).limit == limit


def test_file_name_filters(tree: ToolContext) -> None:
    command = "grep -rn --include='*.py' --include=*.md --exclude-dir=build --exclude x.py foo ."
    assert _search(command, tree).search_filter == SearchFilter(
        include=("*.py", "*.md"),
        exclude=("x.py",),
        exclude_dirs=("build",),
    )


@pytest.mark.parametrize(
    "command",
    [
        "grep -n foo src/app.py",  # не рекурсивный
        "grep -rc foo .",
        "grep -rv foo .",
        "grep -rn -A3 foo .",
        "grep -rn -m1 foo .",
        "grep -rn --max-count=1 foo .",
        "grep -rn --incl='*.py' foo .",  # сокращённый длинный флаг
        "grep -r -f patterns.txt .",
        "egrep -rF foo .",  # два синтаксиса
        "grep -rn foo* .",  # шаблон раскроет оболочка
        "grep -rn $PATTERN .",
        "grep -rn foo . | sort",
        "grep -rn foo . | head -n -2",
        "cat src/app.py | grep -rn foo",
        "grep -rn foo . > out.txt",
        "grep -rn foo . && ls",
        "for d in src; do grep -rn foo $d; done",
        "grep -rn foo /etc",  # вне проекта
        "grep -rn foo missing",
        "grep -rn foo .env",  # файл с секретами назван явно
        "grep -rn '[[:space:]]foo' .",
        r"grep -rn 'foo\' .",
        "grep -rE '(' .",
        "grep -rn --exclude='*.txt' --include='*.py' foo .",
        "grep -rn --include='src/*.py' foo .",
        "grep -rn - .",
        "rg foo",
    ],
)
def test_not_translated(command: str, tree: ToolContext) -> None:
    assert as_grep_search(command, tree) is None


def test_call_shows_equivalent_search(tree: ToolContext) -> None:
    search = _search("grep -rni --include='*.py' 'Foo' src", tree)
    assert search.call == (
        'search_content(pattern="Foo", path="src", glob="*.py", ignore_case=true)'
    )
