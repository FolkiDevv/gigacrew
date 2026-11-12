"""Перевод образцов grep в Python: те же строки, что находит сам GNU grep."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from gigacrew.tools.grep_pattern import translate

_GREP = shutil.which("grep")

# Строки, на которых сравниваются grep и перевод.
_LINES = (
    "a",
    "b",
    "ab",
    "aa",
    "f(x)",
    "x.py",
    "xpy",
    "$HOME",
    "the end",
    "end here",
    "foo bar",
    "food",
    "]",
    "n",
    "\\",
    r"\n",
    "w",
    "a.b",
    "axb",
    "ac",
    "xx",
    "x{2}",
    "word_1",
    "#comment",
    "a|b",
    "a+",
    "^top",
    "1d",
)


def _grep_lines(flag: str, pattern: str, sample: Path) -> list[int]:
    assert _GREP is not None
    proc = subprocess.run(
        [_GREP, "-n", flag, "-e", pattern, str(sample)],
        capture_output=True,
        text=True,
        check=False,
    )
    return [int(line.split(":", 1)[0]) for line in proc.stdout.splitlines()]


@pytest.mark.skipif(_GREP is None, reason="нет grep")
@pytest.mark.parametrize(
    ("flag", "pattern"),
    [
        ("-G", r"a\|b"),
        ("-G", "f(x)"),
        ("-G", r"a\{2\}"),
        ("-G", "*.py"),
        ("-G", "$HOME"),
        ("-G", "end$"),
        ("-G", "^the"),
        ("-G", "a^"),
        ("-G", r"\<foo\>"),
        ("-G", "[]a]"),
        ("-G", r"[\n]"),
        ("-G", r"x\.py"),
        ("-G", r"\w\+_1"),
        ("-G", "a|b"),
        ("-G", "a+"),
        ("-G", r"\(a\|b\)c"),
        ("-E", "a|b"),
        ("-E", "(a|b)c"),
        ("-E", "x{2}"),
        ("-E", r"x\{2\}"),
        ("-E", r"\w+_1"),
        ("-E", r"[\w]"),
        ("-E", "^#"),
        ("-E", r"\^top"),
        ("-E", r"a\+"),
        ("-F", "a.b"),
        ("-F", "f(x)"),
    ],
)
def test_translation_matches_grep(flag: str, pattern: str, tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    text = "\n".join(_LINES)
    sample.write_text(f"{text}\n", encoding="utf-8")
    regex = translate(pattern, flag[1])
    assert regex is not None
    compiled = re.compile(regex)
    ours = [num for num, line in enumerate(_LINES, start=1) if compiled.search(line)]
    assert ours == _grep_lines(flag, pattern, sample)


@pytest.mark.parametrize(
    ("syntax", "pattern"),
    [
        ("G", r"\d"),  # в grep — «d»
        ("E", r"\n"),
        ("G", r"\(a\)\1"),
        ("G", "[[:space:]]"),
        ("E", "a+?"),
        ("E", "(?:a)"),
        ("E", "(a+)+"),
        ("E", "(a|aa)*"),
        ("E", "((a)b+)*"),
        ("P", r"(\w+\s?)*$"),
        ("G", "foo\\"),
        ("G", "[abc"),
    ],
)
def test_untranslatable_patterns(syntax: str, pattern: str) -> None:
    assert translate(pattern, syntax) is None


@pytest.mark.parametrize(
    ("syntax", "pattern"),
    [("E", "(foo)+"), ("P", "(?:ab)?c+"), ("P", r"(?<!\w)foo\d+"), ("E", "[(a+)]+")],
)
def test_simple_repeats_are_kept(syntax: str, pattern: str) -> None:
    assert translate(pattern, syntax) is not None
