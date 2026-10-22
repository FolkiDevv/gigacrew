r"""Правила ``.gitignore``: разбор, стек правил по каталогам, glob → regex.

Поддерживается то, что встречается в реальных проектах: комментарии, ``!`` —
отрицание, хвостовой ``/`` — только каталоги, ``/`` в начале или середине — шаблон
привязан к каталогу файла правил (иначе сравнивается с именем на любой глубине),
``**``, экранирование ``\\``. Правила вложенных ``.gitignore`` действуют на свой
подкаталог и перекрывают правила выше; последнее совпавшее правило побеждает.
Корень дополнительно читает ``.git/info/exclude``.

Не читаются: ``.gitignore`` выше корня проекта и глобальный ``core.excludesFile``.
Решение «обходить ли каталог» принимает :mod:`gigacrew.project.files`.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from pathlib import Path

GITIGNORE_NAME = ".gitignore"
_MAX_RULES_BYTES = 1_000_000  # больше — явно не файл правил


@functools.lru_cache(maxsize=1024)
def compile_glob(pattern: str) -> re.Pattern[str]:
    r"""Glob → regex: ``**`` — любое число каталогов, ``*``/``?`` — в пределах сегмента.

    Как в ``.gitignore``, ``**`` пересекает ``/`` только целым сегментом пути
    (``**/x``, ``a/**``, ``a/**/b``); в остальных местах (``foo**bar``) это ``*``.

    ``\\x`` — символ ``x`` буквально (как в ``.gitignore``). Шаблон, который не
    компилируется (``[z-a]``), сравнивается как обычная строка.
    """
    out: list[str] = []
    i, n = 0, len(pattern)
    while i < n:
        c = pattern[i]
        if c == "\\" and i + 1 < n:
            out.append(re.escape(pattern[i + 1]))
            i += 2
            continue
        if c == "*":
            whole_segment = (i == 0 or pattern[i - 1] == "/") and (
                i + 2 == n or pattern[i + 2 : i + 3] == "/"
            )
            if pattern.startswith("**", i) and whole_segment:
                i += 2
                if i < n and pattern[i] == "/":
                    i += 1
                    out.append("(?:.*/)?")
                else:
                    out.append(".*")
                continue
            while i + 1 < n and pattern[i + 1] == "*":
                i += 1  # "***", "**" внутри сегмента — то же, что "*"
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[":
            j = i + 1
            if j < n and pattern[j] in "!^":
                j += 1
            if j < n and pattern[j] == "]":
                j += 1
            while j < n and pattern[j] != "]":
                j += 1
            if j >= n:
                out.append(re.escape(c))
            else:
                body = pattern[i + 1 : j]
                if body[0] in "!^":
                    body = "^" + body[1:]
                out.append("[" + body.replace("\\", "\\\\") + "]")
                i = j
        else:
            out.append(re.escape(c))
        i += 1
    try:
        return re.compile("".join(out) + r"\Z")
    except re.error:
        return re.compile(re.escape(pattern) + r"\Z")


@dataclass(frozen=True)
class IgnoreRule:
    regex: re.Pattern[str]  # сравнивается с путём относительно каталога файла правил
    negate: bool = False
    dir_only: bool = False

    def matches(self, rel_posix: str, is_dir: bool) -> bool:
        if self.dir_only and not is_dir:
            return False
        return self.regex.match(rel_posix) is not None


def _strip_trailing_spaces(line: str) -> str:
    r"""Хвостовые пробелы отбрасываются, если не экранированы (``foo\\ ``)."""
    end = len(line)
    while end > 0 and line[end - 1] == " ":
        backslashes = 0
        k = end - 2
        while k >= 0 and line[k] == "\\":
            backslashes += 1
            k -= 1
        if backslashes % 2:
            break
        end -= 1
    return line[:end]


def parse_rule(line: str) -> IgnoreRule | None:
    """Одна строка ``.gitignore`` → правило (None — пустая строка или комментарий)."""
    line = _strip_trailing_spaces(line.rstrip("\r\n"))
    if not line or line.startswith("#"):
        return None
    negate = line.startswith("!")
    if negate or line.startswith(("\\!", "\\#")):
        line = line[1:]
    dir_only = line.endswith("/") and not line.endswith("\\/")
    line = line.rstrip("/")
    if not line:
        return None
    anchored = "/" in line
    line = line.lstrip("/")
    if not line:
        return None
    pattern = line if anchored else f"**/{line}"
    return IgnoreRule(compile_glob(pattern), negate=negate, dir_only=dir_only)


def parse_rules(text: str) -> tuple[IgnoreRule, ...]:
    return tuple(rule for rule in map(parse_rule, text.splitlines()) if rule is not None)


def _read_rules(path: Path, *, follow_symlink: bool = True) -> tuple[IgnoreRule, ...]:
    """Правила из файла; отсутствующий, нечитаемый или огромный файл — нет правил."""
    try:
        if not follow_symlink and path.is_symlink():
            return ()
        if not path.is_file() or path.stat().st_size > _MAX_RULES_BYTES:
            return ()
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ()
    return parse_rules(text)


@dataclass(frozen=True)
class IgnoreStack:
    """Правила, действующие в каталоге: пары (каталог от корня, правила), сверху вниз."""

    levels: tuple[tuple[str, tuple[IgnoreRule, ...]], ...] = ()

    def push(self, rel_dir: str, rules: tuple[IgnoreRule, ...]) -> IgnoreStack:
        if not rules:
            return self
        return IgnoreStack((*self.levels, (rel_dir, rules)))

    def enter(self, root: Path, rel_dir: str) -> IgnoreStack:
        """Стек для подкаталога ``rel_dir``: добавляются правила его ``.gitignore``."""
        directory = root / rel_dir if rel_dir else root
        # как git (2.32+): .gitignore-симлинк в рабочем дереве не читается
        return self.push(rel_dir, _read_rules(directory / GITIGNORE_NAME, follow_symlink=False))

    def is_ignored(self, rel_posix: str, is_dir: bool) -> bool:
        """Путь (от корня проекта) исключён правилами; решает последнее совпадение."""
        ignored = False
        for base, rules in self.levels:
            if base:
                if not rel_posix.startswith(base + "/"):
                    continue
                sub = rel_posix[len(base) + 1 :]
            else:
                sub = rel_posix
            for rule in rules:
                if rule.matches(sub, is_dir):
                    ignored = not rule.negate
        return ignored


def root_stack(root: Path) -> IgnoreStack:
    """Правила корня: ``.git/info/exclude`` (слабее), затем ``.gitignore``."""
    stack = IgnoreStack().push("", _read_rules(root / ".git" / "info" / "exclude"))
    return stack.enter(root, "")


def stack_for(root: Path, rel_dir: str) -> IgnoreStack:
    """Стек правил для каталога ``rel_dir`` — с правилами всех каталогов выше него."""
    stack = root_stack(root)
    if not rel_dir:
        return stack
    parts = rel_dir.split("/")
    for i in range(1, len(parts) + 1):
        stack = stack.enter(root, "/".join(parts[:i]))
    return stack
