"""Рекурсивный grep из run_shell — встроенным поиском по содержимому.

Модели ищут по коду привычным ``grep -rn X .``. Если в дереве есть секреты (``.env``),
ссылки наружу (``.venv``) или файлов слишком много, классификатор не признаёт такую
команду чтением, и её пришлось бы проверять рецензенту или пользователю. Когда команда
точно переводится во встроенный поиск (:func:`gigacrew.tools.search.search_paths`),
``run_shell`` выполняет его сам: файлы с секретами, из ``.gitignore``, служебные каталоги
и бинарные файлы пропускаются. Всё, что не переводится точно, остаётся командой.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from gigacrew.project.secret_paths import is_secret_target
from gigacrew.safety import MAX_COMMAND
from gigacrew.safety.bash import Command, Redirect
from gigacrew.safety.bash import parse as parse_command
from gigacrew.tools.base import ToolContext, ToolError
from gigacrew.tools.search import MAX_MATCHES, SearchFilter

__all__ = ["DEFAULT_LIMIT", "GrepSearch", "as_grep_search"]

DEFAULT_LIMIT = 200  # строк вывода без `| head`: grep вывел бы все
_HEAD_DEFAULT = 10  # `head` без числа
_PIPELINE_MAX = 2  # grep и, может быть, head

# Синтаксис образца: G — базовые регулярные выражения, E — расширенные, F — строка,
# P — Perl. egrep и fgrep — grep с -E и -F.
_PROGRAMS = MappingProxyType({"grep": "G", "egrep": "E", "fgrep": "F"})
_SYNTAX_LONG = MappingProxyType({
    "--basic-regexp": "G",
    "--extended-regexp": "E",
    "--fixed-strings": "F",
    "--perl-regexp": "P",
})
_RECURSIVE = "recursive"
_IGNORE_CASE = "ignore_case"
_WORD = "word"
_LINE = "line"
_FILES_ONLY = "files_only"
_SWITCH_SHORT = MappingProxyType({
    "r": _RECURSIVE,
    "R": _RECURSIVE,
    "i": _IGNORE_CASE,
    "y": _IGNORE_CASE,
    "w": _WORD,
    "x": _LINE,
    "l": _FILES_ONLY,
})
_SWITCH_LONG = MappingProxyType({
    "--recursive": _RECURSIVE,
    "--dereference-recursive": _RECURSIVE,
    "--ignore-case": _IGNORE_CASE,
    "--word-regexp": _WORD,
    "--line-regexp": _LINE,
    "--files-with-matches": _FILES_ONLY,
})
# Флаги, которые не меняют, какие строки найдутся: номера строк и имена файлов в выводе
# поиска есть всегда, бинарные файлы он пропускает.
_NEUTRAL_SHORT = frozenset("nHsI")
_NEUTRAL_LONG = frozenset((
    "--line-number",
    "--with-filename",
    "--no-messages",
    "--binary-files=without-match",
    "--color",
    "--colour",
))
_INCLUDE = "--include"
_EXCLUDE = "--exclude"
_EXCLUDE_DIR = "--exclude-dir"
_REGEXP = "--regexp"
_VALUED_LONG = frozenset((_REGEXP, _INCLUDE, _EXCLUDE, _EXCLUDE_DIR))
# Шаблон имён без кавычек оболочка раскрыла бы по файлам каталога: у этих флагов
# совпадений нет (имя начинается с `--`), и слово остаётся как есть.
_GLOB_PREFIXES = ("--include=", "--exclude=", "--exclude-dir=")
# В базовом синтаксисе эти символы — литералы, а с обратной косой — операторы; в Python наоборот.
_BASIC_OPERATORS = frozenset("|(){}?+")
_WORD_EDGES = frozenset("<>")  # \< и \> — границы слова
_BASIC_ENDS = (r"\)", r"\|")  # после `$` — конец выражения в базовом синтаксисе
_HEAD_COUNT = re.compile(r"-n?(\d+)|--lines=(\d+)")
_LINES_FLAGS = frozenset(("-n", "--lines"))
_QUIET_TARGET = "/dev/null"


@dataclass(frozen=True)
class GrepSearch:
    """Рекурсивный grep, переведённый во встроенный поиск.

    ``bases`` — раскрытые пути поиска, ``limit`` — сколько строк вывода (или файлов при
    ``files_only``, как ``grep -l``) показать, ``call`` — тот же поиск вызовом
    search_content (для пометки модели).
    """

    regex: re.Pattern[str]
    bases: tuple[Path, ...]
    search_filter: SearchFilter
    files_only: bool
    limit: int
    call: str


@dataclass
class _GrepArgs:
    """Разобранные аргументы grep; ``chosen`` — синтаксис задан программой или флагом."""

    syntax: str
    chosen: bool
    switches: set[str] = field(default_factory=set)
    patterns: list[str] = field(default_factory=list)
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    exclude_dirs: list[str] = field(default_factory=list)
    positional: list[str] = field(default_factory=list)
    filters: list[str] = field(default_factory=list)  # --include/--exclude по порядку

    def parse(self, words: list[str]) -> bool:
        """Разбирает аргументы; False — есть то, что поиск не повторит точно."""
        index = 0
        options = True
        while index < len(words):
            word = words[index]
            index += 1
            if word == "-":  # stdin
                return False
            if options and word == "--":
                options = False
            elif options and word.startswith("-"):
                taken = self._option(word, words[index : index + 1])
                if taken is None:
                    return False
                index += taken
            else:
                self.positional.append(word)
        return True

    def bases(self, command: Command, ctx: ToolContext) -> tuple[Path, ...] | None:
        """Раскрытые пути поиска (без путей — каталог команды); None — путь не годится.

        Пути — позиционные после образца (если он не задан через -e). Не годится путь вне
        проекта и подключённых каталогов, несуществующий и файл с секретами, названный
        явно: такую команду решает классификатор, как раньше.
        """
        paths = self.positional if self.patterns else self.positional[1:]
        start = ctx.root / command.cwd
        bases: list[Path] = []
        for text in paths or ["."]:
            try:
                base = ctx.resolve(start / text)
            except ToolError:
                return None
            if not base.exists() or (base.is_file() and is_secret_target(ctx.root, base)):
                return None
            bases.append(base)
        return tuple(bases)

    def search_filter(self) -> SearchFilter | None:
        """Фильтр файлов; None — шаблоны, которые поиск понял бы не так, как grep."""
        patterns = (*self.include, *self.exclude, *self.exclude_dirs)
        if any("/" in pattern or "{" in pattern for pattern in patterns):
            return None
        # --exclude перед --include: grep читает и файлы, не попавшие ни под один шаблон
        if self.include and self.filters[0] != _INCLUDE:
            return None
        return SearchFilter(tuple(self.include), tuple(self.exclude), tuple(self.exclude_dirs))

    def regex(self) -> re.Pattern[str] | None:
        """Образец как регулярное выражение Python; None — не переводится точно."""
        sources = self.patterns or self.positional[:1]
        lines = [line for source in sources for line in source.split("\n")]
        parts: list[str] = []
        for line in lines:
            translated = _translate(line, self.syntax)
            if translated is None:
                return None
            parts.append(translated)
        if not parts:
            return None
        body = parts[0] if len(parts) == 1 else "|".join(f"(?:{part})" for part in parts)
        if _WORD in self.switches:
            body = rf"(?<!\w)(?:{body})(?!\w)"
        if _LINE in self.switches:
            body = f"^(?:{body})$"
        flags = re.IGNORECASE if _IGNORE_CASE in self.switches else 0
        try:
            return re.compile(body, flags)
        except re.error:
            return None

    def _option(self, word: str, following: list[str]) -> int | None:
        """Флаг: сколько следующих слов он занял; None — флаг не поддержан."""
        nxt = following[0] if following else None
        if word.startswith("--"):
            return self._long(word, nxt)
        return self._short(word[1:], nxt)

    def _long(self, word: str, following: str | None) -> int | None:
        """Длинный флаг: сколько следующих слов он занял; None — флаг не поддержан."""
        name, equals, inline = word.partition("=")
        if name in _VALUED_LONG:
            if equals:
                self._add(name, inline)
                return 0
            if following is None:
                return None
            self._add(name, following)
            return 1
        if word in _NEUTRAL_LONG or name in {"--color", "--colour"}:
            return 0
        switch = _SWITCH_LONG.get(word)
        if switch is not None:
            self.switches.add(switch)
            return 0
        syntax = _SYNTAX_LONG.get(word)
        return None if syntax is None or not self._choose(syntax) else 0

    def _short(self, letters: str, following: str | None) -> int | None:
        """Группа коротких флагов (``rni`` из ``-rni``); ``-e`` забирает остаток или слово."""
        for pos, letter in enumerate(letters):
            if letter == "e":
                rest = letters[pos + 1 :]
                if rest:
                    self._add(_REGEXP, rest)
                    return 0
                if following is None:
                    return None
                self._add(_REGEXP, following)
                return 1
            if not self._switch(letter):
                return None
        return 0

    def _switch(self, letter: str) -> bool:
        """Короткий флаг без значения; False — не поддержан."""
        if letter in _NEUTRAL_SHORT:
            return True
        switch = _SWITCH_SHORT.get(letter)
        if switch is not None:
            self.switches.add(switch)
            return True
        return letter in "EFGP" and self._choose(letter)

    def _choose(self, syntax: str) -> bool:
        """Задать синтаксис образца; два разных (``egrep -F``) grep считает ошибкой."""
        if self.chosen and syntax != self.syntax:
            return False
        self.syntax = syntax
        self.chosen = True
        return True

    def _add(self, name: str, text: str) -> None:
        """Значение флага: образец или шаблон имён."""
        if name == _REGEXP:
            self.patterns.append(text)
            return
        self.filters.append(name if name == _INCLUDE else _EXCLUDE)
        if name == _INCLUDE:
            self.include.append(text)
        elif name == _EXCLUDE:
            self.exclude.append(text)
        else:
            self.exclude_dirs.append(text)


def as_grep_search(command: str, ctx: ToolContext, cwd: str = ".") -> GrepSearch | None:
    """Команда ``command`` (в каталоге ``cwd`` от корня) как встроенный поиск.

    Переводится только рекурсивный grep (``grep -r``, ``egrep``, ``fgrep``), необязательно
    с ``| head -N`` и перенаправлениями в ``/dev/null``, с флагами, смысл которых поиск
    повторяет точно; пути поиска — внутри проекта или подключённых каталогов. None —
    команда не такая.
    """
    if len(command) > MAX_COMMAND:
        return None
    extra = tuple(ctx.workspace.extra_dirs)
    script = parse_command(command, root=ctx.root, cwd=cwd, extra=extra)
    if script is None or script.problems or script.compound or script.assigns:
        return None
    pipeline = _pipeline(script.commands)
    if pipeline is None:
        return None
    grep, limit = pipeline
    return _grep(grep, limit, ctx)


def _pipeline(commands: list[Command]) -> tuple[Command, int] | None:
    """Grep и лимит вывода из ``| head``; None — другие команды или звенья."""
    if not commands or len(commands) > _PIPELINE_MAX or not all(map(_simple, commands)):
        return None
    grep = commands[0]
    if grep.stdin is not None:
        return None
    if len(commands) == 1:
        return grep, DEFAULT_LIMIT
    head = commands[-1]
    limit = _head_limit(head) if head.stdin is grep else None
    return None if limit is None else (grep, min(limit, MAX_MATCHES))


def _simple(command: Command) -> bool:
    """Простая команда со словами, известными до запуска, и без записи в файлы."""
    if command.assigns or command.substituted or not command.argv:
        return False
    if not all(map(_quiet, command.redirects)):
        return False
    return all(
        word.static and (not word.glob or word.text.startswith(_GLOB_PREFIXES))
        for word in command.argv
    )


def _quiet(redirect: Redirect) -> bool:
    """Перенаправление не меняет, что увидит модель: дублирование или в /dev/null."""
    if redirect.fd_dup:
        return True
    target = redirect.target
    return redirect.writes and target is not None and target.text == _QUIET_TARGET


def _head_limit(command: Command) -> int | None:
    """Сколько строк оставит ``head``; None — это не ``head -N``."""
    if command.argv[0].text != "head":
        return None
    args = [word.text for word in command.args]
    if not args:
        return _HEAD_DEFAULT
    if len(args) == 2 and args[0] in _LINES_FLAGS:
        args = [f"-n{args[1]}"]
    match = _HEAD_COUNT.fullmatch(args[0]) if len(args) == 1 else None
    if match is None:
        return None
    count = int(match.group(1) or match.group(2))
    return count or None


def _grep(command: Command, limit: int, ctx: ToolContext) -> GrepSearch | None:
    """Поиск по аргументам рекурсивного grep; None — перевод не точный."""
    syntax = _PROGRAMS.get(command.argv[0].text)
    if syntax is None:
        return None
    args = _GrepArgs(syntax, chosen=syntax != "G")
    if not args.parse([word.text for word in command.args]) or _RECURSIVE not in args.switches:
        return None
    regex = args.regex()
    search_filter = args.search_filter()
    bases = args.bases(command, ctx)
    if regex is None or search_filter is None or bases is None:
        return None
    files_only = _FILES_ONLY in args.switches
    call = _call(regex, bases, search_filter, ctx)
    return GrepSearch(regex, bases, search_filter, files_only, limit, call)


def _call(
    regex: re.Pattern[str],
    bases: tuple[Path, ...],
    search_filter: SearchFilter,
    ctx: ToolContext,
) -> str:
    """Тот же поиск вызовами search_content (по одному на путь)."""
    extra = ""
    if len(search_filter.include) == 1:
        extra = f", glob={_quote(search_filter.include[0])}"
    if regex.flags & re.IGNORECASE:
        extra = f"{extra}, ignore_case=true"
    pattern = _quote(regex.pattern)
    calls: list[str] = []
    for base in bases:
        path = _quote(ctx.display(base))
        calls.append(f"search_content(pattern={pattern}, path={path}{extra})")
    return "; ".join(calls)


def _quote(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def _translate(pattern: str, syntax: str) -> str | None:
    """Образец grep в синтаксисе Python; None — не переводится точно."""
    if syntax == "F":
        return re.escape(pattern)
    out: list[str] = []
    index = 0
    while index < len(pattern):
        # начало выражения — начало образца, после \( или \|: от этого зависят * и ^
        start = not out or out[-1] in {"(", "|"}
        token = _token(pattern, index, basic=syntax == "G", start=start)
        if token is None:
            return None
        text, index = token
        out.append(text)
    return "".join(out)


def _token(pattern: str, index: int, *, basic: bool, start: bool) -> tuple[str, int] | None:
    """Очередной элемент образца в синтаксисе Python и позиция после него."""
    char = pattern[index]
    if char == "[":
        return _bracket(pattern, index)
    if char == "\\":
        return _escape(pattern, index, basic=basic)
    if basic and _basic_literal(pattern, index, start=start):
        return re.escape(char), index + 1
    return char, index + 1


def _basic_literal(pattern: str, index: int, *, start: bool) -> bool:
    """Символ базового синтаксиса — литерал, хотя в Python он оператор.

    ``| ( ) { } ? +`` — всегда; ``*`` — в начале выражения (``start``), ``^`` — не в
    начале; ``$`` — не в конце выражения.
    """
    char = pattern[index]
    if char in _BASIC_OPERATORS:
        return True
    if char == "*":
        return start
    if char == "^":
        return not start
    if char != "$":
        return False
    rest = pattern[index + 1 :]
    return bool(rest) and not rest.startswith(_BASIC_ENDS)


def _escape(pattern: str, index: int, *, basic: bool) -> tuple[str, int] | None:
    r"""Последовательность с обратной косой: ``\|`` — оператор, ``\<`` — граница слова."""
    nxt = pattern[index + 1 : index + 2]
    if not nxt:
        return None
    end = index + 2
    if nxt in _WORD_EDGES:
        return r"\b", end
    if basic and nxt in _BASIC_OPERATORS:
        return nxt, end
    return pattern[index:end], end


def _bracket(pattern: str, index: int) -> tuple[str, int] | None:
    """Скобочное выражение ``[...]`` как есть; классы POSIX (``[[:space:]]``) — None."""
    pos = index + 1
    if pattern.startswith("^", pos):
        pos += 1
    if pattern.startswith("]", pos):  # `]` сразу после `[` или `[^` — литерал
        pos += 1
    end = pattern.find("]", pos)
    if end < 0:
        return None
    body = pattern[index : end + 1]
    if any(mark in body for mark in ("[:", "[=", "[.")):
        return None
    return body, end + 1
