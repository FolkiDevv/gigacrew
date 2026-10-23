"""Разбор команды оболочки в простые команды — по синтаксическому дереву tree-sitter-bash.

Команду выполняет ``/bin/sh -c`` (на Debian/Ubuntu — dash, на macOS — bash в режиме
POSIX), а грамматика — bash. Поэтому разбор **консервативный**: всё, что оболочки
понимают по-разному или что нельзя оценить без выполнения (функции, фон, ``$'…'``,
раскрытие ``{a,b}``, ``cd`` посреди команды), попадает в ``problems`` — такую команду
классификатор без вопроса не пропустит. Узел незнакомого типа — тоже проблема: значит,
грамматика понимает больше, чем правила разбора.

Результат — :class:`Script`: простые команды (слова со статичностью и глобами,
префиксные присваивания, перенаправления, heredoc, предыдущее звено конвейера),
рабочий каталог после ведущих ``cd X &&`` и сама команда без них.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from gigacrew.project import treesitter

__all__ = ["Command", "Redirect", "Script", "Word", "parse"]

# Подстановки и раскрытия: значение слова известно только при выполнении.
_DYNAMIC_NODES = frozenset((
    "simple_expansion",
    "expansion",
    "command_substitution",
    "process_substitution",
    "arithmetic_expansion",
    "ansi_c_string",  # $'…': dash и bash понимают по-разному
    "translated_string",  # $"…"
))
_SUBSTITUTIONS = frozenset(("command_substitution", "process_substitution"))
_SUBSTITUTION_DELIMITERS = frozenset(("$(", ")", "`", "<(", ">("))
_WORD_NODES = frozenset((
    "word",
    "string",
    "raw_string",
    "concatenation",
    "number",
    "simple_expansion",
    "expansion",
    "command_substitution",
    "process_substitution",
    "arithmetic_expansion",
    "ansi_c_string",
    "translated_string",
))
# Узлы слова, которые разбираются отдельно: простое слово и склейка частей.
_WORD = "word"
_CONCATENATION = "concatenation"
# Составные операторы: внутри — обычные команды (условия, циклы, блоки).
_CONTAINERS = frozenset((
    "do_group",
    "else_clause",
    "elif_clause",
    "case_item",
    "if_statement",
    "while_statement",
    "for_statement",
    "case_statement",
    "negated_command",
    "compound_statement",
))
# Лексемы-разделители и ключевые слова — пропускаются.
_TOKENS = frozenset((
    ";",
    ";;",
    "&",
    "&&",
    "||",
    "|",
    "|&",
    "(",
    ")",
    "{",
    "}",
    "!",
    "if",
    "then",
    "fi",
    "else",
    "elif",
    "for",
    "in",
    "do",
    "done",
    "while",
    "until",
    "case",
    "esac",
    "comment",
    "\n",
))
_SEPARATORS = frozenset(("&&", "||", ";", "&", "\n", "|", "|&", ";;"))
_OUTPUT_OPS = frozenset((">", ">>", ">|", "&>", "&>>", "&>|"))
_GLOB_CHARS = re.compile(r"[*?\[]")
# Экранированный символ: внутри него ни глоб, ни раскрытие {…} не действуют.
_ESCAPED = re.compile(r"\\.")
# Текст без разбора внутри (слово, тело heredoc) и неэкранированная ` в нём.
_PLAIN_TEXT = frozenset(("word", "heredoc_body", "heredoc_content"))
_BACKTICK = re.compile(r"(?<!\\)(?:\\\\)*`")
_BRACE = re.compile(r"\{[^{}\0]*(,|\.\.)[^{}\0]*\}")
# Ссылка на переменную «как есть» ($X, ${X}) — только внутри этих узлов слова: в арифметике
# $((X)) и в ${X:-…}, ${!X}, ${X@P} значение не просто подставляется.
_PLAIN_PARENTS = frozenset(("string", "concatenation", "heredoc_body"))
# Значение присваивания, которое при подстановке не раскроется ни во что другое: без $, `,
# шаблонов, кавычек, пробелов.
_PLAIN_VALUE = re.compile(r"[\w./:,@%+=-]*")
_ASSIGNMENTS = frozenset(("variable_assignment", "variable_assignments", "declaration_command"))
# Встроенные команды, которые могут присвоить переменную оболочки (или выполнить то, что
# присвоит): после них присвоенное раньше значение неизвестно.
_SETS_VARS = frozenset((
    "read",
    "getopts",
    "printf",
    "mapfile",
    "readarray",
    "let",
    "eval",
    ".",
    "source",
    "declare",
    "typeset",
    "local",
    "export",
    "readonly",
    "unset",
    "set",
    "command",
    "builtin",
    "exec",
    "trap",
))


class _Node(Protocol):
    """Узел дерева tree-sitter — то, что из него нужно разбору.

    Сам tree-sitter загружает и импортирует только :mod:`gigacrew.project.treesitter`.
    """

    @property
    def type(self) -> str: ...

    @property
    def text(self) -> bytes | None: ...

    @property
    def children(self) -> Sequence[_Node]: ...

    @property
    def named_children(self) -> Sequence[_Node]: ...

    @property
    def child_count(self) -> int: ...

    @property
    def parent(self) -> _Node | None: ...

    @property
    def is_missing(self) -> bool: ...

    @property
    def start_byte(self) -> int: ...

    def child_by_field_name(self, name: str, /) -> _Node | None: ...


@dataclass(frozen=True)
class Word:
    """Слово команды.

    ``text`` — значение после снятия кавычек (у нестатичного слова — исходный текст);
    ``variables`` — раскрываемые переменные; ``glob`` — оболочка раскроет шаблон имён
    файлов (``*``, ``?``, ``[`` вне кавычек); ``plain`` — те из ``variables``, что только
    подставляются как есть (``$X``, ``${X}``).
    """

    text: str
    raw: str = ""
    static: bool = True
    glob: bool = False
    variables: tuple[str, ...] = ()
    plain: tuple[str, ...] = ()


@dataclass(frozen=True)
class Redirect:
    """Перенаправление.

    ``target`` — файл (``>``, ``>>``, ``<``, ``&>``); у heredoc и herestring — ``body``
    (``body_static`` — без подстановок, ``body_vars`` — раскрываемые в нём переменные).
    """

    op: str
    target: Word | None = None
    fd_dup: bool = False  # >&2, 2>&1 — дублирование дескриптора, не файл
    body: str | None = None
    body_static: bool = True
    body_vars: tuple[str, ...] = ()
    body_plain: tuple[str, ...] = ()  # как Word.plain

    @property
    def writes(self) -> bool:
        """Перенаправление пишет в файл."""
        return self.op in _OUTPUT_OPS or (self.op == ">&" and not self.fd_dup)

    @property
    def reads(self) -> bool:
        """Перенаправление читает файл."""
        return self.op in {"<", "<&"} and not self.fd_dup and self.target is not None


@dataclass
class Command:
    """Простая команда в контексте.

    Контекст — рабочий каталог (от корня, POSIX), перенаправления и предыдущее звено
    конвейера (его вывод — stdin этой команды).
    """

    argv: list[Word]
    assigns: list[tuple[str, Word]] = field(default_factory=list)
    redirects: list[Redirect] = field(default_factory=list)
    cwd: str = "."
    stdin: Command | None = None
    substituted: bool = False  # внутри $(…), `…` или <(…)
    # Переменные, которым команда раньше, в этой же оболочке и без условий, присвоила
    # постоянное значение (``X=1; echo $X``): их значение известно — это не секрет.
    assigned: frozenset[str] = frozenset()

    @property
    def name(self) -> Word | None:
        """Имя команды — первое слово; None — слов нет."""
        return self.argv[0] if self.argv else None

    @property
    def args(self) -> list[Word]:
        """Аргументы — слова после имени команды."""
        return self.argv[1:]

    @property
    def heredoc(self) -> Redirect | None:
        """Ввод из heredoc/herestring (последний — как в оболочке)."""
        found = None
        for redirect in self.redirects:
            if redirect.body is not None:
                found = redirect
        return found


@dataclass
class Script:
    """Разобранная команда: простые команды, проблемы разбора и рабочий каталог."""

    commands: list[Command] = field(default_factory=list)
    # Почему команду нельзя оценить без выполнения («не знаю»).
    problems: list[str] = field(default_factory=list)
    # Присваивания вне команд (``X=1``, ``export X=1``): действуют на всё, что дальше.
    assigns: list[tuple[str, Word]] = field(default_factory=list)
    cwd: str = "."  # рабочий каталог после ведущих ``cd X &&``
    rest: str = ""  # команда без ведущих ``cd X &&``
    compound: bool = False  # есть условия, циклы или блоки { …; }
    consumed: int = 0  # сколько ведущих `cd X &&` превратилось в рабочий каталог


# --------------------------------------------------------------------------- #
# Значения слов
# --------------------------------------------------------------------------- #
def _text(node: _Node) -> str:
    return "" if node.text is None else node.text.decode("utf-8", errors="replace")


def _unescape_word(raw: str) -> str:
    r"""Неквотированное слово: ``\\x`` → ``x``, перенос строки через ``\\`` убирается."""
    return re.sub(
        r"\\(.)",
        lambda escape: "" if escape.group(1) == "\n" else escape.group(1),
        raw,
        flags=re.DOTALL,
    )


def _unescape_dquoted(raw: str) -> str:
    r"""Содержимое ``"…"``: ``\\`` значима только перед ``$ ` " \\`` и переводом строки."""
    return re.sub(
        r'\\([$`"\\\n])',
        lambda escape: "" if escape.group(1) == "\n" else escape.group(1),
        raw,
    )


def _unescape_heredoc(body: str) -> str:
    r"""Тело heredoc без кавычек у разделителя: ``\\$``, ``\\```, ``\\\\``, перенос через ``\\``."""
    return re.sub(
        r"\\([$`\\\n])",
        lambda escape: "" if escape.group(1) == "\n" else escape.group(1),
        body,
    )


def _static_value(node: _Node) -> str:
    kind = node.type
    if kind == "raw_string":
        return _text(node)[1:-1]
    if kind == "string":
        # Текст между кавычками, а не дети узла: string_content рвётся на переводах
        # строк, и сами переводы строк в детей не попадают.
        text = _text(node)
        quoted = len(text) >= 2 and text[0] == text[-1] == '"'
        inner = text[1:-1] if quoted else text
        return _unescape_dquoted(inner)
    if kind == _CONCATENATION:
        return "".join(_static_value(child) for child in node.children)
    if kind == _WORD:
        return _unescape_word(_text(node))
    return _text(node)


def _plain_vars(node: _Node) -> tuple[str, ...]:
    """Переменные, которые в слове ``node`` только подставляются как есть (см. Word.plain).

    Команды в подстановках ``$(…)`` — отдельные команды, в них не заходим.
    """
    plain: set[str] = set()
    other: set[str] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type in _SUBSTITUTIONS:
            continue
        if current.type == "variable_name":
            (plain if _is_plain_ref(current, node) else other).add(_text(current))
        stack.extend(current.children)
    return tuple(sorted(plain - other))


def _is_plain_ref(ref: _Node, top: _Node) -> bool:
    exp = ref.parent
    if exp is None:
        return False
    if exp.type == "expansion":
        if [child.type for child in exp.children] != ["${", "variable_name", "}"]:
            return False
    elif exp.type != "simple_expansion":
        return False
    current = exp
    while current != top:
        parent = current.parent
        if parent is None or (parent != top and parent.type not in _PLAIN_PARENTS):
            return False
        current = parent
    return True


def _has_glob(node: _Node) -> bool:
    if node.type == _WORD:
        return bool(_GLOB_CHARS.search(_ESCAPED.sub("", _text(node))))
    if node.type == _CONCATENATION:
        return any(_has_glob(child) for child in node.children)
    return False


def _has_brace(node: _Node) -> bool:
    """``{a,b}``/``{1..3}`` вне кавычек: bash раскроет, dash — нет."""
    if node.type not in {_WORD, _CONCATENATION}:
        return False
    parts = list(node.children) if node.type == _CONCATENATION else [node]
    pieces = (_ESCAPED.sub("", _text(part)) if part.type == _WORD else "\0" for part in parts)
    return bool(_BRACE.search("".join(pieces)))


class _Walker:
    """Обход дерева команды: простые команды, присваивания и проблемы — в :class:`Script`."""

    def __init__(self, root: Path, cwd: str) -> None:
        self.root = root
        self.script = Script(cwd=cwd)
        self._cwd = cwd  # каталог оператора, который сейчас разбирается
        self._assigned: set[str] = set()  # см. Command.assigned
        self.rest_start: int | None = None  # начало команды после ведущих cd

    def problem(self, text: str) -> None:
        if text not in self.script.problems:
            self.script.problems.append(text)

    # ----------------------------- слова ------------------------------ #
    def scan(self, node: _Node) -> tuple[bool, list[str]]:
        """Обойти поддерево слова: (есть ли раскрытия, имена переменных).

        Команды в подстановках разбираются как самостоятельные (они тоже выполнятся).
        """
        dynamic = False
        names: list[str] = []
        stack = [node]
        while stack:
            current = stack.pop()
            kind = current.type
            if kind in _DYNAMIC_NODES:
                dynamic = True
            if kind in {"variable_name", "special_variable_name"} and dynamic:
                names.append(_text(current))
            if kind in _SUBSTITUTIONS:
                inner = [
                    child
                    for child in current.children
                    if child.type not in _SUBSTITUTION_DELIMITERS
                ]
                self.statements(inner, cwd=self._cwd, substituted=True)
                continue
            if kind == "raw_string":
                continue
            self._check_parsed(current)
            stack.extend(current.children)
        return dynamic, names

    def word(self, node: _Node) -> Word:
        raw = _text(node)
        dynamic, names = self.scan(node)
        if dynamic:
            return Word(raw, raw, static=False, variables=tuple(names), plain=_plain_vars(node))
        if _has_brace(node):
            return Word(raw, raw, static=False)
        return Word(_static_value(node), raw, glob=_has_glob(node))

    # --------------------------- операторы ---------------------------- #
    def statements(
        self,
        nodes: Iterable[_Node],
        *,
        cwd: str,
        substituted: bool = False,
        leading_cd: bool = False,
        top: bool = False,
    ) -> str:
        """Последовательность операторов. Возвращает каталог после ведущих ``cd``."""
        flat: list[tuple[_Node, str]] = []  # (узел, разделитель после него)
        self._flatten(nodes, flat)
        leading = leading_cd
        for index, (node, sep) in enumerate(flat):
            if leading:
                target = self._leading_cd(node, sep, cwd=cwd, last=index == len(flat) - 1)
                if target is not None:
                    cwd = target
                    self.script.consumed += 1
                    continue
            if leading and top:
                self.rest_start = node.start_byte
            leading = False
            if sep == "&":
                self.problem("фоновый процесс (&)")
            saved = self._cwd
            self._cwd = cwd
            known = len(self.script.assigns)
            self.statement(node, cwd=cwd, substituted=substituted)
            self._cwd = saved
            # Присваивание запоминается, только если оно точно выполнено в этой оболочке:
            # на верхнем уровне, не после && или || и не в фоне.
            certain = (
                top
                and node.type in _ASSIGNMENTS
                and (index == 0 or flat[index - 1][1] not in {"&&", "||"})
                and sep != "&"
            )
            self._track_assigned(self.script.assigns[known:], certain=certain)
        return cwd

    def statement(
        self,
        node: _Node,
        *,
        cwd: str,
        substituted: bool,
        redirects: list[Redirect] | None = None,
    ) -> list[Command]:
        """Оператор → его простые команды (последняя — источник для следующего звена конвейера)."""
        kind = node.type
        if kind == "command":
            return [self.command(node, cwd=cwd, substituted=substituted, extra=redirects)]
        if kind == "redirected_statement":
            return self.redirected(node, cwd=cwd, substituted=substituted, outer=redirects)
        if kind == "pipeline":
            return self.pipeline(node, cwd=cwd, substituted=substituted)
        if kind == "subshell":
            return self._subshell(node, cwd=cwd, substituted=substituted, redirects=redirects)
        if kind in _CONTAINERS:
            return self._container(node, cwd=cwd, substituted=substituted, redirects=redirects)
        self._other_statement(node)
        return []

    def assignment(self, node: _Node) -> tuple[str, Word]:
        name = node.child_by_field_name("name")
        value_node = node.child_by_field_name("value")
        word = Word("") if value_node is None else self.word(value_node)
        if name is not None and name.type != "variable_name":  # a[1]=…, a+=…
            self.problem("сложное присваивание")
        return ("?" if name is None else _text(name), word)

    def declaration(self, node: _Node) -> None:
        keyword = _text(node.children[0]) if node.children else ""
        if keyword not in {"export", "local", "readonly", "declare", "typeset"}:
            self.problem(f"конструкция оболочки {keyword}")
        for child in node.named_children:
            if child.type == "variable_assignment":
                self.script.assigns.append(self.assignment(child))
            elif child.type in _WORD_NODES:
                word = self.word(child)
                if word.text.startswith("-") or keyword != "export":
                    self.problem(f"{keyword} {word.text}".strip())
                else:  # export NAME — значение берётся из присваивания раньше
                    self.script.assigns.append((word.text, Word("", static=False)))

    def redirected(
        self,
        node: _Node,
        *,
        cwd: str,
        substituted: bool,
        outer: list[Redirect] | None,
    ) -> list[Command]:
        body: _Node | None = None
        redirects: list[Redirect] = []
        # То, что грамматика поместила внутрь heredoc: `| python`, `&& ls`.
        trailing: list[_Node] = []
        for child in node.children:
            if child.type in {"file_redirect", "herestring_redirect"}:
                redirect, extra = self.redirect(child)
                redirects.append(redirect)
                if extra:
                    self.problem("слова после перенаправления")
            elif child.type == "heredoc_redirect":
                redirect, extra = self.heredoc(child)
                redirects.append(redirect)
                trailing.extend(extra)
            elif body is None:
                body = child
            else:
                self.problem(f"конструкция оболочки {child.type}")
        if body is None:
            self.problem("перенаправление без команды")
            return []
        cmds = self.statement(
            body,
            cwd=cwd,
            substituted=substituted,
            redirects=redirects + list(outer or []),
        )
        if trailing:
            cmds = self._trailing(trailing, cmds, cwd=cwd, substituted=substituted)
        return cmds

    def redirect(self, node: _Node) -> tuple[Redirect, list[_Node]]:
        """Перенаправление и лишние слова (``echo hi > f more`` — ``more`` попадает сюда)."""
        op = ""
        words: list[_Node] = []
        for child in node.children:
            if child.type in {">", ">>", ">|", "&>", "&>>", "<", ">&", "<&", "<>", "<<<", "&>|"}:
                op = child.type
            elif child.type in _WORD_NODES:
                words.append(child)
            elif child.type != "file_descriptor":
                self.problem(f"перенаправление {child.type}")
        if not words:
            self.problem("перенаправление без цели")
            return Redirect(op), []
        first = self.word(words[0])
        if node.type == "herestring_redirect":
            redirect = Redirect(
                op,
                body=first.text,
                body_static=first.static,
                body_vars=first.variables,
                body_plain=first.plain,
            )
            return redirect, words[1:]
        if op in {">&", "<&"} and first.static and (first.text.isdigit() or first.text == "-"):
            return Redirect(op, first, fd_dup=True), words[1:]
        if op == "<>":
            self.problem("перенаправление <>")
        return Redirect(op, first), words[1:]

    def heredoc(self, node: _Node) -> tuple[Redirect, list[_Node]]:
        quoted = False
        body = ""
        static = True
        names: list[str] = []
        plain: tuple[str, ...] = ()
        extra: list[_Node] = []
        for child in node.children:
            if child.type == "heredoc_start":
                quoted = bool(re.search(r"['\"\\]", _text(child)))
            elif child.type == "heredoc_body":
                body = _text(child)
                if not quoted:
                    static, names, plain = self._heredoc_vars(child, body)
            elif child.type in {"<<", "<<-", "heredoc_end", "heredoc_content"}:
                continue
            else:
                extra.append(child)
        if not quoted:
            body = _unescape_heredoc(body)
        redirect = Redirect(
            "<<",
            body=body,
            body_static=static,
            body_vars=tuple(names),
            body_plain=plain,
        )
        return redirect, extra

    def command(
        self,
        node: _Node,
        *,
        cwd: str,
        substituted: bool,
        extra: list[Redirect] | None = None,
    ) -> Command:
        cmd = Command(argv=[], cwd=cwd, substituted=substituted, assigned=frozenset(self._assigned))
        for child in node.children:
            self._command_part(cmd, child)
        if extra:
            cmd.redirects.extend(extra)
        self.script.commands.append(cmd)
        # Префикс `X=… :` у специальной встроенной команды в dash остаётся в оболочке, а
        # встроенные read, printf -v, eval… присваивают сами: значение больше не известно.
        for name, _ in cmd.assigns:
            self._assigned.discard(name)
        if cmd.name is None or not cmd.name.static or cmd.name.text in _SETS_VARS:
            self._assigned.clear()
        return cmd

    def pipeline(self, node: _Node, *, cwd: str, substituted: bool) -> list[Command]:
        prev: Command | None = None
        last: list[Command] = []
        for child in node.children:
            if child.type in {"|", "|&"}:
                continue
            got = self.statement(child, cwd=cwd, substituted=substituted)
            if got and prev is not None:
                got[0].stdin = prev
            prev = got[-1] if got else None
            last = got
        return last

    # ------------------------- вспомогательные ------------------------ #
    def _check_parsed(self, node: _Node) -> None:
        """Проблема, если грамматика не разобрала узел слова (или `…` в нём)."""
        if node.type == "ERROR" or node.is_missing:
            self.problem("команда не разобрана")
        if node.child_count == 0 and node.type in _PLAIN_TEXT and _BACKTICK.search(_text(node)):
            # Грамматика не разбирает `…` в теле heredoc и внутри ${…}, а оболочка
            # выполнит: команда в нём не проверена.
            self.problem("подстановка `…`, которую не удалось разобрать")

    def _leading_cd(self, node: _Node, sep: str, *, cwd: str, last: bool) -> str | None:
        """Каталог после ведущего ``cd X``; None — оператор не такой ``cd``."""
        # Только «cd X && …» (или cd в конце): после «;» и перевода строки (у него
        # нет лексемы, разделитель пуст) команда выполнится и там, где cd не сработал.
        if _is_cd(node) and (sep == "&&" or (not sep and last)):
            return self._cd_target(node, cwd)
        return None

    def _track_assigned(self, assigns: Iterable[tuple[str, Word]], *, certain: bool) -> None:
        """Запомнить постоянные значения присваиваний (см. Command.assigned) или забыть их."""
        for name, word in assigns:
            if certain and word.static and _PLAIN_VALUE.fullmatch(word.text):
                self._assigned.add(name)
            else:
                self._assigned.discard(name)

    def _flatten(self, nodes: Iterable[_Node], flat: list[tuple[_Node, str]]) -> None:
        significant = [child for child in nodes if child.type != "comment"]
        for index, node in enumerate(significant):
            if node.type == "ERROR" or node.is_missing:
                self.problem("команда не разобрана")
                continue
            if node.type in _TOKENS:
                continue
            if node.type == "list":
                self._flatten(node.children, flat)
                if flat:
                    flat[-1] = (flat[-1][0], _separator(significant, index))
                continue
            flat.append((node, _separator(significant, index)))

    def _cd_target(self, node: _Node, cwd: str) -> str | None:
        """Каталог ``cd DIR`` внутри проекта (существующий); None — не ведущий cd."""
        parts = [child for child in node.children if child.type != "command_name"]
        if len(parts) != 1 or parts[0].type not in _WORD_NODES:
            return None
        word = self.word(parts[0])
        if not word.static or word.glob or not word.text or word.text.startswith(("~", "-")):
            return None
        target = (self.root / cwd / word.text).resolve()
        try:
            rel = target.relative_to(self.root)
        except ValueError:
            return None
        if not target.is_dir():
            return None
        return rel.as_posix() or "."

    def _subshell(
        self,
        node: _Node,
        *,
        cwd: str,
        substituted: bool,
        redirects: list[Redirect] | None,
    ) -> list[Command]:
        """Подоболочка ``( … )``: ведущие ``cd`` в ней — свои."""
        before = len(self.script.commands)
        inner = [child for child in node.children if child.type not in {"(", ")"}]
        self.statements(inner, cwd=cwd, substituted=substituted, leading_cd=True)
        return _attach(self.script.commands[before:], redirects)

    def _container(
        self,
        node: _Node,
        *,
        cwd: str,
        substituted: bool,
        redirects: list[Redirect] | None,
    ) -> list[Command]:
        """Составной оператор: условие, цикл, блок ``{ …; }`` или отрицание ``!``."""
        if node.type != "negated_command":
            self.script.compound = True
        before = len(self.script.commands)
        for child in node.children:
            if child.type in _TOKENS or child.type == "variable_name":
                continue
            if child.type in _WORD_NODES:  # for x in СЛОВА; case СЛОВО in
                self.word(child)
            elif child.type == "test_command":
                self.scan(child)
            else:
                self.statements([child], cwd=cwd, substituted=substituted)
        return _attach(self.script.commands[before:], redirects)

    def _other_statement(self, node: _Node) -> None:
        """Оператор без простых команд: проверка, присваивание, объявление, функция."""
        kind = node.type
        if kind == "test_command":
            self.scan(node)
            return
        if kind == "variable_assignment":
            self.script.assigns.append(self.assignment(node))
            return
        if kind == "variable_assignments":
            for child in node.named_children:
                self.script.assigns.append(self.assignment(child))
            return
        if kind == "declaration_command":
            self.declaration(node)
            return
        if kind == "function_definition":
            self.problem("определение функции")
            return
        self.problem(f"конструкция оболочки {kind}")

    def _trailing(
        self,
        nodes: Iterable[_Node],
        cmds: list[Command],
        *,
        cwd: str,
        substituted: bool,
    ) -> list[Command]:
        """Продолжение команды, которое грамматика поместила внутрь heredoc."""
        prev = cmds[-1] if cmds else None
        last = cmds
        op = ""
        for node in nodes:
            if node.type == "&":
                self.problem("фоновый процесс (&)")
            if node.type in _SEPARATORS:
                op = node.type
                continue
            if node.type in {"file_redirect", "herestring_redirect"}:
                self._redirect_each(node, last)
                continue
            if node.type == "pipeline":
                prev, last = self._trailing_pipeline(
                    node,
                    prev,
                    last,
                    cwd=cwd,
                    substituted=substituted,
                )
                continue
            if node.type == "list":
                prev, last = self._trailing_list(
                    node,
                    prev,
                    last,
                    cwd=cwd,
                    substituted=substituted,
                    piped=False,
                )
                continue
            prev, last = self._follow(
                node,
                prev,
                cwd=cwd,
                substituted=substituted,
                piped=op in {"|", "|&"},
            )
        return last

    def _redirect_each(self, node: _Node, cmds: list[Command]) -> None:
        """Перенаправление после heredoc — каждой из команд ``cmds``."""
        redirect, extra = self.redirect(node)
        if extra:
            self.problem("слова после перенаправления")
        for cmd in cmds:
            cmd.redirects.append(redirect)

    def _trailing_pipeline(
        self,
        node: _Node,
        prev: Command | None,
        last: list[Command],
        *,
        cwd: str,
        substituted: bool,
    ) -> tuple[Command | None, list[Command]]:
        """Конвейер после heredoc: продолжает конвейер с ``prev``.

        Возвращает новые ``prev`` и команды последнего оператора (как :meth:`_trailing`).
        """
        for part in node.children:
            if part.type in {"|", "|&"}:
                continue
            if part.type == "list":
                # `cat <<EOF | python3 && ls`: грамматика кладёт `python3 && ls`
                # в одно звено; первое продолжает конвейер, остальное — дальше.
                prev, last = self._trailing_list(
                    part,
                    prev,
                    last,
                    cwd=cwd,
                    substituted=substituted,
                    piped=True,
                )
                continue
            got = self.statement(part, cwd=cwd, substituted=substituted)
            if got and prev is not None:
                got[0].stdin = prev
            prev = got[-1] if got else None
            last = got
        return prev, last

    def _trailing_list(
        self,
        node: _Node,
        prev: Command | None,
        last: list[Command],
        *,
        cwd: str,
        substituted: bool,
        piped: bool,
    ) -> tuple[Command | None, list[Command]]:
        """Список операторов после heredoc; ``piped`` — первый читает вывод ``prev``.

        Возвращает новые ``prev`` и команды последнего оператора (как :meth:`_trailing`).
        """
        flat: list[tuple[_Node, str]] = []
        self._flatten(node.children, flat)
        for index, (child, _) in enumerate(flat):
            prev, last = self._follow(
                child,
                prev,
                cwd=cwd,
                substituted=substituted,
                piped=piped and index == 0,
            )
        return prev, last

    def _follow(
        self,
        node: _Node,
        prev: Command | None,
        *,
        cwd: str,
        substituted: bool,
        piped: bool,
    ) -> tuple[Command | None, list[Command]]:
        """Оператор после ``prev``; ``piped`` — первая его команда читает вывод ``prev``.

        Возвращает новый ``prev`` (последнюю команду оператора, если они есть) и команды
        оператора.
        """
        got = self.statement(node, cwd=cwd, substituted=substituted)
        if piped and got and prev is not None:
            got[0].stdin = prev
        return (got[-1] if got else prev), got

    def _heredoc_vars(self, node: _Node, body: str) -> tuple[bool, list[str], tuple[str, ...]]:
        """Тело heredoc без кавычек у разделителя: (статично ли, переменные, «как есть»)."""
        dynamic, names = self.scan(node)
        plain = _plain_vars(node)
        # `…` и $… в теле без кавычек у разделителя раскрываются оболочкой.
        static = not dynamic and not re.search(r"(?<!\\)[`$]", body)
        return static, names, plain

    def _command_part(self, cmd: Command, node: _Node) -> None:
        """Элемент простой команды: присваивание-префикс, слово или перенаправление."""
        kind = node.type
        if kind == "variable_assignment":
            cmd.assigns.append(self.assignment(node))
            return
        if kind == "command_name":
            cmd.argv.append(self.word(node.children[0] if node.children else node))
            return
        if kind in _WORD_NODES:
            cmd.argv.append(self.word(node))
            return
        if kind in {"file_redirect", "herestring_redirect"}:
            redirect, rest = self.redirect(node)
            cmd.redirects.append(redirect)
            cmd.argv.extend(self.word(word_node) for word_node in rest)
            return
        if kind == "heredoc_redirect":
            redirect, rest = self.heredoc(node)
            cmd.redirects.append(redirect)
            if rest:
                self.problem("сложная команда с heredoc")
            return
        if kind != "comment":
            self.problem(f"элемент команды {kind}")


def _attach(cmds: list[Command], redirects: list[Redirect] | None) -> list[Command]:
    """Перенаправления составного оператора — каждой его простой команде."""
    if redirects:
        for cmd in cmds:
            cmd.redirects.extend(redirects)
    return cmds


def _separator(nodes: Sequence[_Node], index: int) -> str:
    for node in nodes[index + 1 :]:
        if node.type in _SEPARATORS:
            return node.type
        if node.type != "comment":
            return ""
    return ""


def _is_cd(node: _Node) -> bool:
    if node.type != "command":
        return False
    name = node.child_by_field_name("name")
    return name is not None and _text(name) == "cd"


def parse(command: str, *, root: Path, cwd: str = ".") -> Script | None:
    """Разобрать команду; None — грамматика bash не загрузилась.

    ``cwd`` — каталог запуска относительно ``root`` (параметр инструмента).
    """
    tree = treesitter.parse_tree("shell", command)
    if tree is None:
        return None
    walker = _Walker(Path(root).resolve(), cwd)
    if tree.has_error:
        walker.problem("команда не разобрана")
    final = walker.statements(list(tree.children), cwd=cwd, leading_cd=True, top=True)
    script = walker.script
    script.cwd = final
    if final == cwd:
        script.rest = command.strip()
    elif walker.rest_start is None:
        script.rest = ""  # только cd
    else:
        encoded = command.encode("utf-8", errors="surrogateescape")
        script.rest = encoded[walker.rest_start :].decode("utf-8", errors="surrogateescape").strip()
    return script
