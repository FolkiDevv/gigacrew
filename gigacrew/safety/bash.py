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
from dataclasses import dataclass, field
from pathlib import Path

from gigacrew.project import treesitter

__all__ = ["Command", "Redirect", "Script", "Word", "parse"]

# Подстановки и раскрытия: значение слова известно только при выполнении.
_DYNAMIC_NODES = frozenset(
    {
        "simple_expansion",
        "expansion",
        "command_substitution",
        "process_substitution",
        "arithmetic_expansion",
        "ansi_c_string",  # $'…': dash и bash понимают по-разному
        "translated_string",  # $"…"
    }
)
_SUBSTITUTIONS = frozenset({"command_substitution", "process_substitution"})
_SUBSTITUTION_DELIMITERS = frozenset({"$(", ")", "`", "<(", ">("})
_WORD_NODES = frozenset(
    {
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
    }
)
# Составные операторы: внутри — обычные команды (условия, циклы, блоки).
_CONTAINERS = frozenset(
    {
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
    }
)
# Лексемы-разделители и ключевые слова — пропускаются.
_TOKENS = frozenset(
    {
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
    }
)
_SEPARATORS = frozenset({"&&", "||", ";", "&", "\n", "|", "|&", ";;"})
_OUTPUT_OPS = frozenset({">", ">>", ">|", "&>", "&>>", "&>|"})
_GLOB_CHARS = re.compile(r"[*?\[]")
# Текст без разбора внутри (слово, тело heredoc) и неэкранированная ` в нём.
_PLAIN_TEXT = frozenset({"word", "heredoc_body", "heredoc_content"})
_BACKTICK = re.compile(r"(?<!\\)(?:\\\\)*`")
_BRACE = re.compile(r"\{[^{}\0]*(,|\.\.)[^{}\0]*\}")


@dataclass(frozen=True)
class Word:
    """Слово команды. ``value`` — значение после снятия кавычек (у нестатичного слова —
    исходный текст); ``vars`` — раскрываемые переменные; ``glob`` — оболочка раскроет
    шаблон имён файлов (``*``, ``?``, ``[`` вне кавычек)."""

    value: str
    raw: str = ""
    static: bool = True
    glob: bool = False
    vars: tuple[str, ...] = ()


@dataclass(frozen=True)
class Redirect:
    """Перенаправление. ``target`` — файл (``>``, ``>>``, ``<``, ``&>``); у heredoc и
    herestring — ``body`` (``body_static`` — без подстановок, ``body_vars`` —
    раскрываемые в нём переменные)."""

    op: str
    target: Word | None = None
    fd_dup: bool = False  # >&2, 2>&1 — дублирование дескриптора, не файл
    body: str | None = None
    body_static: bool = True
    body_vars: tuple[str, ...] = ()

    @property
    def writes(self) -> bool:
        return self.op in _OUTPUT_OPS or (self.op == ">&" and not self.fd_dup)

    @property
    def reads(self) -> bool:
        return self.op in ("<", "<&") and not self.fd_dup and self.target is not None


@dataclass
class Command:
    """Простая команда в контексте: рабочий каталог (от корня, POSIX), перенаправления
    и предыдущее звено конвейера (его вывод — stdin этой команды)."""

    argv: list[Word]
    assigns: list[tuple[str, Word]] = field(default_factory=list)
    redirects: list[Redirect] = field(default_factory=list)
    cwd: str = "."
    stdin: Command | None = None
    substituted: bool = False  # внутри $(…), `…` или <(…)

    @property
    def name(self) -> Word | None:
        return self.argv[0] if self.argv else None

    @property
    def args(self) -> list[Word]:
        return self.argv[1:]

    @property
    def heredoc(self) -> Redirect | None:
        """Ввод из heredoc/herestring (последний — как в оболочке)."""
        found = None
        for r in self.redirects:
            if r.body is not None:
                found = r
        return found


@dataclass
class Script:
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
def _text(node) -> str:
    return node.text.decode("utf-8", errors="replace") if node.text is not None else ""


def _unescape_word(raw: str) -> str:
    """Неквотированное слово: ``\\x`` → ``x``, перенос строки через ``\\`` убирается."""
    return re.sub(r"\\(.)", lambda m: "" if m.group(1) == "\n" else m.group(1), raw, flags=re.S)


def _unescape_dquoted(raw: str) -> str:
    """Содержимое ``"…"``: ``\\`` значима только перед ``$ ` " \\`` и переводом строки."""
    return re.sub(r'\\([$`"\\\n])', lambda m: "" if m.group(1) == "\n" else m.group(1), raw)


def _unescape_heredoc(body: str) -> str:
    """Тело heredoc без кавычек у разделителя: ``\\$``, ``\\```, ``\\\\``, перенос через ``\\``."""
    return re.sub(r"\\([$`\\\n])", lambda m: "" if m.group(1) == "\n" else m.group(1), body)


def _static_value(node) -> str:
    t = node.type
    if t == "raw_string":
        return _text(node)[1:-1]
    if t == "string":
        # Текст между кавычками, а не дети узла: string_content рвётся на переводах
        # строк, и сами переводы строк в детей не попадают.
        text = _text(node)
        inner = text[1:-1] if len(text) >= 2 and text[0] == text[-1] == '"' else text
        return _unescape_dquoted(inner)
    if t == "concatenation":
        return "".join(_static_value(c) for c in node.children)
    if t == "word":
        return _unescape_word(_text(node))
    return _text(node)


def _has_glob(node) -> bool:
    if node.type == "word":
        return bool(_GLOB_CHARS.search(re.sub(r"\\.", "", _text(node))))
    if node.type == "concatenation":
        return any(_has_glob(c) for c in node.children)
    return False


def _has_brace(node) -> bool:
    """``{a,b}``/``{1..3}`` вне кавычек: bash раскроет, dash — нет."""
    if node.type not in ("word", "concatenation"):
        return False
    parts = list(node.children) if node.type == "concatenation" else [node]
    text = "".join(re.sub(r"\\.", "", _text(p)) if p.type == "word" else "\0" for p in parts)
    return bool(_BRACE.search(text))


class _Walker:
    def __init__(self, root: Path, cwd: str):
        self.root = root
        self.script = Script(cwd=cwd)
        self._cwd = cwd  # каталог оператора, который сейчас разбирается
        self.rest_start: int | None = None  # начало команды после ведущих cd

    def problem(self, text: str) -> None:
        if text not in self.script.problems:
            self.script.problems.append(text)

    # ----------------------------- слова ------------------------------ #
    def scan(self, node) -> tuple[bool, list[str]]:
        """Обойти поддерево слова: (есть ли раскрытия, имена переменных). Команды в
        подстановках разбираются как самостоятельные (они тоже выполнятся)."""
        dynamic = False
        names: list[str] = []
        stack = [node]
        while stack:
            n = stack.pop()
            t = n.type
            if t in _DYNAMIC_NODES:
                dynamic = True
            if t in ("variable_name", "special_variable_name") and dynamic:
                names.append(_text(n))
            if t in _SUBSTITUTIONS:
                inner = [c for c in n.children if c.type not in _SUBSTITUTION_DELIMITERS]
                self.statements(inner, cwd=self._cwd, substituted=True)
                continue
            if t == "raw_string":
                continue
            if t == "ERROR" or n.is_missing:
                self.problem("команда не разобрана")
            if n.child_count == 0 and t in _PLAIN_TEXT and _BACKTICK.search(_text(n)):
                # Грамматика не разбирает `…` в теле heredoc и внутри ${…}, а оболочка
                # выполнит: команда в нём не проверена.
                self.problem("подстановка `…`, которую не удалось разобрать")
            stack.extend(n.children)
        return dynamic, names

    def word(self, node) -> Word:
        raw = _text(node)
        dynamic, names = self.scan(node)
        if dynamic:
            return Word(raw, raw, static=False, vars=tuple(names))
        if _has_brace(node):
            return Word(raw, raw, static=False)
        return Word(_static_value(node), raw, glob=_has_glob(node))

    # --------------------------- операторы ---------------------------- #
    def statements(
        self,
        nodes,
        *,
        cwd: str,
        substituted: bool = False,
        leading_cd: bool = False,
        top: bool = False,
    ) -> str:
        """Последовательность операторов. Возвращает каталог после ведущих ``cd``."""
        items: list[tuple[object, str]] = []  # (узел, разделитель после него)
        self._flatten(nodes, items)
        leading = leading_cd
        for i, (node, sep) in enumerate(items):
            # Только «cd X && …» (или cd в конце): после «;» и перевода строки (у него
            # нет лексемы, разделитель пуст) команда выполнится и там, где cd не сработал.
            last = i == len(items) - 1
            if leading and _is_cd(node) and (sep == "&&" or (sep == "" and last)):
                target = self._cd_target(node, cwd)
                if target is not None:
                    cwd = target
                    self.script.consumed += 1
                    continue
            if leading and top:
                self.rest_start = node.start_byte
            leading = False
            if sep == "&":
                self.problem("фоновый процесс (&)")
            saved, self._cwd = self._cwd, cwd
            self.statement(node, cwd=cwd, substituted=substituted)
            self._cwd = saved
        return cwd

    def _flatten(self, nodes, items: list) -> None:
        nodes = [n for n in nodes if n.type != "comment"]
        for i, n in enumerate(nodes):
            if n.type == "ERROR" or n.is_missing:
                self.problem("команда не разобрана")
                continue
            if n.type in _TOKENS:
                continue
            if n.type == "list":
                self._flatten(n.children, items)
                if items:
                    items[-1] = (items[-1][0], _separator(nodes, i))
                continue
            items.append((n, _separator(nodes, i)))

    def _cd_target(self, node, cwd: str) -> str | None:
        """Каталог ``cd DIR`` внутри проекта (существующий); None — не ведущий cd."""
        parts = [c for c in node.children if c.type != "command_name"]
        if len(parts) != 1 or parts[0].type not in _WORD_NODES:
            return None
        word = self.word(parts[0])
        if not word.static or word.glob or not word.value or word.value.startswith(("~", "-")):
            return None
        target = (self.root / cwd / word.value).resolve()
        try:
            rel = target.relative_to(self.root)
        except ValueError:
            return None
        if not target.is_dir():
            return None
        return rel.as_posix() or "."

    def statement(
        self, node, *, cwd: str, substituted: bool, redirects: list[Redirect] | None = None
    ) -> list[Command]:
        """Оператор → его простые команды (последняя — источник для следующего звена конвейера)."""
        t = node.type
        if t == "command":
            return [self.command(node, cwd=cwd, substituted=substituted, extra=redirects)]
        if t == "redirected_statement":
            return self.redirected(node, cwd=cwd, substituted=substituted, outer=redirects)
        if t == "pipeline":
            return self.pipeline(node, cwd=cwd, substituted=substituted)
        if t == "subshell":
            before = len(self.script.commands)
            inner = [c for c in node.children if c.type not in ("(", ")")]
            self.statements(inner, cwd=cwd, substituted=substituted, leading_cd=True)
            return self._attach(self.script.commands[before:], redirects)
        if t in _CONTAINERS:
            if t != "negated_command":
                self.script.compound = True
            before = len(self.script.commands)
            for c in node.children:
                if c.type in _TOKENS or c.type == "variable_name":
                    continue
                if c.type in _WORD_NODES:  # for x in СЛОВА; case СЛОВО in
                    self.word(c)
                elif c.type == "test_command":
                    self.scan(c)
                else:
                    self.statements([c], cwd=cwd, substituted=substituted)
            return self._attach(self.script.commands[before:], redirects)
        if t == "test_command":
            self.scan(node)
            return []
        if t == "variable_assignment":
            self.script.assigns.append(self.assignment(node))
            return []
        if t == "variable_assignments":
            for a in node.named_children:
                self.script.assigns.append(self.assignment(a))
            return []
        if t == "declaration_command":
            self.declaration(node)
            return []
        if t == "function_definition":
            self.problem("определение функции")
            return []
        self.problem(f"конструкция оболочки {t}")
        return []

    @staticmethod
    def _attach(cmds: list[Command], redirects: list[Redirect] | None) -> list[Command]:
        if redirects:
            for c in cmds:
                c.redirects.extend(redirects)
        return cmds

    def assignment(self, node) -> tuple[str, Word]:
        name = node.child_by_field_name("name")
        value = node.child_by_field_name("value")
        word = self.word(value) if value is not None else Word("")
        if name is not None and name.type != "variable_name":  # a[1]=…, a+=…
            self.problem("сложное присваивание")
        return (_text(name) if name is not None else "?", word)

    def declaration(self, node) -> None:
        keyword = _text(node.children[0]) if node.children else ""
        if keyword not in ("export", "local", "readonly", "declare", "typeset"):
            self.problem(f"конструкция оболочки {keyword}")
        for c in node.named_children:
            if c.type == "variable_assignment":
                self.script.assigns.append(self.assignment(c))
            elif c.type in _WORD_NODES:
                w = self.word(c)
                if w.value.startswith("-") or keyword != "export":
                    self.problem(f"{keyword} {w.value}".strip())
                else:  # export NAME — значение берётся из присваивания раньше
                    self.script.assigns.append((w.value, Word("", static=False)))

    def redirected(
        self, node, *, cwd: str, substituted: bool, outer: list[Redirect] | None
    ) -> list[Command]:
        body = None
        redirects: list[Redirect] = []
        trailing: list = []  # то, что грамматика поместила внутрь heredoc: `| python`, `&& ls`
        for c in node.children:
            if c.type in ("file_redirect", "herestring_redirect"):
                redirect, extra = self.redirect(c)
                redirects.append(redirect)
                if extra:
                    self.problem("слова после перенаправления")
            elif c.type == "heredoc_redirect":
                redirect, extra = self.heredoc(c)
                redirects.append(redirect)
                trailing.extend(extra)
            elif body is None:
                body = c
            else:
                self.problem(f"конструкция оболочки {c.type}")
        if body is None:
            self.problem("перенаправление без команды")
            return []
        cmds = self.statement(
            body, cwd=cwd, substituted=substituted, redirects=redirects + list(outer or [])
        )
        if trailing:
            cmds = self._trailing(trailing, cmds, cwd=cwd, substituted=substituted)
        return cmds

    def _trailing(
        self, nodes, cmds: list[Command], *, cwd: str, substituted: bool
    ) -> list[Command]:
        """Продолжение команды, которое грамматика поместила внутрь heredoc."""
        prev = cmds[-1] if cmds else None
        result = cmds
        op = ""
        for n in nodes:
            if n.type in _SEPARATORS:
                op = n.type
                if op == "&":
                    self.problem("фоновый процесс (&)")
                continue
            if n.type in ("file_redirect", "herestring_redirect"):
                redirect, extra = self.redirect(n)
                if extra:
                    self.problem("слова после перенаправления")
                for c in result:
                    c.redirects.append(redirect)
                continue
            if n.type == "pipeline":
                for part in n.children:
                    if part.type in ("|", "|&"):
                        continue
                    if part.type == "list":
                        # `cat <<EOF | python3 && ls`: грамматика кладёт `python3 && ls`
                        # в одно звено; первое продолжает конвейер, остальное — дальше.
                        items: list = []
                        self._flatten(part.children, items)
                        for k, (item, _sep) in enumerate(items):
                            got = self.statement(item, cwd=cwd, substituted=substituted)
                            if k == 0 and got and prev is not None:
                                got[0].stdin = prev
                            prev = got[-1] if got else prev
                            result = got
                        continue
                    got = self.statement(part, cwd=cwd, substituted=substituted)
                    if got and prev is not None:
                        got[0].stdin = prev
                    prev = got[-1] if got else None
                    result = got
                continue
            if n.type == "list":
                items = []
                self._flatten(n.children, items)
                for item, _sep in items:
                    got = self.statement(item, cwd=cwd, substituted=substituted)
                    prev = got[-1] if got else prev
                    result = got
                continue
            got = self.statement(n, cwd=cwd, substituted=substituted)
            if op in ("|", "|&") and got and prev is not None:
                got[0].stdin = prev
            prev = got[-1] if got else prev
            result = got
        return result

    def redirect(self, node) -> tuple[Redirect, list]:
        """Перенаправление и лишние слова (``echo hi > f more`` — ``more`` попадает сюда)."""
        op = ""
        words = []
        for c in node.children:
            if c.type in (">", ">>", ">|", "&>", "&>>", "<", ">&", "<&", "<>", "<<<", "&>|"):
                op = c.type
            elif c.type in _WORD_NODES:
                words.append(c)
            elif c.type not in ("file_descriptor",):
                self.problem(f"перенаправление {c.type}")
        if not words:
            self.problem("перенаправление без цели")
            return Redirect(op), []
        first = self.word(words[0])
        if node.type == "herestring_redirect":
            redirect = Redirect(
                op, body=first.value, body_static=first.static, body_vars=first.vars
            )
            return redirect, words[1:]
        if op in (">&", "<&") and first.static and (first.value.isdigit() or first.value == "-"):
            return Redirect(op, first, fd_dup=True), words[1:]
        if op == "<>":
            self.problem("перенаправление <>")
        return Redirect(op, first), words[1:]

    def heredoc(self, node) -> tuple[Redirect, list]:
        quoted = False
        body = ""
        static = True
        names: list[str] = []
        extra: list = []
        for c in node.children:
            if c.type == "heredoc_start":
                quoted = bool(re.search(r"['\"\\]", _text(c)))
            elif c.type == "heredoc_body":
                body = _text(c)
                if not quoted:
                    dynamic, names = self.scan(c)
                    # `…` и $… в теле без кавычек у разделителя раскрываются оболочкой.
                    static = not dynamic and not re.search(r"(?<!\\)[`$]", body)
            elif c.type in ("<<", "<<-", "heredoc_end", "heredoc_content"):
                continue
            else:
                extra.append(c)
        if not quoted:
            body = _unescape_heredoc(body)
        return Redirect("<<", body=body, body_static=static, body_vars=tuple(names)), extra

    def command(
        self, node, *, cwd: str, substituted: bool, extra: list[Redirect] | None = None
    ) -> Command:
        cmd = Command(argv=[], cwd=cwd, substituted=substituted)
        for c in node.children:
            t = c.type
            if t == "variable_assignment":
                cmd.assigns.append(self.assignment(c))
            elif t == "command_name":
                cmd.argv.append(self.word(c.children[0] if c.children else c))
            elif t in _WORD_NODES:
                cmd.argv.append(self.word(c))
            elif t in ("file_redirect", "herestring_redirect"):
                redirect, rest = self.redirect(c)
                cmd.redirects.append(redirect)
                cmd.argv.extend(self.word(w) for w in rest)
            elif t == "heredoc_redirect":
                redirect, rest = self.heredoc(c)
                cmd.redirects.append(redirect)
                if rest:
                    self.problem("сложная команда с heredoc")
            elif t != "comment":
                self.problem(f"элемент команды {t}")
        if extra:
            cmd.redirects.extend(extra)
        self.script.commands.append(cmd)
        return cmd

    def pipeline(self, node, *, cwd: str, substituted: bool) -> list[Command]:
        prev: Command | None = None
        last: list[Command] = []
        for c in node.children:
            if c.type in ("|", "|&"):
                continue
            got = self.statement(c, cwd=cwd, substituted=substituted)
            if got and prev is not None:
                got[0].stdin = prev
            prev = got[-1] if got else None
            last = got
        return last


def _separator(nodes, i: int) -> str:
    for n in nodes[i + 1 :]:
        if n.type in _SEPARATORS:
            return n.type
        if n.type != "comment":
            return ""
    return ""


def _is_cd(node) -> bool:
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
    if final != cwd and walker.rest_start is not None:
        data = command.encode("utf-8", errors="surrogateescape")
        script.rest = data[walker.rest_start :].decode("utf-8", errors="surrogateescape").strip()
    elif final != cwd:
        script.rest = ""  # только cd
    else:
        script.rest = command.strip()
    return script
