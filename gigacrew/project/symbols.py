"""Извлечение фактов о файле для индекса проекта: определения, использования, импорты.

Python разбирается модулем :mod:`ast` (точно: вложенность, диапазон строк,
докстринги, использования имён и импорты); JS/TS, Go, Rust, Java, Kotlin, C#, Ruby,
PHP, C/C++, shell, Scala и Swift — tree-sitter (:mod:`gigacrew.project.treesitter`:
определения с диапазоном и вложенностью, вызовы). Построчные регулярные выражения —
запасной путь, если грамматика не загрузилась. Новый язык — расширения в
:data:`EXTENSIONS` и таблицы в ``treesitter.SPECS`` (правила в :data:`_RULES` —
по желанию).
"""

from __future__ import annotations

import ast
import builtins
import re
from dataclasses import dataclass, field
from pathlib import PurePath, PurePosixPath

MAX_SYMBOLS_PER_FILE = 2000
MAX_REFS_PER_FILE = 20_000
_MAX_SIGNATURE = 200
_MAX_SIGNATURE_LINES = 6
_MAX_DOC = 120

EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".swift": "swift",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hh": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".sh": "shell",
    ".bash": "shell",
    ".md": "markdown",
    ".markdown": "markdown",
    # без символов, но полезны для статистики индекса
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".html": "html",
    ".css": "css",
    ".scss": "css",
    ".sql": "sql",
    ".txt": "text",
    ".rst": "text",
}
FILENAMES: dict[str, str] = {
    "Makefile": "make",
    "Dockerfile": "docker",
    "CMakeLists.txt": "cmake",
}


@dataclass(frozen=True)
class Symbol:
    """Определение в файле. ``end_line`` = None — конец неизвестен (эвристика)."""

    name: str
    kind: str  # class, function, method, constant, interface, struct, section, ...
    line: int
    end_line: int | None = None
    parent: str = ""  # объемлющее определение (``Class`` для метода)
    depth: int = 0  # уровень вложенности (для оглавления)
    signature: str = ""
    doc: str = ""  # первая строка докстринга
    col: int = 0  # столбец имени в строке (в символах, с 0) — позиция для LSP

    @property
    def qualname(self) -> str:
        return f"{self.parent}.{self.name}" if self.parent else self.name


@dataclass(frozen=True)
class Ref:
    """Использование имени: вызов (``call``), атрибут (``attr``) или имя (``name``)."""

    name: str
    kind: str
    line: int
    col: int  # в символах, с 0
    scope: str = ""  # объемлющее определение (``Class.method``; "" — уровень модуля)


@dataclass(frozen=True)
class Import:
    """``import module`` (``name`` пустое) или ``from module import name``.

    Относительный импорт разрешается по пути файла в абсолютный; не разрешённый
    (выше корня проекта) остаётся с ведущими точками.
    """

    module: str
    name: str
    alias: str  # имя, под которым импортированное доступно в файле
    line: int


@dataclass(frozen=True)
class FileFacts:
    """Всё, что индекс знает о файле."""

    symbols: list[Symbol]
    refs: list[Ref] = field(default_factory=list)
    imports: list[Import] = field(default_factory=list)


def split_lines(text: str) -> list[str]:
    """Строки как их нумеруют ``ast``, tree-sitter и LSP: разрыв — только ``\\n``,
    ``\\r\\n`` или ``\\r``. ``str.splitlines`` режет ещё и по ``\\f``, ``\\v``,
    ``\\u2028``… — номера строк после таких символов разъезжаются.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def language_of(path: str | PurePath) -> str | None:
    """Язык файла по имени/расширению (None — неизвестен)."""
    p = PurePath(path)
    return FILENAMES.get(p.name) or EXTENSIONS.get(p.suffix.lower())


def module_parts(path: str) -> list[str] | None:
    """Имя Python-модуля по пути от корня: ``a/b/c.py`` → ``[a, b, c]``.

    ``a/b/__init__.py`` — пакет ``[a, b]``; не Python-файл — None. Корень пакетов
    (``src/``) не угадывается: модули сопоставляются файлам по суффиксу пути.
    """
    p = PurePosixPath(path)
    if p.suffix not in (".py", ".pyi"):
        return None
    parts = list(p.parent.parts)
    if p.stem != "__init__":
        parts.append(p.stem)
    return parts


def extract(text: str, language: str | None, path: str = "") -> FileFacts:
    """Факты о файле. ``path`` (от корня проекта) нужен для относительных импортов.

    Python — ``ast`` (при неудаче — tree-sitter, затем регэкспы); Markdown —
    заголовки; остальные языки — tree-sitter, без грамматики — регэкспы.
    """
    # импорт здесь: treesitter сам опирается на типы этого модуля
    from gigacrew.project import treesitter

    if language == "python":
        try:
            tree = ast.parse(text)
            lines = split_lines(text)
            symbols = _python_symbols(tree, lines)
            refs, imports = _python_refs(tree, lines, path)
        except (SyntaxError, ValueError, RecursionError):
            facts = treesitter.extract(text, language, path)
            if facts is not None:
                return facts
            return FileFacts(_regex_symbols(text, _RULES["python"])[:MAX_SYMBOLS_PER_FILE])
        return FileFacts(symbols[:MAX_SYMBOLS_PER_FILE], refs, imports)
    if language == "markdown":
        return FileFacts(_markdown_symbols(text)[:MAX_SYMBOLS_PER_FILE])
    if language is not None and treesitter.supports(language):
        facts = treesitter.extract(text, language, path)
        if facts is not None:
            return facts
    if language in _RULES:
        return FileFacts(_regex_symbols(text, _RULES[language])[:MAX_SYMBOLS_PER_FILE])
    return FileFacts([])


def extract_symbols(text: str, language: str | None) -> list[Symbol]:
    """Символы файла в порядке появления (не больше :data:`MAX_SYMBOLS_PER_FILE`)."""
    return extract(text, language).symbols


def clip_signature(signature: str) -> str:
    """Сигнатура в одну строку, не длиннее :data:`_MAX_SIGNATURE`."""
    signature = " ".join(signature.split())
    if len(signature) > _MAX_SIGNATURE:
        signature = signature[: _MAX_SIGNATURE - 1] + "…"
    return signature


# --------------------------------- Python --------------------------------- #
def _char_col(line: str, byte_col: int) -> int:
    """Столбец в символах по смещению ``ast`` (в байтах UTF-8)."""
    if line.isascii():
        return byte_col
    return len(line.encode("utf-8")[:byte_col].decode("utf-8", errors="ignore"))


def _line(lines: list[str], lineno: int) -> str:
    return lines[lineno - 1] if 0 < lineno <= len(lines) else ""


def _doc(node: ast.AST) -> str:
    try:
        doc = ast.get_docstring(node)  # type: ignore[arg-type]
    except TypeError:
        return ""
    if not doc:
        return ""
    first = next((s.strip() for s in doc.splitlines() if s.strip()), "")
    return first if len(first) <= _MAX_DOC else first[: _MAX_DOC - 1] + "…"


def _def_name_col(lines: list[str], node: ast.AST, name: str) -> int:
    line = _line(lines, node.lineno)  # type: ignore[attr-defined]
    start = _char_col(line, node.col_offset)  # type: ignore[attr-defined]
    m = re.compile(rf"\b(?:def|class)\s+({re.escape(name)})\b").search(line, start)
    return m.start(1) if m else start


def _python_symbols(tree: ast.Module, lines: list[str]) -> list[Symbol]:
    out: list[Symbol] = []

    def signature(node: ast.AST) -> str:
        start = node.lineno
        body = getattr(node, "body", None)
        last = body[0].lineno - 1 if body else start
        last = min(max(last, start), start + _MAX_SIGNATURE_LINES - 1)
        parts = (lines[i - 1].strip() for i in range(start, last + 1) if i <= len(lines))
        # строки-комментарии между заголовком и телом — не часть сигнатуры
        return clip_signature(" ".join(part for part in parts if not part.startswith("#")))

    def visit(body: list[ast.stmt], parent: str, depth: int, in_class: bool) -> None:
        for node in body:
            if len(out) >= MAX_SYMBOLS_PER_FILE:
                return
            if isinstance(node, ast.ClassDef):
                kind = "class"
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "method" if in_class else "function"
            elif isinstance(node, (ast.If, ast.Try)):
                # определения под `if TYPE_CHECKING:` / `try: import ...` — того же уровня
                # в порядке исходника: try → except → else → finally
                visit(node.body, parent, depth, in_class)
                for handler in getattr(node, "handlers", []):
                    visit(handler.body, parent, depth, in_class)
                for block in (node.orelse, getattr(node, "finalbody", [])):
                    visit(block, parent, depth, in_class)
                continue
            elif depth == 0 and isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name) and t.id.isupper() and len(t.id) > 1:
                        out.append(
                            Symbol(
                                name=t.id,
                                kind="constant",
                                line=node.lineno,
                                end_line=node.end_lineno,
                                signature=clip_signature(lines[node.lineno - 1]),
                                col=_char_col(_line(lines, t.lineno), t.col_offset),
                            )
                        )
                continue
            else:
                continue
            out.append(
                Symbol(
                    name=node.name,
                    kind=kind,
                    line=node.lineno,
                    end_line=node.end_lineno,
                    parent=parent,
                    depth=depth,
                    signature=signature(node),
                    doc=_doc(node),
                    col=_def_name_col(lines, node, node.name),
                )
            )
            if kind == "class":  # вложенные функции — шум; методы и вложенные классы — нет
                qual = f"{parent}.{node.name}" if parent else node.name
                visit(node.body, qual, depth + 1, True)

    visit(tree.body, "", 0, False)
    return out


# Имена, использования которых не интересны: встроенные и self/cls.
_SKIP_NAMES = frozenset(dir(builtins)) | {"self", "cls"}


_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _local_names(
    fn: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
) -> tuple[frozenset[str], frozenset[str]]:
    """(локальные, объявленные ``global``) имена самой функции.

    Локальные — параметры, присваивания, ``except … as``, ``match`` (в том числе
    ``case {**rest}``), имена вложенных ``def``/``class``: обращения к ним — не
    использования определений проекта. Не локальные: импорты (вызов
    импортированного имени — использование), связывания внутри вложенных функций,
    классов и включений (у них своя область; ``:=`` во включении связывает имя в
    функции), объявленные ``global`` (они же снимают одноимённые локальные внешних
    функций) и ``nonlocal`` (они остаются локальными — внешней функции).
    """
    a = fn.args
    names = {x.arg for x in (*a.posonlyargs, *a.args, *a.kwonlyargs)}
    names.update(x.arg for x in (a.vararg, a.kwarg) if x is not None)
    declared: set[str] = set()
    nonlocal_: set[str] = set()
    stack: list[ast.AST] = list(fn.body) if isinstance(fn.body, list) else [fn.body]
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)  # тело — своя область
            continue
        if isinstance(node, ast.Lambda):
            continue
        if isinstance(node, ast.Name):
            if not isinstance(node.ctx, ast.Load):
                names.add(node.id)
        elif isinstance(node, ast.Global):
            declared.update(node.names)
        elif isinstance(node, ast.Nonlocal):
            nonlocal_.update(node.names)
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
            if node.name:
                names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.add(node.rest)  # `case {**rest}` — строка, а не узел Name
        if isinstance(node, _COMPREHENSIONS):  # цели включения — его область
            for gen in node.generators:
                stack.append(gen.iter)
                stack.extend(gen.ifs)
            stack.extend(
                child
                for child in (getattr(node, f, None) for f in ("elt", "key", "value"))
                if child is not None
            )
            continue
        stack.extend(ast.iter_child_nodes(node))
    return frozenset(names - declared - nonlocal_), frozenset(declared)


def _absolute_module(module: str | None, level: int, path: str) -> str:
    if not level:
        return module or ""
    package = list(PurePosixPath(path).parent.parts) if path else None
    if package is None or level - 1 > len(package):
        return "." * level + (module or "")
    base = package[: len(package) - (level - 1)]
    return ".".join([*base, *([module] if module else [])])


class _RefCollector(ast.NodeVisitor):
    """Использования имён и импорты. Область — объемлющее индексируемое определение
    (класс или функция уровня модуля/класса; вложенные функции — в своей внешней).
    """

    def __init__(self, lines: list[str], path: str):
        self.lines = lines
        self.path = path
        self.refs: list[Ref] = []
        self.imports: list[Import] = []
        self._seen: set[tuple[str, str, int, int, str]] = set()
        self._scope: list[str] = []
        self._locals: list[frozenset[str]] = []
        self._in_function = 0

    def _add(self, name: str, kind: str, line: int, byte_col: int) -> None:
        if len(name) < 2 or name in _SKIP_NAMES or (name.startswith("__") and name.endswith("__")):
            return
        scope = ".".join(self._scope)
        col = _char_col(_line(self.lines, line), byte_col)
        key = (name, kind, line, col, scope)  # `a.run(); b.run()` — два использования
        if key in self._seen or len(self.refs) >= MAX_REFS_PER_FILE:
            return
        self._seen.add(key)
        self.refs.append(Ref(name=name, kind=kind, line=line, col=col, scope=scope))

    def _local(self, name: str) -> bool:
        return bool(self._locals) and name in self._locals[-1]

    def _with_locals(self, node: ast.AST, names: frozenset[str]) -> None:
        outer = self._locals[-1] if self._locals else frozenset()
        self._locals.append(outer | names)
        self.generic_visit(node)
        self._locals.pop()

    # ------------------------------ области ------------------------------ #
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        indexed = not self._in_function
        if indexed:
            self._scope.append(node.name)
        self.generic_visit(node)
        if indexed:
            self._scope.pop()

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        indexed = not self._in_function
        if indexed:
            self._scope.append(node.name)
        a = node.args
        params = (*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg)
        # декораторы, значения по умолчанию и аннотации вычисляются в объемлющей
        # области: `def convert(Model: Model) -> Model` — использования Model
        for expr in (
            *node.decorator_list,
            *a.defaults,
            *(d for d in a.kw_defaults if d),
            *(p.annotation for p in params if p is not None and p.annotation is not None),
            *([node.returns] if node.returns is not None else []),
            *node.type_params,
        ):
            self.visit(expr)
        self._in_function += 1
        own, declared = _local_names(node)
        outer = self._locals[-1] if self._locals else frozenset()
        self._locals.append((outer - declared) | own)
        for stmt in node.body:
            self.visit(stmt)
        self._locals.pop()
        self._in_function -= 1
        if indexed:
            self._scope.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._with_locals(node, _local_names(node)[0])

    def _visit_comprehension(self, node: ast.AST) -> None:
        generators: list[ast.comprehension] = node.generators  # type: ignore[attr-defined]
        # первый итерируемый вычисляется в объемлющей области: [x for x in x]
        self.visit(generators[0].iter)
        names = frozenset(
            n.id for gen in generators for n in ast.walk(gen.target) if isinstance(n, ast.Name)
        )
        outer = self._locals[-1] if self._locals else frozenset()
        self._locals.append(outer | names)
        for i, gen in enumerate(generators):
            if i:
                self.visit(gen.iter)
            for cond in gen.ifs:
                self.visit(cond)
        for part in ("elt", "key", "value"):
            child = getattr(node, part, None)
            if child is not None:
                self.visit(child)
        self._locals.pop()

    visit_ListComp = _visit_comprehension
    visit_SetComp = _visit_comprehension
    visit_DictComp = _visit_comprehension
    visit_GeneratorExp = _visit_comprehension

    # ---------------------------- использования --------------------------- #
    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and not self._local(node.id):
            self._add(node.id, "name", node.lineno, node.col_offset)

    def _attr(self, node: ast.Attribute, kind: str) -> None:
        end = node.end_col_offset or 0
        line = node.end_lineno or node.lineno
        self._add(node.attr, kind, line, end - len(node.attr.encode("utf-8")))

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._attr(node, "attr")
        self.visit(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name):
            if not self._local(func.id):
                self._add(func.id, "call", func.lineno, func.col_offset)
        elif isinstance(func, ast.Attribute):
            self._attr(func, "call")
            self.visit(func.value)
        else:
            self.visit(func)
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            self.visit(kw)

    # ------------------------------ импорты ------------------------------ #
    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            alias = a.asname or a.name.split(".")[0]
            self.imports.append(Import(module=a.name, name="", alias=alias, line=node.lineno))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = _absolute_module(node.module, node.level, self.path)
        for a in node.names:
            self.imports.append(
                Import(module=module, name=a.name, alias=a.asname or a.name, line=node.lineno)
            )


def _python_refs(tree: ast.Module, lines: list[str], path: str) -> tuple[list[Ref], list[Import]]:
    collector = _RefCollector(lines, path)
    collector.visit(tree)
    return collector.refs, collector.imports


# -------------------------------- Markdown -------------------------------- #
# CommonMark: до 3 пробелов отступа; закрывающие '#' — только после пробела ("# C#").
_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$")
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def _markdown_symbols(text: str) -> list[Symbol]:
    out: list[Symbol] = []
    stack: list[tuple[int, str]] = []  # (уровень, заголовок)
    fence = ""  # открывающая ограда блока кода ("```", "~~~~"…), пусто — вне блока
    for i, line in enumerate(split_lines(text), start=1):
        m = _FENCE_RE.match(line)
        if fence:
            # закрывает только ограда того же символа, не короче, без текста после
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence):
                if not line[m.end() :].strip():
                    fence = ""
            continue
        if m:
            fence = m.group(1)
            continue
        m = _HEADING_RE.match(line)
        if not m:
            continue
        level, title = len(m.group(1)), m.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        out.append(
            Symbol(
                name=title[:_MAX_SIGNATURE],
                kind="section",
                line=i,
                parent=stack[-1][1] if stack else "",
                depth=len(stack),
                signature=clip_signature(line),
            )
        )
        stack.append((level, title))
        if len(out) >= MAX_SYMBOLS_PER_FILE:
            break
    return out


# ------------------------------ Регэкспы ------------------------------ #
# Группы: name — имя; kind — вид (если не задан правилом); parent — владелец.
_Rule = tuple[str, "re.Pattern[str]"]
_NOT_NAMES = frozenset(
    {"if", "for", "while", "switch", "catch", "return", "else", "new", "sizeof", "do", "with"}
)
_ID = r"[A-Za-z_$][\w$]*"


def _rules(*pairs: tuple[str, str]) -> list[_Rule]:
    return [(kind, re.compile(pattern)) for kind, pattern in pairs]


_JS_RULES = _rules(
    (
        "function",
        rf"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*(?P<name>{_ID})",
    ),
    ("class", rf"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(?P<name>{_ID})"),
    ("interface", rf"^\s*(?:export\s+)?(?:declare\s+)?interface\s+(?P<name>{_ID})"),
    ("type", rf"^\s*(?:export\s+)?(?:declare\s+)?type\s+(?P<name>{_ID})\s*(?:<[^=]*>)?\s*="),
    ("enum", rf"^\s*(?:export\s+)?(?:declare\s+)?(?:const\s+)?enum\s+(?P<name>{_ID})"),
    (
        "function",
        rf"^\s*(?:export\s+)?(?:const|let|var)\s+(?P<name>{_ID})\s*(?::[^=]+)?=\s*"
        rf"(?:async\s+)?(?:function\b|\([^)]*\)\s*(?::[^=]+)?=>|{_ID}\s*=>)",
    ),
    (
        "method",
        r"^\s+(?:(?:public|private|protected|static|readonly|async|override|get|set)\s+)*"
        rf"(?P<name>{_ID})\s*\([^)]*\)\s*(?::\s*[^{{]+)?\{{\s*$",
    ),
)
_TYPES_JVM = (
    r"^\s*(?:@\w+(?:\([^)]*\))?\s+)*(?:(?:public|private|protected|internal|static|final|"
    r"abstract|sealed|open|data|partial|inline|value|annotation|case|readonly)\s+)*"
    r"(?P<kind>class|interface|enum|record|struct|object|trait|protocol)\s+(?P<name>[A-Za-z_]\w*)"
)
_JAVA_METHOD = (
    r"^\s+(?:(?:public|private|protected|internal|static|final|abstract|synchronized|native|"
    r"default|override|virtual|async|sealed)\s+)+[\w<>\[\],.?\s]*?\s(?P<name>[A-Za-z_]\w*)\s*\("
)

_RULES: dict[str, list[_Rule]] = {
    "python": _rules(
        ("class", r"^(?P<indent>\s*)class\s+(?P<name>[A-Za-z_]\w*)"),
        ("function", r"^(?P<indent>\s*)(?:async\s+)?def\s+(?P<name>[A-Za-z_]\w*)"),
    ),
    "javascript": _JS_RULES,
    "typescript": _JS_RULES,
    "go": _rules(
        (
            "method",
            r"^func\s+\(\s*\w*\s*\*?\s*(?P<parent>[A-Za-z_]\w*)[^)]*\)\s*(?P<name>[A-Za-z_]\w*)",
        ),
        ("function", r"^func\s+(?P<name>[A-Za-z_]\w*)"),
        ("type", r"^type\s+(?P<name>[A-Za-z_]\w*)\s+(?P<kind>struct|interface)\b"),
        ("type", r"^type\s+(?P<name>[A-Za-z_]\w*)"),
    ),
    "rust": _rules(
        (
            "function",
            r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?"
            r'(?:extern\s+"[^"]*"\s+)?fn\s+(?P<name>[A-Za-z_]\w*)',
        ),
        (
            "type",
            r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?P<kind>struct|enum|trait|union|mod|type)\s+"
            r"(?P<name>[A-Za-z_]\w*)",
        ),
        (
            "impl",
            r"^\s*impl(?:<[^>]*>)?\s+(?:[\w:<>, ]+\s+for\s+)?(?P<name>[A-Za-z_][\w:]*)",
        ),
        ("macro", r"^\s*macro_rules!\s*(?P<name>[A-Za-z_]\w*)"),
    ),
    "java": _rules(("class", _TYPES_JVM), ("method", _JAVA_METHOD)),
    "csharp": _rules(("class", _TYPES_JVM), ("method", _JAVA_METHOD)),
    "kotlin": _rules(
        ("class", _TYPES_JVM),
        (
            "function",
            r"^\s*(?:\w+\s+)*fun\s+(?:<[^>]*>\s*)?(?:[\w.<>?, ]+\.)?(?P<name>[A-Za-z_]\w*)",
        ),
    ),
    "scala": _rules(("class", _TYPES_JVM), ("function", r"^\s*(?:\w+\s+)*def\s+(?P<name>\w+)")),
    "swift": _rules(("class", _TYPES_JVM), ("function", r"^\s*(?:\w+\s+)*func\s+(?P<name>\w+)")),
    "ruby": _rules(
        ("class", r"^\s*(?P<kind>class|module)\s+(?P<name>[A-Z][\w:]*)"),
        ("method", r"^\s*def\s+(?:self\.)?(?P<name>[\w?!=]+)"),
    ),
    "php": _rules(
        (
            "class",
            r"^\s*(?:(?:abstract|final|readonly)\s+)*(?P<kind>class|interface|trait|enum)\s+"
            r"(?P<name>\w+)",
        ),
        (
            "function",
            r"^\s*(?:(?:public|private|protected|static|abstract|final)\s+)*function\s+&?\s*"
            r"(?P<name>\w+)",
        ),
    ),
    "c": _rules(
        ("type", r"^\s*(?:typedef\s+)?(?P<kind>struct|union|enum)\s+(?P<name>[A-Za-z_]\w*)\s*\{"),
        (
            "function",
            r"^(?!\s)(?!(?:return|else|if|while|for|switch|do)\b)[\w\s\*]*?\b"
            r"(?P<name>[A-Za-z_]\w*)\s*\([^;]*\)\s*\{?\s*$",
        ),
    ),
    "cpp": _rules(
        (
            "type",
            r"^\s*(?:template\s*<[^>]*>\s*)?(?:typedef\s+)?(?P<kind>class|struct|union|enum)"
            r"(?:\s+class)?\s+(?P<name>[A-Za-z_]\w*)\s*(?::[^{;]*)?\{?\s*$",
        ),
        ("namespace", r"^\s*namespace\s+(?P<name>[A-Za-z_][\w:]*)"),
        (
            "function",
            r"^(?!\s)(?!(?:return|else|if|while|for|switch|do)\b)[\w\s\*&:<>,~]*?\b"
            r"(?P<name>[A-Za-z_~][\w:~]*)\s*\([^;]*\)\s*(?:const\s*)?(?:noexcept\s*)?"
            r"(?:override\s*)?\{?\s*$",
        ),
    ),
    "shell": _rules(
        ("function", r"^\s*(?:function\s+)?(?P<name>[A-Za-z_][\w-]*)\s*\(\)\s*\{?"),
        ("function", r"^\s*function\s+(?P<name>[A-Za-z_][\w-]*)"),
    ),
}


def _regex_symbols(text: str, rules: list[_Rule]) -> list[Symbol]:
    out: list[Symbol] = []
    for i, line in enumerate(split_lines(text), start=1):
        if not line.strip() or len(line) > 1000:
            continue
        for kind, regex in rules:
            m = regex.match(line)
            if not m:
                continue
            name = m.group("name")
            if name in _NOT_NAMES:
                break
            groups = m.groupdict()
            indent = groups.get("indent")
            out.append(
                Symbol(
                    name=name,
                    kind=(groups.get("kind") or kind).lower(),
                    line=i,
                    parent=groups.get("parent") or "",
                    depth=1 if indent else 0,
                    signature=clip_signature(line),
                    col=m.start("name"),
                )
            )
            break
        if len(out) >= MAX_SYMBOLS_PER_FILE:
            break
    return out
