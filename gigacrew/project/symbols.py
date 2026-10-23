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
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import PurePath, PurePosixPath
from types import MappingProxyType
from typing import override

MAX_SYMBOLS_PER_FILE = 2000
MAX_REFS_PER_FILE = 20_000
_MAX_SIGNATURE = 200
_MAX_SIGNATURE_LINES = 6
_MAX_DOC = 120
_MAX_REGEX_LINE = 1000

# Виды символов и языки, которые повторяются в таблицах ниже.
_CLASS = "class"
_FUNCTION = "function"
_METHOD = "method"
_TYPE = "type"
_CPP = "cpp"

EXTENSIONS: Mapping[str, str] = MappingProxyType({
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
    ".cc": _CPP,
    ".cpp": _CPP,
    ".cxx": _CPP,
    ".hh": _CPP,
    ".hpp": _CPP,
    ".hxx": _CPP,
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
})
FILENAMES: Mapping[str, str] = MappingProxyType({
    "Makefile": "make",
    "Dockerfile": "docker",
    "CMakeLists.txt": "cmake",
})


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
        """Полное имя: ``Parent.name`` или просто ``name`` на уровне модуля."""
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
    r"""Строки, как их нумеруют ``ast``, tree-sitter и LSP.

    Разрыв — только ``\\n``, ``\\r\\n`` или ``\\r``. ``str.splitlines`` режет ещё и
    по ``\\f``, ``\\v``, ``\\u2028``… — номера строк после таких символов разъезжаются.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def language_of(path: str | PurePath) -> str | None:
    """Язык файла по имени/расширению (None — неизвестен)."""
    file_path = PurePath(path)
    return FILENAMES.get(file_path.name) or EXTENSIONS.get(file_path.suffix.lower())


def module_parts(path: str) -> list[str] | None:
    """Имя Python-модуля по пути от корня: ``a/b/c.py`` → ``[a, b, c]``.

    ``a/b/__init__.py`` — пакет ``[a, b]``; не Python-файл — None. Корень пакетов
    (``src/``) не угадывается: модули сопоставляются файлам по суффиксу пути.
    """
    module_path = PurePosixPath(path)
    if module_path.suffix not in {".py", ".pyi"}:
        return None
    parts = list(module_path.parent.parts)
    if module_path.stem != "__init__":
        parts.append(module_path.stem)
    return parts


def extract(text: str, language: str | None, path: str = "") -> FileFacts:
    """Факты о файле. ``path`` (от корня проекта) нужен для относительных импортов.

    Python — ``ast`` (при неудаче — tree-sitter, затем регэкспы); Markdown —
    заголовки; остальные языки — tree-sitter, без грамматики — регэкспы.
    """
    # импорт здесь: treesitter сам опирается на типы этого модуля
    from gigacrew.project import treesitter  # noqa: PLC0415

    if language == "python":
        lines = split_lines(text)
        try:
            tree = ast.parse(text)
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
    rules = None if language is None else _RULES.get(language)
    if rules is not None:
        return FileFacts(_regex_symbols(text, rules)[:MAX_SYMBOLS_PER_FILE])
    return FileFacts([])


def extract_symbols(text: str, language: str | None) -> list[Symbol]:
    """Символы файла в порядке появления (не больше :data:`MAX_SYMBOLS_PER_FILE`)."""
    return extract(text, language).symbols


def clip_signature(signature: str) -> str:
    """Сигнатура в одну строку, не длиннее :data:`_MAX_SIGNATURE`."""
    signature = " ".join(signature.split())
    if len(signature) > _MAX_SIGNATURE:
        head = signature[: _MAX_SIGNATURE - 1]
        signature = f"{head}…"
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
    first = next((doc_line.strip() for doc_line in doc.splitlines() if doc_line.strip()), "")
    if len(first) <= _MAX_DOC:
        return first
    head = first[: _MAX_DOC - 1]
    return f"{head}…"


def _def_name_col(lines: list[str], node: ast.AST, name: str) -> int:
    line = _line(lines, node.lineno)  # type: ignore[attr-defined]
    start = _char_col(line, node.col_offset)  # type: ignore[attr-defined]
    match = re.compile(rf"\b(?:def|class)\s+({re.escape(name)})\b").search(line, start)
    return match.start(1) if match else start


_PythonDefinition = ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef


class _PythonSymbolCollector:
    """Определения Python-модуля: классы, функции, методы и константы уровня модуля."""

    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.symbols: list[Symbol] = []

    def visit(self, body: list[ast.stmt], parent: str, depth: int, *, in_class: bool) -> None:
        """Определения из списка инструкций ``body`` (с вложенными в классы)."""
        for node in body:
            if len(self.symbols) >= MAX_SYMBOLS_PER_FILE:
                return
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                self._visit_definition(node, parent, depth, in_class=in_class)
            elif isinstance(node, (ast.If, ast.Try)):
                self._visit_branches(node, parent, depth, in_class=in_class)
            elif depth == 0 and isinstance(node, (ast.Assign, ast.AnnAssign)):
                self.symbols.extend(self._constants(node))

    def _visit_definition(
        self,
        node: _PythonDefinition,
        parent: str,
        depth: int,
        *,
        in_class: bool,
    ) -> None:
        kind = _METHOD if in_class else _FUNCTION
        if isinstance(node, ast.ClassDef):
            kind = _CLASS
        self.symbols.append(
            Symbol(
                name=node.name,
                kind=kind,
                line=node.lineno,
                end_line=node.end_lineno,
                parent=parent,
                depth=depth,
                signature=self._signature(node),
                doc=_doc(node),
                col=_def_name_col(self.lines, node, node.name),
            ),
        )
        # вложенные функции — шум; методы и вложенные классы — нет
        if isinstance(node, ast.ClassDef):
            qual = f"{parent}.{node.name}" if parent else node.name
            self.visit(node.body, qual, depth + 1, in_class=True)

    def _visit_branches(
        self,
        node: ast.If | ast.Try,
        parent: str,
        depth: int,
        *,
        in_class: bool,
    ) -> None:
        # определения под `if TYPE_CHECKING:` / `try: import ...` — того же уровня
        # в порядке исходника: try → except → else → finally
        self.visit(node.body, parent, depth, in_class=in_class)
        for handler in getattr(node, "handlers", []):
            self.visit(handler.body, parent, depth, in_class=in_class)
        for block in (node.orelse, getattr(node, "finalbody", [])):
            self.visit(block, parent, depth, in_class=in_class)

    def _constants(self, node: ast.Assign | ast.AnnAssign) -> Iterator[Symbol]:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id.isupper() and len(target.id) > 1:
                yield Symbol(
                    name=target.id,
                    kind="constant",
                    line=node.lineno,
                    end_line=node.end_lineno,
                    signature=clip_signature(self.lines[node.lineno - 1]),
                    col=_char_col(_line(self.lines, target.lineno), target.col_offset),
                )

    def _signature(self, node: _PythonDefinition) -> str:
        start = node.lineno
        body = getattr(node, "body", None)
        last = body[0].lineno - 1 if body else start
        last = min(max(last, start), start + _MAX_SIGNATURE_LINES - 1)
        line_numbers = range(start, min(last, len(self.lines)) + 1)
        parts = (self.lines[number - 1].strip() for number in line_numbers)
        # строки-комментарии между заголовком и телом — не часть сигнатуры
        return clip_signature(" ".join(part for part in parts if not part.startswith("#")))


def _python_symbols(tree: ast.Module, lines: list[str]) -> list[Symbol]:
    collector = _PythonSymbolCollector(lines)
    collector.visit(tree.body, "", 0, in_class=False)
    return collector.symbols


# Имена, использования которых не интересны: встроенные и self/cls.
# dir() здесь — не отладка, а перечень имён модуля builtins.
_SKIP_NAMES = frozenset(dir(builtins)) | {"self", "cls"}  # noqa: WPS421


_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_Comprehension = ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp
# Части включения, вычисляемые в его области: элемент, ключ и значение.
_COMPREHENSION_PARTS = ("elt", "key", "value")


def _parameter_names(arguments: ast.arguments) -> set[str]:
    """Имена параметров функции, включая ``*args`` и ``**kwargs``."""
    positional = (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
    names = {argument.arg for argument in positional}
    names.update(
        argument.arg for argument in (arguments.vararg, arguments.kwarg) if argument is not None
    )
    return names


def _bound_name(node: ast.AST) -> str | None:
    """Имя, которое узел связывает в своей области (None — не связывает)."""
    if isinstance(node, ast.Name):
        return None if isinstance(node.ctx, ast.Load) else node.id
    if isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
        return node.name or None
    if isinstance(node, ast.MatchMapping):
        return node.rest or None  # `case {**rest}` — строка, а не узел Name
    return None


def _same_scope_children(node: ast.AST) -> list[ast.AST]:
    """Дочерние узлы той же области: у включения — без его целей (у них своя область)."""
    if not isinstance(node, _COMPREHENSIONS):
        return list(ast.iter_child_nodes(node))
    children: list[ast.AST] = []
    for gen in node.generators:
        children.append(gen.iter)
        children.extend(gen.ifs)
    parts = (getattr(node, part, None) for part in _COMPREHENSION_PARTS)
    children.extend(child for child in parts if child is not None)
    return children


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
    names = _parameter_names(fn.args)
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
        if isinstance(node, ast.Global):
            declared.update(node.names)
        elif isinstance(node, ast.Nonlocal):
            nonlocal_.update(node.names)
        else:
            bound = _bound_name(node)
            if bound is not None:
                names.add(bound)
        stack.extend(_same_scope_children(node))
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
    """Использования имён и импорты.

    Область — объемлющее индексируемое определение (класс или функция уровня
    модуля/класса; вложенные функции — в своей внешней).
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

    # ------------------------------ области ------------------------------ #
    @override
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        indexed = not self._in_function
        if indexed:
            self._scope.append(node.name)
        self.generic_visit(node)
        if indexed:
            self._scope.pop()

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    @override
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    @override
    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._with_locals(node, _local_names(node)[0])

    @override
    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node)

    @override
    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node)

    @override
    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node)

    @override
    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node)

    # ---------------------------- использования --------------------------- #
    @override
    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and not self._local(node.id):
            self._add(node.id, "name", node.lineno, node.col_offset)

    @override
    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._attr(node, "attr")
        self.visit(node.value)

    @override
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
    @override
    def visit_Import(self, node: ast.Import) -> None:
        for imported in node.names:
            alias = imported.asname or imported.name.split(".")[0]
            self.imports.append(
                Import(module=imported.name, name="", alias=alias, line=node.lineno),
            )

    @override
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = _absolute_module(node.module, node.level, self.path)
        for imported in node.names:
            alias = imported.asname or imported.name
            self.imports.append(
                Import(module=module, name=imported.name, alias=alias, line=node.lineno),
            )

    # ---------------------------- вспомогательные --------------------------- #
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

    def _attr(self, node: ast.Attribute, kind: str) -> None:
        end = node.end_col_offset or 0
        line = node.end_lineno or node.lineno
        self._add(node.attr, kind, line, end - len(node.attr.encode("utf-8")))

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        indexed = not self._in_function
        if indexed:
            self._scope.append(node.name)
        # декораторы, значения по умолчанию и аннотации вычисляются в объемлющей
        # области: `def convert(Model: Model) -> Model` — использования Model
        for expr in _outer_scope_exprs(node):
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

    def _visit_comprehension(self, node: _Comprehension) -> None:
        generators = node.generators
        # первый итерируемый вычисляется в объемлющей области: [x for x in x]
        self.visit(generators[0].iter)
        names = frozenset(
            name_node.id
            for gen in generators
            for name_node in ast.walk(gen.target)
            if isinstance(name_node, ast.Name)
        )
        outer = self._locals[-1] if self._locals else frozenset()
        self._locals.append(outer | names)
        for index, gen in enumerate(generators):
            if index:
                self.visit(gen.iter)
            for cond in gen.ifs:
                self.visit(cond)
        for part in _COMPREHENSION_PARTS:
            child = getattr(node, part, None)
            if child is not None:
                self.visit(child)
        self._locals.pop()


def _outer_scope_exprs(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.AST]:
    """Части определения функции, вычисляемые в объемлющей области, в порядке обхода."""
    arguments = node.args
    params = (
        *arguments.posonlyargs,
        *arguments.args,
        *arguments.kwonlyargs,
        arguments.vararg,
        arguments.kwarg,
    )
    exprs: list[ast.AST] = [*node.decorator_list, *arguments.defaults]
    exprs.extend(default for default in arguments.kw_defaults if default)
    exprs.extend(
        argument.annotation
        for argument in params
        if argument is not None and argument.annotation is not None
    )
    if node.returns is not None:
        exprs.append(node.returns)
    exprs.extend(node.type_params)
    return exprs


def _python_refs(tree: ast.Module, lines: list[str], path: str) -> tuple[list[Ref], list[Import]]:
    collector = _RefCollector(lines, path)
    collector.visit(tree)
    return collector.refs, collector.imports


# -------------------------------- Markdown -------------------------------- #
# CommonMark: до 3 пробелов отступа; закрывающие '#' — только после пробела ("# C#").
_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$")
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def _closes_fence(line: str, fence: str) -> bool:
    """Закрывает ли строка блок кода с оградой ``fence``.

    Закрывает только ограда того же символа, не короче, без текста после.
    """
    match = _FENCE_RE.match(line)
    if not match:
        return False
    marker = match.group(1)
    same_marker = marker[0] == fence[0] and len(marker) >= len(fence)
    return same_marker and not line[match.end() :].strip()


def _lines_outside_code(text: str) -> Iterator[tuple[int, str]]:
    """Строки (с номерами от 1) вне блоков кода; сами ограды блоков не выдаются."""
    fence = ""  # открывающая ограда блока кода ("```", "~~~~"…), пусто — вне блока
    for number, line in enumerate(split_lines(text), start=1):
        if fence:
            if _closes_fence(line, fence):
                fence = ""
            continue
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            fence = fence_match.group(1)
            continue
        yield number, line


def _markdown_symbols(text: str) -> list[Symbol]:
    out: list[Symbol] = []
    stack: list[tuple[int, str]] = []  # (уровень, заголовок)
    for number, line in _lines_outside_code(text):
        heading = _HEADING_RE.match(line)
        if not heading:
            continue
        level, title = len(heading.group(1)), heading.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        out.append(
            Symbol(
                name=title[:_MAX_SIGNATURE],
                kind="section",
                line=number,
                parent=stack[-1][1] if stack else "",
                depth=len(stack),
                signature=clip_signature(line),
            ),
        )
        stack.append((level, title))
        if len(out) >= MAX_SYMBOLS_PER_FILE:
            break
    return out


# ------------------------------ Регэкспы ------------------------------ #
# Группы: name — имя; kind — вид (если не задан правилом); parent — владелец.
_Rule = tuple[str, "re.Pattern[str]"]
_NOT_NAMES = frozenset((
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "return",
    "else",
    "new",
    "sizeof",
    "do",
    "with",
))
_ID = r"[A-Za-z_$][\w$]*"


def _rules(*pairs: tuple[str, str]) -> list[_Rule]:
    return [(kind, re.compile(pattern)) for kind, pattern in pairs]


_JS_RULES = _rules(
    (
        _FUNCTION,
        rf"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*(?P<name>{_ID})",
    ),
    (_CLASS, rf"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(?P<name>{_ID})"),
    ("interface", rf"^\s*(?:export\s+)?(?:declare\s+)?interface\s+(?P<name>{_ID})"),
    (_TYPE, rf"^\s*(?:export\s+)?(?:declare\s+)?type\s+(?P<name>{_ID})\s*(?:<[^=]*>)?\s*="),
    ("enum", rf"^\s*(?:export\s+)?(?:declare\s+)?(?:const\s+)?enum\s+(?P<name>{_ID})"),
    (
        _FUNCTION,
        (
            rf"^\s*(?:export\s+)?(?:const|let|var)\s+(?P<name>{_ID})\s*(?::[^=]+)?=\s*"
            rf"(?:async\s+)?(?:function\b|\([^)]*\)\s*(?::[^=]+)?=>|{_ID}\s*=>)"
        ),
    ),
    (
        _METHOD,
        (
            r"^\s+(?:(?:public|private|protected|static|readonly|async|override|get|set)\s+)*"
            rf"(?P<name>{_ID})\s*\([^)]*\)"
            r"\s*(?::\s*[^{]+)?\{\s*$"
        ),
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

_RULES: Mapping[str, list[_Rule]] = MappingProxyType({
    "python": _rules(
        (_CLASS, r"^(?P<indent>\s*)class\s+(?P<name>[A-Za-z_]\w*)"),
        (_FUNCTION, r"^(?P<indent>\s*)(?:async\s+)?def\s+(?P<name>[A-Za-z_]\w*)"),
    ),
    "javascript": _JS_RULES,
    "typescript": _JS_RULES,
    "go": _rules(
        (
            _METHOD,
            r"^func\s+\(\s*\w*\s*\*?\s*(?P<parent>[A-Za-z_]\w*)[^)]*\)\s*(?P<name>[A-Za-z_]\w*)",
        ),
        (_FUNCTION, r"^func\s+(?P<name>[A-Za-z_]\w*)"),
        (_TYPE, r"^type\s+(?P<name>[A-Za-z_]\w*)\s+(?P<kind>struct|interface)\b"),
        (_TYPE, r"^type\s+(?P<name>[A-Za-z_]\w*)"),
    ),
    "rust": _rules(
        (
            _FUNCTION,
            (
                r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?"
                r'(?:extern\s+"[^"]*"\s+)?fn\s+(?P<name>[A-Za-z_]\w*)'
            ),
        ),
        (
            _TYPE,
            (
                r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?P<kind>struct|enum|trait|union|mod|type)\s+"
                r"(?P<name>[A-Za-z_]\w*)"
            ),
        ),
        (
            "impl",
            r"^\s*impl(?:<[^>]*>)?\s+(?:[\w:<>, ]+\s+for\s+)?(?P<name>[A-Za-z_][\w:]*)",
        ),
        ("macro", r"^\s*macro_rules!\s*(?P<name>[A-Za-z_]\w*)"),
    ),
    "java": _rules((_CLASS, _TYPES_JVM), (_METHOD, _JAVA_METHOD)),
    "csharp": _rules((_CLASS, _TYPES_JVM), (_METHOD, _JAVA_METHOD)),
    "kotlin": _rules(
        (_CLASS, _TYPES_JVM),
        (
            _FUNCTION,
            r"^\s*(?:\w+\s+)*fun\s+(?:<[^>]*>\s*)?(?:[\w.<>?, ]+\.)?(?P<name>[A-Za-z_]\w*)",
        ),
    ),
    "scala": _rules((_CLASS, _TYPES_JVM), (_FUNCTION, r"^\s*(?:\w+\s+)*def\s+(?P<name>\w+)")),
    "swift": _rules((_CLASS, _TYPES_JVM), (_FUNCTION, r"^\s*(?:\w+\s+)*func\s+(?P<name>\w+)")),
    "ruby": _rules(
        (_CLASS, r"^\s*(?P<kind>class|module)\s+(?P<name>[A-Z][\w:]*)"),
        (_METHOD, r"^\s*def\s+(?:self\.)?(?P<name>[\w?!=]+)"),
    ),
    "php": _rules(
        (
            _CLASS,
            (
                r"^\s*(?:(?:abstract|final|readonly)\s+)*(?P<kind>class|interface|trait|enum)\s+"
                r"(?P<name>\w+)"
            ),
        ),
        (
            _FUNCTION,
            (
                r"^\s*(?:(?:public|private|protected|static|abstract|final)\s+)*function\s+&?\s*"
                r"(?P<name>\w+)"
            ),
        ),
    ),
    "c": _rules(
        (_TYPE, r"^\s*(?:typedef\s+)?(?P<kind>struct|union|enum)\s+(?P<name>[A-Za-z_]\w*)\s*\{"),
        (
            _FUNCTION,
            (
                r"^(?!\s)(?!(?:return|else|if|while|for|switch|do)\b)[\w\s\*]*?\b"
                r"(?P<name>[A-Za-z_]\w*)\s*\([^;]*\)\s*\{?\s*$"
            ),
        ),
    ),
    _CPP: _rules(
        (
            _TYPE,
            (
                r"^\s*(?:template\s*<[^>]*>\s*)?(?:typedef\s+)?(?P<kind>class|struct|union|enum)"
                r"(?:\s+class)?\s+(?P<name>[A-Za-z_]\w*)\s*(?::[^{;]*)?\{?\s*$"
            ),
        ),
        ("namespace", r"^\s*namespace\s+(?P<name>[A-Za-z_][\w:]*)"),
        (
            _FUNCTION,
            (
                r"^(?!\s)(?!(?:return|else|if|while|for|switch|do)\b)[\w\s\*&:<>,~]*?\b"
                r"(?P<name>[A-Za-z_~][\w:~]*)\s*\([^;]*\)\s*(?:const\s*)?(?:noexcept\s*)?"
                r"(?:override\s*)?\{?\s*$"
            ),
        ),
    ),
    "shell": _rules(
        (_FUNCTION, r"^\s*(?:function\s+)?(?P<name>[A-Za-z_][\w-]*)\s*\(\)\s*\{?"),
        (_FUNCTION, r"^\s*function\s+(?P<name>[A-Za-z_][\w-]*)"),
    ),
})


def _regex_symbol(line: str, number: int, rules: list[_Rule]) -> Symbol | None:
    """Символ строки по первому сработавшему правилу (None — строка не определение)."""
    for kind, regex in rules:
        match = regex.match(line)
        if not match:
            continue
        name = match.group("name")
        if name in _NOT_NAMES:
            return None
        groups = match.groupdict()
        indent = groups.get("indent")
        return Symbol(
            name=name,
            kind=(groups.get("kind") or kind).lower(),
            line=number,
            parent=groups.get("parent") or "",
            depth=1 if indent else 0,
            signature=clip_signature(line),
            col=match.start("name"),
        )
    return None


def _regex_symbols(text: str, rules: list[_Rule]) -> list[Symbol]:
    out: list[Symbol] = []
    for number, line in enumerate(split_lines(text), start=1):
        if not line.strip() or len(line) > _MAX_REGEX_LINE:
            continue
        symbol = _regex_symbol(line, number, rules)
        if symbol is not None:
            out.append(symbol)
        if len(out) >= MAX_SYMBOLS_PER_FILE:
            break
    return out
