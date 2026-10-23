"""Разбор исходников tree-sitter: определения и вызовы.

Определения — с диапазоном строк и вложенностью; разбираются языки, которые индекс
раньше разбирал построчными регэкспами.

Для каждого языка — таблица «тип узла → вид определения» и «тип узла-вызова →
поле с вызываемым»; обход дерева учитывает вложенность (метод — внутри класса).
Из API tree-sitter используются только узлы (тип, поля, позиции, текст): этот
модуль — единственное место, где tree-sitter загружается и импортируется. Ещё один
потребитель узлов — разбор команд оболочки (:func:`parse_tree`, :mod:`gigacrew.safety.bash`).

Грамматики — отдельные колёса ``tree-sitter-<язык>`` (офлайн). Не загрузилась —
:func:`extract` возвращает None, и вызывающий откатывается на регэкспы.
Python индекс разбирает модулем ``ast``; tree-sitter для него — запасной путь,
когда ``ast`` не справился (синтаксическая ошибка посреди правки, синтаксис новее
интерпретатора).
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from gigacrew.project.symbols import (
    MAX_REFS_PER_FILE,
    MAX_SYMBOLS_PER_FILE,
    FileFacts,
    Ref,
    Symbol,
    clip_signature,
)

try:
    from tree_sitter import Language, Node, Parser
except ImportError:  # pragma: no cover — tree-sitter — обязательная зависимость
    _HAS_TREE_SITTER = False
else:
    _HAS_TREE_SITTER = True

_ENCODING = "utf-8"
_MAX_SIGNATURE_LINES = 6
_MAX_SIGNATURE_BYTES = 400  # сигнатура определения без тела — не длиннее
_MAX_NAME_DEPTH = 16  # глубина вложенности составного имени
# Ошибки загрузки грамматики: колеса нет, нет функции языка или она вернула не то.
_LOAD_ERRORS = (ImportError, AttributeError, ValueError, TypeError)

# Виды, внутри которых функция — метод.
_METHOD_OWNERS = frozenset((
    "class",
    "struct",
    "interface",
    "trait",
    "record",
    "object",
    "protocol",
    "enum",
    "union",
    "impl",
    "extension",
))
_FUNCTION_KINDS = frozenset(("function", "method"))

# Узлы-имена, текст которых — вызываемое имя.
_NAME_NODES = frozenset((
    "identifier",
    "simple_identifier",
    "field_identifier",
    "property_identifier",
    "type_identifier",
    "name",
    "constant",
    "word",
    "namespace_identifier",
    "private_property_identifier",
    "destructor_name",
    "operator_name",
    "string_fragment",  # JS: метод с именем в кавычках — `"save"() {}`
))
# Поля, в которых лежит последняя часть составного имени (a.b.c → c).
_NAME_FIELDS = ("name", "field", "property", "attribute", "suffix", "method")
# Составные имена без поля с последней частью: она — последний именованный потомок…
_NAME_LAST_CHILD = frozenset(("navigation_expression", "scoped_identifier"))
# …или первый: `(load)()`, `"save"() {}`.
_NAME_FIRST_CHILD = frozenset((
    "command_name",
    "generic_name",
    "user_type",
    "parenthesized_expression",
    "string",
))

# Узлы-тела: сигнатура — от начала определения до тела.
_BODY_NODES = frozenset((
    "block",
    "body_statement",
    "class_body",
    "compound_statement",
    "declaration_list",
    "enum_body",
    "enum_class_body",
    "field_declaration_list",
    "function_body",
    "interface_body",
    "protocol_body",
    "statement_block",
    "template_body",
    "enum_variant_list",
    "enumerator_list",
))
_COMMENT_PREFIXES = ("//", "#")


@dataclass(frozen=True)
class _Spec:
    """Как разбирать язык: модуль грамматики, определения и вызовы."""

    module: str
    definitions: Mapping[str, str]
    # тип узла-вызова → поле с вызываемым ("" — первый именованный потомок)
    calls: Mapping[str, str] = field(default_factory=dict)
    function: str = "language"


_JS_DEFS = MappingProxyType({
    "class_declaration": "class",
    "function_declaration": "function",
    "generator_function_declaration": "function",
    "method_definition": "method",
    "variable_declarator": "function",  # только `const f = () => …` (см. _is_js_function)
})
_JS_CALLS = MappingProxyType({"call_expression": "function", "new_expression": "constructor"})
_TS_DEFS = MappingProxyType(
    _JS_DEFS
    | {
        "abstract_class_declaration": "class",
        "interface_declaration": "interface",
        "type_alias_declaration": "type",
        "enum_declaration": "enum",
        "internal_module": "namespace",
        "module": "namespace",
        "method_signature": "method",
        "abstract_method_signature": "method",
        "function_signature": "function",
    },
)
_C_DEFS = MappingProxyType({
    "function_definition": "function",
    "struct_specifier": "struct",
    "union_specifier": "union",
    "enum_specifier": "enum",
    "type_definition": "type",
})

SPECS: Mapping[str, _Spec] = MappingProxyType({
    "python": _Spec(
        "tree_sitter_python",
        {"class_definition": "class", "function_definition": "function"},
        {"call": "function"},
    ),
    "javascript": _Spec("tree_sitter_javascript", _JS_DEFS, _JS_CALLS),
    "typescript": _Spec("tree_sitter_typescript", _TS_DEFS, _JS_CALLS, "language_typescript"),
    "tsx": _Spec("tree_sitter_typescript", _TS_DEFS, _JS_CALLS, "language_tsx"),
    "go": _Spec(
        "tree_sitter_go",
        {
            "function_declaration": "function",
            "method_declaration": "method",
            "type_spec": "type",
            "method_elem": "method",
        },
        {"call_expression": "function"},
    ),
    "rust": _Spec(
        "tree_sitter_rust",
        {
            "function_item": "function",
            "function_signature_item": "function",
            "struct_item": "struct",
            "enum_item": "enum",
            "union_item": "union",
            "trait_item": "trait",
            "impl_item": "impl",
            "mod_item": "module",
            "type_item": "type",
            "macro_definition": "macro",
            "const_item": "constant",
            "static_item": "constant",
        },
        {"call_expression": "function"},
    ),
    "java": _Spec(
        "tree_sitter_java",
        {
            "class_declaration": "class",
            "interface_declaration": "interface",
            "enum_declaration": "enum",
            "record_declaration": "record",
            "annotation_type_declaration": "interface",
            "method_declaration": "method",
            "constructor_declaration": "method",
        },
        {"method_invocation": "name", "object_creation_expression": "type"},
    ),
    "kotlin": _Spec(
        "tree_sitter_kotlin",
        {
            "class_declaration": "class",
            "object_declaration": "object",
            "function_declaration": "function",
        },
        {"call_expression": ""},
    ),
    "csharp": _Spec(
        "tree_sitter_c_sharp",
        {
            "namespace_declaration": "namespace",
            "file_scoped_namespace_declaration": "namespace",
            "class_declaration": "class",
            "interface_declaration": "interface",
            "struct_declaration": "struct",
            "enum_declaration": "enum",
            "record_declaration": "record",
            "method_declaration": "method",
            "constructor_declaration": "method",
            "local_function_statement": "function",
        },
        {"invocation_expression": "function", "object_creation_expression": "type"},
    ),
    "ruby": _Spec(
        "tree_sitter_ruby",
        {"class": "class", "module": "module", "method": "method", "singleton_method": "method"},
        {"call": "method"},
    ),
    "php": _Spec(
        "tree_sitter_php",
        {
            "class_declaration": "class",
            "interface_declaration": "interface",
            "trait_declaration": "trait",
            "enum_declaration": "enum",
            "function_definition": "function",
            "method_declaration": "method",
        },
        {
            "function_call_expression": "function",
            "member_call_expression": "name",
            "nullsafe_member_call_expression": "name",
            "scoped_call_expression": "name",
        },
        "language_php",
    ),
    "c": _Spec("tree_sitter_c", _C_DEFS, {"call_expression": "function"}),
    "cpp": _Spec(
        "tree_sitter_cpp",
        _C_DEFS | {"class_specifier": "class", "namespace_definition": "namespace"},
        {"call_expression": "function"},
    ),
    "shell": _Spec("tree_sitter_bash", {"function_definition": "function"}, {"command": "name"}),
    "scala": _Spec(
        "tree_sitter_scala",
        {
            "class_definition": "class",
            "object_definition": "object",
            "trait_definition": "trait",
            "enum_definition": "enum",
            "function_definition": "function",
            "function_declaration": "function",
        },
        {"call_expression": "function"},
    ),
    "swift": _Spec(
        "tree_sitter_swift",
        {
            "class_declaration": "class",
            "protocol_declaration": "protocol",
            "function_declaration": "function",
            "protocol_function_declaration": "function",
        },
        {"call_expression": ""},
    ),
})

_parsers: dict[str, Parser | None] = {}  # язык -> Parser | None (не загрузилась)


def _parser(language: str) -> Parser | None:
    if language not in _parsers:
        parser: Parser | None = None
        spec = SPECS.get(language)
        if spec is not None and _HAS_TREE_SITTER:
            try:
                module = importlib.import_module(spec.module)
                pointer: object = getattr(module, spec.function)()
                parser = Parser(Language(pointer))
            except _LOAD_ERRORS:
                parser = None
        _parsers[language] = parser
    return _parsers[language]


def supports(language: str | None) -> bool:
    """Есть ли для языка таблица разбора tree-sitter."""
    return language in SPECS


def parse_tree(language: str, source: str) -> Node | None:
    """Синтаксическое дерево ``source`` (корневой узел); None — грамматика не загрузилась.

    Для разбора команд оболочки (:mod:`gigacrew.safety.bash`): ему нужны узлы целиком, а
    не определения и вызовы, которые извлекает :func:`extract`.
    """
    parser = _parser(language)
    if parser is None:
        return None
    return parser.parse(source.encode(_ENCODING, errors="surrogateescape")).root_node


# --------------------------------------------------------------------------- #
# Имена
# --------------------------------------------------------------------------- #
def _text(node: Node) -> str:
    raw = node.text
    return "" if raw is None else raw.decode(_ENCODING, errors="replace")


def _leaf_name(node: Node | None) -> Node | None:
    """Последняя часть имени: ``a.b.c`` → узел ``c``."""
    for _ in range(_MAX_NAME_DEPTH):
        if node is None or node.type in _NAME_NODES:
            return node
        inner = _inner_name(node)
        if inner is None:
            return node if node.type == "command_name" else None
        node = inner
    return None


def _inner_name(node: Node) -> Node | None:
    """Узел, в котором продолжается составное имя; None — продолжения нет."""
    for name in _NAME_FIELDS:
        child = node.child_by_field_name(name)
        if child is not None:
            return child
    named = node.named_children
    if not named:
        return None
    if node.type in _NAME_LAST_CHILD:
        return named[-1]
    if node.type in _NAME_FIRST_CHILD:
        return named[0]
    return None


def _declarator_name(node: Node | None) -> Node | None:
    """C/C++: имя из цепочки деклараторов (``*f(…)``, ``ns::A::run(…)``)."""
    for _ in range(_MAX_NAME_DEPTH):
        if node is None:
            return None
        if node.type in _NAME_NODES:
            return node
        if node.type in {"qualified_identifier", "template_function"}:
            node = node.child_by_field_name("name")
            continue
        node = node.child_by_field_name("declarator")
    return None


def _keyword(node: Node, words: frozenset[str]) -> str | None:
    for child in node.children:
        if not child.is_named and child.type in words:
            return child.type
    return None


# --------------------------------------------------------------------------- #
# Определения
# --------------------------------------------------------------------------- #
_C_LANGUAGES = frozenset(("c", "cpp"))
# C/C++: типы, которые без тела — упоминание (`struct S x;`), а не определение.
_C_TYPE_SPECIFIERS = frozenset((
    "struct_specifier",
    "union_specifier",
    "enum_specifier",
    "class_specifier",
))
# C/C++: определения, имя которых — в цепочке деклараторов.
_C_DECLARATIONS = frozenset(("function_definition", "type_definition"))
_JS_FUNCTION_VALUES = frozenset((
    "arrow_function",
    "function_expression",
    "function",
    "generator_function",
))
_GO_TYPE_KINDS = MappingProxyType({"struct_type": "struct", "interface_type": "interface"})
_RUST_TYPE_NAMES = frozenset(("type_identifier", "primitive_type"))
_KOTLIN_INTERFACE = frozenset(("interface",))
_SWIFT_KINDS = frozenset(("class", "struct", "enum", "extension", "actor"))


def _definition(node: Node, kind: str, language: str) -> tuple[Node | None, str, str] | None:
    """(узел имени, вид, владелец) определения или None — узел не определение."""
    node_type = node.type
    if language in _C_LANGUAGES:
        if node_type in _C_TYPE_SPECIFIERS and node.child_by_field_name("body") is None:
            return None  # упоминание типа (`struct S x;`), а не определение
        if node_type in _C_DECLARATIONS:
            declarator = node.child_by_field_name("declarator")
            return _declarator_name(declarator), kind, _qualified_owner(declarator)
    if node_type == "variable_declarator" and not _is_js_function(node):
        return None  # JS/TS: функция — только `const f = () => …`
    return _language_definition(node, kind, language)


def _qualified_owner(declarator: Node | None) -> str:
    """C++: владелец из квалифицированного имени (``void ns::A::run()`` → ``ns::A``)."""
    qualified = declarator
    while qualified is not None and qualified.type != "qualified_identifier":
        qualified = qualified.child_by_field_name("declarator")
    if qualified is None:
        return ""
    return _text(qualified).rsplit("::", 1)[0]


def _is_js_function(node: Node) -> bool:
    """JS/TS: значение переменной — функция (``const f = () => …``)."""
    initializer = node.child_by_field_name("value")
    return initializer is not None and initializer.type in _JS_FUNCTION_VALUES


def _language_definition(node: Node, kind: str, language: str) -> tuple[Node | None, str, str]:
    """Определение с особенностями языка; по умолчанию имя — из поля ``name``."""
    node_type = node.type
    if language == "go" and node_type == "method_declaration":
        return node.child_by_field_name("name"), kind, _go_receiver_type(node)
    if language == "rust" and node_type == "impl_item":
        return _rust_impl_type(node), kind, ""
    if language == "go" and node_type == "type_spec":
        kind = _go_type_kind(node)
    elif language == "kotlin" and node_type == "class_declaration":
        kind = _kotlin_class_kind(node, kind)
    elif language == "swift" and node_type == "class_declaration":
        kind = _keyword(node, _SWIFT_KINDS) or "class"
    name = node.child_by_field_name("name")
    if kind == "namespace":  # `namespace Company.App` — имя целиком, а не последняя часть
        return name, kind, ""
    return _leaf_name(name), kind, ""


def _go_type_kind(node: Node) -> str:
    """Go: вид ``type X …`` — по типу справа (структура, интерфейс или просто тип)."""
    body = node.child_by_field_name("type")
    if body is None:
        return "type"
    return _GO_TYPE_KINDS.get(body.type, "type")


def _go_receiver_type(node: Node) -> str:
    """Go: владелец метода — тип получателя (``func (s *Server) …`` → ``Server``)."""
    receiver = node.child_by_field_name("receiver")
    if receiver is None:
        return ""
    types = [child for child in _walk(receiver) if child.type == "type_identifier"]
    return _text(types[0]) if types else ""


def _rust_impl_type(node: Node) -> Node | None:
    """Rust: имя типа в ``impl`` (``impl<T> Trait for Wrapper<T>`` → ``Wrapper``)."""
    name = node.child_by_field_name("type")
    while name is not None and name.type not in _RUST_TYPE_NAMES:
        inner = name.child_by_field_name("type")
        if inner is None:
            named = name.named_children
            inner = named[0] if named else None
        name = inner
    return name


def _kotlin_class_kind(node: Node, kind: str) -> str:
    """Kotlin: ``class_declaration`` — это и ``interface``, и ``enum class``."""
    if _keyword(node, _KOTLIN_INTERFACE):
        return "interface"
    modifiers = (_text(child).split() for child in node.children if child.type == "modifiers")
    if any("enum" in words for words in modifiers):
        return "enum"
    return kind


def _walk(node: Node) -> Iterator[Node]:
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.named_children))


# --------------------------------------------------------------------------- #
# Разбор
# --------------------------------------------------------------------------- #
class _Source:
    """Текст файла: строки — для столбцов имён, байты UTF-8 — для разбора и сигнатур."""

    def __init__(self, text: str) -> None:
        # строки — по "\n", как у tree-sitter (splitlines режет и по \x0c, \u2028…)
        self.lines = text.split("\n")
        self.encoded = text.encode(_ENCODING)

    def line(self, row: int) -> str:
        return self.lines[row] if 0 <= row < len(self.lines) else ""

    def col(self, point: tuple[int, int]) -> int:
        row, byte_col = point
        line = self.line(row)
        if line.isascii():
            return byte_col
        return len(line.encode(_ENCODING)[:byte_col].decode(_ENCODING, errors="ignore"))

    def signature(self, node: Node) -> str:
        end = _signature_end(node)
        text = self.encoded[node.start_byte : end].decode(_ENCODING, errors="replace")
        parts = [part.strip() for part in text.split("\n")[:_MAX_SIGNATURE_LINES]]
        code = [part for part in parts if part and not part.startswith(_COMMENT_PREFIXES)]
        return clip_signature(" ".join(code))


def _signature_end(node: Node) -> int:
    """Конец сигнатуры (байт): начало тела, а без тела — с ограничением длины."""
    body = _body(node)
    if body is not None and body.start_byte > node.start_byte:
        return body.start_byte
    return min(node.end_byte, node.start_byte + _MAX_SIGNATURE_BYTES)


def _body(node: Node) -> Node | None:
    body = node.child_by_field_name("body")
    if body is None:  # `const f = (x) => {…}` — тело у функции в поле value
        function = node.child_by_field_name("value")
        body = None if function is None else function.child_by_field_name("body")
    if body is None:
        body = next((child for child in node.named_children if child.type in _BODY_NODES), None)
    return body


class _Extractor:
    """Обход дерева: определения с вложенностью и вызовы в их областях."""

    def __init__(self, source: _Source, spec: _Spec, language: str) -> None:
        self.source = source
        self.symbols: list[Symbol] = []
        self.refs: list[Ref] = []
        self._spec = spec
        self._language = language
        self._seen_refs: set[tuple[str, int, str]] = set()
        # открытые определения: (конечный байт, qualname, вид); вложенные в функции
        # определения не записываются (шум, как у ast), их вызовы — в области внешней
        self._scopes: list[tuple[int, str, str]] = []
        self._in_function = 0

    def walk(self, root: Node) -> None:
        """Обходит дерево в прямом порядке, закрывая области определений на выходе."""
        stack: list[tuple[Node, bool]] = [(root, False)]
        while stack:
            node, leaving = stack.pop()
            if leaving:
                self._close_scope()
                continue
            opened = self._open_definition(node)
            self._add_call(node)
            # `namespace App;` (C#) действует до конца файла: объявления после него — его
            # соседи в дереве, поэтому область не закрывается вместе с узлом
            if opened and node.type != "file_scoped_namespace_declaration":
                stack.append((node, True))
            stack.extend((child, False) for child in reversed(node.named_children))

    def _close_scope(self) -> None:
        _, _, kind = self._scopes.pop()
        if kind in _FUNCTION_KINDS:
            self._in_function -= 1

    def _open_definition(self, node: Node) -> bool:
        """Записывает определение в узле и открывает его область; False — не определение."""
        kind = self._spec.definitions.get(node.type)
        if kind is None or self._in_function:
            return False
        found = _definition(node, kind, self._language)
        if found is None:
            return False
        name_node, kind, owner = found
        if name_node is None:
            return False
        if self._scopes:
            _, parent, owner_kind = self._scopes[-1]
        else:
            parent = owner
            owner_kind = "impl" if owner else ""
        if kind == "function" and owner_kind in _METHOD_OWNERS:
            kind = "method"
        name = _text(name_node)
        if len(self.symbols) < MAX_SYMBOLS_PER_FILE and name:
            self._add_symbol(node, name_node, name, kind, parent)
        qualname = f"{parent}.{name}" if parent else name
        self._scopes.append((node.end_byte, qualname, kind))
        if kind in _FUNCTION_KINDS:
            self._in_function += 1
        return True

    def _add_symbol(self, node: Node, name_node: Node, name: str, kind: str, parent: str) -> None:
        self.symbols.append(
            Symbol(
                name=name,
                kind=kind,
                line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                parent=parent,
                depth=len(self._scopes),
                signature=self.source.signature(node),
                col=self.source.col(name_node.start_point),
            ),
        )

    def _add_call(self, node: Node) -> None:
        call_field = self._spec.calls.get(node.type)
        if call_field is None:
            return
        if call_field:
            target = node.child_by_field_name(call_field)
        else:
            named = node.named_children
            target = named[0] if named else None
        self._add_ref(_leaf_name(target))

    def _add_ref(self, name_node: Node | None) -> None:
        if name_node is None or len(self.refs) >= MAX_REFS_PER_FILE:
            return
        name = _text(name_node)
        # однобуквенные — шум (как у Python); `#name` — приватные члены JS
        if len(name) < 2 or not (name[0].isalpha() or name[0] in "_$#"):
            return
        scope = self._scopes[-1][1] if self._scopes else ""
        line = name_node.start_point[0] + 1
        key = (name, line, scope)
        if key in self._seen_refs:
            return
        self._seen_refs.add(key)
        col = self.source.col(name_node.start_point)
        self.refs.append(Ref(name=name, kind="call", line=line, col=col, scope=scope))


def extract(text: str, language: str, path: str = "") -> FileFacts | None:
    """Определения и вызовы файла; None — грамматика недоступна."""
    if language == "typescript" and path.endswith((".tsx", ".TSX")):
        language = "tsx"
    spec = SPECS.get(language)
    parser = None if spec is None else _parser(language)
    if spec is None or parser is None:
        return None
    extractor = _Extractor(_Source(text), spec, language)
    extractor.walk(parser.parse(extractor.source.encoded).root_node)
    return FileFacts(symbols=extractor.symbols, refs=extractor.refs)
