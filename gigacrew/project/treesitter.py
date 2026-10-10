"""Разбор исходников tree-sitter: определения (с диапазоном строк и вложенностью) и
вызовы — для языков, которые индекс раньше разбирал построчными регэкспами.

Для каждого языка — таблица «тип узла → вид определения» и «тип узла-вызова →
поле с вызываемым»; обход дерева учитывает вложенность (метод — внутри класса).
Из API tree-sitter используются только узлы (тип, поля, позиции, текст): этот
модуль — единственное место, где tree-sitter встречается в коде.

Грамматики — отдельные колёса ``tree-sitter-<язык>`` (офлайн). Не загрузилась —
:func:`extract` возвращает None, и вызывающий откатывается на регэкспы.
Python индекс разбирает модулем ``ast``; tree-sitter для него — запасной путь,
когда ``ast`` не справился (синтаксическая ошибка посреди правки, синтаксис новее
интерпретатора).
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

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
    Language = Node = Parser = None  # type: ignore[assignment,misc]

_MAX_SIGNATURE_LINES = 6

# Виды, внутри которых функция — метод.
_METHOD_OWNERS = frozenset(
    {
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
    }
)
_FUNCTION_KINDS = frozenset({"function", "method"})

# Узлы-имена, текст которых — вызываемое имя.
_NAME_NODES = frozenset(
    {
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
    }
)
# Поля, в которых лежит последняя часть составного имени (a.b.c → c).
_NAME_FIELDS = ("name", "field", "property", "attribute", "suffix", "method")

# Узлы-тела: сигнатура — от начала определения до тела.
_BODY_NODES = frozenset(
    {
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
    }
)


@dataclass(frozen=True)
class _Spec:
    """Как разбирать язык: модуль грамматики, определения и вызовы."""

    module: str
    definitions: Mapping[str, str]
    # тип узла-вызова → поле с вызываемым ("" — первый именованный потомок)
    calls: Mapping[str, str] = field(default_factory=dict)
    function: str = "language"


_JS_DEFS = {
    "class_declaration": "class",
    "function_declaration": "function",
    "generator_function_declaration": "function",
    "method_definition": "method",
    "variable_declarator": "function",  # только `const f = () => …` (см. _js_function)
}
_JS_CALLS = {"call_expression": "function", "new_expression": "constructor"}
_TS_DEFS = _JS_DEFS | {
    "abstract_class_declaration": "class",
    "interface_declaration": "interface",
    "type_alias_declaration": "type",
    "enum_declaration": "enum",
    "internal_module": "namespace",
    "module": "namespace",
    "method_signature": "method",
    "abstract_method_signature": "method",
    "function_signature": "function",
}
_C_DEFS = {
    "function_definition": "function",
    "struct_specifier": "struct",
    "union_specifier": "union",
    "enum_specifier": "enum",
    "type_definition": "type",
}

SPECS: dict[str, _Spec] = {
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
}

_parsers: dict[str, Any] = {}  # язык -> Parser | None (не загрузилась)


def _parser(language: str) -> Any:
    if language not in _parsers:
        parser = None
        spec = SPECS.get(language)
        if spec is not None and Parser is not None:
            try:
                module = importlib.import_module(spec.module)
                parser = Parser(Language(getattr(module, spec.function)()))
            except (ImportError, AttributeError, ValueError, TypeError):
                parser = None
        _parsers[language] = parser
    return _parsers[language]


def supports(language: str | None) -> bool:
    return language in SPECS


# --------------------------------------------------------------------------- #
# Имена
# --------------------------------------------------------------------------- #
def _text(node: Node) -> str:
    return node.text.decode("utf-8", errors="replace") if node.text is not None else ""


def _leaf_name(node: Node | None) -> Node | None:
    """Последняя часть имени: ``a.b.c`` → узел ``c``."""
    for _ in range(16):  # глубина вложенности составного имени
        if node is None:
            return None
        if node.type in _NAME_NODES:
            return node
        for name in _NAME_FIELDS:
            child = node.child_by_field_name(name)
            if child is not None:
                node = child
                break
        else:
            named = node.named_children
            if node.type in ("navigation_expression", "scoped_identifier") and named:
                node = named[-1]
            elif node.type in ("command_name", "generic_name", "user_type") and named:
                node = named[0]
            elif node.type in ("parenthesized_expression", "string") and named:
                node = named[0]  # `(load)()`, `"save"() {}`
            elif node.type == "command_name":
                return node
            else:
                return None
    return None


def _declarator_name(node: Node | None) -> Node | None:
    """C/C++: имя из цепочки деклараторов (``*f(…)``, ``ns::A::run(…)``)."""
    for _ in range(16):
        if node is None:
            return None
        if node.type in _NAME_NODES:
            return node
        if node.type in ("qualified_identifier", "template_function"):
            node = node.child_by_field_name("name")
            continue
        node = node.child_by_field_name("declarator")
    return None


def _keyword(node: Node, words: frozenset[str]) -> str | None:
    for child in node.children:
        if not child.is_named and child.type in words:
            return child.type
    return None


_SWIFT_KINDS = frozenset({"class", "struct", "enum", "extension", "actor"})


def _definition(node: Node, kind: str, language: str) -> tuple[Node | None, str, str] | None:
    """(узел имени, вид, владелец) определения или None — узел не определение."""
    t = node.type
    if language in ("c", "cpp"):
        if t in ("struct_specifier", "union_specifier", "enum_specifier", "class_specifier"):
            if node.child_by_field_name("body") is None:
                return None  # упоминание типа (`struct S x;`), а не определение
        if t in ("function_definition", "type_definition"):
            declarator = node.child_by_field_name("declarator")
            name = _declarator_name(declarator)
            owner = ""
            qualified = declarator
            while qualified is not None and qualified.type != "qualified_identifier":
                qualified = qualified.child_by_field_name("declarator")
            if qualified is not None:  # `void ns::A::run()` — владелец ns::A
                owner = _text(qualified).rsplit("::", 1)[0]
            return name, kind, owner
    if t == "variable_declarator":  # JS/TS: функция — только `const f = () => …`
        value = node.child_by_field_name("value")
        if value is None or value.type not in (
            "arrow_function",
            "function_expression",
            "function",
            "generator_function",
        ):
            return None
    if language == "go":
        if t == "type_spec":
            body = node.child_by_field_name("type")
            sub = {"struct_type": "struct", "interface_type": "interface"}
            kind = sub.get(body.type, "type") if body is not None else "type"
        elif t == "method_declaration":
            receiver = node.child_by_field_name("receiver")
            owner = ""
            if receiver is not None:
                types = [n for n in _walk(receiver) if n.type == "type_identifier"]
                owner = _text(types[0]) if types else ""
            return node.child_by_field_name("name"), kind, owner
    if language == "rust" and t == "impl_item":
        target = node.child_by_field_name("type")
        name = target
        while name is not None and name.type not in ("type_identifier", "primitive_type"):
            name = name.child_by_field_name("type") or (
                name.named_children[0] if name.named_children else None
            )
        return name, kind, ""
    if language == "kotlin" and t == "class_declaration":
        if _keyword(node, frozenset({"interface"})):
            kind = "interface"
        elif any(c.type == "modifiers" and "enum" in _text(c).split() for c in node.children):
            kind = "enum"
    if language == "swift" and t == "class_declaration":
        kind = _keyword(node, _SWIFT_KINDS) or "class"
    name = node.child_by_field_name("name")
    if kind == "namespace":  # `namespace Company.App` — имя целиком, а не последняя часть
        return name, kind, ""
    return _leaf_name(name), kind, ""


def _walk(node: Node):
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.named_children))


# --------------------------------------------------------------------------- #
# Разбор
# --------------------------------------------------------------------------- #
class _Source:
    def __init__(self, text: str):
        # строки — по "\n", как у tree-sitter (splitlines режет и по \x0c,  …)
        self.lines = text.split("\n")
        self.data = text.encode("utf-8")

    def line(self, row: int) -> str:
        return self.lines[row] if 0 <= row < len(self.lines) else ""

    def col(self, point: Any) -> int:
        row, byte_col = point
        line = self.line(row)
        if line.isascii():
            return byte_col
        return len(line.encode("utf-8")[:byte_col].decode("utf-8", errors="ignore"))

    def signature(self, node: Node) -> str:
        body = node.child_by_field_name("body")
        if body is None:  # `const f = (x) => {…}` — тело у функции в поле value
            value = node.child_by_field_name("value")
            body = value.child_by_field_name("body") if value is not None else None
        if body is None:
            body = next((c for c in node.named_children if c.type in _BODY_NODES), None)
        end = body.start_byte if body is not None and body.start_byte > node.start_byte else None
        if end is None:
            end = min(node.end_byte, node.start_byte + 400)
        text = self.data[node.start_byte : end].decode("utf-8", errors="replace")
        lines = [s.strip() for s in text.split("\n")[:_MAX_SIGNATURE_LINES]]
        return clip_signature(" ".join(s for s in lines if s and not s.startswith(("//", "#"))))


def extract(text: str, language: str, path: str = "") -> FileFacts | None:
    """Определения и вызовы файла; None — грамматика недоступна."""
    if language == "typescript" and path.endswith((".tsx", ".TSX")):
        language = "tsx"
    spec = SPECS.get(language)
    parser = _parser(language) if spec is not None else None
    if spec is None or parser is None:
        return None
    source = _Source(text)
    tree = parser.parse(source.data)

    symbols: list[Symbol] = []
    refs: list[Ref] = []
    seen_refs: set[tuple[str, int, str]] = set()
    # открытые определения: (конечный байт, qualname, вид); вложенные в функции
    # определения не записываются (шум, как у ast), их вызовы — в области внешней
    scopes: list[tuple[int, str, str]] = []
    in_function = 0

    def add_ref(name_node: Node | None) -> None:
        if name_node is None or len(refs) >= MAX_REFS_PER_FILE:
            return
        name = _text(name_node)
        # однобуквенные — шум (как у Python); `#name` — приватные члены JS
        if len(name) < 2 or not (name[0].isalpha() or name[0] in "_$#"):
            return
        scope = scopes[-1][1] if scopes else ""
        line = name_node.start_point[0] + 1
        key = (name, line, scope)
        if key in seen_refs:
            return
        seen_refs.add(key)
        col = source.col(name_node.start_point)
        refs.append(Ref(name=name, kind="call", line=line, col=col, scope=scope))

    stack: list[tuple[Node, bool]] = [(tree.root_node, False)]
    while stack:
        node, leaving = stack.pop()
        if leaving:
            _, _, kind = scopes.pop()
            if kind in _FUNCTION_KINDS:
                in_function -= 1
            continue
        t = node.type
        kind = spec.definitions.get(t)
        opened = False
        if kind is not None and not in_function:
            found = _definition(node, kind, language)
            if found is not None and found[0] is not None:
                name_node, kind, owner = found
                parent = scopes[-1][1] if scopes else owner
                owner_kind = scopes[-1][2] if scopes else ("impl" if owner else "")
                if kind == "function" and owner_kind in _METHOD_OWNERS:
                    kind = "method"
                name = _text(name_node)
                if len(symbols) < MAX_SYMBOLS_PER_FILE and name:
                    symbols.append(
                        Symbol(
                            name=name,
                            kind=kind,
                            line=node.start_point[0] + 1,
                            end_line=node.end_point[0] + 1,
                            parent=parent,
                            depth=len(scopes),
                            signature=source.signature(node),
                            col=source.col(name_node.start_point),
                        )
                    )
                qualname = f"{parent}.{name}" if parent else name
                scopes.append((node.end_byte, qualname, kind))
                opened = True
                if kind in _FUNCTION_KINDS:
                    in_function += 1
        call_field = spec.calls.get(t)
        if call_field is not None:
            target = node.child_by_field_name(call_field) if call_field else None
            if not call_field:
                named = node.named_children
                target = named[0] if named else None
            add_ref(_leaf_name(target))
        # `namespace App;` (C#) действует до конца файла: объявления после него — его
        # соседи в дереве, поэтому область не закрывается вместе с узлом
        if opened and t != "file_scoped_namespace_declaration":
            stack.append((node, True))
        stack.extend((child, False) for child in reversed(node.named_children))
    return FileFacts(symbols=symbols, refs=refs)
