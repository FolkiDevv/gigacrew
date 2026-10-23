"""Тесты извлечения символов для индекса проекта."""

from __future__ import annotations

import textwrap

import pytest

from gigacrew.project import treesitter
from gigacrew.project.symbols import (
    # Тест «белого ящика»: запасной путь на регэкспах проверяется напрямую, без грамматик.
    _RULES,  # noqa: PLC2701
    MAX_SYMBOLS_PER_FILE,
    FileFacts,
    # Тест «белого ящика»: запасной путь на регэкспах проверяется напрямую, без грамматик.
    _regex_symbols,  # noqa: PLC2701
    extract,
    extract_symbols,
    language_of,
    module_parts,
)


def _names(text: str, language: str) -> list[tuple[str, str]]:
    symbols = extract_symbols(textwrap.dedent(text), language)
    return [(symbol.kind, symbol.qualname) for symbol in symbols]


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("a.py", "python"),
        ("src/x.TS", "typescript"),
        ("Makefile", "make"),
        ("README.md", "markdown"),
        ("data.bin", None),
        ("noext", None),
    ],
)
def test_language_of(path: str, expected: str | None) -> None:
    assert language_of(path) == expected


def test_python_symbols_with_nesting_and_ranges() -> None:
    src = textwrap.dedent(
        '''\
        """Модуль."""
        import os

        MAX_SIZE = 10
        name = "не константа"


        class Agent(Base):
            """Документация."""

            def run(self, x: int) -> int:
                def inner():
                    pass
                return x

            async def stream(
                self,
                prompt: str,
            ) -> None:
                pass

            class Config:
                pass


        async def main():
            pass


        if True:
            def conditional():
                pass
        ''',
    )
    symbols = extract_symbols(src, "python")
    got = [(sym.kind, sym.qualname, sym.line, sym.end_line, sym.depth) for sym in symbols]
    assert got == [
        ("constant", "MAX_SIZE", 4, 4, 0),
        ("class", "Agent", 8, 23, 0),
        ("method", "Agent.run", 11, 14, 1),
        ("method", "Agent.stream", 16, 20, 1),
        ("class", "Agent.Config", 22, 23, 1),
        ("function", "main", 26, 27, 0),
        ("function", "conditional", 31, 32, 0),
    ]
    stream = symbols[3]
    assert stream.signature == "async def stream( self, prompt: str, ) -> None:"
    assert symbols[1].signature == "class Agent(Base):"


def test_python_syntax_error_falls_back_to_tree_sitter() -> None:
    symbols = extract_symbols(
        textwrap.dedent(
            """\
            class Broken:
                def method(self):
                    return (
            def tail():
                pass
            """,
        ),
        "python",
    )
    assert [(sym.kind, sym.qualname) for sym in symbols][:2] == [
        ("class", "Broken"),
        ("method", "Broken.method"),
    ]


def test_python_without_grammar_falls_back_to_regex(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(treesitter, "extract", lambda text, language, path="": None)
    got = _names("class Broken:\n    def method(self):\n        return (\n", "python")
    assert got == [("class", "Broken"), ("function", "method")]


def test_markdown_headings_with_parents_skip_code_fences() -> None:
    symbols = extract_symbols(
        "# Заголовок\n\n## Раздел 1\n```\n# не заголовок\n```\n### Подраздел\n## Раздел 2 ##\n",
        "markdown",
    )
    got = [(sym.name, sym.parent, sym.depth, sym.line) for sym in symbols]
    assert got == [
        ("Заголовок", "", 0, 1),
        ("Раздел 1", "Заголовок", 1, 3),
        ("Подраздел", "Раздел 1", 2, 7),
        ("Раздел 2", "Заголовок", 1, 8),
    ]


@pytest.mark.parametrize(
    ("language", "src", "expected"),
    [
        (
            "typescript",
            """\
            export default class Store {
              private async load(id: string): Promise<void> {
                if (x) {
                }
              }
            }
            export interface Props {}
            export type Id = string;
            export const enum Color { Red }
            export const handler = async (req: Req) => {};
            function* gen() {}
            """,
            [
                ("class", "Store"),
                ("method", "load"),
                ("interface", "Props"),
                ("type", "Id"),
                ("enum", "Color"),
                ("function", "handler"),
                ("function", "gen"),
            ],
        ),
        (
            "go",
            """\
            type Server struct {
            type Handler interface {
            type ID string
            func (s *Server) Serve(addr string) error {
            func main() {
            """,
            [
                ("struct", "Server"),
                ("interface", "Handler"),
                ("type", "ID"),
                ("method", "Server.Serve"),
                ("function", "main"),
            ],
        ),
        (
            "rust",
            """\
            pub struct Index {
            enum Kind {
            pub(crate) trait Search {
            impl Search for Index {
                pub async fn find(&self) -> Vec<Hit> {
            macro_rules! hit {
            """,
            [
                ("struct", "Index"),
                ("enum", "Kind"),
                ("trait", "Search"),
                ("impl", "Index"),
                ("function", "find"),
                ("macro", "hit"),
            ],
        ),
        (
            "java",
            """\
            public final class Service {
                @Override
                public List<String> names(int n) {
                private static void helper() {
            public interface Repo {
            record Point(int x, int y) {
            """,
            [
                ("class", "Service"),
                ("method", "names"),
                ("method", "helper"),
                ("interface", "Repo"),
                ("record", "Point"),
            ],
        ),
        (
            "ruby",
            """\
            module Billing
              class Invoice
                def self.build
                def paid?
            """,
            [
                ("module", "Billing"),
                ("class", "Invoice"),
                ("method", "build"),
                ("method", "paid?"),
            ],
        ),
        (
            "php",
            """\
            final class Controller {
                public static function index() {
            function helper() {
            """,
            [("class", "Controller"), ("function", "index"), ("function", "helper")],
        ),
        (
            "c",
            """\
            struct node {
            static int parse(const char *s)
            {
                if (x) {
                return foo(1);
            int main(void) {
            int declared(void);
            """,
            [("struct", "node"), ("function", "parse"), ("function", "main")],
        ),
        (
            "kotlin",
            """\
            data class User(val id: Int)
            fun <T> List<T>.second(): T = this[1]
            """,
            [("class", "User"), ("function", "second")],
        ),
        (
            "shell",
            """\
            deploy() {
            function cleanup {
            """,
            [("function", "deploy"), ("function", "cleanup")],
        ),
    ],
)
def test_regex_fallback_languages(
    language: str,
    src: str,
    expected: list[tuple[str, str]],
) -> None:
    """Запасной путь без грамматики: построчные шаблоны (здесь — на обрывках кода)."""
    symbols = _regex_symbols(textwrap.dedent(src), _RULES[language])
    assert [(sym.kind, sym.qualname) for sym in symbols] == expected


def test_unknown_language_has_no_symbols() -> None:
    assert extract_symbols("def x(): pass", None) == []
    assert extract_symbols("{}", "json") == []


def test_symbols_limit_per_file() -> None:
    src = "\n".join(f"def f{index}(): pass" for index in range(MAX_SYMBOLS_PER_FILE + 50))
    assert len(extract_symbols(src, "python")) == MAX_SYMBOLS_PER_FILE


def test_markdown_commonmark_headings_and_fences() -> None:
    text = (
        " # С отступом\n"
        "# C#\n"
        "#хештег не заголовок\n"
        "~~~~\n"
        "```\n"
        "# внутри блока\n"
        "~~~~\n"
        "# После блока\n"
        "```python\n"
        "# комментарий\n"
        "``` не закрывает\n"
        "```\n"
        "## Конец ##\n"
    )
    names = [sym.name for sym in extract_symbols(text, "markdown")]
    assert names == ["С отступом", "C#", "После блока", "Конец"]


def test_python_methods_under_class_level_if_and_try() -> None:
    src = textwrap.dedent(
        """\
        class C:
            if True:
                def run(self):
                    pass
            try:
                def fast(self):
                    pass
            except ImportError:
                def slow(self):
                    pass
        """,
    )
    got = [(sym.kind, sym.qualname, sym.depth) for sym in extract_symbols(src, "python")]
    assert got == [
        ("class", "C", 0),
        ("method", "C.run", 1),
        ("method", "C.fast", 1),
        ("method", "C.slow", 1),
    ]


def test_python_try_blocks_in_source_order() -> None:
    src = textwrap.dedent(
        """\
        try:
            def a(): pass
        except ImportError:
            def b(): pass
        else:
            def c(): pass
        finally:
            def d(): pass
        """,
    )
    assert [sym.name for sym in extract_symbols(src, "python")] == ["a", "b", "c", "d"]


# ------------------------- использования и импорты ------------------------- #
def _facts(src: str, path: str = "pkg/mod.py") -> FileFacts:
    return extract(textwrap.dedent(src), "python", path)


def test_python_refs_scopes_kinds_and_locals() -> None:
    facts = _facts(
        """\
        LIMIT = 5


        class Agent(Base):
            def run(self, text, n=LIMIT):
                def inner(y):
                    return helper(y) + text
                local = Thing()
                local.go()
                self.index.refresh(text)
                print(len(text))
                return [x for x in items if x]


        def helper(value):
            return Agent().run(value)
        """,
    )
    got = [(ref.name, ref.kind, ref.line, ref.scope) for ref in facts.refs]
    assert got == [
        ("Base", "name", 4, "Agent"),
        ("LIMIT", "name", 5, "Agent.run"),
        ("helper", "call", 7, "Agent.run"),  # вложенная функция — в области внешней
        ("Thing", "call", 8, "Agent.run"),
        ("go", "call", 9, "Agent.run"),
        ("refresh", "call", 10, "Agent.run"),
        ("index", "attr", 10, "Agent.run"),
        ("items", "name", 12, "Agent.run"),
        ("run", "call", 16, "helper"),
        ("Agent", "call", 16, "helper"),
    ]
    # параметры, локальные переменные, переменные включений, self и builtins — не использования
    names = {ref.name for ref in facts.refs}
    assert not names & {"text", "y", "local", "x", "value", "self", "print", "len", "inner"}


def test_python_ref_columns_are_characters() -> None:
    facts = _facts("ё = 'ж'; obj.метод()\n")
    refs = [ref for ref in facts.refs if ref.name == "метод"]
    assert len(refs) == 1
    assert (refs[0].kind, refs[0].col) == ("call", 13)  # в символах: в байтах было бы 16


def test_python_imports_absolute_relative_and_aliases() -> None:
    facts = _facts(
        """\
        import os.path as osp
        import json, a.b.c
        from . import sibling
        from .core import Engine as E
        from ..up import thing
        from ...top import y
        from ....beyond import x
        """,
        path="pkg/sub/mod.py",
    )
    got = [(imp.module, imp.name, imp.alias) for imp in facts.imports]
    assert got == [
        ("os.path", "", "osp"),
        ("json", "", "json"),
        ("a.b.c", "", "a"),
        ("pkg.sub", "sibling", "sibling"),
        ("pkg.sub.core", "Engine", "E"),
        ("pkg.up", "thing", "thing"),
        ("top", "y", "y"),  # модуль в корне проекта
        ("....beyond", "x", "x"),  # выше корня проекта — не разрешается
    ]
    init = _facts("from . import a\n", path="pkg/__init__.py")
    assert [(imp.module, imp.name) for imp in init.imports] == [("pkg", "a")]


def test_python_docstring_and_name_column() -> None:
    symbols = extract_symbols(
        textwrap.dedent(
            '''\
            class Ёж:
                """

                Первая строка.
                Вторая.
                """

                async def  бег(self):
                    """Очень длинная документация """ + "x" * 0
            ''',
        ),
        "python",
    )
    by_name = {sym.name: sym for sym in symbols}
    assert (by_name["Ёж"].doc, by_name["Ёж"].col) == ("Первая строка.", 6)
    assert (by_name["бег"].doc, by_name["бег"].col) == ("", 15)  # не литерал — не докстринг


def test_python_syntax_error_keeps_calls_but_not_imports() -> None:
    facts = _facts("import os\n\ndef ok():\n    call_me()\n\ndef broken(:\n")
    assert [sym.name for sym in facts.symbols] == ["ok", "broken"]
    assert [(ref.name, ref.kind, ref.scope) for ref in facts.refs] == [("call_me", "call", "ok")]
    assert facts.imports == []  # импорты — только из ast


def test_regex_symbols_have_name_column() -> None:
    symbols = extract_symbols("export function runTurn() {}\n", "typescript")
    assert len(symbols) == 1
    assert symbols[0].col == len("export function ")


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("a/b/c.py", ["a", "b", "c"]),
        ("a/b/__init__.py", ["a", "b"]),
        ("stubs/x.pyi", ["stubs", "x"]),
        ("top.py", ["top"]),
        ("README.md", None),
    ],
)
def test_module_parts(path: str, expected: list[str] | None) -> None:
    assert module_parts(path) == expected


def test_python_signature_skips_comment_lines_before_body() -> None:
    src = "class Conversation:\n    # пояснение к полям\n    # ещё строка\n    items: list = []\n"
    symbols = extract_symbols(src, "python")
    assert len(symbols) == 1
    assert symbols[0].signature == "class Conversation:"


def test_python_scopes_of_comprehensions_imports_and_defaults() -> None:
    facts = _facts(
        """\
        values = [items for items in items]


        def run(factory=factory(), *, mode=MODE):
            from pkg.mod import helper as lazy
            lazy()
            helper()
            return [helper for helper in range(3)]


        @decorate(option)
        def other(option):
            return option
        """,
    )
    got = {(ref.name, ref.kind, ref.scope) for ref in facts.refs}
    assert ("items", "name", "") in got  # первый итерируемый — в объемлющей области
    assert ("factory", "call", "run") in got
    assert ("MODE", "name", "run") in got
    assert ("lazy", "call", "run") in got  # локальный импорт — использование
    assert ("helper", "call", "run") in got  # переменная включения не скрывает глобальное
    assert ("decorate", "call", "other") in got
    assert ("option", "name", "other") in got


def test_python_annotations_globals_and_match_rest() -> None:
    facts = _facts(
        """\
        def convert(Model: Model) -> Model:
            return Model


        def outer(helper):
            def inner():
                global helper
                helper()
            return inner


        def route(event):
            match event:
                case {"kind": kind, **settings}:
                    return settings
        """,
    )
    got = [(ref.name, ref.kind, ref.line) for ref in facts.refs]
    assert ("Model", "name", 1) in got  # аннотации — в объемлющей области
    assert ("helper", "call", 8) in got  # global снимает локальное имя внешней функции
    assert all(ref.name not in {"settings", "kind"} for ref in facts.refs)  # case {**rest}


def test_line_numbers_ignore_form_feed_and_unicode_separators() -> None:
    src = "# \f заголовок\n\ndef after():\n    pass\n\u2028\nclass Later:\n    pass\n"
    symbols = extract_symbols(src, "python")
    assert [(sym.name, sym.line, sym.signature) for sym in symbols] == [
        ("after", 3, "def after():"),
        ("Later", 6, "class Later:"),
    ]


def test_python_nonlocal_stays_local_and_same_line_refs_are_separate() -> None:
    facts = _facts(
        """\
        def outer(callback):
            def inner():
                nonlocal callback
                callback()
            return inner


        def both(a, b):
            a.save(); b.save()
        """,
    )
    assert all(ref.name != "callback" for ref in facts.refs)  # nonlocal — локальное внешней
    assert [(ref.name, ref.line, ref.col) for ref in facts.refs if ref.name == "save"] == [
        ("save", 9, 6),
        ("save", 9, 16),
    ]
