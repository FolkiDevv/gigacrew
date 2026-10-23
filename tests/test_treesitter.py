"""Тесты разбора tree-sitter: определения с диапазоном и вложенностью, вызовы."""

from __future__ import annotations

import pytest

from gigacrew.project import treesitter
from gigacrew.project.symbols import extract

# Ожидаемые определения: (вид, полное имя, первая строка, последняя строка).
type _Symbols = list[tuple[str, str, int, int]]
# Ожидаемые вызовы: (имя вызываемого, область, где вызван).
type _Calls = list[tuple[str, str]]

CASES: tuple[tuple[str, str, str, _Symbols, _Calls], ...] = (
    (
        "javascript",
        "a.js",
        (
            "function load(id) { return fetchItem(id); }\n"
            "class Store extends Base {\n"
            "  save(item) { this.persist(item); }\n"
            "  static create() { return new Store(); }\n"
            "}\n"
            "const handler = async (req) => respond(req);\n"
            "const limit = 5;\n"
        ),
        [
            ("function", "load", 1, 1),
            ("class", "Store", 2, 5),
            ("method", "Store.save", 3, 3),
            ("method", "Store.create", 4, 4),
            ("function", "handler", 6, 6),
        ],
        [
            ("fetchItem", "load"),
            ("persist", "Store.save"),
            ("Store", "Store.create"),
            ("respond", "handler"),
        ],
    ),
    (
        "typescript",
        "a.ts",
        (
            "export interface Props { render(): void }\n"
            "export type Id = string;\n"
            "export const enum Color { Red }\n"
            "export abstract class View {\n"
            "  protected draw(): void { this.canvas.paint(); }\n"
            "}\n"
            "namespace Utils { export function clamp(x: number) { return Math.min(x, 1); } }\n"
        ),
        [
            ("interface", "Props", 1, 1),
            ("method", "Props.render", 1, 1),
            ("type", "Id", 2, 2),
            ("enum", "Color", 3, 3),
            ("class", "View", 4, 6),
            ("method", "View.draw", 5, 5),
            ("namespace", "Utils", 7, 7),
            ("function", "Utils.clamp", 7, 7),
        ],
        [("paint", "View.draw"), ("min", "Utils.clamp")],
    ),
    (
        "typescript",
        "App.tsx",
        "export function App() {\n  return <button onClick={() => submit()}>ok</button>;\n}\n",
        [("function", "App", 1, 3)],
        [("submit", "App")],
    ),
    (
        "go",
        "a.go",
        (
            "package srv\n\n"
            "type Server struct{ addr string }\n"
            "type Handler interface{ Serve() error }\n"
            "type ID string\n\n"
            "func (s *Server) Serve() error {\n\treturn listen(s.addr)\n}\n\n"
            "func main() {\n\tsrv := NewServer()\n\tsrv.Serve()\n}\n"
        ),
        [
            ("struct", "Server", 3, 3),
            ("interface", "Handler", 4, 4),
            ("method", "Handler.Serve", 4, 4),
            ("type", "ID", 5, 5),
            ("method", "Server.Serve", 7, 9),
            ("function", "main", 11, 14),
        ],
        [("listen", "Server.Serve"), ("NewServer", "main"), ("Serve", "main")],
    ),
    (
        "rust",
        "a.rs",
        (
            "pub struct Index { items: Vec<u8> }\n"
            "trait Search { fn find(&self) -> bool; }\n"
            "impl Search for Index {\n"
            "    fn find(&self) -> bool { self.items.contains(&1) }\n"
            "}\n"
            "mod util { pub fn helper() { Index::build(); } }\n"
            "macro_rules! hit { () => {} }\n"
        ),
        [
            ("struct", "Index", 1, 1),
            ("trait", "Search", 2, 2),
            ("method", "Search.find", 2, 2),
            ("impl", "Index", 3, 5),
            ("method", "Index.find", 4, 4),
            ("module", "util", 6, 6),
            ("function", "util.helper", 6, 6),
            ("macro", "hit", 7, 7),
        ],
        [("contains", "Index.find"), ("build", "util.helper")],
    ),
    (
        "java",
        "Service.java",
        (
            "public final class Service {\n"
            "    public Service() {}\n"
            "    @Override\n"
            "    public List<String> names(int n) { return repo.load(n); }\n"
            "}\n"
            "interface Repo { List<String> load(int n); }\n"
            "record Point(int x, int y) {}\n"
        ),
        [
            ("class", "Service", 1, 5),
            ("method", "Service.Service", 2, 2),
            ("method", "Service.names", 3, 4),
            ("interface", "Repo", 6, 6),
            ("method", "Repo.load", 6, 6),
            ("record", "Point", 7, 7),
        ],
        [("load", "Service.names")],
    ),
    (
        "kotlin",
        "a.kt",
        (
            "class User(val id: Int) {\n"
            "    fun greet(): String {\n"
            "        return format(id)\n"
            "    }\n"
            "}\n\n"
            "interface Repo {\n    fun find()\n}\n\n"
            "object Registry\n\n"
            "fun top(a: Int): Int = helper(a)\n"
        ),
        [
            ("class", "User", 1, 5),
            ("method", "User.greet", 2, 4),
            ("interface", "Repo", 7, 9),
            ("method", "Repo.find", 8, 8),
            ("object", "Registry", 11, 11),
            ("function", "top", 13, 13),
        ],
        [("format", "User.greet"), ("helper", "top")],
    ),
    (
        "csharp",
        "a.cs",
        (
            "namespace App {\n"
            "  public class Service {\n"
            '    public void Run() { Logger.Write("x"); }\n'
            "  }\n"
            "  interface IRepo {}\n"
            "}\n"
        ),
        [
            ("namespace", "App", 1, 6),
            ("class", "App.Service", 2, 4),
            ("method", "App.Service.Run", 3, 3),
            ("interface", "App.IRepo", 5, 5),
        ],
        [("Write", "App.Service.Run")],
    ),
    (
        "ruby",
        "a.rb",
        (
            "module Billing\n"
            "  class Invoice < Base\n"
            "    def self.build\n      new\n    end\n"
            "    def paid?\n      payments.any?\n    end\n"
            "  end\n"
            "end\n"
        ),
        [
            ("module", "Billing", 1, 10),
            ("class", "Billing.Invoice", 2, 9),
            ("method", "Billing.Invoice.build", 3, 5),
            ("method", "Billing.Invoice.paid?", 6, 8),
        ],
        [("any?", "Billing.Invoice.paid?")],
    ),
    (
        "php",
        "a.php",
        (
            "<?php\n"
            "final class Controller {\n"
            "    public static function index() { return View::render(); }\n"
            "}\n"
            "function helper() { return strtolower('X'); }\n"
        ),
        [
            ("class", "Controller", 2, 4),
            ("method", "Controller.index", 3, 3),
            ("function", "helper", 5, 5),
        ],
        [("render", "Controller.index"), ("strtolower", "helper")],
    ),
    (
        "c",
        "a.c",
        (
            "struct node { int value; };\n"
            "struct node *head;\n"
            "static int parse(const char *s)\n{\n    return atoi(s);\n}\n"
            "int declared(void);\n"
        ),
        [("struct", "node", 1, 1), ("function", "parse", 3, 6)],
        [("atoi", "parse")],
    ),
    (
        "cpp",
        "a.cpp",
        (
            "namespace net {\n"
            "class Socket {\n  public:\n    int open() { return connect(); }\n};\n"
            "}\n"
            "void net::Socket::close() { shutdown(fd); }\n"
        ),
        [
            ("namespace", "net", 1, 6),
            ("class", "net.Socket", 2, 5),
            ("method", "net.Socket.open", 4, 4),
            ("method", "net::Socket.close", 7, 7),
        ],
        [("connect", "net.Socket.open"), ("shutdown", "net::Socket.close")],
    ),
    (
        "shell",
        "a.sh",
        "deploy() {\n  build_all\n}\nfunction cleanup {\n  rm -rf out\n  deploy\n}\n",
        [("function", "deploy", 1, 3), ("function", "cleanup", 4, 7)],
        [("build_all", "deploy"), ("rm", "cleanup"), ("deploy", "cleanup")],
    ),
    (
        "scala",
        "a.scala",
        "object Main {\n  def run(a: Int): Unit = process(a)\n}\ntrait Shape\n",
        [("object", "Main", 1, 3), ("method", "Main.run", 2, 2), ("trait", "Shape", 4, 4)],
        [("process", "Main.run")],
    ),
    (
        "swift",
        "a.swift",
        (
            "class Player {\n  func play() { engine.start() }\n}\n"
            "struct Point {}\nprotocol Drawable {}\n"
            "extension Player { func stop() { halt() } }\n"
        ),
        [
            ("class", "Player", 1, 3),
            ("method", "Player.play", 2, 2),
            ("struct", "Point", 4, 4),
            ("protocol", "Drawable", 5, 5),
            ("extension", "Player", 6, 6),
            ("method", "Player.stop", 6, 6),
        ],
        [("start", "Player.play"), ("halt", "Player.stop")],
    ),
)


@pytest.mark.parametrize(
    ("language", "path", "src", "symbols", "calls"),
    CASES,
    ids=[f"{case[0]}:{case[1]}" for case in CASES],
)
def test_languages(language: str, path: str, src: str, symbols: _Symbols, calls: _Calls) -> None:
    facts = extract(src, language, path)
    found = [
        (symbol.kind, symbol.qualname, symbol.line, symbol.end_line) for symbol in facts.symbols
    ]
    assert found == symbols
    assert [(ref.name, ref.scope) for ref in facts.refs] == calls
    assert all(ref.kind == "call" for ref in facts.refs)
    assert facts.imports == []


def test_depth_signature_and_column() -> None:
    facts = extract("class Ёлка {\n  fn() { }\n  рост(x) {\n    return x;\n  }\n}\n", "javascript")
    by_name = {symbol.name: symbol for symbol in facts.symbols}
    assert (by_name["Ёлка"].depth, by_name["рост"].depth) == (0, 1)
    assert by_name["рост"].signature == "рост(x)"
    assert by_name["Ёлка"].col == 6
    assert by_name["рост"].col == 2


def test_nested_functions_are_not_definitions_but_calls_count() -> None:
    facts = extract(
        "function outer() {\n  function inner() { deep(); }\n  inner();\n}\n",
        "javascript",
    )
    assert [symbol.qualname for symbol in facts.symbols] == ["outer"]
    calls = [(ref.name, ref.scope) for ref in facts.refs]
    assert calls == [("deep", "outer"), ("inner", "outer")]


def test_unloadable_grammar_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(treesitter._parsers, "go", None)
    assert treesitter.extract("package x\nfunc main() {}\n", "go") is None
    # индекс тогда откатывается на построчные шаблоны
    fallback = extract("package x\nfunc main() {}\n", "go")
    assert [symbol.name for symbol in fallback.symbols] == ["main"]
    assert treesitter.extract("x", "cobol") is None
    assert treesitter.supports("rust")
    assert not treesitter.supports("markdown")


def test_csharp_namespaces_keep_full_name_and_file_scope() -> None:
    facts = extract(
        "namespace Company.App;\n\nclass Service { void Run() { Log.Write(); } }\n",
        "csharp",
    )
    assert [(symbol.qualname, symbol.depth) for symbol in facts.symbols] == [
        ("Company.App", 0),
        ("Company.App.Service", 1),
        ("Company.App.Service.Run", 2),
    ]
    calls = [(ref.name, ref.scope) for ref in facts.refs]
    assert calls == [("Write", "Company.App.Service.Run")]
    block = extract("namespace A.B { class S {} }\nclass Outside {}\n", "csharp")
    assert [symbol.qualname for symbol in block.symbols] == ["A.B", "A.B.S", "Outside"]


def test_js_private_members_are_called() -> None:
    facts = extract("class A {\n  #persist() {}\n  save() { this.#persist(); }\n}\n", "javascript")
    assert [symbol.qualname for symbol in facts.symbols] == ["A", "A.#persist", "A.save"]
    assert [(ref.name, ref.scope) for ref in facts.refs] == [("#persist", "A.save")]


def test_parenthesized_callees_quoted_methods_and_arrow_signatures() -> None:
    facts = extract(
        'class A {\n  "save"() { (load)(); (obj.persist)(); }\n}\n'
        "const handler = (req) => { respond(req); };\n",
        "javascript",
    )
    assert [(symbol.qualname, symbol.signature) for symbol in facts.symbols] == [
        ("A", "class A"),
        ("A.save", '"save"()'),
        ("handler", "handler = (req) =>"),  # без тела функции
    ]
    assert [ref.name for ref in facts.refs] == ["load", "persist", "respond"]
