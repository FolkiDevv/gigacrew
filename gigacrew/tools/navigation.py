"""Инструменты навигации по связям в коде: использования, переход к определению,
иерархия вызовов.

Для Python — точно, через ty (LSP-сервер, :mod:`gigacrew.project.semantic`):
он разрешает импорты, псевдонимы и типы. Без ty (выключен ``GIGACREW_TY=0``,
не запустился, сбой) и для других языков — по индексу проекта: использования по
имени с ранжированием. Перед каждым запросом индекс обновляется, и ty узнаёт об
изменённых файлах, поэтому правки агента видны сразу.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace
from typing import TypeVar

from pydantic import BaseModel, Field

from gigacrew.project import semantic
from gigacrew.project.files import glob_match
from gigacrew.project.index import (
    RESOLVED_IMPORT,
    RESOLVED_NAME,
    RESOLVED_SAME_FILE,
    ProjectIndex,
    RefHit,
    SymbolHit,
)
from gigacrew.project.lsp import LspError
from gigacrew.project.semantic import CallItem, Location, SemanticUnavailable, TyServer
from gigacrew.project.symbols import split_lines
from gigacrew.safety.paths import is_secret_target
from gigacrew.security import resolve_in_root
from gigacrew.tools.base import Tool, ToolContext, ToolError, ToolResult
from gigacrew.tools.index import (
    MAX_SYMBOL_RESULTS,
    definition_line,
    open_index,
    refresh_note,
)
from gigacrew.tools.process import subprocess_env

T = TypeVar("T")

MAX_SOURCE_LINE = 160
MAX_TY_DEFINITIONS = 5  # определений, для которых спрашиваем ty
MAX_DEFINITION_RESULTS = 10
MAX_HIERARCHY_NODES = 80
MAX_HOVER_LINES = 8
MAX_DEPTH = 3

RESOLVED_TY = "ty"
_ORDER = {RESOLVED_TY: 0, RESOLVED_IMPORT: 1, RESOLVED_SAME_FILE: 2, RESOLVED_NAME: 3}

REF_KINDS = ("call", "attr", "name", "import")
_KIND_LABELS = {"call": "вызов", "attr": "атрибут", "name": "имя", "import": "импорт"}
_LABELS = {
    RESOLVED_TY: "точно (ty)",
    RESOLVED_IMPORT: "импортирует модуль с определением",
    RESOLVED_SAME_FILE: "файл определения",
    RESOLVED_NAME: "совпадение только по имени — может быть другое определение",
}
_LABELS_WITH_TY = _LABELS | {
    RESOLVED_NAME: "совпадение только по имени, ty не подтвердил — вероятно, другое определение"
}


# --------------------------------------------------------------------------- #
# Общее
# --------------------------------------------------------------------------- #
class _Ty:
    """Доступ к ty с откатом на индекс: ``run`` возвращает None, если ty недоступен.

    ``note`` — почему ответ получен по индексу (пусто — ty не понадобился или ответил).
    """

    def __init__(self, ctx: ToolContext):
        self._ctx = ctx
        self.note = ""

    def run(self, op: Callable[[TyServer], T]) -> T | None:
        if not self._ctx.semantic:  # выключен пользователем — без пояснений
            return None
        try:
            server = semantic.server_for(self._ctx.workspace, env=subprocess_env())
        except SemanticUnavailable as e:
            self.note = f"ty недоступен: {e}"
            return None
        try:
            return op(server)
        except LspError as e:
            self.note = semantic.report_failure(self._ctx.workspace, e)
            return None

    def fallback_line(self) -> list[str]:
        return [f"({self.note}; ответ по индексу — по именам)"] if self.note else []


class _SourceLines:
    """Строки исходников проекта для показа — читаются по файлу один раз."""

    def __init__(self, ctx: ToolContext):
        self._root = ctx.root
        self._cache: dict[str, list[str]] = {}

    def get(self, path: str, line: int) -> str:
        if path not in self._cache:
            try:
                text = (self._root / path).read_text(encoding="utf-8", errors="replace")
                self._cache[path] = split_lines(text)
            except OSError:
                self._cache[path] = []
        lines = self._cache[path]
        text = lines[line - 1].strip() if 0 < line <= len(lines) else ""
        if len(text) > MAX_SOURCE_LINE:
            text = text[: MAX_SOURCE_LINE - 1] + "…"
        return text


def _python_definitions(definitions: list[SymbolHit]) -> list[SymbolHit]:
    return [d for d in definitions if semantic.is_python(d.path)][:MAX_TY_DEFINITIONS]


_TYPESHED_RE = re.compile(r"/typeshed/[^/]+/(.+)$")


def external_path(path: str) -> str:
    """Короткое имя файла вне проекта: заглушки typeshed из ty, site-packages."""
    m = _TYPESHED_RE.search(path.replace("\\", "/"))
    if m:
        return f"typeshed/{m.group(1)}"
    marker = "site-packages/"
    if marker in path:
        return marker + path.split(marker, 1)[1]
    return path


def _symbol_label(index: ProjectIndex, loc: Location, fallback: str = "") -> str:
    """``Class.method — path:line`` по индексу (или имя из ty, если в индексе нет)."""
    if not loc.in_project:
        return f"{fallback or '?'} — вне проекта: {external_path(loc.path)}"
    symbol = index.symbol_at(loc.path, loc.line)
    name = symbol.qualname if symbol is not None and symbol.line == loc.line else fallback
    return f"{name or '?'} — {loc.path}:{loc.line}"


def _lines_note(lines: list[int]) -> str:
    unique = sorted(set(lines))
    if not unique:
        return ""
    word = "строка" if len(unique) == 1 else "строки"
    shown = ", ".join(map(str, unique[:10])) + (", …" if len(unique) > 10 else "")
    return f" ({word} {shown})"


# --------------------------------------------------------------------------- #
# find_references
# --------------------------------------------------------------------------- #
class FindReferencesParams(BaseModel):
    query: str = Field(
        description=(
            "Name of the definition (exact, case-insensitive): 'ProjectIndex', 'refresh'; "
            "a class method — 'ProjectIndex.refresh'"
        )
    )
    kind: str | None = Field(
        default=None,
        description=(
            "Kind of usage: call (calls), attr (attribute access), name (mentions of the "
            "name), import (import lines); all by default"
        ),
    )
    glob: str | None = Field(
        default=None, description="Limit to files matching a glob: '*.py' or 'src/**'"
    )
    max_results: int = Field(default=50, description="Maximum number of results")


def format_ref_hits(hits: list[RefHit], ctx: ToolContext, labels: dict[str, str]) -> list[str]:
    """Использования, сгруппированные по файлам: заголовок файла — уровень разрешения."""
    source = _SourceLines(ctx)
    out: list[str] = []
    current = None
    for h in hits:
        if (h.path, h.resolution) != current:
            current = (h.path, h.resolution)
            out.append(f"{h.path} — {labels.get(h.resolution, h.resolution)}:")
        where = f" в {h.scope}" if h.scope else ""
        kind = _KIND_LABELS.get(h.kind, h.kind)
        out.append(f"  {h.line} [{kind}]{where} — {source.get(h.path, h.line)}")
    return out


class FindReferencesTool(Tool):
    name = "find_references"
    read_only = True
    description = (
        "Finds usages of a definition: calls, attribute access, mentions, imports — with "
        "the function or class they occur in. Answers 'who calls it', 'where is it used', "
        "'what breaks if it changes'. For Python, usages are confirmed by ty (taking "
        "imports and types into account) and come first, marked 'точно (ty)'; the rest "
        "are matched by name: files importing the definition's module, the definition's "
        "own file, name-only matches. For other languages the index knows only calls "
        "(kind=call); other mentions — search_content."
    )
    Params = FindReferencesParams

    def describe(self, params: FindReferencesParams) -> str:
        return f"{params.query} ({params.kind})" if params.kind else params.query

    def run(self, params: FindReferencesParams, ctx: ToolContext) -> ToolResult:
        query = params.query.strip()
        if not query.strip("."):
            raise ToolError("Пустой запрос.")
        kind = params.kind.strip().lower() if params.kind else None
        if kind and kind not in REF_KINDS:
            raise ToolError(
                f"Неизвестный вид использования {params.kind!r}; допустимые: {', '.join(REF_KINDS)}"
            )
        limit = min(max(params.max_results, 1), MAX_SYMBOL_RESULTS)
        ty = _Ty(ctx)
        with open_index(ctx) as index:
            refreshed = index.refresh()
            result = index.find_refs(query, kind=kind, path_glob=params.glob, limit=10**9)
            hits = list(result.hits)
            confirmed = 0
            py_defs = _python_definitions(result.definitions)
            if py_defs:
                locations = ty.run(
                    lambda srv: [
                        loc
                        for d in py_defs
                        for loc in srv.references(d.path, d.symbol.line, d.symbol.col)
                    ]
                )
                if locations is not None:
                    hits = self._merge(index, hits, locations, py_defs, kind, params.glob)
                    confirmed = sum(h.resolution == RESOLVED_TY for h in hits)
        hits.sort(key=lambda h: (_ORDER[h.resolution], h.path, h.line, h.col))
        total = len(hits)
        shown = hits[:limit]

        lines = [f"определение: {definition_line(d)}" for d in result.definitions]
        if not result.definitions:
            lines.append(
                f"(определение «{query}» в индексе не найдено — ищу использования по имени)"
            )
        if not shown:
            lines.append("(использований не найдено)")
        else:
            lines.append(f"использования ({total}):")
            lines += format_ref_hits(shown, ctx, _LABELS_WITH_TY if confirmed else _LABELS)
            if total > len(shown):
                lines.append(f"… показано {len(shown)} из {total}; уточните запрос, kind или glob")
        if py_defs:
            lines += ty.fallback_line()
        note = f", подтверждено ty: {confirmed}" if confirmed else ""
        return ToolResult(
            content="\n".join(lines),
            summary=f"найдено использований: {total}{note}{refresh_note(refreshed)}",
        )

    @staticmethod
    def _merge(
        index: ProjectIndex,
        hits: list[RefHit],
        locations: list[Location],
        definitions: list[SymbolHit],
        kind: str | None,
        path_glob: str | None,
    ) -> list[RefHit]:
        """Подтверждённые ty — с пометкой ty; найденные только ty — добавляются.

        Строки импорта ty не возвращает — они остаются как есть.
        """
        # сопоставление по позиции: `a.refresh(); b.refresh()` — два разных использования
        found = {(loc.path, loc.line, loc.col) for loc in locations if loc.in_project}
        merged = [
            replace(h, resolution=RESOLVED_TY)
            if (h.path, h.line, h.col) in found and h.kind != "import"
            else h
            for h in hits
        ]
        known = {(h.path, h.line, h.col) for h in merged}
        def_positions = {(d.path, d.symbol.line, d.symbol.col) for d in definitions}
        names = {d.symbol.name for d in definitions}
        for loc in locations:
            key = (loc.path, loc.line, loc.col)
            if not loc.in_project or key in known or key in def_positions:
                continue
            if path_glob and not glob_match(loc.path, path_glob):
                continue
            info = index.ref_at(loc.path, loc.line, names, col=loc.col)
            if info is None:
                symbol = index.symbol_at(loc.path, loc.line)
                info = ("name", symbol.qualname if symbol is not None else "")
            if kind and info[0] != kind:
                continue
            merged.append(RefHit(loc.path, loc.line, loc.col, info[0], info[1], RESOLVED_TY))
            known.add(key)
        return merged


# --------------------------------------------------------------------------- #
# goto_definition
# --------------------------------------------------------------------------- #
class GotoDefinitionParams(BaseModel):
    path: str = Field(description="File relative to the project root")
    line: int = Field(description="Line number (1-based) where the name occurs")
    name: str = Field(
        description="The name on that line: 'ProjectIndex', 'refresh' or 'self.index.refresh'"
    )


def _name_column(text: str, name: str) -> int | None:
    """Столбец (в символах) последней части имени ``a.b.c`` в строке."""
    parts = [p for p in name.strip().split(".") if p]
    if not parts:
        return None
    last = parts[-1]
    if len(parts) > 1:
        dotted = r"\s*\.\s*".join(map(re.escape, parts))
        m = re.search(rf"(?<![\w.]){dotted}(?!\w)", text)
        if m:
            return m.end() - len(last)
    m = re.search(rf"(?<!\w){re.escape(last)}(?!\w)", text)
    return m.start() if m else None


def _hover_signature(hover: str) -> str:
    """Тип или сигнатура из подсказки ty — без документации (она после ``---``)."""
    lines = []
    for line in hover.splitlines():
        if re.fullmatch(r"\s*-{3,}\s*", line):
            break
        lines.append(line.rstrip())
    return "\n  ".join(line for line in lines[:MAX_HOVER_LINES] if line.strip())


class GotoDefinitionTool(Tool):
    name = "goto_definition"
    read_only = True
    description = (
        "Where a name on a specific line of a file leads: its definition (taking imports, "
        "aliases and the object's type into account) and, for Python, its type or "
        "signature. Useful when it is unclear which function `obj.method()` calls or where "
        "a name comes from. For Python the answer comes from ty; without it — from the "
        "index by name and the file's imports."
    )
    Params = GotoDefinitionParams

    def describe(self, params: GotoDefinitionParams) -> str:
        return f"{params.name} в {params.path}:{params.line}"

    def paths(self, params: GotoDefinitionParams) -> list[str]:
        return [params.path]

    def run(self, params: GotoDefinitionParams, ctx: ToolContext) -> ToolResult:
        target = resolve_in_root(ctx.root, params.path)
        if not target.is_file():
            raise ToolError(f"Файл не найден: {params.path}")
        if is_secret_target(ctx.root, target):
            raise ToolError(f"Файлы с секретами не разбираются: {params.path}")
        try:
            lines = split_lines(target.read_text(encoding="utf-8", errors="replace"))
        except OSError as e:
            raise ToolError(f"Не удалось прочитать {params.path}: {e}") from e
        if not 0 < params.line <= len(lines):
            raise ToolError(f"В файле {params.path} нет строки {params.line} (строк: {len(lines)})")
        text = lines[params.line - 1]
        col = _name_column(text, params.name)
        if col is None:
            raise ToolError(f"В строке {params.line} нет имени «{params.name}»: {text.strip()}")
        rel = target.relative_to(ctx.root).as_posix()
        name = params.name.strip().split(".")[-1]

        ty = _Ty(ctx)
        out: list[str] = []
        with open_index(ctx) as index:
            refreshed = index.refresh()
            answer = None
            if semantic.is_python(rel):
                answer = ty.run(
                    lambda srv: (
                        srv.definition(rel, params.line, col),
                        srv.hover(rel, params.line, col),
                    )
                )
            if answer is not None:
                locations, hover = answer
                source = _SourceLines(ctx)
                for loc in locations[:MAX_DEFINITION_RESULTS]:
                    out.append(self._describe(index, loc, source))
                if not locations:
                    out.append("(ty не нашёл определение)")
                signature = _hover_signature(hover)
                if signature:
                    out.append("тип: " + signature)
                summary = f"определений: {len(locations)} (ty)"
            else:
                hits = self._from_index(index, rel, name, params.line)
                out += [definition_line(h) for h in hits] or ["(определение не найдено)"]
                out += ty.fallback_line() or ["(по индексу — по имени и импортам файла)"]
                summary = f"определений: {len(hits)} (по индексу)"
        return ToolResult(content="\n".join(out), summary=summary + refresh_note(refreshed))

    @staticmethod
    def _describe(index: ProjectIndex, loc: Location, source: _SourceLines) -> str:
        if not loc.in_project:
            return f"{external_path(loc.path)}:{loc.line} (вне проекта)"
        symbol = index.symbol_at(loc.path, loc.line)
        if symbol is not None and symbol.line == loc.line:
            return definition_line(SymbolHit(loc.path, symbol))
        return f"{loc.path}:{loc.line} — {source.get(loc.path, loc.line)}"

    @staticmethod
    def _from_index(index: ProjectIndex, rel: str, name: str, line: int) -> list[SymbolHit]:
        """Импорт этого имени в файле → определение в целевом модуле; иначе — в самом
        файле; иначе — все определения с таким именем."""
        for entry in index.imports_of(rel):
            if entry.alias != name or entry.target is None:
                continue
            original = entry.name or name
            hits = [
                h for h in index.definitions(original) if h.path == entry.target
            ] or index.definitions(name)
            if hits:
                return [h for h in hits if h.path == entry.target] or hits
        hits = index.definitions(name)
        local = [h for h in hits if h.path == rel and h.symbol.line != line]
        return local or hits[:MAX_DEFINITION_RESULTS]


# --------------------------------------------------------------------------- #
# call_hierarchy
# --------------------------------------------------------------------------- #
class CallHierarchyParams(BaseModel):
    query: str = Field(
        description="Function or method: 'helper', 'ProjectIndex.refresh' (exact name)"
    )
    direction: str = Field(
        default="incoming",
        description="incoming — who calls it (default); outgoing — what it calls",
    )
    depth: int = Field(default=1, description=f"Depth of the tree: 1–{MAX_DEPTH}")


class CallHierarchyTool(Tool):
    name = "call_hierarchy"
    read_only = True
    description = (
        "Call tree of a function or method: who calls it (incoming) or what it calls "
        f"(outgoing), up to {MAX_DEPTH} levels deep, with the call sites. For Python the "
        "answer comes from ty (taking imports, aliases and types into account); without "
        "it — from the index by name."
    )
    Params = CallHierarchyParams

    def describe(self, params: CallHierarchyParams) -> str:
        return f"{params.query} ({params.direction})"

    def run(self, params: CallHierarchyParams, ctx: ToolContext) -> ToolResult:
        query = params.query.strip()
        if not query.strip("."):
            raise ToolError("Пустой запрос.")
        direction = params.direction.strip().lower()
        if direction not in ("incoming", "outgoing"):
            raise ToolError("direction: incoming (кто вызывает) или outgoing (что вызывает)")
        depth = min(max(params.depth, 1), MAX_DEPTH)
        incoming = direction == "incoming"
        ty = _Ty(ctx)
        with open_index(ctx) as index:
            refreshed = index.refresh()
            definitions = index.definitions(query)
            if not definitions:
                raise ToolError(
                    f"Определение «{query}» не найдено в индексе. Найдите имя через find_symbol."
                )
            py_defs = _python_definitions(definitions)
            tree = None
            if py_defs:
                tree = ty.run(lambda srv: self._ty_tree(srv, index, py_defs, incoming, depth))
            source = "ty"
            if tree is None:
                source = "по индексу"
                tree = self._index_tree(index, definitions[:MAX_TY_DEFINITIONS], incoming, depth)
            else:  # определения на других языках ty не видит — их дерево по индексу
                others = [d for d in definitions if not semantic.is_python(d.path)]
                if others:
                    source = "ty + индекс"
                    tree += self._index_tree(index, others[:MAX_TY_DEFINITIONS], incoming, depth)
        title = "кто вызывает" if incoming else "что вызывает"
        lines = [f"{title} (глубина {depth}, {source}):", *tree]
        if py_defs:
            lines += ty.fallback_line()
        calls = sum(1 for line in tree if line.lstrip().startswith(("←", "→")))
        return ToolResult(
            content="\n".join(lines),
            summary=f"{title}: {calls} ({source}){refresh_note(refreshed)}",
        )

    # ------------------------------ через ty ------------------------------ #
    def _ty_tree(
        self,
        server: TyServer,
        index: ProjectIndex,
        definitions: list[SymbolHit],
        incoming: bool,
        depth: int,
    ) -> list[str]:
        lines: list[str] = []
        budget = [MAX_HIERARCHY_NODES]
        for d in definitions:
            roots = server.prepare_call_hierarchy(d.path, d.symbol.line, d.symbol.col)
            lines.append(f"{d.symbol.qualname} — {d.path}:{d.symbol.line}")
            if not roots:
                lines.append("  (ty не распознал здесь функцию)")
                continue
            seen = {(d.path, d.symbol.line)}
            before = len(lines)
            for root in roots:
                self._ty_walk(server, index, root, incoming, depth, 1, seen, lines, budget)
            if len(lines) == before:
                lines.append("  (вызовов не найдено)")
        if budget[0] <= 0:
            lines.append(f"… дерево обрезано ({MAX_HIERARCHY_NODES} узлов); уменьшите depth")
        return lines

    def _ty_walk(
        self,
        server: TyServer,
        index: ProjectIndex,
        item: CallItem,
        incoming: bool,
        depth: int,
        level: int,
        seen: set[tuple[str, int]],
        lines: list[str],
        budget: list[int],
    ) -> None:
        children = server.incoming_calls(item) if incoming else server.outgoing_calls(item)
        arrow = "←" if incoming else "→"
        # перегрузки (несколько определений вне проекта) — одной строкой
        grouped: dict[str, tuple[CallItem, list[int]]] = {}
        for child in children:
            label = _symbol_label(index, child.location, child.name)
            _, sites = grouped.setdefault(label, (child, []))
            sites += [c.line for c in child.calls]
        for label, (child, sites) in grouped.items():
            if budget[0] <= 0:
                return
            budget[0] -= 1
            lines.append(f"{'  ' * level}{arrow} {label}{_lines_note(sites)}")
            key = (child.location.path, child.location.line)
            if level < depth and child.location.in_project and key not in seen:
                seen.add(key)
                self._ty_walk(server, index, child, incoming, depth, level + 1, seen, lines, budget)

    # ----------------------------- по индексу ----------------------------- #
    def _index_tree(
        self, index: ProjectIndex, definitions: list[SymbolHit], incoming: bool, depth: int
    ) -> list[str]:
        """Дерево по индексу; общий бюджет узлов на все корни и уровни."""
        lines: list[str] = []
        budget = [MAX_HIERARCHY_NODES]
        seen = {d.symbol.qualname for d in definitions}
        for d in definitions:
            lines.append(f"{d.symbol.qualname} — {d.path}:{d.symbol.line}")
            before = len(lines)
            if incoming:
                self._index_callers(index, d.symbol.qualname, depth, 1, seen, lines, budget)
            else:
                for name, call_lines in self._group(index.calls_in(d.path, d.symbol.qualname)):
                    if budget[0] <= 0:
                        break
                    budget[0] -= 1
                    lines.append(f"  → {name}{_lines_note(call_lines)}")
                if depth > 1:
                    lines.append("  (глубже 1 уровня исходящие вызовы — только через ty)")
            if len(lines) == before:
                lines.append("  (вызовов не найдено)")
        if budget[0] <= 0:
            lines.append(f"… дерево обрезано ({MAX_HIERARCHY_NODES} узлов); уменьшите depth")
        return lines

    def _index_callers(
        self,
        index: ProjectIndex,
        qualname: str,
        depth: int,
        level: int,
        seen: set[str],
        lines: list[str],
        budget: list[int],
    ) -> None:
        if budget[0] <= 0:
            return
        result = index.find_refs(qualname, kind="call", limit=budget[0])
        groups: dict[tuple[str, str, str], list[int]] = {}
        for h in result.hits:
            groups.setdefault((h.path, h.scope, h.resolution), []).append(h.line)
        for (path, scope, resolution), call_lines in groups.items():
            if budget[0] <= 0:
                return
            budget[0] -= 1
            guess = " (по имени)" if resolution == RESOLVED_NAME else ""
            caller = scope or "<уровень модуля>"
            lines.append(f"{'  ' * level}← {caller} — {path}{_lines_note(call_lines)}{guess}")
            if level < depth and scope and scope not in seen:
                seen.add(scope)
                self._index_callers(index, scope, depth, level + 1, seen, lines, budget)

    @staticmethod
    def _group(calls: list[tuple[str, int]]) -> list[tuple[str, list[int]]]:
        grouped: dict[str, list[int]] = {}
        for name, line in calls:
            grouped.setdefault(name, []).append(line)
        return list(grouped.items())
