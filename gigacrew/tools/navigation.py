"""Инструменты навигации по связям в коде.

Использования, переход к определению, иерархия вызовов. Для Python — точно, через ty
(LSP-сервер, :mod:`gigacrew.project.semantic`): он разрешает импорты, псевдонимы и
типы. Без ty (выключен ``GIGACREW_TY=0``, не запустился, сбой) и для других языков —
по индексу проекта: использования по имени с ранжированием. Перед каждым запросом
индекс обновляется, и ty узнаёт об изменённых файлах, поэтому правки агента видны сразу.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import override

from pydantic import BaseModel, Field

from gigacrew.project import semantic
from gigacrew.project.files import glob_match
from gigacrew.project.index import (
    RESOLVED_IMPORT,
    RESOLVED_NAME,
    RESOLVED_SAME_FILE,
    ProjectIndex,
    RefHit,
    RefreshStats,
    RefsResult,
    SymbolHit,
)
from gigacrew.project.lsp import LspError
from gigacrew.project.secret_paths import is_secret_target
from gigacrew.project.symbols import split_lines
from gigacrew.security import resolve_in_root, subprocess_env
from gigacrew.tools.base import Tool, ToolContext, ToolError, ToolResult
from gigacrew.tools.index import (
    MAX_SYMBOL_RESULTS,
    definition_line,
    open_index,
    refresh_note,
)

MAX_SOURCE_LINE = 160
MAX_TY_DEFINITIONS = 5  # определений, для которых спрашиваем ty
MAX_DEFINITION_RESULTS = 10
MAX_HIERARCHY_NODES = 80
MAX_HOVER_LINES = 8
MAX_DEPTH = 3

RESOLVED_TY = "ty"
_ORDER = MappingProxyType({
    RESOLVED_TY: 0,
    RESOLVED_IMPORT: 1,
    RESOLVED_SAME_FILE: 2,
    RESOLVED_NAME: 3,
})

REF_KINDS = ("call", "attr", "name", "import")
_KIND_LABELS = MappingProxyType({
    "call": "вызов",
    "attr": "атрибут",
    "name": "имя",
    "import": "импорт",
})
_LABELS = MappingProxyType({
    RESOLVED_TY: "точно (ty)",
    RESOLVED_IMPORT: "импортирует модуль с определением",
    RESOLVED_SAME_FILE: "файл определения",
    RESOLVED_NAME: "совпадение только по имени — может быть другое определение",
})
_LABELS_WITH_TY = MappingProxyType({
    **_LABELS,
    RESOLVED_NAME: "совпадение только по имени, ty не подтвердил — вероятно, другое определение",
})

_EMPTY_QUERY = "Пустой запрос."
_DEFAULT_MAX_RESULTS = 50
_MAX_NOTED_LINES = 10  # номеров строк вызовов в одной строке дерева
_NO_CALLS = "  (вызовов не найдено)"
_INDENT = "  "


# --------------------------------------------------------------------------- #
# Общее
# --------------------------------------------------------------------------- #
class _Ty:
    """Доступ к ty с откатом на индекс: ``run`` возвращает None, если ty недоступен.

    ``note`` — почему ответ получен по индексу (пусто — ty не понадобился или ответил).
    """

    def __init__(self, ctx: ToolContext) -> None:
        self._ctx = ctx
        self.note = ""

    def run[ResultT](self, op: Callable[[semantic.TyServer], ResultT]) -> ResultT | None:
        """Выполняет ``op`` на сервере ty; None — ty выключен, недоступен или сбоит."""
        if not self._ctx.semantic:  # выключен пользователем — без пояснений
            return None
        try:
            server = semantic.server_for(self._ctx.workspace, env=subprocess_env())
        except semantic.SemanticUnavailableError as exc:
            self.note = f"ty недоступен: {exc}"
            return None
        try:
            return op(server)
        except LspError as exc:
            self.note = semantic.report_failure(self._ctx.workspace, exc)
            return None

    def fallback_line(self) -> list[str]:
        """Строка-пояснение, почему ответ получен по индексу (пусто — пояснять нечего)."""
        return [f"({self.note}; ответ по индексу — по именам)"] if self.note else []


class _SourceLines:
    """Строки исходников проекта для показа — читаются по файлу один раз."""

    def __init__(self, ctx: ToolContext) -> None:
        self._root = ctx.root
        self._cache: dict[str, list[str]] = {}

    def get(self, path: str, line: int) -> str:
        """Строка ``line`` файла ``path`` без отступов, обрезанная до ``MAX_SOURCE_LINE``."""
        if path not in self._cache:
            try:
                text = (self._root / path).read_text(encoding="utf-8", errors="replace")
                self._cache[path] = split_lines(text)
            except OSError:
                self._cache[path] = []
        lines = self._cache[path]
        text = lines[line - 1].strip() if 0 < line <= len(lines) else ""
        if len(text) > MAX_SOURCE_LINE:
            cut = text[: MAX_SOURCE_LINE - 1]
            text = f"{cut}…"
        return text


def _python_definitions(definitions: list[SymbolHit]) -> list[SymbolHit]:
    python = [definition for definition in definitions if semantic.is_python(definition.path)]
    return python[:MAX_TY_DEFINITIONS]


_TYPESHED_RE = re.compile(r"/typeshed/[^/]+/(.+)$")


def external_path(path: str) -> str:
    """Короткое имя файла вне проекта: заглушки typeshed из ty, site-packages."""
    typeshed = _TYPESHED_RE.search(path.replace("\\", "/"))
    if typeshed:
        return f"typeshed/{typeshed.group(1)}"
    marker = "site-packages/"
    if marker in path:
        return marker + path.split(marker, 1)[1]
    return path


# --------------------------------------------------------------------------- #
# find_references
# --------------------------------------------------------------------------- #
class FindReferencesParams(BaseModel):
    query: str = Field(
        description=(
            "Name of the definition (exact, case-insensitive): 'ProjectIndex', 'refresh'; "
            "a class method — 'ProjectIndex.refresh'"
        ),
    )
    kind: str | None = Field(
        default=None,
        description=(
            "Kind of usage: call (calls), attr (attribute access), name (mentions of the "
            "name), import (import lines); all by default"
        ),
    )
    glob: str | None = Field(
        default=None,
        description="Limit to files matching a glob: '*.py' or 'src/**'",
    )
    max_results: int = Field(
        default=_DEFAULT_MAX_RESULTS,
        description="Maximum number of results",
    )


def format_ref_hits(
    hits: list[RefHit],
    ctx: ToolContext,
    labels: Mapping[str, str],
) -> list[str]:
    """Использования, сгруппированные по файлам: заголовок файла — уровень разрешения."""
    source = _SourceLines(ctx)
    out: list[str] = []
    current: tuple[str, str] | None = None
    for hit in hits:
        if (hit.path, hit.resolution) != current:
            current = (hit.path, hit.resolution)
            out.append(f"{hit.path} — {labels.get(hit.resolution, hit.resolution)}:")
        where = f" в {hit.scope}" if hit.scope else ""
        kind = _KIND_LABELS.get(hit.kind, hit.kind)
        out.append(f"  {hit.line} [{kind}]{where} — {source.get(hit.path, hit.line)}")
    return out


def _ty_only_hit(
    index: ProjectIndex,
    loc: semantic.Location,
    names: set[str],
    kind: str | None,
) -> RefHit | None:
    """Использование, которое нашёл только ty: вид и область — по индексу.

    None — вид использования не совпал с запрошенным ``kind``.
    """
    ref = index.ref_at(loc.path, loc.line, names, col=loc.col)
    if ref is None:
        symbol = index.symbol_at(loc.path, loc.line)
        ref = ("name", "" if symbol is None else symbol.qualname)
    ref_kind, scope = ref
    if kind and ref_kind != kind:
        return None
    return RefHit(loc.path, loc.line, loc.col, ref_kind, scope, RESOLVED_TY)


class FindReferencesTool(Tool[FindReferencesParams]):
    """Инструмент find_references: использования определения, подтверждённые ty и по имени."""

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

    @override
    def describe(self, params: FindReferencesParams) -> str:
        """Запрос и вид использования, если он задан."""
        return f"{params.query} ({params.kind})" if params.kind else params.query

    @override
    def run(self, params: FindReferencesParams, ctx: ToolContext) -> ToolResult:
        """Использования по файлам: сначала подтверждённые ty, затем найденные по имени."""
        query = params.query.strip()
        if not query.strip("."):
            raise ToolError(_EMPTY_QUERY)
        kind = self._kind(params.kind)
        ty = _Ty(ctx)
        with open_index(ctx) as index:
            refreshed = index.refresh()
            found = index.find_refs(query, kind=kind, path_glob=params.glob, limit=10**9)
            hits, confirmed = self._confirm_by_ty(ty, index, found, kind, params.glob)
        hits.sort(key=lambda hit: (_ORDER[hit.resolution], hit.path, hit.line, hit.col))
        limit = min(max(params.max_results, 1), MAX_SYMBOL_RESULTS)
        lines = [
            *self._definition_lines(query, found.definitions),
            *self._usage_lines(hits[:limit], len(hits), ctx, confirmed),
            # пояснение бывает, только если ty спрашивали — есть определения на Python
            *ty.fallback_line(),
        ]
        return ToolResult(
            content="\n".join(lines),
            summary=self._summary(len(hits), confirmed, refreshed),
        )

    @staticmethod
    def _merge(
        index: ProjectIndex,
        hits: list[RefHit],
        locations: list[semantic.Location],
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
            replace(hit, resolution=RESOLVED_TY)
            if (hit.path, hit.line, hit.col) in found and hit.kind != "import"
            else hit
            for hit in hits
        ]
        known = {(hit.path, hit.line, hit.col) for hit in merged}
        def_positions = {
            (definition.path, definition.symbol.line, definition.symbol.col)
            for definition in definitions
        }
        names = {definition.symbol.name for definition in definitions}
        for loc in locations:
            key = (loc.path, loc.line, loc.col)
            if not loc.in_project or key in known or key in def_positions:
                continue
            if path_glob and not glob_match(loc.path, path_glob):
                continue
            added = _ty_only_hit(index, loc, names, kind)
            if added is not None:
                merged.append(added)
                known.add(key)
        return merged

    def _kind(self, raw: str | None) -> str | None:
        """Вид использования из параметров (None — любой); ToolError — вид неизвестен."""
        kind = raw.strip().lower() if raw else None
        if kind and kind not in REF_KINDS:
            allowed = ", ".join(REF_KINDS)
            raise ToolError(f"Неизвестный вид использования {raw!r}; допустимые: {allowed}")
        return kind

    def _confirm_by_ty(
        self,
        ty: _Ty,
        index: ProjectIndex,
        found: RefsResult,
        kind: str | None,
        path_glob: str | None,
    ) -> tuple[list[RefHit], int]:
        """Использования по индексу, уточнённые ty, и сколько из них подтвердил ty."""
        hits = list(found.hits)
        py_defs = _python_definitions(found.definitions)
        if not py_defs:
            return hits, 0
        locations = ty.run(
            lambda srv: [
                loc
                for definition in py_defs
                for loc in srv.references(
                    definition.path,
                    definition.symbol.line,
                    definition.symbol.col,
                )
            ],
        )
        if locations is None:
            return hits, 0
        merged = self._merge(index, hits, locations, py_defs, kind, path_glob)
        return merged, sum(hit.resolution == RESOLVED_TY for hit in merged)

    def _definition_lines(self, query: str, definitions: list[SymbolHit]) -> list[str]:
        """Найденные определения — или пояснение, что поиск идёт только по имени."""
        if not definitions:
            return [f"(определение «{query}» в индексе не найдено — ищу использования по имени)"]
        return [f"определение: {definition_line(definition)}" for definition in definitions]

    def _usage_lines(
        self,
        shown: list[RefHit],
        total: int,
        ctx: ToolContext,
        confirmed: int,
    ) -> list[str]:
        """Показанные использования по файлам; если показаны не все — как сузить запрос."""
        if not shown:
            return ["(использований не найдено)"]
        labels = _LABELS_WITH_TY if confirmed else _LABELS
        lines = [f"использования ({total}):", *format_ref_hits(shown, ctx, labels)]
        if total > len(shown):
            lines.append(f"… показано {len(shown)} из {total}; уточните запрос, kind или glob")
        return lines

    def _summary(self, total: int, confirmed: int, refreshed: RefreshStats) -> str:
        """Краткий итог: сколько найдено и сколько подтвердил ty."""
        note = f", подтверждено ty: {confirmed}" if confirmed else ""
        return f"найдено использований: {total}{note}{refresh_note(refreshed)}"


# --------------------------------------------------------------------------- #
# goto_definition
# --------------------------------------------------------------------------- #
class GotoDefinitionParams(BaseModel):
    path: str = Field(description="File relative to the project root")
    line: int = Field(description="Line number (1-based) where the name occurs")
    name: str = Field(
        description="The name on that line: 'ProjectIndex', 'refresh' or 'self.index.refresh'",
    )


def _name_column(text: str, name: str) -> int | None:
    """Столбец (в символах) последней части имени ``a.b.c`` в строке."""
    parts = [part for part in name.strip().split(".") if part]
    if not parts:
        return None
    last = parts[-1]
    if len(parts) > 1:
        dotted = r"\s*\.\s*".join(map(re.escape, parts))
        dotted_match = re.search(rf"(?<![\w.]){dotted}(?!\w)", text)
        if dotted_match:
            return dotted_match.end() - len(last)
    name_match = re.search(rf"(?<!\w){re.escape(last)}(?!\w)", text)
    return name_match.start() if name_match else None


def _hover_signature(hover: str) -> str:
    """Тип или сигнатура из подсказки ty — без документации (она после ``---``)."""
    lines: list[str] = []
    for line in hover.splitlines():
        if re.fullmatch(r"\s*-{3,}\s*", line):
            break
        lines.append(line.rstrip())
    return "\n  ".join(kept for kept in lines[:MAX_HOVER_LINES] if kept.strip())


class GotoDefinitionTool(Tool[GotoDefinitionParams]):
    """Инструмент goto_definition: куда ведёт имя в строке файла и каков его тип."""

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

    @override
    def describe(self, params: GotoDefinitionParams) -> str:
        """Имя и место в файле."""
        return f"{params.name} в {params.path}:{params.line}"

    @override
    def paths(self, params: GotoDefinitionParams) -> tuple[str, ...]:
        """Файл, в строке которого ищется имя."""
        return (params.path,)

    @override
    def run(self, params: GotoDefinitionParams, ctx: ToolContext) -> ToolResult:
        """Определение имени и его тип: через ty для Python, иначе — по индексу."""
        rel, col = self._position(params, ctx)
        ty = _Ty(ctx)
        with open_index(ctx) as index:
            refreshed = index.refresh()
            answer = None
            if semantic.is_python(rel):
                answer = ty.run(
                    lambda srv: (
                        srv.definition(rel, params.line, col),
                        srv.hover(rel, params.line, col),
                    ),
                )
            if answer is None:
                out, summary = self._index_answer(index, ty, rel, params)
            else:
                out, summary = self._ty_answer(index, ctx, answer)
        return ToolResult(content="\n".join(out), summary=summary + refresh_note(refreshed))

    def _position(self, params: GotoDefinitionParams, ctx: ToolContext) -> tuple[str, int]:
        """Путь файла от корня проекта и столбец имени в строке; ToolError — их нет."""
        target, lines = self._read_lines(params.path, ctx)
        if not 0 < params.line <= len(lines):
            raise ToolError(f"В файле {params.path} нет строки {params.line} (строк: {len(lines)})")
        text = lines[params.line - 1]
        col = _name_column(text, params.name)
        if col is None:
            raise ToolError(f"В строке {params.line} нет имени «{params.name}»: {text.strip()}")
        return target.relative_to(ctx.root).as_posix(), col

    def _read_lines(self, path: str, ctx: ToolContext) -> tuple[Path, list[str]]:
        """Файл проекта и его строки; ToolError — файла нет, он с секретами или не читается."""
        target = resolve_in_root(ctx.root, path)
        if not target.is_file():
            raise ToolError(f"Файл не найден: {path}")
        if is_secret_target(ctx.root, target):
            raise ToolError(f"Файлы с секретами не разбираются: {path}")
        try:
            lines = split_lines(target.read_text(encoding="utf-8", errors="replace"))
        except OSError as exc:
            raise ToolError(f"Не удалось прочитать {path}: {exc}") from exc
        return target, lines

    def _ty_answer(
        self,
        index: ProjectIndex,
        ctx: ToolContext,
        answer: tuple[list[semantic.Location], str],
    ) -> tuple[list[str], str]:
        """Ответ ty — определения и подсказка с типом — и краткий итог."""
        locations, hover = answer
        source = _SourceLines(ctx)
        out = [
            self._location_line(index, loc, source) for loc in locations[:MAX_DEFINITION_RESULTS]
        ]
        if not locations:
            out.append("(ty не нашёл определение)")
        signature = _hover_signature(hover)
        if signature:
            out.append(f"тип: {signature}")
        return out, f"определений: {len(locations)} (ty)"

    def _index_answer(
        self,
        index: ProjectIndex,
        ty: _Ty,
        rel: str,
        params: GotoDefinitionParams,
    ) -> tuple[list[str], str]:
        """Ответ по индексу — по имени и импортам файла — и краткий итог."""
        name = params.name.strip().split(".")[-1]
        hits = self._from_index(index, rel, name, params.line)
        out = [definition_line(hit) for hit in hits] or ["(определение не найдено)"]
        out += ty.fallback_line() or ["(по индексу — по имени и импортам файла)"]
        return out, f"определений: {len(hits)} (по индексу)"

    def _location_line(
        self,
        index: ProjectIndex,
        loc: semantic.Location,
        source: _SourceLines,
    ) -> str:
        """Строка ответа о месте определения: по индексу, иначе — текст строки."""
        if not loc.in_project:
            return f"{external_path(loc.path)}:{loc.line} (вне проекта)"
        symbol = index.symbol_at(loc.path, loc.line)
        if symbol is not None and symbol.line == loc.line:
            return definition_line(SymbolHit(loc.path, symbol))
        return f"{loc.path}:{loc.line} — {source.get(loc.path, loc.line)}"

    def _from_index(self, index: ProjectIndex, rel: str, name: str, line: int) -> list[SymbolHit]:
        """Определения имени по индексу — от самого вероятного.

        Импорт этого имени в файле → определение в целевом модуле; иначе — в самом
        файле; иначе — все определения с таким именем.
        """
        for entry in index.imports_of(rel):
            if entry.alias != name or entry.target is None:
                continue
            original = entry.name or name
            hits = [
                hit for hit in index.definitions(original) if hit.path == entry.target
            ] or index.definitions(name)
            if hits:
                return [hit for hit in hits if hit.path == entry.target] or hits
        hits = index.definitions(name)
        local = [hit for hit in hits if hit.path == rel and hit.symbol.line != line]
        return local or hits[:MAX_DEFINITION_RESULTS]


# --------------------------------------------------------------------------- #
# call_hierarchy
# --------------------------------------------------------------------------- #
class CallHierarchyParams(BaseModel):
    query: str = Field(
        description="Function or method: 'helper', 'ProjectIndex.refresh' (exact name)",
    )
    direction: str = Field(
        default="incoming",
        description="incoming — who calls it (default); outgoing — what it calls",
    )
    depth: int = Field(default=1, description=f"Depth of the tree: 1–{MAX_DEPTH}")


class _CallTree:
    """Строки одного дерева вызовов: общий бюджет узлов на все корни и уровни."""

    def __init__(self, index: ProjectIndex, *, incoming: bool, depth: int) -> None:
        self._index = index
        self._incoming = incoming
        self._depth = depth
        self._budget = MAX_HIERARCHY_NODES
        self._lines: list[str] = []

    def by_ty(self, server: semantic.TyServer, definitions: list[SymbolHit]) -> list[str]:
        """Дерево через ty: от каждого определения — вызывающие или вызываемые."""
        for definition in definitions:
            symbol = definition.symbol
            roots = server.prepare_call_hierarchy(definition.path, symbol.line, symbol.col)
            self._lines.append(f"{symbol.qualname} — {definition.path}:{symbol.line}")
            if not roots:
                self._lines.append("  (ty не распознал здесь функцию)")
                continue
            seen = {(definition.path, symbol.line)}
            before = len(self._lines)
            for root in roots:
                self._ty_walk(server, root, 1, seen)
            if len(self._lines) == before:
                self._lines.append(_NO_CALLS)
        return self._finish()

    def by_index(self, definitions: list[SymbolHit]) -> list[str]:
        """Дерево по индексу: вызовы ищутся по имени."""
        seen = {definition.symbol.qualname for definition in definitions}
        for definition in definitions:
            symbol = definition.symbol
            self._lines.append(f"{symbol.qualname} — {definition.path}:{symbol.line}")
            before = len(self._lines)
            if self._incoming:
                self._index_callers(symbol.qualname, 1, seen)
            else:
                self._index_callees(definition)
            if len(self._lines) == before:
                self._lines.append(_NO_CALLS)
        return self._finish()

    # ------------------------------ через ty ------------------------------ #
    def _ty_walk(
        self,
        server: semantic.TyServer,
        call_item: semantic.CallItem,
        level: int,
        seen: set[tuple[str, int]],
    ) -> None:
        if self._incoming:
            children = server.incoming_calls(call_item)
        else:
            children = server.outgoing_calls(call_item)
        arrow = "←" if self._incoming else "→"
        # перегрузки (несколько определений вне проекта) — одной строкой
        firsts: dict[str, semantic.CallItem] = {}
        sites: dict[str, list[int]] = {}
        for child in children:
            label = self._label(child.location, child.name)
            firsts.setdefault(label, child)
            sites.setdefault(label, []).extend(call.line for call in child.calls)
        for label, child in firsts.items():
            if not self._take():
                return
            note = self._sites_note(sites[label])
            self._add(level, f"{arrow} {label}{note}")
            key = (child.location.path, child.location.line)
            if level < self._depth and child.location.in_project and key not in seen:
                seen.add(key)
                self._ty_walk(server, child, level + 1, seen)

    def _label(self, loc: semantic.Location, fallback: str) -> str:
        """``Class.method — path:line`` по индексу (или имя из ty, если в индексе нет)."""
        if not loc.in_project:
            shown = fallback or "?"
            return f"{shown} — вне проекта: {external_path(loc.path)}"
        symbol = self._index.symbol_at(loc.path, loc.line)
        name = symbol.qualname if symbol is not None and symbol.line == loc.line else fallback
        shown = name or "?"
        return f"{shown} — {loc.path}:{loc.line}"

    # ----------------------------- по индексу ----------------------------- #
    def _index_callers(self, qualname: str, level: int, seen: set[str]) -> None:
        if self._budget <= 0:
            return
        found = self._index.find_refs(qualname, kind="call", limit=self._budget)
        groups: dict[tuple[str, str, str], list[int]] = {}
        for hit in found.hits:
            groups.setdefault((hit.path, hit.scope, hit.resolution), []).append(hit.line)
        for (path, scope, resolution), call_lines in groups.items():
            if not self._take():
                return
            guess = " (по имени)" if resolution == RESOLVED_NAME else ""
            caller = scope or "<уровень модуля>"
            note = self._sites_note(call_lines)
            self._add(level, f"← {caller} — {path}{note}{guess}")
            if level < self._depth and scope and scope not in seen:
                seen.add(scope)
                self._index_callers(scope, level + 1, seen)

    def _index_callees(self, definition: SymbolHit) -> None:
        calls = self._index.calls_in(definition.path, definition.symbol.qualname)
        for name, call_lines in self._group_calls(calls).items():
            if not self._take():
                break
            self._lines.append(f"  → {name}{self._sites_note(call_lines)}")
        if self._depth > 1:
            self._lines.append("  (глубже 1 уровня исходящие вызовы — только через ty)")

    def _group_calls(self, calls: list[tuple[str, int]]) -> dict[str, list[int]]:
        """Строки вызовов по именам вызываемых — в порядке первого вызова."""
        grouped: dict[str, list[int]] = {}
        for name, line in calls:
            grouped.setdefault(name, []).append(line)
        return grouped

    # ------------------------------- общее ------------------------------- #
    def _sites_note(self, lines: list[int]) -> str:
        """Номера строк мест вызова, например « (строки 3, 7)»; пусто — мест нет."""
        unique = sorted(set(lines))
        if not unique:
            return ""
        word = "строка" if len(unique) == 1 else "строки"
        shown = ", ".join(map(str, unique[:_MAX_NOTED_LINES]))
        if len(unique) > _MAX_NOTED_LINES:
            shown = f"{shown}, …"
        return f" ({word} {shown})"

    def _take(self) -> bool:
        """Забирает узел из бюджета; False — бюджет исчерпан."""
        if self._budget <= 0:
            return False
        self._budget -= 1
        return True

    def _add(self, level: int, text: str) -> None:
        indent = _INDENT * level
        self._lines.append(f"{indent}{text}")

    def _finish(self) -> list[str]:
        if self._budget <= 0:
            self._lines.append(
                f"… дерево обрезано ({MAX_HIERARCHY_NODES} узлов); уменьшите depth",
            )
        return self._lines


class CallHierarchyTool(Tool[CallHierarchyParams]):
    """Инструмент call_hierarchy: дерево вызовов функции или метода в обе стороны."""

    name = "call_hierarchy"
    read_only = True
    description = (
        "Call tree of a function or method: who calls it (incoming) or what it calls "
        f"(outgoing), up to {MAX_DEPTH} levels deep, with the call sites. For Python the "
        "answer comes from ty (taking imports, aliases and types into account); without "
        "it — from the index by name."
    )
    Params = CallHierarchyParams

    @override
    def describe(self, params: CallHierarchyParams) -> str:
        """Запрос и направление обхода."""
        return f"{params.query} ({params.direction})"

    @override
    def run(self, params: CallHierarchyParams, ctx: ToolContext) -> ToolResult:
        """Дерево вызовов: через ty для Python, иначе — по индексу."""
        query = params.query.strip()
        if not query.strip("."):
            raise ToolError(_EMPTY_QUERY)
        incoming = self._incoming(params.direction)
        depth = min(max(params.depth, 1), MAX_DEPTH)
        ty = _Ty(ctx)
        with open_index(ctx) as index:
            refreshed = index.refresh()
            tree, source = self._tree(ty, index, query, incoming=incoming, depth=depth)
        title = "кто вызывает" if incoming else "что вызывает"
        # пояснение бывает, только если ty спрашивали — есть определения на Python
        lines = [f"{title} (глубина {depth}, {source}):", *tree, *ty.fallback_line()]
        return ToolResult(
            content="\n".join(lines),
            summary=self._summary(title, tree, source, refreshed),
        )

    def _incoming(self, direction: str) -> bool:
        """True — кто вызывает, False — что вызывает; ToolError — направление неизвестно."""
        normalized = direction.strip().lower()
        if normalized not in {"incoming", "outgoing"}:
            raise ToolError("direction: incoming (кто вызывает) или outgoing (что вызывает)")
        return normalized == "incoming"

    def _tree(
        self,
        ty: _Ty,
        index: ProjectIndex,
        query: str,
        *,
        incoming: bool,
        depth: int,
    ) -> tuple[list[str], str]:
        """Строки дерева вызовов и откуда они: ty, индекс или оба."""
        definitions = index.definitions(query)
        if not definitions:
            raise ToolError(
                f"Определение «{query}» не найдено в индексе. Найдите имя через find_symbol.",
            )
        new_tree = partial(_CallTree, index, incoming=incoming, depth=depth)
        py_defs = _python_definitions(definitions)
        tree: list[str] | None = None
        if py_defs:
            tree = ty.run(lambda srv: new_tree().by_ty(srv, py_defs))
        if tree is None:
            return new_tree().by_index(definitions[:MAX_TY_DEFINITIONS]), "по индексу"
        # определения на других языках ty не видит — их дерево по индексу
        others = [
            definition for definition in definitions if not semantic.is_python(definition.path)
        ]
        if others:
            tree += new_tree().by_index(others[:MAX_TY_DEFINITIONS])
            return tree, "ty + индекс"
        return tree, "ty"

    def _summary(
        self,
        title: str,
        tree: list[str],
        source: str,
        refreshed: RefreshStats,
    ) -> str:
        """Краткий итог: сколько вызовов в дереве и откуда ответ."""
        calls = sum(1 for line in tree if line.lstrip().startswith(("←", "→")))
        return f"{title}: {calls} ({source}){refresh_note(refreshed)}"
