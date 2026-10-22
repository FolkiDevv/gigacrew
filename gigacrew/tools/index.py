"""Инструменты навигации по индексу проекта: поиск определений, оглавление файла,
карта проекта.

Связи между определениями (использования, переход к определению, вызовы) — в
:mod:`gigacrew.tools.navigation`.

Перед каждым запросом индекс инкрементально обновляется (перечитываются только
изменённые файлы), поэтому результаты отражают текущее состояние файлов, включая
правки самого агента.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager

from pydantic import BaseModel, Field

from gigacrew.project.files import is_excluded, is_secret_path
from gigacrew.project.index import (
    INDEX_ERRORS,
    MAX_INDEX_FILE_BYTES,
    STATUS_BINARY,
    STATUS_ERROR,
    STATUS_LARGE,
    FileEntry,
    ProjectIndex,
    RefreshStats,
    SymbolHit,
    is_corruption,
)
from gigacrew.project.repomap import CHARS_PER_TOKEN, build_repo_map
from gigacrew.project.symbols import Symbol, language_of
from gigacrew.security import resolve_in_root
from gigacrew.tools.base import Tool, ToolContext, ToolError, ToolResult

MAX_SYMBOL_RESULTS = 200
MAX_OUTLINE_FILES = 200  # больше файлов в каталоге — сводка по подкаталогам
MAX_LISTED_FILES = 15  # файлов в строках «импортирует» / «импортируется в»


@contextmanager
def open_index(ctx: ToolContext) -> Iterator[ProjectIndex]:
    index = ProjectIndex(ctx.workspace)
    try:
        with index:
            yield index
    except INDEX_ERRORS as e:
        if is_corruption(e):
            # Испорченные страницы данных видны только на запросе: без пересоздания
            # падал бы каждый вызов, а /index rebuild модели недоступна.
            index.discard()
            raise ToolError(
                f"Индекс проекта был повреждён ({e}) и удалён — повторите вызов: "
                "индекс построится заново."
            ) from e
        raise ToolError(f"Индекс проекта недоступен: {e}") from e


def refresh_note(stats: RefreshStats) -> str:
    return f"; индекс обновлён (файлов: {stats.changed})" if stats.changed else ""


def line_range(symbol: Symbol) -> str:
    if symbol.end_line and symbol.end_line != symbol.line:
        return f"{symbol.line}-{symbol.end_line}"
    return str(symbol.line)


def doc_note(symbol: Symbol) -> str:
    return f"  # {symbol.doc}" if symbol.doc else ""


def definition_line(hit: SymbolHit) -> str:
    s = hit.symbol
    return f"{hit.path}:{line_range(s)} {s.kind} {s.qualname} — {s.signature}{doc_note(s)}"


def short_list(items: list[str]) -> str:
    text = ", ".join(items[:MAX_LISTED_FILES])
    if len(items) > MAX_LISTED_FILES:
        text += f", … и ещё {len(items) - MAX_LISTED_FILES}"
    return text


# --------------------------------------------------------------------------- #
# find_symbol
# --------------------------------------------------------------------------- #
class FindSymbolParams(BaseModel):
    query: str = Field(
        description=(
            "Name or part of the name of a definition (case-insensitive): 'Agent', "
            "'run_turn'; a class method — 'Agent.run_turn'"
        )
    )
    kind: str | None = Field(
        default=None,
        description=(
            "Kind of definition: function (including methods), class (including struct, "
            "interface, enum…), method, constant, section (Markdown heading), etc."
        ),
    )
    glob: str | None = Field(
        default=None, description="Limit to files matching a glob: '*.py' or 'src/**/*.ts'"
    )
    max_results: int = Field(default=50, description="Maximum number of results")


class FindSymbolTool(Tool):
    name = "find_symbol"
    read_only = True
    description = (
        "Finds definitions (classes, functions, methods, types, constants, Markdown "
        "headings) by name in the project index. Returns 'path:lines kind name — "
        "signature'. Faster than search_content for finding where something is defined; "
        "then read the line range you need with read_file."
    )
    Params = FindSymbolParams

    def describe(self, params: FindSymbolParams) -> str:
        return f"{params.query} ({params.kind})" if params.kind else params.query

    def run(self, params: FindSymbolParams, ctx: ToolContext) -> ToolResult:
        if not params.query.strip():
            raise ToolError("Пустой запрос.")
        limit = min(max(params.max_results, 1), MAX_SYMBOL_RESULTS)
        with open_index(ctx) as index:
            refreshed = index.refresh()
            hits, total = index.find_symbols(
                params.query, kind=params.kind, path_glob=params.glob, limit=limit
            )
        if not hits:
            body = (
                "(определений не найдено). Индекс знает только определения; "
                "использования ищите через search_content."
            )
        else:
            body = "\n".join(definition_line(h) for h in hits)
            if total > len(hits):
                body += f"\n… показано {len(hits)} из {total}; уточните запрос, kind или glob"
        return ToolResult(
            content=body,
            summary=f"найдено определений: {total}{refresh_note(refreshed)}",
        )


# --------------------------------------------------------------------------- #
# file_outline
# --------------------------------------------------------------------------- #
class FileOutlineParams(BaseModel):
    path: str = Field(description="File or directory relative to the project root")


def _entry_line(entry: FileEntry, shown_path: str) -> str:
    details = [entry.language or "?", f"строк: {entry.lines}"]
    if entry.symbols:
        details.append(f"определений: {entry.symbols}")
    if entry.status == STATUS_LARGE:
        details.append("слишком большой для разбора")
    elif entry.status == STATUS_BINARY:
        details.append("бинарный")
    return f"{shown_path} ({', '.join(details)})"


class FileOutlineTool(Tool):
    name = "file_outline"
    read_only = True
    description = (
        "Outline from the project index. For a file — its definitions (classes, "
        "functions, methods, headings) with line numbers and nesting, so you can read "
        "only the needed part of a large file with read_file. For a directory — its files "
        "with language, line and definition counts (a summary by subdirectory when there "
        "are many files)."
    )
    Params = FileOutlineParams

    def run(self, params: FileOutlineParams, ctx: ToolContext) -> ToolResult:
        target = resolve_in_root(ctx.root, params.path)
        if not target.exists():
            raise ToolError(f"Путь не найден: {params.path}")
        if target.is_file():
            if is_secret_path(target):
                raise ToolError(f"Файлы с секретами не индексируются: {params.path}")
            if is_excluded(ctx.root, target):
                raise ToolError(
                    f"Файл исключён из индекса (.gitignore или служебный каталог): "
                    f"{params.path}. Используйте read_file."
                )
        with open_index(ctx) as index:
            # импорты Python-файла сопоставляются с другими файлами проекта — нужен весь индекс
            python = target.is_file() and language_of(target.name) == "python"
            refreshed = index.refresh() if python else index.refresh(target)
            rel = target.relative_to(ctx.root).as_posix()
            rel = "" if rel == "." else rel
            if target.is_dir():
                return self._directory(index, rel, refreshed)
            return self._file(index, rel, params.path, refreshed)

    def _file(
        self, index: ProjectIndex, rel: str, shown: str, refreshed: RefreshStats
    ) -> ToolResult:
        entry = index.file_entry(rel)
        if entry is None:
            raise ToolError(f"Файл не попал в индекс: {shown}. Используйте read_file.")
        header = _entry_line(entry, rel)
        if entry.language == "python":
            header += self._imports(index, rel)
        if entry.status == STATUS_LARGE:
            raise ToolError(
                f"Файл слишком большой для разбора (>{MAX_INDEX_FILE_BYTES} байт): {shown}. "
                "Используйте search_content или read_file с диапазоном строк."
            )
        if entry.status == STATUS_BINARY:
            raise ToolError(f"Бинарный файл: {shown}")
        if entry.status == STATUS_ERROR:
            raise ToolError(f"Не удалось разобрать файл: {shown}. Используйте read_file.")
        symbols = index.outline(rel)
        if not symbols:
            body = f"{header}\n(определений не найдено)"
        else:
            body = (
                header
                + "\n"
                + "\n".join(
                    f"{'  ' * s.depth}{line_range(s)}  {s.signature or s.kind + ' ' + s.name}"
                    f"{doc_note(s)}"
                    for s in symbols
                )
            )
        return ToolResult(
            content=body,
            summary=f"{rel}: определений {len(symbols)}{refresh_note(refreshed)}",
        )

    @staticmethod
    def _imports(index: ProjectIndex, rel: str) -> str:
        targets = sorted({e.target for e in index.imports_of(rel) if e.target not in (None, rel)})
        users = [path for path, _ in index.imported_by(rel)]
        text = ""
        if targets:
            text += f"\nимпортирует из проекта: {short_list(targets)}"  # type: ignore[arg-type]
        if users:
            text += f"\nимпортируется в ({len(users)}): {short_list(users)}"
        return text

    def _directory(self, index: ProjectIndex, rel: str, refreshed: RefreshStats) -> ToolResult:
        entries = list(index.files_under(rel))
        prefix = f"{rel}/" if rel else ""
        if not entries:
            body = "(в индексе нет файлов этого каталога)"
        elif len(entries) <= MAX_OUTLINE_FILES:
            body = "\n".join(_entry_line(e, e.path[len(prefix) :]) for e in entries)
        else:
            body = self._grouped(entries, prefix)
        symbols = sum(e.symbols for e in entries)
        where = rel or "."
        return ToolResult(
            content=body,
            summary=f"{where}: файлов {len(entries)}, определений {symbols}"
            f"{refresh_note(refreshed)}",
        )

    @staticmethod
    def _grouped(entries: list[FileEntry], prefix: str) -> str:
        groups: dict[str, list[FileEntry]] = defaultdict(list)
        direct: list[FileEntry] = []
        for e in entries:
            rest = e.path[len(prefix) :]
            if "/" in rest:
                groups[rest.split("/", 1)[0]].append(e)
            else:
                direct.append(e)
        lines = [
            f"{name}/ (файлов: {len(items)}, определений: {sum(i.symbols for i in items)})"
            for name, items in sorted(groups.items())
        ]
        lines += [_entry_line(e, e.path[len(prefix) :]) for e in direct[:MAX_OUTLINE_FILES]]
        if len(direct) > MAX_OUTLINE_FILES:
            lines.append(f"… и ещё {len(direct) - MAX_OUTLINE_FILES} файлов")
        lines.append(
            f"(всего файлов: {len(entries)} — показана сводка по подкаталогам; "
            "уточните путь, чтобы увидеть файлы)"
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# repo_map
# --------------------------------------------------------------------------- #
MIN_MAP_TOKENS = 200
MAX_MAP_TOKENS = 8_000


class RepoMapParams(BaseModel):
    focus: list[str] = Field(
        default_factory=list,
        description=(
            "Focus of the map: project files or directories you are working with and/or "
            "names ('ProjectIndex', 'run_turn'). Empty — what matters in the whole project."
        ),
    )
    max_tokens: int = Field(default=1500, description="Size of the map (estimated tokens)")


class RepoMapTool(Tool):
    name = "repo_map"
    read_only = True
    description = (
        "Project map: the most important definitions (classes, functions, methods with "
        "signatures and line numbers), ranked by how often and from where they are "
        "referenced. With a focus — what is related to the given files and names. The "
        "quickest way to understand an unfamiliar project or the area around a task "
        "without reading whole files."
    )
    Params = RepoMapParams

    def describe(self, params: RepoMapParams) -> str:
        return ", ".join(params.focus) if params.focus else "весь проект"

    def run(self, params: RepoMapParams, ctx: ToolContext) -> ToolResult:
        tokens = min(max(params.max_tokens, MIN_MAP_TOKENS), MAX_MAP_TOKENS)
        with open_index(ctx) as index:
            refreshed = index.refresh()
            files: list[str] = []
            names: list[str] = []
            for item in (f.strip() for f in params.focus):
                if not item:
                    continue
                target = resolve_in_root(ctx.root, item)
                rel = target.relative_to(ctx.root).as_posix()
                if target.is_file():
                    files.append(rel)
                elif target.is_dir():
                    files += [e.path for e in index.files_under("" if rel == "." else rel)]
                else:
                    names.append(item.rsplit(".", 1)[-1])
            repo_map = build_repo_map(
                index,
                focus_files=files,
                focus_names=names,
                max_chars=tokens * CHARS_PER_TOKEN,
            )
        if not repo_map.text:
            body = (
                "(в индексе нет определений)"
                if repo_map.total == 0
                else "(определения не поместились в объём карты — увеличьте max_tokens)"
            )
        else:
            body = (
                f"карта проекта: определений {repo_map.definitions} из {repo_map.total}, "
                f"файлов {repo_map.files}\n{repo_map.text}"
            )
        return ToolResult(
            content=body,
            summary=f"карта: определений {repo_map.definitions}, файлов {repo_map.files}"
            f"{refresh_note(refreshed)}",
        )
