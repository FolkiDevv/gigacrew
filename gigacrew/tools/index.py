"""Инструменты навигации по индексу проекта: поиск определений, оглавление файла, карта проекта.

Связи между определениями (использования, переход к определению, вызовы) — в
:mod:`gigacrew.tools.navigation`.

Перед каждым запросом индекс инкрементально обновляется (перечитываются только
изменённые файлы), поэтому результаты отражают текущее состояние файлов, включая
правки самого агента.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, override

from pydantic import BaseModel, Field

from gigacrew.project.files import is_excluded
from gigacrew.project.index import (
    INDEX_ERRORS,
    MAX_INDEX_FILE_BYTES,
    STATUS_BINARY,
    STATUS_ERROR,
    STATUS_LARGE,
    ProjectIndex,
    is_corruption,
)
from gigacrew.project.repomap import CHARS_PER_TOKEN, build_repo_map
from gigacrew.project.secret_paths import is_secret_target
from gigacrew.project.symbols import Symbol, language_of
from gigacrew.security import resolve_in_root
from gigacrew.tools.base import Tool, ToolContext, ToolError, ToolResult

if TYPE_CHECKING:
    from pathlib import Path

    from gigacrew.project.index import FileEntry, RefreshStats, SymbolHit

MAX_SYMBOL_RESULTS = 200
MAX_OUTLINE_FILES = 200  # больше файлов в каталоге — сводка по подкаталогам
MAX_LISTED_FILES = 15  # файлов в строках «импортирует» / «импортируется в»
_DEFAULT_SYMBOL_RESULTS = 50


@contextmanager
def open_index(ctx: ToolContext) -> Generator[ProjectIndex]:
    """Открывает индекс проекта; его сбои становятся ошибкой инструмента.

    Повреждённый индекс удаляется: следующий вызов построит его заново.
    """
    index = ProjectIndex(ctx.workspace)
    try:
        with index:
            yield index
    except INDEX_ERRORS as exc:
        if is_corruption(exc):
            # Испорченные страницы данных видны только на запросе: без пересоздания
            # падал бы каждый вызов, а /index rebuild модели недоступна.
            index.discard()
            raise ToolError(
                f"Индекс проекта был повреждён ({exc}) и удалён — повторите вызов: "
                "индекс построится заново.",
            ) from exc
        raise ToolError(f"Индекс проекта недоступен: {exc}") from exc


def refresh_note(stats: RefreshStats) -> str:
    """Приписка к сводке о перечитанных файлах (пусто, если индекс уже был актуален)."""
    return f"; индекс обновлён (файлов: {stats.changed})" if stats.changed else ""


def line_range(symbol: Symbol) -> str:
    """Строки определения: ``12-30`` или одна ``12``, если конец неизвестен."""
    if symbol.end_line and symbol.end_line != symbol.line:
        return f"{symbol.line}-{symbol.end_line}"
    return str(symbol.line)


def doc_note(symbol: Symbol) -> str:
    """Первая строка докстринга комментарием (пусто, если её нет)."""
    return f"  # {symbol.doc}" if symbol.doc else ""


def definition_line(hit: SymbolHit) -> str:
    """Строка найденного определения: ``путь:строки вид имя — сигнатура``."""
    symbol = hit.symbol
    location = f"{hit.path}:{line_range(symbol)}"
    return f"{location} {symbol.kind} {symbol.qualname} — {symbol.signature}{doc_note(symbol)}"


def short_list(paths: list[str]) -> str:
    """Пути через запятую: не больше :data:`MAX_LISTED_FILES`, остальные — числом."""
    text = ", ".join(paths[:MAX_LISTED_FILES])
    hidden = len(paths) - MAX_LISTED_FILES
    if hidden > 0:
        return f"{text}, … и ещё {hidden}"
    return text


# --------------------------------------------------------------------------- #
# find_symbol
# --------------------------------------------------------------------------- #
# Докстринг pydantic-модели попал бы в JSON-схему параметров, которую получает модель.
class FindSymbolParams(BaseModel):  # noqa: D101
    query: str = Field(
        description=(
            "Name or part of the name of a definition (case-insensitive): 'Agent', "
            "'run_turn'; a class method — 'Agent.run_turn'"
        ),
    )
    kind: str | None = Field(
        default=None,
        description=(
            "Kind of definition: function (including methods), class (including struct, "
            "interface, enum…), method, constant, section (Markdown heading), etc."
        ),
    )
    glob: str | None = Field(
        default=None,
        description="Limit to files matching a glob: '*.py' or 'src/**/*.ts'",
    )
    max_results: int = Field(
        default=_DEFAULT_SYMBOL_RESULTS,
        description="Maximum number of results",
    )


class FindSymbolTool(Tool[FindSymbolParams]):
    """Поиск определений по имени в индексе проекта."""

    name = "find_symbol"
    read_only = True
    description = (
        "Finds definitions (classes, functions, methods, types, constants, Markdown "
        "headings) by name in the project index. Returns 'path:lines kind name — "
        "signature'. Faster than search_content for finding where something is defined; "
        "then read the line range you need with read_file."
    )
    Params = FindSymbolParams

    @override
    def describe(self, params: FindSymbolParams) -> str:
        """Запрос и вид определения, если он задан."""
        return f"{params.query} ({params.kind})" if params.kind else params.query

    @override
    def run(self, params: FindSymbolParams, ctx: ToolContext) -> ToolResult:
        """Ищет определения в обновлённом индексе."""
        if not params.query.strip():
            raise ToolError("Пустой запрос.")
        limit = min(max(params.max_results, 1), MAX_SYMBOL_RESULTS)
        with open_index(ctx) as index:
            refreshed = index.refresh()
            hits, total = index.find_symbols(
                params.query,
                kind=params.kind,
                path_glob=params.glob,
                limit=limit,
            )
        if hits:
            body = "\n".join(definition_line(hit) for hit in hits)
            if total > len(hits):
                body = f"{body}\n… показано {len(hits)} из {total}; уточните запрос, kind или glob"
        else:
            body = (
                "(определений не найдено). Индекс знает только определения; "
                "использования ищите через search_content."
            )
        return ToolResult(
            content=body,
            summary=f"найдено определений: {total}{refresh_note(refreshed)}",
        )


# --------------------------------------------------------------------------- #
# file_outline
# --------------------------------------------------------------------------- #
# Докстринг pydantic-модели попал бы в JSON-схему параметров, которую получает модель.
class FileOutlineParams(BaseModel):  # noqa: D101
    path: str = Field(description="File or directory relative to the project root")


def _entry_line(entry: FileEntry, shown_path: str) -> str:
    details = [entry.language or "?", f"строк: {entry.lines}"]
    if entry.symbols:
        details.append(f"определений: {entry.symbols}")
    if entry.status == STATUS_LARGE:
        details.append("слишком большой для разбора")
    elif entry.status == STATUS_BINARY:
        details.append("бинарный")
    joined = ", ".join(details)
    return f"{shown_path} ({joined})"


class FileOutlineTool(Tool[FileOutlineParams]):
    """Оглавление файла или каталога по индексу проекта."""

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

    @override
    def run(self, params: FileOutlineParams, ctx: ToolContext) -> ToolResult:
        """Строит оглавление файла или каталога по обновлённому индексу."""
        target = resolve_in_root(ctx.root, params.path)
        if not target.exists():
            raise ToolError(f"Путь не найден: {params.path}")
        if target.is_file():
            if is_secret_target(ctx.root, target):
                raise ToolError(f"Файлы с секретами не индексируются: {params.path}")
            if is_excluded(ctx.root, target):
                raise ToolError(
                    f"Файл исключён из индекса (.gitignore или служебный каталог): "
                    f"{params.path}. Используйте read_file.",
                )
        with open_index(ctx) as index:
            # импорты Python-файла сопоставляются с другими файлами проекта — нужен весь индекс
            python = target.is_file() and language_of(target.name) == "python"
            refreshed = index.refresh() if python else index.refresh(target)
            rel = target.relative_to(ctx.root).as_posix()
            rel = "" if rel == "." else rel
            if target.is_dir():
                return self._directory_outline(index, rel, refreshed)
            return self._file_outline(index, rel, params.path, refreshed)

    def _file_outline(
        self,
        index: ProjectIndex,
        rel: str,
        shown: str,
        refreshed: RefreshStats,
    ) -> ToolResult:
        entry = index.file_entry(rel)
        if entry is None:
            raise ToolError(f"Файл не попал в индекс: {shown}. Используйте read_file.")
        header = _entry_line(entry, rel)
        if entry.language == "python":
            header += self._imports(index, rel)
        self._check_parsed(entry, shown)
        symbols = index.outline(rel)
        if symbols:
            lines = "\n".join(self._outline_line(symbol) for symbol in symbols)
            body = f"{header}\n{lines}"
        else:
            body = f"{header}\n(определений не найдено)"
        return ToolResult(
            content=body,
            summary=f"{rel}: определений {len(symbols)}{refresh_note(refreshed)}",
        )

    def _directory_outline(
        self,
        index: ProjectIndex,
        rel: str,
        refreshed: RefreshStats,
    ) -> ToolResult:
        entries = list(index.files_under(rel))
        prefix = f"{rel}/" if rel else ""
        if entries:
            body = self._listing(entries, prefix)
        else:
            body = "(в индексе нет файлов этого каталога)"
        symbols = sum(entry.symbols for entry in entries)
        where = rel or "."
        return ToolResult(
            content=body,
            summary=f"{where}: файлов {len(entries)}, определений {symbols}"
            f"{refresh_note(refreshed)}",
        )

    def _check_parsed(self, entry: FileEntry, shown: str) -> None:
        """Файл, который индекс не разобрал, — ошибка с подсказкой, чем его читать."""
        if entry.status == STATUS_LARGE:
            raise ToolError(
                f"Файл слишком большой для разбора (>{MAX_INDEX_FILE_BYTES} байт): {shown}. "
                "Используйте search_content или read_file с диапазоном строк.",
            )
        if entry.status == STATUS_BINARY:
            raise ToolError(f"Бинарный файл: {shown}")
        if entry.status == STATUS_ERROR:
            raise ToolError(f"Не удалось разобрать файл: {shown}. Используйте read_file.")

    def _outline_line(self, symbol: Symbol) -> str:
        indent = "  " * symbol.depth
        title = symbol.signature or f"{symbol.kind} {symbol.name}"
        return f"{indent}{line_range(symbol)}  {title}{doc_note(symbol)}"

    def _imports(self, index: ProjectIndex, rel: str) -> str:
        """Строки о связях Python-файла: что он импортирует из проекта и где импортируется сам."""
        targets = sorted({
            imported.target
            for imported in index.imports_of(rel)
            if imported.target not in {None, rel}
        })
        users = [path for path, _ in index.imported_by(rel)]
        note = ""
        if targets:
            note = f"\nимпортирует из проекта: {short_list(targets)}"
        if users:
            note = f"{note}\nимпортируется в ({len(users)}): {short_list(users)}"
        return note

    def _listing(self, entries: list[FileEntry], prefix: str) -> str:
        if len(entries) <= MAX_OUTLINE_FILES:
            return "\n".join(_entry_line(entry, entry.path[len(prefix) :]) for entry in entries)
        return self._grouped(entries, prefix)

    def _grouped(self, entries: list[FileEntry], prefix: str) -> str:
        groups: dict[str, list[FileEntry]] = defaultdict(list)
        direct: list[FileEntry] = []
        for entry in entries:
            rest = entry.path[len(prefix) :]
            if "/" in rest:
                groups[rest.split("/", 1)[0]].append(entry)
            else:
                direct.append(entry)
        lines: list[str] = []
        for name, members in sorted(groups.items()):
            symbols = sum(member.symbols for member in members)
            lines.append(f"{name}/ (файлов: {len(members)}, определений: {symbols})")
        lines += [
            _entry_line(shown, shown.path[len(prefix) :]) for shown in direct[:MAX_OUTLINE_FILES]
        ]
        hidden = len(direct) - MAX_OUTLINE_FILES
        if hidden > 0:
            lines.append(f"… и ещё {hidden} файлов")
        lines.append(
            f"(всего файлов: {len(entries)} — показана сводка по подкаталогам; "
            "уточните путь, чтобы увидеть файлы)",
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# repo_map
# --------------------------------------------------------------------------- #
MIN_MAP_TOKENS = 200
MAX_MAP_TOKENS = 8_000
_DEFAULT_MAP_TOKENS = 1500


# Докстринг pydantic-модели попал бы в JSON-схему параметров, которую получает модель.
class RepoMapParams(BaseModel):  # noqa: D101
    focus: list[str] = Field(
        default_factory=list,
        description=(
            "Focus of the map: project files or directories you are working with and/or "
            "names ('ProjectIndex', 'run_turn'). Empty — what matters in the whole project."
        ),
    )
    max_tokens: int = Field(
        default=_DEFAULT_MAP_TOKENS,
        description="Size of the map (estimated tokens)",
    )


class RepoMapTool(Tool[RepoMapParams]):
    """Карта проекта: важнейшие определения, ранжированные по использованиям."""

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

    @override
    def describe(self, params: RepoMapParams) -> str:
        """Фокус карты или «весь проект»."""
        return ", ".join(params.focus) if params.focus else "весь проект"

    @override
    def run(self, params: RepoMapParams, ctx: ToolContext) -> ToolResult:
        """Строит карту проекта по обновлённому индексу."""
        tokens = min(max(params.max_tokens, MIN_MAP_TOKENS), MAX_MAP_TOKENS)
        with open_index(ctx) as index:
            refreshed = index.refresh()
            files, names = self._split_focus(params.focus, index, ctx.root)
            repo_map = build_repo_map(
                index,
                focus_files=files,
                focus_names=names,
                max_chars=tokens * CHARS_PER_TOKEN,
            )
        if repo_map.text:
            body = (
                f"карта проекта: определений {repo_map.definitions} из {repo_map.total}, "
                f"файлов {repo_map.files}\n{repo_map.text}"
            )
        elif repo_map.total == 0:
            body = "(в индексе нет определений)"
        else:
            body = "(определения не поместились в объём карты — увеличьте max_tokens)"
        return ToolResult(
            content=body,
            summary=f"карта: определений {repo_map.definitions}, файлов {repo_map.files}"
            f"{refresh_note(refreshed)}",
        )

    def _split_focus(
        self,
        focus: list[str],
        index: ProjectIndex,
        root: Path,
    ) -> tuple[list[str], list[str]]:
        """Фокус карты: (файлы проекта, имена). Каталог раскрывается в свои файлы."""
        files: list[str] = []
        names: list[str] = []
        for raw in focus:
            part = raw.strip()
            if not part:
                continue
            target = resolve_in_root(root, part)
            rel = target.relative_to(root).as_posix()
            if target.is_file():
                files.append(rel)
            elif target.is_dir():
                under = "" if rel == "." else rel
                files += [entry.path for entry in index.files_under(under)]
            else:
                names.append(part.rsplit(".", 1)[-1])
        return files, names
