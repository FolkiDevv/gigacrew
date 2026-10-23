"""Поиск по содержимому кодовой базы (аналог grep)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import override

from pydantic import BaseModel, Field

from gigacrew.project.files import glob_match, walk_files
from gigacrew.project.secret_paths import is_secret_target
from gigacrew.security import resolve_in_root
from gigacrew.tools.base import Tool, ToolContext, ToolError, ToolResult
from gigacrew.tools.fs import SECRET_READ_NOTE

_MAX_MATCHES = 500  # потолок max_results, который может запросить модель
_MAX_FILE_BYTES = 1_000_000  # файлы крупнее не читаем (сгенерированное, дампы)
_BYTES_IN_MB = 1_000_000
_BINARY_PROBE = 8192  # NUL в первых байтах — бинарный файл
_MAX_SNIPPET = 200


# Без докстринга: pydantic отдал бы его в JSON-схему параметров, которая уходит модели.
class SearchContentParams(BaseModel):  # noqa: D101
    pattern: str = Field(description="Regular expression to search for in file contents")
    path: str = Field(
        default=".",
        description="Directory or file to search in (relative to the project root)",
    )
    glob: str | None = Field(
        default=None,
        description="Limit the search to files matching a glob: '*.py' (by name) or 'src/**/*.py'",
    )
    ignore_case: bool = Field(default=False, description="Case-insensitive search")
    max_results: int = Field(default=100, description="Maximum number of matches")


@dataclass
class _Found:
    """Совпадения поиска и счётчики файлов."""

    limit: int
    matches: list[str] = field(default_factory=list)
    scanned: int = 0  # файлов просмотрено
    skipped: int = 0  # больших файлов пропущено
    more: bool = False  # совпадений больше лимита


def _compile(params: SearchContentParams) -> re.Pattern[str]:
    """Регулярное выражение поиска; некорректное — ToolError."""
    flags = re.IGNORECASE if params.ignore_case else 0
    try:
        return re.compile(params.pattern, flags)
    except re.error as exc:
        raise ToolError(f"Некорректное регулярное выражение: {exc}") from exc


def _wanted(
    path: Path,
    rel: str,
    params: SearchContentParams,
    ctx: ToolContext,
    *,
    walking: bool,
) -> bool:
    """Искать ли в файле: он подходит под ``glob`` и (при обходе каталога) без секретов."""
    if params.glob and not glob_match(rel, params.glob):
        return False
    # секреты не ищем — то же правило, что у read_file и классификатора команд
    return not (walking and is_secret_target(ctx.root, path))


def _read_text(path: Path, found: _Found) -> str | None:
    """Текст файла для поиска; None — файл не обычный, не читается, большой или бинарный.

    Пропущенные из-за размера файлы считаются в ``found.skipped``.
    """
    try:
        # FIFO, сокеты, устройства: открытие на чтение может заблокироваться
        if not path.is_file():
            return None
        if path.stat().st_size > _MAX_FILE_BYTES:
            found.skipped += 1
            return None
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\0" in raw[:_BINARY_PROBE]:
        return None
    return raw.decode("utf-8", errors="replace")


def _collect(found: _Found, rel: str, text: str, regex: re.Pattern[str]) -> None:
    """Добавляет совпадения из текста файла ``rel``; сверх лимита — отмечает ``more``."""
    # Строки — как у read_file (\n, \r\n, \r), а не str.splitlines: тот режет
    # ещё и по \x0c, \x85, \u2028, и номера строк расходились бы.
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for line_no, line in enumerate(lines, start=1):
        if not regex.search(line):
            continue
        if len(found.matches) >= found.limit:
            found.more = True
            return
        snippet = line.strip()[:_MAX_SNIPPET]
        found.matches.append(f"{rel}:{line_no}:{snippet}")


def _report(found: _Found) -> ToolResult:
    """Итог поиска: совпадения для модели и краткая строка для UI."""
    body = "\n".join(found.matches) if found.matches else "(совпадений не найдено)"
    if found.more:
        body = f"{body}\n… показаны первые {found.limit} совпадений; уточните запрос"
    count = len(found.matches)
    summary = f"совпадений: {count} (файлов просмотрено: {found.scanned})"
    if found.skipped:
        max_mb = _MAX_FILE_BYTES // _BYTES_IN_MB
        body = (
            f"{body}\n… пропущено больших файлов (>{max_mb} МБ): {found.skipped} "
            "— в них ищите через run_shell (grep)"
        )
        summary = f"{summary}, пропущено больших файлов: {found.skipped}"
    return ToolResult(content=body, summary=summary)


class SearchContentTool(Tool[SearchContentParams]):
    """Инструмент search_content: поиск строк по регулярному выражению (как grep -rn)."""

    name = "search_content"
    read_only = True
    description = (
        "Searches project files for lines matching a regular expression, like grep -rn. "
        "Returns matches as path:line:text. Use it for strings, messages, config keys and "
        "other text; to find where a class or function is defined, find_symbol is faster. "
        "Skips service directories, files ignored by .gitignore, binary and very large "
        "files, and files with secrets (.env, keys, credentials)."
    )
    # Модель параметров (класс pydantic), а не константа: имя — часть API инструментов.
    Params = SearchContentParams  # noqa: WPS115

    @override
    def describe(self, params: SearchContentParams) -> str:
        """Шаблон поиска (и путь, если он не корень) для UI."""
        where = "" if params.path in {"", "."} else f" в {params.path}"
        return f"/{params.pattern}/{where}"

    @override
    def run(self, params: SearchContentParams, ctx: ToolContext) -> ToolResult:
        """Ищет строки, совпадающие с шаблоном, в файле или в файлах каталога."""
        base = resolve_in_root(ctx.root, params.path)
        if not base.exists():
            raise ToolError(f"Путь не найден: {params.path}")
        regex = _compile(params)
        limit = min(max(params.max_results, 1), _MAX_MATCHES)

        walking = base.is_dir()
        if not walking and is_secret_target(ctx.root, base):
            raise ToolError(SECRET_READ_NOTE.format(path=params.path))
        files = walk_files(ctx.root, base) if walking else [base]
        found = _Found(limit)
        for path in files:
            rel = path.relative_to(ctx.root).as_posix()
            if not _wanted(path, rel, params, ctx, walking=walking):
                continue
            text = _read_text(path, found)
            if text is None:
                continue
            found.scanned += 1
            _collect(found, rel, text, regex)
            if found.more:
                break
        return _report(found)
