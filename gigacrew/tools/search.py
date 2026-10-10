"""Поиск по содержимому кодовой базы (аналог grep)."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from gigacrew.project.files import glob_match, is_secret_path, walk_files
from gigacrew.security import resolve_in_root
from gigacrew.tools.base import Tool, ToolContext, ToolError, ToolResult

_MAX_MATCHES = 500  # потолок max_results, который может запросить модель
_MAX_FILE_BYTES = 1_000_000  # файлы крупнее не читаем (сгенерированное, дампы)
_BINARY_PROBE = 8192  # NUL в первых байтах — бинарный файл
_MAX_SNIPPET = 200


class SearchContentParams(BaseModel):
    pattern: str = Field(description="Регулярное выражение для поиска по содержимому")
    path: str = Field(default=".", description="Директория или файл для поиска (от корня проекта)")
    glob: str | None = Field(
        default=None,
        description="Ограничить поиск файлами по glob: '*.py' (по имени) или 'src/**/*.py'",
    )
    ignore_case: bool = Field(default=False, description="Игнорировать регистр")
    max_results: int = Field(default=100, description="Максимум совпадений")


class SearchContentTool(Tool):
    name = "search_content"
    description = (
        "Ищет строки, соответствующие регулярному выражению, по файлам проекта. "
        "Возвращает совпадения в формате path:line:текст. Аналог grep -rn. "
        "Пропускает служебные каталоги, файлы из .gitignore, бинарные и очень большие "
        "файлы, а также .env."
    )
    Params = SearchContentParams

    def describe(self, params: SearchContentParams) -> str:
        where = f" в {params.path}" if params.path not in ("", ".") else ""
        return f"/{params.pattern}/{where}"

    def run(self, params: SearchContentParams, ctx: ToolContext) -> ToolResult:
        base = resolve_in_root(ctx.root, params.path)
        if not base.exists():
            raise ToolError(f"Путь не найден: {params.path}")
        flags = re.IGNORECASE if params.ignore_case else 0
        try:
            regex = re.compile(params.pattern, flags)
        except re.error as e:
            raise ToolError(f"Некорректное регулярное выражение: {e}") from e
        limit = min(max(params.max_results, 1), _MAX_MATCHES)

        walking = base.is_dir()
        files = walk_files(ctx.root, base) if walking else [base]
        matches: list[str] = []
        scanned = skipped = 0
        more = False  # совпадений больше лимита
        for f in files:
            rel = f.relative_to(ctx.root).as_posix()
            if params.glob and not glob_match(rel, params.glob):
                continue
            # секреты не ищем при обходе каталога; явно указанный файл — можно
            if walking and is_secret_path(f):
                continue
            try:
                # FIFO, сокеты, устройства: открытие на чтение может заблокироваться
                if not f.is_file():
                    continue
                if f.stat().st_size > _MAX_FILE_BYTES:
                    skipped += 1
                    continue
                data = f.read_bytes()
            except OSError:
                continue
            if b"\0" in data[:_BINARY_PROBE]:
                continue
            scanned += 1
            text = data.decode("utf-8", errors="replace")
            # Строки — как у read_file (\n, \r\n, \r), а не str.splitlines: тот режет
            # ещё и по \x0c, \x85, \u2028, и номера строк расходились бы.
            lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            for i, line in enumerate(lines, start=1):
                if regex.search(line):
                    if len(matches) >= limit:
                        more = True
                        break
                    matches.append(f"{rel}:{i}:{line.strip()[:_MAX_SNIPPET]}")
            if more:
                break

        body = "\n".join(matches) if matches else "(совпадений не найдено)"
        if more:
            body += f"\n… показаны первые {limit} совпадений; уточните запрос"
        if skipped:
            body += (
                f"\n… пропущено больших файлов (>{_MAX_FILE_BYTES // 1_000_000} МБ): {skipped} "
                "— в них ищите через run_shell (grep)"
            )
        summary = f"совпадений: {len(matches)} (файлов просмотрено: {scanned})"
        if skipped:
            summary += f", пропущено больших файлов: {skipped}"
        return ToolResult(content=body, summary=summary)
