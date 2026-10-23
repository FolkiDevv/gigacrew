"""Построение индекса проекта из CLI: при запуске REPL и по команде ``/index``.

Пока индекс строится, ввод не принимается; Esc или Ctrl+C отменяет построение
(тем же путём, что прерывание ответа модели), сессия продолжается. Уже сделанное
сохраняется: следующий запуск или ``/index`` достраивает индекс.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

from gigacrew.project.index import (
    INDEX_ERRORS,
    IndexStats,
    ProjectIndex,
    RefreshStats,
    is_corruption,
)
from gigacrew.project.workspace import Workspace
from gigacrew.ui.console import Console
from gigacrew.ui.format import plural

_MEGABYTE = 1024 * 1024
_LANGUAGES_SHOWN = 8  # языков в ``/index``; остальные — «…»


def _files(count: int) -> str:
    word = plural(count, "файл", "файла", "файлов")
    return f"{count} {word}"


def _symbols(count: int) -> str:
    word = plural(count, "определение", "определения", "определений")
    return f"{count} {word}"


def _refs(count: int) -> str:
    word = plural(count, "использование", "использования", "использований")
    return f"{count} {word}"


def _size(size: int) -> str:
    if size >= _MEGABYTE:
        megabytes = size / 1024 / 1024
        return f"{megabytes:.1f} МБ"
    kilobytes = max(size // 1024, 1)
    return f"{kilobytes} КБ"


class _ScanProgress:
    """Сколько файлов просмотрено: счётчик для индикатора построения индекса."""

    def __init__(self) -> None:
        self.scanned = 0

    def update(self, count: int) -> None:
        """Обратный вызов индекса: просмотрено ``count`` файлов."""
        self.scanned = count

    def describe(self) -> str:
        """Строка индикатора: «1234 файла»."""
        return _files(self.scanned)


def _refresh(
    index: ProjectIndex,
    ui: Console,
    scan: _ScanProgress,
    *,
    rebuild: bool,
) -> tuple[RefreshStats, IndexStats]:
    """Обновляет (или перестраивает) индекс; повреждённую базу строит заново."""
    try:
        refreshed = (
            index.rebuild(progress=scan.update) if rebuild else index.refresh(progress=scan.update)
        )
        stats = index.stats()
    except INDEX_ERRORS as error:
        if not is_corruption(error):
            raise
        ui.warn(f"индекс проекта повреждён ({error}) — строю заново")
        index.discard()
        refreshed = index.refresh(progress=scan.update)
        stats = index.stats()
    return refreshed, stats


def run_indexing(
    index: ProjectIndex,
    ui: Console,
    interrupt: AbstractContextManager[object] | None = None,
    *,
    rebuild: bool = False,
) -> tuple[RefreshStats, IndexStats] | None:
    """Обновляет (или перестраивает) индекс с индикатором. None — отменено или ошибка."""
    scan = _ScanProgress()
    guard: AbstractContextManager[object] = nullcontext() if interrupt is None else interrupt
    try:
        with guard, ui.progress("индексирую проект", scan.describe):
            refreshed, stats = _refresh(index, ui, scan, rebuild=rebuild)
    except KeyboardInterrupt:
        ui.warn(
            f"построение индекса прервано (просмотрено: {scan.describe()}, сделанное "
            "сохранено) — /index достроит",
        )
        return None
    except INDEX_ERRORS as error:
        ui.warn(f"индекс проекта недоступен: {error}")
        return None
    finally:
        index.close()
    return refreshed, stats


NOT_A_REPO_NOTE = (
    "проект не в git-репозитории — индекс не строится автоматически (/index — построить)"
)


def _is_complete(index: ProjectIndex) -> bool:
    """Индекс построен полностью; повреждённая база — нет (пересоздаётся при построении)."""
    try:
        return index.is_complete()
    except INDEX_ERRORS:
        return False
    finally:
        index.close()


def ensure_index(
    workspace: Workspace,
    ui: Console,
    interrupt: AbstractContextManager[object] | None = None,
) -> None:
    """Строит индекс, если его нет или прошлое построение не завершилось.

    Только для проектов в git-репозитории: запуск в домашнем или другом
    «не проектном» каталоге иначе обходил бы всё его дерево при каждом старте.
    Вне репозитория индекс строится по ``/index`` и по требованию инструментов.
    """
    index = ProjectIndex(workspace)
    if _is_complete(index):
        return
    if not workspace.in_git_repo():
        ui.system(NOT_A_REPO_NOTE)
        return
    outcome = run_indexing(index, ui, interrupt)
    if outcome is not None:
        refreshed, stats = outcome
        ui.system(
            f"индекс проекта: {_files(stats.files)}, {_symbols(stats.symbols)} "
            f"({refreshed.duration_s:.1f} с)",
        )


def describe_index(refreshed: RefreshStats, stats: IndexStats, ty_status: str = "") -> str:
    """Текст для ``/index``: состав индекса, итог обновления и состояние ty."""
    shown = stats.languages[:_LANGUAGES_SHOWN]
    languages = ", ".join(f"{lang} {count}" for lang, count in shown) or "—"
    if len(stats.languages) > _LANGUAGES_SHOWN:
        languages = f"{languages}, …"
    changes = (
        f"добавлено {refreshed.added}, обновлено {refreshed.updated}, удалено {refreshed.removed}"
        if refreshed.changed
        else "изменений нет"
    )
    refs = f", {_refs(stats.refs)}" if stats.refs else ""
    ty_line = f"\n  ty: {ty_status}" if ty_status else ""
    return (
        f"индекс проекта: {_files(stats.files)}, {_symbols(stats.symbols)}{refs}\n"
        f"  языки: {languages}\n"
        f"  {changes} ({refreshed.duration_s:.1f} с); "
        f"размер {_size(stats.db_bytes)}{ty_line}"
    )
