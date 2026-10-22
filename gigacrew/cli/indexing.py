"""Построение индекса проекта из CLI: при запуске REPL и по команде ``/index``.

Пока индекс строится, ввод не принимается; Esc или Ctrl+C отменяет построение
(тем же путём, что прерывание ответа модели), сессия продолжается. Уже сделанное
сохраняется: следующий запуск или ``/index`` достраивает индекс.
"""

from __future__ import annotations

import contextlib
from contextlib import AbstractContextManager

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


def _files(n: int) -> str:
    return f"{n} {plural(n, 'файл', 'файла', 'файлов')}"


def _symbols(n: int) -> str:
    return f"{n} {plural(n, 'определение', 'определения', 'определений')}"


def _refs(n: int) -> str:
    return f"{n} {plural(n, 'использование', 'использования', 'использований')}"


def _size(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / 1024 / 1024:.1f} МБ"
    return f"{max(n // 1024, 1)} КБ"


def run_indexing(
    index: ProjectIndex,
    ui: Console,
    interrupt: AbstractContextManager[object] | None = None,
    *,
    rebuild: bool = False,
) -> tuple[RefreshStats, IndexStats] | None:
    """Обновляет (или перестраивает) индекс с индикатором. None — отменено или ошибка."""
    scanned = 0

    def progress(n: int) -> None:
        nonlocal scanned
        scanned = n

    guard = interrupt if interrupt is not None else contextlib.nullcontext()
    try:
        with guard, ui.progress("индексирую проект", lambda: _files(scanned)):
            try:
                refreshed = (
                    index.rebuild(progress=progress)
                    if rebuild
                    else index.refresh(progress=progress)
                )
                stats = index.stats()
            except INDEX_ERRORS as e:
                if not is_corruption(e):
                    raise
                ui.warn(f"индекс проекта повреждён ({e}) — строю заново")
                index.discard()
                refreshed = index.refresh(progress=progress)
                stats = index.stats()
    except KeyboardInterrupt:
        ui.warn(
            f"построение индекса прервано (просмотрено: {_files(scanned)}, сделанное "
            "сохранено) — /index достроит"
        )
        return None
    except INDEX_ERRORS as e:
        ui.warn(f"индекс проекта недоступен: {e}")
        return None
    finally:
        index.close()
    return refreshed, stats


NOT_A_REPO_NOTE = (
    "проект не в git-репозитории — индекс не строится автоматически (/index — построить)"
)


def ensure_index(
    workspace: Workspace, ui: Console, interrupt: AbstractContextManager[object] | None = None
) -> None:
    """Строит индекс, если его нет или прошлое построение не завершилось.

    Только для проектов в git-репозитории: запуск в домашнем или другом
    «не проектном» каталоге иначе обходил бы всё его дерево при каждом старте.
    Вне репозитория индекс строится по ``/index`` и по требованию инструментов.
    """
    index = ProjectIndex(workspace)
    try:
        if index.is_complete():
            return
    except INDEX_ERRORS:
        pass  # повреждённая база пересоздаётся при построении
    finally:
        index.close()
    if not workspace.in_git_repo():
        ui.system(NOT_A_REPO_NOTE)
        return
    result = run_indexing(index, ui, interrupt)
    if result is not None:
        refreshed, stats = result
        ui.system(
            f"индекс проекта: {_files(stats.files)}, {_symbols(stats.symbols)} "
            f"({refreshed.duration_s:.1f} с)"
        )


def describe_index(refreshed: RefreshStats, stats: IndexStats, ty_status: str = "") -> str:
    """Текст для ``/index``: состав индекса, итог обновления и состояние ty."""
    languages = ", ".join(f"{lang} {n}" for lang, n in stats.languages[:8]) or "—"
    if len(stats.languages) > 8:
        languages += ", …"
    changes = (
        f"добавлено {refreshed.added}, обновлено {refreshed.updated}, удалено {refreshed.removed}"
        if refreshed.changed
        else "изменений нет"
    )
    refs = f", {_refs(stats.refs)}" if stats.refs else ""
    return (
        f"индекс проекта: {_files(stats.files)}, {_symbols(stats.symbols)}{refs}\n"
        f"  языки: {languages}\n"
        f"  {changes} ({refreshed.duration_s:.1f} с); "
        f"размер {_size(stats.db_bytes)}" + (f"\n  ty: {ty_status}" if ty_status else "")
    )
