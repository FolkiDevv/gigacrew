"""Оценки частей команды, устаревшие из-за записи этой же командой.

Классификатор смотрит на файлы до запуска команды: скрипт Python, каталоги поиска модулей,
программу проекта. Если более ранняя (или соседняя по конвейеру) часть той же команды
пишет туда, выполнится уже не то, что проверено: ``echo … > x.py; python x.py``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from gigacrew.safety.levels import CommandLevel, Part

__all__ = ["outdated"]

# Файлы, запись которых меняет то, что импортирует Python: модули, пакеты, .pth.
_CODE_SUFFIXES = frozenset((".py", ".pyc", ".pyo", ".pyw", ".pyd", ".so", ".pth", ".zip", ".egg"))


def outdated(parts: list[Part], writes: list[Part]) -> tuple[list[Part], list[Part]]:
    """Части и записи-перенаправления, оценка которых устарела из-за этой же команды.

    Классификатор смотрит на файлы до запуска команды, а ``echo … > x.py; python x.py``
    выполнит уже новый ``x.py`` (как и ``cp b.py json.py; python -c "import json"``).
    Такая часть — «не знаю». Запись частей уровня EXEC не в счёт: код проекта и так
    исполняется; часть уровня WRITE без известных путей могла записать что угодно. Своя
    запись части не в счёт: то, по чему она оценена, она читает до неё.

    Так же — работа во временной папке (:attr:`Part.targets`), если другая часть ставит
    на место её пути ссылку или переносит туда что-то: ``ln -s ~ .gigacode/tmp/x && rm -rf
    .gigacode/tmp/x/``. Вывод через перенаправление и ``tee`` ссылок не создаёт.
    """
    writers = [*parts, *writes]
    fresh: list[Part] = []
    for index, each in enumerate(writers):
        others = [other for pos, other in enumerate(writers) if pos != index]
        relinkers = [other for pos, other in enumerate(parts) if pos != index]
        checked = each
        if _stale(each, others):
            checked = _untrusted(each, "эта же команда меняет код, который он исполнит")
        elif _relinked(each, relinkers):
            checked = _untrusted(each, "эта же команда ставит на место пути ссылку или перенос")
        fresh.append(checked)
    return fresh[: len(parts)], fresh[len(parts) :]


def _stale(part: Part, others: list[Part]) -> bool:
    """Оценку части ``part`` могли изменить записи других частей ``others``."""
    used = [Path(path) for path in part.inputs]
    if not used or part.level is CommandLevel.DANGER:
        return False
    return any(_changes(each, used) for each in others)


def _relinked(part: Part, others: list[Part]) -> bool:
    """На место путей ``part.targets`` другая часть может поставить ссылку или каталог."""
    targets = [Path(path) for path in part.targets]
    if not targets or part.level is CommandLevel.DANGER:
        return False
    for other in others:
        if other.scratch or other.level is not CommandLevel.WRITE:
            continue  # вывод и mkdir во временной папке ссылок не создают; EXEC — доверен
        if not other.writes:
            return True
        written = [Path(path) for path in other.writes]
        if any(path == target or path in target.parents for path in written for target in targets):
            return True
    return False


def _untrusted(part: Part, why: str) -> Part:
    """Часть, чью оценку меняет эта же команда: «не знаю»."""
    head = part.reason.split(":", 1)[0]
    return replace(
        part,
        level=None,
        reason=f"{head}: {why}",
        signature=None,
        edits=False,
        scratch=False,
    )


def _changes(writer: Part, used: list[Path]) -> bool:
    """Часть ``writer`` может изменить файлы ``used``, по которым оценена другая часть.

    Без известных путей: часть уровня WRITE пишет что угодно, а то, что сделает код
    проекта (EXEC), и так доверено ему.
    """
    if not writer.writes:
        return writer.level is CommandLevel.WRITE and not writer.scratch
    return any(_affects(Path(path), source) for path in writer.writes for source in used)


def _affects(written: Path, used: Path) -> bool:
    """Запись ``written`` меняет вход оценки ``used`` (файл или каталог поиска модулей).

    Тот же файл или каталог, в котором он лежит (``cp a.py dir/``); в каталоге поиска
    модулей — модуль, пакет или ``.pth`` (сегменты с точки не импортируются: ``.gigacode``).
    """
    if written == used or written in used.parents:
        return True
    if used not in written.parents:
        return False
    if any(segment.startswith(".") for segment in written.relative_to(used).parts):
        return False
    return written.suffix in _CODE_SUFFIXES or not written.suffix
