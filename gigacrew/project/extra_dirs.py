"""Подключённые каталоги (``--add-dir``, ``/add-dir``): что можно подключить.

Подключённый каталог — например, другой проект, из которого повторяют решение: файловые
инструменты и классификатор команд работают с ним, как с проектом (чтение и правки), а
секреты, служебные папки git/jj и ``.gigacode/`` в нём защищены так же. Список хранит
:class:`gigacrew.project.workspace.Workspace` (``extra_dirs``).
"""

from __future__ import annotations

import os
from pathlib import Path

from gigacrew.project.secret_paths import is_secret_location

__all__ = ["check_extra_dir"]


def check_extra_dir(path: str | Path, root: Path, *, home: Path | None = None) -> Path:
    """Подключаемый каталог (``--add-dir``, ``/add-dir``) — раскрытый; ``ValueError`` — нельзя.

    Относительный путь — от текущего каталога. Нельзя: несуществующий, не каталог,
    сам проект или каталог в нём, корень ФС, домашний каталог и всё, что его содержит,
    каталоги с секретами (``~/.ssh``…).
    """
    # Path.expanduser падает на «~неизвестный», os.path.expanduser оставляет как есть.
    raw = Path(os.path.expanduser(path))  # noqa: PTH111
    directory = raw.resolve()
    if not directory.is_dir():
        raise ValueError(f"каталог не найден: {path}")
    problem = _problem(raw, directory, Path(root).resolve(), home or Path.home())
    if problem:
        raise ValueError(f"{directory} — {problem}")
    return directory


def _problem(raw: Path, directory: Path, project: Path, home: Path) -> str:
    """Почему существующий каталог нельзя подключить ("" — можно)."""
    if directory == project or project in directory.parents:
        return "уже в проекте"
    user_home = home.resolve()
    if directory == user_home or directory in user_home.parents:
        return "слишком широко: подключите каталог конкретного проекта"
    if is_secret_location(directory) or is_secret_location(raw):
        return "каталог с секретами"
    return ""
