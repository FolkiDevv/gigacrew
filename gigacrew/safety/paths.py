"""Где окажется путь из команды: внутри проекта, снаружи, среди секретов или неизвестно.

Путь считается относительно рабочего каталога команды (``cwd`` от корня). Симлинки
раскрываются, как в песочнице файловых инструментов (:func:`gigacrew.security.resolve_in_root`):
ссылка внутри проекта на ``~/.ssh`` — снаружи. Шаблон (``*.py``) раскрывается тем же
способом, что и в оболочке: скрытые файлы ``*`` не совпадают, а сегмент вида ``.*`` в
dash совпадает и с ``..`` — такой шаблон считается выходом наружу.
"""

from __future__ import annotations

import fnmatch
import glob
import os
import re
from collections.abc import Iterator
from pathlib import Path

from gigacrew.project import files
from gigacrew.project.secret_paths import (
    SECRET_DIRS,
    SINKS,
    Scope,
    is_secret_location,
    is_secret_name,
    scope_of_path,
)
from gigacrew.project.workspace import Workspace
from gigacrew.safety.bash import Word

__all__ = [
    "Scope",
    "has_escaping_links",
    "has_secrets_under",
    "is_data_path",
    "is_git_internal",
    "is_secret_name",
    "scope",
    "scope_of_path",
]

# Сколько совпадений шаблона проверять; больше — не знаем, что там.
MAX_GLOB_MATCHES = 2000


def is_git_internal(path: Path, root: Path) -> bool:
    """Путь в служебном каталоге git: хуки и настройки оттуда запускают программы."""
    return files.is_git_internal(root, path)


def is_data_path(path: Path, root: Path) -> bool:
    """Путь в ``.gigacode/`` — служебные данные gigacrew (чаты, планы, скиллы)."""
    return Workspace(root).is_data_path(path)


def _resolve(root: Path, cwd: str, path_text: str) -> Scope:
    if path_text.startswith("~"):
        # Path.expanduser падает на «~неизвестный», os.path.expanduser оставляет как есть.
        lexical = Path(os.path.expanduser(path_text))  # noqa: PTH111
        return Scope.SECRET_OUTSIDE if is_secret_location(lexical) else Scope.OUTSIDE
    if path_text in SINKS:
        return Scope.SINK
    lexical_path = Path(path_text) if path_text.startswith("/") else root / cwd / path_text
    return scope_of_path(lexical_path, root)


def _worst(scopes: list[Scope]) -> Scope:
    order = [
        Scope.SECRET_OUTSIDE,
        Scope.OUTSIDE,
        Scope.DYNAMIC,
        Scope.SECRET,
        Scope.SINK,
        Scope.INSIDE,
    ]
    for candidate in order:
        if candidate in scopes:
            return candidate
    return Scope.INSIDE


# Проверки пути идут подряд, от простых к раскрытию шаблона: так их легче сверять с оболочкой.
def scope(word: Word, *, root: Path, cwd: str = ".") -> Scope:  # noqa: WPS212
    """Область пути ``word`` для команды, запущенной в ``root/cwd``."""
    if not word.static:
        return Scope.DYNAMIC
    pattern = word.text
    if not word.glob:
        return _resolve(root, cwd, pattern)
    # Шаблон: сегмент, начинающийся с точки, в dash совпадает с «..».
    if any(seg.startswith(".") and re.search(r"[*?\[]", seg) for seg in pattern.split("/")):
        return Scope.OUTSIDE
    if pattern.startswith(("~", "/")):
        base_scope = _resolve(root, cwd, pattern.split("*")[0].rsplit("/", 1)[0] or "/")
        if base_scope != Scope.INSIDE:
            return base_scope if base_scope == Scope.SECRET_OUTSIDE else Scope.OUTSIDE
    try:
        # glob.glob, как оболочка, не находит скрытые файлы по «*», а Path.glob находит.
        matches = glob.glob(pattern, root_dir=root / cwd)  # noqa: PTH207
    except (OSError, ValueError):
        return Scope.DYNAMIC
    if len(matches) > MAX_GLOB_MATCHES:
        return Scope.DYNAMIC
    if not matches:  # без совпадений шаблон остаётся словом как есть
        return _resolve(root, cwd, pattern)
    return _worst([_resolve(root, cwd, match) for match in matches])


def _walk(directory: Path, limit: int) -> Iterator[tuple[Path, list[str], list[str]] | None]:
    """Обход дерева целиком, как у ``grep -r`` и ``find``.

    Служебные каталоги (build, node_modules, .venv…) они не пропускают, значит, и проверка
    не пропускает. Отдаёт (каталог, подкаталоги, файлы); None — дерево больше ``limit``
    записей.
    """
    seen = 0
    for current, dirs, names in os.walk(directory):
        seen += len(names) + len(dirs)
        if seen > limit:
            yield None
            return
        yield Path(current), dirs, names


# Обход дерева и проверки найденных файлов — одним циклом, как идёт сама рекурсивная команда.
def has_secrets_under(  # noqa: WPS231
    directory: Path,
    root: Path,
    *,
    name_pattern: str | None = None,
    limit: int = 20000,
) -> bool:
    """Есть ли файлы с секретами в дереве каталога (рекурсивное чтение их покажет).

    ``name_pattern`` — шаблон имён, которые прочитает команда (``*.py``): секреты с
    другими именами не в счёт. Слишком большое дерево — считаем, что есть.
    """
    if not directory.is_dir():
        return is_secret_location(directory)
    for entry in _walk(directory, limit):
        if entry is None:
            return True
        current, dirs, names = entry
        if any(dirname in SECRET_DIRS for dirname in dirs):
            return True
        if ".git" in dirs and _git_config_has_credentials(current / ".git" / "config"):
            return True
        for name in names:
            if name_pattern is not None and not fnmatch.fnmatch(name, name_pattern):
                continue
            if is_secret_name(name) or files.is_secret_path(current / name):
                return True
    return False


_URL_CREDENTIALS = re.compile(r"://[^/\s:@]+:[^/\s@]+@")


def _git_config_has_credentials(path: Path) -> bool:
    """В адресах удалённых репозиториев записан пароль или токен (``https://u:t@host``)."""
    try:
        return bool(_URL_CREDENTIALS.search(path.read_text(encoding="utf-8", errors="replace")))
    except OSError:
        return False


# Обход дерева и проверки ссылок — одним циклом, как идёт сама рекурсивная команда.
def has_escaping_links(  # noqa: WPS231
    directory: Path,
    root: Path,
    *,
    name_pattern: str | None = None,
    limit: int = 20000,
) -> bool:
    """Есть ли в дереве символические ссылки за пределы проекта (чтение по ним — наружу)."""
    if not directory.is_dir():
        return False
    for entry in _walk(directory, limit):
        if entry is None:
            return True
        current, dirs, names = entry
        for name in (*dirs, *names):
            path = current / name
            if not path.is_symlink():
                continue
            if (
                name_pattern is not None
                and name in names
                and not fnmatch.fnmatch(name, name_pattern)
            ):
                continue
            if scope_of_path(path, root) in {Scope.OUTSIDE, Scope.SECRET_OUTSIDE, Scope.DYNAMIC}:
                return True
    return False
