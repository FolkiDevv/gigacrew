"""Файлы проекта: единые правила игнорирования, обход, glob-сопоставление, дерево.

Единственный источник правды о том, какие файлы агент не обходит: служебные
каталоги (:data:`IGNORE_DIRS`) и правила ``.gitignore``
(:mod:`gigacrew.project.gitignore`). Им пользуются дерево проекта в системном
промпте, find_files, search_content и индекс проекта.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

from gigacrew.project.gitignore import IgnoreStack, compile_glob, root_stack, stack_for
from gigacrew.project.workspace import DATA_DIR_NAME

IGNORE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".tox",
        ".eggs",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".idea",
        DATA_DIR_NAME,
    }
)


TREE_TRUNCATED = "... (tree truncated)"


def is_ignored_dir(name: str) -> bool:
    """Каталог служебный (VCS, кеши, окружения, сборка) — не обходим его."""
    return name in IGNORE_DIRS or name.endswith(".egg-info")


def is_secret_file(name: str) -> bool:
    """Файл с секретами (``.env``, ``.env.local``…), кроме шаблона ``.env.example``."""
    return name == ".env" or (name.startswith(".env.") and name != ".env.example")


def is_secret_path(path: Path) -> bool:
    """Файл с секретами — сам или как цель симлинка (``settings.py -> .env``)."""
    if is_secret_file(path.name):
        return True
    if path.is_symlink():
        try:
            return is_secret_file(path.resolve().name)
        except OSError:
            return True
    return False


def is_within(root: Path, path: Path) -> bool:
    """``path`` после раскрытия симлинков лежит внутри ``root``."""
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return resolved == root or root in resolved.parents


def is_excluded(root: Path, path: Path) -> bool:
    """``path`` (внутри ``root``) лежит в служебном каталоге или исключён ``.gitignore``.

    Проверка того же решения, что принимает :func:`walk_files`, для одного пути.
    Путь вне корня считается исключённым.
    """
    root = root.resolve()
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        return True
    if rel == ".":
        return False
    parts = rel.split("/")
    if any(is_ignored_dir(part) for part in parts[:-1]):
        return True
    is_last_dir = path.is_dir() and not path.is_symlink()
    if is_last_dir and is_ignored_dir(parts[-1]):
        return True
    stack = root_stack(root)
    for i in range(len(parts)):
        sub = "/".join(parts[: i + 1])
        is_dir = i < len(parts) - 1 or is_last_dir
        if stack.is_ignored(sub, is_dir):
            return True
        if i < len(parts) - 1:
            stack = stack.enter(root, sub)
    return False


def _rel_dir(root: Path, directory: Path) -> str:
    """Каталог относительно корня в POSIX-виде; сам корень — пустая строка."""
    rel = directory.relative_to(root).as_posix()
    return "" if rel == "." else rel


def walk_files(root: Path, base: Path | None = None, *, gitignore: bool = True) -> Iterator[Path]:
    """Обходит файлы под ``base`` (по умолчанию — ``root``) в отсортированном порядке.

    Служебные каталоги и исключённые ``.gitignore`` отсекаются до спуска в них
    (обход ``node_modules`` не тратит время). Правила каталогов выше ``base``
    действуют и при обходе подкаталога; сам ``base`` (явно указанный путь) не
    проверяется. Симлинки на каталоги не раскрываются, симлинки на файлы вне
    ``root`` пропускаются.
    """
    root = root.resolve()
    start = base if base is not None else root
    if not is_within(root, start):
        gitignore = False  # путь вне корня: правилам проекта не к чему относиться
    elif gitignore:
        start = start.resolve()  # пути от корня считаются по раскрытому пути
    stacks: dict[str, IgnoreStack] = {}
    if gitignore:
        stacks[_rel_dir(root, start)] = stack_for(root, _rel_dir(root, start))
    for dirpath, dirnames, filenames in os.walk(start, followlinks=False):
        current = Path(dirpath)
        rel_dir = _rel_dir(root, current) if gitignore else ""
        stack = stacks.pop(rel_dir, None) if gitignore else None

        def rel(name: str, rel_dir: str = rel_dir) -> str:
            return f"{rel_dir}/{name}" if rel_dir else name

        kept = []
        for d in sorted(dirnames):
            if is_ignored_dir(d):
                continue
            if stack is not None:
                if stack.is_ignored(rel(d), is_dir=True):
                    continue
                stacks[rel(d)] = stack.enter(root, rel(d))
            kept.append(d)
        dirnames[:] = kept
        for name in sorted(filenames):
            if stack is not None and stack.is_ignored(rel(name), is_dir=False):
                continue
            path = current / name
            if path.is_symlink() and not is_within(root, path):
                continue
            yield path


def glob_match(rel_posix: str, pattern: str) -> bool:
    """Сопоставляет относительный POSIX-путь с glob-шаблоном.

    Шаблон без ``/`` сравнивается с именем файла (``*.py`` — любой .py в дереве),
    шаблон с ``/`` — с путём от корня (``src/**/*.py``).
    """
    pattern = pattern.strip()
    while pattern.startswith("./"):
        pattern = pattern[2:]
    target = rel_posix if "/" in pattern else rel_posix.rsplit("/", 1)[-1]
    return compile_glob(pattern).match(target) is not None


TREE_DIR_CAP = 30  # записей одного каталога ниже корня (корень — без ограничения)


def _shares(sizes: list[int], budget: int) -> list[int]:
    """Делит ``budget`` между каталогами поровну, отдавая недобранное остальным."""
    shares = [0] * len(sizes)
    pending = [i for i, size in enumerate(sizes) if size]
    while pending and budget > 0:
        fair = max(budget // len(pending), 1)
        still = []
        for i in pending:
            take = min(sizes[i] - shares[i], fair, budget)
            shares[i] += take
            budget -= take
            if shares[i] < sizes[i]:
                still.append(i)
        pending = still
    return shares


def build_file_tree(root: Path, max_entries: int = 200) -> str:
    """Компактное дерево проекта (отсортированное, с обрезкой).

    Записи отбираются по уровням: сначала корень целиком, затем его каталоги
    (лимит делится между ними поровну, не больше :data:`TREE_DIR_CAP` на каталог)
    и так далее вглубь. Поэтому большой первый каталог не вытесняет файлы корня
    (``README.md``, ``pyproject.toml``) и соседние каталоги. Урезанный каталог
    помечается «… N more», нераскрытый — ``name/ …``.

    Скрытые файлы и каталоги (``.github``, ``.gitignore``) показываются,
    служебные каталоги (``.git``, ``node_modules``…) и исключённое ``.gitignore`` — нет.
    """
    root = root.resolve()

    def is_real_dir(p: Path) -> bool:
        # симлинки на каталоги не раскрываем (возможны циклы и выход из корня)
        return p.is_dir() and not p.is_symlink()

    def listing(directory: Path, rel_dir: str, stack: IgnoreStack) -> list[tuple[str, bool]]:
        try:
            entries = sorted(
                directory.iterdir(), key=lambda e: (not is_real_dir(e), e.name.lower())
            )
        except OSError:
            return []
        kept = []
        for e in entries:
            is_dir = is_real_dir(e)
            rel = f"{rel_dir}/{e.name}" if rel_dir else e.name
            if is_dir and is_ignored_dir(e.name):
                continue
            if stack.is_ignored(rel, is_dir=is_dir):
                continue
            kept.append((e.name, is_dir))
        return kept

    shown: dict[str, list[tuple[str, bool]]] = {}  # каталог → показанные записи
    hidden: dict[str, int] = {}  # каталог → сколько записей не показано
    budget = max_entries
    level = [(root, "", root_stack(root))]
    depth = 0
    while level and budget > 0:
        # Непустых каталогов сверх лимита не раскрыть всё равно — дальше не читаем
        # (пустые и целиком исключённые лимит не тратят и не обрывают перебор).
        listed: list[list[tuple[str, bool]]] = []
        nonempty = 0
        for d, rel, stack in level:
            if nonempty >= budget:
                break
            listed.append(listing(d, rel, stack))
            nonempty += bool(listed[-1])
        level = level[: len(listed)]
        cap = budget if depth == 0 else TREE_DIR_CAP
        shares = _shares([min(len(entries), cap) for entries in listed], budget)
        next_level = []
        for (directory, rel_dir, stack), entries, share in zip(level, listed, shares, strict=True):
            shown[rel_dir] = entries[:share]
            if len(entries) > share:
                hidden[rel_dir] = len(entries) - share
            for name, is_dir in shown[rel_dir]:
                if is_dir:
                    rel = f"{rel_dir}/{name}" if rel_dir else name
                    next_level.append((directory / name, rel, stack.enter(root, rel)))
        budget -= sum(shares)
        level = next_level
        depth += 1

    lines: list[str] = []
    truncated = bool(hidden)

    def render(rel_dir: str, prefix: str) -> None:
        nonlocal truncated
        for name, is_dir in shown.get(rel_dir, []):
            if not is_dir:
                lines.append(f"{prefix}{name}")
                continue
            rel = f"{rel_dir}/{name}" if rel_dir else name
            if rel in shown:
                lines.append(f"{prefix}{name}/")
                render(rel, prefix + "  ")
            else:
                truncated = True
                lines.append(f"{prefix}{name}/ …")
        if hidden.get(rel_dir):
            lines.append(f"{prefix}… {hidden[rel_dir]} more")

    render("", "")
    if truncated:
        lines.append(TREE_TRUNCATED)
    return "\n".join(lines)
