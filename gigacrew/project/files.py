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

IGNORE_DIRS: frozenset[str] = frozenset((
    ".git",
    ".jj",
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
))


TREE_TRUNCATED = "... (tree truncated)"
GIT_DIR_NAME = ".git"
JJ_DIR_NAME = ".jj"
# Служебные каталоги систем контроля версий: настройки оттуда запускают программы.
VCS_DIR_NAMES = frozenset((GIT_DIR_NAME, JJ_DIR_NAME))

# Раскладка каталога git: обязательная запись и признак «это каталог».
_GIT_LAYOUT = (("HEAD", False), ("objects", True), ("refs", True))

# Запись дерева проекта: имя и признак «настоящий каталог» (не симлинк).
_Entry = tuple[str, bool]
# Каталоги одного уровня дерева: путь, путь от корня и правила .gitignore в каталоге.
_Level = list[tuple[Path, str, IgnoreStack]]


def is_ignored_dir(name: str) -> bool:
    """Каталог служебный (VCS, кеши, окружения, сборка) — не обходим его."""
    return name in IGNORE_DIRS or name.endswith(".egg-info")


def is_secret_file(name: str) -> bool:
    """Файл с секретами (``.env``, ``.env.local``…), кроме шаблона ``.env.example``."""
    name = name.casefold()  # .ENV на macOS/Windows — тот же файл
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


def git_dirs(root: Path) -> list[Path]:
    """Каталоги git проекта: ``<root>/.git`` или цель ``gitdir:`` из ``.git``-файла.

    ``<root>/.git`` возвращается раскрытым; ``.git``-файл с ``gitdir:`` бывает у рабочих
    деревьев, сабмодулей и в раскладке с ``.bare``.
    """
    dot_git = root / GIT_DIR_NAME
    try:
        if dot_git.is_dir():
            return [dot_git.resolve()]
        if not dot_git.is_file():
            return []
        for line in dot_git.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("gitdir:"):
                target = Path(line[len("gitdir:") :].strip())
                return [(target if target.is_absolute() else root / target).resolve()]
    except OSError:
        return []
    return []


def looks_like_git_dir(directory: Path, adding: Path | None = None) -> bool:
    """Каталог устроен как каталог git: в нём есть ``HEAD``, ``objects/`` и ``refs/``.

    Так устроены обычный ``.git``, bare-репозиторий и цель ``gitdir:``/симлинка
    вложенного репозитория.

    ``adding`` — путь, который сейчас будет записан: каталог проверяется таким, каким
    станет после записи. Иначе раскладку можно было бы собрать по файлу — ``config``
    первым, ``HEAD`` последним, — и каждая запись по отдельности прошла бы проверку.
    """
    try:
        for name, is_dir in _GIT_LAYOUT:
            entry = directory / name
            if adding is not None and (adding == entry or (is_dir and entry in adding.parents)):
                continue  # запись появится вместе с ``adding``
            exists = entry.is_dir() if is_dir else entry.is_file()
            if not exists:
                return False
    except OSError:
        return False
    return True


def is_vcs_internal(root: Path, path: Path, raw: str = "") -> bool:
    """Путь внутри корня — сам ``.git``/``.jj`` или лежит в служебном каталоге git или jj.

    Проверяются и путь как его прислали (``raw``: ``nested/.git/config``, даже если
    ``nested/.git`` — симлинк, который раскрытие уберёт), и раскрытый: компонент
    ``.git`` или ``.jj`` без учёта регистра (``.GIT`` на macOS/Windows) на любой глубине
    (в ``.jj/`` лежат настройки репозитория jj старых версий и git-хранилище
    неколоцированного репозитория); каталоги по пути от корня, устроенные как каталог git
    (цель симлинка или ``gitdir:`` вложенного репозитория), — в том числе если такими их
    сделает эта запись; каталог git корня. ``.github``, ``.gitignore`` — обычные.
    """
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return False
    raw_parts = Path(raw).parts if raw else ()
    if any(part.casefold() in VCS_DIR_NAMES for part in (*parts, *raw_parts)):
        return True
    if any(path == git_dir or git_dir in path.parents for git_dir in git_dirs(root)):
        return True
    candidates = [path, *path.parents] if path.is_dir() else list(path.parents)
    return any(
        looks_like_git_dir(candidate, adding=path)
        for candidate in candidates
        if candidate != root and root in candidate.parents
    )


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
    last = len(parts) - 1
    sub = ""
    for index, part in enumerate(parts):
        sub = f"{sub}/{part}" if index else part  # первые index + 1 компонентов пути
        is_dir = index < last or is_last_dir
        if stack.is_ignored(sub, is_dir=is_dir):
            return True
        if index < last:
            stack = stack.enter(root, sub)
    return False


def _join_rel(rel_dir: str, name: str) -> str:
    """Путь от корня для записи ``name`` каталога ``rel_dir`` (корень — пустая строка)."""
    return f"{rel_dir}/{name}" if rel_dir else name


class _Walk:
    """Состояние обхода :func:`walk_files`: правила ``.gitignore`` каталогов впереди."""

    def __init__(self, root: Path, start: Path, *, gitignore: bool) -> None:
        self._root = root
        self._gitignore = gitignore
        self._stacks: dict[str, IgnoreStack] = {}
        if gitignore:
            start_rel = self._rel_dir(start)
            self._stacks[start_rel] = stack_for(root, start_rel)

    def files(self, current: Path, dirnames: list[str], filenames: list[str]) -> Iterator[Path]:
        """Файлы каталога ``current``, которые не исключены.

        В ``dirnames`` (на месте — для os.walk) остаются каталоги, в которые спускаться.
        """
        rel_dir = self._rel_dir(current) if self._gitignore else ""
        stack = self._stacks.pop(rel_dir, None) if self._gitignore else None
        self._prune(rel_dir, dirnames, stack)
        for name in sorted(filenames):
            if stack is not None and stack.is_ignored(_join_rel(rel_dir, name), is_dir=False):
                continue
            path = current / name
            if path.is_symlink() and not is_within(self._root, path):
                continue
            yield path

    def _prune(self, rel_dir: str, dirnames: list[str], stack: IgnoreStack | None) -> None:
        """Оставляет в ``dirnames`` каталоги, в которые спускаться, и запоминает их правила."""
        kept = []
        for dirname in sorted(dirnames):
            if is_ignored_dir(dirname):
                continue
            if stack is not None:
                rel = _join_rel(rel_dir, dirname)
                if stack.is_ignored(rel, is_dir=True):
                    continue
                self._stacks[rel] = stack.enter(self._root, rel)
            kept.append(dirname)
        dirnames.clear()
        dirnames.extend(kept)

    def _rel_dir(self, directory: Path) -> str:
        """Каталог относительно корня в POSIX-виде; сам корень — пустая строка."""
        rel = directory.relative_to(self._root).as_posix()
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
    start = root if base is None else base
    # путь вне корня: правилам проекта не к чему относиться
    gitignore = is_within(root, start) and gitignore
    if gitignore:
        start = start.resolve()  # пути от корня считаются по раскрытому пути
    walk = _Walk(root, start, gitignore=gitignore)
    for dirpath, dirnames, filenames in os.walk(start, followlinks=False):
        yield from walk.files(Path(dirpath), dirnames, filenames)


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


class _FileTree:
    """Сборка дерева для :func:`build_file_tree`: отбор записей по уровням и вывод."""

    def __init__(self, root: Path, max_entries: int) -> None:
        self._root = root
        self._budget = max_entries
        self._shown: dict[str, list[_Entry]] = {}  # каталог → показанные записи
        self._hidden: dict[str, int] = {}  # каталог → сколько записей не показано
        self._lines: list[str] = []
        self._truncated = False

    def render(self) -> str:
        """Отбирает записи и выводит дерево строками с отступами."""
        self._select()
        self._truncated = bool(self._hidden)
        self._render_dir("", "")
        if self._truncated:
            self._lines.append(TREE_TRUNCATED)
        return "\n".join(self._lines)

    def _select(self) -> None:
        """Отбирает показываемые записи уровень за уровнем, пока не кончится лимит."""
        level: _Level = [(self._root, "", root_stack(self._root))]
        depth = 0
        while level and self._budget > 0:
            listed = self._list_level(level)
            level = level[: len(listed)]
            cap = self._budget if depth == 0 else TREE_DIR_CAP
            shares = self._shares([min(len(entries), cap) for entries in listed], self._budget)
            level = self._take(level, listed, shares)
            self._budget -= sum(shares)
            depth += 1

    def _list_level(self, level: _Level) -> list[list[_Entry]]:
        """Читает каталоги уровня по порядку, пока их ещё можно раскрыть."""
        # Непустых каталогов сверх лимита не раскрыть всё равно — дальше не читаем
        # (пустые и целиком исключённые лимит не тратят и не обрывают перебор).
        listed: list[list[_Entry]] = []
        nonempty = 0
        for directory, rel_dir, stack in level:
            if nonempty >= self._budget:
                break
            listed.append(self._listing(directory, rel_dir, stack))
            nonempty += bool(listed[-1])
        return listed

    def _take(self, level: _Level, listed: list[list[_Entry]], shares: list[int]) -> _Level:
        """Запоминает показанные и скрытые записи уровня; возвращает следующий уровень."""
        next_level: _Level = []
        for (directory, rel_dir, stack), entries, share in zip(level, listed, shares, strict=True):
            self._shown[rel_dir] = entries[:share]
            if len(entries) > share:
                self._hidden[rel_dir] = len(entries) - share
            for name, is_dir in self._shown[rel_dir]:
                if is_dir:
                    rel = _join_rel(rel_dir, name)
                    next_level.append((directory / name, rel, stack.enter(self._root, rel)))
        return next_level

    def _shares(self, sizes: list[int], budget: int) -> list[int]:
        """Делит ``budget`` между каталогами поровну, отдавая недобранное остальным."""
        shares = [0 for _ in sizes]
        pending = [index for index, size in enumerate(sizes) if size]
        while pending and budget > 0:
            fair = max(budget // len(pending), 1)
            still = []
            for index in pending:
                take = min(sizes[index] - shares[index], fair, budget)
                shares[index] += take
                budget -= take
                if shares[index] < sizes[index]:
                    still.append(index)
            pending = still
        return shares

    def _listing(self, directory: Path, rel_dir: str, stack: IgnoreStack) -> list[_Entry]:
        """Записи каталога для дерева: каталоги первыми, затем по имени."""
        try:
            entries = sorted(
                directory.iterdir(),
                key=lambda entry: (not self._is_real_dir(entry), entry.name.lower()),
            )
        except OSError:
            return []
        kept: list[_Entry] = []
        for entry in entries:
            is_dir = self._is_real_dir(entry)
            rel = _join_rel(rel_dir, entry.name)
            if is_dir and is_ignored_dir(entry.name):
                continue
            if stack.is_ignored(rel, is_dir=is_dir):
                continue
            kept.append((entry.name, is_dir))
        return kept

    def _is_real_dir(self, path: Path) -> bool:
        # симлинки на каталоги не раскрываем (возможны циклы и выход из корня)
        return path.is_dir() and not path.is_symlink()

    def _render_dir(self, rel_dir: str, prefix: str) -> None:
        """Выводит записи каталога ``rel_dir`` и, вглубь, раскрытых подкаталогов."""
        for name, is_dir in self._shown.get(rel_dir, []):
            if not is_dir:
                self._lines.append(f"{prefix}{name}")
                continue
            rel = _join_rel(rel_dir, name)
            if rel in self._shown:
                self._lines.append(f"{prefix}{name}/")
                self._render_dir(rel, f"{prefix}  ")
            else:
                self._truncated = True
                self._lines.append(f"{prefix}{name}/ …")
        hidden = self._hidden.get(rel_dir)
        if hidden:
            self._lines.append(f"{prefix}… {hidden} more")


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
    return _FileTree(root.resolve(), max_entries).render()
