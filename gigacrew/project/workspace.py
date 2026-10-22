"""Рабочее пространство проекта и служебная папка ``.gigacode/``.

``.gigacode/`` в корне проекта — место для данных агента (индекс проекта, планы,
история ввода), скиллов (``skills/``, см. :mod:`gigacrew.project.skills`) и
настроек (``settings.json`` и личный ``settings.local.json``, см.
:mod:`gigacrew.project.settings`).
Папка создаётся лениво, первым компонентом, которому нужно туда писать
(:meth:`Workspace.ensure_data_dir`), и сразу получает собственный ``.gitignore``:
рабочие данные агента в репозиторий не попадают, а скиллы — коммитятся.
Пользовательская папка ``~/.gigacode`` (:func:`user_data_dir`) — замеры окон
моделей, скиллы и настройки, общие для всех проектов, а в ``projects/<ключ>/`` —
данные проекта, которым нельзя прийти вместе с репозиторием: разрешения, журнал
решений (:mod:`gigacrew.agent.decisions`) и чаты (:attr:`Workspace.chats_dir`).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

DATA_DIR_NAME = ".gigacode"
# Данные проекта в каталоге пользователя: ``~/.gigacode/projects/<ключ>/``.
PROJECTS_DIR = "projects"
SKILLS_DIR_NAME = "skills"
HISTORY_FILE_NAME = "history"
SETTINGS_FILE_NAME = "settings.json"
LOCAL_SETTINGS_FILE_NAME = "settings.local.json"
# Игнорируются только рабочие данные агента и личные настройки — не вся папка: skills/ и
# settings.json коммитятся. chats/ — прежнее место чатов: то, что не удалось перенести в
# ~/.gigacode/projects/, тоже не должно попасть в коммит.
GITIGNORED_DATA: tuple[str, ...] = (
    "/chats/",
    "/plans/",
    "/index/",
    f"/{HISTORY_FILE_NAME}",
    f"/{LOCAL_SETTINGS_FILE_NAME}",
)
# Что gigacrew пишет в .gigacode/ сам (ty.log — журнал gigacrew.project.semantic).
_OWN_ENTRIES: tuple[str, ...] = (
    "chats",
    "plans",
    "index",
    HISTORY_FILE_NAME,
    ".gitignore",
    "ty.log",
)
_GITIGNORE_BODY = (
    "# Создано gigacrew: рабочие данные агента и settings.local.json не коммитятся,\n"
    "# скиллы (skills/) и settings.json — коммитятся.\n"
    + "".join(f"{pattern}\n" for pattern in GITIGNORED_DATA)
)


def user_data_dir() -> Path:
    """``~/.gigacode`` — данные пользователя, общие для всех проектов."""
    return Path.home() / DATA_DIR_NAME


def user_settings_file() -> Path:
    """``~/.gigacode/settings.json`` — настройки пользователя (хуки)."""
    return user_data_dir() / SETTINGS_FILE_NAME


def project_key(root: Path) -> str:
    """Ключ проекта: путь корня; рабочее дерево git — путь основного репозитория."""
    return hashlib.sha256(str(project_key_root(root)).encode("utf-8")).hexdigest()[:16]


def project_key_root(root: Path) -> Path:
    """Путь, от которого считается ключ: у рабочего дерева git — основной репозиторий."""
    root = Path(os.path.normcase(str(Path(root).resolve())))
    return _main_worktree(root) or root


def _main_worktree(root: Path) -> Path | None:
    """Основной репозиторий связанного рабочего дерева (``root/.git`` — файл) или None.

    Ссылке верим, только если её подтверждает git: каталог ``<gitdir>`` есть и его файл
    ``gitdir`` указывает обратно на ``root/.git``. Иначе скачанный проект с поддельным
    ``.git`` делил бы с чужим проектом его память решений.
    """
    dot_git = root / ".git"
    try:
        if not dot_git.is_file():
            return None
        for line in dot_git.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith("gitdir:"):
                continue
            target = Path(line[len("gitdir:") :].strip())
            gitdir = (target if target.is_absolute() else root / target).resolve()
            # <основной>/.git/worktrees/<имя>
            if gitdir.parent.name != "worktrees" or gitdir.parent.parent.name != ".git":
                return None
            back = Path((gitdir / "gitdir").read_text(encoding="utf-8").strip())
            back = (back if back.is_absolute() else gitdir / back).resolve()
            if os.path.normcase(str(back)) != os.path.normcase(str(dot_git.resolve())):
                return None
            return gitdir.parent.parent.parent
    except (OSError, ValueError):
        return None
    return None


@dataclass(frozen=True)
class Workspace:
    """Корень проекта и производные пути. Конструктор не трогает файловую систему."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())

    @property
    def data_dir(self) -> Path:
        """``<root>/.gigacode`` (может ещё не существовать)."""
        return self.root / DATA_DIR_NAME

    @property
    def chats_dir(self) -> Path:
        """Каталог сохранённых чатов: ``~/.gigacode/projects/<ключ проекта>/chats``.

        Не в проекте: чат из репозитория при продолжении стал бы «запросами пользователя»
        для рецензента команд и выбрал бы модель. Ключ — тот же, что у разрешений
        (рабочие деревья одного репозитория git делят и чаты).
        """
        return user_data_dir() / PROJECTS_DIR / project_key(self.root) / "chats"

    @property
    def legacy_chats_dir(self) -> Path:
        """Прежний каталог чатов в проекте (``.gigacode/chats``) — только для переноса."""
        return self.data_dir / "chats"

    @property
    def plans_dir(self) -> Path:
        """Каталог планов (режим планирования): ``<id чата>.md``."""
        return self.data_dir / "plans"

    @property
    def index_dir(self) -> Path:
        """Каталог индекса проекта."""
        return self.data_dir / "index"

    @property
    def history_file(self) -> Path:
        """История ввода REPL."""
        return self.data_dir / HISTORY_FILE_NAME

    @property
    def skills_dir(self) -> Path:
        """Скиллы проекта: ``<имя>/SKILL.md``."""
        return self.data_dir / SKILLS_DIR_NAME

    @property
    def settings_file(self) -> Path:
        """Настройки проекта (коммитятся): ``.gigacode/settings.json``."""
        return self.data_dir / SETTINGS_FILE_NAME

    @property
    def local_settings_file(self) -> Path:
        """Личные настройки в проекте (не коммитятся): ``.gigacode/settings.local.json``."""
        return self.data_dir / LOCAL_SETTINGS_FILE_NAME

    def is_data_path(self, path: Path) -> bool:
        """Лежит ли (уже разрешённый) путь внутри ``.gigacode/``.

        Сравнение без учёта регистра: на macOS/Windows ``.GIGACODE`` — та же папка.
        """
        try:
            parts = Path(path).relative_to(self.root).parts
        except ValueError:
            return False
        return bool(parts) and parts[0].casefold() == DATA_DIR_NAME.casefold()

    def in_git_repo(self) -> bool:
        """Проект внутри git-репозитория: ``.git`` (каталог или файл рабочего дерева)
        в корне или выше. Только проверка ФС — git не запускается.
        """
        return any((d / ".git").exists() for d in (self.root, *self.root.parents))

    def ensure_data_dir(self) -> Path:
        """Создаёт ``.gigacode/`` с ``.gitignore`` (идемпотентно) и возвращает путь.

        Существующий ``.gitignore`` не перезаписывается — пользователь мог его изменить.
        Через эту проверку проходит всё, что gigacrew пишет в папку сам (история ввода,
        чаты, индекс, планы, журнал ty): если ``.gigacode`` или одна из этих записей —
        симлинк, — ``OSError``, и запись не идёт. Репозиторий может принести такой симлинк
        (git восстанавливает их при клоне), и запись ушла бы за пределы проекта — например,
        введённые запросы дописывались бы в чужой файл.
        """
        data_dir = self.data_dir
        if data_dir.is_symlink():
            raise OSError(f"{data_dir} — симлинк: gigacrew не пишет по нему")
        data_dir.mkdir(parents=True, exist_ok=True)
        for name in _OWN_ENTRIES:
            if (data_dir / name).is_symlink():
                raise OSError(f"{data_dir / name} — симлинк: gigacrew не пишет по нему")
        try:
            fd = os.open(data_dir / ".gitignore", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            return data_dir
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(_GITIGNORE_BODY)
        return data_dir
