"""Рабочее пространство проекта и служебная папка ``.gigacode/``.

``.gigacode/`` в корне проекта — место для данных агента (индекс проекта, планы,
история ввода, временные файлы модели в ``tmp/``), скиллов (``skills/``, см.
:mod:`gigacrew.project.skills`) и настроек (``settings.json`` и личный
``settings.local.json``, см. :mod:`gigacrew.project.settings`).
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
# Временные файлы модели (черновые скрипты, сохранённый вывод команд) — вместо /tmp:
# запись сюда — обычная правка, а запись вне проекта всегда подтверждает пользователь.
TEMP_DIR_NAME = "tmp"
# Игнорируются только рабочие данные агента и личные настройки — не вся папка: skills/ и
# settings.json коммитятся. chats/ — прежнее место чатов: то, что не удалось перенести в
# ~/.gigacode/projects/, тоже не должно попасть в коммит.
GITIGNORED_DATA: tuple[str, ...] = (
    "/chats/",
    "/plans/",
    "/index/",
    f"/{TEMP_DIR_NAME}/",
    f"/{HISTORY_FILE_NAME}",
    f"/{LOCAL_SETTINGS_FILE_NAME}",
)
# Что gigacrew пишет в .gigacode/ сам (ty.log — журнал gigacrew.project.semantic).
_OWN_ENTRIES: tuple[str, ...] = (
    "chats",
    "plans",
    "index",
    TEMP_DIR_NAME,
    HISTORY_FILE_NAME,
    ".gitignore",
    "ty.log",
)
_GITIGNORE_BODY = (
    "# Создано gigacrew: рабочие данные агента и settings.local.json не коммитятся,\n"
    "# скиллы (skills/) и settings.json — коммитятся.\n"
    + "".join(f"{pattern}\n" for pattern in GITIGNORED_DATA)
)
# Свой .gitignore у tmp/: в прежних проектах .gigacode/.gitignore без /tmp/ (его не
# перезаписываем), а временные файлы не должны попасть в коммит.
_TEMP_GITIGNORE_BODY = "# Создано gigacrew: временные файлы агента не коммитятся.\n*\n"
_GITIGNORE_MODE = 0o644
_PROJECT_KEY_LENGTH = 16  # символов hex-дайджеста sha256


def user_data_dir() -> Path:
    """``~/.gigacode`` — данные пользователя, общие для всех проектов."""
    return Path.home() / DATA_DIR_NAME


def user_settings_file() -> Path:
    """``~/.gigacode/settings.json`` — настройки пользователя (хуки)."""
    return user_data_dir() / SETTINGS_FILE_NAME


def project_key(root: Path) -> str:
    """Ключ проекта: путь корня; рабочее дерево git — путь основного репозитория."""
    digest = hashlib.sha256(str(project_key_root(root)).encode("utf-8")).hexdigest()
    return digest[:_PROJECT_KEY_LENGTH]


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
        return _read_main_worktree(root, dot_git)
    except (OSError, ValueError):
        return None


def _read_main_worktree(root: Path, dot_git: Path) -> Path | None:
    """Тело :func:`_main_worktree`: ошибки ФС и путей (OSError, ValueError) не перехватывает."""
    if not dot_git.is_file():
        return None
    for line in dot_git.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("gitdir:"):
            return _confirmed_main_worktree(root, dot_git, line)
    return None


def _confirmed_main_worktree(root: Path, dot_git: Path, line: str) -> Path | None:
    """Основной репозиторий по строке ``gitdir:`` из ``root/.git``, если git её подтверждает."""
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


@dataclass(frozen=True)
class Workspace:
    """Корень проекта и производные пути. Конструктор не трогает файловую систему."""

    root: Path

    def __post_init__(self) -> None:
        """Приводит корень к абсолютному пути без симлинков."""
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

    def is_temp_path(self, path: Path) -> bool:
        """Лежит ли (уже разрешённый) путь в ``.gigacode/tmp/`` (или это она сама).

        Единственная часть ``.gigacode/``, куда модель пишет как в обычные файлы проекта.
        Сравнение точное, в отличие от :meth:`is_data_path`: на ФС с учётом регистра
        ``.gigacode/TMP`` — другая папка, её не игнорирует git. Свой ``.gitignore`` папки
        модели не принадлежит: без него временные файлы попали бы в коммит.
        """
        temp_dir = self.data_dir / TEMP_DIR_NAME
        path = Path(path)
        if path == temp_dir / ".gitignore":
            return False
        return path == temp_dir or temp_dir in path.parents

    def in_git_repo(self) -> bool:
        """Проект внутри git-репозитория: ``.git`` в корне или выше.

        ``.git`` — каталог или файл рабочего дерева. Только проверка ФС — git не запускается.
        """
        return any((directory / ".git").exists() for directory in (self.root, *self.root.parents))

    def ensure_data_dir(self) -> Path:
        """Создаёт ``.gigacode/`` с ``.gitignore`` (идемпотентно) и возвращает путь.

        Существующий ``.gitignore`` не перезаписывается — пользователь мог его изменить.
        Через эту проверку проходит всё, что gigacrew пишет в папку сам (история ввода,
        чаты, индекс, планы, журнал ty, tmp/): если ``.gigacode`` или одна из этих записей —
        симлинк, — ``OSError``, и запись не идёт. Репозиторий может принести такой симлинк
        (git восстанавливает их при клоне), и запись ушла бы за пределы проекта — например,
        введённые запросы дописывались бы в чужой файл.
        """
        data_dir = self.data_dir
        if data_dir.is_symlink():
            raise OSError(f"{data_dir} — симлинк: gigacrew не пишет по нему")
        data_dir.mkdir(parents=True, exist_ok=True)
        for name in _OWN_ENTRIES:
            entry = data_dir / name
            if entry.is_symlink():
                raise OSError(f"{entry} — симлинк: gigacrew не пишет по нему")
        _create_gitignore(data_dir, _GITIGNORE_BODY)
        return data_dir

    def ensure_temp_dir(self) -> Path:
        """Создаёт ``.gigacode/tmp/`` (временные файлы модели) со своим ``.gitignore``.

        Идемпотентно; возвращает путь папки.

        Симлинк на месте ``tmp`` — ``OSError`` (проверяет :meth:`ensure_data_dir`).
        """
        temp_dir = self.ensure_data_dir() / TEMP_DIR_NAME
        temp_dir.mkdir(exist_ok=True)
        _create_gitignore(temp_dir, _TEMP_GITIGNORE_BODY)
        return temp_dir


def _create_gitignore(directory: Path, body: str) -> None:
    """Создаёт ``directory/.gitignore``, если его нет; существующий не трогает.

    ``O_EXCL`` не идёт по симлинку, даже висячему: на его месте — отказ, как у готового файла.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(directory / ".gitignore", flags, _GITIGNORE_MODE)
    except FileExistsError:
        return
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(body)
