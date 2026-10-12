"""Рабочее пространство проекта и служебная папка ``.gigacrew/``.

``.gigacrew/`` в корне проекта — место для данных агента (индекс проекта, чаты,
планы).
Папка создаётся лениво, первым компонентом, которому нужно туда писать
(:meth:`Workspace.ensure_data_dir`), и сразу получает собственный ``.gitignore``
со звёздочкой, чтобы её содержимое никогда не попадало в репозиторий.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DATA_DIR_NAME = ".gigacrew"
_GITIGNORE_BODY = "# Создано gigacrew: рабочие данные агента не коммитятся.\n*\n"


@dataclass(frozen=True)
class Workspace:
    """Корень проекта и производные пути. Конструктор не трогает файловую систему."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())

    @property
    def data_dir(self) -> Path:
        """``<root>/.gigacrew`` (может ещё не существовать)."""
        return self.root / DATA_DIR_NAME

    @property
    def chats_dir(self) -> Path:
        """Каталог сохранённых чатов."""
        return self.data_dir / "chats"

    @property
    def plans_dir(self) -> Path:
        """Каталог планов (режим планирования): ``<id чата>.md``."""
        return self.data_dir / "plans"

    @property
    def index_dir(self) -> Path:
        """Каталог индекса проекта."""
        return self.data_dir / "index"

    def is_data_path(self, path: Path) -> bool:
        """Лежит ли (уже разрешённый) путь внутри ``.gigacrew/``.

        Сравнение без учёта регистра: на macOS/Windows ``.GIGACREW`` — та же папка.
        """
        try:
            parts = Path(path).relative_to(self.root).parts
        except ValueError:
            return False
        return bool(parts) and parts[0].casefold() == DATA_DIR_NAME.casefold()

    def in_git_repo(self) -> bool:
        """Проект внутри git-репозитория: ``.git`` (каталог или файл рабочего дерева)
        в корне или выше. Только проверка ФС — git не запускается."""
        return any((d / ".git").exists() for d in (self.root, *self.root.parents))

    def ensure_data_dir(self) -> Path:
        """Создаёт ``.gigacrew/`` с ``.gitignore`` (идемпотентно) и возвращает путь.

        Существующий ``.gitignore`` не перезаписывается — пользователь мог его изменить.
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        gitignore = self.data_dir / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text(_GITIGNORE_BODY, encoding="utf-8")
        return self.data_dir
