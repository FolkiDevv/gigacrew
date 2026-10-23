"""Файлы инструкций проекта, которые агент держит в контексте.

Пользователь кладёт в проект соглашения, команды сборки и заметки — они попадают
в системный промпт. Имена (:data:`INSTRUCTION_FILES`) — по возрастанию приоритета:

* ``AGENTS.md`` — общий формат инструкций проекта;
* ``GIGACREW.md`` — инструкции именно для gigacrew;
* ``GIGACREW.local.md`` — личные заметки (обычно в ``.gitignore``).

Файлы корня проекта входят в системный промпт всегда. Те же имена в подкаталогах
подключаются, когда агент начинает работать с файлами каталога
(:class:`NestedInstructions`): инструкции ближайшего к файлу каталога важнее общих.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from gigacrew.errors import SandboxError
from gigacrew.project.files import is_excluded, is_within
from gigacrew.security import resolve_in_root

INSTRUCTION_FILES: tuple[str, ...] = (
    "AGENTS.md",
    "GIGACREW.md",
    "GIGACREW.local.md",
)
MAX_INSTRUCTION_CHARS = 20_000  # один файл
MAX_DIRECTORY_CHARS = 30_000  # все файлы одного каталога вместе


@dataclass(frozen=True)
class InstructionFile:
    """Прочитанный файл инструкций: путь от корня, текст и признак обрезки."""

    name: str  # путь от корня проекта (POSIX): "AGENTS.md", "src/api/AGENTS.md"
    content: str
    truncated: bool = False


def load_instructions(
    root: Path,
    names: Sequence[str] = INSTRUCTION_FILES,
    max_chars: int = MAX_INSTRUCTION_CHARS,
    *,
    directory: Path | None = None,
    max_total: int = MAX_DIRECTORY_CHARS,
) -> list[InstructionFile]:
    """Читает существующие файлы инструкций каталога ``directory`` в порядке ``names``.

    Каталог по умолчанию — корень. Пустые и нечитаемые файлы пропускаются, симлинки
    за пределы корня и повторы одного файла под разными именами (симлинк
    ``GIGACREW.md → AGENTS.md``) — тоже. Файл длиннее ``max_chars`` обрезается; если
    все вместе длиннее ``max_total``, место сначала получают более приоритетные
    (поздние в ``names``).
    """
    root = root.resolve()
    base = root if directory is None else directory
    return _fit_budget(_read_files(root, base, names), max_chars, max_total)


def _read_files(root: Path, base: Path, names: Sequence[str]) -> list[tuple[str, str]]:
    """Непустые файлы инструкций каталога ``base``: (путь от корня, текст) в порядке ``names``."""
    found: list[tuple[str, str]] = []
    seen: set[Path] = set()
    for name in names:
        path = base / name
        if not path.is_file() or not is_within(root, path):
            continue
        real = path.resolve()
        if real in seen:
            continue
        try:
            text = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            continue
        if text:
            seen.add(real)
            found.append((path.relative_to(root).as_posix(), text))
    return found


def _fit_budget(
    found: list[tuple[str, str]],
    max_chars: int,
    max_total: int,
) -> list[InstructionFile]:
    """Обрезка по лимитам; общий бюджет сначала получают поздние (приоритетные) файлы."""
    budget = max_total
    fitted: list[InstructionFile] = []
    for name, text in reversed(found):
        limit = max(min(max_chars, budget), 0)
        truncated = len(text) > limit
        content = text[:limit] if truncated else text
        budget -= len(content)
        fitted.append(InstructionFile(name=name, content=content, truncated=truncated))
    fitted.reverse()
    return fitted


class NestedInstructions:
    """Инструкции подкаталогов, с которыми агент уже работал в этом диалоге.

    :meth:`add_paths` получает пути из вызовов инструментов и ищет файлы инструкций
    в каталогах от корня (не включая — его инструкции и так в промпте) до каталога
    пути, снаружи вовнутрь. Каждый каталог проверяется один раз. Служебные и
    исключённые ``.gitignore`` каталоги (``node_modules``, ``.gigacode``…) не
    рассматриваются: чужие инструкции из зависимостей агенту не нужны. Сами файлы
    инструкций при этом могут быть в ``.gitignore`` (``GIGACREW.local.md``).

    Конструктор не трогает файловую систему.
    """

    def __init__(self, root: Path):
        """Инструкции подкаталогов проекта с корнем ``root``."""
        self._root = root
        self._checked: set[str] = set()
        self._files: list[InstructionFile] = []

    @property
    def files(self) -> tuple[InstructionFile, ...]:
        """Найденные инструкции — в порядке обнаружения, внешние каталоги раньше."""
        return tuple(self._files)

    def reset(self) -> None:
        """Забыть найденные инструкции и проверенные каталоги (смена диалога)."""
        self._checked.clear()
        self._files.clear()

    def add_paths(self, paths: Iterable[str]) -> list[InstructionFile]:
        """Ищет инструкции для ``paths``; возвращает только новые файлы."""
        root = self._root.resolve()
        new: list[InstructionFile] = []
        for raw in paths:
            directory = _directory_of(root, raw)
            if directory is None:
                continue
            parts = directory.relative_to(root).parts
            for depth in range(1, len(parts) + 1):
                rel = "/".join(parts[:depth])
                if rel in self._checked:
                    continue
                self._checked.add(rel)
                current = root.joinpath(*parts[:depth])
                if is_excluded(root, current):
                    continue
                new += load_instructions(root, directory=current)
        self._files += new
        return new


def _directory_of(root: Path, raw: str) -> Path | None:
    """Каталог пути внутри корня: сам путь, если это каталог, иначе родитель."""
    if not raw or not raw.strip():
        return None
    try:
        path = resolve_in_root(root, raw.strip())
    except (SandboxError, OSError, ValueError):
        return None
    directory = path if path.is_dir() else path.parent
    if root not in directory.parents:
        return None  # сам корень: его инструкции и так в системном промпте
    return directory
