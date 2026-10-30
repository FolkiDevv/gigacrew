"""Уровни команды и вердикт классификатора."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from types import MappingProxyType

__all__ = ["CommandLevel", "Part", "Verdict", "worst"]

# Версия правил: в журнале решений — чтобы переоценить старые записи после правок.
RULES_VERSION = 4


class CommandLevel(IntEnum):
    """Что команда может сделать. Составная команда — максимум своих частей.

    Не путать с :class:`gigacrew.security.RiskLevel` — тот решает, спрашивать ли;
    уровень команды переводится в него в :attr:`Verdict.risk_level`.
    """

    READ = 0  # только читает внутри проекта — выполняется без вопроса в любом режиме
    EXEC = 1  # исполняет код проекта или инструментов (тесты, сборка, линтеры, скрипты)
    WRITE = 2  # меняет файлы проекта, локальное окружение (в т.ч. установка пакетов)
    DANGER = 3  # необратимо, привилегии, вне проекта, секреты, публикация, сеть

    @property
    def label(self) -> str:
        """Название уровня для пользователя."""
        return _LABELS[self]


_LABELS = MappingProxyType({
    CommandLevel.READ: "только чтение",
    CommandLevel.EXEC: "выполняет код проекта",
    CommandLevel.WRITE: "меняет файлы",
    CommandLevel.DANGER: "опасно",
})


@dataclass(frozen=True)
class Part:
    """Оценка одной простой команды.

    ``signature`` — программа с подкомандой у команд уровня EXEC (``pytest``,
    ``npm test``, ``make test``; ``python -m pytest`` и ``uv run pytest`` — тоже ``pytest``):
    по ней работает разрешение «эта программа с любыми аргументами». ``edits`` — запись
    только в обычные файлы проекта (как правка файла инструментом): в режиме «авто»
    не спрашивается. ``prelude`` — подготовка окружения (активация venv проекта), не
    мешает совпадению с разрешением.
    """

    level: CommandLevel | None
    reason: str
    signature: str | None = None
    edits: bool = False
    prelude: bool = False
    text: str = ""  # простая команда как написана (для точного разрешения)


@dataclass(frozen=True)
class Verdict:
    """Решение классификатора о команде целиком.

    ``level`` None — «не знаю» (данных недостаточно): решает пользователь, его выбор
    сохраняется. ``cwd`` — рабочий каталог от корня (параметр инструмента и ведущие
    ``cd X &&``), ``command`` — команда без ведущих ``cd``. ``parts`` — оценки простых
    команд; ``composable`` — команда состоит только из частей, соединённых ``&&``,
    ``||``, ``;`` и конвейерами, без записи через перенаправления и без конструкций,
    которые не удалось оценить, и все части выполняются в ``cwd`` (не в подоболочке
    с ``cd``): такую можно сверять с сохранёнными разрешениями по частям.
    ``redirects_edit`` — запись через перенаправления (если есть) только в обычные файлы
    проекта: ``> .env`` — не правка, как и ``cp a.txt .env``.
    """

    level: CommandLevel | None
    reason: str
    cwd: str = "."
    command: str = ""
    parts: tuple[Part, ...] = field(default_factory=tuple)
    composable: bool = False
    redirects_edit: bool = True

    @property
    def known(self) -> bool:
        """Уровень известен (не «не знаю»)."""
        return self.level is not None

    @property
    def edits_only(self) -> bool:
        """Только правки обычных файлов проекта (как write_file/edit_file)."""
        return (
            self.level is CommandLevel.WRITE
            and self.redirects_edit
            and bool(self.parts)
            and all(part.level is CommandLevel.READ or part.edits for part in self.parts)
        )

    @property
    def signatures(self) -> tuple[str, ...]:
        """Сигнатуры частей уровня EXEC (без подготовки окружения), по порядку."""
        out: list[str] = []
        for part in self.parts:
            if (
                part.level is CommandLevel.EXEC
                and not part.prelude
                and part.signature
                and part.signature not in out
            ):
                out.append(part.signature)
        return tuple(out)


def worst(*levels: CommandLevel | None) -> CommandLevel | None:
    """Худший уровень; «не знаю» (None) хуже всего, кроме явной опасности."""
    if any(lv is CommandLevel.DANGER for lv in levels):
        return CommandLevel.DANGER
    if any(lv is None for lv in levels):
        return None
    return max((lv for lv in levels if lv is not None), default=CommandLevel.READ)
