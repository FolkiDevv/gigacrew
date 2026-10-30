"""Скиллы: готовые инструкции для типовых задач, ``skills/<имя>/SKILL.md``.

Источники (по возрастанию приоритета — проектный скилл перекрывает
пользовательский с тем же именем):

* ``~/.gigacode/skills/`` — скиллы пользователя, общие для всех проектов;
* ``<проект>/.gigacode/skills/`` — скиллы проекта (коммитятся вместе с ним).

``SKILL.md`` начинается с frontmatter::

    ---
    name: release
    description: Как выпустить новую версию пакета
    ---
    Тело — инструкция для агента.

Без ``name`` имя берётся из имени каталога. Имя должно годиться для команды REPL
(``/release``), иначе скилл пропускается. Каталог скилла и ``SKILL.md`` и после раскрытия
симлинков должны лежать в своём каталоге скиллов, а скиллы проекта — ещё и в проекте:
иначе репозиторий подсунул бы модели под видом скилла любой файл пользователя
(``~/.ssh/…``) — инструмент ``skill`` читает без подтверждения. Пользователь вызывает скилл командой
``/<имя> [аргументы]``, модель — инструментом ``skill`` (список скиллов с
описаниями — в системном промпте).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from gigacrew.project.files import is_within
from gigacrew.project.workspace import SKILLS_DIR_NAME, Workspace, user_data_dir

SKILL_FILE = "SKILL.md"
MAX_SKILL_BYTES = 200_000  # SKILL.md больше — не скилл, а случайный файл
# Имя команды REPL — общее правило для скиллов и cli.commands: иначе скилл виден в
# списках, но «/имя» не распознаётся как команда.
COMMAND_NAME_PATTERN = r"[a-zA-Zа-яА-ЯёЁ][\w-]*"
_NAME_RE = re.compile(COMMAND_NAME_PATTERN)
_KEY_RE = re.compile(r"([A-Za-z_][\w-]*)\s*:\s?(.*)")

SkillSource = Literal["project", "user"]


@dataclass(frozen=True)
class Skill:
    """Скилл: имя команды, описание, путь к ``SKILL.md`` и источник."""

    name: str
    description: str
    path: Path  # SKILL.md
    source: SkillSource = "project"

    @property
    def directory(self) -> Path:
        """Каталог скилла (в нём ``SKILL.md`` и файлы скилла)."""
        return self.path.parent

    @property
    def command(self) -> str:
        """Команда REPL: ``/<имя>``."""
        return f"/{self.name}"

    def body(self) -> str:
        """Текст инструкции — без frontmatter. Ошибка чтения — OSError/UnicodeDecodeError."""
        return split_frontmatter(self.path.read_text(encoding="utf-8"))[1].strip()


def split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Разбирает frontmatter (``---`` … ``---`` в начале файла): (поля, тело).

    Минимальное подмножество YAML: ``ключ: значение`` (в кавычках или без) и
    блочные значения ``|``/``>`` со строками продолжения, сдвинутыми отступом.
    Нет frontmatter — пустые поля и весь текст как тело.
    """
    text = text.removeprefix("﻿")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    try:
        end = next(index for index in range(1, len(lines)) if lines[index].strip() == "---")
    except StopIteration:
        return {}, text
    return _parse_fields(lines[1:end]), "\n".join(lines[end + 1 :])


def _parse_fields(header: list[str]) -> dict[str, str]:
    """Строки frontmatter между ``---`` → поля (ключи — в нижнем регистре)."""
    fields: dict[str, str] = {}
    pos = 0
    while pos < len(header):
        match = _KEY_RE.fullmatch(header[pos].rstrip())
        pos += 1
        if match is None:
            continue
        block_end = _continuation_end(header, pos)
        block = [line.strip() for line in header[pos:block_end]]
        pos = block_end
        fields[match.group(1).lower()] = _field_text(match.group(2).strip(), block)
    return fields


def _continuation_end(header: list[str], start: int) -> int:
    """Конец строк продолжения с ``start``: пустых или сдвинутых отступом."""
    end = start
    while end < len(header):
        line = header[end]
        if line.strip() and line[:1] not in " \t":
            break
        end += 1
    return end


def _field_text(inline: str, block: list[str]) -> str:
    """Значение поля: текст после ``:`` и строки продолжения ``block`` (без отступа)."""
    text = inline
    if inline[:1] in "|>":
        joiner = "\n" if inline[:1] == "|" else " "
        text = joiner.join(block).strip()
    elif block:  # значение, перенесённое на следующие строки
        text = " ".join([inline, *block]).strip()
    return _unquote(text)


def _unquote(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    return text


def parse_skill(path: Path, source: SkillSource = "project") -> Skill | None:
    """Скилл из ``SKILL.md``; None — файл нечитаем, слишком велик или имя не годится."""
    try:
        if path.stat().st_size > MAX_SKILL_BYTES:
            return None
        fields, _body = split_frontmatter(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return None
    name = fields.get("name") or path.parent.name
    if not _NAME_RE.fullmatch(name):
        return None
    description = " ".join(fields.get("description", "").split())
    return Skill(name=name, description=description, path=path, source=source)


def _scan(directory: Path, source: SkillSource, within: Path | None = None) -> list[Skill]:
    """Скиллы ``directory``; ``within`` — где ещё должен лежать раскрытый каталог (проект).

    Сам каталог скиллов может быть симлинком (dotfiles), но скилл, раскрытый за его
    пределы, — не скилл.
    """
    try:
        root = directory.resolve()
        entries = sorted(directory.iterdir())
    except OSError:  # каталога нет или он нечитаем
        return []
    if within is not None and not is_within(within, root):
        return []
    skills = []
    for entry in entries:
        path = entry / SKILL_FILE
        if is_within(root, entry) and is_within(root, path) and path.is_file():
            skill = parse_skill(path, source)
            if skill is not None:
                skills.append(skill)
    return skills


def user_skills_dir() -> Path:
    """``~/.gigacode/skills``."""
    return user_data_dir() / SKILLS_DIR_NAME


def discover_skills(workspace: Workspace, user_dir: Path | None = None) -> list[Skill]:
    """Скиллы пользователя и проекта (проектный важнее), по имени без учёта регистра.

    У недоверенного проекта (``workspace.untrusted``) — только скиллы пользователя.
    """
    user_dir = user_skills_dir() if user_dir is None else user_dir
    found: dict[str, Skill] = {}
    project = (
        [] if workspace.untrusted else _scan(workspace.skills_dir, "project", within=workspace.root)
    )
    for candidate in (*_scan(user_dir, "user"), *project):
        found[candidate.name.casefold()] = candidate
    return sorted(found.values(), key=lambda skill: skill.name.casefold())


def find_skill(skills: list[Skill], name: str) -> Skill | None:
    """Скилл по имени (с ведущим ``/`` или без, без учёта регистра)."""
    key = name.removeprefix("/").casefold()
    return next((skill for skill in skills if skill.name.casefold() == key), None)


class SkillCatalog:
    """Найденные скиллы; :meth:`refresh` перечитывает каталоги.

    Конструктор не трогает файловую систему: список пуст до первого ``refresh``.
    """

    def __init__(self, workspace: Workspace, user_dir: Path | None = None):
        """Каталог скиллов проекта ``workspace`` и пользователя (``user_dir``)."""
        self._workspace = workspace
        self._user_dir = user_dir
        self._skills: list[Skill] = []

    def refresh(self) -> None:
        """Перечитать скиллы с диска."""
        self._skills = discover_skills(self._workspace, self._user_dir)

    def get(self, name: str) -> Skill | None:
        """Скилл по имени (с ``/`` или без, без учёта регистра); None — нет такого."""
        return find_skill(self._skills, name)

    def __iter__(self) -> Iterator[Skill]:
        """Скиллы по имени."""
        return iter(self._skills)

    def __len__(self) -> int:
        """Число скиллов."""
        return len(self._skills)
