"""Инструмент ``skill``: загрузить инструкцию скилла (:mod:`gigacrew.project.skills`).

Список скиллов с описаниями модель видит в системном промпте; подходящий она
загружает этим инструментом и следует ему. ``file`` — дополнительный файл из
каталога скилла (справка, шаблон): скиллы пользователя лежат в ``~/.gigacode``,
вне корня проекта, и read_file до них не дотянется. Выйти за пределы каталога
скилла нельзя, в том числе симлинком: сам каталог и ``SKILL.md`` проверяет поиск
скиллов, файлы — :func:`~gigacrew.security.resolve_in_root`.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from gigacrew.errors import SandboxError
from gigacrew.project.files import is_within
from gigacrew.project.skills import (
    SKILL_FILE,
    Skill,
    discover_skills,
    find_skill,
    split_frontmatter,
)
from gigacrew.security import resolve_in_root
from gigacrew.tools.base import Tool, ToolContext, ToolError, ToolResult

MAX_SKILL_CHARS = 60_000  # как MAX_READ_CHARS у read_file
MAX_LISTED_FILES = 50


class SkillParams(BaseModel):
    name: str = Field(description="Skill name from the Skills list in your context")
    file: str | None = Field(
        default=None,
        description="A file of the skill directory to read (relative to it), e.g. a "
        "reference the skill mentions. Omit to load the skill itself",
    )


def _skill_files(skill: Skill) -> list[str]:
    """Файлы каталога скилла (кроме SKILL.md), не больше :data:`MAX_LISTED_FILES`.

    Симлинки за пределы каталога не показываются: прочитать их всё равно нельзя.
    """
    files: list[str] = []
    root = skill.directory.resolve()
    for path in sorted(skill.directory.rglob("*")):
        rel = path.relative_to(skill.directory).as_posix()
        if rel == SKILL_FILE or rel.startswith(".") or not path.is_file():
            continue
        if not is_within(root, path):
            continue
        if len(files) == MAX_LISTED_FILES:
            files.append("…")
            break
        files.append(rel)
    return files


def _read(path: Path, shown: str) -> str:
    try:
        with path.open(encoding="utf-8") as fh:  # не больше лимита: файл может быть огромным
            text = fh.read(MAX_SKILL_CHARS + 1)
    except UnicodeDecodeError:
        raise ToolError(f"Не текстовый файл (не UTF-8): {shown}") from None
    except OSError as e:
        raise ToolError(f"Не удалось прочитать {shown}: {e}") from None
    if len(text) > MAX_SKILL_CHARS:
        text = text[:MAX_SKILL_CHARS] + f"\n… (обрезано: файл длиннее {MAX_SKILL_CHARS} символов)"
    return text


class SkillTool(Tool):
    name = "skill"
    description = (
        "Loads a skill — ready-made instructions for a typical task — by its name from "
        "the Skills list in your context. Returns the instructions and the list of the "
        "skill's files; pass file to read one of them."
    )
    Params = SkillParams

    def describe(self, params: SkillParams) -> str:
        return f"{params.name}/{params.file}" if params.file else params.name

    def paths(self, params: SkillParams) -> tuple[str, ...]:
        return ()

    def run(self, params: SkillParams, ctx: ToolContext) -> ToolResult:
        skills = discover_skills(ctx.workspace)
        skill = find_skill(skills, params.name.strip())
        if skill is None:
            names = ", ".join(s.name for s in skills) or "скиллов нет"
            raise ToolError(f"Скилл не найден: {params.name}. Доступные: {names}")
        if params.file:
            try:
                path = resolve_in_root(skill.directory, params.file)
            except SandboxError:
                raise ToolError(
                    f"Путь '{params.file}' выходит за пределы каталога скилла {skill.name}"
                ) from None
            shown = f"{skill.name}/{params.file}"
            if not path.is_file():
                raise ToolError(f"Файл скилла не найден: {shown}")
            return ToolResult(content=_read(path, shown), summary=f"файл скилла {shown}")

        body = split_frontmatter(_read(skill.path, f"{skill.name}/{SKILL_FILE}"))[1].strip()
        parts = [f"Skill {skill.name} (directory: {skill.directory}).", body]
        files = _skill_files(skill)
        if files:
            parts.append(
                "Files of the skill (read with the skill tool and file; commands run in the "
                "project root, so use the directory path above):\n"
                + "\n".join(f"- {f}" for f in files)
            )
        return ToolResult(content="\n\n".join(parts), summary=f"скилл {skill.name}")
