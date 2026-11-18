"""Слэш-команды сессии ACP: ``/compact`` и скиллы.

Команды REPL для терминала (выход, буфер обмена, ``/model``, ``/mode``…) в ACP не нужны:
режим и модель клиент меняет своими методами. Остаются ``/compact`` и скиллы; скилл
раскрывается в запрос агенту, как в CLI (:func:`~gigacrew.cli.commands.expand_skill`), и
встроенная ``/compact`` важнее скилла с тем же именем. Клиенту команды объявляются без ``/``
(его добавляет клиент).
"""

from __future__ import annotations

from typing import Any, Final

from gigacrew.cli.acp.rpc import INTERNAL_ERROR, RpcError
from gigacrew.cli.commands import CommandRegistry, default_commands, expand_skill, is_repl_command
from gigacrew.project.skills import SkillCatalog
from gigacrew.project.workspace import Workspace

__all__ = ["SlashCommands", "compact_instructions"]

_COMPACT: Final = "/compact"
_SKILL_HINT: Final = "аргументы"

type Command = dict[str, Any]


def _builtin_commands() -> CommandRegistry:
    """Команды REPL, которые работают и без терминала: только ``/compact``."""
    registry = CommandRegistry()
    compact = default_commands().get(_COMPACT)
    if compact is not None:
        registry.register(compact)
    return registry


_COMMANDS: Final = _builtin_commands()


class SlashCommands:
    """Слэш-команды сессии: ``/compact`` и скиллы проекта (скиллы читаются при создании)."""

    def __init__(self, workspace: Workspace) -> None:
        """Команды проекта ``workspace``."""
        self._skills = SkillCatalog(workspace)
        self._skills.refresh()

    def advertised(self) -> list[Command]:
        """Команды для ``available_commands_update``: встроенные, затем скиллы."""
        commands = [
            _command(command.name, command.summary, command.usage.removeprefix(command.name))
            for command in _COMMANDS
        ]
        commands.extend(
            _command(skill.command, skill.description or skill.name, _SKILL_HINT)
            for skill in self._skills
            if _COMMANDS.get(skill.command) is None
        )
        return commands

    def expand(self, text: str) -> str:
        """Запрос агенту: ``/<скилл> аргументы`` раскрывается, остальное — как есть.

        Скилл не прочитать — ошибка JSON-RPC с причиной.
        """
        try:
            expanded = expand_skill(text, _COMMANDS, self._skills)
        except OSError as error:
            raise RpcError(INTERNAL_ERROR, str(error)) from None
        return text if expanded is None else expanded


def compact_instructions(text: str) -> str | None:
    """Пожелания к сжатию, если ``text`` — команда ``/compact``; иначе None."""
    if not is_repl_command(text):
        return None
    parts = text.strip().split(maxsplit=1)
    if _COMMANDS.get(parts[0]) is None:
        return None
    return parts[1].strip() if len(parts) > 1 else ""


def _command(name: str, description: str, hint: str) -> Command:
    """Команда ACP: имя без ``/``, описание и подсказка к аргументам."""
    command: Command = {"name": name.removeprefix("/"), "description": description}
    hint = hint.strip(" []")
    if hint:
        command["input"] = {"hint": hint}
    return command
