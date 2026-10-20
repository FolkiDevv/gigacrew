"""Настройки gigacrew в JSON: пока только хуки (секция ``"hooks"``).

Источники (хуки из всех объединяются в этом порядке):

* ``~/.gigacode/settings.json`` — пользователь, общие для всех проектов;
* ``<проект>/.gigacode/settings.json`` — проект (коммитится);
* ``<проект>/.gigacode/settings.local.json`` — личные настройки в проекте (в ``.gitignore``).

Формат секции — как у Claude Code::

    {"hooks": {"PostToolUse": [{"matcher": "write_file|edit_file",
        "hooks": [{"type": "command", "command": "make fmt", "timeout": 30}]}]}}

Ошибки в файлах не мешают запуску: запись с ошибкой пропускается, вызывающий получает
текст предупреждения. Хуки проекта приходят вместе с репозиторием — это запуск чужих
команд, поэтому они работают только после подтверждения пользователя; подтверждённые
отпечатки хранит :class:`TrustStore`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from gigacrew.project.workspace import Workspace, user_data_dir, user_settings_file

SettingsSource = Literal["user", "project", "local"]
PROJECT_SOURCES: tuple[SettingsSource, ...] = ("project", "local")

DEFAULT_HOOK_TIMEOUT = 60
MIN_HOOK_TIMEOUT = 1
MAX_HOOK_TIMEOUT = 600

TRUST_FILE_NAME = "trusted_hooks.json"
TRUST_FORMAT_VERSION = 1


class HookEvent(StrEnum):
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    USER_PROMPT_SUBMIT = "UserPromptSubmit"
    STOP = "Stop"
    SUBAGENT_STOP = "SubagentStop"
    SESSION_START = "SessionStart"
    SESSION_END = "SessionEnd"
    PRE_COMPACT = "PreCompact"
    NOTIFICATION = "Notification"


# У этих событий нечего сравнивать с matcher: хуки срабатывают всегда.
UNMATCHED_EVENTS = frozenset({HookEvent.USER_PROMPT_SUBMIT, HookEvent.STOP})


@dataclass(frozen=True)
class HookCommand:
    command: str
    timeout: int = DEFAULT_HOOK_TIMEOUT  # секунды


@dataclass(frozen=True)
class HookGroup:
    """Запись события: matcher и команды, которые запускаются при совпадении."""

    event: HookEvent
    matcher: str
    hooks: tuple[HookCommand, ...]
    source: SettingsSource = "user"

    def matches(self, target: str) -> bool:
        """Совпадает ли ``target`` (имя инструмента, источник, причина…) с matcher.

        Пустой matcher и ``*`` — всё; иначе регулярное выражение целиком
        (``write_file|edit_file``, ``.*_file``).
        """
        if self.event in UNMATCHED_EVENTS or self.matcher in ("", "*"):
            return True
        return re.fullmatch(self.matcher, target) is not None


@dataclass(frozen=True)
class HookSettings:
    groups: tuple[HookGroup, ...] = ()
    disabled: bool = False  # "disableAllHooks": true хотя бы в одном файле

    def commands(self, event: HookEvent, target: str = "") -> list[HookCommand]:
        """Команды события для ``target`` по порядку; одинаковая команда — один раз."""
        if self.disabled:
            return []
        seen: set[str] = set()
        found: list[HookCommand] = []
        for group in self.groups:
            if group.event is not event or not group.matches(target):
                continue
            for hook in group.hooks:
                if hook.command not in seen:
                    seen.add(hook.command)
                    found.append(hook)
        return found

    def has(self, event: HookEvent, target: str = "") -> bool:
        return bool(self.commands(event, target))

    @property
    def project_groups(self) -> tuple[HookGroup, ...]:
        """Хуки из файлов проекта — им нужно подтверждение."""
        return tuple(g for g in self.groups if g.source in PROJECT_SOURCES)

    def without_project(self) -> HookSettings:
        """Только хуки пользователя (хуки проекта не подтверждены)."""
        return replace(
            self, groups=tuple(g for g in self.groups if g.source not in PROJECT_SOURCES)
        )


def settings_files(workspace: Workspace) -> list[tuple[Path, SettingsSource]]:
    """Файлы настроек по порядку: пользователь, проект, личные настройки в проекте."""
    return [
        (user_settings_file(), "user"),
        (workspace.settings_file, "project"),
        (workspace.local_settings_file, "local"),
    ]


def load_settings(workspace: Workspace) -> tuple[HookSettings, list[str]]:
    """Хуки из всех файлов настроек и предупреждения об ошибках в них.

    Нет файла — не ошибка. Нечитаемый файл или не JSON-объект — предупреждение, файл
    пропускается; ошибочная запись — предупреждение, пропускается только она.
    """
    groups: list[HookGroup] = []
    warnings: list[str] = []
    disabled = False
    for path, source in settings_files(workspace):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        except (OSError, UnicodeDecodeError, ValueError) as e:
            warnings.append(f"не удалось прочитать {path}: {e} — файл пропущен")
            continue
        if not isinstance(data, dict):
            warnings.append(f"{path}: ожидается JSON-объект — файл пропущен")
            continue
        if data.get("disableAllHooks") is True:
            disabled = True
        hooks = data.get("hooks")
        if hooks is None:
            continue
        if not isinstance(hooks, dict):
            warnings.append(f"{path}: hooks — не объект, хуки пропущены")
            continue
        for name, entries in hooks.items():
            where = f"{path}: hooks.{name}"
            try:
                event = HookEvent(name)
            except ValueError:
                known = ", ".join(e.value for e in HookEvent)
                warnings.append(f"{where}: неизвестное событие (поддерживаются: {known})")
                continue
            if not isinstance(entries, list):
                warnings.append(f"{where}: ожидается список записей")
                continue
            for i, entry in enumerate(entries):
                group = _parse_group(event, entry, source, f"{where}[{i}]", warnings)
                if group is not None:
                    groups.append(group)
    return HookSettings(tuple(groups), disabled=disabled), warnings


def _parse_group(
    event: HookEvent, entry: Any, source: SettingsSource, where: str, warnings: list[str]
) -> HookGroup | None:
    if not isinstance(entry, dict):
        warnings.append(f"{where}: ожидается объект с полями matcher и hooks — пропущено")
        return None
    matcher = entry.get("matcher", "")
    if matcher is None:
        matcher = ""
    if not isinstance(matcher, str):
        warnings.append(f"{where}: matcher — не строка, запись пропущена")
        return None
    if matcher not in ("", "*"):
        try:
            re.compile(matcher)
        except re.error as e:
            warnings.append(f"{where}: неверное регулярное выражение matcher {matcher!r}: {e}")
            return None
    raw_hooks = entry.get("hooks")
    if not isinstance(raw_hooks, list) or not raw_hooks:
        warnings.append(f"{where}: hooks — ожидается непустой список команд")
        return None
    hooks: list[HookCommand] = []
    for j, raw in enumerate(raw_hooks):
        hook = _parse_command(raw, f"{where}.hooks[{j}]", warnings)
        if hook is not None:
            hooks.append(hook)
    if not hooks:
        return None
    return HookGroup(event, matcher, tuple(hooks), source)


def _parse_command(raw: Any, where: str, warnings: list[str]) -> HookCommand | None:
    if not isinstance(raw, dict):
        warnings.append(f"{where}: ожидается объект — пропущено")
        return None
    kind = raw.get("type")
    if kind != "command":
        warnings.append(f'{where}: тип {kind!r} не поддерживается (только "command") — пропущено')
        return None
    command = raw.get("command")
    if not isinstance(command, str) or not command.strip():
        warnings.append(f"{where}: command — ожидается непустая строка")
        return None
    timeout = raw.get("timeout", DEFAULT_HOOK_TIMEOUT)
    if isinstance(timeout, bool) or not isinstance(timeout, int | float):
        warnings.append(f"{where}: timeout — ожидается число секунд, взято {DEFAULT_HOOK_TIMEOUT}")
        timeout = DEFAULT_HOOK_TIMEOUT
    clamped = min(max(int(timeout), MIN_HOOK_TIMEOUT), MAX_HOOK_TIMEOUT)
    if not MIN_HOOK_TIMEOUT <= timeout <= MAX_HOOK_TIMEOUT:
        warnings.append(
            f"{where}: timeout {timeout} вне {MIN_HOOK_TIMEOUT}…{MAX_HOOK_TIMEOUT} с, "
            f"взято {clamped}"
        )
    return HookCommand(command, clamped)


# ---------------------------------------------------------------------------- #
# Доверие к хукам проекта


def project_fingerprint(settings: HookSettings) -> str:
    """Отпечаток хуков проекта (sha256); пустая строка — хуков проекта нет.

    Любое изменение команд, matcher, таймаутов или файла, где они записаны, меняет
    отпечаток — и хуки снова требуют подтверждения.
    """
    groups = settings.project_groups
    if not groups:
        return ""
    canonical = [
        [g.source, g.event.value, g.matcher, [[h.command, h.timeout] for h in g.hooks]]
        for g in groups
    ]
    data = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


class TrustStore:
    """Подтверждённые хуки проектов: ``~/.gigacode/trusted_hooks.json``.

    Формат: ``{"version": 1, "projects": {"<корень проекта>": "<отпечаток>"}}``. Битый
    файл читается как пустой (хуки просто спросят подтверждение снова).
    """

    def __init__(self, path: Path | None = None):
        self.path = user_data_dir() / TRUST_FILE_NAME if path is None else path

    def is_trusted(self, root: Path, fingerprint: str) -> bool:
        return bool(fingerprint) and self._read().get(_project_key(root)) == fingerprint

    def trust(self, root: Path, fingerprint: str) -> None:
        """Запомнить отпечаток проекта. Ошибка записи — ``OSError``."""
        projects = self._read()
        projects[_project_key(root)] = fingerprint
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"version": TRUST_FORMAT_VERSION, "projects": dict(sorted(projects.items()))}
        fd, tmp = tempfile.mkstemp(
            prefix=f".{self.path.stem}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def _read(self) -> dict[str, str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            return {}
        if not isinstance(data, dict) or data.get("version") != TRUST_FORMAT_VERSION:
            return {}
        projects = data.get("projects")
        if not isinstance(projects, dict):
            return {}
        return {str(k): v for k, v in projects.items() if isinstance(v, str)}


def _project_key(root: Path) -> str:
    return str(Path(root).resolve())
