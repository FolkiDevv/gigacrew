"""Настройки gigacrew в JSON: пока только хуки (секция ``"hooks"``).

Источники (хуки из всех объединяются в этом порядке):

* ``~/.gigacode/settings.json`` — пользователь, общие для всех проектов;
* ``<проект>/.gigacode/settings.json`` — проект (коммитится);
* ``<проект>/.gigacode/settings.local.json`` — личные настройки в проекте (в ``.gitignore``).

Формат секции — как у Claude Code::

    {
        "hooks": {
            "PostToolUse": [
                {
                    "matcher": "write_file|edit_file",
                    "hooks": [{"type": "command", "command": "make fmt", "timeout": 30}],
                }
            ]
        }
    }

Ошибки в файлах не мешают запуску: запись с ошибкой пропускается, вызывающий получает
текст предупреждения. Хуки — запуск команд без вопросов, поэтому они работают только после
подтверждения пользователя: хуки проекта приходят вместе с репозиторием, а в файл
пользователя команду может дописать и сама модель (через ``run_shell``). Подтверждённые
отпечатки хранит :class:`TrustStore`: изменились хуки — нужно подтверждение снова.
"""

from __future__ import annotations

import hashlib
import json
import math
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
_TRUST_USER_KEY = "user"  # отпечаток хуков пользователя в файле доверия

# Нечитаемый файл настроек. RecursionError — глубоко вложенный JSON ([[[…]]]) из репозитория:
# не повод падать.
_READ_ERRORS = (OSError, UnicodeDecodeError, ValueError, RecursionError)


class HookEvent(StrEnum):
    """События, на которые вешаются хуки (имена — как в Claude Code)."""

    PRE_REQUEST = "PreRequest"  # перед каждым запросом к GigaCode (matcher — вид запроса)
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
UNMATCHED_EVENTS = frozenset((HookEvent.USER_PROMPT_SUBMIT, HookEvent.STOP))


@dataclass(frozen=True)
class HookCommand:
    """Команда хука и её таймаут."""

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
        if self.event in UNMATCHED_EVENTS or self.matcher in {"", "*"}:
            return True
        return re.fullmatch(self.matcher, target) is not None


@dataclass(frozen=True)
class HookSettings:
    """Все хуки из файлов настроек и выключатели ``disableAllHooks``."""

    groups: tuple[HookGroup, ...] = ()
    disabled: bool = False  # "disableAllHooks": true в файле пользователя — все хуки
    # "disableAllHooks": true в файле проекта: выключены только хуки проекта — файл приходит
    # с репозиторием и не может выключить хуки пользователя (его проверки, запреты).
    project_disabled: bool = False

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
        """Есть ли у события команды для ``target``."""
        return bool(self.commands(event, target))

    def from_sources(self, *sources: SettingsSource) -> tuple[HookGroup, ...]:
        """Группы хуков из файлов ``sources``."""
        return tuple(group for group in self.groups if group.source in sources)

    @property
    def project_groups(self) -> tuple[HookGroup, ...]:
        """Хуки из файлов проекта."""
        return self.from_sources(*PROJECT_SOURCES)

    @property
    def user_groups(self) -> tuple[HookGroup, ...]:
        """Хуки из ``~/.gigacode/settings.json``."""
        return self.from_sources("user")

    def without(self, *sources: SettingsSource) -> HookSettings:
        """Без хуков из ``sources`` (не подтверждены)."""
        kept = tuple(group for group in self.groups if group.source not in sources)
        return replace(self, groups=kept)


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
    project_disabled = False
    for path, source in settings_files(workspace):
        document = _read_settings_file(path, warnings)
        if document is None:
            continue
        switch = document.get("disableAllHooks")
        if switch is True and source in PROJECT_SOURCES:
            project_disabled = True
        elif switch is True:
            disabled = True
        elif switch not in {None, False}:  # "true" или 1 молча не выключили бы ничего
            warnings.append(f"{path}: disableAllHooks — ожидается true или false, пропущено")
        groups.extend(_parse_hooks(document.get("hooks"), path, source, warnings))
    # Хуки недоверенного проекта не запускаются ни с каким подтверждением.
    if project_disabled or workspace.untrusted:
        groups = [group for group in groups if group.source not in PROJECT_SOURCES]
    settings = HookSettings(tuple(groups), disabled=disabled, project_disabled=project_disabled)
    return settings, warnings


def _read_settings_file(path: Path, warnings: list[str]) -> dict[str, Any] | None:
    """JSON-объект файла настроек; None — файла нет или он пропущен (с предупреждением)."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except _READ_ERRORS as exc:
        warnings.append(f"не удалось прочитать {path}: {exc} — файл пропущен")
        return None
    if not isinstance(document, dict):
        warnings.append(f"{path}: ожидается JSON-объект — файл пропущен")
        return None
    return document


def _parse_hooks(
    hooks: object,
    path: Path,
    source: SettingsSource,
    warnings: list[str],
) -> list[HookGroup]:
    """Секция ``hooks`` одного файла → группы хуков; ошибки — в ``warnings``."""
    if hooks is None:
        return []
    if not isinstance(hooks, dict):
        warnings.append(f"{path}: hooks — не объект, хуки пропущены")
        return []
    groups: list[HookGroup] = []
    for name, entries in hooks.items():
        where = f"{path}: hooks.{name}"
        try:
            event = HookEvent(name)
        except ValueError:
            known = ", ".join(member.value for member in HookEvent)
            warnings.append(f"{where}: неизвестное событие (поддерживаются: {known})")
            continue
        if not isinstance(entries, list):
            warnings.append(f"{where}: ожидается список записей")
            continue
        for index, entry in enumerate(entries):
            group = _parse_group(event, entry, source, f"{where}[{index}]", warnings)
            if group is not None:
                groups.append(group)
    return groups


def _parse_group(
    event: HookEvent,
    entry: Any,
    source: SettingsSource,
    where: str,
    warnings: list[str],
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
    if matcher not in {"", "*"}:
        try:
            re.compile(matcher)
        except re.error as exc:
            warnings.append(f"{where}: неверное регулярное выражение matcher {matcher!r}: {exc}")
            return None
    raw_hooks = entry.get("hooks")
    if not isinstance(raw_hooks, list) or not raw_hooks:
        warnings.append(f"{where}: hooks — ожидается непустой список команд")
        return None
    hooks: list[HookCommand] = []
    for index, raw in enumerate(raw_hooks):
        hook = _parse_command(raw, f"{where}.hooks[{index}]", warnings)
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
    # json.loads пропускает \u0000 и одиночные суррогаты — такую команду не запустить.
    if "\0" in command or _has_surrogate(command):
        warnings.append(f"{where}: command — NUL или неверный символ Unicode, пропущено")
        return None
    timeout = raw.get("timeout", DEFAULT_HOOK_TIMEOUT)
    # json.loads пропускает NaN, Infinity и 1e400 — int() на них упал бы. Огромное целое
    # (10**400) isfinite не переводит во float — оно доходит до ограничения сверху.
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, int | float)
        or (isinstance(timeout, float) and not math.isfinite(timeout))
    ):
        warnings.append(f"{where}: timeout — ожидается число секунд, взято {DEFAULT_HOOK_TIMEOUT}")
        timeout = DEFAULT_HOOK_TIMEOUT
    clamped = min(max(int(timeout), MIN_HOOK_TIMEOUT), MAX_HOOK_TIMEOUT)
    if not MIN_HOOK_TIMEOUT <= timeout <= MAX_HOOK_TIMEOUT:
        warnings.append(
            f"{where}: timeout {timeout} вне {MIN_HOOK_TIMEOUT}…{MAX_HOOK_TIMEOUT} с, "
            f"взято {clamped}",
        )
    return HookCommand(command, clamped)


def _has_surrogate(text: str) -> bool:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return True
    return False


# ---------------------------------------------------------------------------- #
# Доверие к хукам


def project_fingerprint(settings: HookSettings) -> str:
    """Отпечаток хуков проекта (sha256); пустая строка — хуков проекта нет."""
    return hooks_fingerprint(settings.project_groups)


def user_fingerprint(settings: HookSettings) -> str:
    """Отпечаток хуков пользователя (sha256); пустая строка — их нет."""
    return hooks_fingerprint(settings.user_groups)


def hooks_fingerprint(groups: tuple[HookGroup, ...]) -> str:
    """Отпечаток групп хуков (sha256); пустая строка — групп нет.

    Любое изменение команд, matcher, таймаутов или файла, где они записаны, меняет
    отпечаток — и хуки снова требуют подтверждения.
    """
    if not groups:
        return ""
    canonical = [
        [
            group.source,
            group.event.value,
            group.matcher,
            [[hook.command, hook.timeout] for hook in group.hooks],
        ]
        for group in groups
    ]
    serialized = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class TrustStore:
    """Подтверждённые хуки: ``~/.gigacode/trusted_hooks.json``.

    Формат: ``{"version": 1, "user": "<отпечаток>", "projects": {"<корень проекта>":
    "<отпечаток>"}}``; хуки пользователя подтверждаются одним отпечатком на все проекты.
    Битый файл читается как пустой (хуки просто спросят подтверждение снова).
    """

    def __init__(self, path: Path | None = None):
        """Хранилище в ``path``; по умолчанию — в каталоге данных пользователя."""
        self.path = user_data_dir() / TRUST_FILE_NAME if path is None else path

    def is_trusted(self, root: Path, fingerprint: str) -> bool:
        """Подтверждены ли хуки проекта ``root`` с этим отпечатком."""
        return bool(fingerprint) and self._read()[1].get(self._project_key(root)) == fingerprint

    def is_user_trusted(self, fingerprint: str) -> bool:
        """Подтверждены ли хуки пользователя с этим отпечатком."""
        return bool(fingerprint) and self._read()[0] == fingerprint

    def trust(self, root: Path, fingerprint: str) -> None:
        """Запомнить отпечаток проекта. Ошибка записи — ``OSError``."""
        user, projects = self._read()
        projects[self._project_key(root)] = fingerprint
        self._write(user, projects)

    def trust_user(self, fingerprint: str) -> None:
        """Запомнить отпечаток хуков пользователя. Ошибка записи — ``OSError``."""
        self._write(fingerprint, self._read()[1])

    def _write(self, user: str, projects: dict[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        document: dict[str, object] = {"version": TRUST_FORMAT_VERSION}
        if user:
            document[_TRUST_USER_KEY] = user
        document["projects"] = dict(sorted(projects.items()))
        stem = self.path.stem
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{stem}.",
            suffix=".tmp",
            dir=self.path.parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(document, stream, ensure_ascii=False, indent=2)
            Path(temp_name).replace(self.path)
        except BaseException:
            Path(temp_name).unlink(missing_ok=True)
            raise

    def _read(self) -> tuple[str, dict[str, str]]:
        """(отпечаток хуков пользователя, отпечатки проектов)."""
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except _READ_ERRORS:
            return "", {}
        if not isinstance(document, dict) or document.get("version") != TRUST_FORMAT_VERSION:
            return "", {}
        user = document.get(_TRUST_USER_KEY)
        projects = document.get("projects")
        if not isinstance(projects, dict):
            projects = {}
        return (
            user if isinstance(user, str) else "",
            {
                str(key): fingerprint
                for key, fingerprint in projects.items()
                if isinstance(fingerprint, str)
            },
        )

    def _project_key(self, root: Path) -> str:
        """Ключ проекта в файле доверия — абсолютный путь корня без симлинков."""
        return str(Path(root).resolve())
