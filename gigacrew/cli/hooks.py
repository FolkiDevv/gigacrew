"""Хуки пользователя в CLI: загрузка настроек, подтверждение хуков проекта, хуки
сессии (SessionStart/SessionEnd) и описание для ``/hooks``.

Хуки проекта (``.gigacode/settings.json`` и ``settings.local.json``) приходят вместе с
репозиторием — это запуск чужих команд. Они работают, только если пользователь их
подтвердил (отпечаток запоминается в ``~/.gigacode/trusted_hooks.json``; изменились —
спросим снова) или передал ``--trust-hooks``. Без терминала и в ``-p`` спросить некого:
хуки проекта пропускаются с предупреждением.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

from gigacrew.agent.hooks import HookRunner
from gigacrew.agent.loop import Agent
from gigacrew.project.settings import (
    HookEvent,
    HookGroup,
    HookSettings,
    SettingsSource,
    TrustStore,
    load_settings,
    project_fingerprint,
)
from gigacrew.project.workspace import Workspace
from gigacrew.ui.console import Console, sanitize

SOURCE_LABELS: dict[SettingsSource, str] = {
    "user": "~/.gigacode/settings.json",
    "project": ".gigacode/settings.json",
    "local": ".gigacode/settings.local.json",
}
UNTRUSTED_WARNING = (
    "хуки проекта (.gigacode/settings*.json) не подтверждены и пропущены: запустите gigacrew "
    "интерактивно, чтобы подтвердить их, или передайте --trust-hooks"
)


def load_hooks(
    workspace: Workspace,
    ui: Console,
    *,
    interactive: bool,
    trust: bool = False,
    store: TrustStore | None = None,
) -> HookRunner:
    """Хуки из файлов настроек; хуки проекта — только подтверждённые.

    ``interactive`` — можно спросить пользователя; ``trust`` (``--trust-hooks``) —
    хуки проекта доверены на этот запуск, без вопроса и без записи отпечатка.
    """
    settings, warnings = load_settings(workspace)
    for warning in warnings:
        ui.warn(warning)
    fingerprint = project_fingerprint(settings)
    if settings.disabled or not fingerprint or trust:
        return HookRunner(settings)
    store = TrustStore() if store is None else store
    if store.is_trusted(workspace.root, fingerprint):
        return HookRunner(settings)
    if not interactive:
        ui.warn(UNTRUSTED_WARNING)
        return HookRunner(settings.without_project())
    ui.warn("в проекте настроены хуки — gigacrew будет сам запускать эти команды:")
    ui.print(describe_groups(settings.project_groups), markup=False, highlight=False)
    if not ui.ask("Разрешить хуки проекта?", dangerous=True):
        ui.info("хуки проекта выключены до конца сессии (/hooks reload — спросить снова)")
        return HookRunner(settings.without_project())
    try:
        store.trust(workspace.root, fingerprint)
    except OSError as e:
        ui.warn(f"подтверждение не сохранено ({e}) — при следующем запуске спрошу снова")
    return HookRunner(settings)


def describe_groups(groups: tuple[HookGroup, ...]) -> str:
    """Хуки по событиям: matcher, полная команда (без управляющих символов), файл."""
    lines: list[str] = []
    for event in HookEvent:
        found = [group for group in groups if group.event is event]
        if not found:
            continue
        lines.append(f"  {event.value}")
        for group in found:
            matcher = f"[{group.matcher}] " if group.matcher not in ("", "*") else ""
            for hook in group.hooks:
                command = sanitize(hook.command).strip().replace("\n", "\n      ")
                lines.append(f"    {matcher}{command}")
                lines.append(f"      таймаут {hook.timeout} с · {SOURCE_LABELS[group.source]}")
    return "\n".join(lines)


def describe_hooks(settings: HookSettings) -> str:
    """Текст для ``/hooks``."""
    files = ", ".join(SOURCE_LABELS.values())
    if settings.disabled:
        return f"хуки выключены (disableAllHooks); настройки: {files}"
    if not settings.groups:
        return f"хуков нет; настройки: {files}"
    return f"хуки (настройки: {files}):\n{describe_groups(settings.groups)}"


def _session_hook(ui: Console, event: HookEvent, run: Callable[[], None]) -> None:
    """Хуки сессии: Ctrl+C прерывает только их, а не сессию."""
    try:
        run()
    except KeyboardInterrupt:
        ui.stop_live()
        ui.system(f"(хук {event} прерван)")


@contextmanager
def hook_session(agent: Agent, ui: Console, *, source: str, end_reason: str) -> Iterator[None]:
    """SessionStart (``source``) при входе и SessionEnd (``end_reason``) при выходе."""
    _session_hook(ui, HookEvent.SESSION_START, lambda: agent.start_session(source))
    try:
        yield
    finally:
        _session_hook(ui, HookEvent.SESSION_END, lambda: agent.end_session(end_reason))


def restart_session[T](
    agent: Agent, ui: Console, *, end_reason: str, source: str, switch: Callable[[], T]
) -> T:
    """Смена чата (``/clear``, ``/resume``): SessionEnd старого, ``switch``, SessionStart
    нового. SessionEnd идёт до ``switch`` — хуки видят id прежнего чата."""
    _session_hook(ui, HookEvent.SESSION_END, lambda: agent.end_session(end_reason))
    result = switch()
    _session_hook(ui, HookEvent.SESSION_START, lambda: agent.start_session(source))
    return result
