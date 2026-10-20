"""Хуки пользователя в CLI: загрузка настроек, подтверждение хуков, хук PreConnect,
хуки сессии (SessionStart/SessionEnd) и описание для ``/hooks``.

Хуки — запуск команд без вопросов: хуки проекта (``.gigacode/settings.json`` и
``settings.local.json``) приходят вместе с репозиторием, а в ``~/.gigacode/settings.json``
команду может дописать и сама модель (через ``run_shell``). Поэтому любые хуки работают,
только если пользователь подтвердил их (отпечатки — в ``~/.gigacode/trusted_hooks.json``;
изменились — спросим снова) или передал ``--trust-hooks``. Без терминала и в ``-p``
спросить некого: неподтверждённые хуки пропускаются с предупреждением.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from gigacrew.agent.hooks import HookRunner
from gigacrew.agent.loop import Agent
from gigacrew.config import Config
from gigacrew.project.settings import (
    PROJECT_SOURCES,
    HookEvent,
    HookGroup,
    HookSettings,
    SettingsSource,
    TrustStore,
    load_settings,
    project_fingerprint,
    user_fingerprint,
)
from gigacrew.project.workspace import Workspace
from gigacrew.ui.console import Console, sanitize

SOURCE_LABELS: dict[SettingsSource, str] = {
    "user": "~/.gigacode/settings.json",
    "project": ".gigacode/settings.json",
    "local": ".gigacode/settings.local.json",
}


@dataclass(frozen=True)
class _Pending:
    """Неподтверждённые хуки одного владельца (пользователь или проект)."""

    owner: str  # «пользователя», «проекта»
    sources: tuple[SettingsSource, ...]
    groups: tuple[HookGroup, ...]
    remember: Callable[[], None]  # записать отпечаток в TrustStore (OSError)


def load_hooks(
    workspace: Workspace,
    ui: Console,
    *,
    interactive: bool,
    trust: bool = False,
    store: TrustStore | None = None,
) -> HookRunner:
    """Хуки из файлов настроек — только подтверждённые.

    Новые или изменённые хуки пользователя и проекта подтверждаются одним вопросом.
    ``interactive`` — можно спросить пользователя; ``trust`` (``--trust-hooks``) — все хуки
    доверены на этот запуск, без вопроса и без записи отпечатков.
    """
    settings, warnings = load_settings(workspace)
    for warning in warnings:
        ui.warn(warning)
    if settings.disabled or trust:
        return HookRunner(settings)
    store = TrustStore() if store is None else store
    pending: list[_Pending] = []
    user = user_fingerprint(settings)
    if user and not store.is_user_trusted(user):
        pending.append(
            _Pending(
                "пользователя", ("user",), settings.user_groups, lambda: store.trust_user(user)
            )
        )
    project = project_fingerprint(settings)
    if project and not store.is_trusted(workspace.root, project):
        pending.append(
            _Pending(
                "проекта",
                PROJECT_SOURCES,
                settings.project_groups,
                lambda: store.trust(workspace.root, project),
            )
        )
    if not pending:
        return HookRunner(settings)
    owners = " и ".join(p.owner for p in pending)
    sources = tuple(source for p in pending for source in p.sources)
    if not interactive:
        files = ", ".join(SOURCE_LABELS[source] for source in sources)
        ui.warn(
            f"хуки {owners} ({files}) не подтверждены и пропущены: запустите gigacrew "
            "интерактивно, чтобы подтвердить их, или передайте --trust-hooks"
        )
        return HookRunner(settings.without(*sources))
    ui.warn(f"новые или изменённые хуки {owners} — gigacrew будет сам запускать эти команды:")
    groups = tuple(group for p in pending for group in p.groups)
    ui.print(describe_groups(groups), markup=False, highlight=False)
    if not ui.ask(f"Разрешить хуки {owners}?", dangerous=True):
        ui.info(f"хуки {owners} выключены до конца сессии (/hooks reload — спросить снова)")
        return HookRunner(settings.without(*sources))
    try:
        for p in pending:
            p.remember()
    except OSError as e:
        ui.warn(f"подтверждение не сохранено ({e}) — при следующем запуске спрошу снова")
    return HookRunner(settings)


def run_pre_connect(runner: HookRunner, ui: Console, config: Config) -> str | None:
    """Хуки PreConnect — до подключения к бэкенду: причина остановить запуск или None.

    Агента и чата ещё нет: в stdin — провайдер, адрес и модель из настроек. Ctrl+C
    пробрасывается (вызывающий прерывает запуск, как при подключении).
    """
    root = Path(config.project_root).resolve()
    payload = {
        "hook_event_name": HookEvent.PRE_CONNECT.value,
        "cwd": str(root),
        "provider": config.provider,
        "api_url": config.base_url,
        "model": config.model,
    }
    result = runner.run(HookEvent.PRE_CONNECT, payload, cwd=root, events=ui, target=config.provider)
    return result.reason if result.blocked else None


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
