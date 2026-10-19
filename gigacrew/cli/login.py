"""Вход в бэкенд через браузер: ссылка в терминале и попытка открыть браузер.

Ссылка печатается всегда — её можно открыть и на другом компьютере. Браузер
открывается сам, только если есть графическая среда и сессия не по SSH; модуль
``webbrowser`` не используется: без графики он запускает консольный браузер (lynx,
w3m) прямо посреди хода.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable, Mapping

from gigacrew.ui.console import Console


def can_open_browser(environ: Mapping[str, str] | None = None, platform: str | None = None) -> bool:
    """Есть ли графический браузер: не SSH и (macOS, Windows или ``DISPLAY``/Wayland)."""
    env = os.environ if environ is None else environ
    platform = sys.platform if platform is None else platform
    if env.get("SSH_CONNECTION") or env.get("SSH_CLIENT") or env.get("SSH_TTY"):
        return False
    if platform == "darwin" or platform.startswith(("win", "cygwin")):
        return True
    return bool(env.get("DISPLAY") or env.get("WAYLAND_DISPLAY"))


def open_in_browser(url: str) -> bool:
    """Открыть ``url`` браузером по умолчанию; False — нельзя или не получилось."""
    if not can_open_browser():
        return False
    try:
        if sys.platform.startswith("win"):
            os.startfile(url)  # type: ignore[attr-defined]  # есть только в Windows
            return True
        command = ["open", url] if sys.platform == "darwin" else ["xdg-open", url]
        subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # Ctrl+C в gigacrew не убивает браузер
        )
    except OSError:
        return False
    return True


def login_notifier(
    ui: Console, opener: Callable[[str], bool] = open_in_browser
) -> Callable[[str], None]:
    """``on_login`` провайдера: показать ссылку на страницу входа и ждать подтверждения."""

    def notify(url: str) -> None:
        opened = opener(url)
        action = "открываю страницу входа в браузере" if opened else "откройте ссылку в браузере"
        ui.warn(f"нужен вход в GigaCode — {action}:")
        ui.link(url)
        ui.system("жду подтверждения входа… Ctrl+C — отмена")

    return notify
