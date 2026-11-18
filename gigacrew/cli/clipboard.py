"""Буфер обмена для /copy: исходный Markdown ответа, а не его отрисовка.

Выделение мышью копирует то, что нарисовал rich: без ``#`` и ``**``, с маркерами «•»,
отступами и переносами по ширине окна. Поэтому текст ответа кладётся в буфер напрямую.

Сначала — системная утилита (``pbcopy``, ``wl-copy``, ``xclip``, ``xsel``). Её нет (сервер
по SSH, контейнер) — последовательность OSC 52: буфер заполняет сам терминал, в том числе
на машине пользователя по ту сторону SSH. Получилось ли, терминал не сообщает: OSC 52
поддерживают не все терминалы, а tmux пропускает его только с ``set-clipboard on``.
"""

from __future__ import annotations

import base64
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from typing import IO

from gigacrew.agent.loop import Agent
from gigacrew.security import subprocess_env
from gigacrew.ui.console import Console, sanitize
from gigacrew.ui.format import plural

__all__ = ["COPY_USAGE", "OSC52", "clipboard_commands", "copy_command", "copy_text"]

COPY_USAGE = "/copy [N]"
_LEADING_BLANK_LINES = re.compile(r"\A(?:[ \t]*\n)+")

# Способ копирования через терминал (возвращает copy_text).
OSC52 = "OSC 52"
# Утилите хватает доли секунды; зависшая (нет доступа к дисплею) не держит REPL.
_TIMEOUT = 5


def clipboard_commands(environ: Mapping[str, str], platform: str) -> list[list[str]]:
    """Утилиты буфера обмена в порядке предпочтения (без проверки, установлены ли)."""
    if platform == "darwin":
        return [["pbcopy"]]
    commands: list[list[str]] = []
    if environ.get("WAYLAND_DISPLAY"):
        # Без --type wl-copy угадывает тип по содержимому и может выдать текст не текстом.
        commands.append(["wl-copy", "--type", "text/plain;charset=utf-8"])
    if environ.get("DISPLAY"):
        commands.extend((["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]))
    return commands


def _run_tool(command: list[str], text: str, environ: Mapping[str, str]) -> bool:
    """Отдаёт текст утилите на stdin; True — она завершилась успешно.

    Вывод утилиты не читается: xclip и wl-copy оставляют потомка, который держит буфер, и
    чтение их stdout ждало бы его выхода.
    """
    try:
        completed = subprocess.run(
            command,
            input=text.encode(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=subprocess_env(environ),
            timeout=_TIMEOUT,
            check=False,
            start_new_session=True,  # Ctrl+C в REPL не обрывает потомка с буфером
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _osc52(text: str, terminal: IO[str]) -> bool:
    """Просит терминал положить текст в буфер (OSC 52); False — вывод не терминал."""
    if not terminal.isatty():
        return False
    encoded = base64.b64encode(text.encode()).decode("ascii")
    terminal.write(f"\x1b]52;c;{encoded}\x07")
    terminal.flush()
    return True


def copy_text(
    text: str,
    *,
    terminal: IO[str] | None = None,
    environ: Mapping[str, str] | None = None,
    platform: str | None = None,
) -> str | None:
    """Кладёт текст в буфер обмена. Итог — чем скопировано; None — скопировать нечем.

    ``terminal`` — поток для OSC 52 (по умолчанию stdout), ``environ`` — окружение для
    выбора и запуска утилиты (по умолчанию — окружение процесса), ``platform`` — как
    ``sys.platform``.
    """
    env = os.environ if environ is None else environ
    path = env.get("PATH")
    for command in clipboard_commands(env, platform or sys.platform):
        program = shutil.which(command[0], path=path)
        if program is not None and _run_tool([program, *command[1:]], text, env):
            return command[0]
    if terminal is None:
        terminal = sys.stdout
    return OSC52 if _osc52(text, terminal) else None


def _answers(agent: Agent) -> list[str]:
    """Тексты ответов модели в чате, от первого к последнему (без пустых).

    Ответ — любой текст модели, который видел пользователь, в том числе пояснение перед
    вызовом инструментов. Управляющие символы убраны, как при выводе; ведущие пустые
    строки и концевые пробелы — тоже, а отступ первой строки (код отступом) сохранён.
    """
    texts = (
        sanitize(message.content)
        for message in agent.conversation.messages
        if message.role == "assistant"
    )
    return [_LEADING_BLANK_LINES.sub("", text).rstrip() for text in texts if text.strip()]


def _report(ui: Console, method: str | None, text: str) -> None:
    """Сообщает, чем скопирован ответ (``method`` — итог :func:`copy_text`)."""
    if method is None:
        ui.error(
            "не удалось скопировать: ни pbcopy, ни wl-copy, xclip или xsel не сработали, "
            "а OSC 52 нужен вывод в терминал",
        )
    elif method == OSC52:
        ui.info(
            "ответ передан терминалу (OSC 52); буфер пуст — терминал или tmux не "
            "поддерживает OSC 52 (в tmux: set -g set-clipboard on)",
        )
    else:
        lines = text.count("\n") + 1
        word = plural(lines, "строка", "строки", "строк")
        ui.success(f"ответ скопирован ({method}): {lines} {word}")


def copy_command(agent: Agent, ui: Console, arg: str) -> bool:
    """``/copy [N]``: исходный Markdown ответа модели — в буфер (N-й с конца, 1 — последний)."""
    if arg and not re.fullmatch(r"[1-9]\d{0,5}", arg):
        ui.error(f"использование: {COPY_USAGE}")
        return True
    answers = _answers(agent)
    number = int(arg or "1")
    if number > len(answers):
        count = len(answers)
        word = plural(count, "ответ", "ответа", "ответов")
        ui.error(f"в чате {count} {word} модели" if count else "ответов модели ещё нет")
        return True
    text = answers[-number]
    _report(ui, copy_text(text), text)
    return True
