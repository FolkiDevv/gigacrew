"""Строка ввода REPL на prompt_toolkit.

* автодополнение слеш-команд из :class:`~gigacrew.cli.commands.CommandRegistry`
  (имя + краткое описание), меню появляется при вводе ``/``; аргументы команд —
  из ``arg_choices`` (``/model`` — список моделей);
* история ввода в ``.gigacrew/history`` (папка создаётся при первом сохранённом
  вводе) и подсказки из истории;
* многострочный ввод: ``\\`` в конце строки + Enter или Alt+Enter — новая строка;
  Esc Esc — очистить ввод (черновик сохраняется в истории);
  вставка многострочного текста не отправляет его;
* Shift+Tab — сменить режим разрешений (ручной → авто-правки → план);
* статус-строка: модель, режим, заполнение контекста, потраченные токены, ``-y``.

Если stdin/stdout не терминал (pipe, CI) или prompt_toolkit не смог запуститься,
используется обычный ``input()``.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prompt_toolkit import PromptSession
from prompt_toolkit.application import get_app
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition, has_completions
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.output import ColorDepth
from prompt_toolkit.styles import Style

from gigacrew.permissions import PermissionMode
from gigacrew.project.workspace import Workspace
from gigacrew.ui.format import format_tokens, mode_badge
from gigacrew.ui.theme import PROMPT_STYLES

if TYPE_CHECKING:  # commands импортирует KEY_HELP отсюда
    from gigacrew.cli.commands import CommandRegistry

# Чтение строки; ``default`` — текст, которым заполнить ввод (набранный во время хода).
InputReader = Callable[..., str]
# Варианты аргумента команды: пары (значение, пояснение); вызывается при каждом дополнении.
ArgChoices = Callable[[], Iterable[tuple[str, str]]]

HISTORY_FILE = "history"
PROMPT = "❯ "
PLAIN_PROMPT = "gigacrew> "
PLACEHOLDER = "сообщение…  / — команды · \\ и Enter — новая строка"

# Подсказки по клавишам для /help.
KEY_HELP: tuple[tuple[str, str], ...] = (
    ("Enter", "отправить"),
    ("\\ + Enter, Alt+Enter", "новая строка (вставка многострочного текста не отправляет)"),
    ("Tab, ↑ ↓", "автодополнение команд, история ввода"),
    ("→", "принять подсказку из истории"),
    ("Shift+Tab", "сменить режим: ручной → авто-правки → план (и во время ответа)"),
    ("Esc", "во время ответа — прервать ход"),
    ("Esc Esc", "очистить ввод (черновик остаётся в истории, ↑ вернёт его)"),
    ("Ctrl+C", "сбросить ввод; во время ответа — прервать ход"),
    ("Ctrl+D", "выход"),
)

_COMMAND_PREFIX_RE = re.compile(r"/[\w-]*")
_COMMAND_ARG_RE = re.compile(r"(/[\w-]+)[ \t]+(\S*)")
_CONTEXT_WARN = 50  # % заполнения контекста — жёлтый
_CONTEXT_DANGER = 80  # % — красный


# ------------------------------ статус-строка ------------------------------ #
@dataclass(frozen=True)
class StatusInfo:
    model: str
    context_tokens: int = 0
    context_budget: int = 0
    billed_tokens: int = 0
    auto_approve: bool = False
    mode: PermissionMode = PermissionMode.MANUAL


def toolbar_fragments(status: StatusInfo) -> list[tuple[str, str]]:
    """Статус-строка в формате prompt_toolkit: список пар (стиль, текст)."""
    sep = ("", "  ·  ")
    parts: list[tuple[str, str]] = [("class:toolbar.model", f" {status.model}")]
    parts += [
        sep,
        (f"class:toolbar.mode.{status.mode.value}", mode_badge(status.mode)),
        ("", " (Shift+Tab)"),
    ]
    if status.context_budget > 0:
        percent = round(100 * status.context_tokens / status.context_budget)
        if percent >= _CONTEXT_DANGER:
            style = "class:toolbar.danger"
        elif percent >= _CONTEXT_WARN:
            style = "class:toolbar.warn"
        else:
            style = "class:toolbar.ok"
        used, budget = format_tokens(status.context_tokens), format_tokens(status.context_budget)
        parts += [sep, ("", "контекст "), (style, f"{used}/{budget} ({percent}%)")]
    parts += [sep, ("", f"потрачено {format_tokens(status.billed_tokens)}")]
    if status.auto_approve:
        parts += [sep, ("class:toolbar.danger", "⚠ авто-подтверждение")]
    return parts


# ------------------------------ автодополнение ----------------------------- #
class SlashCommandCompleter(Completer):
    """Дополняет первое слово, если это начало слеш-команды (не путь ``/home/...``),
    и аргумент команды, для которой есть источник вариантов в ``arg_choices``
    (ключ — имя команды): сначала совпадения по началу, затем по подстроке."""

    def __init__(self, commands: CommandRegistry, arg_choices: Mapping[str, ArgChoices] = {}):
        self._commands = commands
        self._arg_choices = {name.lower(): source for name, source in arg_choices.items()}

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        text = document.text_before_cursor
        if _COMMAND_PREFIX_RE.fullmatch(text):
            yield from self._complete_command(text)
            return
        match = _COMMAND_ARG_RE.fullmatch(text)
        if match is not None:
            yield from self._complete_arg(match.group(1), match.group(2))

    def _complete_arg(self, name: str, prefix: str) -> Iterable[Completion]:
        command = self._commands.get(name)
        source = self._arg_choices.get(command.name.lower()) if command is not None else None
        if source is None:
            return
        needle = prefix.lower()
        choices = list(source())
        starts = [c for c in choices if c[0].lower().startswith(needle)]
        inside = [c for c in choices if needle in c[0].lower() and c not in starts]
        for value, meta in starts + inside:
            yield Completion(value, start_position=-len(prefix), display_meta=meta)

    def _complete_command(self, text: str) -> Iterable[Completion]:
        prefix = text.lower()
        for cmd in self._commands:
            for name in (cmd.name, *cmd.aliases):
                if name.lower().startswith(prefix):
                    yield Completion(name, start_position=-len(text), display_meta=cmd.summary)
                    break


# --------------------------------- история --------------------------------- #
class LazyFileHistory(FileHistory):
    """История в ``.gigacrew/history``; папка создаётся при первом сохранении.

    Если записать не удалось (проект только для чтения), история остаётся в памяти
    до конца сессии — ошибка не должна закрывать REPL.
    """

    def __init__(self, workspace: Workspace):
        self._workspace = workspace
        self._persist = True
        super().__init__(workspace.data_dir / HISTORY_FILE)

    def store_string(self, string: str) -> None:
        if not self._persist:
            return
        try:
            self._workspace.ensure_data_dir()
            path = Path(self.filename)
            if not path.exists():
                # В истории бывают вставленные секреты — права как у чатов (0600).
                os.close(os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600))
            super().store_string(string)
        except OSError:
            self._persist = False


# --------------------------------- клавиши --------------------------------- #
def _key_bindings(on_cycle_mode: Callable[[], object] | None = None) -> KeyBindings:
    kb = KeyBindings()

    if on_cycle_mode is not None:
        # При открытом меню дополнения Shift+Tab остаётся штатным «назад по меню».
        @kb.add("s-tab", filter=~has_completions)
        def _cycle_mode(event: KeyPressEvent) -> None:
            on_cycle_mode()
            event.app.invalidate()  # статус-строка с новым режимом — сразу

    @kb.add("escape", "escape")
    def _clear(event: KeyPressEvent) -> None:
        buffer = event.current_buffer
        if buffer.text.strip():
            buffer.reset(append_to_history=True)  # черновик не теряется: ↑ вернёт его
        else:
            buffer.reset()

    @kb.add("escape", "enter")
    def _newline(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("\n")

    @Condition
    def _line_continues() -> bool:
        return get_app().current_buffer.document.text_before_cursor.endswith("\\")

    @kb.add("enter", filter=_line_continues)
    def _continue_line(event: KeyPressEvent) -> None:
        buffer = event.current_buffer
        buffer.delete_before_cursor(1)
        buffer.insert_text("\n")

    return kb


def _continuation(width: int, line_number: int, wrap_count: int) -> list[tuple[str, str]]:
    return [("class:continuation", "·".rjust(max(width - 1, 1)) + " ")]


# --------------------------------- сборка ---------------------------------- #
def create_prompt_session(
    *,
    commands: CommandRegistry,
    workspace: Workspace,
    status: Callable[[], StatusInfo],
    no_color: bool = False,
    arg_choices: Mapping[str, ArgChoices] = {},
    on_cycle_mode: Callable[[], object] | None = None,
    **kwargs: Any,
) -> PromptSession[str]:
    """Сессия ввода. ``kwargs`` (``input``/``output``) подменяются в тестах.

    ``on_cycle_mode`` — что сделать по Shift+Tab (сменить режим агента).
    """

    @Condition
    def _typing_command() -> bool:
        return get_app().current_buffer.text.startswith("/")

    return PromptSession(
        message=[("class:prompt", PROMPT)],
        history=LazyFileHistory(workspace),
        auto_suggest=AutoSuggestFromHistory(),
        completer=SlashCommandCompleter(commands, arg_choices),
        complete_while_typing=_typing_command,
        reserve_space_for_menu=min(8, len(list(commands)) + 1),
        key_bindings=_key_bindings(on_cycle_mode),
        bottom_toolbar=lambda: toolbar_fragments(status()),
        prompt_continuation=_continuation,
        placeholder=[("class:placeholder", PLACEHOLDER)],
        style=Style.from_dict(PROMPT_STYLES),
        color_depth=ColorDepth.MONOCHROME if no_color else None,
        **kwargs,
    )


def _read_plain(default: str = "") -> str:
    """``input()`` не умеет заполнять строку: набранное заранее печатается после
    приглашения и приклеивается к введённому."""
    return default + input(PLAIN_PROMPT + default)


def make_input_reader(
    *,
    commands: CommandRegistry,
    workspace: Workspace,
    status: Callable[[], StatusInfo],
    no_color: bool = False,
    arg_choices: Mapping[str, ArgChoices] = {},
    on_cycle_mode: Callable[[], object] | None = None,
) -> InputReader:
    """Функция чтения строки: prompt_toolkit в терминале, иначе ``input()``."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return _read_plain
    try:
        session = create_prompt_session(
            commands=commands,
            workspace=workspace,
            status=status,
            no_color=no_color,
            arg_choices=arg_choices,
            on_cycle_mode=on_cycle_mode,
        )
    except Exception:  # терминал не поддерживается (например, mintty без консоли)
        return _read_plain

    def read(default: str = "") -> str:
        return session.prompt(default=default)

    return read
