r"""Строка ввода REPL на prompt_toolkit.

* автодополнение слеш-команд из :class:`~gigacrew.cli.commands.CommandRegistry`
  и скиллов (имя + краткое описание), меню появляется при вводе ``/``; аргументы
  команд — из ``arg_choices`` (``/model`` — список моделей);
* история ввода в ``.gigacode/history`` (папка создаётся при первом сохранённом
  вводе) и подсказки из истории;
* многострочный ввод: ``\\`` в конце строки + Enter или Alt+Enter — новая строка;
  Esc Esc — очистить ввод (черновик сохраняется в истории);
  вставка многострочного текста не отправляет его;
* Shift+Tab — сменить режим разрешений (авто ↔ план);
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
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, override

from prompt_toolkit import PromptSession
from prompt_toolkit.application import get_app
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition, has_completions
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent

from gigacrew.permissions import PermissionMode
from gigacrew.project.workspace import Workspace
from gigacrew.ui.colors import prompt_color_depth, prompt_style
from gigacrew.ui.format import format_tokens, mode_badge

if TYPE_CHECKING:  # commands импортирует KEY_HELP отсюда
    from gigacrew.cli.commands import CommandRegistry

# Чтение строки; ``default`` — текст, которым заполнить ввод (набранный во время хода).
InputReader = Callable[..., str]
# Варианты аргумента команды: пары (значение, пояснение); вызывается при каждом дополнении.
ArgChoices = Callable[[], Iterable[tuple[str, str]]]
# Скиллы для дополнения: пары (команда «/имя», описание).
SkillChoices = Callable[[], Iterable[tuple[str, str]]]

PROMPT = "❯ "
PLAIN_PROMPT = "gigacrew> "
PLACEHOLDER = r"сообщение…  / — команды · \ и Enter — новая строка"

# Подсказки по клавишам для /help.
KEY_HELP: tuple[tuple[str, str], ...] = (
    ("Enter", "отправить"),
    (r"\ + Enter, Alt+Enter", "новая строка (вставка многострочного текста не отправляет)"),
    ("Tab, ↑ ↓", "автодополнение команд и скиллов, история ввода"),
    ("→", "принять подсказку из истории"),
    ("Shift+Tab", "сменить режим: авто ↔ план (и во время ответа)"),
    ("Esc", "во время ответа — прервать ход"),
    ("Esc Esc", "очистить ввод (черновик остаётся в истории, ↑ вернёт его)"),
    ("Ctrl+C", "сбросить ввод; во время ответа — прервать ход"),
    ("Ctrl+D", "выход"),
)

_COMMAND_PREFIX_RE = re.compile(r"/[\w-]*")
_COMMAND_ARG_RE = re.compile(r"(/[\w-]+)[ \t]+(\S*)")
_CONTEXT_WARN = 50  # % заполнения контекста — жёлтый
_CONTEXT_DANGER = 80  # % — красный
_HISTORY_FILE_MODE = 0o600  # в истории бывают вставленные секреты — права как у чатов
# Нет источников вариантов аргументов (значение по умолчанию, только для чтения).
_NO_ARG_CHOICES: Mapping[str, ArgChoices] = MappingProxyType({})


# ------------------------------ статус-строка ------------------------------ #
@dataclass(frozen=True)
class StatusInfo:
    """Данные статус-строки: модель, заполнение контекста, потраченные токены, режим."""

    model: str
    context_tokens: int = 0
    context_budget: int = 0
    billed_tokens: int = 0
    auto_approve: bool = False
    mode: PermissionMode = PermissionMode.AUTO


def toolbar_fragments(status: StatusInfo) -> list[tuple[str, str]]:
    """Статус-строка в формате prompt_toolkit: список пар (стиль, текст)."""
    sep = ("", "  ·  ")
    mode = status.mode
    parts: list[tuple[str, str]] = [("class:toolbar.model", f" {status.model}")]
    parts += [
        sep,
        (f"class:toolbar.mode.{mode.value}", mode_badge(mode)),
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


def _bottom_toolbar(status: StatusInfo) -> StyleAndTextTuples:
    """Статус-строка как ``StyleAndTextTuples`` — тип, который ждёт prompt_toolkit."""
    return list(toolbar_fragments(status))


# ------------------------------ автодополнение ----------------------------- #
class SlashCommandCompleter(Completer):
    """Дополняет слеш-команды и скиллы, а также аргументы команд.

    Первое слово дополняется, если это начало слеш-команды или скилла (не путь
    ``/home/...``); аргумент — у команды, для которой есть источник вариантов в
    ``arg_choices`` (ключ — имя команды): сначала совпадения по началу, затем по
    подстроке. Скилл с именем встроенной команды не предлагается — её не перекрыть.
    """

    def __init__(
        self,
        commands: CommandRegistry,
        arg_choices: Mapping[str, ArgChoices] = _NO_ARG_CHOICES,
        skills: SkillChoices | None = None,
    ) -> None:
        """Команды ``commands``, источники аргументов ``arg_choices`` и скиллы ``skills``."""
        self._commands = commands
        self._arg_choices = {name.lower(): source for name, source in arg_choices.items()}
        self._skills = skills

    @override
    def get_completions(
        self,
        document: Document,
        complete_event: CompleteEvent,
    ) -> Iterable[Completion]:
        text = document.text_before_cursor
        if _COMMAND_PREFIX_RE.fullmatch(text):
            yield from self._complete_command(text)
        else:
            match = _COMMAND_ARG_RE.fullmatch(text)
            if match is not None:
                yield from self._complete_arg(match.group(1), match.group(2))

    def _complete_arg(self, name: str, prefix: str) -> Iterable[Completion]:
        command = self._commands.get(name)
        source = None if command is None else self._arg_choices.get(command.name.lower())
        if source is None:
            return
        needle = prefix.lower()
        choices = list(source())
        starts = [choice for choice in choices if choice[0].lower().startswith(needle)]
        inside = [
            choice for choice in choices if needle in choice[0].lower() and choice not in starts
        ]
        for choice_text, meta in starts + inside:
            yield Completion(choice_text, start_position=-len(prefix), display_meta=meta)

    def _complete_command(self, text: str) -> Iterable[Completion]:
        prefix = text.lower()
        for cmd in self._commands:
            for name in (cmd.name, *cmd.aliases):
                if name.lower().startswith(prefix):
                    yield Completion(name, start_position=-len(text), display_meta=cmd.summary)
                    break
        if self._skills is None:
            return
        for name, description in self._skills():
            if name.lower().startswith(prefix) and self._commands.get(name) is None:
                meta = f"скилл · {description}" if description else "скилл"
                yield Completion(name, start_position=-len(text), display_meta=meta)


# --------------------------------- история --------------------------------- #
class LazyFileHistory(FileHistory):
    """История в ``.gigacode/history``; папка создаётся при первом сохранении.

    Если записать не удалось (проект только для чтения, история — симлинк из
    репозитория), история остаётся в памяти до конца сессии — ошибка не должна
    закрывать REPL.
    """

    def __init__(self, workspace: Workspace) -> None:
        """История ввода проекта ``workspace``."""
        self._workspace = workspace
        self._persist = True
        history_file = workspace.history_file
        self._path = history_file
        super().__init__(history_file)

    @override
    def load_history_strings(self) -> Iterable[str]:
        # Симлинк на месте папки или файла истории принёс репозиторий: это не наша история
        # (писать по нему не даст Workspace.ensure_data_dir).
        if self._workspace.data_dir.is_symlink() or self._path.is_symlink():
            return []
        return super().load_history_strings()

    @override
    def store_string(self, string: str) -> None:
        if not self._persist:
            return
        try:
            self._workspace.ensure_data_dir()
            if not self._path.exists():
                # В истории бывают вставленные секреты — права как у чатов (0600).
                flags = os.O_CREAT | os.O_WRONLY | os.O_APPEND
                os.close(os.open(self._path, flags, _HISTORY_FILE_MODE))
            super().store_string(string)
        except OSError:
            self._persist = False


# --------------------------------- клавиши --------------------------------- #
def _cycle_mode(on_cycle_mode: Callable[[], object], event: KeyPressEvent) -> None:
    """Shift+Tab: сменить режим разрешений."""
    on_cycle_mode()
    event.app.invalidate()  # статус-строка с новым режимом — сразу


def _clear(event: KeyPressEvent) -> None:
    """Esc Esc: очистить ввод."""
    buffer = event.current_buffer
    if buffer.text.strip():
        buffer.reset(append_to_history=True)  # черновик не теряется: ↑ вернёт его
    else:
        buffer.reset()


def _newline(event: KeyPressEvent) -> None:
    """Alt+Enter: новая строка."""
    event.current_buffer.insert_text("\n")


def _continue_line(event: KeyPressEvent) -> None:
    """Enter после обратной косой черты: она заменяется новой строкой."""
    buffer = event.current_buffer
    buffer.delete_before_cursor(1)
    buffer.insert_text("\n")


def _key_bindings(on_cycle_mode: Callable[[], object] | None = None) -> KeyBindings:
    bindings = KeyBindings()
    if on_cycle_mode is not None:
        # При открытом меню дополнения Shift+Tab остаётся штатным «назад по меню».
        bindings.add("s-tab", filter=~has_completions)(partial(_cycle_mode, on_cycle_mode))
    bindings.add("escape", "escape")(_clear)
    bindings.add("escape", "enter")(_newline)
    line_continues = Condition(
        lambda: get_app().current_buffer.document.text_before_cursor.endswith("\\"),
    )
    bindings.add("enter", filter=line_continues)(_continue_line)
    return bindings


def _continuation(width: int, line_number: int, wrap_count: int) -> StyleAndTextTuples:
    marker = "·".rjust(max(width - 1, 1))
    return [("class:continuation", f"{marker} ")]


# --------------------------------- сборка ---------------------------------- #
# Параметры — только по имени: источники автодополнения и подмены ввода для тестов.
def create_prompt_session(  # noqa: WPS211
    *,
    commands: CommandRegistry,
    workspace: Workspace,
    status: Callable[[], StatusInfo],
    no_color: bool = False,
    arg_choices: Mapping[str, ArgChoices] = _NO_ARG_CHOICES,
    on_cycle_mode: Callable[[], object] | None = None,
    skills: SkillChoices | None = None,
    **kwargs: Any,
) -> PromptSession[str]:
    """Сессия ввода. ``kwargs`` (``input``/``output``) подменяются в тестах.

    ``on_cycle_mode`` — что сделать по Shift+Tab (сменить режим агента);
    ``skills`` — скиллы для автодополнения.
    """
    return PromptSession(
        message=[("class:prompt", PROMPT)],
        history=LazyFileHistory(workspace),
        auto_suggest=AutoSuggestFromHistory(),
        completer=SlashCommandCompleter(commands, arg_choices, skills),
        complete_while_typing=Condition(lambda: get_app().current_buffer.text.startswith("/")),
        reserve_space_for_menu=min(8, len(list(commands)) + 1),
        key_bindings=_key_bindings(on_cycle_mode),
        bottom_toolbar=lambda: _bottom_toolbar(status()),
        prompt_continuation=_continuation,
        placeholder=[("class:placeholder", PLACEHOLDER)],
        style=prompt_style(),
        color_depth=prompt_color_depth(no_color=no_color),
        **kwargs,
    )


def _read_plain(default: str = "") -> str:
    """Строка через ``input()``: набранное заранее — после приглашения.

    ``input()`` не умеет заполнять строку: набранное заранее печатается после
    приглашения и приклеивается к введённому.
    """
    # Запасной ввод без терминала (pipe, CI) — и есть назначение ``input()`` здесь.
    return default + input(PLAIN_PROMPT + default)  # noqa: WPS421


def _read_line(session: PromptSession[str], default: str = "") -> str:
    """Строка из prompt_toolkit; ``default`` — текст, которым заполнить ввод."""
    return session.prompt(default=default)


def make_input_reader(
    *,
    commands: CommandRegistry,
    workspace: Workspace,
    status: Callable[[], StatusInfo],
    no_color: bool = False,
    arg_choices: Mapping[str, ArgChoices] = _NO_ARG_CHOICES,
    on_cycle_mode: Callable[[], object] | None = None,
    skills: SkillChoices | None = None,
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
            skills=skills,
        )
    except Exception:  # терминал не поддерживается (например, mintty без консоли)
        return _read_plain
    return partial(_read_line, session)
