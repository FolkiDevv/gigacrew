"""Меню настроек ``/settings``: просмотр и правка ``~/.gigacode/gigacrew.json``.

Встроенное (не полноэкранное) приложение prompt_toolkit, как выбор чата
(:mod:`gigacrew.ui.chat_picker`): ↑↓ — настройка, Enter/Пробел — переключить флаг,
следующий вариант или ввести число и имя модели, ←→ — варианты, Del — значение по
умолчанию, Esc — выход. Каждое изменение сразу уходит обработчику (запись в файл и
применение) — отдельного «сохранить» нет. Сверху — подключение (только просмотр: оно
задаётся переменными окружения и ``.env``), снизу — пояснение к настройке под курсором.
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from prompt_toolkit.application import Application, get_app
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.key_binding.key_bindings import KeyHandlerCallable
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import FormattedTextControl, Layout, Window
from prompt_toolkit.utils import get_cwidth

from gigacrew.project.preferences import SECTIONS, Setting, SettingKind, SettingSpec
from gigacrew.ui.colors import prompt_color_depth, prompt_style

__all__ = [
    "Change",
    "ChangeHandler",
    "SettingRow",
    "SettingsState",
    "edit_settings",
    "render",
]

WINDOW = 14  # сколько строк списка видно одновременно
HINT = "↑↓ выбор · Enter — изменить · ←→ — варианты · Del — по умолчанию · Esc — выход"
EDIT_HINT = "Enter — сохранить · Esc — отмена · пусто — значение по умолчанию"
_ESC_TIMEOUT = 0.05
_MIN_WIDTH = 30
_LABEL_WIDTH = 32  # подпись самой длинной настройки и отступ
_HINT_STYLE = "class:choice.hint"
_DESCRIPTION_STYLE = "class:choice.description"
_ERROR_STYLE = "class:choice.error"
_NEWLINE = ("", "\n")
_INDENT = "   "
_EDITABLE = frozenset((SettingKind.INTEGER, SettingKind.NUMBER, SettingKind.MODEL))
_CYCLED = frozenset((SettingKind.CHOICE, SettingKind.MODE))


@dataclass(frozen=True)
class SettingRow:
    """Настройка в меню: действующее значение и откуда оно."""

    spec: SettingSpec
    setting: Setting  # действующее значение
    stored: bool = False  # задано в gigacrew.json
    source: str | None = None  # перекрыто переменной: откуда она (окружение, .env…)
    choices: tuple[str, ...] = ()  # известные модели (для MODEL)

    def options(self) -> tuple[Setting, ...]:
        """Варианты для ←→: флаг, список выбора, модели каталога (и «не задана»)."""
        spec = self.spec
        if spec.kind is SettingKind.FLAG:
            return (True, False)
        if spec.kind in _CYCLED:
            return spec.choices
        if spec.kind is SettingKind.MODEL and self.choices:
            return (None, *self.choices)
        return ()


@dataclass(frozen=True)
class Change:
    """Итог изменения: обновлённая строка и сообщение (``failed`` — ошибка)."""

    row: SettingRow
    message: str = ""
    failed: bool = False


# Обработчик изменения: строка и новое значение → итог (запись в файл и применение).
ChangeHandler = Callable[[SettingRow, Setting], Change]


@dataclass
class SettingsState:
    """Состояние меню: строки настроек, курсор, окно прокрутки, ввод и сообщение."""

    rows: list[SettingRow]
    on_change: ChangeHandler
    title: str = "Настройки"
    connection: Sequence[tuple[str, str]] = ()  # подключение: (подпись, значение)
    cursor: int = 0  # индекс в rows
    offset: int = 0  # первая видимая строка списка (с заголовками разделов)
    window: int = WINDOW
    editing: str | None = None  # текст ввода; None — не вводится
    status: str = ""
    failed: bool = False
    _lines: list[SettingRow | str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        """Строки списка: заголовок раздела перед его первой настройкой."""
        section = ""
        for row in self.rows:
            if row.spec.section != section:
                section = row.spec.section
                self._lines.append(SECTIONS.get(section, section))
            self._lines.append(row)

    @property
    def lines(self) -> list[SettingRow | str]:
        """Строки списка: настройки и заголовки разделов (строки)."""
        return self._lines

    @property
    def selected(self) -> SettingRow:
        """Настройка под курсором."""
        return self.rows[self.cursor]

    def move(self, step: int) -> None:
        """Стрелки — по кругу; ввод отменяется."""
        self.editing = None
        self.cursor = (self.cursor + step) % len(self.rows)
        self._scroll()

    def jump(self, step: int) -> None:
        """PgUp/PgDn/Home/End — без перехода через край."""
        self.editing = None
        self.cursor = min(max(self.cursor + step, 0), len(self.rows) - 1)
        self._scroll()

    def activate(self) -> None:
        """Enter/Пробел: флаг — переключить, выбор — следующий вариант, иначе — ввод."""
        row = self.selected
        if row.spec.kind in _EDITABLE:
            self.editing = row.spec.text(row.setting)
            self.status = ""
            return
        self.cycle(1)

    def cycle(self, step: int) -> None:
        """Следующий (``step`` = 1) или предыдущий вариант значения."""
        options = self.selected.options()
        if not options or self.editing is not None:
            return
        current = self.selected.setting
        index = options.index(current) if current in options else -1
        self._change(options[(index + step) % len(options)])

    def reset(self) -> None:
        """Del: значение по умолчанию (настройка уходит из файла)."""
        if self.editing is None:
            self._change(self.selected.spec.default)

    def type_text(self, text: str) -> None:
        """Набор в строке ввода."""
        if self.editing is not None:
            self.editing += text

    def erase(self) -> None:
        """Backspace в строке ввода."""
        if self.editing:
            self.editing = self.editing[:-1]

    def clear(self) -> None:
        """Ctrl+U: очистить строку ввода."""
        if self.editing is not None:
            self.editing = ""

    def commit(self) -> None:
        """Enter в строке ввода: проверить и сохранить; ошибка — ввод остаётся."""
        if self.editing is None:
            return
        try:
            setting = self.selected.spec.parse(self.editing)
        except ValueError as error:
            self.status = str(error)
            self.failed = True
            return
        self.editing = None
        self._change(setting)

    def cancel(self) -> None:
        """Esc в строке ввода: выйти из ввода без изменений."""
        self.editing = None
        self.status = ""
        self.failed = False

    def _change(self, setting: Setting) -> None:
        change = self.on_change(self.selected, setting)
        line = self._lines.index(self.selected)
        self.rows[self.cursor] = change.row
        self._lines[line] = change.row
        self.status = change.message
        self.failed = change.failed

    def _scroll(self) -> None:
        line = self._lines.index(self.selected)
        # Заголовок раздела — вместе с первой настройкой раздела.
        top = line - 1 if line and isinstance(self._lines[line - 1], str) else line
        if top < self.offset:
            self.offset = top
        elif line >= self.offset + self.window:
            self.offset = line - self.window + 1


def _fit(text: str, width: int) -> str:
    """Обрезает по ширине терминала (с учётом широких символов)."""
    if get_cwidth(text) <= width:
        return text
    out, used = "", 0
    for ch in text:
        char_width = get_cwidth(ch)
        if used + char_width > width - 1:
            break
        out += ch
        used += char_width
    return f"{out}…"


def _connection(state: SettingsState, width: int) -> list[tuple[str, str]]:
    """Подключение — только просмотр: оно задаётся окружением и .env."""
    if not state.connection:
        return []
    out = [(_HINT_STYLE, " Подключение (переменные окружения и .env, только просмотр)\n")]
    for label, text in state.connection:
        line = f"{_INDENT}{label}: {text}"
        out.extend(((_DESCRIPTION_STYLE, _fit(line, width)), _NEWLINE))
    return out


def _value_cell(state: SettingsState, row: SettingRow, *, current: bool) -> tuple[str, str]:
    """Значение настройки: в строке ввода — набираемый текст."""
    if current and state.editing is not None:
        return ("class:choice.custom", f"{state.editing}▏")
    style = "class:choice.checked" if row.stored and row.source is None else "class:choice"
    return (style, row.spec.show(row.setting))


def _row_line(state: SettingsState, row: SettingRow, width: int) -> list[tuple[str, str]]:
    """Строка настройки: указатель, подпись, значение и источник, если она перекрыта."""
    current = row is state.selected
    pointer = "❯" if current else " "
    name = row.spec.label
    label = f" {pointer} {name}"
    pad = max(_LABEL_WIDTH - get_cwidth(label), 1)
    style = "class:choice.selected" if current else "class:choice"
    cell_style, cell = _value_cell(state, row, current=current)
    tag = f"  перекрыто: {row.source}" if row.source else ""
    room = max(width - get_cwidth(label) - pad - get_cwidth(tag), _MIN_WIDTH // 2)
    return [
        (style, label + " " * pad),
        (cell_style, _fit(cell, room)),
        (_HINT_STYLE, tag),
        _NEWLINE,
    ]


def _list(state: SettingsState, width: int) -> list[tuple[str, str]]:
    """Окно списка со стрелками «ещё N» сверху и снизу."""
    lines = state.lines
    end = min(state.offset + state.window, len(lines))
    out: list[tuple[str, str]] = []
    if state.offset:
        out.append((_HINT_STYLE, f"   ↑ ещё {state.offset}\n"))
    for line in lines[state.offset : end]:
        if isinstance(line, str):
            out.append(("class:question.counter", f" {line}\n"))
        else:
            out.extend(_row_line(state, line, width))
    rest = len(lines) - end
    if rest:
        out.append((_HINT_STYLE, f"   ↓ ещё {rest}\n"))
    return out


def _details(state: SettingsState, width: int) -> list[tuple[str, str]]:
    """Пояснение к настройке под курсором, переменная окружения и сообщение."""
    row = state.selected
    spec = row.spec
    text = spec.hint
    if spec.restart:
        text = f"{text} Действует со следующего запуска."
    out = [
        (_DESCRIPTION_STYLE, f"{_INDENT}{line}\n")
        for line in textwrap.wrap(text, max(width - len(_INDENT), _MIN_WIDTH))
    ]
    env_note = f"переменная {spec.env}"
    if row.source:
        env_note = f"{env_note} задана ({row.source}) и главнее файла"
    out.extend(((_HINT_STYLE, _fit(f"{_INDENT}{env_note}", width)), _NEWLINE))
    if state.status:
        style = _ERROR_STYLE if state.failed else "class:choice.checked"
        out.extend(((style, _fit(f" {state.status}", width)), _NEWLINE))
    return out


def render(state: SettingsState, width: int = 80) -> list[tuple[str, str]]:
    """Отрисовка меню: список пар (стиль prompt_toolkit, текст)."""
    width = max(width - 1, _MIN_WIDTH)  # последний столбец — без автопереноса строки
    out: list[tuple[str, str]] = [("class:question.counter", f" {state.title}"), _NEWLINE]
    out += _connection(state, width)
    out += _list(state, width)
    out += _details(state, width)
    hint = HINT if state.editing is None else EDIT_HINT
    out.append((_HINT_STYLE, _fit(f" {hint}", width)))
    return out


def _bind(kb: KeyBindings, handler: KeyHandlerCallable, *keys: Keys | str) -> None:
    """Привязывает ``handler`` к каждой из клавиш ``keys`` по порядку."""
    for key in keys:
        kb.add(key)(handler)


def _run(action: Callable[[], None], event: KeyPressEvent) -> None:
    """Обработчик клавиши, которому событие не нужно."""
    action()


def _on(kb: KeyBindings, action: Callable[[], None], *keys: str) -> None:
    """Клавиши ``keys`` вызывают ``action``."""
    _bind(kb, partial(_run, action), *keys)


class _SettingsKeys:
    """Обработчики клавиш, зависящие от строки ввода; Esc вне ввода закрывает меню."""

    def __init__(self, state: SettingsState) -> None:
        self._state = state

    def bindings(self) -> KeyBindings:
        """Привязки клавиш меню."""
        state = self._state
        kb = KeyBindings()
        self._bind_moves(kb)
        _bind(kb, self._type, Keys.BracketedPaste, Keys.Any)
        _on(kb, state.erase, "backspace")
        _on(kb, state.reset, "delete")
        _on(kb, state.clear, "c-u")
        kb.add("enter")(self._enter)
        _bind(kb, self._close, "c-d", "c-c")
        kb.add("escape", eager=True)(self._escape)
        return kb

    def _bind_moves(self, kb: KeyBindings) -> None:
        """Клавиши перемещения курсора и перебора вариантов."""
        state = self._state
        _on(kb, partial(state.move, -1), "s-tab", "c-p", "up")
        _on(kb, partial(state.move, 1), "tab", "c-n", "down")
        _on(kb, lambda: state.jump(-state.window), "pageup")
        _on(kb, lambda: state.jump(state.window), "pagedown")
        _on(kb, lambda: state.jump(-len(state.rows)), "home")
        _on(kb, lambda: state.jump(len(state.rows)), "end")
        _on(kb, partial(state.cycle, -1), "left")
        _on(kb, partial(state.cycle, 1), "right")

    def _type(self, event: KeyPressEvent) -> None:
        typed = "".join(ch for ch in event.data if ch.isprintable())
        if self._state.editing is not None:
            self._state.type_text(typed)
        elif typed == " ":
            self._state.activate()

    def _enter(self, event: KeyPressEvent) -> None:
        if self._state.editing is None:
            self._state.activate()
        else:
            self._state.commit()

    def _escape(self, event: KeyPressEvent) -> None:
        if self._state.editing is None:
            event.app.exit()
        else:
            self._state.cancel()

    def _close(self, event: KeyPressEvent) -> None:
        event.app.exit()


def edit_settings(
    state: SettingsState,
    *,
    no_color: bool = False,
    **io: Any,
) -> None:
    """Меню настроек до выхода (Esc); изменения уходят в ``state.on_change`` сразу.

    ``io`` — ``input``/``output`` prompt_toolkit (подменяются в тестах).
    """
    kb = _SettingsKeys(state).bindings()
    control = FormattedTextControl(
        lambda: list(render(state, get_app().output.get_size().columns)),
        focusable=True,
        show_cursor=False,
    )
    app: Application[None] = Application(
        layout=Layout(Window(control, wrap_lines=True, dont_extend_height=True)),
        key_bindings=kb,
        style=prompt_style(),
        color_depth=prompt_color_depth(no_color=no_color),
        full_screen=False,
        erase_when_done=True,
        **io,
    )
    app.ttimeoutlen = _ESC_TIMEOUT
    app.run()
