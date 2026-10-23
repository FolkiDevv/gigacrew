"""Меню выбора ответа на вопрос агента (инструмент ``ask_user``).

Встроенное (не полноэкранное) приложение prompt_toolkit: ↑↓ — выбор, цифры — сразу
выбрать (в мультивыборе — отметить), Пробел — отметить, Enter — подтвердить,
Esc/Ctrl+C — отказаться отвечать. Последний пункт — «Свой ответ…» (подпись задаёт
вопрос: у плана — «Замечания…»): когда он выбран, набираемый текст попадает прямо в
него (Backspace — стереть, Ctrl+U — очистить). Слева — полоса, как у подтверждения
команды. После ответа меню стирается — итог печатает вызывающий код.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from prompt_toolkit.application import Application, get_app
from prompt_toolkit.filters import Condition, FilterOrBool
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.key_binding.key_bindings import KeyHandlerCallable
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import FormattedTextControl, Layout, Window

from gigacrew.tools.questions import Answer, Question
from gigacrew.ui.colors import prompt_color_depth, prompt_style

CUSTOM_LABEL = "Свой ответ…"
CUSTOM_DESCRIPTION = "Начните печатать — ответ своими словами"
_ESC_TIMEOUT = 0.05  # одиночный Esc срабатывает сразу, без ожидания Alt-последовательности
_MIN_WRAP_WIDTH = 20
_BOX_WIDTH = 4  # ширина отметки «[✔] » в мультивыборе
_DIGIT_KEYS = 9  # цифры 1–9 выбирают пункт сразу
_SELECTED_STYLE = "class:choice.selected"
_DESCRIPTION_STYLE = "class:choice.description"
_BOX_STYLE = "class:choice.box"
_HINT_STYLE = "class:choice.hint"


@dataclass
class ChoiceState:
    """Состояние меню: вопрос, курсор, отмеченные пункты и текст своего ответа."""

    question: Question
    index: int  # номер вопроса (с 1)
    total: int
    cursor: int = 0
    checked: set[int] = field(default_factory=set)
    custom: str = ""  # текст своего ответа
    error: str = ""

    @property
    def custom_index(self) -> int:
        """Номер пункта «Свой ответ…» — последнего в меню."""
        return len(self.question.options)

    @property
    def size(self) -> int:
        """Число пунктов меню вместе со «Свой ответ…»."""
        return len(self.question.options) + 1

    @property
    def on_custom(self) -> bool:
        """Курсор на пункте «Свой ответ…»: набираемый текст попадает в него."""
        return self.cursor == self.custom_index

    def answer(self) -> Answer | None:
        """Ответ по текущему состоянию (Enter); None — ответа ещё нет."""
        custom = self.custom.strip()
        options = self.question.options
        if self.question.multi_select:
            picked = sorted(number for number in self.checked if number < self.custom_index)
            if not picked and not custom and not self.on_custom:
                picked = [self.cursor]  # ничего не отмечено — берём текущий
        else:
            if not self.on_custom:
                return Answer((options[self.cursor].label,))
            picked = []
        if not picked and not custom:
            self.error = "введите свой ответ или выберите вариант"
            return None
        return Answer(tuple(options[number].label for number in picked), custom)


_LEFT_BAR = "  ▌ "  # полоса слева — как у подтверждения команды: всё, что ждёт ответа


def _wrap(text: str, width: int, indent: str = "") -> list[str]:
    room = max(width - len(_LEFT_BAR) - 1, _MIN_WRAP_WIDTH)
    return textwrap.wrap(text, room, initial_indent=indent, subsequent_indent=indent)


def _line(*parts: tuple[str, str]) -> list[tuple[str, str]]:
    """Строка меню: полоса слева, части, перевод строки."""
    return [("class:question.bar", _LEFT_BAR), *parts, ("", "\n")]


def custom_label(question: Question) -> str:
    """Подпись пункта «свой ответ»: заданная вопросом или общая."""
    return question.custom_label or CUSTOM_LABEL


def _question_lines(state: ChoiceState, width: int) -> list[tuple[str, str]]:
    """Текст вопроса с меткой темы и номером вопроса, пометка о мультивыборе."""
    question = state.question
    tag = question.header
    if state.total > 1:
        counter = f"{state.index} из {state.total}"
        tag = f"{tag} · {counter}" if tag else counter
    text_lines = _wrap(question.text, width) or [" "]
    out: list[tuple[str, str]] = []
    for line_index, line in enumerate(text_lines):
        mark = ("class:question.mark", "? " if line_index == 0 else "  ")
        parts = [mark, ("class:question.text", line)]
        if tag and line_index == len(text_lines) - 1:
            parts.append(("class:question.tag", f"   {tag}"))
        out += _line(*parts)
    if question.multi_select:
        out += _line((_HINT_STYLE, "можно выбрать несколько вариантов"))
    out += _line()
    return out


def _box(state: ChoiceState, position: int) -> list[tuple[str, str]]:
    """Отметка пункта в мультивыборе: «[✔] », «[ ] », у своего ответа — пробелы."""
    if not state.question.multi_select:
        return []
    if position == state.custom_index:
        return [("", "    ")]
    if position in state.checked:
        return [
            (_BOX_STYLE, "["),
            ("class:choice.box.checked", "✔"),
            (_BOX_STYLE, "] "),
        ]
    return [(_BOX_STYLE, "[ ] ")]


def _custom_parts(state: ChoiceState, *, current: bool) -> list[tuple[str, str]]:
    """Набранный свой ответ после подписи пункта; под курсором — с курсором ввода."""
    label = custom_label(state.question).rstrip("…")
    parts = [(_DESCRIPTION_STYLE, f"{label}: "), ("class:choice.custom", state.custom)]
    if current:
        parts.append(("[SetCursorPosition]", ""))
    return parts


def _label_style(state: ChoiceState, position: int, *, current: bool) -> str:
    """Стиль подписи пункта: под курсором, отмеченный или обычный."""
    if current:
        return _SELECTED_STYLE
    if state.question.multi_select and position in state.checked:
        return "class:choice.checked"
    return "class:choice"


def _description_lines(
    description: str,
    width: int,
    indent: str,
    *,
    current: bool,
) -> list[tuple[str, str]]:
    """Описание пункта с отступом под подписью; у пункта под курсором — ярче."""
    faint = _DESCRIPTION_STYLE if current else "class:choice.description.faint"
    lines: list[tuple[str, str]] = []
    for line in _wrap(description, width, indent):
        lines.extend(_line((faint, line)))
    return lines


def _option_lines(
    state: ChoiceState,
    position: int,
    entry: tuple[str, str],
    width: int,
) -> list[tuple[str, str]]:
    """Строки пункта меню: указатель, номер, отметка, подпись и описание под ней."""
    label, description = entry
    current = position == state.cursor
    ordinal = position + 1
    number = f"{ordinal}. "
    parts = [
        (_SELECTED_STYLE, "❯ " if current else "  "),
        ("class:choice.number.selected" if current else "class:choice.number", number),
        *_box(state, position),
    ]
    if position == state.custom_index and (state.custom or current):
        lines = _line(*parts, *_custom_parts(state, current=current))
        if state.custom:
            return lines
    else:
        lines = _line(*parts, (_label_style(state, position, current=current), label))
    if description:
        box_width = _BOX_WIDTH if state.question.multi_select else 0
        indent = " " * (2 + len(number) + box_width)
        lines += _description_lines(description, width, indent, current=current)
    return lines


def _hint_lines(state: ChoiceState, width: int) -> list[tuple[str, str]]:
    """Подсказка по клавишам внизу меню (без перевода строки в конце)."""
    if state.question.multi_select:
        hint = (
            f"отмечено {len(state.checked)} · Пробел/цифра — отметить · Enter — готово · "
            "Esc — отказаться"
        )
    else:
        hint = "↑↓ выбор · цифра — сразу · Enter — подтвердить · Esc — отказаться"
    hint_lines = _wrap(hint, width)
    out: list[tuple[str, str]] = []
    for line_index, line in enumerate(hint_lines):
        out += [("class:question.bar", _LEFT_BAR), (_HINT_STYLE, line)]
        if line_index < len(hint_lines) - 1:
            out.append(("", "\n"))
    return out


def render(state: ChoiceState, width: int = 80) -> list[tuple[str, str]]:
    """Отрисовка меню: список пар (стиль prompt_toolkit, текст)."""
    question = state.question
    out = _question_lines(state, width)
    entries = [(option.label, option.description) for option in question.options]
    entries.append((custom_label(question), question.custom_hint or CUSTOM_DESCRIPTION))
    for position, entry in enumerate(entries):
        out.extend(_option_lines(state, position, entry, width))
    out += _line()
    if state.error:
        out += _line(("class:choice.error", state.error))
    out += _hint_lines(state, width)
    return out


def _bind(
    kb: KeyBindings,
    handler: KeyHandlerCallable,
    *keys: Keys | str,
    active: FilterOrBool = True,
) -> None:
    """Привязывает ``handler`` к каждой из клавиш ``keys`` по порядку."""
    for key in keys:
        kb.add(key, filter=active)(handler)


class _ChoiceKeys:
    """Обработчики клавиш меню: меняют состояние, Enter и Esc закрывают меню."""

    def __init__(self, state: ChoiceState) -> None:
        self._state = state
        self._multi = state.question.multi_select
        # Печать своего ответа: курсор на пункте «Свой ответ…».
        self._typing = Condition(lambda: state.on_custom)

    def bindings(self) -> KeyBindings:
        """Привязки клавиш меню."""
        kb = KeyBindings()
        _bind(kb, self._up, "s-tab", "c-p", "up")
        _bind(kb, self._down, "tab", "c-n", "down")
        for position in range(min(self._state.size, _DIGIT_KEYS)):
            kb.add(str(position + 1), filter=~self._typing)(partial(self._digit, position))
        kb.add("space", filter=~self._typing)(self._space)
        _bind(kb, self._type, Keys.BracketedPaste, Keys.Any, active=self._typing)
        kb.add("backspace", filter=self._typing)(self._erase)
        kb.add("c-u", filter=self._typing)(self._clear)
        kb.add("enter")(self._enter)
        _bind(kb, self._cancel, "c-d", "c-c")
        kb.add("escape", eager=True)(self._cancel)
        return kb

    def _move(self, step: int) -> None:
        self._state.cursor = (self._state.cursor + step) % self._state.size
        self._state.error = ""

    def _up(self, event: KeyPressEvent) -> None:
        self._move(-1)

    def _down(self, event: KeyPressEvent) -> None:
        self._move(1)

    def _digit(self, position: int, event: KeyPressEvent) -> None:
        state = self._state
        state.cursor = position
        state.error = ""
        if position == state.custom_index:
            return  # дальше — печать своего ответа
        if self._multi:
            state.checked ^= {position}
        else:
            event.app.exit(result=Answer((state.question.options[position].label,)))

    def _space(self, event: KeyPressEvent) -> None:
        if self._multi:
            self._state.checked ^= {self._state.cursor}

    def _type(self, event: KeyPressEvent) -> None:
        text = event.data.replace("\r", " ").replace("\n", " ")
        self._state.custom += "".join(ch for ch in text if ch.isprintable())
        self._state.error = ""

    def _erase(self, event: KeyPressEvent) -> None:
        self._state.custom = self._state.custom[:-1]

    def _clear(self, event: KeyPressEvent) -> None:
        self._state.custom = ""

    def _enter(self, event: KeyPressEvent) -> None:
        answer = self._state.answer()
        if answer is not None:
            event.app.exit(result=answer)

    def _cancel(self, event: KeyPressEvent) -> None:
        event.app.exit(result=None)


def choose(
    question: Question,
    index: int,
    total: int,
    *,
    no_color: bool = False,
    **io: Any,
) -> Answer | None:
    """Задаёт один вопрос; None — пользователь отказался отвечать.

    ``io`` — ``input``/``output`` prompt_toolkit (подменяются в тестах).
    """
    state = ChoiceState(question, index, total)
    kb = _ChoiceKeys(state).bindings()
    control = FormattedTextControl(
        lambda: list(render(state, get_app().output.get_size().columns)),
        focusable=True,
        show_cursor=False,
    )
    app: Application[Answer | None] = Application(
        layout=Layout(Window(control, wrap_lines=True, dont_extend_height=True)),
        key_bindings=kb,
        style=prompt_style(),
        color_depth=prompt_color_depth(no_color=no_color),
        full_screen=False,
        erase_when_done=True,
        **io,
    )
    app.ttimeoutlen = _ESC_TIMEOUT
    return app.run()
