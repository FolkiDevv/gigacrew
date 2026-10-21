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
from typing import Any

from prompt_toolkit.application import Application, get_app
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import FormattedTextControl, Layout, Window

from gigacrew.tools.questions import Answer, Question
from gigacrew.ui.colors import prompt_color_depth, prompt_style

CUSTOM_LABEL = "Свой ответ…"
CUSTOM_DESCRIPTION = "Начните печатать — ответ своими словами"
_ESC_TIMEOUT = 0.05  # одиночный Esc срабатывает сразу, без ожидания Alt-последовательности


@dataclass
class ChoiceState:
    question: Question
    index: int  # номер вопроса (с 1)
    total: int
    cursor: int = 0
    checked: set[int] = field(default_factory=set)
    custom: str = ""  # текст своего ответа
    error: str = ""

    @property
    def custom_index(self) -> int:
        return len(self.question.options)

    @property
    def size(self) -> int:
        return len(self.question.options) + 1

    @property
    def on_custom(self) -> bool:
        return self.cursor == self.custom_index

    def answer(self) -> Answer | None:
        """Ответ по текущему состоянию (Enter); None — ответа ещё нет."""
        custom = self.custom.strip()
        options = self.question.options
        if self.question.multi_select:
            picked = sorted(k for k in self.checked if k < self.custom_index)
            if not picked and not custom and not self.on_custom:
                picked = [self.cursor]  # ничего не отмечено — берём текущий
        else:
            if not self.on_custom:
                return Answer((options[self.cursor].label,))
            picked = []
        if not picked and not custom:
            self.error = "введите свой ответ или выберите вариант"
            return None
        return Answer(tuple(options[k].label for k in picked), custom)


BAR = "  ▌ "  # полоса слева — как у подтверждения команды: всё, что ждёт ответа


def _wrap(text: str, width: int, indent: str = "") -> list[str]:
    room = max(width - len(BAR) - 1, 20)
    return textwrap.wrap(text, room, initial_indent=indent, subsequent_indent=indent)


def _line(*parts: tuple[str, str]) -> list[tuple[str, str]]:
    """Строка меню: полоса слева, части, перевод строки."""
    return [("class:question.bar", BAR), *parts, ("", "\n")]


def custom_label(question: Question) -> str:
    return question.custom_label or CUSTOM_LABEL


def render(state: ChoiceState, width: int = 80) -> list[tuple[str, str]]:
    """Отрисовка меню: список пар (стиль prompt_toolkit, текст)."""
    q = state.question
    multi = q.multi_select
    out: list[tuple[str, str]] = []
    tag = q.header
    if state.total > 1:
        counter = f"{state.index} из {state.total}"
        tag = f"{tag} · {counter}" if tag else counter
    text_lines = _wrap(q.text, width) or [" "]
    for i, line in enumerate(text_lines):
        mark = ("class:question.mark", "? " if i == 0 else "  ")
        parts = [mark, ("class:question.text", line)]
        if tag and i == len(text_lines) - 1:
            parts.append(("class:question.tag", f"   {tag}"))
        out += _line(*parts)
    if multi:
        out += _line(("class:choice.hint", "можно выбрать несколько вариантов"))
    out += _line()

    items = [(o.label, o.description) for o in q.options]
    items.append((custom_label(q), q.custom_hint or CUSTOM_DESCRIPTION))
    for k, (label, description) in enumerate(items):
        current = k == state.cursor
        is_custom = k == state.custom_index
        number = f"{k + 1}. "
        parts: list[tuple[str, str]] = [
            ("class:choice.selected", "❯ " if current else "  "),
            ("class:choice.number.selected" if current else "class:choice.number", number),
        ]
        box_width = 0
        if multi:
            box_width = 4
            if is_custom:
                parts.append(("", "    "))
            elif k in state.checked:
                parts += [
                    ("class:choice.box", "["),
                    ("class:choice.box.checked", "✔"),
                    ("class:choice.box", "] "),
                ]
            else:
                parts.append(("class:choice.box", "[ ] "))
        if is_custom and (state.custom or current):
            prefix = custom_label(q).rstrip("…") + ": "
            parts += [("class:choice.description", prefix), ("class:choice.custom", state.custom)]
            if current:
                parts.append(("[SetCursorPosition]", ""))
            out += _line(*parts)
            if state.custom:
                continue
        else:
            if current:
                style = "class:choice.selected"
            elif multi and k in state.checked:
                style = "class:choice.checked"
            else:
                style = "class:choice"
            out += _line(*parts, (style, label))
        if description:
            indent = " " * (2 + len(number) + box_width)
            faint = "class:choice.description" if current else "class:choice.description.faint"
            for line in _wrap(description, width, indent):
                out += _line((faint, line))

    out += _line()
    if state.error:
        out += _line(("class:choice.error", state.error))
    if multi:
        hint = (
            f"отмечено {len(state.checked)} · Пробел/цифра — отметить · Enter — готово · "
            "Esc — отказаться"
        )
    else:
        hint = "↑↓ выбор · цифра — сразу · Enter — подтвердить · Esc — отказаться"
    hint_lines = _wrap(hint, width)
    for i, line in enumerate(hint_lines):
        out += [("class:question.bar", BAR), ("class:choice.hint", line)]
        if i < len(hint_lines) - 1:
            out.append(("", "\n"))
    return out


def choose(
    question: Question, index: int, total: int, *, no_color: bool = False, **io: Any
) -> Answer | None:
    """Задаёт один вопрос; None — пользователь отказался отвечать.

    ``io`` — ``input``/``output`` prompt_toolkit (подменяются в тестах).
    """
    state = ChoiceState(question, index, total)
    kb = KeyBindings()
    typing = Condition(lambda: state.on_custom)
    multi = question.multi_select

    def move(step: int) -> None:
        state.cursor = (state.cursor + step) % state.size
        state.error = ""

    @kb.add("up")
    @kb.add("c-p")
    @kb.add("s-tab")
    def _up(event: KeyPressEvent) -> None:
        move(-1)

    @kb.add("down")
    @kb.add("c-n")
    @kb.add("tab")
    def _down(event: KeyPressEvent) -> None:
        move(1)

    for k in range(min(state.size, 9)):

        @kb.add(str(k + 1), filter=~typing)
        def _digit(event: KeyPressEvent, k: int = k) -> None:
            state.cursor = k
            state.error = ""
            if k == state.custom_index:
                return  # дальше — печать своего ответа
            if multi:
                state.checked ^= {k}
            else:
                event.app.exit(result=Answer((question.options[k].label,)))

    @kb.add("space", filter=~typing)
    def _space(event: KeyPressEvent) -> None:
        if multi:
            state.checked ^= {state.cursor}

    @kb.add(Keys.Any, filter=typing)
    @kb.add(Keys.BracketedPaste, filter=typing)
    def _type(event: KeyPressEvent) -> None:
        text = event.data.replace("\r", " ").replace("\n", " ")
        state.custom += "".join(ch for ch in text if ch.isprintable())
        state.error = ""

    @kb.add("backspace", filter=typing)
    def _erase(event: KeyPressEvent) -> None:
        state.custom = state.custom[:-1]

    @kb.add("c-u", filter=typing)
    def _clear(event: KeyPressEvent) -> None:
        state.custom = ""

    @kb.add("enter")
    def _enter(event: KeyPressEvent) -> None:
        answer = state.answer()
        if answer is not None:
            event.app.exit(result=answer)

    @kb.add("escape", eager=True)
    @kb.add("c-c")
    @kb.add("c-d")
    def _cancel(event: KeyPressEvent) -> None:
        event.app.exit(result=None)

    control = FormattedTextControl(
        lambda: render(state, get_app().output.get_size().columns),
        focusable=True,
        show_cursor=False,
    )
    app: Application[Answer | None] = Application(
        layout=Layout(Window(control, wrap_lines=True, dont_extend_height=True)),
        key_bindings=kb,
        style=prompt_style(),
        color_depth=prompt_color_depth(no_color),
        full_screen=False,
        erase_when_done=True,
        **io,
    )
    app.ttimeoutlen = _ESC_TIMEOUT
    return app.run()
