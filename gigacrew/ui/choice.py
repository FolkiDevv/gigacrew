"""Меню выбора ответа на вопрос агента (инструмент ``ask_user``).

Встроенное (не полноэкранное) приложение prompt_toolkit: ↑↓ — выбор, цифры — сразу
выбрать (в мультивыборе — отметить), Пробел — отметить, Enter — подтвердить,
Esc/Ctrl+C — отказаться отвечать. Последний пункт — «Свой ответ…»: когда он выбран,
набираемый текст попадает прямо в него (Backspace — стереть, Ctrl+U — очистить).
После ответа меню стирается — итог печатает вызывающий код.
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
from prompt_toolkit.output import ColorDepth
from prompt_toolkit.styles import Style

from gigacrew.tools.questions import Answer, Question
from gigacrew.ui.theme import PROMPT_STYLES

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


def _wrap(text: str, width: int, indent: str = " ") -> list[str]:
    return textwrap.wrap(text, max(width - 1, 20), initial_indent=indent, subsequent_indent=indent)


def render(state: ChoiceState, width: int = 80) -> list[tuple[str, str]]:
    """Отрисовка меню: список пар (стиль prompt_toolkit, текст)."""
    q = state.question
    out: list[tuple[str, str]] = [
        ("class:question.counter", f" Вопрос {state.index} из {state.total}")
    ]
    if q.header:
        out += [("class:choice.hint", " · "), ("class:question.header", q.header)]
    out.append(("", "\n"))
    for line in _wrap(q.text, width) or [" "]:
        out.append(("class:question.text", line + "\n"))
    if q.multi_select:
        out.append(("class:choice.hint", " можно выбрать несколько вариантов\n"))

    items = [(o.label, o.description) for o in q.options] + [(CUSTOM_LABEL, CUSTOM_DESCRIPTION)]
    for k, (label, description) in enumerate(items):
        current = k == state.cursor
        style = "class:choice.selected" if current else "class:choice"
        box = ""
        if q.multi_select and k != state.custom_index:
            box = "[x] " if k in state.checked else "[ ] "
        prefix = f" {'❯' if current else ' '} {k + 1}. "
        if k == state.custom_index and (state.custom or current):
            out += [(style, prefix + "Свой ответ: "), ("class:choice.custom", state.custom)]
            if current:
                out.append(("[SetCursorPosition]", ""))
            out.append(("", "\n"))
            if state.custom:
                continue
        else:
            out += [(style, prefix + box + label), ("", "\n")]
        if description:
            indent = " " * (len(prefix) + len(box))
            for line in _wrap(description, width, indent):
                out.append(("class:choice.description", line + "\n"))

    if state.error:
        out.append(("class:choice.error", f" {state.error}\n"))
    if q.multi_select:
        hint = "↑↓ выбор · Пробел или цифра — отметить · Enter — готово · Esc — отказаться"
    else:
        hint = "↑↓ выбор · цифра — сразу · Enter — подтвердить · Esc — отказаться"
    out.append(("class:choice.hint", "\n".join(_wrap(hint, width))))
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
        style=Style.from_dict(PROMPT_STYLES),
        color_depth=ColorDepth.MONOCHROME if no_color else None,
        full_screen=False,
        erase_when_done=True,
        **io,
    )
    app.ttimeoutlen = _ESC_TIMEOUT
    return app.run()
