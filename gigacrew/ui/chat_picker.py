"""Селектор сохранённых чатов (``/resume``, ``gigacrew --resume``).

Встроенное (не полноэкранное) приложение prompt_toolkit, как меню вопросов
(:mod:`gigacrew.ui.choice`): ↑↓ — выбор, PgUp/PgDn, Home/End — быстрые переходы,
печать — поиск по заголовку и последнему ответу (Backspace — стереть, Ctrl+U —
очистить), Enter — открыть чат, Esc/Ctrl+C — отмена. Видно окно из нескольких
чатов, список прокручивается за курсором. После выбора меню стирается.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from prompt_toolkit.application import Application, get_app
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import FormattedTextControl, Layout, Window
from prompt_toolkit.utils import get_cwidth

from gigacrew.agent.chat_store import ChatInfo
from gigacrew.ui.colors import prompt_color_depth, prompt_style
from gigacrew.ui.format import chat_size, format_when

WINDOW = 8  # сколько чатов видно одновременно
_ESC_TIMEOUT = 0.05
HINT = "↑↓ выбор · Enter — открыть · печатайте для поиска · Esc — отмена"


@dataclass
class PickerState:
    chats: Sequence[ChatInfo]
    current_id: str = ""
    query: str = ""
    cursor: int = 0  # индекс в visible
    offset: int = 0  # первый видимый чат окна
    window: int = WINDOW
    now: datetime | None = None  # для тестов
    _visible: list[ChatInfo] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._refilter()

    @property
    def visible(self) -> list[ChatInfo]:
        return self._visible

    @property
    def selected(self) -> ChatInfo | None:
        return self._visible[self.cursor] if self._visible else None

    def _refilter(self) -> None:
        words = self.query.lower().split()
        self._visible = [
            c
            for c in self.chats
            if all(w in f"{c.title}\n{c.preview}\n{c.id}".lower() for w in words)
        ]
        self.cursor = 0
        self.offset = 0

    def set_query(self, query: str) -> None:
        self.query = query
        self._refilter()

    def _scroll(self) -> None:
        if self.cursor < self.offset:
            self.offset = self.cursor
        elif self.cursor >= self.offset + self.window:
            self.offset = self.cursor - self.window + 1

    def move(self, step: int) -> None:
        """Стрелки — по кругу."""
        if self._visible:
            self.cursor = (self.cursor + step) % len(self._visible)
            self._scroll()

    def jump(self, step: int) -> None:
        """PgUp/PgDn/Home/End — без перехода через край."""
        if self._visible:
            self.cursor = min(max(self.cursor + step, 0), len(self._visible) - 1)
            self._scroll()


def _fit(text: str, width: int) -> str:
    """Обрезает по ширине терминала (с учётом широких символов)."""
    if get_cwidth(text) <= width:
        return text
    out, used = "", 0
    for ch in text:
        w = get_cwidth(ch)
        if used + w > width - 1:
            break
        out += ch
        used += w
    return out + "…"


def _meta(chat: ChatInfo, now: datetime | None) -> str:
    return f"{format_when(chat.updated_at, now)} · {chat_size(chat.requests, chat.messages)}"


def render(state: PickerState, width: int = 80) -> list[tuple[str, str]]:
    """Отрисовка селектора: список пар (стиль prompt_toolkit, текст)."""
    width = max(width - 1, 30)  # последний столбец — без автопереноса строки
    total = len(state.chats)
    out: list[tuple[str, str]] = [
        ("class:question.counter", f" Чаты проекта · {total}"),
        ("", "\n"),
    ]
    if state.query:
        out += [
            ("class:choice.hint", " поиск: "),
            ("class:choice.custom", state.query),
            ("[SetCursorPosition]", ""),
            ("class:choice.hint", f"  · найдено {len(state.visible)}"),
            ("", "\n"),
        ]

    if not state.visible:
        out.append(("class:choice.error", " ничего не найдено\n"))
    else:
        end = min(state.offset + state.window, len(state.visible))
        if state.offset:
            out.append(("class:choice.hint", f"   ↑ ещё {state.offset}\n"))
        for k in range(state.offset, end):
            chat = state.visible[k]
            current = k == state.cursor
            style = "class:choice.selected" if current else "class:choice"
            prefix = f" {'❯' if current else ' '} "
            meta = _meta(chat, state.now)
            mark = " (текущий)" if chat.id == state.current_id else ""
            room = width - get_cwidth(prefix) - get_cwidth(meta) - get_cwidth(mark) - 2
            title = _fit(chat.title, max(room, 10))
            gap = width - get_cwidth(prefix + title + mark) - get_cwidth(meta)
            out += [
                (style, prefix + title),
                ("class:choice.hint", mark),
                ("", " " * max(gap, 2)),
                ("class:choice.description", meta),
                ("", "\n"),
            ]
            if current:
                details = chat.preview or "(ответа ещё нет)"
                out.append(("class:choice.description", _fit(f"     {details}", width) + "\n"))
                extra = f"     {chat.id}" + (f" · {chat.model}" if chat.model else "")
                out.append(("class:choice.hint", _fit(extra, width) + "\n"))
        if end < len(state.visible):
            out.append(("class:choice.hint", f"   ↓ ещё {len(state.visible) - end}\n"))
    out.append(("class:choice.hint", _fit(" " + HINT, width)))
    return out


def pick_chat(
    chats: Sequence[ChatInfo],
    current_id: str = "",
    *,
    no_color: bool = False,
    now: datetime | None = None,
    **io: Any,
) -> ChatInfo | None:
    """Выбор чата из списка; None — отмена.

    ``io`` — ``input``/``output`` prompt_toolkit (подменяются в тестах).
    """
    state = PickerState(chats, current_id, now=now)
    kb = KeyBindings()

    @kb.add("up")
    @kb.add("c-p")
    @kb.add("s-tab")
    def _up(event: KeyPressEvent) -> None:
        state.move(-1)

    @kb.add("down")
    @kb.add("c-n")
    @kb.add("tab")
    def _down(event: KeyPressEvent) -> None:
        state.move(1)

    @kb.add("pageup")
    def _page_up(event: KeyPressEvent) -> None:
        state.jump(-state.window)

    @kb.add("pagedown")
    def _page_down(event: KeyPressEvent) -> None:
        state.jump(state.window)

    @kb.add("home")
    def _home(event: KeyPressEvent) -> None:
        state.jump(-len(state.visible))

    @kb.add("end")
    def _end(event: KeyPressEvent) -> None:
        state.jump(len(state.visible))

    @kb.add(Keys.Any)
    @kb.add(Keys.BracketedPaste)
    def _type(event: KeyPressEvent) -> None:
        text = event.data.replace("\r", " ").replace("\n", " ")
        typed = "".join(ch for ch in text if ch.isprintable())
        if typed:
            state.set_query(state.query + typed)

    @kb.add("backspace")
    def _erase(event: KeyPressEvent) -> None:
        if state.query:
            state.set_query(state.query[:-1])

    @kb.add("c-u")
    def _clear(event: KeyPressEvent) -> None:
        state.set_query("")

    @kb.add("enter")
    def _enter(event: KeyPressEvent) -> None:
        if state.selected is not None:
            event.app.exit(result=state.selected)

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
    app: Application[ChatInfo | None] = Application(
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
