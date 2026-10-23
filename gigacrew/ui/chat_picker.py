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
from prompt_toolkit.key_binding.key_bindings import KeyHandlerCallable
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import FormattedTextControl, Layout, Window
from prompt_toolkit.utils import get_cwidth

from gigacrew.agent.chat_store import ChatInfo
from gigacrew.ui.colors import prompt_color_depth, prompt_style
from gigacrew.ui.format import chat_size, format_when

WINDOW = 8  # сколько чатов видно одновременно
_ESC_TIMEOUT = 0.05
HINT = "↑↓ выбор · Enter — открыть · печатайте для поиска · Esc — отмена"
_MIN_WIDTH = 30
_MIN_TITLE_WIDTH = 10
_HINT_STYLE = "class:choice.hint"
_DESCRIPTION_STYLE = "class:choice.description"
_NEWLINE = ("", "\n")
_DETAILS_INDENT = "     "


@dataclass
class PickerState:
    """Состояние селектора: список чатов, поисковый запрос, курсор и окно прокрутки."""

    chats: Sequence[ChatInfo]
    current_id: str = ""
    query: str = ""
    cursor: int = 0  # индекс в visible
    offset: int = 0  # первый видимый чат окна
    window: int = WINDOW
    now: datetime | None = None  # для тестов
    _visible: list[ChatInfo] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        """Отбирает чаты по начальному запросу."""
        self._refilter()

    @property
    def visible(self) -> list[ChatInfo]:
        """Чаты, подходящие под запрос."""
        return self._visible

    @property
    def selected(self) -> ChatInfo | None:
        """Чат под курсором; None — ничего не найдено."""
        return self._visible[self.cursor] if self._visible else None

    def set_query(self, query: str) -> None:
        """Меняет поисковый запрос: список отбирается заново, курсор — в начало."""
        self.query = query
        self._refilter()

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

    def _refilter(self) -> None:
        words = self.query.lower().split()
        self._visible = [
            chat
            for chat in self.chats
            if all(word in f"{chat.title}\n{chat.preview}\n{chat.id}".lower() for word in words)
        ]
        self.cursor = 0
        self.offset = 0

    def _scroll(self) -> None:
        if self.cursor < self.offset:
            self.offset = self.cursor
        elif self.cursor >= self.offset + self.window:
            self.offset = self.cursor - self.window + 1


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


def _meta(chat: ChatInfo, now: datetime | None) -> str:
    return f"{format_when(chat.updated_at, now)} · {chat_size(chat.requests, chat.messages)}"


def _search_line(state: PickerState) -> list[tuple[str, str]]:
    """Строка поиска: запрос (с курсором ввода) и сколько чатов найдено."""
    return [
        (_HINT_STYLE, " поиск: "),
        ("class:choice.custom", state.query),
        ("[SetCursorPosition]", ""),
        (_HINT_STYLE, f"  · найдено {len(state.visible)}"),
        _NEWLINE,
    ]


def _title_and_gap(
    title: str,
    width: int,
    *,
    prefix: str,
    meta: str,
    mark: str,
) -> tuple[str, int]:
    """Заголовок, обрезанный под ширину, и отступ до даты и размера чата справа."""
    room = width - get_cwidth(prefix) - get_cwidth(meta) - get_cwidth(mark) - 2
    fitted = _fit(title, max(room, _MIN_TITLE_WIDTH))
    gap = width - get_cwidth(prefix + fitted + mark) - get_cwidth(meta)
    return fitted, gap


def _details(chat: ChatInfo, width: int) -> list[tuple[str, str]]:
    """Подробности выбранного чата: последний ответ, id и модель."""
    details = chat.preview or "(ответа ещё нет)"
    model = f" · {chat.model}" if chat.model else ""
    preview = _fit(f"{_DETAILS_INDENT}{details}", width)
    extra = _fit(f"{_DETAILS_INDENT}{chat.id}{model}", width)
    return [(_DESCRIPTION_STYLE, f"{preview}\n"), (_HINT_STYLE, f"{extra}\n")]


def _chat_lines(state: PickerState, position: int, width: int) -> list[tuple[str, str]]:
    """Строка чата номер ``position`` в списке; у выбранного — ещё и подробности."""
    chat = state.visible[position]
    current = position == state.cursor
    style = "class:choice.selected" if current else "class:choice"
    pointer = "❯" if current else " "
    prefix = f" {pointer} "
    meta = _meta(chat, state.now)
    mark = " (текущий)" if chat.id == state.current_id else ""
    title, gap = _title_and_gap(chat.title, width, prefix=prefix, meta=meta, mark=mark)
    lines = [
        (style, prefix + title),
        (_HINT_STYLE, mark),
        ("", " " * max(gap, 2)),
        (_DESCRIPTION_STYLE, meta),
        _NEWLINE,
    ]
    if current:
        lines += _details(chat, width)
    return lines


def _chat_list(state: PickerState, width: int) -> list[tuple[str, str]]:
    """Окно списка чатов со стрелками «ещё N» сверху и снизу."""
    out: list[tuple[str, str]] = []
    end = min(state.offset + state.window, len(state.visible))
    if state.offset:
        out.append((_HINT_STYLE, f"   ↑ ещё {state.offset}\n"))
    for position in range(state.offset, end):
        out.extend(_chat_lines(state, position, width))
    if end < len(state.visible):
        rest = len(state.visible) - end
        out.append((_HINT_STYLE, f"   ↓ ещё {rest}\n"))
    return out


def render(state: PickerState, width: int = 80) -> list[tuple[str, str]]:
    """Отрисовка селектора: список пар (стиль prompt_toolkit, текст)."""
    width = max(width - 1, _MIN_WIDTH)  # последний столбец — без автопереноса строки
    total = len(state.chats)
    out: list[tuple[str, str]] = [
        ("class:question.counter", f" Чаты проекта · {total}"),
        _NEWLINE,
    ]
    if state.query:
        out += _search_line(state)

    if state.visible:
        out += _chat_list(state, width)
    else:
        out.append(("class:choice.error", " ничего не найдено\n"))
    out.append((_HINT_STYLE, _fit(f" {HINT}", width)))
    return out


def _bind(kb: KeyBindings, handler: KeyHandlerCallable, *keys: Keys | str) -> None:
    """Привязывает ``handler`` к каждой из клавиш ``keys`` по порядку."""
    for key in keys:
        kb.add(key)(handler)


class _PickerKeys:
    """Обработчики клавиш селектора: меняют состояние, Enter и Esc закрывают меню."""

    def __init__(self, state: PickerState) -> None:
        self._state = state

    def bindings(self) -> KeyBindings:
        """Привязки клавиш селектора."""
        kb = KeyBindings()
        self._bind_moves(kb)
        _bind(kb, self._type, Keys.BracketedPaste, Keys.Any)
        kb.add("backspace")(self._erase)
        kb.add("c-u")(self._clear)
        kb.add("enter")(self._enter)
        _bind(kb, self._cancel, "c-d", "c-c")
        kb.add("escape", eager=True)(self._cancel)
        return kb

    def _bind_moves(self, kb: KeyBindings) -> None:
        """Клавиши перемещения курсора."""
        _bind(kb, self._up, "s-tab", "c-p", "up")
        _bind(kb, self._down, "tab", "c-n", "down")
        kb.add("pageup")(self._page_up)
        kb.add("pagedown")(self._page_down)
        kb.add("home")(self._home)
        kb.add("end")(self._end)

    def _up(self, event: KeyPressEvent) -> None:
        self._state.move(-1)

    def _down(self, event: KeyPressEvent) -> None:
        self._state.move(1)

    def _page_up(self, event: KeyPressEvent) -> None:
        self._state.jump(-self._state.window)

    def _page_down(self, event: KeyPressEvent) -> None:
        self._state.jump(self._state.window)

    def _home(self, event: KeyPressEvent) -> None:
        self._state.jump(-len(self._state.visible))

    def _end(self, event: KeyPressEvent) -> None:
        self._state.jump(len(self._state.visible))

    def _type(self, event: KeyPressEvent) -> None:
        text = event.data.replace("\r", " ").replace("\n", " ")
        typed = "".join(ch for ch in text if ch.isprintable())
        if typed:
            self._state.set_query(self._state.query + typed)

    def _erase(self, event: KeyPressEvent) -> None:
        if self._state.query:
            self._state.set_query(self._state.query[:-1])

    def _clear(self, event: KeyPressEvent) -> None:
        self._state.set_query("")

    def _enter(self, event: KeyPressEvent) -> None:
        if self._state.selected is not None:
            event.app.exit(result=self._state.selected)

    def _cancel(self, event: KeyPressEvent) -> None:
        event.app.exit(result=None)


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
    kb = _PickerKeys(state).bindings()
    control = FormattedTextControl(
        lambda: list(render(state, get_app().output.get_size().columns)),
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
