"""Селектор чатов: стрелки, прокрутка, поиск, отмена; запасной режим и итог продолжения."""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta, timezone

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from gigacrew.agent.chat_store import ChatInfo
from gigacrew.agent.events import ToolCallInfo
from gigacrew.llm.types import FunctionCall, Message
from gigacrew.permissions import ToolKind
from gigacrew.ui.chat_picker import PickerState, pick_chat, render
from gigacrew.ui.console import Console
from gigacrew.ui.format import format_when

DOWN, UP, ESC = "\x1b[B", "\x1b[A", "\x1b"
PGDN, END, HOME = "\x1b[6~", "\x1b[F", "\x1b[H"
BACKSPACE, CTRL_U = "\x7f", "\x15"

NOW = datetime(2026, 10, 9, 15, 0, tzinfo=timezone(timedelta(hours=3)))


def _chat(i: int, title: str = "", **kw) -> ChatInfo:
    when = NOW - timedelta(hours=i)
    return ChatInfo(
        id=f"20261009-{i:06d}-abcd",
        title=title or f"чат номер {i}",
        created_at=when,
        updated_at=when,
        requests=i + 1,
        **kw,
    )


CHATS = [_chat(i) for i in range(12)]


def _pick(chats, *keys: str, current_id: str = ""):
    with create_pipe_input() as pipe:
        for key in keys:
            pipe.send_text(key)
        return pick_chat(chats, current_id, now=NOW, input=pipe, output=DummyOutput())


@pytest.mark.parametrize(
    ("keys", "index"),
    [
        (("\r",), 0),
        ((DOWN, DOWN, "\r"), 2),
        ((UP, "\r"), 11),  # по кругу
        ((PGDN, "\r"), 8),
        ((PGDN, PGDN, PGDN, "\r"), 11),  # без перехода через край
        ((END, "\r"), 11),
        ((END, HOME, "\r"), 0),
    ],
)
def test_navigation(keys, index):
    assert _pick(CHATS, *keys) == CHATS[index]


def test_search_filters_and_backspace_restores():
    chats = [_chat(0, "почини тесты"), _chat(1, "добавь логи"), _chat(2, "тесты для CLI")]
    assert _pick(chats, "ТЕСТЫ", DOWN, "\r") == chats[2]  # без учёта регистра
    assert _pick(chats, "логи", "\r") == chats[1]
    assert _pick(chats, "логиx", BACKSPACE, "\r") == chats[1]
    assert _pick(chats, "zzz", CTRL_U, DOWN, "\r") == chats[1]
    assert _pick(chats, "тесты cli", "\r") == chats[2]  # все слова
    with_preview = [_chat(0, "a"), _chat(1, "b", preview="готов отчёт")]
    assert _pick(with_preview, "отчёт", "\r") == with_preview[1]


def test_enter_with_nothing_found_does_nothing():
    assert _pick(CHATS, "нет такого", "\r", ESC) is None


def test_cancel():
    assert _pick(CHATS, ESC) is None
    assert _pick(CHATS, "\x03") is None  # Ctrl+C


def _text(state: PickerState, width: int = 80) -> str:
    return "".join(text for _, text in render(state, width))


def test_render_window_and_scroll():
    state = PickerState(CHATS, now=NOW)
    text = _text(state)
    assert "Чаты проекта · 12" in text
    assert "❯ чат номер 0" in text and "чат номер 7" in text and "чат номер 8" not in text
    assert "↓ ещё 4" in text and "↑ ещё" not in text
    assert "только что · 1 запрос" in text
    for _ in range(9):
        state.move(1)
    text = _text(state)
    assert "❯ чат номер 9" in text and "↑ ещё 2" in text and "↓ ещё 2" in text


def test_meta_counts_messages_besides_requests(monkeypatch):
    """Ответы на вопросы модели и шаги с инструментами — не запросы: длинный чат из двух
    запросов не должен выглядеть коротким."""
    chat = _chat(1, "долгий чат", messages=108)
    assert "сегодня 14:00 · 2 запроса · 108 сообщений" in _text(PickerState([chat], now=NOW), 100)
    ui, buf = _console(monkeypatch, "")
    ui.pick_chat([chat])
    assert "2 запроса · 108 сообщений" in buf.getvalue()


def test_render_selected_details_current_mark_and_query():
    chats = [_chat(0, "первый", preview="последний ответ", model="Alpha-2-Max"), _chat(1)]
    text = _text(PickerState(chats, current_id=chats[1].id, now=NOW))
    assert "последний ответ" in text and "Alpha-2-Max" in text and chats[0].id in text
    assert "чат номер 1 (текущий)" in text
    state = PickerState(chats, now=NOW)
    state.set_query("нет")
    text = _text(state)
    assert "поиск: нет" in text and "найдено 0" in text and "ничего не найдено" in text


def test_render_fits_width():
    chats = [_chat(0, "очень длинный заголовок " * 10, preview="ответ " * 50)]
    for width in (40, 80):
        lines = _text(PickerState(chats, now=NOW), width).splitlines()
        assert all(len(line) < width for line in lines)
        assert "…" in lines[1]


def test_format_when():
    assert format_when(NOW - timedelta(seconds=10), NOW) == "только что"
    assert format_when(NOW - timedelta(minutes=5), NOW) == "5 мин назад"
    assert format_when(NOW.replace(hour=9, minute=5), NOW) == "сегодня 09:05"
    assert format_when(NOW - timedelta(days=1), NOW) == "вчера 15:00"
    assert format_when(datetime(2026, 3, 1, 8, 0, tzinfo=NOW.tzinfo), NOW) == "01.03 08:00"
    assert format_when(datetime(2025, 3, 1, tzinfo=NOW.tzinfo), NOW) == "01.03.2025"
    # время в другом поясе приводится к местному; наивное считается местным
    assert format_when(NOW.astimezone(UTC), NOW) == "только что"
    assert format_when(datetime.now(), None) == "только что"


# ------------------------------ Console ------------------------------ #
def _console(monkeypatch, *replies):
    buf = io.StringIO()
    ui = Console(no_color=True, file=buf, width=100)
    queue = list(replies)

    def fake_input(prompt):
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr(ui._c, "input", fake_input)
    return ui, buf


def test_plain_picker(monkeypatch):
    ui, buf = _console(monkeypatch, "0", "x", "3")
    assert ui.pick_chat(CHATS[:3], current_id=CHATS[1].id) == CHATS[2]
    out = buf.getvalue()
    assert "Чаты проекта · 3" in out and "чат номер 1 (текущий)" in out
    assert "введите номер от 1 до 3" in out


@pytest.mark.parametrize("reply", ["", EOFError(), KeyboardInterrupt()])
def test_plain_picker_cancel(monkeypatch, reply):
    ui, _ = _console(monkeypatch, reply)
    assert ui.pick_chat(CHATS[:3]) is None


def test_picker_without_chats(monkeypatch):
    ui, _ = _console(monkeypatch)
    assert ui.pick_chat([]) is None


def _call(command: str) -> list[Message]:
    call = FunctionCall(name="run_shell", arguments={"command": command})
    return [
        Message(role="assistant", function_calls=[call]),
        Message(role="function", name="run_shell", content="ok"),
    ]


def test_chat_resumed_shows_last_exchanges(monkeypatch):
    ui, buf = _console(monkeypatch)
    messages = [
        Message(role="user", content="нулевой вопрос"),
        Message(role="assistant", content="нулевой ответ"),
        Message(role="user", content="первый вопрос"),
        Message(role="assistant", content="первый ответ"),
        Message(role="user", content="второй вопрос"),
        Message(role="assistant", content="второй ответ"),
        Message(role="user", content="почини \x1b[31mтесты"),
        *_call("pytest -x"),
        Message(role="assistant", content="**Готово**: тесты проходят"),
    ]
    ui.chat_resumed(
        _chat(3, "почини тесты", messages=10),
        messages,
        lambda call: ToolCallInfo(call.name, str(call.arguments["command"]), ToolKind.COMMAND),
    )
    out = buf.getvalue()
    assert "продолжаем чат «почини тесты» · 4 запроса · 10 сообщений" in out
    # Ранние обмены — строкой: сохранены в чате, но не печатаются.
    assert "раньше: 1 запрос · 2 сообщения — сохранены в чате, на экран не выводятся" in out
    assert "нулевой" not in out
    assert out.index("❯ первый вопрос") < out.index("первый ответ") < out.index("❯ второй вопрос")
    assert "\x1b" not in out and "❯ почини [31mтесты" in out  # sanitize убрал ESC
    assert "● $ pytest -x" in out  # шаг — как в ленте, без итога: его нет в истории
    assert "Готово: тесты проходят" in out


def test_chat_resumed_folds_tool_calls_and_clips_earlier_answers(monkeypatch):
    ui, buf = _console(monkeypatch)
    long_answer = "\n\n".join(f"абзац {i}" for i in range(20))
    messages = [
        Message(role="user", content="разбери логи"),
        *[m for i in range(8) for m in _call(f"cat log{i}")],
        Message(role="assistant", content=long_answer),
        Message(role="user", content="а теперь итог"),
        Message(role="assistant", content=long_answer),
    ]
    ui.chat_resumed(_chat(1, "разбери логи"), messages)
    out = buf.getvalue()
    assert "раньше" not in out  # показаны все обмены
    assert "… ещё 3 вызова инструментов" in out and out.count("● run_shell") == 5
    first, last = out.split("❯ а теперь итог")
    assert "пропущено" in first and "абзац 19" in first  # ранний ответ сокращён
    assert "пропущено" not in last and "абзац 10" in last  # последний — целиком


def test_chat_resumed_skips_meta_messages(monkeypatch):
    ui, buf = _console(monkeypatch)
    messages = [
        Message(role="user", content="спланируй промокоды"),
        Message(role="assistant", content="Вот план"),
        Message(role="user", content="<system-reminder>план одобрен</system-reminder>", meta=True),
        Message(role="assistant", content="Промокоды готовы"),
    ]
    ui.chat_resumed(_chat(0, "спланируй промокоды"), messages)
    out = buf.getvalue()
    assert "❯ спланируй промокоды" in out and "Промокоды готовы" in out
    assert "system-reminder" not in out


def test_chat_resumed_after_interrupted_turn_shows_no_stale_answer(monkeypatch):
    ui, buf = _console(monkeypatch)
    messages = [
        Message(role="user", content="первый вопрос"),
        Message(role="assistant", content="ответ на первый"),
        Message(role="user", content="второй вопрос"),
        Message(role="assistant", function_calls=[FunctionCall(name="run_shell")]),
        Message(role="function", name="run_shell", content="прервано"),
    ]
    ui.chat_resumed(_chat(1, "первый вопрос"), messages)
    out = buf.getvalue()
    # Ответ на первый вопрос — в своём обмене, а не ответом на прерванный второй.
    before, after = out.split("❯ второй вопрос")
    assert "ответ на первый" in before
    assert "ответ на первый" not in after and "ответа нет" in after


def test_chat_resumed_folds_consecutive_reads_and_edits(monkeypatch):
    ui, buf = _console(monkeypatch)

    def call(name: str, path: str) -> FunctionCall:
        return FunctionCall(name=name, arguments={"path": path})

    def info(fc: FunctionCall) -> ToolCallInfo:
        if fc.name == "read_file":
            return ToolCallInfo(fc.name, fc.arguments["path"], read_only=True)
        return ToolCallInfo(fc.name, fc.arguments["path"], ToolKind.EDIT)

    reads = [call("read_file", "a.py"), call("read_file", "b.py")]
    edits = [call("edit_file", "a.py"), call("edit_file", "b.py"), call("edit_file", "a.py")]
    messages = [
        Message(role="user", content="поправь"),
        Message(role="assistant", function_calls=reads),
        Message(role="assistant", function_calls=edits),
        Message(role="assistant", content="Готово"),
    ]
    ui.chat_resumed(_chat(0, "поправь"), messages, info)
    out = buf.getvalue()
    assert "● Изучил код  прочитал 2 файла\n  a.py  b.py\n● Изменил a.py, b.py\n" in out
