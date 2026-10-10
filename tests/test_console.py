"""Тесты терминального UI: очистка управляющих символов, рендер событий."""

from __future__ import annotations

import io
import re

import pytest

from gigacrew.agent.events import ToolCallInfo, TurnStats
from gigacrew.tools.base import Display, ToolResult
from gigacrew.ui.console import Console, sanitize
from gigacrew.ui.format import clip_lines, format_tokens, plural


def _console():
    buf = io.StringIO()
    return Console(no_color=True, file=buf), buf


def test_sanitize_strips_escape_sequences():
    assert sanitize("a\x1b[31mred\x1b[0m\tb\nc\x07\x9b") == "a[31mred[0m\tb\nc"


def test_stream_and_output_are_sanitized():
    ui, buf = _console()
    ui.on_stream_start()
    ui.on_stream_delta("hi \x1b]0;pwned\x07there")
    ui.on_stream_end()
    ui.on_tool_result(
        ToolCallInfo("run_shell", "ls"),
        ToolResult(content="", summary="ok", display=Display("\x1b[2Jcleared", title="$ ls")),
        previewed=False,
    )
    out = buf.getvalue()
    assert "\x1b" not in out and "\x07" not in out
    assert "there" in out and "cleared" in out


def test_previewed_diff_not_repeated():
    ui, buf = _console()
    result = ToolResult(
        content="", summary="изменён a.py", display=Display("+UNIQUE", "diff", "a.py")
    )
    ui.on_tool_result(ToolCallInfo("edit_file", "a.py"), result, previewed=True)
    assert "UNIQUE" not in buf.getvalue()
    ui.on_tool_result(ToolCallInfo("edit_file", "a.py"), result, previewed=False)
    assert "UNIQUE" in buf.getvalue()


def test_turn_stats_line():
    ui, buf = _console()
    ui.on_turn_end(
        TurnStats(
            steps=2,
            tool_calls=1,
            prompt_tokens=200,
            completion_tokens=20,
            context_tokens=12_345,
            duration_s=3.21,
        )
    )
    out = buf.getvalue()
    assert "2 шага" in out and "1 инструмент " in out and "3.2 с" in out
    assert "контекст ~12.3k" in out and "потрачено 220" in out


# ------------------------------ живой режим ------------------------------ #
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _live_console(width: int = 60):
    buf = io.StringIO()
    return Console(file=buf, force_terminal=True, width=width, auto_refresh=False), buf


def _plain(buf: io.StringIO) -> str:
    return _ANSI_RE.sub("", buf.getvalue())


def test_streamed_markdown_is_formatted():
    ui, buf = _live_console()
    ui.on_stream_start()
    for chunk in ["# Заго", "ловок\n\nЭто **жир", "ный** текст\n", "\n- пункт\n", "\nконец"]:
        ui.on_stream_delta(chunk)
    ui.on_stream_end()
    out = _plain(buf)
    assert "Заголовок" in out and "жирный текст" in out and "конец" in out
    assert "**" not in out and "# " not in out
    assert "\x1b[1m" in buf.getvalue()  # жирный — настоящим стилем терминала
    assert ui._live is None


def test_live_area_stopped_after_interrupt(tmp_path):
    from fakes import ScriptedProvider, text_turn

    from gigacrew.agent.loop import Agent
    from gigacrew.config import Config
    from gigacrew.tools.base import build_default_registry

    class Interrupting(ScriptedProvider):
        def stream(self, messages, tools=None, *, on_delta=None, **kwargs):
            on_delta("Начало ответа\n\nещё")
            raise KeyboardInterrupt

    ui, buf = _live_console()
    cfg = Config(access_key="x", project_root=tmp_path, stream=True)
    agent = Agent(Interrupting([text_turn("x")]), build_default_registry(), cfg, ui)
    with pytest.raises(KeyboardInterrupt):
        agent.run_turn("привет")
    assert ui._live is None
    assert "Начало ответа" in _plain(buf) and "ещё" in _plain(buf)


def test_tool_spinner_lifecycle():
    ui, _ = _live_console()
    call = ToolCallInfo("run_shell", "sleep 1")
    ui.on_tool_start(call)
    assert ui._live is not None
    ui.on_tool_end(call)
    assert ui._live is None
    ui.stop_live()  # идемпотентно


def test_single_live_area():
    ui, _ = _live_console()
    ui.on_stream_start()
    first = ui._live
    ui.on_tool_start(ToolCallInfo("x"))
    assert ui._live is not first and not first.active
    ui.stop_live()


def test_progress_live_label_updates_and_stops_on_interrupt():
    ui, buf = _live_console()
    counter = {"n": 0}
    with (
        pytest.raises(KeyboardInterrupt),
        ui.progress("индексирую", lambda: f"{counter['n']} файлов"),
    ):
        assert ui._live is not None
        counter["n"] = 42
        ui._live._live.refresh()
        raise KeyboardInterrupt
    assert ui._live is None
    assert "индексирую: 42 файлов" in _plain(buf)


def test_progress_without_terminal_prints_title_once():
    ui, buf = _console()
    with ui.progress("индексирую", lambda: "1 файл"):
        assert ui._live is None
    assert buf.getvalue() == "индексирую…\n"


def test_compaction_progress_and_result():
    from gigacrew.agent.events import CompactResult

    ui, buf = _console()
    ui.on_compact_start(auto=True)
    ui.on_compact_end(CompactResult(before_tokens=24_000, after_tokens=6_500, messages=31))
    out = buf.getvalue()
    assert "контекст почти заполнен — сжимаю историю…" in out
    assert "контекст сжат: ~24k → ~6.5k ток." in out and "31 сообщение" in out

    ui, buf = _live_console()
    ui.on_compact_start(auto=False)
    assert ui._live is not None
    ui.on_compact_end(None)  # не удалось — только убрать индикатор
    assert ui._live is None and "сжат" not in _plain(buf)


def test_no_live_area_without_terminal():
    ui, buf = _console()
    ui.on_stream_start()
    ui.on_tool_start(ToolCallInfo("x"))
    assert ui._live is None
    ui.on_stream_end()
    assert buf.getvalue() == ""


def test_long_tool_output_is_clipped():
    ui, buf = _console()
    text = "\n".join(f"line{i}" for i in range(100))
    ui.on_tool_result(
        ToolCallInfo("run_shell", "seq"),
        ToolResult(content=text, summary="ok", display=Display(text, title="$ seq")),
        previewed=False,
    )
    out = buf.getvalue()
    assert "line0" in out and "line99" in out and "line50" not in out
    assert "пропущено 84 строки" in out


def test_confirm_question_includes_summary(monkeypatch):
    from gigacrew.agent.events import Approval
    from gigacrew.permissions import ToolKind

    ui, buf = _console()
    answers = iter(["д", "", "a", "a"])
    monkeypatch.setattr(ui._c, "input", lambda prompt: (ui._c.print(prompt), next(answers))[1])
    edit = ToolCallInfo("edit_file", "a.py", ToolKind.EDIT)
    assert ui.confirm(edit, None, dangerous=False) is Approval.YES
    out = buf.getvalue()
    assert "Применить edit_file (a.py)? [y/N/a]" in out and "режим «авто-правки»" in out
    danger = ToolCallInfo("run_shell", "rm -rf x", ToolKind.COMMAND)
    assert ui.confirm(danger, None, dangerous=True) is Approval.NO
    assert "ОПАСНО: Выполнить run_shell (rm -rf x)? [y/N] " in buf.getvalue()
    command = ToolCallInfo("run_shell", "pytest -q", ToolKind.COMMAND)
    assert ui.confirm(command, None, dangerous=False) is Approval.ALWAYS
    assert "Выполнить run_shell (pytest -q)?" in buf.getvalue()
    assert ui.confirm(danger, None, dangerous=True) is Approval.NO  # «a» для опасного — нет


def test_format_helpers():
    assert [format_tokens(n) for n in (950, 12_345, 120_000, 1_234_567)] == [
        "950",
        "12.3k",
        "120k",
        "1.2M",
    ]
    assert [plural(n, "шаг", "шага", "шагов") for n in (1, 2, 5, 11, 21, 112)] == [
        "шаг",
        "шага",
        "шагов",
        "шагов",
        "шаг",
        "шагов",
    ]
    assert clip_lines("a\nb") == "a\nb"
    assert len(clip_lines("x" * 10_000)) <= 300
    long_lines = "\n".join(f"{i:03d} " + "x" * 290 for i in range(40)) + "\nИТОГ: 3 failed"
    clipped = clip_lines(long_lines)
    assert len(clipped) <= 4000
    assert clipped.startswith("000") and clipped.endswith("ИТОГ: 3 failed")


def test_animation_thread_stops():
    buf = io.StringIO()
    ui = Console(file=buf, force_terminal=True, width=60)  # с потоком анимации
    ui.on_stream_start()
    area = ui._live
    ui.on_stream_delta("текст\n\nещё\n")
    ui.on_stream_end()
    assert area._thread is not None and not area._thread.is_alive()
    assert "текст" in _plain(buf)


# ---------------------------- вопросы агента ---------------------------- #
from gigacrew.tools.questions import Answer, Question, QuestionOption  # noqa: E402

_DB = Question(
    "Какую БД?",
    (QuestionOption("PostgreSQL", "с сервером"), QuestionOption("SQLite", "файл")),
    header="БД",
)
_PARTS = Question(
    "Что добавить?", (QuestionOption("Логи"), QuestionOption("Метрики")), multi_select=True
)


def _asking_console(monkeypatch, *replies):
    import gigacrew.ui.console as console_mod

    monkeypatch.setattr(console_mod, "_stdin_is_terminal", lambda: True)
    ui, buf = _console()
    queue = list(replies)

    def fake_input(prompt):
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr(ui._c, "input", fake_input)
    return ui, buf


@pytest.mark.parametrize(
    ("replies", "expected"),
    [
        (["2"], [Answer(("SQLite",)), Answer(("Логи",))]),
        (["MongoDB"], [Answer(custom="MongoDB")]),
        (["3", "Redis"], [Answer(custom="Redis")]),
        (["1 2", "1"], [Answer(("PostgreSQL",))]),  # один вариант — повтор вопроса
        ([" , ", "2"], [Answer(("SQLite",))]),  # одни запятые — повтор
        (["²"], [Answer(custom="²")]),  # не номер — свой ответ, без падения
    ],
)
def test_plain_questions(monkeypatch, replies, expected):
    second = ["1, 2"] if len(expected) == 1 else ["1"]
    ui, buf = _asking_console(monkeypatch, *replies, *second)
    answers = ui.ask_user([_DB, _PARTS])
    assert answers[0] == expected[0]
    out = buf.getvalue()
    assert "Вопрос 1 из 2 · БД" in out and "с сервером" in out and "3. Свой ответ" in out
    assert "? Какую БД?" in out  # итог в ленте


def test_plain_multi_select(monkeypatch):
    ui, _ = _asking_console(monkeypatch, "1", "1, 2")
    assert ui.ask_user([_DB, _PARTS])[1] == Answer(("Логи", "Метрики"))


def test_questions_declined(monkeypatch):
    ui, buf = _asking_console(monkeypatch, KeyboardInterrupt())
    assert ui.ask_user([_DB, _PARTS]) is None
    assert "отказался" in buf.getvalue()


def test_questions_need_keyboard(monkeypatch):
    from gigacrew.tools.questions import QuestionsUnavailable

    ui, _ = _console()
    with pytest.raises(QuestionsUnavailable):
        ui.ask_user([_DB])


def test_questions_pause_esc_interrupt(monkeypatch):
    import contextlib

    ui, _ = _asking_console(monkeypatch, "1")
    entered = []

    @contextlib.contextmanager
    def guard():
        entered.append(True)
        yield

    ui.set_interrupt_keys("Esc — прервать", guard)
    ui.ask_user([_DB])
    assert entered == [True]


def test_banner_shows_rocket_title_and_meta():
    ui, buf = _console()
    ui.banner(version="9.9.9", model="Alpha-Max", root="/work/proj", hints=[("/help", "справка")])
    out = _plain(buf)
    assert "▟█▙" in out and "┃╺┓" in out  # ракета и крупное название
    assert "v9.9.9" in out and "Alpha-Max" in out and "/work/proj" in out
    assert "/help справка" in out


def test_banner_shows_mode():
    from gigacrew.permissions import PermissionMode

    ui, buf = _console()
    ui.banner(version="1", model="m", root="/r", mode=PermissionMode.PLAN)
    assert "режим  ⏸ план  Shift+Tab — сменить" in _plain(buf)


def test_question_body_is_shown_before_options(monkeypatch):
    from dataclasses import replace

    ui, buf = _asking_console(monkeypatch, "1")
    question = replace(_DB, header="План", body="## Шаги\n1. **Поправить** `a.py`")
    assert ui.ask_user([question]) == [Answer(("PostgreSQL",))]
    out = buf.getvalue()
    assert "План" in out and "Шаги" in out and "Поправить" in out and "**" not in out
    assert out.index("Поправить") < out.index("Вопрос 1 из 1")


def test_turn_indicator_shows_current_mode():
    import contextlib

    from gigacrew.permissions import PermissionMode

    ui, buf = _live_console(width=120)
    mode = [PermissionMode.MANUAL]
    ui.set_interrupt_keys("Esc — прервать", contextlib.nullcontext)
    ui.set_mode_hint(lambda: f"{mode[0].label} (Shift+Tab)")
    ui.on_stream_start()
    mode[0] = PermissionMode.PLAN  # Shift+Tab во время хода
    ui._live._live.refresh()
    ui.stop_live()
    assert "Esc — прервать  ·  план (Shift+Tab)" in _plain(buf)


# ------------------------------ суб-агенты ------------------------------ #
def test_subagent_steps_nested_and_text_hidden(tmp_path):
    from fakes import ScriptedProvider, text_turn, tool_turn

    from gigacrew.agent.loop import Agent
    from gigacrew.config import Config
    from gigacrew.tools.base import build_default_registry

    (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
    provider = ScriptedProvider(
        [
            tool_turn("task", {"agent": "explore", "prompt": "найди foo", "description": "foo"}),
            tool_turn("search_content", {"pattern": "def foo"}),
            tool_turn("read_file", {"path": "nope.py"}),
            text_turn("СЕКРЕТНЫЙ ОТЧЁТ суб-агента"),
            text_turn("Итог основного"),
        ]
    )
    ui, buf = _live_console(width=100)
    cfg = Config(access_key="x", project_root=tmp_path, stream=True, auto_approve=True)
    Agent(provider, build_default_registry(), cfg, ui).run_turn("где foo?")
    out = _plain(buf)
    assert "task  explore · foo" in out
    assert "○ search_content  /def foo/" in out and "○ read_file  nope.py" in out
    assert "✘ Файл не найден: nope.py" in out  # неудача шага суб-агента видна
    assert "СЕКРЕТНЫЙ ОТЧЁТ" not in out  # отчёт получает основной агент, не пользователь
    assert "Итог основного" in out
    assert out.count("gigacrew") == 1  # метка ответа — только у основного агента
    assert "explore · инструментов: 2" in out
    assert ui._live is None


def test_subagent_live_area_survives_confirm():
    from gigacrew.agent.events import SubagentInfo

    ui, _ = _live_console()
    info = SubagentInfo("general", "правка")
    ui.on_subagent_start(info)
    assert ui._live is not None
    seen = {}

    def question(*args, **kwargs):
        seen["live"] = ui._live
        return "y"

    ui._question = question
    ui.confirm(ToolCallInfo("edit_file", "a.py"), Display("+x", "diff", "a.py"), dangerous=False)
    assert seen["live"] is None  # вопрос — без временной области
    ui.on_subagent_activity(info, "edit_file")
    assert ui._live is not None  # индикатор суб-агента вернулся
    ui.on_subagent_end(info, None)
    assert ui._live is None


def test_subagent_hint_names_stop_keys():
    import contextlib

    from gigacrew.agent.events import SubagentInfo

    ui, _ = _live_console()
    assert "Esc" not in ui._subagent_hint()  # Esc не слушается — подсказки нет
    ui.set_interrupt_keys("Esc — прервать", contextlib.nullcontext, stop_subagent=True)
    info = SubagentInfo("explore")
    ui.on_subagent_start(info)
    assert "Esc — завершить с итогом" in ui._subagent_hint()
    assert "Ctrl+C — прервать всё" in ui._subagent_hint()
    ui.on_subagent_activity(info, "подвожу итог", wrapping_up=True)
    assert "Esc — оборвать суб-агента" in ui._subagent_hint()
    assert ui._subagent_label() == "explore · подвожу итог"
    ui.on_subagent_end(info, None)
