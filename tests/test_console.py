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
    from gigacrew.permissions import ToolKind

    ui.on_tool_result(
        ToolCallInfo("run_shell", "ls", ToolKind.COMMAND),
        ToolResult(content="", summary="ok", display=Display("\x1b[2Jcleared", title="$ ls")),
        previewed=False,
    )
    out = buf.getvalue()
    assert "\x1b" not in out and "\x07" not in out
    assert "there" in out and "cleared" in out


def test_edit_shows_file_and_line_counts_not_diff():
    from gigacrew.permissions import ToolKind

    ui, buf = _console()
    diff = "--- a/a.py\n+++ b/a.py\n@@ -1 +1,2 @@\n-old\n+UNIQUE\n+new\n"
    result = ToolResult(content="", summary="изменён a.py", display=Display(diff, "diff", "a.py"))
    ui.on_tool_result(ToolCallInfo("edit_file", "a.py", ToolKind.EDIT), result, previewed=False)
    out = buf.getvalue()
    assert "● Изменил a.py  +2 −1" in out and "UNIQUE" not in out


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


def test_command_shows_last_line_or_tail_of_failure():
    from gigacrew.permissions import ToolKind

    ui, buf = _console()
    text = "\n".join(f"line{i}" for i in range(100))
    call = ToolCallInfo("run_shell", "seq 100", ToolKind.COMMAND)
    ui.on_tool_result(
        call, ToolResult(content=text, summary="ok", display=Display(text)), previewed=False
    )
    assert buf.getvalue() == "● $ seq 100  ✔ line99\n"
    failed = text + "\n" + "=" * 30 + " short test summary " + "=" * 30
    ui.on_tool_result(
        call,
        ToolResult(content="", ok=False, summary="код 1", display=Display(failed), exit_code=1),
        previewed=False,
    )
    out = buf.getvalue()
    assert "● $ seq 100  ✘ код 1" in out
    assert "│ line97\n" in out and "│ line99\n" in out and "line96" not in out
    assert "short test summary" not in out  # строка-разделитель в хвост не идёт


def test_confirm_question_includes_summary(monkeypatch):
    from gigacrew.agent.events import Approval
    from gigacrew.permissions import ToolKind

    ui, buf = _console()
    answers = iter(["д", "", "a", "a"])
    monkeypatch.setattr(ui._c, "input", lambda prompt: (ui._c.print(prompt), next(answers))[1])
    edit = ToolCallInfo("edit_file", "a.py", ToolKind.EDIT)
    assert ui.confirm(edit, None, dangerous=False).approval is Approval.YES
    out = buf.getvalue()
    assert "▌ edit_file a.py\n" in out and "▌ ? Применить правку? [y/N/a]" in out
    assert "до конца сессии" in out
    danger = ToolCallInfo("run_shell", "rm -rf x", ToolKind.COMMAND, reason="удаление без возврата")
    assert ui.confirm(danger, None, dangerous=True).approval is Approval.NO
    out = buf.getvalue()
    assert "▌ $ rm -rf x\n" in out and "▌ почему опасно  удаление без возврата" in out
    assert "▌ ⚠ Выполнить опасную команду? [y/N] " in out and "запомнить нельзя" in out
    command = ToolCallInfo(
        "run_shell",
        "pytest -q",
        ToolKind.COMMAND,
        reason="выполняет код проекта",
        remember="всегда разрешать «pytest»",
        history="раньше: разрешали 2",
    )
    assert ui.confirm(command, None, dangerous=False).approval is Approval.ALWAYS
    out = buf.getvalue()
    assert "▌ $ pytest -q\n" in out and "▌ ? Выполнить команду? [y/N/a]" in out
    assert "▌ почему спрашиваю  выполняет код проекта" in out and "раньше: разрешали 2" in out
    assert "a — всегда разрешать «pytest»" in out
    assert (
        ui.confirm(danger, None, dangerous=True).approval is Approval.NO
    )  # «a» для опасного — нет


def test_reviewer_lines(monkeypatch):
    from gigacrew.agent.events import Approval, Review
    from gigacrew.permissions import ToolKind

    ui, buf = _console()
    command = ToolCallInfo("run_shell", "./deploy-check", ToolKind.COMMAND)
    ui.on_review_start(command)  # без терминала индикатора нет
    ui.on_review_end(command, Review("R", True, "проверка проекта"))
    ui.on_review_end(command, Review("R", False, "не просили"))  # отказ — в вопросе
    ui.on_review_end(command, None)
    assert buf.getvalue().strip() == "рецензент (R) разрешил: проверка проекта"
    # у шага — строкой под ним, когда он выполнится
    ui, buf = _console()
    ui.on_tool_call(command)
    ui.on_review_end(command, Review("R", True, "проверка проекта"))
    assert buf.getvalue() == ""
    ui.on_tool_result(command, ToolResult(content="", summary="ok", exit_code=0), previewed=False)
    assert buf.getvalue() == (
        "● $ ./deploy-check  ✔ код 0\n  рецензент (R) разрешил: проверка проекта\n"
    )
    ui, buf = _console()
    monkeypatch.setattr(ui._c, "input", lambda prompt: "y")
    asked = ToolCallInfo(
        "run_shell",
        "./deploy-check",
        ToolKind.COMMAND,
        reason="выполняет код проекта",
        review="рецензент (R): не просили",
    )
    assert ui.confirm(asked, None, dangerous=False).approval is Approval.YES
    out = buf.getvalue()
    assert out.index("почему спрашиваю") < out.index("рецензент (R): не просили")


def test_confirm_free_text_is_feedback_and_no_answer_is_marked(monkeypatch):
    from gigacrew.agent.events import Approval
    from gigacrew.permissions import ToolKind

    ui, _ = _console()
    answers = iter(["Используй uv run pytest", "ф", "нет"])
    monkeypatch.setattr(ui._c, "input", lambda prompt: next(answers))
    command = ToolCallInfo("run_shell", "pytest", ToolKind.COMMAND)
    result = ui.confirm(command, None, dangerous=False)
    assert result.approval is Approval.NO and result.feedback == "Используй uv run pytest"
    assert ui.confirm(command, None, dangerous=False).approval is Approval.ALWAYS  # «ф» = «a»
    assert ui.confirm(command, None, dangerous=False).feedback == ""

    def eof(prompt):
        raise EOFError

    monkeypatch.setattr(ui._c, "input", eof)
    result = ui.confirm(command, None, dangerous=False)
    assert result.approval is Approval.NO and not result.answered


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
    assert "▌ ? Какую БД?   БД · 1 из 2" in out and "с сервером" in out
    assert "3. Свой ответ" in out
    assert "● Какую БД?  " in out  # итог в ленте


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
    assert "▄▄▄██▀▀▀▀█" in out and "┃╺┓" in out  # ракета и крупное название
    assert "v9.9.9" in out and "Alpha-Max" in out and "/work/proj" in out
    assert "/help справка" in out


def test_banner_without_rocket_in_narrow_terminal():
    buf = io.StringIO()
    ui = Console(no_color=True, file=buf, width=60)
    ui.banner(version="1", model="m", root="/r")
    out = buf.getvalue()
    assert "┃╺┓" in out and "▀" not in out


def test_banner_shows_mode():
    from gigacrew.permissions import PermissionMode

    ui, buf = _console()
    ui.banner(version="1", model="m", root="/r", mode=PermissionMode.PLAN)
    assert "режим   ⏸ план   Shift+Tab — сменить" in _plain(buf)


def test_question_body_is_shown_before_options(monkeypatch):
    from dataclasses import replace

    ui, buf = _asking_console(monkeypatch, "1")
    question = replace(
        _DB, header="План", body="## Шаги\n1. **Поправить** `a.py`", source="plans/x.md"
    )
    assert ui.ask_user([question]) == [Answer(("PostgreSQL",))]
    out = buf.getvalue()
    assert "План" in out and "Шаги" in out and "Поправить" in out and "**" not in out
    assert "plans/x.md" in out  # файл — в подписи рамки
    assert out.index("Поправить") < out.index("? Какую БД?")


def test_turn_indicator_shows_current_mode():
    import contextlib

    from gigacrew.permissions import PermissionMode

    ui, buf = _live_console(width=120)
    mode = [PermissionMode.AUTO]
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
    assert "● task explore · foo  ✔ explore · инструментов: 2" in out
    assert "  1 поиск\n" in out  # что сделал суб-агент — под строкой его вызова
    assert "✘ read_file nope.py  Файл не найден: nope.py" in out  # неудача шага видна
    assert "СЕКРЕТНЫЙ ОТЧЁТ" not in out  # отчёт получает основной агент, не пользователь
    assert "✦ Итог основного" in out
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

    ui._question_raw = question
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


# ------------------- подтверждение: команда целиком ------------------- #
def test_confirm_shows_long_command_entirely_and_offers_pager(monkeypatch):
    from gigacrew.agent.events import Approval
    from gigacrew.permissions import ToolKind

    buf = io.StringIO()
    ui = Console(no_color=True, file=buf, width=80)
    from rich.console import ConsoleDimensions

    monkeypatch.setattr(type(ui._c), "size", property(lambda self: ConsoleDimensions(80, 24)))
    body = "\n".join(f"print('line {i:03d}')" for i in range(200))
    command = f"python3 - <<'EOF'\n{body}\nEOF"
    pages = []
    monkeypatch.setattr(ui, "_page", lambda preview: pages.append(preview.text))
    answers = iter(["v", "y"])
    monkeypatch.setattr(ui._c, "input", lambda prompt: (ui._c.print(prompt), next(answers))[1])
    call = ToolCallInfo("run_shell", command[:70], ToolKind.COMMAND, purpose="печать строк")
    result = ui.confirm(call, Display(f"$ {command}", title="команда"), dangerous=False)
    assert result.approval is Approval.YES
    out = buf.getvalue()
    assert all(f"line {i:03d}" in out for i in range(200))  # ничего не обрезано
    assert "202 │ EOF" in out and "202 строки" in out  # номера строк и размер
    assert "v — открыть целиком" in out and "[y/N/a/v]" in out
    assert "▌ $ python3 - <<'EOF' …\n" in out  # первая строка команды — над рамкой
    assert "▌ зачем  печать строк  со слов модели" in out
    assert pages == [f"$ {command}"]  # просмотрщик открыт, потом вопрос задан снова


def test_confirm_makes_invisible_characters_visible():
    from gigacrew.permissions import ToolKind

    ui, buf = _console()
    ui._c.input = lambda prompt: "n"
    command = "ls ‮txt.exe\x1b[2K"
    ui.confirm(
        ToolCallInfo("run_shell", "ls", ToolKind.COMMAND), Display(f"$ {command}"), dangerous=True
    )
    out = buf.getvalue()
    assert "⟨U+202E⟩" in out and "⟨U+001B⟩" in out and "‮" not in out
    assert "v — открыть" not in out  # короткая команда помещается на экран


# ------------------------- группы шагов в ленте ------------------------- #
def _read(path: str) -> ToolCallInfo:
    return ToolCallInfo("read_file", path, read_only=True)


def _edit(path: str) -> ToolCallInfo:
    from gigacrew.permissions import ToolKind

    return ToolCallInfo("edit_file", path, ToolKind.EDIT)


def _diff(added: int, removed: int) -> Display:
    body = "".join(f"-old{i}\n" for i in range(removed)) + "".join(
        f"+new{i}\n" for i in range(added)
    )
    return Display(f"--- a/x\n+++ b/x\n@@ -1 +1 @@\n{body}", "diff", "x")


def _step(ui, call, result):
    ui.on_tool_call(call)
    ui.on_tool_start(call)
    ui.on_tool_end(call)
    ui.on_tool_result(call, result, previewed=False)


def test_read_calls_collapse_into_group_until_answer():
    ui, buf = _console()
    _step(ui, _read("a.py"), ToolResult(content="", summary="прочитан a.py (3 строк)"))
    search = ToolCallInfo("search_content", "/foo/", read_only=True)
    _step(ui, search, ToolResult(content="", summary="совпадений: 2"))
    _step(ui, _read("nope.py"), ToolResult(content="", ok=False, summary="Файл не найден"))
    assert buf.getvalue() == ""  # группа ещё идёт — в ленте ничего
    ui.on_assistant_text("Нашёл")
    out = "".join(line.rstrip() + "\n" for line in buf.getvalue().splitlines())
    assert out == (
        "● Изучил код  прочитал 1 файл, 1 поиск\n"
        "  a.py  /foo/\n"
        "  ✘ read_file nope.py  Файл не найден\n"
        "\n"
        "✦ Нашёл\n"
    )


def test_single_read_is_one_line_without_repeated_path():
    ui, buf = _console()
    _step(ui, _read("a.py"), ToolResult(content="", summary="прочитан a.py (3 строк)"))
    ui.on_turn_end(TurnStats(steps=1))
    assert buf.getvalue().startswith("● read_file a.py  ✔ прочитан (3 строк)\n")


def test_edits_collapse_per_file():
    ui, buf = _console()
    for path, stat in (("a.py", (3, 1)), ("b.py", (1, 0)), ("a.py", (2, 2))):
        _step(ui, _edit(path), ToolResult(content="", summary="изменён", display=_diff(*stat)))
    command = ToolCallInfo("run_shell", "pytest")
    ui.on_tool_call(command)  # другой вызов закрывает группу правок
    assert buf.getvalue() == ("● Изменил 2 файла  +6 −3\n  a.py  +5 −3\n  b.py  +1 −0\n")


def test_group_is_shown_in_live_area_while_running():
    ui, buf = _live_console(width=100)
    ui.on_stream_start()
    _step(ui, _read("a.py"), ToolResult(content="", summary="прочитан"))
    ui.on_tool_call(_read("b.py"))
    rows = [row.plain for row in ui._live_steps()]
    assert rows[0].startswith("● Изучаю код  2 вызова")
    assert rows[1].startswith("  ✔ read_file  a.py") and rows[2].startswith("  ◌ read_file  b.py")
    ui.stop_live()


def test_asked_call_leaves_group_and_gets_own_line(monkeypatch):
    ui, buf = _console()
    monkeypatch.setattr(ui._c, "input", lambda prompt: "y")
    _step(ui, _edit("a.py"), ToolResult(content="", summary="изменён", display=_diff(1, 0)))
    asked = _edit("b.py")
    ui.on_tool_call(asked)
    from dataclasses import replace

    ui.confirm(replace(asked, reason="хук требует подтверждения"), _diff(2, 0), dangerous=False)
    ui.on_tool_result(
        asked, ToolResult(content="", summary="изменён", display=_diff(2, 0)), previewed=True
    )
    out = buf.getvalue()
    # группа до вопроса — итогом, спрошенная правка — своей строкой после ответа
    assert out.index("● Изменил a.py  +1 −0") < out.index("▌ edit_file b.py")
    assert out.rstrip().endswith("● Изменил b.py  +2 −0")


def test_sixteen_color_terminal_gets_ansi_palette(monkeypatch):
    from gigacrew.ui.theme import MINT, MINT_ANSI

    monkeypatch.delenv("COLORTERM", raising=False)
    monkeypatch.setenv("TERM", "xterm")
    ui, _ = _live_console()
    assert ui._p is MINT_ANSI
    monkeypatch.setenv("COLORTERM", "truecolor")
    ui, _ = _live_console()
    assert ui._p is MINT


def test_prompt_toolkit_color_depth_follows_rich(monkeypatch):
    from prompt_toolkit.output import ColorDepth

    from gigacrew.ui import colors

    monkeypatch.delenv("PROMPT_TOOLKIT_COLOR_DEPTH", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    for system, depth in [
        ("truecolor", ColorDepth.TRUE_COLOR),
        ("256", ColorDepth.DEPTH_8_BIT),
        ("standard", ColorDepth.DEPTH_4_BIT),
        (None, None),
    ]:
        monkeypatch.setattr(colors, "color_system", lambda s=system: s)
        assert colors.prompt_color_depth(False) == depth
    assert colors.prompt_color_depth(True) == ColorDepth.MONOCHROME
