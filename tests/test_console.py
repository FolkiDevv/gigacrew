"""Тесты терминального UI: очистка управляющих символов, рендер событий."""

from __future__ import annotations

import contextlib
import io
import re
from collections.abc import Callable, Generator, Iterator
from dataclasses import replace
from pathlib import Path
from typing import NoReturn, override

import pytest
from fakes import ScriptedProvider, text_turn, tool_turn
from prompt_toolkit.output import ColorDepth
from rich.console import ConsoleDimensions

from gigacrew.agent.events import (
    Approval,
    CompactResult,
    Review,
    SubagentInfo,
    ToolCallInfo,
    TurnStats,
)
from gigacrew.agent.loop import Agent
from gigacrew.config import Config
from gigacrew.permissions import PermissionMode, ToolKind
from gigacrew.project.workspace import Workspace
from gigacrew.tools.base import Display, ToolContext, ToolResult, build_default_registry
from gigacrew.tools.questions import Answer, Question, QuestionOption, QuestionsUnavailableError
from gigacrew.tools.shell import RunShellParams, RunShellTool
from gigacrew.ui import colors
from gigacrew.ui.console import Console, sanitize
from gigacrew.ui.format import clip_lines, format_tokens, plural
from gigacrew.ui.theme import MINT, MINT_ANSI


def _console() -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return Console(no_color=True, file=buf), buf


def _has_live_area(ui: Console) -> bool:
    # Проверка через функцию, а не ``ui._live is None`` в тесте: после assert mypy
    # держит суженный тип ``ui._live`` до конца теста, хотя вызовы UI его меняют.
    return ui._live is not None


def _echoing_input(ui: Console, answers: Iterator[str]) -> Callable[[str], str]:
    # Ввод, как в терминале: приглашение остаётся в выводе, ответ — следующий заготовленный.
    def echo_input(prompt: str) -> str:
        ui._c.print(prompt)
        return next(answers)

    return echo_input


def test_sanitize_strips_escape_sequences() -> None:
    assert sanitize("a\x1b[31mred\x1b[0m\tb\nc\x07\x9b") == "a[31mred[0m\tb\nc"


def test_stream_and_output_are_sanitized() -> None:
    ui, buf = _console()
    ui.on_stream_start()
    ui.on_stream_delta("hi \x1b]0;pwned\x07there")
    ui.on_stream_end()
    ui.on_tool_result(
        ToolCallInfo("run_shell", "ls", ToolKind.COMMAND),
        ToolResult(content="", summary="ok", display=Display("\x1b[2Jcleared", title="$ ls")),
        previewed=False,
    )
    out = buf.getvalue()
    assert "\x1b" not in out
    assert "\x07" not in out
    assert "there" in out
    assert "cleared" in out


def test_edit_shows_file_and_line_counts_not_diff() -> None:
    ui, buf = _console()
    diff = "--- a/a.py\n+++ b/a.py\n@@ -1 +1,2 @@\n-old\n+UNIQUE\n+new\n"
    edit = ToolResult(content="", summary="изменён a.py", display=Display(diff, "diff", "a.py"))
    ui.on_tool_result(ToolCallInfo("edit_file", "a.py", ToolKind.EDIT), edit, previewed=False)
    out = buf.getvalue()
    assert "● Изменил a.py  +2 −1" in out
    assert "UNIQUE" not in out


def test_turn_stats_line() -> None:
    ui, buf = _console()
    ui.on_turn_end(
        TurnStats(
            steps=2,
            tool_calls=1,
            prompt_tokens=200,
            completion_tokens=20,
            context_tokens=12_345,
            duration_s=3.21,
        ),
    )
    out = buf.getvalue()
    assert "2 шага" in out
    assert "1 инструмент " in out
    assert "3.2 с" in out
    assert "контекст ~12.3k" in out
    assert "потрачено 220" in out


# ------------------------------ живой режим ------------------------------ #
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _live_console(width: int = 60) -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return Console(file=buf, force_terminal=True, width=width, auto_refresh=False), buf


def _plain(buf: io.StringIO) -> str:
    return _ANSI_RE.sub("", buf.getvalue())


def test_streamed_markdown_is_formatted() -> None:
    ui, buf = _live_console()
    ui.on_stream_start()
    for chunk in ("# Заго", "ловок\n\nЭто **жир", "ный** текст\n", "\n- пункт\n", "\nконец"):
        ui.on_stream_delta(chunk)
    ui.on_stream_end()
    out = _plain(buf)
    assert "Заголовок" in out
    assert "жирный текст" in out
    assert "конец" in out
    assert "**" not in out
    assert "# " not in out
    assert "\x1b[1m" in buf.getvalue()  # жирный — настоящим стилем терминала
    assert ui._live is None


def test_live_area_stopped_after_interrupt(tmp_path: Path) -> None:
    class Interrupting(ScriptedProvider):
        @override
        def stream(
            self,
            messages: object,
            tools: object = None,
            *,
            on_delta: Callable[[str], None] | None = None,
            **kwargs: object,
        ) -> NoReturn:
            assert on_delta is not None
            on_delta("Начало ответа\n\nещё")
            raise KeyboardInterrupt

    ui, buf = _live_console()
    cfg = Config(access_key="x", project_root=tmp_path, stream=True)
    agent = Agent(Interrupting([text_turn("x")]), build_default_registry(), cfg, ui)
    with pytest.raises(KeyboardInterrupt):
        agent.run_turn("привет")
    assert ui._live is None
    assert "Начало ответа" in _plain(buf)
    assert "ещё" in _plain(buf)


def test_tool_spinner_lifecycle() -> None:
    ui, _ = _live_console()
    call = ToolCallInfo("run_shell", "sleep 1")
    ui.on_tool_start(call)
    assert _has_live_area(ui)
    ui.on_tool_end(call)
    assert not _has_live_area(ui)
    ui.stop_live()  # идемпотентно


def test_single_live_area() -> None:
    ui, _ = _live_console()
    ui.on_stream_start()
    first = ui._live
    ui.on_tool_start(ToolCallInfo("x"))
    assert ui._live is not first
    assert first is not None
    assert not first.active
    ui.stop_live()


def _interrupt_progress(ui: Console, counter: dict[str, int]) -> None:
    with ui.progress("индексирую", lambda: f"{counter['n']} файлов"):
        assert ui._live is not None
        counter["n"] = 42
        ui._live._live.refresh()
        raise KeyboardInterrupt


def test_progress_live_label_updates_and_stops_on_interrupt() -> None:
    ui, buf = _live_console()
    counter = {"n": 0}
    with pytest.raises(KeyboardInterrupt):
        _interrupt_progress(ui, counter)
    assert ui._live is None
    assert "индексирую: 42 файлов" in _plain(buf)


def test_progress_without_terminal_prints_title_once() -> None:
    ui, buf = _console()
    with ui.progress("индексирую", lambda: "1 файл"):
        assert ui._live is None
    assert buf.getvalue() == "индексирую…\n"


def test_compaction_progress_and_result() -> None:
    ui, buf = _console()
    ui.on_compact_start(auto=True)
    ui.on_compact_end(CompactResult(before_tokens=24_000, after_tokens=6_500, messages=31))
    out = buf.getvalue()
    assert "контекст почти заполнен — сжимаю историю…" in out
    assert "контекст сжат: ~24k → ~6.5k ток." in out
    assert "31 сообщение" in out

    ui, buf = _live_console()
    ui.on_compact_start(auto=False)
    assert _has_live_area(ui)
    ui.on_compact_end(None)  # не удалось — только убрать индикатор
    assert not _has_live_area(ui)
    assert "сжат" not in _plain(buf)


def test_no_live_area_without_terminal() -> None:
    ui, buf = _console()
    ui.on_stream_start()
    ui.on_tool_start(ToolCallInfo("x"))
    assert ui._live is None
    ui.on_stream_end()
    assert not buf.getvalue()


def test_command_shows_last_line_or_tail_of_failure() -> None:
    ui, buf = _console()
    text = "\n".join(f"line{index}" for index in range(100))
    call = ToolCallInfo("run_shell", "seq 100", ToolKind.COMMAND)
    ui.on_tool_result(
        call,
        ToolResult(content=text, summary="ok", display=Display(text)),
        previewed=False,
    )
    assert buf.getvalue() == "● $ seq 100  ✔ line99\n"
    separator = "=" * 30
    failed = f"{text}\n{separator} short test summary {separator}"
    ui.on_tool_result(
        call,
        ToolResult(content="", ok=False, summary="код 1", display=Display(failed), exit_code=1),
        previewed=False,
    )
    out = buf.getvalue()
    assert "● $ seq 100  ✘ код 1" in out
    assert "│ line97\n" in out
    assert "│ line99\n" in out
    assert "line96" not in out
    assert "short test summary" not in out  # строка-разделитель в хвост не идёт


def test_confirm_question_includes_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    ui, buf = _console()
    answers = iter(["д", "", "a", "a"])
    monkeypatch.setattr(ui._c, "input", _echoing_input(ui, answers))
    edit = ToolCallInfo("edit_file", "a.py", ToolKind.EDIT)
    assert ui.confirm(edit, None, dangerous=False).approval is Approval.YES
    out = buf.getvalue()
    assert "▌ edit_file a.py\n" in out
    assert "▌ ? Применить правку? [y/N/a]" in out
    assert "до конца сессии" in out
    danger = ToolCallInfo("run_shell", "rm -rf x", ToolKind.COMMAND, reason="удаление без возврата")
    assert ui.confirm(danger, None, dangerous=True).approval is Approval.NO
    out = buf.getvalue()
    assert "▌ $ rm -rf x\n" in out
    assert "▌ почему опасно  удаление без возврата" in out
    assert "▌ ⚠ Выполнить опасную команду? [y/N] " in out
    assert "запомнить нельзя" in out
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
    assert "▌ $ pytest -q\n" in out
    assert "▌ ? Выполнить команду? [y/N/a]" in out
    assert "▌ почему спрашиваю  выполняет код проекта" in out
    assert "раньше: разрешали 2" in out
    assert "a — всегда разрешать «pytest»" in out
    assert (
        ui.confirm(danger, None, dangerous=True).approval is Approval.NO
    )  # «a» для опасного — нет


def test_reviewer_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    ui, buf = _console()
    command = ToolCallInfo("run_shell", "./deploy-check", ToolKind.COMMAND)
    ui.on_review_start(command)  # без терминала индикатора нет
    ui.on_review_end(command, Review("R", allowed=True, reason="проверка проекта"))
    ui.on_review_end(command, Review("R", allowed=False, reason="не просили"))  # отказ — в вопросе
    ui.on_review_end(command, None)
    assert buf.getvalue().strip() == "рецензент (R) разрешил: проверка проекта"
    # у шага — строкой под ним, когда он выполнится
    ui, buf = _console()
    ui.on_tool_call(command)
    ui.on_review_end(command, Review("R", allowed=True, reason="проверка проекта"))
    assert not buf.getvalue()
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


def test_confirm_free_text_is_feedback_and_no_answer_is_marked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ui, _ = _console()
    answers = iter(["Используй uv run pytest", "ф", "нет"])
    monkeypatch.setattr(ui._c, "input", lambda prompt: next(answers))
    command = ToolCallInfo("run_shell", "pytest", ToolKind.COMMAND)
    outcome = ui.confirm(command, None, dangerous=False)
    assert outcome.approval is Approval.NO
    assert outcome.feedback == "Используй uv run pytest"
    assert ui.confirm(command, None, dangerous=False).approval is Approval.ALWAYS  # «ф» = «a»
    assert not ui.confirm(command, None, dangerous=False).feedback

    def eof(prompt: str) -> NoReturn:
        raise EOFError

    monkeypatch.setattr(ui._c, "input", eof)
    outcome = ui.confirm(command, None, dangerous=False)
    assert outcome.approval is Approval.NO
    assert not outcome.answered


def test_format_helpers() -> None:
    assert [format_tokens(count) for count in (950, 12_345, 120_000, 1_234_567)] == [
        "950",
        "12.3k",
        "120k",
        "1.2M",
    ]
    assert [plural(count, "шаг", "шага", "шагов") for count in (1, 2, 5, 11, 21, 112)] == [
        "шаг",
        "шага",
        "шагов",
        "шагов",
        "шаг",
        "шагов",
    ]
    assert clip_lines("a\nb") == "a\nb"
    assert len(clip_lines("x" * 10_000)) <= 300
    padding = "x" * 290
    numbers = [format(index, "03d") for index in range(40)]
    rows = "\n".join(f"{number} {padding}" for number in numbers)
    clipped = clip_lines(f"{rows}\nИТОГ: 3 failed")
    assert len(clipped) <= 4000
    assert clipped.startswith("000")
    assert clipped.endswith("ИТОГ: 3 failed")


def test_animation_thread_stops() -> None:
    buf = io.StringIO()
    ui = Console(file=buf, force_terminal=True, width=60)  # с потоком анимации
    ui.on_stream_start()
    area = ui._live
    ui.on_stream_delta("текст\n\nещё\n")
    ui.on_stream_end()
    assert area is not None
    assert area._thread is not None
    assert not area._thread.is_alive()
    assert "текст" in _plain(buf)


# ---------------------------- вопросы агента ---------------------------- #
_DB = Question(
    "Какую БД?",
    (QuestionOption("PostgreSQL", "с сервером"), QuestionOption("SQLite", "файл")),
    header="БД",
)
_PARTS = Question(
    "Что добавить?",
    (QuestionOption("Логи"), QuestionOption("Метрики")),
    multi_select=True,
)


def _asking_console(
    monkeypatch: pytest.MonkeyPatch,
    *replies: str | BaseException,
) -> tuple[Console, io.StringIO]:
    monkeypatch.setattr("gigacrew.ui.console._stdin_is_terminal", lambda: True)
    ui, buf = _console()
    queue = list(replies)

    def fake_input(prompt: str) -> str:
        reply = queue.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

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
def test_plain_questions(
    monkeypatch: pytest.MonkeyPatch,
    replies: list[str],
    expected: list[Answer],
) -> None:
    second = ["1, 2"] if len(expected) == 1 else ["1"]
    ui, buf = _asking_console(monkeypatch, *replies, *second)
    answers = ui.ask_user([_DB, _PARTS])
    assert answers is not None
    assert answers[0] == expected[0]
    out = buf.getvalue()
    assert "▌ ? Какую БД?   БД · 1 из 2" in out
    assert "с сервером" in out
    assert "3. Свой ответ" in out
    assert "● Какую БД?  " in out  # итог в ленте


def test_plain_multi_select(monkeypatch: pytest.MonkeyPatch) -> None:
    ui, _ = _asking_console(monkeypatch, "1", "1, 2")
    answers = ui.ask_user([_DB, _PARTS])
    assert answers is not None
    assert answers[1] == Answer(("Логи", "Метрики"))


def test_questions_declined(monkeypatch: pytest.MonkeyPatch) -> None:
    ui, buf = _asking_console(monkeypatch, KeyboardInterrupt())
    assert ui.ask_user([_DB, _PARTS]) is None
    assert "отказался" in buf.getvalue()


def test_questions_need_keyboard() -> None:
    ui, _ = _console()
    with pytest.raises(QuestionsUnavailableError):
        ui.ask_user([_DB])


def test_questions_pause_esc_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    ui, _ = _asking_console(monkeypatch, "1")
    entered: list[bool] = []

    @contextlib.contextmanager
    def guard() -> Generator[None]:
        entered.append(True)
        yield

    ui.set_interrupt_keys("Esc — прервать", guard)
    ui.ask_user([_DB])
    assert entered == [True]


def test_banner_shows_rocket_title_and_meta() -> None:
    ui, buf = _console()
    ui.banner(version="9.9.9", model="Alpha-Max", root="/work/proj", hints=[("/help", "справка")])
    out = _plain(buf)
    assert "▄▄▄██▀▀▀▀█" in out
    assert "┃╺┓" in out
    assert "v9.9.9" in out
    assert "Alpha-Max" in out
    assert "/work/proj" in out
    assert "/help справка" in out


def test_banner_without_rocket_in_narrow_terminal() -> None:
    buf = io.StringIO()
    ui = Console(no_color=True, file=buf, width=60)
    ui.banner(version="1", model="m", root="/r")
    out = buf.getvalue()
    assert "┃╺┓" in out
    assert "▀" not in out


def test_banner_shows_mode() -> None:
    ui, buf = _console()
    ui.banner(version="1", model="m", root="/r", mode=PermissionMode.PLAN)
    assert "режим   ⏸ план   Shift+Tab — сменить" in _plain(buf)


def test_question_body_is_shown_before_options(monkeypatch: pytest.MonkeyPatch) -> None:
    ui, buf = _asking_console(monkeypatch, "1")
    question = replace(
        _DB,
        header="План",
        body="## Шаги\n1. **Поправить** `a.py`",
        source="plans/x.md",
    )
    assert ui.ask_user([question]) == [Answer(("PostgreSQL",))]
    out = buf.getvalue()
    assert "План" in out
    assert "Шаги" in out
    assert "Поправить" in out
    assert "**" not in out
    assert "plans/x.md" in out  # файл — в подписи рамки
    assert out.index("Поправить") < out.index("? Какую БД?")


def test_turn_indicator_shows_current_mode() -> None:
    ui, buf = _live_console(width=120)
    mode = [PermissionMode.AUTO]
    ui.set_interrupt_keys("Esc — прервать", contextlib.nullcontext)
    ui.set_mode_hint(lambda: f"{mode[0].label} (Shift+Tab)")
    ui.on_stream_start()
    mode[0] = PermissionMode.PLAN  # Shift+Tab во время хода
    assert ui._live is not None
    ui._live._live.refresh()
    ui.stop_live()
    assert "Esc — прервать  ·  план (Shift+Tab)" in _plain(buf)


# ------------------------------ суб-агенты ------------------------------ #
def test_subagent_steps_nested_and_text_hidden(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
    provider = ScriptedProvider(
        [
            tool_turn("task", {"agent": "explore", "prompt": "найди foo", "description": "foo"}),
            tool_turn("search_content", {"pattern": "def foo"}),
            tool_turn("read_file", {"path": "nope.py"}),
            text_turn("СЕКРЕТНЫЙ ОТЧЁТ суб-агента"),
            text_turn("Итог основного"),
        ],
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


def test_subagent_live_area_survives_confirm(monkeypatch: pytest.MonkeyPatch) -> None:
    ui, _ = _live_console()
    subagent = SubagentInfo("general", "правка")
    ui.on_subagent_start(subagent)
    assert _has_live_area(ui)
    seen: dict[str, object] = {}

    def question(*args: object, **kwargs: object) -> str:
        seen["live"] = ui._live
        return "y"

    monkeypatch.setattr(ui, "_question_raw", question)
    ui.confirm(ToolCallInfo("edit_file", "a.py"), Display("+x", "diff", "a.py"), dangerous=False)
    assert seen["live"] is None  # вопрос — без временной области
    ui.on_subagent_activity(subagent, "edit_file")
    assert _has_live_area(ui)  # индикатор суб-агента вернулся
    ui.on_subagent_end(subagent, None)
    assert not _has_live_area(ui)


def test_subagent_hint_names_stop_keys() -> None:
    ui, _ = _live_console()
    assert "Esc" not in ui._subagent_hint()  # Esc не слушается — подсказки нет
    ui.set_interrupt_keys("Esc — прервать", contextlib.nullcontext, stop_subagent=True)
    subagent = SubagentInfo("explore")
    ui.on_subagent_start(subagent)
    assert "Esc — завершить с итогом" in ui._subagent_hint()
    assert "Ctrl+C — прервать всё" in ui._subagent_hint()
    ui.on_subagent_activity(subagent, "подвожу итог", wrapping_up=True)
    assert "Esc — оборвать суб-агента" in ui._subagent_hint()
    assert ui._subagent_label() == "explore · подвожу итог"
    ui.on_subagent_end(subagent, None)


# ------------------- подтверждение: команда целиком ------------------- #
def test_confirm_shows_long_command_entirely_and_offers_pager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buf = io.StringIO()
    ui = Console(no_color=True, file=buf, width=80)
    monkeypatch.setattr(type(ui._c), "size", property(lambda _: ConsoleDimensions(80, 24)))
    numbers = [format(index, "03d") for index in range(200)]
    body = "\n".join(f"print('line {number}')" for number in numbers)
    command = f"python3 - <<'EOF'\n{body}\nEOF"
    pages: list[str] = []
    monkeypatch.setattr(ui, "_page", lambda preview: pages.append(preview.text))
    answers = iter(["v", "y"])
    monkeypatch.setattr(ui._c, "input", _echoing_input(ui, answers))
    call = ToolCallInfo("run_shell", command[:70], ToolKind.COMMAND, purpose="печать строк")
    outcome = ui.confirm(call, Display(f"$ {command}", title="команда"), dangerous=False)
    assert outcome.approval is Approval.YES
    out = buf.getvalue()
    assert all(f"line {number}" in out for number in numbers)  # ничего не обрезано
    assert "202 │ EOF" in out
    assert "202 строки" in out
    assert "v — открыть целиком" in out
    assert "[y/N/a/v]" in out
    assert "▌ $ python3 - <<'EOF' …\n" in out  # первая строка команды — над рамкой
    assert "▌ зачем  печать строк  со слов модели" in out
    assert pages == [f"$ {command}"]  # просмотрщик открыт, потом вопрос задан снова


def test_confirm_makes_invisible_characters_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    ui, buf = _console()
    monkeypatch.setattr(ui._c, "input", lambda prompt: "n")
    command = "ls \u202etxt.exe\x1b[2K"
    ui.confirm(
        ToolCallInfo("run_shell", "ls", ToolKind.COMMAND),
        Display(f"$ {command}"),
        dangerous=True,
    )
    out = buf.getvalue()
    assert "⟨U+202E⟩" in out
    assert "⟨U+001B⟩" in out
    assert "\u202e" not in out
    assert "v — открыть" not in out  # короткая команда помещается на экран


# ------------------------- группы шагов в ленте ------------------------- #
def _read(path: str) -> ToolCallInfo:
    return ToolCallInfo("read_file", path, read_only=True)


def _edit(path: str) -> ToolCallInfo:
    return ToolCallInfo("edit_file", path, ToolKind.EDIT)


def _diff(added: int, removed: int) -> Display:
    body = "".join(f"-old{index}\n" for index in range(removed)) + "".join(
        f"+new{index}\n" for index in range(added)
    )
    return Display(f"--- a/x\n+++ b/x\n@@ -1 +1 @@\n{body}", "diff", "x")


def _step(ui: Console, call: ToolCallInfo, tool_result: ToolResult) -> None:
    ui.on_tool_call(call)
    ui.on_tool_start(call)
    ui.on_tool_end(call)
    ui.on_tool_result(call, tool_result, previewed=False)


def test_read_calls_collapse_into_group_until_answer() -> None:
    ui, buf = _console()
    _step(ui, _read("a.py"), ToolResult(content="", summary="прочитан a.py (3 строк)"))
    search = ToolCallInfo("search_content", "/foo/", read_only=True)
    _step(ui, search, ToolResult(content="", summary="совпадений: 2"))
    _step(ui, _read("nope.py"), ToolResult(content="", ok=False, summary="Файл не найден"))
    assert not buf.getvalue()  # группа ещё идёт — в ленте ничего
    ui.on_assistant_text("Нашёл")
    out = "".join(f"{line.rstrip()}\n" for line in buf.getvalue().splitlines())
    assert out == (
        "● Изучил код  прочитал 1 файл, 1 поиск\n"
        "  a.py  /foo/\n"
        "  ✘ read_file nope.py  Файл не найден\n"
        "\n"
        "✦ Нашёл\n"
    )


def test_single_read_is_one_line_without_repeated_path() -> None:
    ui, buf = _console()
    _step(ui, _read("a.py"), ToolResult(content="", summary="прочитан a.py (3 строк)"))
    ui.on_turn_end(TurnStats(steps=1))
    assert buf.getvalue().startswith("● read_file a.py  ✔ прочитан (3 строк)\n")


def test_edits_collapse_per_file() -> None:
    ui, buf = _console()
    for path, stat in (("a.py", (3, 1)), ("b.py", (1, 0)), ("a.py", (2, 2))):
        _step(ui, _edit(path), ToolResult(content="", summary="изменён", display=_diff(*stat)))
    command = ToolCallInfo("run_shell", "pytest")
    ui.on_tool_call(command)  # другой вызов закрывает группу правок
    assert buf.getvalue() == ("● Изменил 2 файла  +6 −3\n  a.py  +5 −3\n  b.py  +1 −0\n")


def test_group_is_shown_in_live_area_while_running() -> None:
    ui, _buf = _live_console(width=100)
    ui.on_stream_start()
    _step(ui, _read("a.py"), ToolResult(content="", summary="прочитан"))
    ui.on_tool_call(_read("b.py"))
    rows = [row.plain for row in ui._live_steps()]
    assert rows[0].startswith("● Изучаю код  2 вызова")
    assert rows[1].startswith("  ✔ read_file  a.py")
    assert rows[2].startswith("  ◌ read_file  b.py")
    ui.stop_live()


def test_asked_call_leaves_group_and_gets_own_line(monkeypatch: pytest.MonkeyPatch) -> None:
    ui, buf = _console()
    monkeypatch.setattr(ui._c, "input", lambda prompt: "y")
    _step(ui, _edit("a.py"), ToolResult(content="", summary="изменён", display=_diff(1, 0)))
    asked = _edit("b.py")
    ui.on_tool_call(asked)
    ui.confirm(replace(asked, reason="хук требует подтверждения"), _diff(2, 0), dangerous=False)
    ui.on_tool_result(
        asked,
        ToolResult(content="", summary="изменён", display=_diff(2, 0)),
        previewed=True,
    )
    out = buf.getvalue()
    # группа до вопроса — итогом, спрошенная правка — своей строкой после ответа
    assert out.index("● Изменил a.py  +1 −0") < out.index("▌ edit_file b.py")
    assert out.rstrip().endswith("● Изменил b.py  +2 −0")


def test_sixteen_color_terminal_gets_ansi_palette(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COLORTERM", raising=False)
    monkeypatch.setenv("TERM", "xterm")
    ui, _ = _live_console()
    assert ui._p is MINT_ANSI
    monkeypatch.setenv("COLORTERM", "truecolor")
    ui, _ = _live_console()
    assert ui._p is MINT


def test_prompt_toolkit_color_depth_follows_rich(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROMPT_TOOLKIT_COLOR_DEPTH", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    for system, depth in (
        ("truecolor", ColorDepth.TRUE_COLOR),
        ("256", ColorDepth.DEPTH_8_BIT),
        ("standard", ColorDepth.DEPTH_4_BIT),
        (None, None),
    ):
        monkeypatch.setattr(colors, "color_system", lambda detected=system: detected)
        assert colors.prompt_color_depth(no_color=False) == depth
    assert colors.prompt_color_depth(no_color=True) == ColorDepth.MONOCHROME


# --------------- управляющие символы: все пути вывода в терминал --------------- #
_ESC = "x\x1b[2K\x1b[1A\x1b]0;owned\x07\x9bевил"


def _sink_tool_name(ui: Console, monkeypatch: pytest.MonkeyPatch) -> None:
    _step(ui, ToolCallInfo(f"read{_ESC}", "a.py"), ToolResult(content="", summary="ok"))
    ui.on_turn_end(TurnStats(steps=1))


def _sink_diff_title(ui: Console, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ui._c, "input", lambda prompt: "n")
    preview = Display("--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n", "diff", f"a{_ESC}.py")
    ui.confirm(ToolCallInfo("edit_file", "a.py", ToolKind.EDIT), preview, dangerous=False)


def _sink_exact_title(ui: Console, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ui._c, "input", lambda prompt: "n")
    preview = Display("$ make\nmake test", title=f"команда · {_ESC}")
    ui.confirm(ToolCallInfo("run_shell", "make", ToolKind.COMMAND), preview, dangerous=False)


def _sink_purpose(ui: Console, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ui._c, "input", lambda prompt: "n")
    call = ToolCallInfo("run_shell", "rm -rf x", ToolKind.COMMAND, purpose=_ESC)
    ui.confirm(call, None, dangerous=True)


def _sink_plain_question(ui: Console, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ui._c, "input", lambda prompt: "1")
    question = Question(f"Q{_ESC}?", (QuestionOption(f"A{_ESC}", f"d{_ESC}"), QuestionOption("B")))
    ui._ask_plain(question, 1, 1)


def _sink_help(ui: Console, monkeypatch: pytest.MonkeyPatch) -> None:
    ui.help([("/deploy [аргументы]", f"скилл · {_ESC}")])


def _sink_messages(ui: Console, monkeypatch: pytest.MonkeyPatch) -> None:
    ui.info(f"последние решения: {_ESC}")
    ui.warn(_ESC)
    ui.on_notice(_ESC, level="warn")


@pytest.mark.parametrize(
    "sink",
    [
        _sink_tool_name,
        _sink_diff_title,
        _sink_exact_title,
        _sink_purpose,
        _sink_plain_question,
        _sink_help,
        _sink_messages,
    ],
)
def test_control_characters_never_reach_terminal(
    sink: Callable[[Console, pytest.MonkeyPatch], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Rich (Text) не убирает ESC и C1: очистка — в одной точке на весь вывод консоли.
    ui, buf = _console()
    sink(ui, monkeypatch)
    out = buf.getvalue()
    assert "x[2K[1A]0;owned" in out
    assert "евил" in out
    assert "\x1b" not in out
    assert "\x07" not in out
    assert "\x9b" not in out


def test_control_characters_stripped_in_live_terminal_but_own_codes_kept() -> None:
    ui, buf = _live_console()
    ui.on_tool_call(ToolCallInfo(f"read{_ESC}", "a.py", read_only=True))
    ui.on_tool_start(ToolCallInfo(f"read{_ESC}", "a.py", read_only=True))
    ui.stop_live()
    out = buf.getvalue()
    assert "\x1b]0;owned" not in out
    assert "\x07" not in out
    assert "\x1b[" in out  # свои последовательности Rich (цвет, курсор) — на месте


# ------------------- невидимые символы в том, что подтверждается ------------------- #
@pytest.mark.parametrize(
    "char",
    [
        "\u200b",  # нулевой ширины
        "\u200d",  # соединитель нулевой ширины
        "\u200f",  # RLM
        "\u061c",  # ARABIC LETTER MARK — тоже управляет направлением текста
        "\u00ad",  # мягкий перенос
        "\u115f",  # заполнитель хангыля
        "\u3164",
        "\ufe0f",  # селектор варианта
        "\U000e0041",  # символ-тег
        "\u2028",  # разделитель строк
    ],
)
def test_confirm_marks_every_invisible_character_in_command(
    char: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ui, buf = _console()
    monkeypatch.setattr(ui._c, "input", lambda prompt: "n")
    call = ToolCallInfo("run_shell", "rm", ToolKind.COMMAND)
    ui.confirm(call, Display(f"$ rm{char} -rf x"), dangerous=True)
    out = buf.getvalue()
    code = format(ord(char), "04X")
    assert f"$ rm⟨U+{code}⟩ -rf x" in out
    assert char not in out


def test_confirm_marks_invisible_characters_in_diff(monkeypatch: pytest.MonkeyPatch) -> None:
    ui, buf = _console()
    monkeypatch.setattr(ui._c, "input", lambda prompt: "n")
    diff = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-ok = 1\n+# \u202e; rm -rf ~ \u202c\x01\n"
    ui.confirm(
        ToolCallInfo("edit_file", "x.py", ToolKind.EDIT),
        Display(diff, "diff", "x.py"),
        dangerous=False,
    )
    out = buf.getvalue()
    assert "+# ⟨U+202E⟩; rm -rf ~ ⟨U+202C⟩⟨U+0001⟩" in out  # и управляющие — не молча
    assert "\u202e" not in out
    assert "-ok = 1" in out


@pytest.mark.parametrize("dangerous", [True, False])
@pytest.mark.parametrize("command", ["rm -rf build", "make\nmake install"])
def test_confirm_shows_directory_command_runs_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    dangerous: bool,
    command: str,
) -> None:
    (tmp_path / "vendor" / "lib").mkdir(parents=True)
    ctx = ToolContext(workspace=Workspace(tmp_path))
    tool = RunShellTool()
    ui, buf = _console()
    monkeypatch.setattr(ui._c, "input", lambda prompt: "n")
    call = ToolCallInfo("run_shell", command, ToolKind.COMMAND)
    params = RunShellParams(command=command, cwd="vendor/lib/")
    ui.confirm(call, tool.preview(params, ctx), dangerous=dangerous)
    assert "▌ каталог  vendor/lib\n" in buf.getvalue()
    # в корне проекта — без строки о каталоге
    ui, buf = _console()
    monkeypatch.setattr(ui._c, "input", lambda prompt: "n")
    ui.confirm(call, tool.preview(RunShellParams(command=command), ctx), dangerous=dangerous)
    assert "каталог" not in buf.getvalue()


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("", False),  # Enter — «нет»: вопрос о доверии хукам и о платном замере
        ("x", False),
        ("н", False),  # н и т — клавиши y и n в русской раскладке: не «да»
        ("т", False),
        ("нет", False),
        ("y", True),
        ("Да", True),
        ("д", True),
    ],
)
def test_ask_defaults_to_no(
    monkeypatch: pytest.MonkeyPatch,
    *,
    reply: str,
    expected: bool,
) -> None:
    ui, _ = _console()
    monkeypatch.setattr(ui._c, "input", lambda prompt: reply)
    assert ui.ask("Разрешить хуки?", dangerous=True) is expected


@pytest.mark.parametrize("reply", ["н", "т", "нет", "N", ""])
def test_confirm_russian_layout_no_is_a_plain_refusal(
    reply: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ui, _ = _console()
    monkeypatch.setattr(ui._c, "input", lambda prompt: reply)
    outcome = ui.confirm(ToolCallInfo("run_shell", "ls", ToolKind.COMMAND), None, dangerous=False)
    assert outcome.approval is Approval.NO
    assert not outcome.feedback
    assert outcome.answered
