"""Терминальный UI gigacrew.

Обёртка над rich, изолирующая остальной код от деталей вывода. Задаёт единый
визуальный язык: брендовые цвета, панели, рендеринг вызовов инструментов,
потоковый вывод ответа модели с живым Markdown, индикаторы ожидания, диффы и
подтверждения.

Временная область внизу экрана (``rich.live.Live``) — одна на всю консоль: в ней
крутится индикатор ожидания модели или работы инструмента и рисуется ещё растущий
хвост ответа. Перед запуском новой области предыдущая всегда останавливается,
перед вопросом пользователю — тоже. Анимацию двигает собственный поток под общей
с печатью блокировкой: встроенное авто-обновление rich перерисовывает область
параллельно с печатью над ней, и при гонке стирает уже напечатанные строки.

В не-терминале (pipe, файл, «глупый» терминал) живой области нет: ответ модели
печатается как есть, по мере поступления — так его удобно перенаправлять в файл.
"""

from __future__ import annotations

import contextlib
import re
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager
from typing import IO

from rich.box import ROUNDED
from rich.console import Console as RichConsole
from rich.console import ConsoleOptions, Group, RenderResult
from rich.live import Live
from rich.panel import Panel
from rich.segment import Segment, SegmentLines
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from gigacrew.agent.chat_store import ChatInfo
from gigacrew.agent.conversation import request_text
from gigacrew.agent.events import (
    AgentEvents,
    Approval,
    CompactResult,
    ConfirmResult,
    NoticeLevel,
    Review,
    SubagentInfo,
    ToolCallInfo,
    TurnStats,
)
from gigacrew.agent.hooks import short_command
from gigacrew.llm.types import Message
from gigacrew.permissions import PermissionMode, ToolKind
from gigacrew.tools.ask_user import format_answer
from gigacrew.tools.base import Display, ToolResult
from gigacrew.tools.questions import Answer, Question, QuestionsUnavailable
from gigacrew.ui import banner as banner_art
from gigacrew.ui.format import (
    SKIPPED_MARK,
    clip_lines,
    format_tokens,
    format_when,
    mode_badge,
    plural,
)
from gigacrew.ui.markdown import Markdown
from gigacrew.ui.markdown_stream import MarkdownStream
from gigacrew.ui.spinner import INTERVAL as SPINNER_INTERVAL
from gigacrew.ui.spinner import rocket_frame
from gigacrew.ui.theme import (
    ACCENT,
    BRAND,
    DANGER,
    ICON_ARROW,
    ICON_BRAND,
    ICON_FAIL,
    ICON_OK,
    ICON_SUBSTEP,
    ICON_TOOL,
    MARKDOWN_STYLES,
    MUTED,
    OK,
    USER,
    WARN,
)

# Управляющие символы, которые нельзя пропускать в терминал из текста модели и
# вывода команд: ESC-последовательности могут перекрасить/стереть экран,
# подменить заголовок окна и т.п. Оставляем только \n и \t.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_SKIPPED_RE = re.compile(rf"^{SKIPPED_MARK} пропущено .*$", re.MULTILINE)
_REFRESH_PER_SECOND = 2 / SPINNER_INTERVAL  # два обновления на кадр — ровный шаг ракеты
_MODE_COLORS = {
    PermissionMode.AUTO: WARN,
    PermissionMode.PLAN: BRAND,
}


def sanitize(text: str) -> str:
    """Убирает управляющие символы (кроме переводов строк и табуляции)."""
    return _CONTROL_RE.sub("", text.replace("\r\n", "\n"))


# Невидимые символы: управляющие, смена направления текста, нулевой ширины. В команде,
# которую подтверждает пользователь, их нельзя убирать молча — она выполнится с ними.
_INVISIBLE_RE = re.compile(
    "[\x00-\x08\x0b-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff]"
)
# Многострочная команда длиннее — с номерами строк.
_NUMBER_LINES_FROM = 4


def visible_text(text: str, *, numbered: bool | None = None) -> Text:
    """Текст как есть, но с видимыми невидимыми символами (``⟨U+202E⟩``) и, у длинного
    многострочного, номерами строк. Ничего не обрезается: длинные строки переносятся."""
    lines = text.replace("\r\n", "\n").split("\n")
    if numbered is None:
        numbered = len(lines) >= _NUMBER_LINES_FROM
    width = len(str(len(lines)))
    out = Text(overflow="fold")
    for i, line in enumerate(lines, 1):
        if numbered:
            out.append(f"{i:>{width}} │ ", style=MUTED)
        pos = 0
        for m in _INVISIBLE_RE.finditer(line):
            out.append(line[pos : m.start()])
            out.append(f"⟨U+{ord(m.group()):04X}⟩", style=f"bold {DANGER}")
            pos = m.end()
        out.append(line[pos:])
        if i < len(lines):
            out.append("\n")
    return out


class _LiveView:
    """Содержимое временной области: хвост ответа (если есть) + строка индикатора.

    Вызывается из потока обновления ``Live``. Сама обрезает себя по высоте экрана:
    то, что ушло за верх экрана, временная область уже не сможет стереть.
    """

    def __init__(
        self,
        label: str | Callable[[], str],
        tail: Callable[[], list[list[Segment]]] | None = None,
        *,
        hint: str | Callable[[], str] = "Ctrl+C — прервать",
        started: float | None = None,
    ):
        self._label = label
        self._tail = tail
        self._hint = hint
        # Отсчёт времени — с ``started`` (например, с запуска суб-агента, а не с
        # последнего перезапуска области после вопроса).
        self._started = time.monotonic() if started is None else started

    def __rich_console__(self, console: RichConsole, options: ConsoleOptions) -> RenderResult:
        elapsed = time.monotonic() - self._started
        indicator = Text.assemble(
            rocket_frame(elapsed),
            (f"  {self._label() if callable(self._label) else self._label}… ", MUTED),
            (f"{int(elapsed)} с", MUTED),
            (f"  ·  {self._hint() if callable(self._hint) else self._hint}", MUTED),
        )
        lines = self._tail() if self._tail else []
        if lines:
            room = max(console.size.height - 3, 1)  # индикатор + отступ + запас
            yield SegmentLines(lines[-room:], new_lines=True)
            yield Text("")
        yield indicator


class _LiveArea:
    """Временная область + поток анимации; печать над ней — только через :meth:`print`."""

    def __init__(self, console: RichConsole, view: _LiveView, *, animate: bool):
        self._live = Live(
            view,
            console=console,
            auto_refresh=False,
            transient=True,
            redirect_stdout=False,
            redirect_stderr=False,
            vertical_overflow="crop",
        )
        self._lock = threading.RLock()
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._animate, daemon=True) if animate else None

    @property
    def active(self) -> bool:
        return not self._stopped.is_set()

    def start(self) -> None:
        self._live.start(refresh=True)
        if self._thread is not None:
            self._thread.start()

    def _animate(self) -> None:
        while not self._stopped.wait(1 / _REFRESH_PER_SECOND):
            with self._lock:
                if not self._stopped.is_set():
                    self._live.refresh()

    def print(self, renderable, **options) -> None:
        with self._lock:
            self._live.console.print(renderable, **options)

    def stop(self) -> None:
        self._stopped.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1)
        with self._lock:
            self._live.stop()


class Console(AgentEvents):
    """Терминальный интерфейс на rich; реализует события агента.

    ``force_terminal``/``width``/``auto_refresh`` нужны тестам: живую область
    можно проверить без настоящего терминала и без фонового потока обновления.
    """

    def __init__(
        self,
        *,
        no_color: bool = False,
        file: IO[str] | None = None,
        force_terminal: bool | None = None,
        width: int | None = None,
        auto_refresh: bool = True,
    ):
        self._c = RichConsole(
            no_color=no_color,
            highlight=False,
            emoji=False,
            file=file,
            force_terminal=force_terminal,
            width=width,
            theme=Theme(MARKDOWN_STYLES),
        )
        self._no_color = no_color
        self._live_ok = self._c.is_terminal and not self._c.is_dumb_terminal
        self._auto_refresh = auto_refresh
        self._live: _LiveArea | None = None
        self._md: MarkdownStream | None = None  # потоковый ответ (живой режим)
        self._raw_started = False  # потоковый ответ (не-терминал): метка уже напечатана
        self._interrupt_hint = "Ctrl+C — прервать"
        self._mode_hint: Callable[[], str] | None = None
        self._input_guard: Callable[[], AbstractContextManager[None]] = contextlib.nullcontext
        self._esc_stops_subagent = False
        # Работающий суб-агент: кто, с какого момента, чем занят, подводит ли итог.
        self._sub: SubagentInfo | None = None
        self._sub_started = 0.0
        self._sub_activity = ""
        self._sub_wrapping = False

    def set_interrupt_keys(
        self,
        hint: str,
        input_guard: Callable[[], AbstractContextManager[None]],
        *,
        stop_subagent: bool = False,
    ) -> None:
        """Подсказка о прерывании в индикаторе и защита ввода во время хода.

        ``input_guard`` оборачивает вопрос пользователю (подтверждение) — например,
        чтобы на это время отпустить терминал, который слушает клавишу Esc.
        ``stop_subagent`` — Esc во время суб-агента останавливает его (подсказка в
        индикаторе суб-агента).
        """
        self._interrupt_hint = hint
        self._input_guard = input_guard
        self._esc_stops_subagent = stop_subagent

    def set_mode_hint(self, hint: Callable[[], str] | None) -> None:
        """Режим разрешений в индикаторе хода; перечитывается при каждой отрисовке,
        поэтому смена режима во время хода (Shift+Tab) видна сразу."""
        self._mode_hint = hint

    def _turn_hint(self) -> str:
        if self._mode_hint is None:
            return self._interrupt_hint
        return f"{self._interrupt_hint}  ·  {self._mode_hint()}"

    def _subagent_hint(self) -> str:
        if not self._esc_stops_subagent:
            return self._turn_hint()
        keys = "Esc — оборвать суб-агента" if self._sub_wrapping else "Esc — завершить с итогом"
        hint = f"{keys}  ·  Ctrl+C — прервать всё"
        return hint if self._mode_hint is None else f"{hint}  ·  {self._mode_hint()}"

    @property
    def no_color(self) -> bool:
        return self._no_color

    def interactive(self) -> bool:
        """Можно ли спросить пользователя: ввод идёт с терминала (не pipe/файл)."""
        return _stdin_is_terminal()

    # ----------------------------- базовое ----------------------------- #
    def print(self, *args, **kwargs) -> None:
        self._c.print(*args, **kwargs)

    def blank(self) -> None:
        self._c.print()

    def rule(self, title: str = "") -> None:
        self._c.rule(Text(title, style=MUTED) if title else "", style=MUTED)

    # Сообщения печатаются над временной областью, если она есть (например,
    # уведомление суб-агента во время его работы).
    def system(self, text: str) -> None:
        self._print_above_live(Text(text, style=MUTED))

    def error(self, text: str) -> None:
        self._print_above_live(Text(f"{ICON_FAIL} {text}", style=f"bold {DANGER}"))

    def warn(self, text: str) -> None:
        self._print_above_live(Text(f"⚠ {text}", style=WARN))

    def info(self, text: str) -> None:
        self._print_above_live(Text(text, style=ACCENT))

    def success(self, text: str) -> None:
        self._print_above_live(Text(f"{ICON_OK} {text}", style=f"bold {OK}"))

    # ------------------------- временная область ------------------------ #
    def _start_live(self, view: _LiveView) -> None:
        self.stop_live()
        if not self._live_ok:
            return
        area = _LiveArea(self._c, view, animate=self._auto_refresh)
        area.start()
        self._live = area

    def _print_above_live(self, renderable, **options) -> None:
        if self._live is not None:
            self._live.print(renderable, **options)
        else:
            self._c.print(renderable, **options)

    def stop_live(self) -> None:
        """Убирает временную область (идемпотентно). Безопасно вызывать всегда."""
        live, self._live = self._live, None
        if live is not None:
            live.stop()

    @contextlib.contextmanager
    def progress(self, title: str, detail: Callable[[], str] | None = None) -> Iterator[None]:
        """Индикатор долгой операции вне хода агента (например, построения индекса).

        ``detail`` перечитывается при каждой отрисовке («1234 файлов»). Вне
        терминала печатается одна строка с ``title``.
        """
        if not self._live_ok:
            self.system(f"{title}…")
            yield
            return

        def label() -> str:
            return f"{title}: {detail()}" if detail is not None else title

        self._start_live(_LiveView(label, hint=self._interrupt_hint))
        try:
            yield
        finally:
            self.stop_live()

    # ------------------------------ баннер ------------------------------ #
    def banner(
        self,
        *,
        version: str,
        model: str,
        root: str,
        hints: Sequence[tuple[str, str]] = (),
        auto_approve: bool = False,
        mode: PermissionMode | None = None,
    ) -> None:
        """Приветственная панель; ``hints`` — пары (команда, краткое описание)."""
        rows = [
            banner_art.title(),
            Text(f"v{version}", style=MUTED),
            Text(""),
            Text.assemble(("модель ", MUTED), (model, f"bold {ACCENT}")),
            Text.assemble(("проект ", MUTED), root),
        ]
        if mode is not None:
            rows.append(
                Text.assemble(
                    ("режим  ", MUTED),
                    (mode_badge(mode), f"bold {_MODE_COLORS[mode]}"),
                    ("  Shift+Tab — сменить", MUTED),
                )
            )
        info = Group(*rows)
        body = Table.grid(padding=(0, 3))
        body.add_column(vertical="middle", no_wrap=True)
        body.add_column(vertical="middle")
        body.add_row(banner_art.rocket(), info)
        self._c.print(Panel(body, box=ROUNDED, border_style=BRAND, padding=(1, 2), expand=False))
        hint = Text("  ", style=MUTED)
        for i, (cmd, desc) in enumerate(hints):
            if i:
                hint.append("   ", style=MUTED)
            hint.append(cmd, style=ACCENT)
            hint.append(f" {desc}", style=MUTED)
        self._c.print(hint)
        if auto_approve:
            self.warn(
                "авто-подтверждение включено (-y): изменения применяются без вопросов, "
                "опасные команды — с вопросом (--yes-all — и они без вопроса)"
            )
        self._c.print()

    def help(
        self, commands: Sequence[tuple[str, str]], keys: Sequence[tuple[str, str]] = ()
    ) -> None:
        """Справка: таблица команд и клавиш."""
        table = Table.grid(padding=(0, 2))
        table.add_column(style=ACCENT, no_wrap=True)
        table.add_column(style=MUTED)
        for name, desc in commands:
            # Text, а не строка: «/resume [id]» и описания скиллов — не разметка Rich.
            table.add_row(Text(name), Text(desc))
        if keys:
            table.add_row("", "")
            for key, desc in keys:
                table.add_row(Text(key, style="bold"), desc)
        self._c.print(table)

    # --------------------------- сообщения LLM -------------------------- #
    def on_assistant_text(self, text: str) -> None:
        """Печать ответа модели целиком (не потоковый режим)."""
        if not text.strip():
            return
        self._assistant_label()
        self._c.print(Markdown(sanitize(text)))

    def _assistant_label(self) -> None:
        self._print_above_live(Text(f"{ICON_BRAND} gigacrew", style=f"bold {BRAND}"))

    # ----------------------- потоковый вывод LLM ------------------------ #
    def on_stream_start(self) -> None:
        """Запрос отправлен: индикатор ожидания до первого токена."""
        self._md = None
        self._raw_started = False
        self._start_live(_LiveView("думаю", hint=self._turn_hint))

    def on_stream_delta(self, text: str) -> None:
        """Очередной кусок текста модели."""
        text = sanitize(text)
        if not text:
            return
        if not self._live_ok:
            if not self._raw_started:
                self._raw_started = True
                self._assistant_label()
            self._c.file.write(text)
            self._c.file.flush()
            return
        if self._md is None:
            self._md = MarkdownStream(self._c)
            self._assistant_label()  # при активной области печатается над ней
            self._start_live(_LiveView("печатает", tail=self._md.tail_lines, hint=self._turn_hint))
        lines = self._md.feed(text)
        if lines:
            self._print_above_live(SegmentLines(lines, new_lines=True))

    def on_stream_end(self) -> None:
        """Ответ закончен (или прерван): убрать индикатор, дорисовать хвост."""
        md, self._md = self._md, None
        self.stop_live()
        if md is not None:
            lines = md.finish()
            if lines:
                self._c.print(SegmentLines(lines, new_lines=True))
        if self._raw_started:
            self._raw_started = False
            self._c.file.write("\n")
            self._c.file.flush()

    # --------------------------- инструменты ---------------------------- #
    def on_tool_call(self, call: ToolCallInfo) -> None:
        line = Text()
        line.append(f"{ICON_TOOL} ", style=f"bold {OK}")
        line.append(call.name, style="bold")
        if call.summary:
            line.append("  ", style=MUTED)
            line.append(sanitize(call.summary), style=ACCENT)
        self._c.print(line)

    def on_review_start(self, call: ToolCallInfo) -> None:
        self._start_live(_LiveView("рецензент проверяет команду", hint=self._turn_hint))

    def on_review_end(self, call: ToolCallInfo, review: Review | None) -> None:
        self.stop_live()
        if review is not None and review.allowed:
            line = Text("  ")
            line.append(f"{ICON_OK} ", style=OK)
            line.append(sanitize(review.approval), style=MUTED)
            self._c.print(line)

    def on_tool_start(self, call: ToolCallInfo) -> None:
        self._start_live(_LiveView("выполняется", hint=self._turn_hint))

    def on_tool_end(self, call: ToolCallInfo) -> None:
        self.stop_live()

    def on_tool_result(self, call: ToolCallInfo, result: ToolResult, *, previewed: bool) -> None:
        self.tool_result(result.summary or ("готово" if result.ok else "ошибка"), ok=result.ok)
        shown = result.display
        if not shown or not shown.text.strip():
            return
        if shown.kind == "diff":
            # дифф уже показан при подтверждении — не дублируем
            if not previewed:
                self.diff(shown.text, title=shown.title or None)
        else:
            self.output_block(clip_lines(shown.text), title=shown.title or "вывод")

    def tool_result(self, summary: str, ok: bool = True, *, indent: str = "  ") -> None:
        summary = sanitize(summary)
        line = Text(indent)
        if ok:
            line.append(f"{ICON_ARROW} ", style=OK)
            line.append(summary, style=MUTED)
        else:
            line.append(f"{ICON_ARROW} {ICON_FAIL} ", style=f"bold {DANGER}")
            line.append(summary, style=DANGER)
        self._print_above_live(line)

    def diff(self, diff_text: str, *, title: str | None = None) -> None:
        if not diff_text.strip():
            return
        syntax = Syntax(
            sanitize(diff_text).rstrip(),
            "diff",
            theme="ansi_dark",
            background_color="default",
            word_wrap=True,
        )
        self._print_above_live(
            Panel(
                syntax,
                title=Text(title or "изменения", style=ACCENT),
                title_align="left",
                box=ROUNDED,
                border_style=MUTED,
                padding=(0, 1),
                expand=False,
            )
        )

    def _exact_block(self, preview: Display) -> bool:
        """Превью того, что выполнится: целиком, без обрезки и молча удалённых символов.

        Возвращает True, если оно не помещается на экран (тогда предлагается просмотрщик).
        """
        text = preview.text.rstrip("\n")
        lines = text.split("\n")
        title = preview.title or "превью"
        if len(lines) >= _NUMBER_LINES_FROM:
            n = len(lines)
            title += f" · {n} {plural(n, 'строка', 'строки', 'строк')}"
        self._print_above_live(
            Panel(
                visible_text(text),
                title=Text(title, style=MUTED),
                title_align="left",
                box=ROUNDED,
                border_style=MUTED,
                padding=(0, 1),
                expand=False,
            )
        )
        width = max(self._c.size.width - 4, 20)
        height = sum(max(1, -(-len(line) // width)) for line in lines)
        return height > max(self._c.size.height - 8, 10)

    def _page(self, preview: Display) -> None:
        """Показать превью в просмотрщике (less и т.п.) — длинную команду целиком."""
        body = visible_text(preview.text.rstrip("\n"), numbered=True)
        with self._input_guard(), self._c.pager(styles=True):
            self._c.print(Text(preview.title or "превью", style=MUTED))
            self._c.print(body)

    def output_block(self, text: str, *, title: str = "вывод") -> None:
        if not text.strip():
            return
        body = Text(sanitize(text).rstrip())
        body.highlight_regex(_SKIPPED_RE, f"italic {MUTED}")
        self._print_above_live(
            Panel(
                body,
                title=Text(title, style=MUTED),
                title_align="left",
                box=ROUNDED,
                border_style=MUTED,
                padding=(0, 1),
                expand=False,
            )
        )

    # ---------------------------- суб-агенты ----------------------------- #
    def on_subagent_start(self, info: SubagentInfo) -> None:
        self._sub = info
        self._sub_started = time.monotonic()
        self._sub_activity = "думаю"
        self._sub_wrapping = False
        self._start_subagent_live()

    def _subagent_label(self) -> str:
        name = self._sub.name if self._sub is not None else "суб-агент"
        return f"{name} · {self._sub_activity}"

    def _start_subagent_live(self) -> None:
        if self._sub is None:
            return
        self._start_live(
            _LiveView(self._subagent_label, hint=self._subagent_hint, started=self._sub_started)
        )

    def on_subagent_activity(
        self, info: SubagentInfo, activity: str, *, wrapping_up: bool = False
    ) -> None:
        self._sub_activity = activity
        self._sub_wrapping = wrapping_up
        if self._live is None:  # область убрал вопрос пользователю — вернуть
            self._start_subagent_live()

    def on_subagent_tool_call(self, info: SubagentInfo, call: ToolCallInfo) -> None:
        line = Text(f"    {ICON_SUBSTEP} ", style=MUTED)
        line.append(call.name, style="bold")
        if call.summary:
            line.append("  ", style=MUTED)
            line.append(sanitize(call.summary), style=MUTED)
        self._print_above_live(line)

    def on_subagent_tool_result(
        self, info: SubagentInfo, call: ToolCallInfo, result: ToolResult, *, previewed: bool
    ) -> None:
        # Шаги суб-агента — коротко: только неудачи и правки, не показанные при
        # подтверждении (вывод команд и найденное читает сам суб-агент).
        if not result.ok:
            self.tool_result(result.summary or "ошибка", ok=False, indent="      ")
            return
        shown = result.display
        if shown and shown.kind == "diff" and shown.text.strip() and not previewed:
            self.diff(shown.text, title=shown.title or None)

    def on_subagent_end(self, info: SubagentInfo, stats: TurnStats | None) -> None:
        self.stop_live()  # до сброса: последняя отрисовка ещё читает метку
        self._sub = None
        self._sub_wrapping = False

    # ---------------------------- статистика ---------------------------- #
    def on_compact_start(self, *, auto: bool) -> None:
        title = "контекст почти заполнен — сжимаю историю" if auto else "сжимаю историю"
        if not self._live_ok:
            self.system(f"{title}…")
            return
        self._start_live(_LiveView(title, hint=self._turn_hint))

    def on_compact_end(self, result: CompactResult | None) -> None:
        self.stop_live()
        if result is None:
            return
        n = result.messages
        self.system(
            f"✂ контекст сжат: ~{format_tokens(result.before_tokens)} → "
            f"~{format_tokens(result.after_tokens)} ток. · в краткое содержание свёрнуто "
            f"{n} {plural(n, 'сообщение', 'сообщения', 'сообщений')}"
        )

    def on_hook_start(self, event: str, command: str) -> None:
        title = f"хук {event} · {sanitize(short_command(command))}"
        if not self._live_ok:
            self.system(f"{title}…")
            return
        self._start_live(_LiveView(title, hint=self._turn_hint))

    def on_hook_end(self) -> None:
        self.stop_live()

    def on_notice(self, text: str, *, level: NoticeLevel = "info") -> None:
        # В тексте может быть вывод хуков пользователя — escape-последовательности убираются.
        {"info": self.info, "warn": self.warn, "error": self.error}[level](sanitize(text))

    def on_turn_end(self, stats: TurnStats) -> None:
        steps = stats.steps
        parts = [f"{steps} {plural(steps, 'шаг', 'шага', 'шагов')}"]
        if stats.tool_calls:
            n = stats.tool_calls
            parts.append(f"{n} {plural(n, 'инструмент', 'инструмента', 'инструментов')}")
        if stats.duration_s:
            parts.append(f"{stats.duration_s:.1f} с")
        if stats.context_tokens:
            parts.append(f"контекст ~{format_tokens(stats.context_tokens)}")
        if stats.billed_tokens:
            parts.append(f"потрачено {format_tokens(stats.billed_tokens)} ток.")
        self._c.print(Text("  " + "  ·  ".join(parts), style=MUTED))

    # -------------------------- подтверждения --------------------------- #
    def confirm(
        self, call: ToolCallInfo, preview: Display | None, *, dangerous: bool
    ) -> ConfirmResult:
        self.stop_live()  # превью печатается до вопроса — область уже не нужна
        long_preview = False
        if preview and preview.text.strip():
            if preview.kind == "diff":
                self.diff(preview.text, title=preview.title or None)
            else:
                long_preview = self._exact_block(preview)
        if call.purpose:
            self._c.print(Text(f"  зачем (со слов модели): {sanitize(call.purpose)}", style=MUTED))
        if call.reason:
            self._c.print(Text(f"  почему спрашиваю: {sanitize(call.reason)}", style=MUTED))
        if call.review:
            self._c.print(Text(f"  {sanitize(call.review)}", style=MUTED))
        if call.history:
            self._c.print(Text(f"  {sanitize(call.history)}", style=MUTED))
        edit = call.kind is ToolKind.EDIT
        action = "Применить" if edit else "Выполнить"
        summary = sanitize(call.summary).strip()
        if "\n" in summary:  # многострочная команда целиком — в превью выше
            summary = summary.split("\n", 1)[0].rstrip() + " …"
        target = f" ({summary})" if summary else ""
        always = None
        if not dangerous:  # опасное «навсегда» не разрешается
            if call.remember:
                always = sanitize(call.remember)
            else:
                always = "не спрашивать этот вызов до конца сессии"
        question = f"{action} {call.name}{target}?"
        while True:
            raw = self._question_raw(
                question, dangerous=dangerous, always=always, feedback=True, view=long_preview
            )
            if raw is None:
                return ConfirmResult(Approval.NO, answered=False)
            answer = raw.strip().lower()
            if preview is not None and answer in _VIEW:
                self._page(preview)  # и спросить снова
                continue
            break
        if answer in _YES:
            return ConfirmResult(Approval.YES)
        if always is not None and answer in _ALWAYS:
            return ConfirmResult(Approval.ALWAYS)
        if not answer or answer in _NO:
            return ConfirmResult(Approval.NO)
        # Свой ответ вместо y/n — отказ с объяснением: его получит модель.
        return ConfirmResult(Approval.NO, feedback=raw.strip())

    def ask(self, question: str, *, dangerous: bool = False) -> bool:
        """Вопрос да/нет. Ctrl+C/Ctrl+D — «нет»."""
        return self._question(question, dangerous=dangerous) in _YES

    def _question(self, question: str, *, dangerous: bool, always: str | None = None) -> str:
        """Ответ на вопрос (в нижнем регистре); Ctrl+C/Ctrl+D — пустой."""
        raw = self._question_raw(question, dangerous=dangerous, always=always)
        return "" if raw is None else raw.strip().lower()

    def _question_raw(
        self,
        question: str,
        *,
        dangerous: bool,
        always: str | None = None,
        feedback: bool = False,
        view: bool = False,
    ) -> str | None:
        """Ответ как введён; None — ответа не было (Ctrl+C/Ctrl+D, конец ввода)."""
        self.stop_live()
        color = DANGER if dangerous else WARN
        if always is not None:
            self._c.print(Text(f"  a — {always}", style=MUTED))
        if view:
            self._c.print(Text("  v — открыть целиком в просмотрщике", style=MUTED))
        if feedback:
            self._c.print(Text("  или напишите, что сделать вместо этого", style=MUTED))
        prompt = Text("  ")
        prompt.append("⚠ ОПАСНО: " if dangerous else "? ", style=f"bold {color}")
        prompt.append(question, style="bold")
        keys = "y/N" + ("/a" if always is not None else "") + ("/v" if view else "")
        prompt.append(f" [{keys}] ", style=MUTED)
        try:
            with self._input_guard():
                answer = self._c.input(prompt)
        except (EOFError, KeyboardInterrupt):
            self._c.print()
            return None
        return answer

    # ---------------------------- вопросы агента ---------------------------- #
    def ask_user(self, questions: Sequence[Question]) -> list[Answer] | None:
        """Задаёт вопросы по очереди (меню со стрелками или ввод номера). None — отказ."""
        self.stop_live()
        if not _stdin_is_terminal():
            raise QuestionsUnavailable("ввод не с клавиатуры")
        answers: list[Answer] = []
        with self._input_guard():
            for i, question in enumerate(questions, 1):
                if question.body.strip():
                    self._question_body(question)
                answer = self._ask_one(question, i, len(questions))
                if answer is None:
                    self._c.print(Text("  ? пользователь отказался отвечать", style=MUTED))
                    return None
                line = Text("  ? ", style=f"bold {BRAND}")
                line.append(sanitize(question.text), style="bold")
                line.append(f" {ICON_ARROW} ", style=MUTED)
                line.append(sanitize(format_answer(answer)), style=ACCENT)
                self._c.print(line)
                answers.append(answer)
        return answers

    def _question_body(self, question: Question) -> None:
        """Подробности к вопросу (например, план на одобрение) — Markdown в рамке."""
        self._c.print(
            Panel(
                Markdown(sanitize(question.body)),
                title=Text(question.header or "подробности", style=f"bold {BRAND}"),
                title_align="left",
                box=ROUNDED,
                border_style=BRAND,
                padding=(0, 1),
            )
        )

    def _ask_one(self, question: Question, index: int, total: int) -> Answer | None:
        if self._live_ok:
            try:
                from gigacrew.ui.choice import choose

                return choose(question, index, total, no_color=self._no_color)
            except (KeyboardInterrupt, EOFError):
                return None
        return self._ask_plain(question, index, total)

    def _ask_plain(self, question: Question, index: int, total: int) -> Answer | None:
        """Запасной режим без интерактивного меню: номер(а) варианта или свой текст."""
        title = Text(f"Вопрос {index} из {total}", style=f"bold {BRAND}")
        if question.header:
            title.append(f" · {question.header}", style=ACCENT)
        self._c.print(title)
        self._c.print(Text(question.text, style="bold"))
        options = question.options
        for k, option in enumerate(options, 1):
            self._c.print(Text(f"  {k}. {option.label}"))
            if option.description:
                self._c.print(Text(f"     {option.description}", style=MUTED))
        custom = len(options) + 1
        self._c.print(Text(f"  {custom}. Свой ответ (или просто напишите текст)"))
        hint = "номера через пробел" if question.multi_select else "номер"
        while True:
            try:
                raw = self._c.input(Text(f"  {hint} или ответ: ", style=MUTED)).strip()
                if not raw:
                    continue
                numbers = raw.replace(",", " ").split()
                if not numbers:
                    continue  # одни запятые — спросить заново
                # isdecimal, а не isdigit: «²» — цифра, но int() её не разберёт
                if not all(n.isdecimal() and 1 <= int(n) <= custom for n in numbers):
                    return Answer(custom=raw)  # не номера — это свой ответ
                picked = sorted({int(n) for n in numbers})
                if len(picked) > 1 and not question.multi_select:
                    self.warn("здесь можно выбрать только один вариант")
                    continue
                text = ""
                if custom in picked:
                    text = self._c.input(Text("  ваш ответ: ", style=MUTED)).strip()
                    if not text:
                        continue
                labels = tuple(options[n - 1].label for n in picked if n != custom)
                return Answer(labels, text)
            except (EOFError, KeyboardInterrupt):
                self._c.print()
                return None

    # ---------------------------- сохранённые чаты ---------------------------- #
    def pick_chat(self, chats: Sequence[ChatInfo], current_id: str = "") -> ChatInfo | None:
        """Выбор чата: меню со стрелками и поиском, без терминала — номер. None — отмена."""
        self.stop_live()
        if not chats:
            return None
        if self._live_ok and _stdin_is_terminal():
            try:
                from gigacrew.ui.chat_picker import pick_chat

                return pick_chat(chats, current_id, no_color=self._no_color)
            except (KeyboardInterrupt, EOFError):
                return None
        return self._pick_chat_plain(chats, current_id)

    def _pick_chat_plain(self, chats: Sequence[ChatInfo], current_id: str) -> ChatInfo | None:
        table = Table.grid(padding=(0, 2))
        table.add_column(style=ACCENT, justify="right", no_wrap=True)
        table.add_column(style=MUTED, no_wrap=True)
        table.add_column()
        for k, chat in enumerate(chats, 1):
            n = chat.requests
            meta = (
                f"{format_when(chat.updated_at)} · {n} {plural(n, 'запрос', 'запроса', 'запросов')}"
            )
            title = Text(sanitize(chat.title))
            if chat.id == current_id:
                title.append(" (текущий)", style=MUTED)
            table.add_row(f"{k}.", meta, title)
        self._c.print(Text(f"Чаты проекта · {len(chats)}", style=f"bold {BRAND}"))
        self._c.print(table)
        while True:
            try:
                raw = self._c.input(Text("  номер чата (Enter — отмена): ", style=MUTED))
            except (EOFError, KeyboardInterrupt):
                self._c.print()
                return None
            raw = raw.strip()
            if not raw:
                return None
            if raw.isdecimal() and 1 <= int(raw) <= len(chats):
                return chats[int(raw) - 1]
            self.warn(f"введите номер от 1 до {len(chats)}")

    def chat_resumed(self, info: ChatInfo, messages: Sequence[Message]) -> None:
        """Сообщение о продолжении чата и последний обмен репликами."""
        n = info.requests
        line = Text("↺ продолжаем чат ", style=MUTED)
        line.append(f"«{sanitize(info.title)}»", style=f"bold {ACCENT}")
        line.append(
            f" · {n} {plural(n, 'запрос', 'запроса', 'запросов')} · {format_when(info.updated_at)}",
            style=MUTED,
        )
        self._c.print(line)
        last = next(
            (
                k
                for k in range(len(messages) - 1, -1, -1)
                if messages[k].is_request and messages[k].content.strip()
            ),
            None,
        )
        if last is not None:
            shown = clip_lines(request_text(messages[last].content), head=3, tail=2, max_chars=600)
            self._c.print(Text("› ", style=f"bold {USER}").append(sanitize(shown), style=USER))
            # Ответ ищется только после этого запроса: у прерванного хода его нет, и
            # ответ на предыдущий запрос здесь выглядел бы ответом на этот.
            answer = next(
                (
                    m.content
                    for m in reversed(messages[last + 1 :])
                    if m.role == "assistant" and not m.function_calls and m.content.strip()
                ),
                "",
            )
            if answer:
                self._assistant_label()
                self._c.print(Markdown(sanitize(answer)))
            else:
                self._c.print(Text("  (ответа нет — ход не был завершён)", style=MUTED))
        self._c.print()


_YES = ("y", "yes", "д", "да")
_NO = ("n", "no", "н", "нет", "т")  # н, т — клавиши y и n в русской раскладке: не «да»
_ALWAYS = ("a", "always", "в", "всегда", "ф")  # ф — клавиша a в русской раскладке
_VIEW = ("v", "view", "м", "показать")  # м — клавиша v в русской раскладке


def _stdin_is_terminal() -> bool:
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False
