"""Терминальный UI gigacrew.

Обёртка над rich, изолирующая остальной код от деталей вывода. Задаёт единый
визуальный язык (палитра — :mod:`gigacrew.ui.theme`): шаги агента строкой,
потоковый вывод ответа модели с живым Markdown, индикаторы ожидания, вопросы и
подтверждения.

Лента — это scrollback терминала: напечатанное в ней уже не изменить. Поэтому
подряд идущие вызовы, которые только смотрят проект, и подряд идущие правки
собираются в группу: пока она идёт, её шаги видны во временной области, а в ленту
уходит итог одной-двумя строками, когда группа закончилась (начался ответ модели,
другой вызов, вопрос пользователю, любое сообщение или конец хода). Диффы правок в
ленте не показываются — файл и число строк; дифф виден в вопросе о правке.
Всё, что ждёт ответа пользователя, отмечено полосой слева: жёлтой, у опасного —
красной.

Временная область внизу экрана (``rich.live.Live``) — одна на всю консоль: в ней
крутится индикатор ожидания модели или работы инструмента, идёт группа шагов и
рисуется ещё растущий хвост ответа. Перед запуском новой области предыдущая всегда
останавливается, перед вопросом пользователю — тоже. Анимацию двигает собственный
поток под общей с печатью блокировкой: встроенное авто-обновление rich
перерисовывает область параллельно с печатью над ней, и при гонке стирает уже
напечатанные строки.

В не-терминале (pipe, файл, «глупый» терминал) живой области нет: ответ модели
печатается как есть, по мере поступления — так его удобно перенаправлять в файл.
"""

from __future__ import annotations

import contextlib
import re
import sys
import threading
import time
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import IO

from rich.box import ROUNDED
from rich.console import Console as RichConsole
from rich.console import ConsoleOptions, Group, RenderableType, RenderResult
from rich.live import Live
from rich.padding import Padding
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
from gigacrew.llm.types import FunctionCall, Message
from gigacrew.permissions import PermissionMode, ToolKind
from gigacrew.tools.base import Display, ToolResult
from gigacrew.tools.questions import Answer, Question, QuestionsUnavailable
from gigacrew.ui import banner as banner_art
from gigacrew.ui.format import (
    chat_size,
    clip_lines,
    format_tokens,
    format_when,
    mode_badge,
    plural,
)
from gigacrew.ui.markdown import Markdown
from gigacrew.ui.markdown_stream import Lines, MarkdownStream
from gigacrew.ui.spinner import INTERVAL as SPINNER_INTERVAL
from gigacrew.ui.spinner import rocket_frame
from gigacrew.ui.theme import (
    ICON_BAR,
    ICON_BRAND,
    ICON_DANGER,
    ICON_FAIL,
    ICON_OK,
    ICON_PENDING,
    ICON_TOOL,
    MINT,
    Palette,
    markdown_styles,
    palette_for,
)

# Управляющие символы, которые нельзя пропускать в терминал из текста модели и
# вывода команд: ESC-последовательности могут перекрасить/стереть экран,
# подменить заголовок окна и т.п. Оставляем только \n и \t.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
# Строка-разделитель в выводе команды («===== FAILURES =====», «-----»): в хвост не идёт.
_SEPARATOR_RE = re.compile(r"^[=_\-*~#]{3,}.*[=_\-*~#]{3,}$|^[=_\-*~#]{3,}$")
_REFRESH_PER_SECOND = 2 / SPINNER_INTERVAL  # два обновления на кадр — ровный шаг ракеты
_TAIL_LINES = 3  # строк вывода упавшей команды
_LIVE_STEPS = 5  # шагов группы во временной области
_DETAIL_CHARS = 100  # итог шага (последняя строка вывода) — не длиннее
_BANNER_MIN_WIDTH = 76  # уже — стартовый экран без ракеты
_SHOW_SECONDS = 1.0  # время шага в ленте — от секунды
# Вызовы, итог которых уже показан пользователю их же диалогом.
_SELF_REPORTING = ("ask_user",)
# Обзорные инструменты в счётчиках группы; read_file — «прочитал», остальное — поиски.
_OVERVIEW_TOOLS = ("file_outline", "repo_map", "list_dir")
# Продолжение чата: сколько последних обменов показать и шагов (групп вызовов) в обмене.
RESUME_EXCHANGES = 3
RESUME_CALLS = 5


def sanitize(text: str) -> str:
    """Убирает управляющие символы (кроме переводов строк и табуляции)."""
    return _CONTROL_RE.sub("", text.replace("\r\n", "\n"))


class _SafeConsole(RichConsole):
    """Консоль Rich, которая не пропускает в терминал управляющие символы из текста.

    Rich (``Text``) убирает лишь BEL/BS/VT/FF/CR, а ESC и C1 оставляет. Очистка на
    месте каждой печати то и дело забывалась (имя инструмента от модели, заголовок
    превью, описание скилла), поэтому здесь — одна точка на весь вывод: из текста
    сегментов убирается всё, кроме ``\\n`` и ``\\t``. Свои последовательности Rich
    (цвет — в стилях, курсор живой области — в сегментах ``control``) не трогаются.
    Через неё же идёт и просмотрщик (``pager``). Старая консоль Windows печатает в
    обход неё, но ESC-последовательности она и не исполняет.
    """

    def _render_buffer(self, buffer: Iterable[Segment]) -> str:
        return super()._render_buffer(
            Segment(_CONTROL_RE.sub("", s.text), s.style)
            if s.control is None and _CONTROL_RE.search(s.text)
            else s
            for s in buffer
        )


# Невидимые символы: управляющие, смена направления текста, нулевой ширины и прочие
# невидимые. В команде или правке, которую подтверждает пользователь, их нельзя ни
# убирать молча, ни печатать как есть — выполнится или запишется текст с ними.
# Кандидаты — всё, кроме печатного ASCII, \n и \t; невидимые среди них — по категории
# Юникода (управляющие, форматирующие — в том числе смена направления, теги —,
# разделители строк и абзацев, половинки суррогатных пар) и по свойству
# Default_Ignorable_Code_Point (мягкий перенос, заполнители хангыля, селекторы вариантов).
_INVISIBLE_CANDIDATE_RE = re.compile(r"[^\t\n\x20-\x7e]")
_INVISIBLE_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp", "Cs"})
_IGNORABLE_RE = re.compile(
    "[­͏؜ᅟᅠ឴឵᠋-᠏​-‏‪-‮⁠-⁯ㅤ︀-️﻿ﾠ￰-￸\U0001bca0-\U0001bca3\U0001d173-\U0001d17a\U000e0000-\U000e0fff]"
)
_INVISIBLE_MARK_RE = re.compile(r"⟨U\+[0-9A-F]{4,6}⟩")
# Многострочная команда длиннее — с номерами строк.
_NUMBER_LINES_FROM = 4


def _invisible(text: str) -> Iterator[re.Match[str]]:
    """Невидимые символы строки по порядку."""
    for m in _INVISIBLE_CANDIDATE_RE.finditer(text):
        char = m.group()
        if unicodedata.category(char) in _INVISIBLE_CATEGORIES or _IGNORABLE_RE.match(char):
            yield m


def _mark(char: str) -> str:
    return f"⟨U+{ord(char):04X}⟩"


def mark_invisible(text: str) -> str:
    """Текст с видимыми невидимыми символами (``⟨U+202E⟩``); ``\\r\\n`` — как ``\\n``."""
    text = text.replace("\r\n", "\n")
    out, pos = [], 0
    for m in _invisible(text):
        out += [text[pos : m.start()], _mark(m.group())]
        pos = m.end()
    out.append(text[pos:])
    return "".join(out)


def visible_text(text: str, *, numbered: bool | None = None, palette: Palette = MINT) -> Text:
    """Текст как есть, но с видимыми невидимыми символами (``⟨U+202E⟩``) и, у длинного
    многострочного, номерами строк. Ничего не обрезается: длинные строки переносятся."""
    lines = text.replace("\r\n", "\n").split("\n")
    if numbered is None:
        numbered = len(lines) >= _NUMBER_LINES_FROM
    width = len(str(len(lines)))
    out = Text(overflow="fold")
    for i, line in enumerate(lines, 1):
        if numbered:
            out.append(f"{i:>{width}} │ ", style=palette.dim)
        pos = 0
        for m in _invisible(line):
            out.append(line[pos : m.start()])
            out.append(_mark(m.group()), style=f"bold {palette.danger}")
            pos = m.end()
        out.append(line[pos:])
        if i < len(lines):
            out.append("\n")
    return out


class _LiveView:
    """Содержимое временной области: идущая группа шагов или хвост ответа + индикатор.

    Вызывается из потока обновления ``Live``. Сама обрезает себя по высоте экрана:
    то, что ушло за верх экрана, временная область уже не сможет стереть.
    """

    def __init__(
        self,
        label: str | Callable[[], str],
        tail: Callable[[], Lines] | None = None,
        *,
        hint: str | Callable[[], str] = "Ctrl+C — прервать",
        started: float | None = None,
        above: Callable[[], list[Text]] | None = None,
        palette: Palette = MINT,
    ):
        self._label = label
        self._tail = tail
        self._hint = hint
        self._above = above
        self._p = palette
        # Отсчёт времени — с ``started`` (например, с запуска суб-агента, а не с
        # последнего перезапуска области после вопроса).
        self._started = time.monotonic() if started is None else started

    def __rich_console__(self, console: RichConsole, options: ConsoleOptions) -> RenderResult:
        p = self._p
        elapsed = time.monotonic() - self._started
        label = self._label() if callable(self._label) else self._label
        indicator = Text.assemble(
            rocket_frame(elapsed, p),
            (f"  {label.removesuffix('…')}… ", p.muted),
            (f"{int(elapsed)} с", p.muted),
            (f"  ·  {self._hint() if callable(self._hint) else self._hint}", p.dim),
        )
        room = max(console.size.height - 3, 1)  # индикатор + отступ + запас
        steps = self._above() if self._above else []
        if steps:
            yield from steps[-room:]
            yield Text("")
            room = max(room - len(steps) - 1, 1)
        lines = self._tail() if self._tail else []
        if lines:
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


@dataclass
class _Step:
    """Вызов инструмента в ленте: строка печатается, когда известен результат."""

    call: ToolCallInfo
    result: ToolResult | None = None
    started: float = 0.0  # начало выполнения (on_tool_start)
    seconds: float = 0.0
    review: Review | None = None  # рецензент разрешил — строка под шагом


@dataclass
class _Group:
    """Подряд идущие вызовы одного вида: ``read`` — только смотрят проект, ``edit`` — правки."""

    kind: str
    steps: list[_Step] = field(default_factory=list)


def _group_kind(call: ToolCallInfo) -> str:
    if call.read_only:
        return "read"
    if call.kind is ToolKind.EDIT:
        return "edit"
    return ""


def _target(call: ToolCallInfo) -> str:
    """Цель вызова одной строкой: путь, шаблон, первая строка команды (+ «…»)."""
    summary = sanitize(call.summary).strip()
    first = summary.split("\n", 1)[0].rstrip()
    return f"{first} …" if "\n" in summary else first


def _diff_stat(result: ToolResult) -> tuple[int, int] | None:
    """Добавлено и удалено строк по диффу правки; None — диффа нет."""
    shown = result.display
    if shown is None or shown.kind != "diff":
        return None
    added = removed = 0
    for line in shown.text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed


def _clip(text: str, limit: int = _DETAIL_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _read_counts(calls: Sequence[ToolCallInfo]) -> str:
    """«прочитал 2 файла, 3 поиска, 1 обзор» — по вызовам группы."""
    reads = sum(1 for c in calls if c.name == "read_file")
    overviews = sum(1 for c in calls if c.name in _OVERVIEW_TOOLS)
    searches = len(calls) - reads - overviews
    parts = []
    if reads:
        parts.append(f"прочитал {reads} {plural(reads, 'файл', 'файла', 'файлов')}")
    if searches:
        parts.append(f"{searches} {plural(searches, 'поиск', 'поиска', 'поисков')}")
    if overviews:
        parts.append(f"{overviews} {plural(overviews, 'обзор', 'обзора', 'обзоров')}")
    return ", ".join(parts)


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
        self._c = _SafeConsole(
            no_color=no_color,
            highlight=False,
            emoji=False,
            file=file,
            force_terminal=force_terminal,
            width=width,
        )
        # Точные цвета или, в терминале на 16 цветов, цвета ANSI его темы.
        self._p = palette_for(self._c.color_system)
        self._c.push_theme(Theme(markdown_styles(self._p)))
        self._no_color = no_color
        self._live_ok = self._c.is_terminal and not self._c.is_dumb_terminal
        self._auto_refresh = auto_refresh
        self._live: _LiveArea | None = None
        self._md: MarkdownStream | None = None  # потоковый ответ (живой режим)
        self._raw_started = False  # потоковый ответ (не-терминал): знак уже напечатан
        self._answer_open = False  # у текущего ответа уже напечатан знак ✦
        self._blank = True  # последняя напечатанная строка — пустая
        self._gap = False  # после ответа модели — пустая строка перед следующим
        self._interrupt_hint = "Ctrl+C — прервать"
        self._mode_hint: Callable[[], str] | None = None
        self._input_guard: Callable[[], AbstractContextManager[None]] = contextlib.nullcontext
        self._esc_stops_subagent = False
        # Шаги: идущая группа и вызов вне группы, ждущий результата.
        self._group: _Group | None = None
        self._step: _Step | None = None
        # Работающий суб-агент: кто, с какого момента, чем занят, подводит ли итог, шаги.
        self._sub: SubagentInfo | None = None
        self._sub_started = 0.0
        self._sub_activity = ""
        self._sub_wrapping = False
        self._sub_steps: list[_Step] = []

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
        self._flush_group()
        self._gap = False
        self._c.print(*args, **kwargs)
        self._blank = not args

    def blank(self) -> None:
        self.print()

    def rule(self, title: str = "") -> None:
        self._flush_group()
        self._c.rule(Text(title, style=self._p.muted) if title else "", style=self._p.dim)
        self._blank = False

    # Сообщения печатаются над временной областью, если она есть (например,
    # уведомление суб-агента во время его работы).
    def system(self, text: str) -> None:
        self._emit(Text(text, style=self._p.muted))

    def error(self, text: str) -> None:
        self._emit(Text(f"{ICON_FAIL} {text}", style=f"bold {self._p.danger}"))

    def warn(self, text: str) -> None:
        self._emit(Text(f"{ICON_DANGER} {text}", style=self._p.warn))

    def info(self, text: str) -> None:
        self._emit(Text(text, style=self._p.muted))

    def success(self, text: str) -> None:
        self._emit(Text(f"{ICON_OK} {text}", style=f"bold {self._p.ok}"))

    # ------------------------------ вывод ------------------------------ #
    def _out(self, renderable: RenderableType = "", **options) -> None:
        """Печать над временной областью (если она есть)."""
        if self._live is not None:
            self._live.print(renderable, **options)
        else:
            self._c.print(renderable, **options)
        self._blank = isinstance(renderable, str) and not renderable

    def _gap_line(self) -> None:
        """Пустая строка после ответа модели — перед первым, что печатается за ним."""
        if self._gap:
            self._gap = False
            if not self._blank:
                self._out()

    def _emit(self, renderable: RenderableType, **options) -> None:
        """Строка ленты: сначала — итог идущей группы шагов и отступ после ответа."""
        self._flush_group()
        self._gap_line()
        self._out(renderable, **options)

    def _fit(self, text: Text) -> Text:
        """Строка по ширине экрана: не переносится, а обрезается многоточием."""
        text.truncate(max(self._c.width - 1, 20), overflow="ellipsis")
        return text

    # ------------------------- временная область ------------------------ #
    def _view(
        self,
        label: str | Callable[[], str],
        tail: Callable[[], Lines] | None = None,
        *,
        hint: str | Callable[[], str] | None = None,
        started: float | None = None,
    ) -> _LiveView:
        return _LiveView(
            label,
            tail,
            hint=self._turn_hint if hint is None else hint,
            started=started,
            above=self._live_steps,
            palette=self._p,
        )

    def _start_live(self, view: _LiveView) -> None:
        self.stop_live()
        if not self._live_ok:
            return
        area = _LiveArea(self._c, view, animate=self._auto_refresh)
        area.start()
        self._live = area

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

        self._start_live(self._view(label, hint=self._interrupt_hint))
        try:
            yield
        finally:
            self.stop_live()

    # ------------------------------ баннер ------------------------------ #
    def _mode_color(self, mode: PermissionMode) -> str:
        return self._p.warn if mode is PermissionMode.AUTO else self._p.brand

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
        p = self._p
        rows = [
            banner_art.title(p),
            Text(f"v{version}", style=p.dim),
            Text(""),
            Text.assemble(("модель  ", p.muted), (model, f"bold {p.fg}")),
            Text.assemble(("проект  ", p.muted), (root, p.fg)),
        ]
        if mode is not None:
            rows.append(
                Text.assemble(
                    ("режим   ", p.muted),
                    (mode_badge(mode), f"bold {self._mode_color(mode)}"),
                    ("   Shift+Tab — сменить", p.dim),
                )
            )
        info = Group(*rows)
        if self._c.width < _BANNER_MIN_WIDTH:
            body: RenderableType = info
        else:
            grid = Table.grid(padding=(0, 3))
            grid.add_column(vertical="middle", no_wrap=True)
            grid.add_column(vertical="middle")
            grid.add_row(banner_art.rocket(p, mono=self._c.no_color), info)
            body = grid
        self._c.print(Panel(body, box=ROUNDED, border_style=p.dim, padding=(1, 2), expand=False))
        hint = Text("  ")
        for i, (cmd, desc) in enumerate(hints):
            if i:
                hint.append("   ")
            hint.append(cmd, style=p.muted)
            hint.append(f" {desc}", style=p.dim)
        self._c.print(hint)
        if auto_approve:
            self.warn(
                "авто-подтверждение включено (-y): изменения применяются без вопросов, "
                "опасные команды — с вопросом (--yes-all — и они без вопроса)"
            )
        self.print()

    def help(
        self, commands: Sequence[tuple[str, str]], keys: Sequence[tuple[str, str]] = ()
    ) -> None:
        """Справка: таблица команд и клавиш."""
        table = Table.grid(padding=(0, 2))
        table.add_column(style=self._p.fg, no_wrap=True)
        table.add_column(style=self._p.muted)
        for name, desc in commands:
            # Text, а не строка: «/resume [id]» и описания скиллов — не разметка Rich.
            table.add_row(Text(name), Text(desc))
        if keys:
            table.add_row("", "")
            for key, desc in keys:
                table.add_row(Text(key, style="bold"), desc)
        self.print(table)

    # --------------------------- ответ модели --------------------------- #
    def _answer_start(self) -> None:
        """Начало ответа модели: итог группы шагов и пустая строка перед ответом."""
        self._flush_group()
        self._gap = False
        if not self._blank:
            self._out()
        self._answer_open = False

    def _gutter(self, lines: Lines, *, commit: bool = True) -> Lines:
        """Строки ответа: знак ✦ слева у первой строки ответа, отступ у остальных."""
        mark = Segment(f"{ICON_BRAND} ", self._c.get_style(f"bold {self._p.brand}"))
        pad = Segment("  ")
        first = not self._answer_open
        out = []
        for line in lines:
            out.append([mark if first else pad, *line])
            first = False
        if commit and lines:
            self._answer_open = True
        return out

    def _answer_width(self) -> int:
        return max(self._c.width - 2, 20)

    def on_assistant_text(self, text: str) -> None:
        """Печать ответа модели целиком (не потоковый режим)."""
        if not text.strip():
            return
        self._answer_start()
        options = self._c.options.update_width(self._answer_width())
        lines = self._c.render_lines(Markdown(sanitize(text)), options, pad=False)
        self._out(SegmentLines(self._gutter(lines), new_lines=True))
        self._gap = True

    # ----------------------- потоковый вывод LLM ------------------------ #
    def on_stream_start(self) -> None:
        """Запрос отправлен: индикатор ожидания до первого токена."""
        self._md = None
        self._raw_started = False
        self._start_live(self._view("думаю"))

    def _answer_tail(self) -> Lines:
        md = self._md
        return self._gutter(md.tail_lines(), commit=False) if md is not None else []

    def on_stream_delta(self, text: str) -> None:
        """Очередной кусок текста модели."""
        text = sanitize(text)
        if not text:
            return
        if not self._live_ok:
            if not self._raw_started:
                self._raw_started = True
                self._answer_start()
                self._c.file.write(f"{ICON_BRAND} ")
                self._blank = False
            self._c.file.write(text)
            self._c.file.flush()
            return
        if self._md is None:
            self._answer_start()
            self._md = MarkdownStream(self._c, width=self._answer_width())
            self._start_live(self._view("печатает", tail=self._answer_tail))
        lines = self._md.feed(text)
        if lines:
            self._out(SegmentLines(self._gutter(lines), new_lines=True))

    def on_stream_end(self) -> None:
        """Ответ закончен (или прерван): убрать индикатор, дорисовать хвост."""
        md, self._md = self._md, None
        self.stop_live()
        if md is not None:
            lines = md.finish()
            if lines:
                self._out(SegmentLines(self._gutter(lines), new_lines=True))
            self._gap = True
        if self._raw_started:
            self._raw_started = False
            self._c.file.write("\n")
            self._c.file.flush()
            self._gap = True

    # --------------------------- инструменты ---------------------------- #
    def _find(self, call: ToolCallInfo, steps: Sequence[_Step] = ()) -> _Step | None:
        """Шаг вызова: тот же объект или (для копии с подробностями вопроса) последний
        шаг без результата с тем же именем и описанием."""
        candidates = list(steps)
        if not candidates:
            if self._group is not None:
                candidates += self._group.steps
            if self._step is not None:
                candidates.append(self._step)
        for step in reversed(candidates):
            if step.call is call:
                return step
        for step in reversed(candidates):
            if step.result is None and (step.call.name, step.call.summary) == (
                call.name,
                call.summary,
            ):
                return step
        return None

    def _in_group(self, step: _Step) -> bool:
        return self._group is not None and any(s is step for s in self._group.steps)

    def on_tool_call(self, call: ToolCallInfo) -> None:
        kind = _group_kind(call)
        if self._group is not None and self._group.kind != kind:
            self._flush_group()
        step = _Step(call)
        if kind:
            if self._group is None:
                self._group = _Group(kind)
            self._group.steps.append(step)
        else:
            self._step = step

    def on_review_start(self, call: ToolCallInfo) -> None:
        self._start_live(self._view("рецензент проверяет команду"))

    def on_review_end(self, call: ToolCallInfo, review: Review | None) -> None:
        self.stop_live()
        if review is None or not review.allowed:
            return  # отказ или сбой — дальше вопрос пользователю с подробностями
        step = self._find(call)
        if step is not None:
            step.review = review  # строкой под шагом, когда он выполнится
        else:
            self._emit(Text(f"  {sanitize(review.approval)}", style=self._p.dim))

    def on_tool_start(self, call: ToolCallInfo) -> None:
        step = self._find(call)
        if step is not None:
            step.started = time.monotonic()
        target = _clip(_target(call), 50)
        if call.kind is ToolKind.COMMAND and target:
            label = f"$ {target}"
        else:
            label = f"{call.name} · {target}" if target else call.name
        self._start_live(self._view(label))

    def on_tool_end(self, call: ToolCallInfo) -> None:
        step = self._find(call)
        if step is not None and step.started:
            step.seconds = time.monotonic() - step.started
        self.stop_live()

    def on_tool_result(self, call: ToolCallInfo, result: ToolResult, *, previewed: bool) -> None:
        step = self._find(call)
        if step is None:  # результат без on_tool_call — строкой сразу
            step = _Step(call)
        step.result = result
        if self._in_group(step):
            return  # итог группы — когда она закончится
        if step is self._step:
            self._step = None
        if call.name in _SELF_REPORTING and result.ok:
            return
        for line in self._step_lines(step):
            self._emit(line)
        if self._sub_steps:
            for line in self._subagent_lines(self._sub_steps):
                self._emit(line)
            self._sub_steps = []

    # -------- строки шагов -------- #
    def _stat(self, added: int, removed: int) -> Text:
        return Text.assemble((f"+{added}", self._p.ok), " ", (f"−{removed}", self._p.danger))

    def _detail(self, call: ToolCallInfo, result: ToolResult) -> str:
        """Итог удачного шага: у команды — последняя строка вывода (итог тестов, сборки),
        у прочего — размер вывода, код выхода или итог без цели."""
        shown = result.display
        if shown is not None and shown.kind == "text":
            lines = [x.strip() for x in shown.text.splitlines() if x.strip()]
            if lines and call.kind is ToolKind.COMMAND:
                return _clip(sanitize(lines[-1]))
            if lines:
                n = len(lines)
                return f"вывод: {n} {plural(n, 'строка', 'строки', 'строк')}"
        if result.exit_code is not None:
            return f"код {result.exit_code}"
        summary = sanitize(result.summary or "готово")
        target = _target(call)
        if target and target in summary:  # «прочитан a.py (12 строк)» — путь уже в шаге
            summary = summary.replace(target, " ").strip(" :")
            summary = re.sub(r"\s{2,}", " ", summary)
        return _clip(summary or "готово")

    def _failure(self, result: ToolResult) -> str:
        if result.exit_code is not None:
            return f"код {result.exit_code}"
        return sanitize(result.summary or "ошибка")

    def _edit_line(self, call: ToolCallInfo, result: ToolResult) -> Text:
        p = self._p
        path = _target(call)
        if not result.ok:
            return Text.assemble(
                (f"{ICON_TOOL} ", p.danger),
                (f"{call.name} {path}".strip(), f"bold {p.fg}"),
                (f"  {ICON_FAIL} {sanitize(result.summary or 'ошибка')}", f"bold {p.danger}"),
            )
        verb = "Создал" if result.summary.startswith("создан") else "Изменил"
        line = Text.assemble((f"{ICON_TOOL} ", p.ok), (f"{verb} {path}", f"bold {p.fg}"))
        stat = _diff_stat(result)
        if stat is not None:
            line.append("  ")
            line.append_text(self._stat(*stat))
        return line

    def _step_lines(self, step: _Step) -> list[Text]:
        """Шаг одной строкой: ● что сделано  ✔ итог · время; под ней — рецензент и,
        если шаг упал, последние строки вывода."""
        p = self._p
        call, result = step.call, step.result
        assert result is not None
        if call.kind is ToolKind.EDIT:
            lines = [self._edit_line(call, result)]
        else:
            target = _target(call)
            head = f"$ {target}" if call.kind is ToolKind.COMMAND else f"{call.name} {target}"
            status = Text()
            if result.ok:
                status.append(f"{ICON_OK} ", style=p.ok)
                status.append(self._detail(call, result), style=p.muted)
            else:
                status.append(f"{ICON_FAIL} {self._failure(result)}", style=f"bold {p.danger}")
            if step.seconds >= _SHOW_SECONDS:
                status.append(f"  ·  {step.seconds:.1f} с", style=p.dim)
            # Не помещается в строку — короче команда, а не итог.
            room = self._c.width - 1 - len(status) - 4
            head = head.strip()
            if len(head) > room:
                head = _clip(head, max(room, 24))
            line = Text.assemble(
                (f"{ICON_TOOL} ", p.ok if result.ok else p.danger),
                (head, f"bold {p.fg}"),
                ("  ", ""),
            )
            lines = [self._fit(line + status)]
        if step.review is not None:
            lines.append(Text(f"  {sanitize(step.review.approval)}", style=p.dim))
        if not result.ok and result.display is not None and result.display.kind == "text":
            output = [x for x in sanitize(result.display.text).splitlines() if x.strip()]
            output = [x for x in output if not _SEPARATOR_RE.match(x.strip())]
            for out_line in output[-_TAIL_LINES:]:
                lines.append(self._fit(Text.assemble(("  │ ", p.danger), (out_line, ""))))
        return lines

    def _failed_line(self, step: _Step) -> Text:
        """Неудачный шаг внутри группы: ✘ что — почему."""
        p = self._p
        assert step.result is not None
        return Text.assemble(
            (f"  {ICON_FAIL} ", f"bold {p.danger}"),
            (f"{step.call.name} {_target(step.call)}".strip(), p.danger),
            ("  ", ""),
            (sanitize(step.result.summary or "ошибка"), p.danger),
        )

    def _read_lines(self, steps: Sequence[_Step]) -> list[Text]:
        """Итог группы чтений: «Изучил код» со счётчиками, цели, ошибки."""
        p = self._p
        if len(steps) == 1:
            return self._step_lines(steps[0])
        ok = [s for s in steps if s.result is not None and s.result.ok]
        failed = [s for s in steps if s.result is not None and not s.result.ok]
        head = Text.assemble(
            (f"{ICON_TOOL} ", p.ok if ok else p.danger), ("Изучил код", f"bold {p.fg}")
        )
        rest = _read_counts([s.call for s in ok])
        seconds = sum(s.seconds for s in steps)
        if seconds >= _SHOW_SECONDS:
            rest = f"{rest}  ·  {seconds:.1f} с" if rest else f"{seconds:.1f} с"
        if rest:
            head.append(f"  {rest}", style=p.muted)
        lines = [head]
        targets = list(dict.fromkeys(_target(s.call) for s in ok if _target(s.call)))
        if targets:
            lines.append(self._fit(Text("  " + "  ".join(targets), style=p.muted)))
        lines += [self._failed_line(s) for s in failed]
        return lines

    def _edit_lines(self, steps: Sequence[_Step]) -> list[Text]:
        """Итог группы правок: файл и +N −M; несколько файлов — строка на файл."""
        p = self._p
        ok = [s for s in steps if s.result is not None and s.result.ok]
        failed = [s for s in steps if s.result is not None and not s.result.ok]
        files: dict[str, list[int]] = {}
        for s in ok:
            stat = _diff_stat(s.result) or (0, 0)  # type: ignore[arg-type]
            total = files.setdefault(_target(s.call), [0, 0])
            total[0] += stat[0]
            total[1] += stat[1]
        if len(files) <= 1:
            # Один файл (в том числе несколько правок подряд) — как одна правка.
            lines = []
            if ok:
                last = ok[-1]
                line = self._edit_line(last.call, last.result)  # type: ignore[arg-type]
                if len(ok) > 1:
                    line = Text.assemble(
                        (f"{ICON_TOOL} ", p.ok), (f"Изменил {_target(last.call)}", f"bold {p.fg}")
                    )
                    added, removed = next(iter(files.values()))
                    line.append("  ")
                    line.append_text(self._stat(added, removed))
                lines.append(line)
            if not ok:  # все правки упали — каждая своей строкой
                return [line for s in failed for line in self._step_lines(s)]
            return lines + [self._failed_line(s) for s in failed]
        n = len(files)
        head = Text.assemble(
            (f"{ICON_TOOL} ", p.ok),
            (f"Изменил {n} {plural(n, 'файл', 'файла', 'файлов')}", f"bold {p.fg}"),
            ("  ", ""),
        )
        head.append_text(
            self._stat(sum(a for a, _ in files.values()), sum(r for _, r in files.values()))
        )
        lines = [head]
        width = max(len(path) for path in files)
        for path, (added, removed) in files.items():
            line = Text(f"  {path:<{width}}  ", style=p.fg)
            line.append_text(self._stat(added, removed))
            lines.append(line)
        return lines + [self._failed_line(s) for s in failed]

    def _flush_group(self) -> None:
        """Итог идущей группы — в ленту (идемпотентно)."""
        group, self._group = self._group, None
        if group is None:
            return
        steps = [s for s in group.steps if s.result is not None]
        if not steps:
            return
        lines = self._read_lines(steps) if group.kind == "read" else self._edit_lines(steps)
        self._gap_line()
        for line in lines:
            self._out(line)

    def _live_rows(self, steps: Sequence[_Step]) -> list[Text]:
        """Последние шаги для временной области: ✔/✘/◌, инструмент, цель, итог."""
        p = self._p
        rows = []
        hidden = len(steps) - _LIVE_STEPS
        if hidden > 0:
            rows.append(Text(f"  … ещё {hidden}", style=p.dim))
            steps = steps[-_LIVE_STEPS:]
        for s in steps:
            result = s.result
            if result is None:
                mark = (f"  {ICON_PENDING} ", p.warn)
            elif result.ok:
                mark = (f"  {ICON_OK} ", p.ok)
            else:
                mark = (f"  {ICON_FAIL} ", p.danger)
            row = Text.assemble(mark, (s.call.name, p.muted), ("  ", ""), (_target(s.call), p.fg))
            if result is not None:
                row.append("  ")
                stat = _diff_stat(result) if result.ok else None
                if stat is not None:
                    row.append_text(self._stat(*stat))
                else:
                    row.append(
                        sanitize(result.summary or ""), style=p.dim if result.ok else p.danger
                    )
            rows.append(self._fit(row))
        return rows

    def _live_steps(self) -> list[Text]:
        """Что показать над индикатором: идущую группу или шаги суб-агента."""
        group = self._group
        if group is not None and group.steps:
            p = self._p
            steps = list(group.steps)
            n = len(steps)
            title = "Изучаю код" if group.kind == "read" else "Правлю файлы"
            head = Text.assemble(
                (f"{ICON_TOOL} ", p.warn),
                (title, f"bold {p.fg}"),
                (f"  {n} {plural(n, 'вызов', 'вызова', 'вызовов')}", p.muted),
            )
            return [head, *self._live_rows(steps)]
        if self._sub is not None and self._sub_steps:
            return self._live_rows(list(self._sub_steps))
        return []

    # ---------------------------- суб-агенты ----------------------------- #
    def on_subagent_start(self, info: SubagentInfo) -> None:
        self._flush_group()
        self._sub = info
        self._sub_started = time.monotonic()
        self._sub_activity = "думаю"
        self._sub_wrapping = False
        self._sub_steps = []
        self._start_subagent_live()

    def _subagent_label(self) -> str:
        name = self._sub.name if self._sub is not None else "суб-агент"
        return f"{name} · {self._sub_activity}"

    def _start_subagent_live(self) -> None:
        if self._sub is None:
            return
        self._start_live(
            self._view(self._subagent_label, hint=self._subagent_hint, started=self._sub_started)
        )

    def on_subagent_activity(
        self, info: SubagentInfo, activity: str, *, wrapping_up: bool = False
    ) -> None:
        self._sub_activity = activity
        self._sub_wrapping = wrapping_up
        if self._live is None:  # область убрал вопрос пользователю — вернуть
            self._start_subagent_live()

    def on_subagent_tool_call(self, info: SubagentInfo, call: ToolCallInfo) -> None:
        self._sub_steps.append(_Step(call))

    def on_subagent_tool_result(
        self, info: SubagentInfo, call: ToolCallInfo, result: ToolResult, *, previewed: bool
    ) -> None:
        step = self._find(call, self._sub_steps) if self._sub_steps else None
        if step is None:
            step = _Step(call)
            self._sub_steps.append(step)
        step.result = result

    def on_subagent_end(self, info: SubagentInfo, stats: TurnStats | None) -> None:
        self.stop_live()  # до сброса: последняя отрисовка ещё читает метку
        self._sub = None
        self._sub_wrapping = False

    def _subagent_lines(self, steps: Sequence[_Step]) -> list[Text]:
        """Что сделал суб-агент — под строкой его вызова: счётчики, правки, неудачи."""
        p = self._p
        done = [s for s in steps if s.result is not None]
        ok = [s for s in done if s.result.ok]  # type: ignore[union-attr]
        parts = []
        reads = [s for s in ok if s.call.read_only]
        if reads:
            parts.append(_read_counts([s.call for s in reads]))
        edits: dict[str, list[int]] = {}
        for s in ok:
            if s.call.kind is ToolKind.EDIT:
                stat = _diff_stat(s.result) or (0, 0)  # type: ignore[arg-type]
                total = edits.setdefault(_target(s.call), [0, 0])
                total[0] += stat[0]
                total[1] += stat[1]
        if edits:
            if len(edits) <= 3:
                shown = ", ".join(f"{path} +{a} −{r}" for path, (a, r) in edits.items())
                parts.append(f"изменил {shown}")
            else:
                n = len(edits)
                parts.append(f"изменил {n} {plural(n, 'файл', 'файла', 'файлов')}")
        commands = sum(1 for s in ok if s.call.kind is ToolKind.COMMAND)
        if commands:
            parts.append(f"{commands} {plural(commands, 'команда', 'команды', 'команд')}")
        lines = []
        if parts:
            lines.append(Text("  " + ", ".join(parts), style=p.muted))
        lines += [self._failed_line(s) for s in done if not s.result.ok]  # type: ignore[union-attr]
        return lines

    # ---------------------------- статистика ---------------------------- #
    def on_compact_start(self, *, auto: bool) -> None:
        title = "контекст почти заполнен — сжимаю историю" if auto else "сжимаю историю"
        if not self._live_ok:
            self.system(f"{title}…")
            return
        self._start_live(self._view(title))

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
        self._start_live(self._view(title))

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
        self._emit(Text("  " + "  ·  ".join(parts), style=self._p.dim))

    # -------------------------- подтверждения --------------------------- #
    def _bar(self, *, dangerous: bool = False) -> tuple[str, str]:
        return (f"  {ICON_BAR} ", self._p.danger if dangerous else self._p.warn)

    def _confirm_head(self, call: ToolCallInfo, preview: Display | None, bar: tuple[str, str]):
        """Что спрашивается — первой строкой вопроса. Однострочное превью команды
        показывается здесь же, как выполнится (с видимыми невидимыми символами)."""
        p = self._p
        text = preview.text.rstrip("\n") if preview is not None else ""
        if preview is not None and preview.kind != "diff" and text.strip() and "\n" not in text:
            head = visible_text(text, numbered=False, palette=p)
            head.stylize("bold")
            self._out(Text.assemble(bar) + head)
            return False
        target = _target(call)
        if call.kind is ToolKind.COMMAND:
            head_text = f"$ {target}"
        else:
            head_text = f"{call.name} {target}".strip()
        self._out(Text.assemble(bar, (head_text, f"bold {p.fg}")))
        if not text.strip():
            return False
        if preview is not None and preview.kind == "diff":
            self._out(Padding(self._diff_panel(preview), (0, 0, 0, 4)))
            return False
        return self._exact_block(preview, border=bar[1])  # type: ignore[arg-type]

    def confirm(
        self, call: ToolCallInfo, preview: Display | None, *, dangerous: bool
    ) -> ConfirmResult:
        step = self._find(call)
        if step is not None and self._group is not None and self._in_group(step):
            # Спрошенный вызов — не часть группы: его строка — после ответа.
            self._group.steps = [s for s in self._group.steps if s is not step]
            self._step = step
        self._flush_group()
        self.stop_live()  # превью печатается до вопроса — область уже не нужна
        self._gap_line()
        p = self._p
        bar = self._bar(dangerous=dangerous)
        long_preview = self._confirm_head(call, preview, bar)
        labeled = []
        where = _command_dir(call, preview)
        if where:  # где выполнится команда — тоже часть того, что подтверждается
            labeled.append(("каталог", p.muted, Text(where, style=f"bold {p.fg}")))
        if call.reason:
            label = "почему опасно" if dangerous else "почему спрашиваю"
            style = f"bold {p.danger}" if dangerous else p.muted
            labeled.append((label, style, Text(sanitize(call.reason), style=p.fg)))
        if call.purpose:
            purpose = Text.assemble((sanitize(call.purpose), p.muted), ("  со слов модели", p.dim))
            labeled.append(("зачем", p.muted, purpose))
        width = max((len(label) for label, _, _ in labeled), default=0)
        for label, style, text in labeled:
            self._out(Text.assemble(bar, (f"{label:<{width}}  ", style)) + text)
        if call.review:
            self._out(Text.assemble(bar, (sanitize(call.review), p.muted)))
        if call.history:
            self._out(Text.assemble(bar, (sanitize(call.history), p.muted)))
        self._out(Text.assemble(bar))
        if call.kind is ToolKind.EDIT:
            question = "Применить правку?"
        elif call.kind is ToolKind.COMMAND:
            question = "Выполнить опасную команду?" if dangerous else "Выполнить команду?"
        else:
            question = f"Выполнить {call.name}?"
        always = None
        if not dangerous:  # опасное «навсегда» не разрешается
            if call.remember:
                always = sanitize(call.remember)
            else:
                always = "не спрашивать этот вызов до конца сессии"
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

    def _diff_panel(self, preview: Display) -> Panel:
        # Невидимые символы правки — видимыми (как в команде), с подсветкой диффа.
        code = mark_invisible(preview.text).rstrip()
        body = Syntax(code, "diff", theme="ansi_dark", background_color="default").highlight(code)
        body.rstrip()
        body.no_wrap, body.overflow = False, "fold"  # длинные строки — переносом
        body.highlight_regex(_INVISIBLE_MARK_RE, f"bold {self._p.danger}")
        return Panel(
            body,
            title=Text(preview.title or "изменения", style=self._p.fg),
            title_align="left",
            box=ROUNDED,
            border_style=self._p.dim,
            padding=(0, 1),
            expand=False,
        )

    def _exact_block(self, preview: Display, *, border: str) -> bool:
        """Превью того, что выполнится: целиком, без обрезки и молча удалённых символов.

        Возвращает True, если оно не помещается на экран (тогда предлагается просмотрщик).
        """
        p = self._p
        text = preview.text.rstrip("\n")
        lines = text.split("\n")
        title = preview.title or "превью"
        if len(lines) >= _NUMBER_LINES_FROM:
            n = len(lines)
            title += f" · {n} {plural(n, 'строка', 'строки', 'строк')}"
        panel = Panel(
            visible_text(text, palette=p),
            title=Text(title, style=p.muted),
            title_align="left",
            box=ROUNDED,
            border_style=border,
            padding=(0, 1),
            expand=False,
        )
        self._out(Padding(panel, (0, 0, 0, 4)))
        width = max(self._c.size.width - 8, 20)
        height = sum(max(1, -(-len(line) // width)) for line in lines)
        return height > max(self._c.size.height - 8, 10)

    def _page(self, preview: Display) -> None:
        """Показать превью в просмотрщике (less и т.п.) — длинную команду целиком."""
        body = visible_text(preview.text.rstrip("\n"), numbered=True, palette=self._p)
        with self._input_guard(), self._c.pager(styles=True):
            self._c.print(Text(preview.title or "превью", style=self._p.muted))
            self._c.print(body)

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
        self._flush_group()
        self.stop_live()
        p = self._p
        bar = self._bar(dangerous=dangerous)
        notes = []
        if always is not None:
            notes.append(f"a — {always}")
        elif dangerous and feedback:
            notes.append("запомнить нельзя")
        if view:
            notes.append("v — открыть целиком в просмотрщике")
        if feedback:
            notes.append("или напишите, что сделать вместо этого")
        joined = " · ".join(notes)
        if len(joined) + len(bar[0]) < self._c.width:
            notes = [joined] if notes else []
        for note in notes:  # не помещаются в строку — по строке на подсказку
            self._out(Text.assemble(bar, (note, p.dim)))
        mark = f"{ICON_DANGER} " if dangerous else "? "
        keys = "y/N" + ("/a" if always is not None else "") + ("/v" if view else "")
        prompt = Text.assemble(
            bar,
            (mark, f"bold {bar[1]}"),
            (question, f"bold {p.fg}"),
            (f" [{keys}] ", p.muted),
        )
        try:
            with self._input_guard():
                answer = self._c.input(prompt)
        except (EOFError, KeyboardInterrupt):
            self._c.print()
            return None
        finally:
            self._blank = False
        return answer

    # ---------------------------- вопросы агента ---------------------------- #
    def ask_user(self, questions: Sequence[Question]) -> list[Answer] | None:
        """Задаёт вопросы по очереди (меню со стрелками или ввод номера). None — отказ."""
        self._flush_group()
        self.stop_live()
        if not _stdin_is_terminal():
            raise QuestionsUnavailable("ввод не с клавиатуры")
        p = self._p
        answers: list[Answer] = []
        with self._input_guard():
            for i, question in enumerate(questions, 1):
                if question.body.strip():
                    self._question_body(question)
                answer = self._ask_one(question, i, len(questions))
                if answer is None:
                    self._emit(
                        Text.assemble(
                            (f"{ICON_TOOL} ", p.dim),
                            (sanitize(question.text), p.muted),
                            ("  пользователь отказался отвечать", p.dim),
                        )
                    )
                    return None
                self._emit(self._answered(question, answer))
                answers.append(answer)
        return answers

    def _answered(self, question: Question, answer: Answer) -> Text:
        """Строка ленты после ответа: вопрос серым, ответ ярким."""
        p = self._p
        line = Text.assemble(
            (f"{ICON_TOOL} ", p.ok), (sanitize(question.text), p.muted), ("  ", "")
        )
        for i, label in enumerate(answer.selected):
            if i:
                line.append(", ", style=p.muted)
            line.append(sanitize(label), style=f"bold {p.fg}")
        if answer.custom:
            if answer.selected:
                line.append(", ", style=p.muted)
            label = (question.custom_label or "Свой ответ…").rstrip("…").lower()
            line.append(f"{label} ", style=p.muted)
            line.append(f"«{sanitize(answer.custom)}»", style=f"bold {p.fg}")
        if not answer.selected and not answer.custom:
            line.append("(без ответа)", style=p.muted)
        return line

    def _question_body(self, question: Question) -> None:
        """Подробности к вопросу (например, план на одобрение) — Markdown в рамке."""
        p = self._p
        self._emit(
            Panel(
                Markdown(sanitize(question.body)),
                title=Text(f" {question.header or 'подробности'} ", style=f"bold {p.brand}"),
                title_align="left",
                subtitle=Text(f" {sanitize(question.source)} ", style=p.dim)
                if question.source
                else None,
                subtitle_align="right",
                box=ROUNDED,
                border_style=p.dim,
                padding=(1, 2),
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
        p = self._p
        bar = self._bar()
        head = Text.assemble(bar, ("? ", f"bold {p.warn}"), (question.text, f"bold {p.fg}"))
        tag = question.header
        if total > 1:
            tag = f"{tag} · {index} из {total}" if tag else f"{index} из {total}"
        if tag:
            head.append(f"   {tag}", style=p.muted)
        self._out(head)
        options = question.options
        for k, option in enumerate(options, 1):
            self._out(Text.assemble(bar, (f"  {k}. ", p.muted), (option.label, p.fg)))
            if option.description:
                self._out(Text.assemble(bar, (f"     {option.description}", p.dim)))
        custom = len(options) + 1
        label = (question.custom_label or "Свой ответ…").rstrip("…")
        self._out(
            Text.assemble(
                bar, (f"  {custom}. ", p.muted), (f"{label} (или просто напишите текст)", p.fg)
            )
        )
        hint = "номера через пробел" if question.multi_select else "номер"
        while True:
            try:
                prompt = Text.assemble(bar, (f"{hint} или ответ: ", p.muted))
                raw = self._c.input(prompt).strip()
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
                    text = self._c.input(Text.assemble(bar, ("ваш ответ: ", p.muted))).strip()
                    if not text:
                        continue
                labels = tuple(options[n - 1].label for n in picked if n != custom)
                return Answer(labels, text)
            except (EOFError, KeyboardInterrupt):
                self._c.print()
                return None
            finally:
                self._blank = False

    # ---------------------------- сохранённые чаты ---------------------------- #
    def pick_chat(self, chats: Sequence[ChatInfo], current_id: str = "") -> ChatInfo | None:
        """Выбор чата: меню со стрелками и поиском, без терминала — номер. None — отмена."""
        self._flush_group()
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
        p = self._p
        table = Table.grid(padding=(0, 2))
        table.add_column(style=p.muted, justify="right", no_wrap=True)
        table.add_column(style=p.dim, no_wrap=True)
        table.add_column()
        for k, chat in enumerate(chats, 1):
            meta = f"{format_when(chat.updated_at)} · {chat_size(chat.requests, chat.messages)}"
            title = Text(sanitize(chat.title), style=p.fg)
            if chat.id == current_id:
                title.append(" (текущий)", style=p.muted)
            table.add_row(f"{k}.", meta, title)
        self.print(Text(f"Чаты проекта · {len(chats)}", style=f"bold {p.brand}"))
        self.print(table)
        while True:
            try:
                raw = self._c.input(Text("  номер чата (Enter — отмена): ", style=p.muted))
            except (EOFError, KeyboardInterrupt):
                self._c.print()
                return None
            raw = raw.strip()
            if not raw:
                return None
            if raw.isdecimal() and 1 <= int(raw) <= len(chats):
                return chats[int(raw) - 1]
            self.warn(f"введите номер от 1 до {len(chats)}")

    def chat_resumed(
        self,
        info: ChatInfo,
        messages: Sequence[Message],
        describe: Callable[[FunctionCall], ToolCallInfo] | None = None,
    ) -> None:
        """Сообщение о продолжении чата и последние обмены репликами.

        Обмен — запрос пользователя и всё до следующего: шаги (вызовы инструментов,
        ``describe`` — как они выглядели во время хода) и итоговый ответ. Шаги — как в
        ленте, но приглушены и без итогов: результатов в истории нет. Более ранние
        обмены — одной строкой со счётчиками: они сохранены в чате, но не печатаются.
        """
        p = self._p
        line = Text("↺ продолжаем чат ", style=p.muted)
        line.append(f"«{sanitize(info.title)}»", style=f"bold {p.fg}")
        line.append(
            f" · {chat_size(info.requests, info.messages)} · {format_when(info.updated_at)}",
            style=p.muted,
        )
        self.print(line)
        starts = _exchange_starts(messages)[-RESUME_EXCHANGES:]
        if starts and starts[0] > 0:
            earlier = messages[: starts[0]]
            size = chat_size(sum(1 for m in earlier if m.is_request), len(earlier))
            self.print(
                Text(f"  … раньше: {size} — сохранены в чате, на экран не выводятся", style=p.dim)
            )
        for k, start in enumerate(starts):
            end = starts[k + 1] if k + 1 < len(starts) else len(messages)
            if k:
                self.print()
            self._resumed_exchange(messages[start:end], describe, last=k == len(starts) - 1)
        self.print()

    def _resumed_exchange(
        self,
        messages: Sequence[Message],
        describe: Callable[[FunctionCall], ToolCallInfo] | None,
        *,
        last: bool,
    ) -> None:
        """Обмен продолженного чата (``messages[0]`` — запрос); ответ ранних — сокращён."""
        p = self._p
        shown = clip_lines(request_text(messages[0].content), head=3, tail=2, max_chars=600)
        self.print(Text.assemble(("❯ ", f"bold {p.brand}"), (sanitize(shown), p.fg)))
        calls = [
            describe(call) if describe is not None else ToolCallInfo(call.name)
            for m in messages
            if m.role == "assistant"
            for call in m.function_calls
        ]
        for line in self._history_lines(calls):
            self.print(line)
        # Ответ ищется только в этом обмене: у прерванного хода его нет, и ответ на
        # предыдущий запрос здесь выглядел бы ответом на этот.
        answer = next(
            (
                m.content
                for m in reversed(messages)
                if m.role == "assistant" and not m.function_calls and m.content.strip()
            ),
            "",
        )
        if not answer:
            self.print(Text("  (ответа нет — ход не был завершён)", style=p.muted))
            return
        if not last:
            answer = clip_lines(answer, head=6, tail=2, max_chars=1200)
        self.on_assistant_text(answer)
        self._gap = False

    def _history_lines(self, calls: Sequence[ToolCallInfo]) -> list[Text]:
        """Шаги из истории: подряд идущие чтения и правки — одной строкой, как в ленте;
        не больше :data:`RESUME_CALLS` строк, ранние — счётчиком."""
        p = self._p
        entries: list[list[ToolCallInfo]] = []
        for call in calls:
            kind = _group_kind(call)
            if kind and entries and _group_kind(entries[-1][0]) == kind:
                entries[-1].append(call)
            else:
                entries.append([call])
        lines = []
        if len(entries) > RESUME_CALLS:
            hidden = sum(len(entry) for entry in entries[:-RESUME_CALLS])
            word = plural(hidden, "вызов", "вызова", "вызовов")
            lines.append(Text(f"  … ещё {hidden} {word} инструментов", style=p.dim))
            entries = entries[-RESUME_CALLS:]
        for entry in entries:
            first = entry[0]
            kind = _group_kind(first)
            targets = list(dict.fromkeys(t for call in entry if (t := _target(call))))
            detail = ""
            if kind == "read" and len(entry) > 1:
                head = f"Изучил код  {_read_counts(entry)}"
                detail = "  ".join(targets)
            elif kind == "edit" and targets:
                n = len(targets)
                files = ", ".join(targets)
                head = f"Изменил {files}" if n <= 3 else f"Изменил {n} файлов: {files}"
            elif first.kind is ToolKind.COMMAND and targets:
                head = f"$ {targets[0]}"
            else:
                head = f"{first.name} {_target(first)}".strip()
            lines.append(self._fit(Text.assemble((f"{ICON_TOOL} ", p.dim), (head, p.muted))))
            if detail:
                lines.append(self._fit(Text(f"  {detail}", style=p.dim)))
        return lines


def _command_dir(call: ToolCallInfo, preview: Display | None) -> str:
    """Каталог, где выполнится команда, если не корень проекта: превью команды
    подписано «команда · <каталог>». Пусто — в корне (или это не команда)."""
    if call.kind is not ToolKind.COMMAND or preview is None:
        return ""
    head, sep, where = preview.title.partition(" · ")
    return where.strip() if sep and head == "команда" else ""


def _exchange_starts(messages: Sequence[Message]) -> list[int]:
    """Индексы запросов пользователя с текстом — начала обменов (служебные — не запросы)."""
    return [k for k, m in enumerate(messages) if m.is_request and m.content.strip()]


_YES = ("y", "yes", "д", "да")
_NO = ("n", "no", "н", "нет", "т")  # н, т — клавиши y и n в русской раскладке: не «да»
_ALWAYS = ("a", "always", "в", "всегда", "ф")  # ф — клавиша a в русской раскладке
_VIEW = ("v", "view", "м", "показать")  # м — клавиша v в русской раскладке


def _stdin_is_terminal() -> bool:
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False
