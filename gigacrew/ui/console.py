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

import functools
import re
import sys
import threading
import time
import unicodedata
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import IO, Any, override

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

from gigacrew.agent import events
from gigacrew.agent.chat_store import ChatInfo
from gigacrew.agent.conversation import request_text
from gigacrew.agent.hooks import short_command
from gigacrew.llm.types import FunctionCall, Message
from gigacrew.permissions import PermissionMode, ToolKind
from gigacrew.tools.base import Display, ToolResult
from gigacrew.tools.questions import Answer, Question, QuestionsUnavailableError
from gigacrew.ui import banner as banner_art
from gigacrew.ui import theme
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
_LABEL_CHARS = 50  # цель вызова в индикаторе — не длиннее
_MIN_WIDTH = 20  # строка ленты, ответ модели и превью — не уже, как бы ни был узок экран
_BANNER_MIN_WIDTH = 76  # уже — стартовый экран без ракеты
_SHOW_SECONDS = 1.0  # время шага в ленте — от секунды
_SPACER = "  "  # промежуток между частями строки и отступ под ней
_COMMA = ", "
# Вызовы, итог которых уже показан пользователю их же диалогом.
_SELF_REPORTING = ("ask_user",)
# Обзорные инструменты в счётчиках группы; read_file — «прочитал», остальное — поиски.
_OVERVIEW_TOOLS = ("file_outline", "repo_map", "list_dir")
# Продолжение чата: сколько последних обменов показать и шагов (групп вызовов) в обмене.
RESUME_EXCHANGES = 3
RESUME_CALLS = 5
_RESUME_REQUEST_CHARS = 600  # запрос продолженного чата — не длиннее
_RESUME_ANSWER_CHARS = 1200  # ответ раннего обмена продолженного чата — не длиннее


def sanitize(text: str) -> str:
    """Убирает управляющие символы (кроме переводов строк и табуляции)."""
    return _CONTROL_RE.sub("", text.replace("\r\n", "\n"))


class _SafeConsole(RichConsole):
    r"""Консоль Rich, которая не пропускает в терминал управляющие символы из текста.

    Rich (``Text``) убирает лишь BEL/BS/VT/FF/CR, а ESC и C1 оставляет. Очистка на
    месте каждой печати то и дело забывалась (имя инструмента от модели, заголовок
    превью, описание скилла), поэтому здесь — одна точка на весь вывод: из текста
    сегментов убирается всё, кроме ``\\n`` и ``\\t``. Свои последовательности Rich
    (цвет — в стилях, курсор живой области — в сегментах ``control``) не трогаются.
    Через неё же идёт и просмотрщик (``pager``). Старая консоль Windows печатает в
    обход неё, но ESC-последовательности она и не исполняет.
    """

    @override
    def _render_buffer(self, buffer: Iterable[Segment]) -> str:
        return super()._render_buffer(
            Segment(_CONTROL_RE.sub("", segment.text), segment.style)
            if segment.control is None and _CONTROL_RE.search(segment.text)
            else segment
            for segment in buffer
        )


# Невидимые символы: управляющие, смена направления текста, нулевой ширины и прочие
# невидимые. В команде или правке, которую подтверждает пользователь, их нельзя ни
# убирать молча, ни печатать как есть — выполнится или запишется текст с ними.
# Кандидаты — всё, кроме печатного ASCII, \n и \t; невидимые среди них — по категории
# Юникода (управляющие, форматирующие — в том числе смена направления, теги —,
# разделители строк и абзацев, половинки суррогатных пар) и по свойству
# Default_Ignorable_Code_Point (мягкий перенос, заполнители хангыля, селекторы вариантов).
_INVISIBLE_CANDIDATE_RE = re.compile(r"[^\t\n\x20-\x7e]")
_INVISIBLE_CATEGORIES = frozenset(("Cc", "Cf", "Zl", "Zp", "Cs"))
_IGNORABLE_RE = re.compile(
    r"[\u00ad\u034f\u061c\u115f\u1160\u17b4\u17b5\u180b-\u180f\u200b-\u200f\u202a-\u202e\u2060-\u206f\u3164\ufe00-\ufe0f\ufeff\uffa0\ufff0-\ufff8\U0001bca0-\U0001bca3\U0001d173-\U0001d17a\U000e0000-\U000e0fff]",
)
_INVISIBLE_MARK_RE = re.compile(r"⟨U\+[0-9A-F]{4,6}⟩")
# Многострочная команда длиннее — с номерами строк.
_NUMBER_LINES_FROM = 4


def _invisible(text: str) -> Iterator[re.Match[str]]:
    """Невидимые символы строки по порядку."""
    for match in _INVISIBLE_CANDIDATE_RE.finditer(text):
        char = match.group()
        if unicodedata.category(char) in _INVISIBLE_CATEGORIES or _IGNORABLE_RE.match(char):
            yield match


def _mark(char: str) -> str:
    hex_code = format(ord(char), "04X")
    return f"⟨U+{hex_code}⟩"


def mark_invisible(text: str) -> str:
    r"""Текст с видимыми невидимыми символами (``⟨U+202E⟩``); ``\\r\\n`` — как ``\\n``."""
    text = text.replace("\r\n", "\n")
    out, pos = [], 0
    for match in _invisible(text):
        out += [text[pos : match.start()], _mark(match.group())]
        pos = match.end()
    out.append(text[pos:])
    return "".join(out)


def visible_text(
    text: str,
    *,
    numbered: bool | None = None,
    palette: theme.Palette = theme.MINT,
) -> Text:
    """Текст как есть, но с видимыми невидимыми символами (``⟨U+202E⟩``).

    У длинного многострочного текста — ещё и номера строк. Ничего не обрезается:
    длинные строки переносятся.
    """
    lines = text.replace("\r\n", "\n").split("\n")
    if numbered is None:
        numbered = len(lines) >= _NUMBER_LINES_FROM
    width = len(str(len(lines)))
    out = Text(overflow="fold")
    for number, line in enumerate(lines, 1):
        if numbered:
            prefix = str(number).rjust(width)
            out.append(f"{prefix} │ ", style=palette.dim)
        pos = 0
        for match in _invisible(line):
            out.append(line[pos : match.start()])
            out.append(_mark(match.group()), style=f"bold {palette.danger}")
            pos = match.end()
        out.append(line[pos:])
        if number < len(lines):
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
        palette: theme.Palette = theme.MINT,
    ):
        self._label = label
        self._tail = tail
        self._hint = hint
        self._above = above
        self._palette = palette
        # Отсчёт времени — с ``started`` (например, с запуска суб-агента, а не с
        # последнего перезапуска области после вопроса).
        self._started = time.monotonic() if started is None else started

    # Протокол отрисовки rich: метод с этим именем вызывает сама rich.
    def __rich_console__(self, console: RichConsole, options: ConsoleOptions) -> RenderResult:  # noqa: PLW3201
        palette = self._palette
        elapsed = time.monotonic() - self._started
        label = self._label() if callable(self._label) else self._label
        hint = self._hint() if callable(self._hint) else self._hint
        indicator = Text.assemble(
            rocket_frame(elapsed, palette),
            (f"  {label.removesuffix('…')}… ", palette.muted),
            (f"{int(elapsed)} с", palette.muted),
            (f"  ·  {hint}", palette.dim),
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

    def print(self, renderable: RenderableType, **options: Any) -> None:
        with self._lock:
            self._live.console.print(renderable, **options)

    def stop(self) -> None:
        self._stopped.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1)
        with self._lock:
            self._live.stop()

    def _animate(self) -> None:
        while not self._stopped.wait(1 / _REFRESH_PER_SECOND):
            with self._lock:
                if not self._stopped.is_set():
                    self._live.refresh()


@dataclass
class _Step:
    """Вызов инструмента в ленте: строка печатается, когда известен результат."""

    call: events.ToolCallInfo
    outcome: ToolResult | None = None
    started: float = 0  # начало выполнения (on_tool_start)
    seconds: float = 0
    review: events.Review | None = None  # рецензент разрешил — строка под шагом


@dataclass
class _Group:
    """Подряд идущие вызовы одного вида: ``read`` — только смотрят проект, ``edit`` — правки."""

    kind: str
    steps: list[_Step] = field(default_factory=list)


def _group_kind(call: events.ToolCallInfo) -> str:
    if call.read_only:
        return "read"
    if call.kind is ToolKind.EDIT:
        return "edit"
    return ""


def _target(call: events.ToolCallInfo) -> str:
    """Цель вызова одной строкой: путь, шаблон, первая строка команды (+ «…»)."""
    summary = sanitize(call.summary).strip()
    first = summary.split("\n", 1)[0].rstrip()
    return f"{first} …" if "\n" in summary else first


def _diff_stat(tool_result: ToolResult) -> tuple[int, int] | None:
    """Добавлено и удалено строк по диффу правки; None — диффа нет."""
    shown = tool_result.display
    if shown is None or shown.kind != "diff":
        return None
    added = 0
    removed = 0
    for line in shown.text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed


def _clip(text: str, limit: int = _DETAIL_CHARS) -> str:
    if len(text) <= limit:
        return text
    clipped = text[: limit - 1]
    return f"{clipped}…"


def _counted(count: int, one: str, few: str, many: str) -> str:
    """Число со словом в нужной форме: «3 файла»."""
    word = plural(count, one, few, many)
    return f"{count} {word}"


def _read_counts(calls: Sequence[events.ToolCallInfo]) -> str:
    """«прочитал 2 файла, 3 поиска, 1 обзор» — по вызовам группы."""
    reads = sum(1 for call in calls if call.name == "read_file")
    overviews = sum(1 for call in calls if call.name in _OVERVIEW_TOOLS)
    searches = len(calls) - reads - overviews
    parts = []
    if reads:
        files = _counted(reads, "файл", "файла", "файлов")
        parts.append(f"прочитал {files}")
    if searches:
        parts.append(_counted(searches, "поиск", "поиска", "поисков"))
    if overviews:
        parts.append(_counted(overviews, "обзор", "обзора", "обзоров"))
    return _COMMA.join(parts)


def _step_outcome(step: _Step) -> ToolResult:
    """Результат шага, который уже выполнен: строки шага печатаются только после него."""
    if step.outcome is None:
        raise AssertionError
    return step.outcome


def _split_done(steps: Sequence[_Step]) -> tuple[list[_Step], list[_Step]]:
    """Выполненные шаги: удачные и неудачные; шаги без результата пропускаются."""
    ok = [step for step in steps if step.outcome is not None and step.outcome.ok]
    failed = [step for step in steps if step.outcome is not None and not step.outcome.ok]
    return ok, failed


def _edit_totals(steps: Iterable[_Step]) -> dict[str, list[int]]:
    """Добавлено и удалено строк по файлам — по выполненным правкам."""
    totals: dict[str, list[int]] = {}
    for step in steps:
        stat = _diff_stat(_step_outcome(step)) or (0, 0)
        total = totals.setdefault(_target(step.call), [0, 0])
        total[0] += stat[0]
        total[1] += stat[1]
    return totals


def _sum_totals(totals: dict[str, list[int]]) -> tuple[int, int]:
    """Добавлено и удалено строк во всех файлах вместе."""
    added = sum(file_added for file_added, _ in totals.values())
    removed = sum(file_removed for _, file_removed in totals.values())
    return added, removed


def _subagent_parts(ok: Sequence[_Step]) -> list[str]:
    """Счётчики удачных шагов суб-агента: чтения, правки по файлам, команды."""
    parts = []
    reads = [step.call for step in ok if step.call.read_only]
    if reads:
        parts.append(_read_counts(reads))
    edits = _edit_totals(step for step in ok if step.call.kind is ToolKind.EDIT)
    if len(edits) > 3:
        files = _counted(len(edits), "файл", "файла", "файлов")
        parts.append(f"изменил {files}")
    elif edits:
        shown = _COMMA.join(
            f"{path} +{added} −{removed}" for path, (added, removed) in edits.items()
        )
        parts.append(f"изменил {shown}")
    commands = sum(1 for step in ok if step.call.kind is ToolKind.COMMAND)
    if commands:
        parts.append(_counted(commands, "команда", "команды", "команд"))
    return parts


def _progress_label(title: str, detail: Callable[[], str] | None) -> str:
    """Подпись индикатора долгой операции: заголовок и, если есть, подробности."""
    if detail is None:
        return title
    return f"{title}: {detail()}"


# Методы сгруппированы по разделам ленты: порядок по видимости разорвал бы разделы.
class Console(events.AgentEvents):  # noqa: WPS338
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
        """Консоль поверх rich; ``file`` — куда печатать (по умолчанию — stdout)."""
        self._c = _SafeConsole(
            no_color=no_color,
            highlight=False,
            emoji=False,
            file=file,
            force_terminal=force_terminal,
            width=width,
        )
        # Точные цвета или, в терминале на 16 цветов, цвета ANSI его темы.
        self._p = theme.palette_for(self._c.color_system)
        self._c.push_theme(Theme(theme.markdown_styles(self._p)))
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
        self._mode_hint_source: Callable[[], str] | None = None
        self._input_guard: Callable[[], AbstractContextManager[None]] = nullcontext
        self._esc_stops_subagent = False
        # Шаги: идущая группа и вызов вне группы, ждущий результата.
        self._group: _Group | None = None
        self._step: _Step | None = None
        # Работающий суб-агент: кто, с какого момента, чем занят, подводит ли итог, шаги.
        self._sub: events.SubagentInfo | None = None
        self._sub_started: float = 0
        self._sub_activity = ""
        self._sub_wrapping = False
        self._sub_steps: list[_Step] = []
        self._keyboard_only = False  # отвечать на вопросы — только с терминала (-p)

    def keyboard_only(self) -> None:
        """Вопросы (подтверждения, выбор чата) — только с клавиатуры.

        Ввод не с терминала — ответа нет. Для ``-p``: stdin там не канал ввода (запрос —
        аргумент), и строка «y» или «a» из перенаправленных в него данных не должна ничего
        подтверждать. В REPL ввод не с терминала — это и запросы, и ответы.
        """
        self._keyboard_only = True

    def _no_keyboard(self) -> bool:
        return self._keyboard_only and not _stdin_is_terminal()

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
        """Режим разрешений в индикаторе хода.

        Перечитывается при каждой отрисовке, поэтому смена режима во время хода
        (Shift+Tab) видна сразу.
        """
        self._mode_hint_source = hint

    def _turn_hint(self) -> str:
        if self._mode_hint_source is None:
            return self._interrupt_hint
        return f"{self._interrupt_hint}  ·  {self._mode_hint_source()}"

    def _subagent_hint(self) -> str:
        if not self._esc_stops_subagent:
            return self._turn_hint()
        keys = "Esc — оборвать суб-агента" if self._sub_wrapping else "Esc — завершить с итогом"
        hint = f"{keys}  ·  Ctrl+C — прервать всё"
        return hint if self._mode_hint_source is None else f"{hint}  ·  {self._mode_hint_source()}"

    @property
    def no_color(self) -> bool:
        """Вывод без цвета."""
        return self._no_color

    def interactive(self) -> bool:
        """Можно ли спросить пользователя: ввод идёт с терминала (не pipe/файл)."""
        return _stdin_is_terminal()

    # ----------------------------- базовое ----------------------------- #
    def print(self, *args: Any, **kwargs: Any) -> None:
        """Печать в ленту (как у rich) — после итога идущей группы шагов."""
        self._flush_group()
        self._gap = False
        self._c.print(*args, **kwargs)
        self._blank = not args

    def blank(self) -> None:
        """Пустая строка в ленте."""
        self.print()

    def rule(self, title: str = "") -> None:
        """Горизонтальная черта во всю ширину, с заголовком или без."""
        self._flush_group()
        self._c.rule(Text(title, style=self._p.muted) if title else "", style=self._p.dim)
        self._blank = False

    # Сообщения печатаются над временной областью, если она есть (например,
    # уведомление суб-агента во время его работы).
    def system(self, text: str) -> None:
        """Служебное сообщение."""
        self._emit(Text(text, style=self._p.muted))

    def error(self, text: str) -> None:
        """Сообщение об ошибке."""
        danger = self._p.danger
        self._emit(Text(f"{theme.ICON_FAIL} {text}", style=f"bold {danger}"))

    def warn(self, text: str) -> None:
        """Предупреждение."""
        self._emit(Text(f"{theme.ICON_DANGER} {text}", style=self._p.warn))

    def info(self, text: str) -> None:
        """Сведения для пользователя."""
        self._emit(Text(text, style=self._p.muted))

    def success(self, text: str) -> None:
        """Сообщение об успехе."""
        ok = self._p.ok
        self._emit(Text(f"{theme.ICON_OK} {text}", style=f"bold {ok}"))

    # ------------------------------ вывод ------------------------------ #
    def _out(self, renderable: RenderableType = "", **options: Any) -> None:
        """Печать над временной областью (если она есть)."""
        if self._live is None:
            self._c.print(renderable, **options)
        else:
            self._live.print(renderable, **options)
        self._blank = isinstance(renderable, str) and not renderable

    def _gap_line(self) -> None:
        """Пустая строка после ответа модели — перед первым, что печатается за ним."""
        if self._gap:
            self._gap = False
            if not self._blank:
                self._out()

    def _emit(self, renderable: RenderableType, **options: Any) -> None:
        """Строка ленты: сначала — итог идущей группы шагов и отступ после ответа."""
        self._flush_group()
        self._gap_line()
        self._out(renderable, **options)

    def _fit(self, text: Text) -> Text:
        """Строка по ширине экрана: не переносится, а обрезается многоточием."""
        text.truncate(max(self._c.width - 1, _MIN_WIDTH), overflow="ellipsis")
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
        live = self._live
        self._live = None
        if live is not None:
            live.stop()

    @contextmanager
    def progress(self, title: str, detail: Callable[[], str] | None = None) -> Generator[None]:
        """Индикатор долгой операции вне хода агента (например, построения индекса).

        ``detail`` перечитывается при каждой отрисовке («1234 файлов»). Вне
        терминала печатается одна строка с ``title``.
        """
        if not self._live_ok:
            self.system(f"{title}…")
            yield
            return

        label = functools.partial(_progress_label, title, detail)
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
        palette = self._p
        rows = [
            banner_art.title(palette),
            Text(f"v{version}", style=palette.dim),
            Text(""),
            Text.assemble(("модель  ", palette.muted), (model, f"bold {palette.fg}")),
            Text.assemble(("проект  ", palette.muted), (root, palette.fg)),
        ]
        if mode is not None:
            rows.append(
                Text.assemble(
                    ("режим   ", palette.muted),
                    (mode_badge(mode), f"bold {self._mode_color(mode)}"),
                    ("   Shift+Tab — сменить", palette.dim),
                ),
            )
        details = Group(*rows)
        if self._c.width < _BANNER_MIN_WIDTH:
            body: RenderableType = details
        else:
            grid = Table.grid(padding=(0, 3))
            grid.add_column(vertical="middle", no_wrap=True)
            grid.add_column(vertical="middle")
            grid.add_row(banner_art.rocket(palette, mono=self._c.no_color), details)
            body = grid
        self._c.print(
            Panel(body, box=ROUNDED, border_style=palette.dim, padding=(1, 2), expand=False),
        )
        hint = Text(_SPACER)
        for index, (cmd, desc) in enumerate(hints):
            if index:
                hint.append("   ")
            hint.append(cmd, style=palette.muted)
            hint.append(f" {desc}", style=palette.dim)
        self._c.print(hint)
        if auto_approve:
            self.warn(
                "авто-подтверждение включено (-y): изменения применяются без вопросов, "
                "опасные команды — с вопросом (--yes-all — и они без вопроса)",
            )
        self.print()

    def help(
        self,
        commands: Sequence[tuple[str, str]],
        keys: Sequence[tuple[str, str]] = (),
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
        brand = self._p.brand
        mark = Segment(f"{theme.ICON_BRAND} ", self._c.get_style(f"bold {brand}"))
        pad = Segment(_SPACER)
        first = not self._answer_open
        out = []
        for line in lines:
            out.append([mark if first else pad, *line])
            first = False
        if commit and lines:
            self._answer_open = True
        return out

    def _answer_width(self) -> int:
        return max(self._c.width - 2, _MIN_WIDTH)

    @override
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
    @override
    def on_stream_start(self) -> None:
        """Запрос отправлен: индикатор ожидания до первого токена."""
        self._md = None
        self._raw_started = False
        self._start_live(self._view("думаю"))

    def _answer_tail(self) -> Lines:
        md = self._md
        return [] if md is None else self._gutter(md.tail_lines(), commit=False)

    @override
    def on_stream_delta(self, text: str) -> None:
        """Очередной кусок текста модели."""
        text = sanitize(text)
        if not text:
            return
        if not self._live_ok:
            if not self._raw_started:
                self._raw_started = True
                self._answer_start()
                self._c.file.write(f"{theme.ICON_BRAND} ")
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

    @override
    def on_stream_end(self) -> None:
        """Ответ закончен (или прерван): убрать индикатор, дорисовать хвост."""
        md = self._md
        self._md = None
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
    def _find(self, call: events.ToolCallInfo, steps: Sequence[_Step] = ()) -> _Step | None:
        """Шаг вызова: тот же объект или последний шаг без результата с тем же вызовом.

        Второе — для копии вызова с подробностями вопроса: совпадают имя и описание.
        """
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
            if step.outcome is None and (step.call.name, step.call.summary) == (
                call.name,
                call.summary,
            ):
                return step
        return None

    def _in_group(self, step: _Step) -> bool:
        return self._group is not None and any(grouped is step for grouped in self._group.steps)

    @override
    def on_tool_call(self, call: events.ToolCallInfo) -> None:
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

    @override
    def on_review_start(self, call: events.ToolCallInfo) -> None:
        self._start_live(self._view("рецензент проверяет команду"))

    @override
    def on_review_end(self, call: events.ToolCallInfo, review: events.Review | None) -> None:
        self.stop_live()
        if review is None or not review.allowed:
            return  # отказ или сбой — дальше вопрос пользователю с подробностями
        step = self._find(call)
        if step is None:
            self._emit(Text(f"  {sanitize(review.approval)}", style=self._p.dim))
        else:
            step.review = review  # строкой под шагом, когда он выполнится

    @override
    def on_tool_start(self, call: events.ToolCallInfo) -> None:
        step = self._find(call)
        if step is not None:
            step.started = time.monotonic()
        target = _clip(_target(call), _LABEL_CHARS)
        if call.kind is ToolKind.COMMAND and target:
            label = f"$ {target}"
        else:
            label = f"{call.name} · {target}" if target else call.name
        self._start_live(self._view(label))

    @override
    def on_tool_end(self, call: events.ToolCallInfo) -> None:
        step = self._find(call)
        if step is not None and step.started:
            step.seconds = time.monotonic() - step.started
        self.stop_live()

    @override
    def on_tool_result(
        self,
        call: events.ToolCallInfo,
        result: ToolResult,
        *,
        previewed: bool,
    ) -> None:
        step = self._find(call)
        if step is None:  # результат без on_tool_call — строкой сразу
            step = _Step(call)
        step.outcome = result
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

    def _detail(self, call: events.ToolCallInfo, tool_result: ToolResult) -> str:
        """Итог удачного шага: у команды — последняя строка вывода (итог тестов, сборки).

        У прочего — размер вывода, код выхода или итог без цели.
        """
        shown = tool_result.display
        if shown is not None and shown.kind == "text":
            lines = [line.strip() for line in shown.text.splitlines() if line.strip()]
            if lines and call.kind is ToolKind.COMMAND:
                return _clip(sanitize(lines[-1]))
            if lines:
                count = _counted(len(lines), "строка", "строки", "строк")
                return f"вывод: {count}"
        if tool_result.exit_code is not None:
            return f"код {tool_result.exit_code}"
        summary = sanitize(tool_result.summary or "готово")
        target = _target(call)
        if target and target in summary:  # «прочитан a.py (12 строк)» — путь уже в шаге
            summary = summary.replace(target, " ").strip(" :")
            summary = re.sub(r"\s{2,}", " ", summary)
        return _clip(summary or "готово")

    def _failure(self, tool_result: ToolResult) -> str:
        if tool_result.exit_code is not None:
            return f"код {tool_result.exit_code}"
        return sanitize(tool_result.summary or "ошибка")

    def _edit_line(self, call: events.ToolCallInfo, tool_result: ToolResult) -> Text:
        palette = self._p
        path = _target(call)
        if not tool_result.ok:
            reason = sanitize(tool_result.summary or "ошибка")
            return Text.assemble(
                (f"{theme.ICON_TOOL} ", palette.danger),
                (f"{call.name} {path}".strip(), f"bold {palette.fg}"),
                (f"  {theme.ICON_FAIL} {reason}", f"bold {palette.danger}"),
            )
        verb = "Создал" if tool_result.summary.startswith("создан") else "Изменил"
        line = Text.assemble(
            (f"{theme.ICON_TOOL} ", palette.ok),
            (f"{verb} {path}", f"bold {palette.fg}"),
        )
        stat = _diff_stat(tool_result)
        if stat is not None:
            line.append(_SPACER)
            line.append_text(self._stat(*stat))
        return line

    def _step_lines(self, step: _Step) -> list[Text]:
        """Шаг одной строкой: ● что сделано  ✔ итог · время.

        Под ней — рецензент и, если шаг упал, последние строки вывода.
        """
        outcome = _step_outcome(step)
        if step.call.kind is ToolKind.EDIT:
            lines = [self._edit_line(step.call, outcome)]
        else:
            lines = [self._tool_line(step, outcome)]
        if step.review is not None:
            lines.append(Text(f"  {sanitize(step.review.approval)}", style=self._p.dim))
        if not outcome.ok:
            lines += self._output_tail(outcome)
        return lines

    def _tool_line(self, step: _Step, outcome: ToolResult) -> Text:
        """Строка шага, кроме правки: что сделано, итог и время."""
        palette = self._p
        call = step.call
        target = _target(call)
        head = f"$ {target}" if call.kind is ToolKind.COMMAND else f"{call.name} {target}"
        status = Text()
        if outcome.ok:
            status.append(f"{theme.ICON_OK} ", style=palette.ok)
            status.append(self._detail(call, outcome), style=palette.muted)
        else:
            failure = self._failure(outcome)
            status.append(f"{theme.ICON_FAIL} {failure}", style=f"bold {palette.danger}")
        if step.seconds >= _SHOW_SECONDS:
            status.append(f"  ·  {step.seconds:.1f} с", style=palette.dim)
        # Не помещается в строку — короче команда, а не итог.
        room = self._c.width - 1 - len(status) - 4
        head = head.strip()
        if len(head) > room:
            head = _clip(head, max(room, 24))
        line = Text.assemble(
            (f"{theme.ICON_TOOL} ", palette.ok if outcome.ok else palette.danger),
            (head, f"bold {palette.fg}"),
            (_SPACER, ""),
        )
        return self._fit(line + status)

    def _output_tail(self, outcome: ToolResult) -> list[Text]:
        """Последние строки вывода упавшего шага, без строк-разделителей."""
        shown = outcome.display
        if shown is None or shown.kind != "text":
            return []
        output = [line for line in sanitize(shown.text).splitlines() if line.strip()]
        output = [line for line in output if not _SEPARATOR_RE.match(line.strip())]
        danger = self._p.danger
        return [
            self._fit(Text.assemble(("  │ ", danger), (out_line, "")))
            for out_line in output[-_TAIL_LINES:]
        ]

    def _failed_line(self, step: _Step) -> Text:
        """Неудачный шаг внутри группы: ✘ что — почему."""
        palette = self._p
        outcome = _step_outcome(step)
        call = step.call
        target = _target(call)
        return Text.assemble(
            (f"  {theme.ICON_FAIL} ", f"bold {palette.danger}"),
            (f"{call.name} {target}".strip(), palette.danger),
            (_SPACER, ""),
            (sanitize(outcome.summary or "ошибка"), palette.danger),
        )

    def _read_lines(self, steps: Sequence[_Step]) -> list[Text]:
        """Итог группы чтений: «Изучил код» со счётчиками, цели, ошибки."""
        palette = self._p
        if len(steps) == 1:
            return self._step_lines(steps[0])
        ok, failed = _split_done(steps)
        head = Text.assemble(
            (f"{theme.ICON_TOOL} ", palette.ok if ok else palette.danger),
            ("Изучил код", f"bold {palette.fg}"),
        )
        rest = _read_counts([step.call for step in ok])
        seconds = sum(step.seconds for step in steps)
        if seconds >= _SHOW_SECONDS:
            rest = f"{rest}  ·  {seconds:.1f} с" if rest else f"{seconds:.1f} с"
        if rest:
            head.append(f"  {rest}", style=palette.muted)
        lines = [head]
        targets = list(dict.fromkeys(_target(step.call) for step in ok if _target(step.call)))
        if targets:
            joined = _SPACER.join(targets)
            lines.append(self._fit(Text(f"  {joined}", style=palette.muted)))
        lines += [self._failed_line(step) for step in failed]
        return lines

    def _edit_lines(self, steps: Sequence[_Step]) -> list[Text]:
        """Итог группы правок: файл и +N −M; несколько файлов — строка на файл."""
        ok, failed = _split_done(steps)
        if not ok:  # все правки упали — каждая своей строкой
            return [line for step in failed for line in self._step_lines(step)]
        totals = _edit_totals(ok)
        # Один файл (в том числе несколько правок подряд) — как одна правка.
        lines = [self._one_file_line(ok, totals)] if len(totals) <= 1 else self._files_lines(totals)
        return lines + [self._failed_line(step) for step in failed]

    def _one_file_line(self, ok: Sequence[_Step], totals: dict[str, list[int]]) -> Text:
        """Удачные правки одного файла: одна — как есть, несколько — с суммой строк."""
        last = ok[-1]
        if len(ok) == 1:
            return self._edit_line(last.call, _step_outcome(last))
        palette = self._p
        line = Text.assemble(
            (f"{theme.ICON_TOOL} ", palette.ok),
            (f"Изменил {_target(last.call)}", f"bold {palette.fg}"),
        )
        added, removed = next(iter(totals.values()))
        line.append(_SPACER)
        line.append_text(self._stat(added, removed))
        return line

    def _files_lines(self, totals: dict[str, list[int]]) -> list[Text]:
        """Правки нескольких файлов: общий итог и строка на файл."""
        palette = self._p
        files = _counted(len(totals), "файл", "файла", "файлов")
        head = Text.assemble(
            (f"{theme.ICON_TOOL} ", palette.ok),
            (f"Изменил {files}", f"bold {palette.fg}"),
            (_SPACER, ""),
        )
        head.append_text(self._stat(*_sum_totals(totals)))
        lines = [head]
        width = max(len(path) for path in totals)
        for path, (added, removed) in totals.items():
            padded = path.ljust(width)
            line = Text(f"  {padded}  ", style=palette.fg)
            line.append_text(self._stat(added, removed))
            lines.append(line)
        return lines

    def _flush_group(self) -> None:
        """Итог идущей группы — в ленту (идемпотентно)."""
        group = self._group
        self._group = None
        if group is None:
            return
        steps = [step for step in group.steps if step.outcome is not None]
        if not steps:
            return
        lines = self._read_lines(steps) if group.kind == "read" else self._edit_lines(steps)
        self._gap_line()
        for line in lines:
            self._out(line)

    def _live_rows(self, steps: Sequence[_Step]) -> list[Text]:
        """Последние шаги для временной области: ✔/✘/◌, инструмент, цель, итог."""
        rows = []
        hidden = len(steps) - _LIVE_STEPS
        if hidden > 0:
            rows.append(Text(f"  … ещё {hidden}", style=self._p.dim))
            steps = steps[-_LIVE_STEPS:]
        rows += [self._live_row(step) for step in steps]
        return rows

    def _live_row(self, step: _Step) -> Text:
        """Шаг во временной области: ✔/✘/◌, инструмент, цель и итог, если он есть."""
        palette = self._p
        outcome = step.outcome
        if outcome is None:
            mark = (f"  {theme.ICON_PENDING} ", palette.warn)
        elif outcome.ok:
            mark = (f"  {theme.ICON_OK} ", palette.ok)
        else:
            mark = (f"  {theme.ICON_FAIL} ", palette.danger)
        row = Text.assemble(
            mark,
            (step.call.name, palette.muted),
            (_SPACER, ""),
            (_target(step.call), palette.fg),
        )
        if outcome is not None:
            row.append(_SPACER)
            stat = _diff_stat(outcome) if outcome.ok else None
            if stat is None:
                row.append(
                    sanitize(outcome.summary or ""),
                    style=palette.dim if outcome.ok else palette.danger,
                )
            else:
                row.append_text(self._stat(*stat))
        return self._fit(row)

    def _live_steps(self) -> list[Text]:
        """Что показать над индикатором: идущую группу или шаги суб-агента."""
        group = self._group
        if group is not None and group.steps:
            palette = self._p
            steps = list(group.steps)
            title = "Изучаю код" if group.kind == "read" else "Правлю файлы"
            calls = _counted(len(steps), "вызов", "вызова", "вызовов")
            head = Text.assemble(
                (f"{theme.ICON_TOOL} ", palette.warn),
                (title, f"bold {palette.fg}"),
                (f"  {calls}", palette.muted),
            )
            return [head, *self._live_rows(steps)]
        if self._sub is not None and self._sub_steps:
            return self._live_rows(list(self._sub_steps))
        return []

    # ---------------------------- суб-агенты ----------------------------- #
    @override
    def on_subagent_start(self, info: events.SubagentInfo) -> None:
        self._flush_group()
        self._sub = info
        self._sub_started = time.monotonic()
        self._sub_activity = "думаю"
        self._sub_wrapping = False
        self._sub_steps = []
        self._start_subagent_live()

    def _subagent_label(self) -> str:
        name = "суб-агент" if self._sub is None else self._sub.name
        return f"{name} · {self._sub_activity}"

    def _start_subagent_live(self) -> None:
        if self._sub is None:
            return
        self._start_live(
            self._view(self._subagent_label, hint=self._subagent_hint, started=self._sub_started),
        )

    @override
    def on_subagent_activity(
        self,
        info: events.SubagentInfo,
        activity: str,
        *,
        wrapping_up: bool = False,
    ) -> None:
        self._sub_activity = activity
        self._sub_wrapping = wrapping_up
        if self._live is None:  # область убрал вопрос пользователю — вернуть
            self._start_subagent_live()

    @override
    def on_subagent_tool_call(self, info: events.SubagentInfo, call: events.ToolCallInfo) -> None:
        self._sub_steps.append(_Step(call))

    @override
    def on_subagent_tool_result(
        self,
        info: events.SubagentInfo,
        call: events.ToolCallInfo,
        result: ToolResult,
        *,
        previewed: bool,
    ) -> None:
        step = self._find(call, self._sub_steps) if self._sub_steps else None
        if step is None:
            step = _Step(call)
            self._sub_steps.append(step)
        step.outcome = result

    @override
    def on_subagent_end(self, info: events.SubagentInfo, stats: events.TurnStats | None) -> None:
        self.stop_live()  # до сброса: последняя отрисовка ещё читает метку
        self._sub = None
        self._sub_wrapping = False

    def _subagent_lines(self, steps: Sequence[_Step]) -> list[Text]:
        """Что сделал суб-агент — под строкой его вызова: счётчики, правки, неудачи."""
        ok, failed = _split_done(steps)
        parts = _subagent_parts(ok)
        lines = []
        if parts:
            summary = _COMMA.join(parts)
            lines.append(Text(f"  {summary}", style=self._p.muted))
        lines += [self._failed_line(step) for step in failed]
        return lines

    # ---------------------------- статистика ---------------------------- #
    @override
    def on_compact_start(self, *, auto: bool) -> None:
        title = "контекст почти заполнен — сжимаю историю" if auto else "сжимаю историю"
        if not self._live_ok:
            self.system(f"{title}…")
            return
        self._start_live(self._view(title))

    @override
    def on_compact_end(self, result: events.CompactResult | None) -> None:
        self.stop_live()
        if result is None:
            return
        before = format_tokens(result.before_tokens)
        after = format_tokens(result.after_tokens)
        folded = _counted(result.messages, "сообщение", "сообщения", "сообщений")
        self.system(
            f"✂ контекст сжат: ~{before} → ~{after} ток. · в краткое содержание свёрнуто {folded}",
        )

    @override
    def on_hook_start(self, event: str, command: str) -> None:
        title = f"хук {event} · {sanitize(short_command(command))}"
        if not self._live_ok:
            self.system(f"{title}…")
            return
        self._start_live(self._view(title))

    @override
    def on_hook_end(self) -> None:
        self.stop_live()

    @override
    def on_notice(self, text: str, *, level: events.NoticeLevel = "info") -> None:
        # В тексте может быть вывод хуков пользователя — escape-последовательности убираются.
        {"info": self.info, "warn": self.warn, "error": self.error}[level](sanitize(text))

    @override
    def on_turn_end(self, stats: events.TurnStats) -> None:
        parts = [_counted(stats.steps, "шаг", "шага", "шагов")]
        if stats.tool_calls:
            parts.append(_counted(stats.tool_calls, "инструмент", "инструмента", "инструментов"))
        if stats.duration_s:
            parts.append(f"{stats.duration_s:.1f} с")
        if stats.context_tokens:
            parts.append(f"контекст ~{format_tokens(stats.context_tokens)}")
        if stats.billed_tokens:
            parts.append(f"потрачено {format_tokens(stats.billed_tokens)} ток.")
        summary = "  ·  ".join(parts)
        self._emit(Text(f"  {summary}", style=self._p.dim))

    # -------------------------- подтверждения --------------------------- #
    def _stripe(self, *, dangerous: bool = False) -> tuple[str, str]:
        return (f"  {theme.ICON_BAR} ", self._p.danger if dangerous else self._p.warn)

    def _confirm_head(
        self,
        call: events.ToolCallInfo,
        preview: Display | None,
        stripe: tuple[str, str],
    ) -> bool:
        """Что спрашивается — первой строкой вопроса.

        Однострочное превью команды показывается здесь же, как выполнится (с видимыми
        невидимыми символами). True — превью не помещается на экран.
        """
        palette = self._p
        text = "" if preview is None else preview.text.rstrip("\n")
        if preview is not None and preview.kind != "diff" and text.strip() and "\n" not in text:
            head = visible_text(text, numbered=False, palette=palette)
            head.stylize("bold")
            self._out(Text.assemble(stripe) + head)
            return False
        target = _target(call)
        if call.kind is ToolKind.COMMAND:
            head_text = f"$ {target}"
        else:
            head_text = f"{call.name} {target}".strip()
        self._out(Text.assemble(stripe, (head_text, f"bold {palette.fg}")))
        if preview is None or not text.strip():
            return False
        if preview.kind == "diff":
            self._out(Padding(self._diff_panel(preview), (0, 0, 0, 4)))
            return False
        return self._exact_block(preview, border=stripe[1])

    @override
    def confirm(
        self,
        call: events.ToolCallInfo,
        preview: Display | None,
        *,
        dangerous: bool,
    ) -> events.ConfirmResult:
        self._detach_step(call)
        self._flush_group()
        self.stop_live()  # превью печатается до вопроса — область уже не нужна
        self._gap_line()
        stripe = self._stripe(dangerous=dangerous)
        long_preview = self._confirm_head(call, preview, stripe)
        self._confirm_details(call, preview, stripe, dangerous=dangerous)
        question = _confirm_question(call, dangerous=dangerous)
        always = _always_note(call, dangerous=dangerous)
        raw = self._confirm_answer(
            question,
            preview,
            dangerous=dangerous,
            always=always,
            view=long_preview,
        )
        if raw is None:
            return events.ConfirmResult(events.Approval.NO, answered=False)
        return _confirm_result(raw, always)

    def _detach_step(self, call: events.ToolCallInfo) -> None:
        """Спрошенный вызов — не часть группы: его строка — после ответа."""
        step = self._find(call)
        if step is not None and self._group is not None and self._in_group(step):
            self._group.steps = [grouped for grouped in self._group.steps if grouped is not step]
            self._step = step

    def _confirm_details(
        self,
        call: events.ToolCallInfo,
        preview: Display | None,
        stripe: tuple[str, str],
        *,
        dangerous: bool,
    ) -> None:
        """Подробности вопроса о вызове: каталог, причина, цель, рецензент, история."""
        palette = self._p
        labeled = self._confirm_labels(call, preview, dangerous=dangerous)
        width = max((len(label) for label, _, _ in labeled), default=0)
        for label, style, text in labeled:
            padded = label.ljust(width)
            self._out(Text.assemble(stripe, (f"{padded}  ", style)) + text)
        if call.review:
            self._out(Text.assemble(stripe, (sanitize(call.review), palette.muted)))
        if call.history:
            self._out(Text.assemble(stripe, (sanitize(call.history), palette.muted)))
        self._out(Text.assemble(stripe))

    def _confirm_labels(
        self,
        call: events.ToolCallInfo,
        preview: Display | None,
        *,
        dangerous: bool,
    ) -> list[tuple[str, str, Text]]:
        """Подписанные строки вопроса о вызове: (подпись, её стиль, текст)."""
        palette = self._p
        labeled = []
        where = _command_dir(call, preview)
        if where:  # где выполнится команда — тоже часть того, что подтверждается
            labeled.append(("каталог", palette.muted, Text(where, style=f"bold {palette.fg}")))
        if call.reason:
            label = "почему опасно" if dangerous else "почему спрашиваю"
            style = f"bold {palette.danger}" if dangerous else palette.muted
            labeled.append((label, style, Text(sanitize(call.reason), style=palette.fg)))
        if call.purpose:
            purpose = Text.assemble(
                (sanitize(call.purpose), palette.muted),
                ("  со слов модели", palette.dim),
            )
            labeled.append(("зачем", palette.muted, purpose))
        return labeled

    def _confirm_answer(
        self,
        question: str,
        preview: Display | None,
        *,
        dangerous: bool,
        always: str | None,
        view: bool,
    ) -> str | None:
        """Ответ на вопрос о вызове как введён; «v» — превью в просмотрщике и вопрос снова."""
        while True:
            raw = self._question_raw(
                question,
                dangerous=dangerous,
                always=always,
                feedback=True,
                view=view,
            )
            if raw is None or preview is None or raw.strip().lower() not in _VIEW_ANSWERS:
                return raw
            self._page(preview)  # и спросить снова

    def _diff_panel(self, preview: Display) -> Panel:
        # Невидимые символы правки — видимыми (как в команде), с подсветкой диффа.
        code = mark_invisible(preview.text).rstrip()
        body = Syntax(code, "diff", theme="ansi_dark", background_color="default").highlight(code)
        body.rstrip()
        body.no_wrap = False  # длинные строки — переносом
        body.overflow = "fold"
        danger = self._p.danger
        body.highlight_regex(_INVISIBLE_MARK_RE, f"bold {danger}")
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
        palette = self._p
        text = preview.text.rstrip("\n")
        lines = text.split("\n")
        title = preview.title or "превью"
        if len(lines) >= _NUMBER_LINES_FROM:
            count = _counted(len(lines), "строка", "строки", "строк")
            title = f"{title} · {count}"
        panel = Panel(
            visible_text(text, palette=palette),
            title=Text(title, style=palette.muted),
            title_align="left",
            box=ROUNDED,
            border_style=border,
            padding=(0, 1),
            expand=False,
        )
        self._out(Padding(panel, (0, 0, 0, 4)))
        width = max(self._c.size.width - 8, _MIN_WIDTH)
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
        return self._question(question, dangerous=dangerous) in _YES_ANSWERS

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
        """Ответ как введён; None — ответа не было.

        Ответа нет при Ctrl+C/Ctrl+D, конце ввода и вводе не с терминала там, где отвечать
        можно только с клавиатуры.
        """
        self._flush_group()
        self.stop_live()
        palette = self._p
        stripe = self._stripe(dangerous=dangerous)
        if self._no_keyboard():
            self._out(
                Text.assemble(
                    stripe,
                    (f"{question} ", f"bold {palette.fg}"),
                    (
                        "— ответить можно только с клавиатуры, а ввод не с терминала: ",
                        palette.muted,
                    ),
                    ("нет", f"bold {palette.fg}"),
                ),
            )
            self._blank = False
            return None
        notes = _question_notes(always, dangerous=dangerous, feedback=feedback, view=view)
        joined = " · ".join(notes)
        if len(joined) + len(stripe[0]) < self._c.width:
            notes = [joined] if notes else []
        for note in notes:  # не помещаются в строку — по строке на подсказку
            self._out(Text.assemble(stripe, (note, palette.dim)))
        mark = f"{theme.ICON_DANGER} " if dangerous else "? "
        keys = "y/N"
        if always is not None:
            keys = f"{keys}/a"
        if view:
            keys = f"{keys}/v"
        prompt = Text.assemble(
            stripe,
            (mark, f"bold {stripe[1]}"),
            (question, f"bold {palette.fg}"),
            (f" [{keys}] ", palette.muted),
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
    @override
    def ask_user(self, questions: Sequence[Question]) -> list[Answer] | None:
        """Задаёт вопросы по очереди (меню со стрелками или ввод номера). None — отказ."""
        self._flush_group()
        self.stop_live()
        if not _stdin_is_terminal():
            raise QuestionsUnavailableError("ввод не с клавиатуры")
        palette = self._p
        answers: list[Answer] = []
        with self._input_guard():
            for number, question in enumerate(questions, 1):
                if question.body.strip():
                    self._question_body(question)
                answer = self._ask_one(question, number, len(questions))
                if answer is None:
                    self._emit(
                        Text.assemble(
                            (f"{theme.ICON_TOOL} ", palette.dim),
                            (sanitize(question.text), palette.muted),
                            ("  пользователь отказался отвечать", palette.dim),
                        ),
                    )
                    return None
                self._emit(self._answered(question, answer))
                answers.append(answer)
        return answers

    def _answered(self, question: Question, answer: Answer) -> Text:
        """Строка ленты после ответа: вопрос серым, ответ ярким."""
        palette = self._p
        line = Text.assemble(
            (f"{theme.ICON_TOOL} ", palette.ok),
            (sanitize(question.text), palette.muted),
            (_SPACER, ""),
        )
        for index, selected in enumerate(answer.selected):
            if index:
                line.append(_COMMA, style=palette.muted)
            line.append(sanitize(selected), style=f"bold {palette.fg}")
        if answer.custom:
            if answer.selected:
                line.append(_COMMA, style=palette.muted)
            label = (question.custom_label or "Свой ответ…").rstrip("…").lower()
            line.append(f"{label} ", style=palette.muted)
            line.append(f"«{sanitize(answer.custom)}»", style=f"bold {palette.fg}")
        if not answer.selected and not answer.custom:
            line.append("(без ответа)", style=palette.muted)
        return line

    def _question_body(self, question: Question) -> None:
        """Подробности к вопросу (например, план на одобрение) — Markdown в рамке."""
        palette = self._p
        header = question.header or "подробности"
        self._emit(
            Panel(
                Markdown(sanitize(question.body)),
                title=Text(f" {header} ", style=f"bold {palette.brand}"),
                title_align="left",
                subtitle=Text(f" {sanitize(question.source)} ", style=palette.dim)
                if question.source
                else None,
                subtitle_align="right",
                box=ROUNDED,
                border_style=palette.dim,
                padding=(1, 2),
            ),
        )

    def _ask_one(self, question: Question, index: int, total: int) -> Answer | None:
        if self._live_ok:
            try:
                # Меню на prompt_toolkit грузится лишь к первому вопросу: запуск быстрее.
                from gigacrew.ui.choice import choose  # noqa: PLC0415

                return choose(question, index, total, no_color=self._no_color)
            except (KeyboardInterrupt, EOFError):
                return None
        return self._ask_plain(question, index, total)

    def _ask_plain(self, question: Question, index: int, total: int) -> Answer | None:
        """Запасной режим без интерактивного меню: номер(а) варианта или свой текст."""
        stripe = self._stripe()
        self._plain_menu(question, index, total, stripe)
        while True:
            try:
                answer = self._plain_attempt(question, stripe)
            except (EOFError, KeyboardInterrupt):
                self._c.print()
                return None
            finally:
                self._blank = False
            if answer is not None:
                return answer

    def _plain_menu(
        self,
        question: Question,
        index: int,
        total: int,
        stripe: tuple[str, str],
    ) -> None:
        """Вопрос запасного режима: заголовок и пронумерованные варианты."""
        palette = self._p
        head = Text.assemble(
            stripe,
            ("? ", f"bold {palette.warn}"),
            (question.text, f"bold {palette.fg}"),
        )
        tag = question.header
        if total > 1:
            tag = f"{tag} · {index} из {total}" if tag else f"{index} из {total}"
        if tag:
            head.append(f"   {tag}", style=palette.muted)
        self._out(head)
        for number, option in enumerate(question.options, 1):
            self._out(
                Text.assemble(stripe, (f"  {number}. ", palette.muted), (option.label, palette.fg)),
            )
            if option.description:
                self._out(Text.assemble(stripe, (f"     {option.description}", palette.dim)))
        custom = len(question.options) + 1
        label = (question.custom_label or "Свой ответ…").rstrip("…")
        self._out(
            Text.assemble(
                stripe,
                (f"  {custom}. ", palette.muted),
                (f"{label} (или просто напишите текст)", palette.fg),
            ),
        )

    def _plain_attempt(self, question: Question, stripe: tuple[str, str]) -> Answer | None:
        """Одна попытка ответа в запасном режиме; None — спросить заново."""
        palette = self._p
        options = question.options
        custom = len(options) + 1
        hint = "номера через пробел" if question.multi_select else "номер"
        prompt = Text.assemble(stripe, (f"{hint} или ответ: ", palette.muted))
        raw = self._c.input(prompt).strip()
        numbers = raw.replace(",", " ").split()
        if not numbers:
            return None  # пусто или одни запятые — спросить заново
        # isdecimal, а не isdigit: «²» — цифра, но int() её не разберёт
        if not all(number.isdecimal() and 1 <= int(number) <= custom for number in numbers):
            return Answer(custom=raw)  # не номера — это свой ответ
        picked = sorted({int(number) for number in numbers})
        if len(picked) > 1 and not question.multi_select:
            self.warn("здесь можно выбрать только один вариант")
            return None
        text = ""
        if custom in picked:
            text = self._c.input(Text.assemble(stripe, ("ваш ответ: ", palette.muted))).strip()
            if not text:
                return None
        labels = tuple(options[number - 1].label for number in picked if number != custom)
        return Answer(labels, text)

    # ---------------------------- сохранённые чаты ---------------------------- #
    def pick_chat(self, chats: Sequence[ChatInfo], current_id: str = "") -> ChatInfo | None:
        """Выбор чата: меню со стрелками и поиском, без терминала — номер. None — отмена."""
        self._flush_group()
        self.stop_live()
        if not chats:
            return None
        if self._no_keyboard():
            self.warn("выбрать чат можно только с клавиатуры (или по id: -r ID) — начат новый")
            return None
        if self._live_ok and _stdin_is_terminal():
            try:
                # Меню на prompt_toolkit грузится лишь к выбору чата: запуск быстрее.
                from gigacrew.ui.chat_picker import pick_chat  # noqa: PLC0415

                return pick_chat(chats, current_id, no_color=self._no_color)
            except (KeyboardInterrupt, EOFError):
                return None
        return self._pick_chat_plain(chats, current_id)

    def _pick_chat_plain(self, chats: Sequence[ChatInfo], current_id: str) -> ChatInfo | None:
        palette = self._p
        table = Table.grid(padding=(0, 2))
        table.add_column(style=palette.muted, justify="right", no_wrap=True)
        table.add_column(style=palette.dim, no_wrap=True)
        table.add_column()
        for number, chat in enumerate(chats, 1):
            meta = f"{format_when(chat.updated_at)} · {chat_size(chat.requests, chat.messages)}"
            title = Text(sanitize(chat.title), style=palette.fg)
            if chat.id == current_id:
                title.append(" (текущий)", style=palette.muted)
            table.add_row(f"{number}.", meta, title)
        self.print(Text(f"Чаты проекта · {len(chats)}", style=f"bold {palette.brand}"))
        self.print(table)
        while True:
            try:
                raw = self._c.input(Text("  номер чата (Enter — отмена): ", style=palette.muted))
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
        describe: Callable[[FunctionCall], events.ToolCallInfo] | None = None,
    ) -> None:
        """Сообщение о продолжении чата и последние обмены репликами.

        Обмен — запрос пользователя и всё до следующего: шаги (вызовы инструментов,
        ``describe`` — как они выглядели во время хода) и итоговый ответ. Шаги — как в
        ленте, но приглушены и без итогов: результатов в истории нет. Более ранние
        обмены — одной строкой со счётчиками: они сохранены в чате, но не печатаются.
        """
        palette = self._p
        line = Text("↺ продолжаем чат ", style=palette.muted)
        line.append(f"«{sanitize(info.title)}»", style=f"bold {palette.fg}")
        line.append(
            f" · {chat_size(info.requests, info.messages)} · {format_when(info.updated_at)}",
            style=palette.muted,
        )
        self.print(line)
        starts = _exchange_starts(messages)[-RESUME_EXCHANGES:]
        if starts and starts[0] > 0:
            earlier = messages[: starts[0]]
            size = chat_size(sum(1 for message in earlier if message.is_request), len(earlier))
            self.print(
                Text(
                    f"  … раньше: {size} — сохранены в чате, на экран не выводятся",
                    style=palette.dim,
                ),
            )
        for index, start in enumerate(starts):
            end = starts[index + 1] if index + 1 < len(starts) else len(messages)
            if index:
                self.print()
            self._resumed_exchange(messages[start:end], describe, last=index == len(starts) - 1)
        self.print()

    def _resumed_exchange(
        self,
        messages: Sequence[Message],
        describe: Callable[[FunctionCall], events.ToolCallInfo] | None,
        *,
        last: bool,
    ) -> None:
        """Обмен продолженного чата (``messages[0]`` — запрос); ответ ранних — сокращён."""
        palette = self._p
        shown = clip_lines(
            request_text(messages[0].content),
            head=3,
            tail=2,
            max_chars=_RESUME_REQUEST_CHARS,
        )
        self.print(Text.assemble(("❯ ", f"bold {palette.brand}"), (sanitize(shown), palette.fg)))
        calls = [
            events.ToolCallInfo(call.name) if describe is None else describe(call)
            for message in messages
            if message.role == "assistant"
            for call in message.function_calls
        ]
        for line in self._history_lines(calls):
            self.print(line)
        # Ответ ищется только в этом обмене: у прерванного хода его нет, и ответ на
        # предыдущий запрос здесь выглядел бы ответом на этот.
        answer = next(
            (
                message.content
                for message in reversed(messages)
                if message.role == "assistant"
                and not message.function_calls
                and message.content.strip()
            ),
            "",
        )
        if not answer:
            self.print(Text("  (ответа нет — ход не был завершён)", style=palette.muted))
            return
        if not last:
            answer = clip_lines(answer, head=6, tail=2, max_chars=_RESUME_ANSWER_CHARS)
        self.on_assistant_text(answer)
        self._gap = False

    def _history_lines(self, calls: Sequence[events.ToolCallInfo]) -> list[Text]:
        """Шаги из истории: подряд идущие чтения и правки — одной строкой, как в ленте.

        Не больше :data:`RESUME_CALLS` строк, ранние — счётчиком.
        """
        entries = _history_entries(calls)
        lines = []
        if len(entries) > RESUME_CALLS:
            hidden = sum(len(entry) for entry in entries[:-RESUME_CALLS])
            word = plural(hidden, "вызов", "вызова", "вызовов")
            lines.append(Text(f"  … ещё {hidden} {word} инструментов", style=self._p.dim))
            entries = entries[-RESUME_CALLS:]
        lines += [line for entry in entries for line in self._history_entry(entry)]
        return lines

    def _history_entry(self, entry: Sequence[events.ToolCallInfo]) -> list[Text]:
        """Строка шага (группы вызовов) из истории и, у группы чтений, её цели."""
        palette = self._p
        head, detail = _history_head(entry)
        lines = [
            self._fit(Text.assemble((f"{theme.ICON_TOOL} ", palette.dim), (head, palette.muted))),
        ]
        if detail:
            lines.append(self._fit(Text(f"  {detail}", style=palette.dim)))
        return lines


def _confirm_question(call: events.ToolCallInfo, *, dangerous: bool) -> str:
    """Вопрос о вызове: применить правку, выполнить команду или инструмент."""
    if call.kind is ToolKind.EDIT:
        return "Применить правку?"
    if call.kind is ToolKind.COMMAND:
        return "Выполнить опасную команду?" if dangerous else "Выполнить команду?"
    return f"Выполнить {call.name}?"


def _always_note(call: events.ToolCallInfo, *, dangerous: bool) -> str | None:
    """Что запомнит ответ «всегда»; None — так ответить нельзя."""
    if dangerous:  # опасное «навсегда» не разрешается
        return None
    if call.remember:
        return sanitize(call.remember)
    return "не спрашивать этот вызов до конца сессии"


def _confirm_result(raw: str, always: str | None) -> events.ConfirmResult:
    """Решение по ответу на вопрос о вызове (ответ — как введён)."""
    answer = raw.strip().lower()
    if answer in _YES_ANSWERS:
        return events.ConfirmResult(events.Approval.YES)
    if always is not None and answer in _ALWAYS_ANSWERS:
        return events.ConfirmResult(events.Approval.ALWAYS)
    if not answer or answer in _NO_ANSWERS:
        return events.ConfirmResult(events.Approval.NO)
    # Свой ответ вместо y/n — отказ с объяснением: его получит модель.
    return events.ConfirmResult(events.Approval.NO, feedback=raw.strip())


def _question_notes(
    always: str | None,
    *,
    dangerous: bool,
    feedback: bool,
    view: bool,
) -> list[str]:
    """Подсказки к вопросу: что запомнит «a», что откроет «v», можно ли ответить текстом."""
    notes = []
    if always is not None:
        notes.append(f"a — {always}")
    elif dangerous and feedback:
        notes.append("запомнить нельзя")
    if view:
        notes.append("v — открыть целиком в просмотрщике")
    if feedback:
        notes.append("или напишите, что сделать вместо этого")
    return notes


def _command_dir(call: events.ToolCallInfo, preview: Display | None) -> str:
    """Каталог, где выполнится команда, если не корень проекта.

    Превью команды подписано «команда · <каталог>». Пусто — в корне (или это не команда).
    """
    if call.kind is not ToolKind.COMMAND or preview is None:
        return ""
    head, sep, where = preview.title.partition(" · ")
    return where.strip() if sep and head == "команда" else ""


def _exchange_starts(messages: Sequence[Message]) -> list[int]:
    """Индексы запросов пользователя с текстом — начала обменов (служебные — не запросы)."""
    return [
        index
        for index, message in enumerate(messages)
        if message.is_request and message.content.strip()
    ]


def _history_entries(calls: Sequence[events.ToolCallInfo]) -> list[list[events.ToolCallInfo]]:
    """Вызовы из истории по группам: подряд идущие чтения и правки — вместе."""
    entries: list[list[events.ToolCallInfo]] = []
    for call in calls:
        kind = _group_kind(call)
        if kind and entries and _group_kind(entries[-1][0]) == kind:
            entries[-1].append(call)
        else:
            entries.append([call])
    return entries


def _history_head(entry: Sequence[events.ToolCallInfo]) -> tuple[str, str]:
    """Строка шага из истории и подробности под ней (цели группы чтений)."""
    first = entry[0]
    kind = _group_kind(first)
    targets = list(dict.fromkeys(target for call in entry if (target := _target(call))))
    if kind == "read" and len(entry) > 1:
        counts = _read_counts(entry)
        return f"Изучил код  {counts}", _SPACER.join(targets)
    if kind == "edit" and targets:
        files = _COMMA.join(targets)
        if len(targets) <= 3:
            return f"Изменил {files}", ""
        return f"Изменил {len(targets)} файлов: {files}", ""
    if first.kind is ToolKind.COMMAND and targets:
        return f"$ {targets[0]}", ""
    target = _target(first)
    return f"{first.name} {target}".strip(), ""


_YES_ANSWERS = ("y", "yes", "д", "да")
_NO_ANSWERS = ("n", "no", "н", "нет", "т")  # н, т — клавиши y и n в русской раскладке: не «да»
_ALWAYS_ANSWERS = ("a", "always", "в", "всегда", "ф")  # ф — клавиша a в русской раскладке
_VIEW_ANSWERS = ("v", "view", "м", "показать")  # м — клавиша v в русской раскладке


def _stdin_is_terminal() -> bool:
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False
