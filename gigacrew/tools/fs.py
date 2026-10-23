"""Файловые инструменты: чтение, запись, точечное редактирование, листинг, поиск файлов."""

from __future__ import annotations

import difflib
import functools
import itertools
import os
import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import override

from pydantic import BaseModel, Field

from gigacrew.permissions import PermissionMode, ToolKind
from gigacrew.project.files import GIT_DIR_NAME, glob_match, is_git_internal, walk_files
from gigacrew.project.secret_paths import is_secret_target
from gigacrew.project.workspace import DATA_DIR_NAME
from gigacrew.security import RiskLevel, SandboxError, resolve_in_root
from gigacrew.tools.base import Assessment, Display, Tool, ToolContext, ToolError, ToolResult

# Жёсткий предел размера читаемого файла (дальше — только search_content).
MAX_READ_BYTES = 20_000_000
# Сколько отдаётся модели за один вызов: строки, символы на строку, символы всего.
MAX_READ_LINES = 1000
MAX_LINE_CHARS = 1000
MAX_READ_CHARS = 60_000
_BINARY_PROBE = 8192


def _rel(ctx: ToolContext, path: Path) -> str:
    try:
        return str(path.relative_to(ctx.root))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- #
# Файлы с секретами
# --------------------------------------------------------------------------- #
# Из .env проекта gigacrew берёт ключ доступа и обычные настройки (config.load_environment;
# адреса и проверку TLS — нет). Правка без вопроса подменила бы ключ или модель при
# следующем запуске — поэтому её подтверждает пользователь всегда, даже с -y.
CONFIG_FILE_NAME = ".env"
# Не пароль, а текст отказа: S105 срабатывает на слово SECRET в имени.
SECRET_READ_NOTE = (
    "Файл с секретами (ключи, пароли, токены) инструменты не читают: {path}. Если его "  # noqa: S105
    "содержимое действительно нужно, спросите пользователя или прочитайте его через "
    "run_shell — такая команда требует подтверждения."
)


def _write_assessment(ctx: ToolContext, path: str) -> Assessment:
    """Оценка записи в ``path``: план, настройки gigacrew (.env) и другие секреты — особо.

    План — без риска, настройки gigacrew (.env) — как опасная операция (``-y`` и хук
    не снимают вопрос), другие секреты — с вопросом в режиме «авто» (вид OTHER: правка
    без вопроса — только для обычных файлов проекта).
    """
    if _is_plan_file(ctx, path):
        return Assessment(RiskLevel.SAFE)  # план — данные агента: в любом режиме без вопроса
    try:
        target = resolve_in_root(ctx.root, path)
    except SandboxError:
        return Assessment(RiskLevel.WRITE)  # откажет сама запись
    if target.name.casefold() == CONFIG_FILE_NAME:
        return Assessment(
            RiskLevel.DANGEROUS,
            reason=(
                f"{_rel(ctx, target)} — настройки gigacrew: ключ доступа, адреса API, проверка TLS"
            ),
            kind=ToolKind.OTHER,
        )
    if is_secret_target(ctx.root, target):
        return Assessment(
            RiskLevel.WRITE,
            reason=f"файл с секретами ({_rel(ctx, target)})",
            kind=ToolKind.OTHER,
        )
    return Assessment(RiskLevel.WRITE)


# --------------------------------------------------------------------------- #
# Текст файла: переводы строк, BOM, атомарная запись
# --------------------------------------------------------------------------- #
_BOM = "﻿"
_LF = "\n"
_CRLF = "\r\n"
_CR = "\r"  # одиночный перевод строки старых файлов Mac
_EOL_NAMES = MappingProxyType({_LF: "LF", _CRLF: "CRLF", _CR: "CR"})


@dataclass(frozen=True)
class TextFile:
    r"""Текст файла с переводами строк ``\\n`` и исходное оформление файла.

    Инструменты сопоставляют, сравнивают и правят текст только в виде с ``\\n``, а
    при записи возвращают файлу его переводы строк (``eol``) и BOM — правка одной
    строки CRLF-файла не превращается в дифф на весь файл.
    """

    text: str
    eol: str = "\n"
    bom: bool = False
    mixed: bool = False  # в файле были разные окончания строк — запишутся как ``eol``

    def encode(self, text: str | None = None) -> bytes:
        """Байты для записи: ``text`` (по умолчанию — свой текст) с оформлением файла."""
        body = self.text if text is None else text
        if self.eol != "\n":
            body = body.replace("\n", self.eol)
        prefix = _BOM if self.bom else ""
        return f"{prefix}{body}".encode()

    def eol_note(self) -> str:
        """Пометка для результата: окончания строк файла будут выровнены."""
        if not self.mixed:
            return ""
        eol_name = _EOL_NAMES[self.eol]
        return f" (окончания строк приведены к {eol_name})"


def decode_text(raw: str) -> TextFile:
    r"""Разбор текста: преобладающий перевод строки, BOM, вид с ``\\n``.

    Одиночный ``\\r`` считается переводом строки, только если он преобладает
    (старые файлы Mac); иначе это обычный символ и сохраняется как есть.
    """
    bom = raw.startswith(_BOM)
    if bom:
        raw = raw[1:]
    crlf = raw.count(_CRLF)
    counts = {_LF: raw.count(_LF) - crlf, _CRLF: crlf, _CR: raw.count(_CR) - crlf}
    eol = max((_LF, _CRLF, _CR), key=lambda candidate: (counts[candidate], candidate == _LF))
    if counts[eol] == 0:
        eol = _LF
    if eol == _CR:
        text = raw.replace(_CRLF, _LF).replace(_CR, _LF)
        mixed = counts[_LF] + counts[_CRLF] > 0
    else:
        text = raw.replace(_CRLF, _LF)
        mixed = counts[_LF] > 0 and counts[_CRLF] > 0
    return TextFile(text, eol, bom, mixed)


def write_atomic(path: Path, data: bytes) -> None:
    """Записывает файл целиком или не трогает его вовсе.

    Существующий файл заменяется через временный файл в том же каталоге
    (``os.replace``) с прежними правами и группой: прерывание (Esc, Ctrl+C) или
    нехватка места посреди записи не оставят его пустым или обрезанным.

    Замена создаёт новый inode, поэтому если она потеряла бы метаданные файла —
    чужой владелец, ACL/xattr, жёсткие ссылки — или во временный файл писать
    нельзя (каталог без права записи), файл перезаписывается на месте, как раньше.
    """
    if not path.exists() or not _replace_keeps_metadata(path):
        path.write_bytes(data)
        return
    try:
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    except PermissionError:
        path.write_bytes(data)
        return
    tmp_path = Path(tmp_name)
    try:
        _replace_through_temp(path, data, fd, tmp_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _replace_through_temp(path: Path, payload: bytes, fd: int, tmp_path: Path) -> None:
    """Пишет ``payload`` во временный файл ``tmp_path`` (открыт как ``fd``) и заменяет им файл.

    Если группу файла временному файлу не передать, временный файл удаляется, а файл
    перезаписывается на месте.
    """
    with os.fdopen(fd, "wb") as tmp_file:
        tmp_file.write(payload)
    shutil.copymode(path, tmp_path)
    if not _copy_group(path, tmp_path):
        tmp_path.unlink(missing_ok=True)
        path.write_bytes(payload)
        return
    tmp_path.replace(path)


def _copy_group(path: Path, tmp_path: Path) -> bool:
    """Даёт временному файлу группу файла ``path``; False — сменить группу нельзя."""
    gid = path.stat().st_gid
    if hasattr(os, "chown") and tmp_path.stat().st_gid != gid:
        try:
            os.chown(tmp_path, -1, gid)  # чужая группа файла или каталог с setgid
        except PermissionError:
            return False
    return True


def _replace_keeps_metadata(path: Path) -> bool:
    """Замену через новый inode можно делать без потерь.

    Файл наш, без жёстких ссылок и расширенных атрибутов (ACL хранятся в xattr).
    """
    st = path.stat()
    if st.st_nlink > 1:
        return False
    if hasattr(os, "geteuid") and st.st_uid != os.geteuid():
        return False
    return not (hasattr(os, "listxattr") and _has_xattrs(path))


def _has_xattrs(path: Path) -> bool:
    """У файла есть расширенные атрибуты (ФС без xattr — нет)."""
    try:
        return bool(os.listxattr(path))
    except OSError:
        return False  # ФС без xattr


def _require_regular_file(path: Path, shown: str) -> None:
    """Путь — обычный файл, а не каталог, FIFO, сокет или устройство; иначе ToolError."""
    if path.is_dir():
        raise ToolError(f"Это директория, а не файл: {shown}")
    if not path.is_file():
        raise ToolError(f"Не обычный файл (FIFO, сокет или устройство): {shown}")


def load_existing(path: Path, shown: str) -> TextFile:
    """Существующий файл для правки или перезаписи; иначе ToolError.

    Каталог, FIFO/устройство, бинарный и не-UTF-8 файл не правятся: превью
    выглядело бы как создание нового файла, а запись в UTF-8 испортила бы
    содержимое (или зависла бы на FIFO).
    """
    _require_regular_file(path, shown)
    raw = path.read_bytes()
    if b"\0" in raw[:_BINARY_PROBE]:
        raise ToolError(f"Бинарный файл — write_file/edit_file меняют только текст: {shown}")
    try:
        return decode_text(raw.decode("utf-8"))
    except UnicodeDecodeError:
        raise ToolError(
            f"Файл не в UTF-8 (например, cp1251): {shown}. write_file/edit_file его не "
            "меняют, чтобы не испортить кодировку; перекодировать можно через run_shell "
            "(iconv -f cp1251 -t utf-8).",
        ) from None


def _lines(text: str) -> list[str]:
    r"""Строки с окончаниями — только по ``\\n``.

    ``str.splitlines`` режет ещё и по ``\\x0c``, ``\\x85``, ``\\u2028`` — номера строк
    разошлись бы с ``read_file``, а строки диффа склеились бы.
    """
    parts = text.split("\n")
    lines = [f"{part}\n" for part in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


def _is_plan_file(ctx: ToolContext, path: str) -> bool:
    """Путь ведёт к файлу плана текущего чата (его можно писать без вопроса)."""
    if ctx.plan is None:
        return False
    try:
        return ctx.plan.is_target(resolve_in_root(ctx.root, path))
    except Exception:  # путь вне корня — ошибку сообщит сама запись
        return False


def _plan_written(ctx: ToolContext, path: Path) -> bool:
    """Отмечает изменение плана после записи; True — записан файл плана.

    «План ждёт показа» (``dirty``) — только после правки в режиме плана: отметки о
    выполненных шагах в режиме «авто» — не новый план, иначе в следующем цикле
    планирования harness показал бы старый план сам.
    """
    if ctx.plan is None or not ctx.plan.is_target(path):
        return False
    if ctx.get_mode is None or ctx.get_mode() is PermissionMode.PLAN:
        ctx.plan.dirty = True
    ctx.plan.writes += 1
    return True


def _writable_path(ctx: ToolContext, path: str) -> Path:
    """Путь для записи: внутри корня, не в ``.gigacode/`` и не во внутренностях git.

    В ``.gigacode/`` лежат индекс, история ввода и чаты агента — модель не должна
    их править. Исключение — файл плана текущего чата (``ctx.plan``). Запись в
    ``.git/`` (``config``, хуки) превратила бы правку файла в выполнение команды:
    ``core.fsmonitor`` и ``diff.external`` запускаются уже на ``git status``/``git
    diff``, которые выполняются без подтверждения. Проверка срабатывает и в превью,
    то есть до вопроса о подтверждении.
    """
    target = resolve_in_root(ctx.root, path)
    if is_git_internal(ctx.root, target, path):
        raise ToolError(
            f"Внутренности git ({GIT_DIR_NAME}/) недоступны для записи: {path}. "
            "Для операций с репозиторием используйте инструмент git.",
        )
    if ctx.plan is not None and ctx.plan.is_target(target):
        return target
    data_dir = ctx.workspace.data_dir.resolve()  # .gigacode может быть симлинком
    # is_data_path — ещё и без учёта регистра: на macOS/Windows .GIGACODE — та же папка.
    if target == data_dir or data_dir in target.parents or ctx.workspace.is_data_path(target):
        plan_hint = ""
        if ctx.plan is not None:
            plan_hint = f". Файл плана этого чата: {_rel(ctx, ctx.plan.path)}"
        raise ToolError(
            f"Служебная папка {DATA_DIR_NAME}/ (индекс, история, чаты агента, скиллы) "
            f"недоступна для записи: {path}{plan_hint}",
        )
    return target


_NO_NEWLINE_MARK = r"\ No newline at end of file"


def make_diff(old: str, new: str, path: str) -> str:
    r"""Unified diff двух текстов (с ``\\n``), как у git.

    Последняя строка без перевода строки помечается ``\\ No newline at end of file``,
    а не склеивается со следующей строкой диффа.
    """
    diff = difflib.unified_diff(_lines(old), _lines(new), fromfile=f"a/{path}", tofile=f"b/{path}")
    return "".join(
        line if line.endswith("\n") else f"{line}\n{_NO_NEWLINE_MARK}\n" for line in diff
    )


# Последовательности двойной экранизации: ``\r\n`` — один перевод строки (текст
# сравнивается в виде с ``\n``), ``\\`` — обратная косая черта. Разбор — за один проход
# слева направо: в ``\\n`` экранирована косая черта, а ``n`` — обычная буква.
_ESCAPE_RE = re.compile(r"\\r\\n|\\([nt\"'\\])")
_ESCAPES = MappingProxyType({"n": "\n", "t": "\t", '"': '"', "'": "'", "\\": "\\"})


def _unescaped(match: re.Match[str]) -> str:
    r"""Значение одной экранированной последовательности (``\\r\\n`` — перевод строки)."""
    escaped = match.group(1)
    return _ESCAPES[escaped] if escaped else "\n"


def _unescape(text: str) -> str:
    r"""Разэкранирует частые последовательности (\\n, \\t, \\", \\', \\\\) без условий.

    Используется как один из вариантов-кандидатов при поиске old_string —
    применяется только если разэкранированная форма реально нашлась в файле.
    """
    return _ESCAPE_RE.sub(_unescaped, text)


_LINE_NUM_RE = re.compile(r"^\s*\d+\t")
# Префиксы-номера снимаются, если ими помечена хотя бы такая доля непустых строк.
_NUMBERED_SHARE = 0.6


def _strip_line_numbers(text: str) -> str:
    r"""Удаляет префиксы-номера строк (``N\\t``), если ими помечено большинство строк.

    read_file показывает содержимое с номерами (как ``cat -n``); слабые модели
    иногда копируют эти префиксы прямо в old_string/new_string. Снимаем их,
    чтобы фрагмент совпал с реальным файлом.
    """
    lines = text.split("\n")
    nonempty = [ln for ln in lines if ln.strip()]
    if not nonempty:
        return text
    marked = sum(1 for ln in nonempty if _LINE_NUM_RE.match(ln))
    if marked < max(2, _NUMBERED_SHARE * len(nonempty)):
        return text
    return "\n".join(_LINE_NUM_RE.sub("", ln) for ln in lines)


_NUMBERED_LINE_RE = re.compile(r"^\d+\t")


def repair_escaped_content(content: str) -> tuple[str, bool]:
    r"""Чинит «двойную экранизацию», которую иногда выдаёт модель.

    Слабые модели порой кладут в write_file одну физическую строку, где все
    переносы — это литералы ``\\n`` (а кавычки — ``\\"``), иногда ещё и с
    префиксами номеров строк (``1\\timport ...``), скопированными из подсказки.
    Чинится только однозначный случай: ни одного настоящего переноса, не меньше двух
    литеральных ``\\n`` и либо текст кончается литеральным ``\\n`` (файл с переводом
    строки в конце, экранированный ещё раз), либо все строки — с номерами. Однострочный
    код или JSON без такого хвоста (``print("a\\nb\\nc")``) верен как есть и не меняется.

    Возвращает (исправленный_текст, был_ли_ремонт).
    """
    if "\n" in content or content.count(r"\n") < 2:
        return content, False
    fixed = _unescape(content)
    lines = fixed.split("\n")
    nonempty = [ln for ln in lines if ln.strip()]
    if nonempty and all(_NUMBERED_LINE_RE.match(ln) for ln in nonempty):
        return "\n".join(_NUMBERED_LINE_RE.sub("", ln) for ln in lines), True
    if not fixed.endswith("\n"):
        return content, False
    return fixed, True


def _numbered(number: int, width: int, line: str) -> str:
    """Строка с номером, выровненным вправо по ширине ``width``, через табуляцию."""
    column = str(number).rjust(width)
    return f"{column}\t{line}"


def _numbered_excerpt(text: str, max_lines: int = 60) -> str:
    """Содержимое файла с номерами строк (для подсказки при неудачном edit)."""
    lines = [line.rstrip("\n") for line in _lines(text)]
    shown = lines[:max_lines]
    width = len(str(len(shown)))
    body = "\n".join(_numbered(number, width, line) for number, line in enumerate(shown, start=1))
    if len(lines) > max_lines:
        rest = len(lines) - max_lines
        body = f"{body}\n… (ещё {rest} строк)"
    return body


def _leading(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" \t"))]


IndentRule = Callable[[str], "str | None"]


def _strip_suffix(text: str, suffix: str) -> str | None:
    """``text`` без окончания ``suffix``; None — ``text`` им не кончается."""
    if not text.endswith(suffix):
        return None
    return text[: len(text) - len(suffix)]


def _single(candidates: set[str | None]) -> str | None:
    """Единственный вариант; None — вариантов несколько, нет ни одного или среди них None."""
    if len(candidates) == 1 and None not in candidates:
        return candidates.pop()
    return None


def _add_prefix(prefix: str, indent: str) -> str | None:
    """Отступ файла: к отступу шаблона добавлен общий префикс."""
    return prefix + indent


def _remove_prefix(extra: str, indent: str) -> str | None:
    """Отступ файла: у отступа шаблона убран лишний общий префикс; None — его там нет."""
    return indent[len(extra) :] if indent.startswith(extra) else None


def _scale_units(pattern_unit: str, file_unit: str, indent: str) -> str | None:
    """Отступ файла: единицы отступа шаблона заменены единицами файла; None — не кратен."""
    repeats = len(indent) // len(pattern_unit)
    return file_unit * repeats if indent == pattern_unit * repeats else None


def _unit_rule(mapping: dict[str, str]) -> IndentRule | None:
    """Соответствие отступов кратными единицами (пробелы ↔ табы); None — не оно."""
    units = [
        (pat_indent, file_indent)
        for pat_indent, file_indent in mapping.items()
        if pat_indent and file_indent
    ]
    if not units:
        return None
    pattern_unit, file_unit = min(units, key=lambda unit: len(unit[0]))
    for pat_indent, file_indent in mapping.items():
        repeats = len(pat_indent) // len(pattern_unit)
        if pat_indent != pattern_unit * repeats or file_indent != file_unit * repeats:
            return None
    return functools.partial(_scale_units, pattern_unit, file_unit)


def _indent_rule(pairs: list[tuple[str, str]]) -> IndentRule | None:
    """Как перевести отступ из фрагмента модели в отступ файла.

    ``pairs`` — (отступ строки шаблона, отступ совпавшей строки файла). Годится
    единый добавленный префикс (модель потеряла общий отступ), единый убранный
    (лишний общий отступ) или согласованное соответствие отступов (пробелы ↔ табы,
    в том числе кратными единицами). Иначе None — правку не угадываем.
    """
    added = _single({_strip_suffix(file_indent, pat_indent) for pat_indent, file_indent in pairs})
    if added is not None:
        return functools.partial(_add_prefix, added)
    removed = _single(set(itertools.starmap(_strip_suffix, pairs)))
    if removed is not None:
        return functools.partial(_remove_prefix, removed)
    mapping: dict[str, str] = {}
    for pat_indent, file_indent in pairs:
        if mapping.setdefault(pat_indent, file_indent) != file_indent:
            return None
    by_units = _unit_rule(mapping)
    return mapping.get if by_units is None else by_units


@dataclass(frozen=True)
class _Match:
    """Несторогое совпадение: где заменять и как подогнать ``new_string``."""

    start: int
    end: int
    lead: int  # сколько пустых строк срезано с начала шаблона
    trail: int  # … и с конца
    indent: IndentRule | None = None  # None — отступы шаблона совпали с файлом

    def adapt(self, new_string: str) -> str | None:
        """``new_string`` под совпавший блок; None — отступы не перевести."""
        lines = new_string.split("\n")
        for _ in range(self.lead):
            if len(lines) > 1 and not lines[0].strip():
                lines.pop(0)
        for _ in range(self.trail):
            if len(lines) > 1 and not lines[-1].strip():
                lines.pop()
        if self.indent is None:
            return "\n".join(lines)
        out = []
        for line in lines:
            if not line.strip():
                out.append(line)
                continue
            indent = self.indent(_leading(line))
            if indent is None:
                return None
            out.append(indent + line.lstrip(" \t"))
        return "\n".join(out)


def _trim_blank_lines(pattern: str) -> tuple[list[str], int, int]:
    """Строки шаблона без пустых строк по краям: (строки, срезано в начале, срезано в конце)."""
    pat_lines = pattern.split("\n")
    lead = 0
    while pat_lines and not pat_lines[0].strip():
        pat_lines.pop(0)
        lead += 1
    trail = 0
    while pat_lines and not pat_lines[-1].strip():
        pat_lines.pop()
        trail += 1
    return pat_lines, lead, trail


def _normalized(line: str, *, strict: bool) -> str:
    """Строка для сравнения: strict — без хвостовых пробелов, иначе — без пробелов по краям."""
    return line.rstrip() if strict else line.strip()


def _unique_block(file_lines: list[str], pat_lines: list[str], *, strict: bool) -> int | None:
    """Номер первой строки блока, совпавшего с шаблоном; None — совпадений нет или несколько."""
    target = [_normalized(line, strict=strict) for line in pat_lines]
    size = len(pat_lines)
    hits = [
        first
        for first in range(len(file_lines) - size + 1)
        if [_normalized(line, strict=strict) for line in file_lines[first : first + size]] == target
    ]
    return hits[0] if len(hits) == 1 else None


def _block_span(raw: list[str], first: int, size: int) -> tuple[int, int]:
    r"""Границы блока из ``size`` строк с ``first`` в тексте (конец — без последнего ``\\n``)."""
    last = first + size - 1
    start = sum(len(line) for line in raw[:first])
    before_last = sum(len(line) for line in raw[:last])
    end = before_last + len(raw[last].rstrip("\n"))
    return start, end


def _indent_pairs(pat_lines: list[str], block: list[str]) -> list[tuple[str, str]]:
    """Пары (отступ строки шаблона, отступ строки файла) для непустых строк шаблона."""
    return [
        (_leading(pat), _leading(line))
        for pat, line in zip(pat_lines, block, strict=True)
        if pat.strip()
    ]


def _tolerant_find(text: str, pattern: str) -> _Match | None:
    """Ищет блок строк, совпадающий с pattern с точностью до пробелов/отступов.

    Возвращает совпадение при ЕДИНСТВЕННОМ вхождении, иначе None. Используется
    как запасной вариант, когда точного совпадения нет (модель часто слегка путает
    отступы/хвостовые пробелы). Блок заменяется целыми строками, поэтому при
    совпадении «без учёта отступов» ``new_string`` переотступается под файл.
    """
    raw = _lines(text)
    if not raw:
        return None
    file_lines = [line.rstrip("\n") for line in raw]
    pat_lines, lead, trail = _trim_blank_lines(pattern)
    if not pat_lines:
        return None

    # От более строгой нормализации (только хвостовые пробелы) к более мягкой
    # (полный strip — игнор отступов). Берём первый режим с уникальным совпадением.
    exact = _unique_block(file_lines, pat_lines, strict=True)
    if exact is not None:
        start, end = _block_span(raw, exact, len(pat_lines))
        return _Match(start, end, lead, trail)
    loose = _unique_block(file_lines, pat_lines, strict=False)
    if loose is None:
        return None
    rule = _indent_rule(_indent_pairs(pat_lines, file_lines[loose : loose + len(pat_lines)]))
    if rule is None:
        return None
    start, end = _block_span(raw, loose, len(pat_lines))
    return _Match(start, end, lead, trail, rule)


# --------------------------------------------------------------------------- #
# read_file
# --------------------------------------------------------------------------- #
class ReadFileParams(BaseModel):
    path: str = Field(description="File path relative to the project root")
    start_line: int | None = Field(default=None, description="First line (1-based), inclusive")
    end_line: int | None = Field(default=None, description="Last line (1-based), inclusive")


class ReadFileTool(Tool[ReadFileParams]):
    """Инструмент read_file: текст файла с номерами строк, по частям."""

    name = "read_file"
    read_only = True
    description = (
        "Reads a text file. Returns the text with line numbers. Pass a line range "
        "(start_line/end_line) to read only the part you need — use file_outline or "
        f"search to find it. At most {MAX_READ_LINES} lines per call; to continue, pass "
        "start_line."
    )
    Params = ReadFileParams

    @override
    def run(self, params: ReadFileParams, ctx: ToolContext) -> ToolResult:
        """Читает файл и возвращает выбранные строки с номерами."""
        path = resolve_in_root(ctx.root, params.path)
        _check_readable(ctx, path, params.path)
        with path.open("rb") as fh:
            if b"\0" in fh.read(_BINARY_PROBE):
                raise ToolError(f"Бинарный файл, чтение не поддерживается: {params.path}")

        start = max(params.start_line or 1, 1)
        encoding_note = ""
        try:
            selected, total, stopped_at = _select_lines(path, start, params.end_line, "utf-8")
        except UnicodeDecodeError:
            # Чаще всего — старый русский текст в cp1251: показать его полезнее отказа.
            selected, total, stopped_at = _select_lines(path, start, params.end_line, "cp1251")
            encoding_note = f"{CP1251_NOTE}\n"

        numbered = _numbered_body(selected, total) + _range_note(
            selected,
            total,
            start,
            stopped_at,
        )
        encoding = " · cp1251" if encoding_note else ""
        return ToolResult(
            content=encoding_note + numbered,
            summary=f"прочитан {_rel(ctx, path)} ({len(selected)} строк){encoding}",
        )


CP1251_NOTE = "(файл не в UTF-8 — показан как cp1251; edit_file и write_file такие файлы не меняют)"

_NumberedLine = tuple[int, str]


def _check_readable(ctx: ToolContext, path: Path, shown: str) -> None:
    """Файл можно читать: он есть, обычный, не с секретами и не слишком большой."""
    if not path.exists():
        raise ToolError(f"Файл не найден: {shown}")
    # FIFO/сокет/устройство — чтение может заблокироваться.
    _require_regular_file(path, shown)
    if is_secret_target(ctx.root, path):
        raise ToolError(SECRET_READ_NOTE.format(path=shown))
    if path.stat().st_size > MAX_READ_BYTES:
        raise ToolError(
            f"Файл слишком большой (>{MAX_READ_BYTES} байт). Используйте search_content.",
        )


def _numbered_body(selected: list[_NumberedLine], total: int) -> str:
    """Выбранные строки с номерами; если строк нет — пометка, почему."""
    if not selected:
        return "(пусто)" if total == 0 else f"(нет строк в диапазоне; всего строк: {total})"
    width = len(str(selected[-1][0]))
    return "\n".join(_numbered(number, width, line) for number, line in selected)


def _range_note(
    selected: list[_NumberedLine],
    total: int,
    start: int,
    stopped_at: int | None,
) -> str:
    """Какая часть файла показана и откуда продолжить; пусто — показан весь файл."""
    first = selected[0][0] if selected else start
    if stopped_at is not None:
        shown_last = stopped_at - 1
        return (
            f"\n… показаны строки {first}–{shown_last} из {total}. "
            f"Продолжение: start_line={stopped_at}."
        )
    if not selected:
        return ""
    last = selected[-1][0]
    if first <= 1 and last >= total:
        return ""
    note = f"\n… показаны строки {first}–{last} из {total}."
    if last < total:
        next_line = last + 1
        note = f"{note} Продолжение: start_line={next_line}."
    return note


def _clip_line(raw: str) -> str:
    """Строка файла без перевода строки, обрезанная до ``MAX_LINE_CHARS`` символов."""
    line = raw.rstrip("\r\n")
    if len(line) > MAX_LINE_CHARS:
        head = line[:MAX_LINE_CHARS]
        return f"{head} …[строка обрезана]"
    return line


def _select_lines(
    path: Path,
    start: int,
    end: int | None,
    encoding: str,
) -> tuple[list[_NumberedLine], int, int | None]:
    """Строки ``start..end`` в пределах лимитов чтения.

    Возвращает (выбранные (номер, строка), всего строк в файле, первая строка,
    не поместившаяся в лимит, или None). Для UTF-8 — ``UnicodeDecodeError`` на
    неверных байтах, иначе неверные байты заменяются.
    """
    selected: list[_NumberedLine] = []
    budget = MAX_READ_CHARS
    total = 0
    stopped_at: int | None = None
    errors = "strict" if encoding == "utf-8" else "replace"
    with path.open(encoding=encoding, errors=errors) as fh:
        for number, raw in enumerate(fh, start=1):
            total = number
            if number < start or (end is not None and number > end):
                continue
            if stopped_at is not None:
                continue  # досчитываем общее число строк
            line = _clip_line(raw)
            if len(selected) >= MAX_READ_LINES or len(line) + 1 > budget:
                stopped_at = number
                continue
            selected.append((number, line))
            budget -= len(line) + 1
    return selected, total, stopped_at


# --------------------------------------------------------------------------- #
# write_file
# --------------------------------------------------------------------------- #
class WriteFileParams(BaseModel):
    path: str = Field(description="File path relative to the project root")
    content: str = Field(description="The complete new content of the file")


class WriteFileTool(Tool[WriteFileParams]):
    """Инструмент write_file: создаёт файл или перезаписывает его целиком."""

    name = "write_file"
    description = (
        "Creates a new file or completely overwrites an existing one with the given "
        "content. Creates parent directories when needed. For changes to an existing "
        "file prefer edit_file."
    )
    Params = WriteFileParams
    kind = ToolKind.EDIT

    @override
    def assess(self, params: WriteFileParams, ctx: ToolContext) -> Assessment:
        """Оценка записи: план, настройки gigacrew и секреты — особо."""
        return _write_assessment(ctx, params.path)

    @override
    def risk(self, params: WriteFileParams, ctx: ToolContext) -> RiskLevel:
        """Уровень риска — из оценки записи."""
        return self.assess(params, ctx).risk

    @override
    def preview(self, params: WriteFileParams, ctx: ToolContext) -> Display | None:
        """Дифф прежнего и нового содержимого файла."""
        _, old, new, _ = self._prepare(params, ctx)
        diff = make_diff(old.text if old else "", new.text, params.path)
        if not diff:
            diff = "(новый пустой файл)" if old is None else "(без изменений)"
        return Display(diff, kind="diff", title=f"{params.path}{new.eol_note()}")

    @override
    def run(self, params: WriteFileParams, ctx: ToolContext) -> ToolResult:
        """Записывает файл (каталоги создаются при необходимости)."""
        target, old, new, repaired = self._prepare(params, ctx)
        if ctx.plan is not None and ctx.plan.is_target(target):
            ctx.workspace.ensure_data_dir()  # .gitignore папки — до первого файла в ней
        target.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(target, new.encode())
        verb = "создан" if old is None else "перезаписан"
        shown = "план" if _plan_written(ctx, target) else _rel(ctx, target)
        line_count = len(_lines(new.text))
        repair_note = " (автокоррекция экранирования)" if repaired else ""
        note = repair_note + new.eol_note()
        return ToolResult(
            content=f"Файл {verb}: {params.path} ({line_count} строк).{note}",
            summary=f"{verb} {shown}{note}",
            display=Display(
                make_diff(old.text if old else "", new.text, params.path),
                kind="diff",
                title=params.path,
            ),
        )

    def _prepare(
        self,
        params: WriteFileParams,
        ctx: ToolContext,
    ) -> tuple[Path, TextFile | None, TextFile, bool]:
        """Путь, прежний текст и новый текст с оформлением, с которым он запишется.

        Существующий файл сохраняет свои переводы строк и BOM; новый — как прислано.
        """
        target = _writable_path(ctx, params.path)
        old = load_existing(target, params.path) if target.exists() else None
        content, repaired = repair_escaped_content(params.content)
        new = decode_text(content)
        if old is not None:
            new = TextFile(new.text, old.eol, old.bom, old.mixed)
        return target, old, new, repaired


# --------------------------------------------------------------------------- #
# edit_file
# --------------------------------------------------------------------------- #
class EditFileParams(BaseModel):
    path: str = Field(description="File path relative to the project root")
    old_string: str = Field(description="The exact fragment to replace (must be present)")
    new_string: str = Field(description="The replacement text")
    replace_all: bool = Field(
        default=False,
        description="Replace all occurrences instead of exactly one",
    )


def _as_sent(fragment: str) -> str:
    """Фрагмент как прислан моделью."""
    return fragment


def _unescape_and_strip(fragment: str) -> str:
    """Фрагмент без экранизации и без префиксов-номеров строк."""
    return _strip_line_numbers(_unescape(fragment))


# Слабые модели часто слегка искажают old_string: экранируют переносы
# (литералы \n, \"), либо копируют префиксы-номера строк (N\t) прямо из
# вывода read_file. Пробуем набор согласованных преобразований к ОБОИМ
# фрагментам и берём первое, которое реально находится в файле — это
# безопасно: преобразованная форма применяется, только если совпала.
_FRAGMENT_TRANSFORMS: tuple[Callable[[str], str], ...] = (
    _as_sent,  # как прислано
    _unescape,  # снять экранизацию \n/\"
    _strip_line_numbers,  # снять префиксы номеров строк
    _unescape_and_strip,  # и то, и другое
)


def _check_edit_fragments(old_param: str, new_param: str) -> None:
    """Заменяемый фрагмент не пуст и отличается от замены; иначе ToolError."""
    if not old_param:
        # "" нашлось бы между каждой парой символов: replace_all вставил бы
        # new_string повсюду.
        raise ToolError(
            "old_string пуст — укажите заменяемый фрагмент. Чтобы создать файл "
            "или заменить его целиком, используйте write_file.",
        )
    if old_param == new_param:
        raise ToolError("old_string и new_string совпадают — нечего менять.")


def _replace_fragment(
    text: str,
    old_string: str,
    new_string: str,
    *,
    replace_all: bool,
) -> tuple[str, int] | None:
    """Новый текст и число замен; None — фрагмент не нашёлся ни точно, ни с поправкой."""
    # Тир 1: точное совпадение.
    count = text.count(old_string)
    if count == 1 or (count > 1 and replace_all):
        return text.replace(old_string, new_string), count
    if count > 1 and not replace_all:
        raise ToolError(
            f"old_string встречается {count} раз. Добавьте контекста, чтобы "
            "фрагмент стал уникальным, или укажите replace_all=true.",
        )
    # Тир 2: совпадение с поправкой на пробелы/отступы (одиночная замена).
    match = _tolerant_find(text, old_string)
    if match is None:
        return None
    replacement = match.adapt(new_string)
    if replacement is None:
        return None
    return text[: match.start] + replacement + text[match.end :], 1


def _apply_edit(
    text: str,
    old_param: str,
    new_param: str,
    *,
    replace_all: bool,
) -> tuple[str, int] | None:
    """Правка с первым преобразованием фрагментов, которое нашлось в тексте; None — ни одно."""
    seen: set[str] = set()
    for transform in _FRAGMENT_TRANSFORMS:
        old_string = transform(old_param)
        if not old_string or old_string in seen:
            continue
        seen.add(old_string)
        edited = _replace_fragment(
            text,
            old_string,
            transform(new_param),
            replace_all=replace_all,
        )
        if edited is not None:
            return edited
    return None


def _not_found_error(ctx: ToolContext, path: Path, text: str) -> ToolError:
    """Ошибка «old_string не найден» с содержимым файла, чтобы модель скопировала точно.

    Файл с секретами не показываем: ошибка превью уходит модели до вопроса пользователю.
    """
    missing = "old_string не найден в файле (ни точно, ни с поправкой на пробелы)."
    if is_secret_target(ctx.root, path):
        return ToolError(f"{missing} Это файл с секретами — его содержимое не показывается.")
    excerpt = _numbered_excerpt(text)
    return ToolError(f"{missing} Скопируйте фрагмент дословно из содержимого ниже:\n\n{excerpt}")


class EditFileTool(Tool[EditFileParams]):
    """Инструмент edit_file: заменяет фрагмент файла на месте."""

    name = "edit_file"
    description = (
        "Edits a file in place: replaces old_string with new_string. old_string must "
        "occur exactly once (otherwise add surrounding context, or set replace_all=true "
        "to replace every occurrence). Matching tolerates minor differences in "
        "indentation and trailing spaces, but copy the fragment verbatim from the file "
        "content you have read."
    )
    Params = EditFileParams
    kind = ToolKind.EDIT

    @override
    def assess(self, params: EditFileParams, ctx: ToolContext) -> Assessment:
        """Оценка записи: план, настройки gigacrew и секреты — особо."""
        return _write_assessment(ctx, params.path)

    @override
    def risk(self, params: EditFileParams, ctx: ToolContext) -> RiskLevel:
        """Уровень риска — из оценки записи."""
        return self.assess(params, ctx).risk

    @override
    def preview(self, params: EditFileParams, ctx: ToolContext) -> Display | None:
        """Дифф правки."""
        _, source, new_text, _ = self._compute(params, ctx)
        return Display(
            make_diff(source.text, new_text, params.path),
            kind="diff",
            title=f"{params.path}{source.eol_note()}",
        )

    @override
    def run(self, params: EditFileParams, ctx: ToolContext) -> ToolResult:
        """Применяет правку и записывает файл."""
        target, source, new_text, count = self._compute(params, ctx)
        write_atomic(target, source.encode(new_text))
        replaced = count if params.replace_all else 1
        note = source.eol_note()
        shown = "план" if _plan_written(ctx, target) else _rel(ctx, target)
        return ToolResult(
            content=f"Отредактирован {params.path}: заменено вхождений — {replaced}.{note}",
            summary=f"изменён {shown}{note}",
            display=Display(
                make_diff(source.text, new_text, params.path),
                kind="diff",
                title=params.path,
            ),
        )

    def _compute(self, params: EditFileParams, ctx: ToolContext) -> tuple[Path, TextFile, str, int]:
        """Путь, файл, новый текст и число замен; ToolError — правку не применить."""
        target = _writable_path(ctx, params.path)
        if not target.exists():
            raise ToolError(f"Файл не найден: {params.path}")
        source = load_existing(target, params.path)
        # Сравнение и правка — в виде с \n; переводы строк файла вернёт запись.
        old_param = params.old_string.replace("\r\n", "\n")
        new_param = params.new_string.replace("\r\n", "\n")
        _check_edit_fragments(old_param, new_param)
        edited = _apply_edit(source.text, old_param, new_param, replace_all=params.replace_all)
        if edited is None:
            # Не найдено — даём модели контекст файла, чтобы скопировать точно.
            raise _not_found_error(ctx, target, source.text)
        new_text, count = edited
        return target, source, new_text, count


# --------------------------------------------------------------------------- #
# list_dir
# --------------------------------------------------------------------------- #
MAX_LIST_ENTRIES = 500


class ListDirParams(BaseModel):
    path: str = Field(default=".", description="Directory path (the project root by default)")


class ListDirTool(Tool[ListDirParams]):
    """Инструмент list_dir: содержимое каталога."""

    name = "list_dir"
    read_only = True
    description = (
        "Lists files and subdirectories (including hidden ones) of a directory. For an "
        "overview with languages and definitions use file_outline."
    )
    Params = ListDirParams

    @override
    def run(self, params: ListDirParams, ctx: ToolContext) -> ToolResult:
        """Перечисляет файлы и подкаталоги: сначала каталоги, затем файлы."""
        directory = resolve_in_root(ctx.root, params.path)
        if not directory.is_dir():
            raise ToolError(f"Не директория: {params.path}")
        entries = sorted(
            directory.iterdir(),
            key=lambda entry: (entry.is_file(), entry.name.lower()),
        )
        lines = [
            f"{entry.name}/" if entry.is_dir() else entry.name
            for entry in entries[:MAX_LIST_ENTRIES]
        ]
        body = "\n".join(lines) if lines else "(пусто)"
        if len(entries) > MAX_LIST_ENTRIES:
            body = f"{body}\n… показано {MAX_LIST_ENTRIES} из {len(entries)}"
        return ToolResult(
            content=body,
            summary=f"{_rel(ctx, directory)}: {len(entries)} элементов",
        )


# --------------------------------------------------------------------------- #
# find_files
# --------------------------------------------------------------------------- #
MAX_FIND_RESULTS = 1000
_DEFAULT_FIND_RESULTS = 200


class FindFilesParams(BaseModel):
    pattern: str = Field(
        description="Glob pattern: '*.py' — by name in any directory, 'src/**/*.ts' — by path "
        "from the root",
    )
    max_results: int = Field(
        default=_DEFAULT_FIND_RESULTS,
        description="Maximum number of results",
    )


def _check_pattern(pattern: str) -> None:
    """Шаблон должен быть относительным и не подниматься выше корня проекта."""
    if not pattern:
        raise ToolError("Пустой шаблон.")
    if PurePosixPath(pattern).is_absolute() or PureWindowsPath(pattern).is_absolute():
        raise ToolError("Шаблон должен быть относительным путём от корня проекта.")
    if ".." in re.split(r"[\\/]", pattern):
        raise ToolError("Шаблон не может содержать '..' — поиск только внутри проекта.")


class FindFilesTool(Tool[FindFilesParams]):
    """Инструмент find_files: поиск файлов проекта по glob-шаблону."""

    name = "find_files"
    read_only = True
    description = (
        "Finds files by a glob pattern (recursively, inside the project). A pattern "
        "without '/' is matched against the file name, with '/' — against the path from "
        "the root ('**' — any directories). Skips service directories (.git, "
        "node_modules, etc.) and files ignored by .gitignore."
    )
    Params = FindFilesParams

    @override
    def run(self, params: FindFilesParams, ctx: ToolContext) -> ToolResult:
        """Находит файлы проекта, подходящие под шаблон (пути от корня, по алфавиту)."""
        pattern = params.pattern.strip()
        _check_pattern(pattern)
        limit = min(max(params.max_results, 1), MAX_FIND_RESULTS)
        root = ctx.root
        relative = (path.relative_to(root).as_posix() for path in walk_files(root))
        matches = sorted(rel for rel in relative if glob_match(rel, pattern))
        shown = matches[:limit]
        body = "\n".join(shown) if shown else "(ничего не найдено)"
        if len(matches) > limit:
            body = f"{body}\n… показано {limit} из {len(matches)}; уточните шаблон"
        return ToolResult(content=body, summary=f"найдено файлов: {len(matches)}")
