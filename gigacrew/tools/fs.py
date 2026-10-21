"""Файловые инструменты: чтение, запись, точечное редактирование, листинг, поиск файлов."""

from __future__ import annotations

import difflib
import os
import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from pydantic import BaseModel, Field

from gigacrew.permissions import PermissionMode, ToolKind
from gigacrew.project.files import GIT_DIR_NAME, glob_match, is_git_internal, walk_files
from gigacrew.project.workspace import DATA_DIR_NAME
from gigacrew.security import RiskLevel, resolve_in_root
from gigacrew.tools.base import Display, Tool, ToolContext, ToolError, ToolResult

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
# Текст файла: переводы строк, BOM, атомарная запись
# --------------------------------------------------------------------------- #
_BOM = "﻿"
_EOL_NAMES = {"\n": "LF", "\r\n": "CRLF", "\r": "CR"}


@dataclass(frozen=True)
class TextFile:
    """Текст файла с переводами строк ``\\n`` и исходное оформление файла.

    Инструменты сопоставляют, сравнивают и правят текст только в виде с ``\\n``, а
    при записи возвращают файлу его переводы строк (``eol``) и BOM — правка одной
    строки CRLF-файла не превращается в дифф на весь файл.
    """

    text: str
    eol: str = "\n"
    bom: bool = False
    mixed: bool = False  # в файле были разные окончания строк — запишутся как ``eol``

    def encode(self, text: str | None = None) -> bytes:
        body = self.text if text is None else text
        if self.eol != "\n":
            body = body.replace("\n", self.eol)
        return ((_BOM if self.bom else "") + body).encode("utf-8")

    def eol_note(self) -> str:
        """Пометка для результата: окончания строк файла будут выровнены."""
        if not self.mixed:
            return ""
        return f" (окончания строк приведены к {_EOL_NAMES[self.eol]})"


def decode_text(raw: str) -> TextFile:
    """Разбор текста: преобладающий перевод строки, BOM, вид с ``\\n``.

    Одиночный ``\\r`` считается переводом строки, только если он преобладает
    (старые файлы Mac); иначе это обычный символ и сохраняется как есть.
    """
    bom = raw.startswith(_BOM)
    if bom:
        raw = raw[1:]
    crlf = raw.count("\r\n")
    counts = {"\n": raw.count("\n") - crlf, "\r\n": crlf, "\r": raw.count("\r") - crlf}
    eol = max(("\n", "\r\n", "\r"), key=lambda k: (counts[k], k == "\n"))
    if counts[eol] == 0:
        eol = "\n"
    if eol == "\r":
        text = raw.replace("\r\n", "\n").replace("\r", "\n")
        mixed = counts["\n"] + counts["\r\n"] > 0
    else:
        text = raw.replace("\r\n", "\n")
        mixed = counts["\n"] > 0 and counts["\r\n"] > 0
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
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    except PermissionError:
        path.write_bytes(data)
        return
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        shutil.copymode(path, tmp)
        gid = path.stat().st_gid
        if hasattr(os, "chown") and os.stat(tmp).st_gid != gid:
            try:
                os.chown(tmp, -1, gid)  # чужая группа файла или каталог с setgid
            except PermissionError:
                Path(tmp).unlink(missing_ok=True)
                path.write_bytes(data)
                return
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _replace_keeps_metadata(path: Path) -> bool:
    """Замену через новый inode можно делать без потерь: файл наш, без жёстких
    ссылок и расширенных атрибутов (ACL хранятся в xattr)."""
    st = path.stat()
    if st.st_nlink > 1:
        return False
    if hasattr(os, "geteuid") and st.st_uid != os.geteuid():
        return False
    if hasattr(os, "listxattr"):
        try:
            if os.listxattr(path):
                return False
        except OSError:
            pass  # ФС без xattr
    return True


def load_existing(path: Path, shown: str) -> TextFile:
    """Существующий файл для правки или перезаписи; иначе ToolError.

    Каталог, FIFO/устройство, бинарный и не-UTF-8 файл не правятся: превью
    выглядело бы как создание нового файла, а запись в UTF-8 испортила бы
    содержимое (или зависла бы на FIFO).
    """
    if path.is_dir():
        raise ToolError(f"Это директория, а не файл: {shown}")
    if not path.is_file():
        raise ToolError(f"Не обычный файл (FIFO, сокет или устройство): {shown}")
    data = path.read_bytes()
    if b"\0" in data[:_BINARY_PROBE]:
        raise ToolError(f"Бинарный файл — write_file/edit_file меняют только текст: {shown}")
    try:
        return decode_text(data.decode("utf-8"))
    except UnicodeDecodeError:
        raise ToolError(
            f"Файл не в UTF-8 (например, cp1251): {shown}. write_file/edit_file его не "
            "меняют, чтобы не испортить кодировку; перекодировать можно через run_shell "
            "(iconv -f cp1251 -t utf-8)."
        ) from None


def _lines(text: str) -> list[str]:
    """Строки с окончаниями — только по ``\\n``.

    ``str.splitlines`` режет ещё и по ``\\x0c``, ``\\x85``, ``\\u2028`` — номера строк
    разошлись бы с ``read_file``, а строки диффа склеились бы.
    """
    parts = text.split("\n")
    lines = [part + "\n" for part in parts[:-1]]
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


def _plan_written(ctx: ToolContext, p: Path) -> bool:
    """Отмечает изменение плана после записи; True — записан файл плана.

    «План ждёт показа» (``dirty``) — только после правки в режиме плана: отметки о
    выполненных шагах в режиме «авто» — не новый план, иначе в следующем цикле
    планирования harness показал бы старый план сам.
    """
    if ctx.plan is None or not ctx.plan.is_target(p):
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
    p = resolve_in_root(ctx.root, path)
    if is_git_internal(ctx.root, p, path):
        raise ToolError(
            f"Внутренности git ({GIT_DIR_NAME}/) недоступны для записи: {path}. "
            "Для операций с репозиторием используйте инструмент git."
        )
    if ctx.plan is not None and ctx.plan.is_target(p):
        return p
    data_dir = ctx.workspace.data_dir.resolve()  # .gigacode может быть симлинком
    # is_data_path — ещё и без учёта регистра: на macOS/Windows .GIGACODE — та же папка.
    if p == data_dir or data_dir in p.parents or ctx.workspace.is_data_path(p):
        raise ToolError(
            f"Служебная папка {DATA_DIR_NAME}/ (индекс, история, чаты агента, скиллы) "
            f"недоступна для записи: {path}"
            + (
                f". Файл плана этого чата: {_rel(ctx, ctx.plan.path)}"
                if ctx.plan is not None
                else ""
            )
        )
    return p


_NO_NEWLINE = "\\ No newline at end of file\n"


def make_diff(old: str, new: str, path: str) -> str:
    """Unified diff двух текстов (с ``\\n``), как у git: последняя строка без
    перевода строки помечается ``\\ No newline at end of file``, а не склеивается
    со следующей строкой диффа."""
    diff = difflib.unified_diff(_lines(old), _lines(new), fromfile=f"a/{path}", tofile=f"b/{path}")
    return "".join(line if line.endswith("\n") else f"{line}\n{_NO_NEWLINE}" for line in diff)


def _unescape_simple(s: str) -> str:
    """Разэкранирует частые последовательности (\\n, \\t, \\", \\') без условий.

    Используется как один из вариантов-кандидатов при поиске old_string —
    применяется только если разэкранированная форма реально нашлась в файле.
    """
    return (
        s.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace('\\"', '"')
        .replace("\\'", "'")
    )


_LINE_NUM_RE = re.compile(r"^\s*\d+\t")


def _strip_line_numbers(s: str) -> str:
    """Удаляет префиксы-номера строк (``N\\t``), если ими помечено большинство строк.

    read_file показывает содержимое с номерами (как ``cat -n``); слабые модели
    иногда копируют эти префиксы прямо в old_string/new_string. Снимаем их,
    чтобы фрагмент совпал с реальным файлом.
    """
    lines = s.split("\n")
    nonempty = [ln for ln in lines if ln.strip()]
    if not nonempty:
        return s
    marked = sum(1 for ln in nonempty if _LINE_NUM_RE.match(ln))
    if marked < max(2, 0.6 * len(nonempty)):
        return s
    return "\n".join(_LINE_NUM_RE.sub("", ln) for ln in lines)


def repair_escaped_content(content: str) -> tuple[str, bool]:
    """Чинит «двойную экранизацию», которую иногда выдаёт модель.

    Слабые модели порой кладут в write_file одну физическую строку, где все
    переносы — это литералы ``\\n`` (а кавычки — ``\\"``), иногда ещё и с
    префиксами номеров строк (``1\\timport ...``), скопированными из подсказки.
    Такой контент однозначно битый: ноль настоящих переносов при множестве
    литеральных ``\\n``. В этом случае разэкранируем и снимаем нумерацию.

    Возвращает (исправленный_текст, был_ли_ремонт).
    """
    if content.count("\n") == 0 and content.count("\\n") >= 2:
        fixed = (
            content.replace("\\r\\n", "\n")
            .replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace('\\"', '"')
            .replace("\\'", "'")
            .replace("\\\\", "\\")
        )
        lines = fixed.split("\n")
        nonempty = [ln for ln in lines if ln.strip()]
        if nonempty and all(re.match(r"^\d+\t", ln) for ln in nonempty):
            lines = [re.sub(r"^\d+\t", "", ln) for ln in lines]
            fixed = "\n".join(lines)
        return fixed, True
    return content, False


def _numbered_excerpt(text: str, max_lines: int = 60) -> str:
    """Содержимое файла с номерами строк (для подсказки при неудачном edit)."""
    lines = [line.rstrip("\n") for line in _lines(text)]
    shown = lines[:max_lines]
    width = len(str(len(shown)))
    body = "\n".join(f"{str(i + 1).rjust(width)}\t{ln}" for i, ln in enumerate(shown))
    if len(lines) > max_lines:
        body += f"\n… (ещё {len(lines) - max_lines} строк)"
    return body


def _leading(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" \t"))]


IndentRule = Callable[[str], "str | None"]


def _indent_rule(pairs: list[tuple[str, str]]) -> IndentRule | None:
    """Как перевести отступ из фрагмента модели в отступ файла.

    ``pairs`` — (отступ строки шаблона, отступ совпавшей строки файла). Годится
    единый добавленный префикс (модель потеряла общий отступ), единый убранный
    (лишний общий отступ) или согласованное соответствие отступов (пробелы ↔ табы,
    в том числе кратными единицами). Иначе None — правку не угадываем.
    """
    added = {f[: len(f) - len(p)] if f.endswith(p) else None for p, f in pairs}
    if len(added) == 1 and None not in added:
        prefix = added.pop()
        return lambda indent: prefix + indent
    removed = {p[: len(p) - len(f)] if p.endswith(f) else None for p, f in pairs}
    if len(removed) == 1 and None not in removed:
        extra = removed.pop()
        return lambda indent: indent[len(extra) :] if indent.startswith(extra) else None
    mapping: dict[str, str] = {}
    for p, f in pairs:
        if mapping.setdefault(p, f) != f:
            return None
    units = [(p, f) for p, f in mapping.items() if p and f]
    if units:
        pu, fu = min(units, key=lambda pf: len(pf[0]))
        if all(
            p == pu * (len(p) // len(pu)) and f == fu * (len(p) // len(pu))
            for p, f in mapping.items()
        ):

            def by_units(indent: str) -> str | None:
                k = len(indent) // len(pu)
                return fu * k if indent == pu * k else None

            return by_units
    return mapping.get


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
    offsets = []
    pos = 0
    for ln in raw:
        offsets.append(pos)
        pos += len(ln)
    contents = [ln.rstrip("\n") for ln in raw]

    pat_lines = pattern.split("\n")
    lead = trail = 0
    while pat_lines and not pat_lines[0].strip():
        pat_lines.pop(0)
        lead += 1
    while pat_lines and not pat_lines[-1].strip():
        pat_lines.pop()
        trail += 1
    n = len(pat_lines)
    if n == 0:
        return None

    # От более строгой нормализации (только хвостовые пробелы) к более мягкой
    # (полный strip — игнор отступов). Берём первый режим с уникальным совпадением.
    for strict in (True, False):

        def norm(s: str, strict: bool = strict) -> str:
            return s.rstrip() if strict else s.strip()

        target = [norm(ln) for ln in pat_lines]
        hits = [
            i
            for i in range(len(contents) - n + 1)
            if [norm(contents[j]) for j in range(i, i + n)] == target
        ]
        if len(hits) != 1:
            continue
        i = hits[0]
        start = offsets[i]
        end = offsets[i + n - 1] + len(contents[i + n - 1])
        if strict:
            return _Match(start, end, lead, trail)
        pairs = [
            (_leading(pat), _leading(contents[i + j]))
            for j, pat in enumerate(pat_lines)
            if pat.strip()
        ]
        rule = _indent_rule(pairs)
        return None if rule is None else _Match(start, end, lead, trail, rule)
    return None


# --------------------------------------------------------------------------- #
# read_file
# --------------------------------------------------------------------------- #
class ReadFileParams(BaseModel):
    path: str = Field(description="File path relative to the project root")
    start_line: int | None = Field(default=None, description="First line (1-based), inclusive")
    end_line: int | None = Field(default=None, description="Last line (1-based), inclusive")


class ReadFileTool(Tool):
    name = "read_file"
    read_only = True
    description = (
        "Reads a text file. Returns the text with line numbers. Pass a line range "
        "(start_line/end_line) to read only the part you need — use file_outline or "
        f"search to find it. At most {MAX_READ_LINES} lines per call; to continue, pass "
        "start_line."
    )
    Params = ReadFileParams

    def run(self, params: ReadFileParams, ctx: ToolContext) -> ToolResult:
        p = resolve_in_root(ctx.root, params.path)
        if not p.exists():
            raise ToolError(f"Файл не найден: {params.path}")
        if p.is_dir():
            raise ToolError(f"Это директория, а не файл: {params.path}")
        if not p.is_file():  # FIFO/сокет/устройство — чтение может заблокироваться
            raise ToolError(f"Не обычный файл (FIFO, сокет или устройство): {params.path}")
        if p.stat().st_size > MAX_READ_BYTES:
            raise ToolError(
                f"Файл слишком большой (>{MAX_READ_BYTES} байт). Используйте search_content."
            )
        with p.open("rb") as fh:
            if b"\0" in fh.read(_BINARY_PROBE):
                raise ToolError(f"Бинарный файл, чтение не поддерживается: {params.path}")

        start = max(params.start_line or 1, 1)
        encoding_note = ""
        try:
            selected, total, stopped_at = _select_lines(p, start, params.end_line, "utf-8")
        except UnicodeDecodeError:
            # Чаще всего — старый русский текст в cp1251: показать его полезнее отказа.
            selected, total, stopped_at = _select_lines(p, start, params.end_line, "cp1251")
            encoding_note = CP1251_NOTE + "\n"

        if not selected:
            numbered = "(пусто)" if total == 0 else f"(нет строк в диапазоне; всего строк: {total})"
        else:
            width = len(str(selected[-1][0]))
            numbered = "\n".join(f"{str(n).rjust(width)}\t{line}" for n, line in selected)
        first = selected[0][0] if selected else start
        if stopped_at is not None:
            numbered += (
                f"\n… показаны строки {first}–{stopped_at - 1} из {total}. "
                f"Продолжение: start_line={stopped_at}."
            )
        elif selected and (first > 1 or selected[-1][0] < total):
            last = selected[-1][0]
            numbered += f"\n… показаны строки {first}–{last} из {total}."
            if last < total:
                numbered += f" Продолжение: start_line={last + 1}."
        encoding = " · cp1251" if encoding_note else ""
        return ToolResult(
            content=encoding_note + numbered,
            summary=f"прочитан {_rel(ctx, p)} ({len(selected)} строк){encoding}",
        )


CP1251_NOTE = "(файл не в UTF-8 — показан как cp1251; edit_file и write_file такие файлы не меняют)"


def _select_lines(
    path: Path, start: int, end: int | None, encoding: str
) -> tuple[list[tuple[int, str]], int, int | None]:
    """Строки ``start..end`` в пределах лимитов чтения.

    Возвращает (выбранные (номер, строка), всего строк в файле, первая строка,
    не поместившаяся в лимит, или None). Для UTF-8 — ``UnicodeDecodeError`` на
    неверных байтах, иначе неверные байты заменяются.
    """
    selected: list[tuple[int, str]] = []
    budget = MAX_READ_CHARS
    total = 0
    stopped_at: int | None = None
    errors = "strict" if encoding == "utf-8" else "replace"
    with path.open(encoding=encoding, errors=errors) as fh:
        for total, raw in enumerate(fh, start=1):
            if total < start or (end is not None and total > end):
                continue
            if stopped_at is not None:
                continue  # досчитываем общее число строк
            line = raw.rstrip("\r\n")
            if len(line) > MAX_LINE_CHARS:
                line = line[:MAX_LINE_CHARS] + " …[строка обрезана]"
            if len(selected) >= MAX_READ_LINES or len(line) + 1 > budget:
                stopped_at = total
                continue
            selected.append((total, line))
            budget -= len(line) + 1
    return selected, total, stopped_at


# --------------------------------------------------------------------------- #
# write_file
# --------------------------------------------------------------------------- #
class WriteFileParams(BaseModel):
    path: str = Field(description="File path relative to the project root")
    content: str = Field(description="The complete new content of the file")


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Creates a new file or completely overwrites an existing one with the given "
        "content. Creates parent directories when needed. For changes to an existing "
        "file prefer edit_file."
    )
    Params = WriteFileParams
    kind = ToolKind.EDIT

    def risk(self, params: WriteFileParams, ctx: ToolContext) -> RiskLevel:
        # План — данные агента, а не проект: пишется в любом режиме без вопроса.
        return RiskLevel.SAFE if _is_plan_file(ctx, params.path) else RiskLevel.WRITE

    def _prepare(
        self, params: WriteFileParams, ctx: ToolContext
    ) -> tuple[Path, TextFile | None, TextFile, bool]:
        """Путь, прежний текст и новый текст с оформлением, с которым он запишется.

        Существующий файл сохраняет свои переводы строк и BOM; новый — как прислано.
        """
        p = _writable_path(ctx, params.path)
        old = load_existing(p, params.path) if p.exists() else None
        content, repaired = repair_escaped_content(params.content)
        new = decode_text(content)
        if old is not None:
            new = TextFile(new.text, old.eol, old.bom, old.mixed)
        return p, old, new, repaired

    def preview(self, params: WriteFileParams, ctx: ToolContext) -> Display | None:
        _, old, new, _ = self._prepare(params, ctx)
        diff = make_diff(old.text if old else "", new.text, params.path)
        if not diff:
            diff = "(без изменений)" if old is not None else "(новый пустой файл)"
        return Display(diff, kind="diff", title=f"{params.path}{new.eol_note()}")

    def run(self, params: WriteFileParams, ctx: ToolContext) -> ToolResult:
        p, old, new, repaired = self._prepare(params, ctx)
        if ctx.plan is not None and ctx.plan.is_target(p):
            ctx.workspace.ensure_data_dir()  # .gitignore папки — до первого файла в ней
        p.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(p, new.encode())
        verb = "перезаписан" if old is not None else "создан"
        target = "план" if _plan_written(ctx, p) else _rel(ctx, p)
        n = len(_lines(new.text))
        note = (" (автокоррекция экранирования)" if repaired else "") + new.eol_note()
        return ToolResult(
            content=f"Файл {verb}: {params.path} ({n} строк).{note}",
            summary=f"{verb} {target}{note}",
            display=Display(
                make_diff(old.text if old else "", new.text, params.path),
                kind="diff",
                title=params.path,
            ),
        )


# --------------------------------------------------------------------------- #
# edit_file
# --------------------------------------------------------------------------- #
class EditFileParams(BaseModel):
    path: str = Field(description="File path relative to the project root")
    old_string: str = Field(description="The exact fragment to replace (must be present)")
    new_string: str = Field(description="The replacement text")
    replace_all: bool = Field(
        default=False, description="Replace all occurrences instead of exactly one"
    )


class EditFileTool(Tool):
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

    def risk(self, params: EditFileParams, ctx: ToolContext) -> RiskLevel:
        return RiskLevel.SAFE if _is_plan_file(ctx, params.path) else RiskLevel.WRITE

    def _compute(self, params: EditFileParams, ctx: ToolContext):
        p = _writable_path(ctx, params.path)
        if not p.exists():
            raise ToolError(f"Файл не найден: {params.path}")
        file = load_existing(p, params.path)
        old = file.text
        # Сравнение и правка — в виде с \n; переводы строк файла вернёт запись.
        old_param = params.old_string.replace("\r\n", "\n")
        new_param = params.new_string.replace("\r\n", "\n")
        if not old_param:
            # "" нашлось бы между каждой парой символов: replace_all вставил бы
            # new_string повсюду.
            raise ToolError(
                "old_string пуст — укажите заменяемый фрагмент. Чтобы создать файл "
                "или заменить его целиком, используйте write_file."
            )
        if old_param == new_param:
            raise ToolError("old_string и new_string совпадают — нечего менять.")

        # Слабые модели часто слегка искажают old_string: экранируют переносы
        # (литералы \n, \"), либо копируют префиксы-номера строк (N\t) прямо из
        # вывода read_file. Пробуем набор согласованных преобразований к ОБОИМ
        # фрагментам и берём первое, которое реально находится в файле — это
        # безопасно: преобразованная форма применяется, только если совпала.
        def both(s):
            return _strip_line_numbers(_unescape_simple(s))

        transforms = [
            lambda s: s,  # как прислано
            _unescape_simple,  # снять экранизацию \n/\"
            _strip_line_numbers,  # снять префиксы номеров строк
            both,  # и то, и другое
        ]

        seen = set()
        for tf in transforms:
            old_string = tf(old_param)
            if not old_string or old_string in seen:
                continue
            seen.add(old_string)
            new_string = tf(new_param)

            # Тир 1: точное совпадение.
            count = old.count(old_string)
            if count == 1 or (count > 1 and params.replace_all):
                new = old.replace(old_string, new_string)
                return p, file, new, count
            if count > 1 and not params.replace_all:
                raise ToolError(
                    f"old_string встречается {count} раз. Добавьте контекста, чтобы "
                    "фрагмент стал уникальным, или укажите replace_all=true."
                )
            # Тир 2: совпадение с поправкой на пробелы/отступы (одиночная замена).
            match = _tolerant_find(old, old_string)
            replacement = match.adapt(new_string) if match is not None else None
            if match is not None and replacement is not None:
                new = old[: match.start] + replacement + old[match.end :]
                return p, file, new, 1

        # Не найдено — даём модели контекст файла, чтобы скопировать точно.
        raise ToolError(
            "old_string не найден в файле (ни точно, ни с поправкой на пробелы). "
            "Скопируйте фрагмент дословно из содержимого ниже:\n\n" + _numbered_excerpt(old)
        )

    def preview(self, params: EditFileParams, ctx: ToolContext) -> Display | None:
        _, file, new, _ = self._compute(params, ctx)
        return Display(
            make_diff(file.text, new, params.path),
            kind="diff",
            title=f"{params.path}{file.eol_note()}",
        )

    def run(self, params: EditFileParams, ctx: ToolContext) -> ToolResult:
        p, file, new, count = self._compute(params, ctx)
        write_atomic(p, file.encode(new))
        replaced = count if params.replace_all else 1
        note = file.eol_note()
        target = "план" if _plan_written(ctx, p) else _rel(ctx, p)
        return ToolResult(
            content=f"Отредактирован {params.path}: заменено вхождений — {replaced}.{note}",
            summary=f"изменён {target}{note}",
            display=Display(make_diff(file.text, new, params.path), kind="diff", title=params.path),
        )


# --------------------------------------------------------------------------- #
# list_dir
# --------------------------------------------------------------------------- #
MAX_LIST_ENTRIES = 500


class ListDirParams(BaseModel):
    path: str = Field(default=".", description="Directory path (the project root by default)")


class ListDirTool(Tool):
    name = "list_dir"
    read_only = True
    description = (
        "Lists files and subdirectories (including hidden ones) of a directory. For an "
        "overview with languages and definitions use file_outline."
    )
    Params = ListDirParams

    def run(self, params: ListDirParams, ctx: ToolContext) -> ToolResult:
        p = resolve_in_root(ctx.root, params.path)
        if not p.is_dir():
            raise ToolError(f"Не директория: {params.path}")
        entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
        lines = [f"{e.name}/" if e.is_dir() else e.name for e in entries[:MAX_LIST_ENTRIES]]
        body = "\n".join(lines) if lines else "(пусто)"
        if len(entries) > MAX_LIST_ENTRIES:
            body += f"\n… показано {MAX_LIST_ENTRIES} из {len(entries)}"
        return ToolResult(content=body, summary=f"{_rel(ctx, p)}: {len(entries)} элементов")


# --------------------------------------------------------------------------- #
# find_files
# --------------------------------------------------------------------------- #
MAX_FIND_RESULTS = 1000


class FindFilesParams(BaseModel):
    pattern: str = Field(
        description="Glob pattern: '*.py' — by name in any directory, 'src/**/*.ts' — by path "
        "from the root"
    )
    max_results: int = Field(default=200, description="Maximum number of results")


def _check_pattern(pattern: str) -> None:
    """Шаблон должен быть относительным и не подниматься выше корня проекта."""
    if not pattern:
        raise ToolError("Пустой шаблон.")
    if PurePosixPath(pattern).is_absolute() or PureWindowsPath(pattern).is_absolute():
        raise ToolError("Шаблон должен быть относительным путём от корня проекта.")
    if ".." in re.split(r"[\\/]", pattern):
        raise ToolError("Шаблон не может содержать '..' — поиск только внутри проекта.")


class FindFilesTool(Tool):
    name = "find_files"
    read_only = True
    description = (
        "Finds files by a glob pattern (recursively, inside the project). A pattern "
        "without '/' is matched against the file name, with '/' — against the path from "
        "the root ('**' — any directories). Skips service directories (.git, "
        "node_modules, etc.) and files ignored by .gitignore."
    )
    Params = FindFilesParams

    def run(self, params: FindFilesParams, ctx: ToolContext) -> ToolResult:
        pattern = params.pattern.strip()
        _check_pattern(pattern)
        limit = min(max(params.max_results, 1), MAX_FIND_RESULTS)
        root = ctx.root
        matches = sorted(
            rel
            for rel in (path.relative_to(root).as_posix() for path in walk_files(root))
            if glob_match(rel, pattern)
        )
        shown = matches[:limit]
        body = "\n".join(shown) if shown else "(ничего не найдено)"
        if len(matches) > limit:
            body += f"\n… показано {limit} из {len(matches)}; уточните шаблон"
        return ToolResult(content=body, summary=f"найдено файлов: {len(matches)}")
