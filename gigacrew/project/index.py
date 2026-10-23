"""Индекс проекта: файлы, определения (символы), использования и импорты в ``.gigacode/index/``.

Индекс — база SQLite (stdlib): выборки и точечные правки не требуют загружать его
целиком в память, поэтому он годится и для больших кодовых баз. Обновление
инкрементальное: обходятся файлы проекта (с теми же правилами игнорирования, что
у поиска, — :func:`~gigacrew.project.files.walk_files`), перечитываются только
файлы с изменившимися размером или временем изменения, исчезнувшие — удаляются.

Изменения фиксируются пачками: прерванное построение (Ctrl+C) сохраняет уже
сделанное, а следующее обновление его продолжает. Признак полноты
(:meth:`ProjectIndex.is_complete`) выставляется только в конце полного обхода.

Повреждённая база пересоздаётся: несовместимая или с испорченным заголовком — при
открытии, с испорченными страницами данных — когда это обнаружит запрос
(:func:`is_corruption`, :meth:`ProjectIndex.discard`).

Файлы с секретами (``.env``) не индексируются; большие и бинарные файлы попадают
в список файлов без разбора содержимого.

Использования (:meth:`ProjectIndex.find_refs`) — по имени, без вывода типов:
результат ранжируется по тому, насколько вероятно, что имя ссылается на найденное
определение (файл импортирует модуль определения > тот же файл > только имя).

Запросы с переменным числом условий собираются из фиксированных фрагментов SQL и
плейсхолдеров «?»; сами значения всегда передаются параметрами запроса.
"""

from __future__ import annotations

import os
import sqlite3
import stat
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from gigacrew.project.files import glob_match, is_excluded, walk_files
from gigacrew.project.secret_paths import is_secret_entry
from gigacrew.project.symbols import Symbol, extract, language_of, module_parts
from gigacrew.project.workspace import Workspace

SCHEMA_VERSION = 3  # 3 — tree-sitter: другие языки с диапазонами и вызовами
INDEX_FILE_NAME = "index.sqlite3"
MAX_INDEX_FILE_BYTES = 1_000_000  # крупнее — сгенерированное/дампы: без разбора
_BINARY_PROBE = 8192
_BATCH = 200  # изменённых файлов на одну транзакцию
_COUNT_CAP = 10_000  # дальше совпадения не досчитываются
_DELETE_CHUNK = 500  # путей в одном ``DELETE … WHERE path IN (…)``
_PRIMARY_CODE_MASK = 0xFF  # основной код ошибки SQLite — младший байт расширенного

# Ошибки, которые вызывающий код ловит при работе с индексом.
INDEX_ERRORS: tuple[type[BaseException], ...] = (sqlite3.Error, OSError)


def is_corruption(error: BaseException) -> bool:
    """Ошибка SQLite «база повреждена».

    Испорченные страницы данных (сбой диска, запись наполовину) или не база вовсе.
    Заголовок и схему проверяет :meth:`ProjectIndex.open`, а порча страниц всплывает
    только на запросе — такой индекс надо пересоздать (:meth:`ProjectIndex.discard`),
    иначе каждый запрос к нему будет падать.
    """
    code = getattr(error, "sqlite_errorcode", None)
    return (
        isinstance(error, sqlite3.DatabaseError)
        and code is not None
        and code & _PRIMARY_CODE_MASK in {sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB}
    )


# Подписчики на изменения файлов, замеченные обновлением индекса (корень проекта,
# добавленные, изменённые и удалённые пути) — например, LSP-сервер, который сам за
# файлами не следит. Обновление индекса — единственный детектор изменений.
_Paths = tuple[str, ...]
ChangeListener = Callable[[Path, _Paths, _Paths, _Paths], None]
_change_listeners: list[ChangeListener] = []


def add_change_listener(listener: ChangeListener) -> None:
    """Подписывает ``listener`` на изменения файлов; повторная подписка ничего не меняет."""
    if listener not in _change_listeners:
        _change_listeners.append(listener)


def remove_change_listener(listener: ChangeListener) -> None:
    """Отписывает ``listener``; если он не подписан — ничего не делает."""
    if listener in _change_listeners:
        _change_listeners.remove(listener)


def _notify_changes(
    root: Path,
    added: tuple[str, ...],
    updated: tuple[str, ...],
    removed: tuple[str, ...],
) -> None:
    if not (added or updated or removed):
        return
    for listener in _change_listeners:
        # подписчик не должен ломать обновление индекса
        with suppress(Exception):
            listener(root, added, updated, removed)


# Обобщённые виды для фильтра: "function" находит и методы и т.п.
KIND_GROUPS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "function": ("function", "method"),
    "class": (
        "class",
        "struct",
        "interface",
        "trait",
        "record",
        "object",
        "protocol",
        "enum",
        "union",
        "module",
    ),
})

_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE files (
    path TEXT PRIMARY KEY,
    size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    language TEXT,
    lines INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL
);
CREATE TABLE symbols (
    path TEXT NOT NULL,
    name TEXT NOT NULL,
    name_lower TEXT NOT NULL,
    parent TEXT NOT NULL,
    parent_lower TEXT NOT NULL,
    kind TEXT NOT NULL,
    line INTEGER NOT NULL,
    end_line INTEGER,
    depth INTEGER NOT NULL,
    signature TEXT NOT NULL,
    doc TEXT NOT NULL,
    col INTEGER NOT NULL
);
CREATE INDEX symbols_name ON symbols (name_lower);
CREATE INDEX symbols_path ON symbols (path);
CREATE TABLE refs (
    path TEXT NOT NULL,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    line INTEGER NOT NULL,
    col INTEGER NOT NULL,
    scope TEXT NOT NULL
);
CREATE INDEX refs_name ON refs (name);
CREATE INDEX refs_path ON refs (path);
CREATE TABLE imports (
    path TEXT NOT NULL,
    module TEXT NOT NULL,
    name TEXT NOT NULL,
    alias TEXT NOT NULL,
    line INTEGER NOT NULL
);
CREATE INDEX imports_path ON imports (path);
CREATE INDEX imports_module ON imports (module);
"""

# Удаление фактов о файле (определений, использований, импортов) перед переиндексацией.
_DELETE_FACTS = (
    "DELETE FROM symbols WHERE path = ?",
    "DELETE FROM refs WHERE path = ?",
    "DELETE FROM imports WHERE path = ?",
)

# Уровни разрешения использования — от надёжного к эвристике.
RESOLVED_IMPORT = "import"  # файл импортирует модуль, где имя определено
RESOLVED_SAME_FILE = "same_file"
RESOLVED_NAME = "name_only"
_RESOLUTION_ORDER = MappingProxyType({RESOLVED_IMPORT: 0, RESOLVED_SAME_FILE: 1, RESOLVED_NAME: 2})
REF_IMPORT = "import"  # вид использования: строка импорта

# Состояния файла в индексе.
STATUS_INDEXED = "indexed"
STATUS_LARGE = "large"
STATUS_BINARY = "binary"
STATUS_ERROR = "error"

# Строка использования из базы: (путь, строка, столбец, вид, область).
_RefRow = tuple[str, int, int, str, str]


@dataclass(frozen=True)
class RefreshStats:
    """Итог обновления индекса: сколько файлов просмотрено, добавлено, изменено, удалено."""

    scanned: int = 0
    added: int = 0
    updated: int = 0
    removed: int = 0
    duration_s: float = float(0)
    added_paths: tuple[str, ...] = ()
    updated_paths: tuple[str, ...] = ()
    removed_paths: tuple[str, ...] = ()

    @property
    def changed(self) -> int:
        """Сколько файлов изменилось всего: добавленные, изменённые и удалённые."""
        return self.added + self.updated + self.removed


@dataclass(frozen=True)
class IndexStats:
    """Сводка по индексу: размеры таблиц, языки, полнота и размер базы."""

    files: int
    symbols: int
    refs: int
    languages: list[tuple[str, int]]  # (язык, файлов) по убыванию
    complete: bool
    updated_at: float | None  # время последнего полного обновления (epoch)
    db_bytes: int


@dataclass(frozen=True)
class FileEntry:
    """Файл в индексе: язык, число строк, состояние (``STATUS_*``) и число определений."""

    path: str
    language: str | None
    lines: int
    status: str
    symbols: int = 0


@dataclass(frozen=True)
class SymbolHit:
    """Найденное определение и файл, где оно лежит."""

    path: str
    symbol: Symbol


@dataclass(frozen=True)
class RefHit:
    """Использование имени. ``kind``: call, attr, name или import (строка импорта)."""

    path: str
    line: int
    col: int
    kind: str
    scope: str
    resolution: str  # RESOLVED_*


@dataclass(frozen=True)
class RefsResult:
    """Найденные определения и их использования (первые ``limit`` и общее число)."""

    definitions: list[SymbolHit]
    hits: list[RefHit]  # первые ``limit``
    total: int


@dataclass(frozen=True)
class ImportEntry:
    """Импорт в файле: модуль, имя, псевдоним, строка и файл проекта, куда он ведёт."""

    module: str
    name: str
    alias: str
    line: int
    target: str | None  # файл проекта, куда ведёт импорт (None — внешний/не найден)


@dataclass
class _Scan:
    """Ход обновления индекса; виден и при прерывании — для уведомления подписчиков."""

    known: dict[str, tuple[int, int] | None]  # путь → (размер, mtime_ns); None — перечитать
    scanned: int = 0
    pending: int = 0  # изменённых файлов с последней фиксации
    seen: set[str] = field(default_factory=set)
    new: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    gone: list[str] = field(default_factory=list)


def _escape_like(text: str) -> str:
    return text.replace("\\", r"\\").replace("%", r"\%").replace("_", r"\_")


def _marks(count: int) -> str:
    """Плейсхолдеры «?» через запятую для ``IN (…)``."""
    return ",".join("?" * count)


def _under(prefix: str) -> tuple[str, list[str | int]]:
    """SQL-условие «путь равен ``prefix`` или лежит под ним» (пустой — весь проект)."""
    if not prefix:
        return "1", []
    # не LIKE: он не различает регистр ASCII, а `Src/` и `src/` — разные каталоги
    return "(path = ? OR substr(path, 1, ?) = ?)", [prefix, len(prefix) + 1, f"{prefix}/"]


def _count_lines(text: str) -> int:
    if not text:
        return 0
    newlines = text.count("\n")
    if text.endswith("\n"):
        return newlines
    return newlines + 1


class ProjectIndex:
    """Индекс проекта. Конструктор не трогает ФС; база открывается в :meth:`open`.

    Использование::

        with ProjectIndex(workspace) as index:
            index.refresh()
            hits, total = index.find_symbols("Agent")
    """

    def __init__(self, workspace: Workspace) -> None:
        """Индекс проекта ``workspace``; база не открывается до первого обращения."""
        self._ws = workspace
        self._conn: sqlite3.Connection | None = None
        self._modules: dict[str, str | None] = {}  # кэш resolve_module до refresh

    # ------------------------------------------------------------------ #
    @property
    def path(self) -> Path:
        """Файл базы индекса."""
        return self._ws.index_dir / INDEX_FILE_NAME

    @property
    def root(self) -> Path:
        """Корень проекта."""
        return self._ws.root

    def __enter__(self) -> ProjectIndex:
        """Открывает базу на время блока ``with``."""
        return self.open()

    def __exit__(self, *exc: object) -> None:
        """Закрывает базу по выходе из блока ``with``."""
        self.close()

    def open(self) -> ProjectIndex:
        """Открывает (создаёт) базу; несовместимая или повреждённая — пересоздаётся."""
        if self._conn is not None:
            return self
        self._ws.ensure_data_dir()
        self._ws.index_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = self._connect_checked()
        except sqlite3.DatabaseError:
            self._recreate_file()
            self._conn = self._connect_checked()
        return self

    def close(self) -> None:
        """Закрывает соединение с базой (если оно открыто)."""
        conn = self._conn
        self._conn = None
        if conn is not None:
            conn.close()

    def discard(self) -> None:
        """Удаляет базу (повреждённую, см. :func:`is_corruption`).

        Следующее открытие создаст пустую, а обновление построит индекс заново.
        """
        self.close()
        self._recreate_file()
        self._modules.clear()

    def is_complete(self) -> bool:
        """Индекс существует и полностью построен (не создаёт базу, если её нет)."""
        if not self.path.is_file():
            return False
        return self._meta("complete") == "1"

    # ---------------------------- обновление ---------------------------- #
    def refresh(
        self,
        base: Path | None = None,
        *,
        progress: Callable[[int], None] | None = None,
    ) -> RefreshStats:
        """Синхронизирует индекс с файлами под ``base`` (по умолчанию — весь проект).

        ``progress`` получает число просмотренных файлов. Прерывание (в т.ч. Ctrl+C)
        откатывает только незафиксированную пачку изменений.
        """
        started = time.monotonic()
        db = self._db
        base = self.root if base is None else base
        full = base == self.root
        scan = _Scan(known=self._known_files(base))
        try:
            self._sync_files(db, scan, base, full=full, progress=progress)
        except BaseException:
            db.rollback()
            raise
        finally:
            self._modules.clear()
            # и при прерывании: лишнее уведомление безвредно, пропущенное — нет
            _notify_changes(self.root, tuple(scan.new), tuple(scan.changed), tuple(scan.gone))
        return RefreshStats(
            scanned=scan.scanned,
            added=len(scan.new),
            updated=len(scan.changed),
            removed=len(scan.gone),
            duration_s=time.monotonic() - started,
            added_paths=tuple(scan.new),
            updated_paths=tuple(scan.changed),
            removed_paths=tuple(scan.gone),
        )

    def rebuild(self, *, progress: Callable[[int], None] | None = None) -> RefreshStats:
        """Удаляет базу и строит индекс заново."""
        self.close()
        self._recreate_file()
        self.open()
        return self.refresh(progress=progress)

    # ------------------------------ запросы ----------------------------- #
    def stats(self) -> IndexStats:
        """Сводка по индексу: число файлов, определений, использований, языки, размер базы."""
        db = self._db
        files = db.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        symbols = db.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
        refs = db.execute("SELECT COUNT(*) FROM refs").fetchone()[0]
        languages = [
            (lang, count)
            for lang, count in db.execute(
                "SELECT language, COUNT(*) AS n FROM files WHERE language IS NOT NULL "
                "GROUP BY language ORDER BY n DESC, language",
            )
        ]
        updated = self._meta("updated_at")
        try:
            size = sum(
                Path(f"{self.path}{suffix}").stat().st_size
                for suffix in ("", "-wal")
                if Path(f"{self.path}{suffix}").exists()
            )
        except OSError:
            size = 0
        return IndexStats(
            files=files,
            symbols=symbols,
            refs=refs,
            languages=languages,
            complete=self._meta("complete") == "1",
            updated_at=float(updated) if updated else None,
            db_bytes=size,
        )

    def find_symbols(
        self,
        query: str,
        *,
        kind: str | None = None,
        path_glob: str | None = None,
        limit: int = 50,
    ) -> tuple[list[SymbolHit], int]:
        """Определения, чьё имя содержит ``query`` (без учёта регистра).

        ``Class.method`` — метод с именем ``method`` у владельца, оканчивающегося на
        ``Class``. Порядок: точное совпадение имени, начало имени, вхождение; затем
        короткие имена и путь. Возвращает (первые ``limit``, сколько всего — не
        больше :data:`_COUNT_CAP`).
        """
        query = query.strip().lower()
        if not query:
            raise ValueError("пустой запрос")
        parent = ""
        if "." in query.strip("."):
            parent, query = query.rsplit(".", 1)
        where, args = _symbol_filter(query, parent, kind)
        sql = (
            # Подставляется только условие из :func:`_symbol_filter` — с плейсхолдерами «?».
            "SELECT path, name, kind, line, end_line, parent, depth, signature, doc, col, "  # noqa: S608
            r"CASE WHEN name_lower = ? THEN 0 WHEN name_lower LIKE ? ESCAPE '\' THEN 1 "
            "ELSE 2 END AS rank "
            f"FROM symbols WHERE {where} "
            "ORDER BY rank, length(name), path, line"
        )
        params = [query, f"{_escape_like(query)}%", *args]
        hits: list[SymbolHit] = []
        total = 0
        for row in self._db.execute(sql, params):
            path = row[0]
            if path_glob and not glob_match(path, path_glob):
                continue
            total += 1
            if len(hits) < limit:
                hits.append(SymbolHit(path, _symbol(row[1:10])))
            if total >= _COUNT_CAP:
                break
        return hits, total

    def file_entry(self, rel_path: str) -> FileEntry | None:
        """Файл ``rel_path`` в индексе с числом определений; None — его там нет."""
        row = self._db.execute(
            "SELECT path, language, lines, status FROM files WHERE path = ?",
            (rel_path,),
        ).fetchone()
        if row is None:
            return None
        count = self._db.execute(
            "SELECT COUNT(*) FROM symbols WHERE path = ?",
            (rel_path,),
        ).fetchone()[0]
        path, language, lines, status = row
        return FileEntry(path, language, lines, status, symbols=count)

    def symbol_at(self, rel_path: str, line: int) -> Symbol | None:
        """Определение в строке ``line``, иначе самое вложенное, содержащее её."""
        row = self._db.execute(
            "SELECT name, kind, line, end_line, parent, depth, signature, doc, col FROM symbols "
            "WHERE path = ? AND (line = ? OR (line < ? AND end_line >= ?)) "
            "ORDER BY line = ? DESC, line DESC, depth DESC LIMIT 1",
            (rel_path, line, line, line, line),
        ).fetchone()
        return _symbol(row) if row else None

    def ref_at(
        self,
        rel_path: str,
        line: int,
        names: Iterable[str],
        *,
        col: int | None = None,
    ) -> tuple[str, str] | None:
        """Вид и область использования одного из ``names`` в строке ``line``.

        Возвращает (вид, область); столбец ``col`` учитывается, если задан. Из
        нескольких использований — вызов первым.
        """
        names = list(names)
        if not names:
            return None
        marks = _marks(len(names))
        at_col = "" if col is None else " AND col = ?"
        row = self._db.execute(
            # Подставляются только плейсхолдеры «?» и условие на столбец — без значений.
            "SELECT kind, scope FROM refs "  # noqa: S608
            f"WHERE path = ? AND line = ? AND name IN ({marks}){at_col} "
            "ORDER BY CASE kind WHEN 'call' THEN 0 WHEN 'attr' THEN 1 ELSE 2 END, col LIMIT 1",
            (rel_path, line, *names, *([] if col is None else [col])),
        ).fetchone()
        return (row[0], row[1]) if row else None

    def outline(self, rel_path: str) -> list[Symbol]:
        """Определения файла в порядке строк."""
        rows = self._db.execute(
            "SELECT name, kind, line, end_line, parent, depth, signature, doc, col FROM symbols "
            "WHERE path = ? ORDER BY line, depth",
            (rel_path,),
        )
        return [_symbol(row) for row in rows]

    # ------------------------- использования, импорты ------------------------- #
    def resolve_module(self, module: str) -> str | None:
        """Файл проекта с модулем ``a.b.c``: ``…/a/b/c.py``, ``.pyi`` или ``…/__init__.py``.

        Совпадение — по суффиксу пути (корень пакетов, например ``src/``, не нужен);
        из нескольких — ближайший к корню, ``.py`` раньше ``.pyi``.
        """
        if module not in self._modules:
            self._modules[module] = self._find_module(module)
        return self._modules[module]

    def imports_of(self, rel_path: str) -> list[ImportEntry]:
        """Импорты файла с разрешёнными целями, в порядке строк."""
        rows = self._db.execute(
            "SELECT module, name, alias, line FROM imports WHERE path = ? ORDER BY line",
            (rel_path,),
        ).fetchall()
        return [
            ImportEntry(module, name, alias, line, self._import_target(module, name))
            for module, name, alias, line in rows
        ]

    def imported_by(self, rel_path: str) -> list[tuple[str, int]]:
        """Файлы проекта, импортирующие ``rel_path``: (путь, первая строка импорта)."""
        parts = module_parts(rel_path)
        if not parts:
            return []
        found: dict[str, int] = {}
        for path, module, name, line in self._import_rows(parts):
            if path == rel_path or path in found:
                continue
            if self._import_target(module, name) == rel_path:
                found[path] = line
        return sorted(found.items())

    def definitions(self, query: str) -> list[SymbolHit]:
        """Определения с точным именем (без учёта регистра): ``name`` или ``Class.method``."""
        if not query.strip(" ."):
            raise ValueError("пустой запрос")
        name = _last_name(query)
        hits, _ = self.find_symbols(query.strip(), limit=_COUNT_CAP)
        return [hit for hit in hits if hit.symbol.name.lower() == name.lower()]

    def calls_in(self, rel_path: str, scope: str) -> list[tuple[str, int]]:
        """Вызовы внутри определения ``scope`` (qualname): (имя, строка) по порядку."""
        rows = self._db.execute(
            "SELECT name, line FROM refs WHERE path = ? AND scope = ? AND kind = 'call' "
            "ORDER BY line, col",
            (rel_path, scope),
        )
        return [(name, line) for name, line in rows]

    def find_refs(
        self,
        query: str,
        *,
        kind: str | None = None,
        path_glob: str | None = None,
        limit: int = 50,
    ) -> RefsResult:
        """Использования определения ``query`` (``name`` или ``Class.method``).

        Определения ищутся без учёта регистра (точное имя), использования — по
        точному имени найденных определений (или по самому запросу, если их нет).
        ``kind``: call, attr, name, import; None — все.
        """
        definitions = self.definitions(query)
        names = sorted({hit.symbol.name for hit in definitions}) or [_last_name(query)]
        kinds = {kind.strip().lower()} if kind else None
        rows = self._ref_rows(names, kinds)
        hits = self._ref_hits(rows, definitions, kinds, path_glob)
        hits.sort(key=lambda hit: (_RESOLUTION_ORDER[hit.resolution], hit.path, hit.line, hit.col))
        return RefsResult(definitions=definitions[:limit], hits=hits[:limit], total=len(hits))

    def files_under(self, rel_dir: str) -> Iterator[FileEntry]:
        """Файлы каталога (рекурсивно) с числом символов, по пути."""
        cond, args = _under(rel_dir)
        cond = cond.replace("path", "f.path")
        rows = self._db.execute(
            # Подставляется только условие из :func:`_under` — с плейсхолдерами «?».
            "SELECT f.path, f.language, f.lines, f.status, "  # noqa: S608
            "(SELECT COUNT(*) FROM symbols s WHERE s.path = f.path) "
            f"FROM files f WHERE {cond} ORDER BY f.path",
            args,
        )
        for path, language, lines, status, count in rows:
            yield FileEntry(path, language, lines, status, symbols=count)

    # ------------------------- соединение и meta ------------------------- #
    @property
    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            self.open()
        if self._conn is None:  # open() открывает базу или бросает исключение
            raise AssertionError
        return self._conn

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5)
        try:
            # ФС без поддержки WAL — работаем в режиме по умолчанию
            with suppress(sqlite3.OperationalError):
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
        except BaseException:
            conn.close()
            raise
        return conn

    def _connect_checked(self) -> sqlite3.Connection:
        conn = self._connect()
        try:
            compatible = _prepare_schema(conn)
        except BaseException:
            conn.close()
            raise
        if compatible:
            return conn
        conn.close()
        raise sqlite3.DatabaseError("несовместимая версия индекса")

    def _recreate_file(self) -> None:
        for suffix in ("", "-wal", "-shm", "-journal"):
            Path(f"{self.path}{suffix}").unlink(missing_ok=True)

    def _meta(self, key: str) -> str | None:
        row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _set_meta(self, key: str, text: str) -> None:
        self._db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, text))

    # ----------------------- обновление: служебное ----------------------- #
    def _rel(self, path: Path) -> str:
        rel = path.relative_to(self.root).as_posix()
        return "" if rel == "." else rel

    def _candidates(self, base: Path) -> Iterable[Path]:
        if is_excluded(self.root, base):
            return ()
        if base.is_dir():
            return walk_files(self.root, base)
        return (base,) if base.exists() else ()

    def _known_files(self, base: Path) -> dict[str, tuple[int, int] | None]:
        """Файлы индекса под ``base``: путь → (размер, mtime_ns); None — перечитать."""
        cond, args = _under(self._rel(base))
        rows = self._db.execute(
            # Подставляется только условие из :func:`_under` — с плейсхолдерами «?».
            f"SELECT path, size, mtime_ns, status FROM files WHERE {cond}",  # noqa: S608
            args,
        )
        # файлы с ошибкой чтения перечитываются, даже если не менялись (права могли починить)
        return {
            path: None if status == STATUS_ERROR else (size, mtime)
            for path, size, mtime, status in rows
        }

    def _sync_files(
        self,
        db: sqlite3.Connection,
        scan: _Scan,
        base: Path,
        *,
        full: bool,
        progress: Callable[[int], None] | None,
    ) -> None:
        """Переиндексирует изменённые файлы под ``base`` и удаляет исчезнувшие."""
        if full:
            with db:
                self._set_meta("complete", "0")
        for path in self._candidates(base):
            self._scan_file(db, scan, path, progress)
        scan.gone = [rel for rel in scan.known if rel not in scan.seen]
        for chunk in _chunks(scan.gone, _DELETE_CHUNK):
            marks = _marks(len(chunk))
            for table in ("symbols", "refs", "imports", "files"):
                # Подставляются только имя таблицы из списка выше и плейсхолдеры «?».
                db.execute(f"DELETE FROM {table} WHERE path IN ({marks})", chunk)  # noqa: S608
        if full:
            self._set_meta("complete", "1")
            self._set_meta("updated_at", str(time.time()))
        db.commit()

    def _scan_file(
        self,
        db: sqlite3.Connection,
        scan: _Scan,
        path: Path,
        progress: Callable[[int], None] | None,
    ) -> None:
        """Учитывает файл ``path`` в обходе; изменённый — переиндексирует."""
        file_stat = self._regular_file_stat(path)
        if file_stat is None:
            return
        rel = self._rel(path)
        scan.seen.add(rel)
        scan.scanned += 1
        if progress is not None:
            progress(scan.scanned)
        if scan.known.get(rel) == (file_stat.st_size, file_stat.st_mtime_ns):
            return
        self._index_file(rel, path, file_stat)
        (scan.changed if rel in scan.known else scan.new).append(rel)
        scan.pending += 1
        if scan.pending >= _BATCH:
            db.commit()
            scan.pending = 0

    def _regular_file_stat(self, path: Path) -> os.stat_result | None:
        """Сведения об обычном файле, который индексируется; None — файл пропускается."""
        # Секреты — по правилу read_file и классификатора команд (не только .env).
        if is_secret_entry(self.root, path):
            return None
        # симлинк на исключённое (.gitignore, служебные каталоги) — тоже мимо
        if path.is_symlink() and is_excluded(self.root, path.resolve()):
            return None
        try:
            file_stat = path.stat()
        except OSError:
            return None
        return file_stat if stat.S_ISREG(file_stat.st_mode) else None

    def _index_file(self, rel: str, path: Path, file_stat: os.stat_result) -> None:
        language = language_of(rel)
        status, lines, facts = STATUS_INDEXED, 0, None
        if file_stat.st_size > MAX_INDEX_FILE_BYTES:
            status = STATUS_LARGE
        else:
            try:
                raw = path.read_bytes()
            except OSError:
                raw, status = b"", STATUS_ERROR
            if b"\0" in raw[:_BINARY_PROBE]:
                status = STATUS_BINARY
            elif status == STATUS_INDEXED:
                text = raw.decode("utf-8", errors="replace")
                lines = _count_lines(text)
                try:
                    facts = extract(text, language, rel)
                except Exception:  # разбор — эвристика: сбой на одном файле не валит индекс
                    status = STATUS_ERROR
        db = self._db
        for delete_sql in _DELETE_FACTS:
            db.execute(delete_sql, (rel,))
        db.execute(
            "INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?, ?, ?)",
            (rel, file_stat.st_size, file_stat.st_mtime_ns, language, lines, status),
        )
        if facts is None:
            return
        db.executemany(
            "INSERT INTO symbols VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    rel,
                    sym.name,
                    sym.name.lower(),
                    sym.parent,
                    sym.parent.lower(),
                    sym.kind,
                    sym.line,
                    sym.end_line,
                    sym.depth,
                    sym.signature,
                    sym.doc,
                    sym.col,
                )
                for sym in facts.symbols
            ],
        )
        db.executemany(
            "INSERT INTO refs VALUES (?, ?, ?, ?, ?, ?)",
            [(rel, ref.name, ref.kind, ref.line, ref.col, ref.scope) for ref in facts.refs],
        )
        db.executemany(
            "INSERT INTO imports VALUES (?, ?, ?, ?, ?)",
            [(rel, imp.module, imp.name, imp.alias, imp.line) for imp in facts.imports],
        )

    # ------------------- использования, импорты: служебное ------------------- #
    def _find_module(self, module: str) -> str | None:
        """Файл модуля ``module`` по правилам :meth:`resolve_module` (без кэша)."""
        if not module or module.startswith("."):
            return None
        base = module.replace(".", "/")
        candidates: list[str] = []
        for suffix in (
            f"{base}.py",
            f"{base}/__init__.py",
            f"{base}.pyi",
            f"{base}/__init__.pyi",
        ):
            rows = self._db.execute(
                r"SELECT path FROM files WHERE path = ? OR path LIKE ? ESCAPE '\'",
                (suffix, f"%/{_escape_like(suffix)}"),
            )
            paths = [row[0] for row in rows]
            # LIKE не различает регистр ASCII — сверяем точно
            candidates += [path for path in paths if path == suffix or path.endswith(f"/{suffix}")]
        if not candidates:
            return None
        return min(candidates, key=lambda path: (path.count("/"), path.endswith(".pyi"), path))

    def _import_target(self, module: str, name: str) -> str | None:
        """Куда ведёт импорт: ``from pkg import mod`` — подмодуль, иначе сам модуль."""
        if name and name != "*":
            sub = self.resolve_module(f"{module}.{name}" if module else name)
            if sub is not None:
                return sub
        return self.resolve_module(module)

    def _import_rows(self, parts: list[str]) -> sqlite3.Cursor:
        """Импорты, которые могут вести к модулю ``parts``: (путь, модуль, имя, строка)."""
        modules = _module_suffixes(parts)
        parents = _module_suffixes(parts[:-1]) if len(parts) > 1 else [""]
        marks = _marks(len(modules))
        pmarks = _marks(len(parents))
        return self._db.execute(
            # Подставляются только плейсхолдеры «?»: имена модулей — параметрами.
            "SELECT path, module, name, line FROM imports "  # noqa: S608
            f"WHERE module IN ({marks}) OR (module IN ({pmarks}) AND name = ?) "
            "ORDER BY path, line",
            [*modules, *parents, parts[-1]],
        )

    def _ref_rows(self, names: list[str], kinds: set[str] | None) -> list[_RefRow]:
        """Использования ``names`` и строки их импорта — для видов ``kinds`` (None — всех)."""
        marks = _marks(len(names))
        imported: list[tuple[str, int, str]] = self._db.execute(
            # Подставляются только плейсхолдеры «?»: имена — параметрами.
            f"SELECT path, line, alias FROM imports WHERE name IN ({marks})",  # noqa: S608
            names,
        ).fetchall()
        rows: list[_RefRow] = []
        if kinds is None or kinds - {REF_IMPORT}:
            rows.extend(
                self._db.execute(
                    # Подставляются только плейсхолдеры «?»: имена — параметрами.
                    f"SELECT path, line, col, kind, scope FROM refs WHERE name IN ({marks})",  # noqa: S608
                    names,
                ).fetchall(),
            )
            rows.extend(self._alias_rows(imported, names))
        if kinds is None or REF_IMPORT in kinds:
            rows.extend((path, line, 0, REF_IMPORT, "") for path, line, _alias in imported)
        return rows

    def _alias_rows(
        self,
        imported: list[tuple[str, int, str]],
        names: list[str],
    ) -> list[_RefRow]:
        """Использования под псевдонимом импорта.

        ``from m import f as g`` — в этом файле f используется как g.
        """
        aliases = {(path, alias) for path, _line, alias in imported if alias not in names}
        rows: list[_RefRow] = []
        for alias_path, alias_name in aliases:
            rows.extend(
                self._db.execute(
                    "SELECT path, line, col, kind, scope FROM refs WHERE name = ? AND path = ?",
                    (alias_name, alias_path),
                ).fetchall(),
            )
        return rows

    def _ref_hits(
        self,
        rows: list[_RefRow],
        definitions: list[SymbolHit],
        kinds: set[str] | None,
        path_glob: str | None,
    ) -> list[RefHit]:
        """Использования из ``rows`` нужных видов и путей, кроме самих определений."""
        def_paths = {hit.path for hit in definitions}
        def_positions = {(hit.path, hit.symbol.line, hit.symbol.col) for hit in definitions}
        imports_def: dict[str, bool] = {}
        hits: list[RefHit] = []
        for path, line, col, ref_kind, scope in rows:
            if kinds is not None and ref_kind not in kinds:
                continue
            if path_glob and not glob_match(path, path_glob):
                continue
            # имя в самом определении — не использование (рекурсивный вызов в той же строке — да)
            if (path, line, col) in def_positions:
                continue
            resolution = self._resolution(path, def_paths, imports_def)
            hits.append(RefHit(path, line, col, ref_kind, scope, resolution))
        return hits

    def _resolution(self, path: str, def_paths: set[str], imports_def: dict[str, bool]) -> str:
        """Уровень разрешения (``RESOLVED_*``) использования в файле ``path``.

        ``imports_def`` — кэш на один поиск: импортирует ли файл модуль определения.
        """
        if path in def_paths:
            return RESOLVED_SAME_FILE
        if path not in imports_def:
            imports_def[path] = any(entry.target in def_paths for entry in self.imports_of(path))
        return RESOLVED_IMPORT if imports_def[path] else RESOLVED_NAME


def _prepare_schema(conn: sqlite3.Connection) -> bool:
    """Создаёт схему в пустой базе; True — схема базы текущей версии."""
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
    if not tables:
        conn.executescript(_SCHEMA)
        with conn:
            conn.execute(
                "INSERT INTO meta VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
        return True
    if "meta" not in tables:
        return False
    row: tuple[str] | None = conn.execute(
        "SELECT value FROM meta WHERE key = 'schema_version'",
    ).fetchone()
    return row is not None and row[0] == str(SCHEMA_VERSION)


def _symbol_filter(query: str, parent: str, kind: str | None) -> tuple[str, list[object]]:
    """Условие поиска определений (имя, владелец, вид) и его параметры."""
    where = [r"name_lower LIKE ? ESCAPE '\'"]
    args: list[object] = [f"%{_escape_like(query)}%"]
    if parent:
        where.append(r"(parent_lower = ? OR parent_lower LIKE ? ESCAPE '\')")
        args += [parent, f"%.{_escape_like(parent)}"]
    if kind:
        group = kind.strip().lower()
        kinds = KIND_GROUPS.get(group, (group,))
        marks = _marks(len(kinds))
        where.append(f"kind IN ({marks})")
        args += list(kinds)
    return " AND ".join(where), args


def _last_name(query: str) -> str:
    query = query.strip().strip(".")
    return query.rsplit(".", 1)[-1]


def _module_suffixes(parts: list[str]) -> list[str]:
    """``[a, b, c]`` → ``a.b.c``, ``b.c``, ``c`` (корень пакетов неизвестен)."""
    return [".".join(parts[start:]) for start in range(len(parts))]


def _symbol(row: Sequence[Any]) -> Symbol:
    """Определение из строки ``name, kind, line, end_line, parent, depth, signature, doc, col``."""
    return Symbol(
        name=row[0],
        kind=row[1],
        line=row[2],
        end_line=row[3],
        parent=row[4],
        depth=row[5],
        signature=row[6],
        doc=row[7],
        col=row[8],
    )


def _chunks(paths: list[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(paths), size):
        yield paths[start : start + size]
