"""Индекс проекта: файлы, определения (символы), использования и импорты в
``.gigacode/index/``.

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
"""

from __future__ import annotations

import sqlite3
import stat
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from gigacrew.project.files import glob_match, is_excluded, walk_files
from gigacrew.project.symbols import Symbol, extract, language_of, module_parts
from gigacrew.project.workspace import Workspace
from gigacrew.safety.paths import is_secret_entry

SCHEMA_VERSION = 3  # 3 — tree-sitter: другие языки с диапазонами и вызовами
INDEX_FILE_NAME = "index.sqlite3"
MAX_INDEX_FILE_BYTES = 1_000_000  # крупнее — сгенерированное/дампы: без разбора
_BINARY_PROBE = 8192
_BATCH = 200  # изменённых файлов на одну транзакцию
_COUNT_CAP = 10_000  # дальше совпадения не досчитываются

# Ошибки, которые вызывающий код ловит при работе с индексом.
INDEX_ERRORS: tuple[type[BaseException], ...] = (sqlite3.Error, OSError)


def is_corruption(error: BaseException) -> bool:
    """Ошибка SQLite «база повреждена»: испорченные страницы данных (сбой диска, запись
    наполовину) или не база вовсе. Заголовок и схему проверяет :meth:`ProjectIndex.open`,
    а порча страниц всплывает только на запросе — такой индекс надо пересоздать
    (:meth:`ProjectIndex.discard`), иначе каждый запрос к нему будет падать."""
    code = getattr(error, "sqlite_errorcode", None)
    return (
        isinstance(error, sqlite3.DatabaseError)
        and code is not None
        and code & 0xFF in (sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB)
    )


# Подписчики на изменения файлов, замеченные обновлением индекса (корень проекта,
# добавленные, изменённые и удалённые пути) — например, LSP-сервер, который сам за
# файлами не следит. Обновление индекса — единственный детектор изменений.
ChangeListener = Callable[[Path, tuple[str, ...], tuple[str, ...], tuple[str, ...]], None]
_change_listeners: list[ChangeListener] = []


def add_change_listener(listener: ChangeListener) -> None:
    if listener not in _change_listeners:
        _change_listeners.append(listener)


def remove_change_listener(listener: ChangeListener) -> None:
    if listener in _change_listeners:
        _change_listeners.remove(listener)


def _notify_changes(
    root: Path, added: tuple[str, ...], updated: tuple[str, ...], removed: tuple[str, ...]
) -> None:
    if not (added or updated or removed):
        return
    for listener in list(_change_listeners):
        try:
            listener(root, added, updated, removed)
        except Exception:  # подписчик не должен ломать обновление индекса
            pass


# Обобщённые виды для фильтра: "function" находит и методы и т.п.
KIND_GROUPS: dict[str, tuple[str, ...]] = {
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
}

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

# Уровни разрешения использования — от надёжного к эвристике.
RESOLVED_IMPORT = "import"  # файл импортирует модуль, где имя определено
RESOLVED_SAME_FILE = "same_file"
RESOLVED_NAME = "name_only"
_RESOLUTION_ORDER = {RESOLVED_IMPORT: 0, RESOLVED_SAME_FILE: 1, RESOLVED_NAME: 2}
REF_IMPORT = "import"  # вид использования: строка импорта

# Состояния файла в индексе.
STATUS_INDEXED = "indexed"
STATUS_LARGE = "large"
STATUS_BINARY = "binary"
STATUS_ERROR = "error"


@dataclass(frozen=True)
class RefreshStats:
    scanned: int = 0
    added: int = 0
    updated: int = 0
    removed: int = 0
    duration_s: float = 0.0
    added_paths: tuple[str, ...] = ()
    updated_paths: tuple[str, ...] = ()
    removed_paths: tuple[str, ...] = ()

    @property
    def changed(self) -> int:
        return self.added + self.updated + self.removed


@dataclass(frozen=True)
class IndexStats:
    files: int
    symbols: int
    refs: int
    languages: list[tuple[str, int]]  # (язык, файлов) по убыванию
    complete: bool
    updated_at: float | None  # время последнего полного обновления (epoch)
    db_bytes: int


@dataclass(frozen=True)
class FileEntry:
    path: str
    language: str | None
    lines: int
    status: str
    symbols: int = 0


@dataclass(frozen=True)
class SymbolHit:
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
    definitions: list[SymbolHit]
    hits: list[RefHit]  # первые ``limit``
    total: int


@dataclass(frozen=True)
class ImportEntry:
    module: str
    name: str
    alias: str
    line: int
    target: str | None  # файл проекта, куда ведёт импорт (None — внешний/не найден)


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _under(prefix: str) -> tuple[str, list[str]]:
    """SQL-условие «путь равен ``prefix`` или лежит под ним» (пустой — весь проект)."""
    if not prefix:
        return "1", []
    # не LIKE: он не различает регистр ASCII, а `Src/` и `src/` — разные каталоги
    return "(path = ? OR substr(path, 1, ?) = ?)", [prefix, len(prefix) + 1, prefix + "/"]


def _count_lines(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


class ProjectIndex:
    """Индекс проекта. Конструктор не трогает ФС; база открывается в :meth:`open`.

    Использование::

        with ProjectIndex(workspace) as index:
            index.refresh()
            hits, total = index.find_symbols("Agent")
    """

    def __init__(self, workspace: Workspace):
        self._ws = workspace
        self._conn: sqlite3.Connection | None = None
        self._modules: dict[str, str | None] = {}  # кэш resolve_module до refresh

    # ------------------------------------------------------------------ #
    @property
    def path(self) -> Path:
        return self._ws.index_dir / INDEX_FILE_NAME

    @property
    def root(self) -> Path:
        return self._ws.root

    def __enter__(self) -> ProjectIndex:
        return self.open()

    def __exit__(self, *exc: object) -> None:
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
        conn, self._conn = self._conn, None
        if conn is not None:
            conn.close()

    @property
    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            self.open()
        assert self._conn is not None
        return self._conn

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.OperationalError:
            pass  # ФС без поддержки WAL — работаем в режиме по умолчанию
        except BaseException:
            conn.close()
            raise
        return conn

    def _connect_checked(self) -> sqlite3.Connection:
        conn = self._connect()
        try:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
            if not tables:
                conn.executescript(_SCHEMA)
                with conn:
                    conn.execute(
                        "INSERT INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
                    )
                return conn
            row = None
            if "meta" in tables:
                row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
            if row is not None and row[0] == str(SCHEMA_VERSION):
                return conn
        except BaseException:
            conn.close()
            raise
        conn.close()
        raise sqlite3.DatabaseError("несовместимая версия индекса")

    def discard(self) -> None:
        """Удаляет базу (повреждённую, см. :func:`is_corruption`): следующее открытие
        создаст пустую, а обновление построит индекс заново."""
        self.close()
        self._recreate_file()
        self._modules.clear()

    def _recreate_file(self) -> None:
        for suffix in ("", "-wal", "-shm", "-journal"):
            Path(f"{self.path}{suffix}").unlink(missing_ok=True)

    # ------------------------------ meta ------------------------------- #
    def _meta(self, key: str) -> str | None:
        row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self._db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))

    def is_complete(self) -> bool:
        """Индекс существует и полностью построен (не создаёт базу, если её нет)."""
        if not self.path.is_file():
            return False
        return self._meta("complete") == "1"

    # ---------------------------- обновление ---------------------------- #
    def _rel(self, path: Path) -> str:
        rel = path.relative_to(self.root).as_posix()
        return "" if rel == "." else rel

    def _candidates(self, base: Path) -> Iterable[Path]:
        if is_excluded(self.root, base):
            return ()
        if base.is_dir():
            return walk_files(self.root, base)
        return (base,) if base.exists() else ()

    def refresh(
        self, base: Path | None = None, *, progress: Callable[[int], None] | None = None
    ) -> RefreshStats:
        """Синхронизирует индекс с файлами под ``base`` (по умолчанию — весь проект).

        ``progress`` получает число просмотренных файлов. Прерывание (в т.ч. Ctrl+C)
        откатывает только незафиксированную пачку изменений.
        """
        started = time.monotonic()
        db = self._db
        base = self.root if base is None else base
        full = base == self.root
        prefix = self._rel(base)
        cond, args = _under(prefix)
        # файлы с ошибкой чтения перечитываются, даже если не менялись (права могли починить)
        known = {
            path: (size, mtime) if status != STATUS_ERROR else None
            for path, size, mtime, status in db.execute(
                f"SELECT path, size, mtime_ns, status FROM files WHERE {cond}", args
            )
        }
        scanned = added = updated = pending = 0
        seen: set[str] = set()
        new: list[str] = []
        changed: list[str] = []
        gone: list[str] = []
        try:
            if full:
                with db:
                    self._set_meta("complete", "0")
            for path in self._candidates(base):
                # Секреты — по правилу read_file и классификатора команд (не только .env).
                if is_secret_entry(self.root, path):
                    continue
                # симлинк на исключённое (.gitignore, служебные каталоги) — тоже мимо
                if path.is_symlink() and is_excluded(self.root, path.resolve()):
                    continue
                try:
                    st = path.stat()
                except OSError:
                    continue
                if not stat.S_ISREG(st.st_mode):
                    continue
                rel = self._rel(path)
                seen.add(rel)
                scanned += 1
                if progress is not None:
                    progress(scanned)
                old = known.get(rel)
                if old == (st.st_size, st.st_mtime_ns):
                    continue
                self._index_file(rel, path, st)
                (changed if rel in known else new).append(rel)
                if rel not in known:
                    added += 1
                else:
                    updated += 1
                pending += 1
                if pending >= _BATCH:
                    db.commit()
                    pending = 0
            gone = [p for p in known if p not in seen]
            for chunk in _chunks(gone, 500):
                marks = ",".join("?" * len(chunk))
                for table in ("symbols", "refs", "imports", "files"):
                    db.execute(f"DELETE FROM {table} WHERE path IN ({marks})", chunk)
            if full:
                self._set_meta("complete", "1")
                self._set_meta("updated_at", str(time.time()))
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            self._modules.clear()
            # и при прерывании: лишнее уведомление безвредно, пропущенное — нет
            _notify_changes(self.root, tuple(new), tuple(changed), tuple(gone))
        return RefreshStats(
            scanned=scanned,
            added=added,
            updated=updated,
            removed=len(gone),
            duration_s=time.monotonic() - started,
            added_paths=tuple(new),
            updated_paths=tuple(changed),
            removed_paths=tuple(gone),
        )

    def _index_file(self, rel: str, path: Path, st) -> None:
        language = language_of(rel)
        status, lines, facts = STATUS_INDEXED, 0, None
        if st.st_size > MAX_INDEX_FILE_BYTES:
            status = STATUS_LARGE
        else:
            try:
                data = path.read_bytes()
            except OSError:
                data, status = b"", STATUS_ERROR
            if b"\0" in data[:_BINARY_PROBE]:
                status = STATUS_BINARY
            elif status == STATUS_INDEXED:
                text = data.decode("utf-8", errors="replace")
                lines = _count_lines(text)
                try:
                    facts = extract(text, language, rel)
                except Exception:  # разбор — эвристика: сбой на одном файле не валит индекс
                    status = STATUS_ERROR
        db = self._db
        for table in ("symbols", "refs", "imports"):
            db.execute(f"DELETE FROM {table} WHERE path = ?", (rel,))
        db.execute(
            "INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?, ?, ?)",
            (rel, st.st_size, st.st_mtime_ns, language, lines, status),
        )
        if facts is None:
            return
        db.executemany(
            "INSERT INTO symbols VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    rel,
                    s.name,
                    s.name.lower(),
                    s.parent,
                    s.parent.lower(),
                    s.kind,
                    s.line,
                    s.end_line,
                    s.depth,
                    s.signature,
                    s.doc,
                    s.col,
                )
                for s in facts.symbols
            ],
        )
        db.executemany(
            "INSERT INTO refs VALUES (?, ?, ?, ?, ?, ?)",
            [(rel, r.name, r.kind, r.line, r.col, r.scope) for r in facts.refs],
        )
        db.executemany(
            "INSERT INTO imports VALUES (?, ?, ?, ?, ?)",
            [(rel, i.module, i.name, i.alias, i.line) for i in facts.imports],
        )

    def rebuild(self, *, progress: Callable[[int], None] | None = None) -> RefreshStats:
        """Удаляет базу и строит индекс заново."""
        self.close()
        self._recreate_file()
        self.open()
        return self.refresh(progress=progress)

    # ------------------------------ запросы ----------------------------- #
    def stats(self) -> IndexStats:
        db = self._db
        files = db.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        symbols = db.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
        refs = db.execute("SELECT COUNT(*) FROM refs").fetchone()[0]
        languages = [
            (lang, n)
            for lang, n in db.execute(
                "SELECT language, COUNT(*) AS n FROM files WHERE language IS NOT NULL "
                "GROUP BY language ORDER BY n DESC, language"
            )
        ]
        updated = self._meta("updated_at")
        try:
            size = sum(
                Path(f"{self.path}{s}").stat().st_size
                for s in ("", "-wal")
                if Path(f"{self.path}{s}").exists()
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
        where = ["name_lower LIKE ? ESCAPE '\\'"]
        args: list[object] = [f"%{_escape_like(query)}%"]
        if parent:
            where.append("(parent_lower = ? OR parent_lower LIKE ? ESCAPE '\\')")
            args += [parent, f"%.{_escape_like(parent)}"]
        if kind:
            kinds = KIND_GROUPS.get(kind.strip().lower(), (kind.strip().lower(),))
            where.append(f"kind IN ({','.join('?' * len(kinds))})")
            args += list(kinds)
        sql = (
            "SELECT path, name, kind, line, end_line, parent, depth, signature, doc, col, "
            "CASE WHEN name_lower = ? THEN 0 WHEN name_lower LIKE ? ESCAPE '\\' THEN 1 "
            "ELSE 2 END AS rank "
            f"FROM symbols WHERE {' AND '.join(where)} "
            "ORDER BY rank, length(name), path, line"
        )
        params = [query, _escape_like(query) + "%", *args]
        hits: list[SymbolHit] = []
        total = 0
        for row in self._db.execute(sql, params):
            if path_glob and not glob_match(row[0], path_glob):
                continue
            total += 1
            if len(hits) < limit:
                hits.append(SymbolHit(row[0], _symbol(row[1:10])))
            if total >= _COUNT_CAP:
                break
        return hits, total

    def file_entry(self, rel_path: str) -> FileEntry | None:
        row = self._db.execute(
            "SELECT path, language, lines, status FROM files WHERE path = ?", (rel_path,)
        ).fetchone()
        if row is None:
            return None
        count = self._db.execute(
            "SELECT COUNT(*) FROM symbols WHERE path = ?", (rel_path,)
        ).fetchone()[0]
        return FileEntry(*row, symbols=count)

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
        self, rel_path: str, line: int, names: Iterable[str], *, col: int | None = None
    ) -> tuple[str, str] | None:
        """(вид, область) использования одного из ``names`` в строке (и столбце, если
        задан); из нескольких — вызов первым."""
        names = list(names)
        if not names:
            return None
        marks = ",".join("?" * len(names))
        at_col = "" if col is None else " AND col = ?"
        row = self._db.execute(
            "SELECT kind, scope FROM refs "
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
        return [_symbol(r) for r in rows]

    # ------------------------- использования, импорты ------------------------- #
    def resolve_module(self, module: str) -> str | None:
        """Файл проекта с модулем ``a.b.c``: ``…/a/b/c.py``, ``.pyi`` или ``…/__init__.py``.

        Совпадение — по суффиксу пути (корень пакетов, например ``src/``, не нужен);
        из нескольких — ближайший к корню, ``.py`` раньше ``.pyi``.
        """
        if module in self._modules:
            return self._modules[module]
        found: str | None = None
        if module and not module.startswith("."):
            base = module.replace(".", "/")
            candidates = []
            for suffix in (
                f"{base}.py",
                f"{base}/__init__.py",
                f"{base}.pyi",
                f"{base}/__init__.pyi",
            ):
                rows = self._db.execute(
                    "SELECT path FROM files WHERE path = ? OR path LIKE ? ESCAPE '\\'",
                    (suffix, f"%/{_escape_like(suffix)}"),
                )
                # LIKE не различает регистр ASCII — сверяем точно
                candidates += [r[0] for r in rows if r[0] == suffix or r[0].endswith(f"/{suffix}")]
            if candidates:
                found = min(candidates, key=lambda p: (p.count("/"), p.endswith(".pyi"), p))
        self._modules[module] = found
        return found

    def _import_target(self, module: str, name: str) -> str | None:
        """Куда ведёт импорт: ``from pkg import mod`` — подмодуль, иначе сам модуль."""
        if name and name != "*":
            sub = self.resolve_module(f"{module}.{name}" if module else name)
            if sub is not None:
                return sub
        return self.resolve_module(module)

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
        modules = _module_suffixes(parts)
        parents = _module_suffixes(parts[:-1]) if len(parts) > 1 else [""]
        marks = ",".join("?" * len(modules))
        pmarks = ",".join("?" * len(parents))
        rows = self._db.execute(
            "SELECT path, module, name, line FROM imports "
            f"WHERE module IN ({marks}) OR (module IN ({pmarks}) AND name = ?) "
            "ORDER BY path, line",
            [*modules, *parents, parts[-1]],
        )
        found: dict[str, int] = {}
        for path, module, name, line in rows:
            if path != rel_path and path not in found:
                if self._import_target(module, name) == rel_path:
                    found[path] = line
        return sorted(found.items())

    def definitions(self, query: str) -> list[SymbolHit]:
        """Определения с точным именем (без учёта регистра): ``name`` или ``Class.method``."""
        if not query.strip(" ."):
            raise ValueError("пустой запрос")
        name = _last_name(query)
        hits, _ = self.find_symbols(query.strip(), limit=_COUNT_CAP)
        return [h for h in hits if h.symbol.name.lower() == name.lower()]

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
        name = _last_name(query)
        names = sorted({h.symbol.name for h in definitions}) or [name]
        def_paths = {h.path for h in definitions}
        kinds = None if not kind else {kind.strip().lower()}

        marks = ",".join("?" * len(names))
        imported = self._db.execute(
            f"SELECT path, line, alias FROM imports WHERE name IN ({marks})", names
        ).fetchall()
        rows: list[tuple[str, int, int, str, str]] = []
        if kinds is None or kinds - {REF_IMPORT}:
            rows += self._db.execute(
                f"SELECT path, line, col, kind, scope FROM refs WHERE name IN ({marks})", names
            ).fetchall()
            # `from m import f as g` — в этом файле f используется как g
            for path, alias in {(p, a) for p, _, a in imported if a not in names}:
                rows += self._db.execute(
                    "SELECT path, line, col, kind, scope FROM refs WHERE name = ? AND path = ?",
                    (alias, path),
                ).fetchall()
        if kinds is None or REF_IMPORT in kinds:
            rows += [(path, line, 0, REF_IMPORT, "") for path, line, _ in imported]

        imports_def: dict[str, bool] = {}

        def resolution(path: str) -> str:
            if path in def_paths:
                return RESOLVED_SAME_FILE
            if path not in imports_def:
                imports_def[path] = any(e.target in def_paths for e in self.imports_of(path))
            return RESOLVED_IMPORT if imports_def[path] else RESOLVED_NAME

        def_positions = {(h.path, h.symbol.line, h.symbol.col) for h in definitions}
        out: list[RefHit] = []
        for path, line, col, ref_kind, scope in rows:
            if kinds is not None and ref_kind not in kinds:
                continue
            if path_glob and not glob_match(path, path_glob):
                continue
            # имя в самом определении — не использование (рекурсивный вызов в той же строке — да)
            if (path, line, col) in def_positions:
                continue
            out.append(RefHit(path, line, col, ref_kind, scope, resolution(path)))
        out.sort(key=lambda h: (_RESOLUTION_ORDER[h.resolution], h.path, h.line, h.col))
        return RefsResult(definitions=definitions[:limit], hits=out[:limit], total=len(out))

    def files_under(self, rel_dir: str) -> Iterator[FileEntry]:
        """Файлы каталога (рекурсивно) с числом символов, по пути."""
        cond, args = _under(rel_dir)
        cond = cond.replace("path", "f.path")
        rows = self._db.execute(
            "SELECT f.path, f.language, f.lines, f.status, "
            "(SELECT COUNT(*) FROM symbols s WHERE s.path = f.path) "
            f"FROM files f WHERE {cond} ORDER BY f.path",
            args,
        )
        for path, language, lines, status, count in rows:
            yield FileEntry(path, language, lines, status, symbols=count)


def _last_name(query: str) -> str:
    query = query.strip().strip(".")
    return query.rsplit(".", 1)[-1]


def _module_suffixes(parts: list[str]) -> list[str]:
    """``[a, b, c]`` → ``a.b.c``, ``b.c``, ``c`` (корень пакетов неизвестен)."""
    return [".".join(parts[i:]) for i in range(len(parts))]


def _symbol(row) -> Symbol:
    name, kind, line, end_line, parent, depth, signature, doc, col = row
    return Symbol(
        name=name,
        kind=kind,
        line=line,
        end_line=end_line,
        parent=parent,
        depth=depth,
        signature=signature,
        doc=doc,
        col=col,
    )


def _chunks(items: list[str], size: int) -> Iterator[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
