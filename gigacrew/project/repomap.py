"""Ранжированная карта проекта: самые важные определения в пределах бюджета.

Алгоритм на таблицах индекса:

* граф файлов: ребро «файл, где имя используется → файл, где оно определено» для
  каждого общего имени (``refs`` × ``symbols``);
* вес ребра: ×10 — имя из фокуса; ×10 — длинное составное имя (``snake_case``,
  ``CamelCase``, от 8 символов); ×0,1 — ``_приватное``; ×0,1 — имя определено
  больше чем в 5 файлах (``run``); ×0,1 — имя совпадает с методом встроенных типов
  (``get``, ``items``: ``d.get()`` по имени не отличить от своего ``get``);
  ×√(число использований);
* персонализированный PageRank (телепорт — в файлы фокуса, без фокуса —
  равномерно); ранг файла делится между его рёбрами и достаётся определениям, а
  файл в фокусе отдаёт тем, что использует, ×50;
* пары «использует × определяет» не строятся (имя, определённое и вызываемое в
  тысячах файлов, дало бы их миллионы): время и память линейны;
* в карту попадает столько лучших определений, сколько влезает в бюджет (двоичный
  поиск по числу).

Определения, которые нигде не используются, идут после используемых — по рангу
своего файла (иначе файлы без связей, например на языках без использований в
индексе, в карту не попали бы вовсе). В PageRank они не участвуют: петля отдала бы
им весь ранг файла.
"""

from __future__ import annotations

import inspect
import math
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from gigacrew.project.index import ProjectIndex
from gigacrew.project.symbols import Symbol

CHARS_PER_TOKEN = 3  # грубая оценка, как у бюджета контекста агента
DEFAULT_MAP_CHARS = 6_000
_DAMPING = 0.85
_MAX_ITERATIONS = 100
_TOLERANCE = 1e-8
_FOCUS_SHARE = 50  # во сколько раз файл в фокусе щедрее к определениям, которые использует
_UNUSED_RANK_SCALE = 1e-6  # неиспользуемые определения — после всех используемых
_SKIP_KINDS = ("section",)  # заголовки Markdown — не определения кода
# Методы встроенных коллекций и строк: вызовы ``x.get()`` чаще всего не про свой ``get``.
_COMMON_METHODS = frozenset(
    name
    for builtin in (dict, list, set, str, bytes)
    for name, _member in inspect.getmembers(builtin)
    if not name.startswith("_")
)

# В запрос подставляются только плейсхолдеры «?»: сами виды передаются параметрами.
_SKIP_MARKS = ",".join("?" * len(_SKIP_KINDS))
_DEFINITIONS_SQL = f"SELECT DISTINCT path, name FROM symbols WHERE kind NOT IN ({_SKIP_MARKS})"  # noqa: S608
_REFERENCES_SQL = (
    "SELECT r.path, r.name, COUNT(*) FROM refs r "
    "WHERE r.name IN (SELECT name FROM symbols) GROUP BY r.path, r.name"
)


@dataclass(frozen=True)
class RankedDefinition:
    """Определение ``name`` из файла ``path`` и его ранг."""

    path: str
    name: str
    rank: float


@dataclass(frozen=True)
class RepoMap:
    """Отрисованная карта проекта и её счётчики."""

    text: str
    definitions: int  # показано определений
    files: int  # показано файлов
    total: int  # ранжировано определений


# --------------------------------------------------------------------------- #
# PageRank
# --------------------------------------------------------------------------- #
def pagerank(
    edges: Mapping[str, Mapping[str, float]],
    nodes: Iterable[str],
    personalization: Mapping[str, float] | None = None,
    *,
    damping: float = _DAMPING,
) -> dict[str, float]:
    """Взвешенный персонализированный PageRank степенным методом.

    ``edges[u][v]`` — вес ребра u→v. Телепорт и «висячие» вершины (без исходящих
    рёбер) распределяют ранг по ``personalization`` (без неё — равномерно).
    Сумма рангов — 1.
    """
    names = sorted(
        set(nodes) | set(edges) | {target for outgoing in edges.values() for target in outgoing},
    )
    if not names:
        return {}
    teleport = _teleport(names, personalization)
    out_weight = {name: sum(edges.get(name, {}).values()) for name in names}
    rank = dict.fromkeys(names, 1.0 / len(names))
    for _ in range(_MAX_ITERATIONS):
        new = _teleport_step(names, rank, teleport, out_weight, damping)
        for source, targets in edges.items():
            if out_weight[source] <= 0:
                continue
            share = damping * rank[source] / out_weight[source]
            for target, weight in targets.items():
                new[target] += share * weight
        if _converged(new, rank, names):
            return new
        rank = new
    return rank


def _teleport(
    names: Sequence[str],
    personalization: Mapping[str, float] | None,
) -> dict[str, float]:
    """Вероятности телепорта: по ``personalization`` (отрицательные — 0), без неё — поровну."""
    if personalization:
        weights = {name: max(personalization.get(name, 0), 0) for name in names}
        total = sum(weights.values())
        if total > 0:
            return {name: weight / total for name, weight in weights.items()}
    return {name: 1.0 / len(names) for name in names}


def _teleport_step(
    names: Sequence[str],
    rank: Mapping[str, float],
    teleport: Mapping[str, float],
    out_weight: Mapping[str, float],
    damping: float,
) -> dict[str, float]:
    """Ранг шага до раздачи по рёбрам: телепорт и ранг «висячих» вершин (без рёбер)."""
    dangling = sum(rank[name] for name in names if out_weight.get(name, 0) <= 0)
    return {name: (1 - damping + damping * dangling) * teleport[name] for name in names}


def _converged(
    new: Mapping[str, float],
    rank: Mapping[str, float],
    names: Sequence[str],
) -> bool:
    """Сошёлся ли степенной метод: суммарное изменение рангов — в пределах допуска."""
    error = sum(abs(new[name] - rank[name]) for name in names)
    return error < _TOLERANCE * len(names)


# --------------------------------------------------------------------------- #
# Ранжирование определений
# --------------------------------------------------------------------------- #
def _is_compound(name: str) -> bool:
    letters = name.strip("_")
    snake = "_" in letters
    camel = any(char.isupper() for char in letters) and any(char.islower() for char in letters)
    return (snake or camel) and len(name) >= 8


def _name_weight(name: str, definers: int, mentioned: set[str]) -> float:
    """Множитель веса рёбер имени (см. описание модуля)."""
    mul = 1.0
    if name in mentioned:
        mul *= 10
    if _is_compound(name):
        mul *= 10
    if name.startswith("_"):
        mul *= 0.1
    if definers > 5:
        mul *= 0.1
    if name in _COMMON_METHODS:
        mul *= 0.1
    return mul


def _load_definitions(db: sqlite3.Connection) -> dict[str, list[str]]:
    """Файлы, где определено каждое имя (без заголовков Markdown)."""
    found: dict[str, set[str]] = defaultdict(set)
    for path, defined in db.execute(_DEFINITIONS_SQL, _SKIP_KINDS):
        found[defined].add(path)
    return {name: sorted(paths) for name, paths in found.items()}


def _load_references(
    db: sqlite3.Connection,
    defines: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, int]]:
    """Число использований определённых имён по файлам: ``{имя: {файл: число}}``."""
    references: dict[str, dict[str, int]] = defaultdict(dict)
    for path, name, count in db.execute(_REFERENCES_SQL):
        if name in defines:
            references[name][path] = count
    return references


class _FileGraph:
    """Граф файлов по общим именам: от файлов, где имя используется, к файлам, где определено.

    Пары «использует × определяет» не строятся: имя, которое определено и вызывается
    в тысячах файлов (``handle``, ``String``), дало бы их миллионы. Хранятся веса
    рёбер по имени — ``uses[имя][файл]`` — и список файлов, где имя определено.
    """

    def __init__(
        self,
        defines: Mapping[str, Sequence[str]],
        references: Mapping[str, Mapping[str, int]],
        mentioned: set[str],
    ) -> None:
        self._defines = defines
        # Вес ребра «использующий файл → каждый определяющий» по имени: uses[имя][файл].
        self._uses: dict[str, dict[str, float]] = {}
        # Сумма исходящих рёбер файла.
        self._out_weight: defaultdict[str, float] = defaultdict(float)
        # (файл, имя) — определено, но не используется.
        self._unused: list[tuple[str, str]] = []
        for name, definers in defines.items():
            self._add_name(name, definers, references.get(name), mentioned)

    def file_ranks(self, focus: set[str]) -> dict[str, float]:
        """PageRank графа файлов без самого графа; телепорт — в файлы фокуса.

        Это :func:`pagerank` по рёбрам «файл, где имя используется → каждый файл, где оно
        определено» (вес — ``uses[name][user]``). Поток по имени считается один раз —
        сумма от всех использующих — и достаётся каждому определяющему: время и память
        линейны по числу использований и определений.
        """
        nodes = {path for paths in self._defines.values() for path in paths}
        personalization = {
            path: 1.0 for path in focus if path in nodes or path in self._out_weight
        } or None
        names = sorted(set(nodes) | set(self._out_weight))
        if not names:
            return {}
        total = sum(personalization.values()) if personalization else 0
        if personalization and total > 0:
            teleport = {name: personalization.get(name, 0) / total for name in names}
        else:
            teleport = {name: 1.0 / len(names) for name in names}
        rank = dict.fromkeys(names, 1.0 / len(names))
        for _ in range(_MAX_ITERATIONS):
            new = _teleport_step(names, rank, teleport, self._out_weight, _DAMPING)
            self._spread(new, rank)
            if _converged(new, rank, names):
                return new
            rank = new
        return rank

    def ranked_used(
        self,
        ranks: Mapping[str, float],
        focus: set[str],
    ) -> list[RankedDefinition]:
        """Используемые определения по убыванию ранга."""
        # Ранг файла делится между его рёбрами. Файл в фокусе отдаёт определениям, которые
        # использует, ×50: множитель на рёбрах внутри PageRank ничего бы не менял — и там,
        # и здесь вес ребра делится на сумму исходящих рёбер того же файла.
        boost = dict.fromkeys(focus, _FOCUS_SHARE)
        ranked: dict[tuple[str, str], float] = {}
        for name, users in self._uses.items():
            share = self._share(users, ranks, boost)
            for definer in self._defines[name]:
                ranked[definer, name] = share
        return sorted(
            (
                RankedDefinition(file_path, symbol_name, rank)
                for (file_path, symbol_name), rank in ranked.items()
            ),
            key=lambda definition: (-definition.rank, definition.path, definition.name),
        )

    def ranked_unused(self, ranks: Mapping[str, float]) -> list[RankedDefinition]:
        """Неиспользуемые определения — по рангу своего файла."""
        per_file = Counter(path for path, _name in self._unused)
        return sorted(
            (
                RankedDefinition(
                    path,
                    name,
                    ranks.get(path, 0) / per_file[path] * _UNUSED_RANK_SCALE,
                )
                for path, name in self._unused
            ),
            key=lambda definition: (-definition.rank, definition.path, definition.name),
        )

    def _add_name(
        self,
        name: str,
        definers: Sequence[str],
        users: Mapping[str, int] | None,
        mentioned: set[str],
    ) -> None:
        """Рёбра имени: от каждого использующего файла ко всем определяющим."""
        if not users:
            self._unused.extend((definer, name) for definer in definers)
            return
        mul = _name_weight(name, len(definers), mentioned)
        weights = {user: mul * math.sqrt(count) for user, count in users.items()}
        self._uses[name] = weights
        for user, weight in weights.items():
            self._out_weight[user] += weight * len(definers)

    def _spread(self, new: dict[str, float], rank: Mapping[str, float]) -> None:
        """Шаг PageRank по рёбрам: поток по имени достаётся каждому определяющему."""
        for name, users in self._uses.items():
            flow = _DAMPING * sum(
                rank[user] * weight / self._out_weight[user] for user, weight in users.items()
            )
            for definer in self._defines[name]:
                new[definer] += flow

    def _share(
        self,
        users: Mapping[str, float],
        ranks: Mapping[str, float],
        boost: Mapping[str, int],
    ) -> float:
        """Доля ранга, которую использующие файлы отдают каждому определению имени."""
        return sum(
            ranks.get(user, 0) * weight / self._out_weight[user] * boost.get(user, 1)
            for user, weight in users.items()
        )


def rank_definitions(
    index: ProjectIndex,
    *,
    focus_files: Iterable[str] = (),
    focus_names: Iterable[str] = (),
) -> list[RankedDefinition]:
    """Определения (файл, имя) по убыванию важности."""
    # Карта — часть слоя индекса: агрегаты нужны на SQL.
    db = index._db  # noqa: SLF001
    defines = _load_definitions(db)
    if not defines:
        return []
    references = _load_references(db, defines)
    focus = set(focus_files)
    graph = _FileGraph(defines, references, set(focus_names))
    ranks = graph.file_ranks(focus)
    return graph.ranked_used(ranks, focus) + graph.ranked_unused(ranks)


# --------------------------------------------------------------------------- #
# Отрисовка
# --------------------------------------------------------------------------- #
def _render(
    index: ProjectIndex,
    chosen: list[RankedDefinition],
    outlines: dict[str, list[Symbol]],
) -> tuple[str, int, int]:
    """Файлы в порядке важности, в каждом — выбранные определения по строкам.

    ``outlines`` — кэш оглавлений между попытками.
    """
    by_file: dict[str, set[str]] = {}
    for definition in chosen:
        by_file.setdefault(definition.path, set()).add(definition.name)
    out: list[str] = []
    shown = 0
    for path, names in by_file.items():
        if path not in outlines:
            outlines[path] = [
                symbol for symbol in index.outline(path) if symbol.kind not in _SKIP_KINDS
            ]
        lines, picked = _file_lines(outlines[path], names)
        out.append(f"{path}:")
        out += lines
        shown += picked
    return "\n".join(out), shown, len(by_file)


def _file_lines(symbols: Sequence[Symbol], names: set[str]) -> tuple[list[str], int]:
    """Строки определений файла в карте и число выбранных среди них.

    Для метода показывается и его класс (контекст), пропущенные определения файла —
    строкой «⋮ ещё N».
    """
    picked = {symbol for symbol in symbols if symbol.name in names}
    parents = {symbol.parent for symbol in picked if symbol.parent}
    lines: list[str] = []
    for symbol in symbols:
        if symbol in picked or symbol.qualname in parents:
            text = symbol.signature or f"{symbol.kind} {symbol.name}"
            indent = "  " * symbol.depth
            lines.append(f"{symbol.line:>6}  {indent}{text}")
    hidden = len(symbols) - len(lines)
    if hidden > 0:
        lines.append(f"  ⋮ ещё {hidden}")
    return lines, len(picked)


def build_repo_map(
    index: ProjectIndex,
    *,
    focus_files: Iterable[str] = (),
    focus_names: Iterable[str] = (),
    max_chars: int = DEFAULT_MAP_CHARS,
) -> RepoMap:
    """Карта: сколько лучших определений влезает в ``max_chars``."""
    ranked = rank_definitions(index, focus_files=focus_files, focus_names=focus_names)
    if not ranked:
        return RepoMap("", 0, 0, 0)
    outlines: dict[str, list[Symbol]] = {}
    lo, hi = 0, len(ranked)
    best = ("", 0, 0)
    while lo < hi:  # наибольшее k, при котором карта влезает
        mid = (lo + hi + 1) // 2
        rendered = _render(index, ranked[:mid], outlines)
        if len(rendered[0]) <= max_chars:
            best, lo = rendered, mid
        else:
            hi = mid - 1
    text, shown, files = best
    return RepoMap(text, shown, files, len(ranked))
