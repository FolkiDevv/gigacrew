"""Ранжированная карта проекта: самые важные определения в пределах бюджета.

Алгоритм на таблицах индекса:

* граф файлов: ребро «файл, где имя используется → файл, где оно определено» для
  каждого общего имени (``refs`` × ``symbols``);
* вес ребра: ×10 — имя из фокуса; ×10 — длинное составное имя (``snake_case``,
  ``CamelCase``, от 8 символов); ×0,1 — ``_приватное``; ×0,1 — имя определено
  больше чем в 5 файлах (``run``); ×0,1 — имя совпадает с методом встроенных типов
  (``get``, ``items``: ``d.get()`` по имени не отличить от своего ``get``); ×50 —
  ссылка из файла в фокусе; ×√(число использований);
* персонализированный PageRank (телепорт — в файлы фокуса, без фокуса —
  равномерно); ранг файла делится между его рёбрами и достаётся определениям;
* в карту попадает столько лучших определений, сколько влезает в бюджет (двоичный
  поиск по числу).

Определения, которые нигде не используются, идут после используемых — по рангу
своего файла (иначе файлы без связей, например на языках без использований в
индексе, в карту не попали бы вовсе). В PageRank они не участвуют: петля отдала бы
им весь ранг файла.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from gigacrew.project.index import ProjectIndex
from gigacrew.project.symbols import Symbol

CHARS_PER_TOKEN = 3  # грубая оценка, как у бюджета контекста агента
DEFAULT_MAP_CHARS = 6_000
_DAMPING = 0.85
_MAX_ITERATIONS = 100
_TOLERANCE = 1e-8
_SKIP_KINDS = ("section",)  # заголовки Markdown — не определения кода
# Методы встроенных коллекций и строк: вызовы ``x.get()`` чаще всего не про свой ``get``.
_COMMON_METHODS = frozenset(
    name for t in (dict, list, set, str, bytes) for name in dir(t) if not name.startswith("_")
)


@dataclass(frozen=True)
class RankedDefinition:
    path: str
    name: str
    rank: float


@dataclass(frozen=True)
class RepoMap:
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
    names = sorted(set(nodes) | set(edges) | {v for t in edges.values() for v in t})
    if not names:
        return {}
    if personalization:
        total = sum(max(personalization.get(n, 0.0), 0.0) for n in names)
    else:
        total = 0.0
    if total > 0:
        teleport = {n: max(personalization.get(n, 0.0), 0.0) / total for n in names}  # type: ignore[union-attr]
    else:
        teleport = {n: 1.0 / len(names) for n in names}
    out_weight = {n: sum(edges.get(n, {}).values()) for n in names}
    rank = dict.fromkeys(names, 1.0 / len(names))
    for _ in range(_MAX_ITERATIONS):
        dangling = sum(rank[n] for n in names if out_weight[n] <= 0)
        new = {n: (1 - damping + damping * dangling) * teleport[n] for n in names}
        for u, targets in edges.items():
            if out_weight[u] <= 0:
                continue
            share = damping * rank[u] / out_weight[u]
            for v, w in targets.items():
                new[v] += share * w
        error = sum(abs(new[n] - rank[n]) for n in names)
        rank = new
        if error < _TOLERANCE * len(names):
            break
    return rank


# --------------------------------------------------------------------------- #
# Ранжирование определений
# --------------------------------------------------------------------------- #
def _is_compound(name: str) -> bool:
    letters = name.strip("_")
    snake = "_" in letters
    camel = any(c.isupper() for c in letters) and any(c.islower() for c in letters)
    return (snake or camel) and len(name) >= 8


def rank_definitions(
    index: ProjectIndex,
    *,
    focus_files: Iterable[str] = (),
    focus_names: Iterable[str] = (),
) -> list[RankedDefinition]:
    """Определения (файл, имя) по убыванию важности."""
    db = index._db  # noqa: SLF001 — карта — часть слоя индекса, агрегаты нужны на SQL
    marks = ",".join("?" * len(_SKIP_KINDS))
    defines: dict[str, set[str]] = defaultdict(set)
    for path, name in db.execute(
        f"SELECT DISTINCT path, name FROM symbols WHERE kind NOT IN ({marks})", _SKIP_KINDS
    ):
        defines[name].add(path)
    if not defines:
        return []
    references: dict[str, dict[str, int]] = defaultdict(dict)
    for path, name, count in db.execute(
        "SELECT r.path, r.name, COUNT(*) FROM refs r "
        "WHERE r.name IN (SELECT name FROM symbols) GROUP BY r.path, r.name"
    ):
        if name in defines:
            references[name][path] = count

    focus = set(focus_files)
    mentioned = set(focus_names)
    # вес ребра по каждому имени: (откуда, куда, имя) -> вес
    weights: dict[tuple[str, str, str], float] = defaultdict(float)
    unused: list[tuple[str, str]] = []  # (файл, имя) — определено, но не используется
    for name, definers in defines.items():
        users = references.get(name)
        if not users:
            unused += [(definer, name) for definer in definers]
            continue
        mul = 1.0
        if name in mentioned:
            mul *= 10
        if _is_compound(name):
            mul *= 10
        if name.startswith("_"):
            mul *= 0.1
        if len(definers) > 5:
            mul *= 0.1
        if name in _COMMON_METHODS:
            mul *= 0.1
        for user, count in users.items():
            use_mul = mul * (50 if user in focus else 1)
            for definer in definers:
                weights[(user, definer, name)] += use_mul * math.sqrt(count)

    edges: dict[str, dict[str, float]] = defaultdict(dict)
    out_total: dict[str, float] = defaultdict(float)
    for (src, dst, _), weight in weights.items():
        edges[src][dst] = edges[src].get(dst, 0.0) + weight
        out_total[src] += weight
    nodes = {p for paths in defines.values() for p in paths}
    personalization = {p: 1.0 for p in focus if p in nodes or p in edges} or None
    ranks = pagerank(edges, nodes, personalization)

    ranked: dict[tuple[str, str], float] = defaultdict(float)
    for (src, dst, name), weight in weights.items():
        ranked[(dst, name)] += ranks.get(src, 0.0) * weight / out_total[src]
    used = sorted(
        (RankedDefinition(path, name, rank) for (path, name), rank in ranked.items()),
        key=lambda d: (-d.rank, d.path, d.name),
    )
    per_file: dict[str, int] = defaultdict(int)
    for path, _ in unused:
        per_file[path] += 1
    rest = sorted(
        (
            RankedDefinition(path, name, ranks.get(path, 0.0) / per_file[path] * 1e-6)
            for path, name in unused
        ),
        key=lambda d: (-d.rank, d.path, d.name),
    )
    return used + rest


# --------------------------------------------------------------------------- #
# Отрисовка
# --------------------------------------------------------------------------- #
def _render(
    index: ProjectIndex, chosen: list[RankedDefinition], outlines: dict[str, list[Symbol]]
) -> tuple[str, int, int]:
    """Файлы в порядке важности, в каждом — выбранные определения по строкам.

    Для метода показывается и его класс (контекст), пропущенные определения файла —
    строкой «⋮ ещё N». ``outlines`` — кэш оглавлений между попытками.
    """
    by_file: dict[str, set[str]] = {}
    for d in chosen:
        by_file.setdefault(d.path, set()).add(d.name)
    out: list[str] = []
    shown = 0
    for path, names in by_file.items():
        if path not in outlines:
            outlines[path] = [s for s in index.outline(path) if s.kind not in _SKIP_KINDS]
        symbols = outlines[path]
        picked = {s for s in symbols if s.name in names}
        parents = {s.parent for s in picked if s.parent}
        keep = [s for s in symbols if s in picked or s.qualname in parents]
        out.append(f"{path}:")
        out += [_symbol_line(s) for s in keep]
        rest = len(symbols) - len(keep)
        if rest > 0:
            out.append(f"  ⋮ ещё {rest}")
        shown += len(picked)
    return "\n".join(out), shown, len(by_file)


def _symbol_line(symbol: Symbol) -> str:
    text = symbol.signature or f"{symbol.kind} {symbol.name}"
    return f"{symbol.line:>6}  {'  ' * symbol.depth}{text}"


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
