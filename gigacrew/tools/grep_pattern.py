"""Образец grep в синтаксисе регулярных выражений Python — только там, где смысл тот же.

GNU grep понимает базовый (``-G``, по умолчанию) и расширенный (``-E``) синтаксис POSIX,
строку (``-F``) и Perl (``-P``). Перевод либо точный, либо его нет (None): тогда команда
остаётся командой. Отказ получают и образцы с вложенными повторами (``(a+)+``): поиск
Python идёт в процессе агента, и на таких образцах он может работать экспоненциально долго.
"""

from __future__ import annotations

import re

__all__ = ["translate"]

# В базовом синтаксисе эти символы — литералы, а с обратной косой — операторы; в Python
# наоборот.
_BASIC_OPERATORS = frozenset("|(){}?+")
_BASIC_ENDS = (r"\)", r"\|")  # после `$` — конец выражения в базовом синтаксисе
_WORD_EDGES = frozenset("<>")  # \< и \> — границы слова
# Последовательности с обратной косой, которые grep и Python понимают одинаково.
_SHARED_ESCAPES = frozenset("wWsSbB")
_QUANTIFIERS = frozenset("*+?")
# Повтор группы, при котором вложенный повтор опасен (следующий токен после `)`).
_REPEATS = frozenset("*+{")
_MARKS = frozenset("*+?{|")  # повтор или выбор внутри группы
# Токены регулярного выражения Python: экранированный знак, скобочное выражение, начало
# группы (с `?` у `(?:`, `(?=`…), любой другой знак.
_ATOM = re.compile(r"\\.|\[\^?\]?[^\]]*\]|\(\??|.", re.DOTALL)
# В скобках `[...]` Python видит в этих парах операции над множествами (и предупреждает).
_SET_OPERATIONS = ("--", "&&", "~~", "||")


def translate(pattern: str, syntax: str) -> str | None:
    """Образец grep с синтаксисом ``syntax`` (G, E, F, P) для Python; None — не переводится.

    Perl-синтаксис (``-P``) берётся как есть: Python понимает его почти так же, а то, чего
    не понимает, не скомпилируется.
    """
    if syntax == "F":
        return re.escape(pattern)
    regex = pattern if syntax == "P" else _posix(pattern, basic=syntax == "G")
    if regex is None or _backtracks(regex):
        return None
    return regex


def _posix(pattern: str, *, basic: bool) -> str | None:
    """Образец в синтаксисе POSIX (базовом или расширенном) — в синтаксис Python."""
    out: list[str] = []
    index = 0
    while index < len(pattern):
        # начало выражения — начало образца, после ( или |: от этого зависят * и ^
        start = not out or out[-1] in {"(", "|"}
        token = _token(pattern, index, basic=basic, start=start)
        if token is None:
            return None
        text, index = token
        # повтор повтора (`a+?`) и повтор в начале выражения значат в grep не то, что в Python
        if text in _QUANTIFIERS and (start or out[-1] in _QUANTIFIERS or out[-1] == "}"):
            return None
        out.append(text)
    return "".join(out)


def _token(pattern: str, index: int, *, basic: bool, start: bool) -> tuple[str, int] | None:
    """Очередной элемент образца в синтаксисе Python и позиция после него."""
    char = pattern[index]
    if char == "[":
        return _bracket(pattern, index)
    if char == "\\":
        return _escape(pattern, index, basic=basic)
    if basic and _basic_literal(pattern, index, start=start):
        return re.escape(char), index + 1
    return char, index + 1


def _basic_literal(pattern: str, index: int, *, start: bool) -> bool:
    """Символ базового синтаксиса — литерал, хотя в Python он оператор.

    ``| ( ) { } ? +`` — всегда; ``*`` — в начале выражения (``start``), ``^`` — не в
    начале; ``$`` — не в конце выражения.
    """
    char = pattern[index]
    if char in _BASIC_OPERATORS:
        return True
    if char == "*":
        return start
    if char == "^":
        return not start
    if char != "$":
        return False
    rest = pattern[index + 1 :]
    return bool(rest) and not rest.startswith(_BASIC_ENDS)


def _escape(pattern: str, index: int, *, basic: bool) -> tuple[str, int] | None:
    r"""Последовательность с обратной косой.

    ``\|`` в базовом синтаксисе — оператор, ``\<`` — граница слова, ``\w`` и подобные —
    как в Python, экранированный знак — литерал. Остальное (``\d``, ``\n``, ссылки ``\1``)
    в grep значит не то, что в Python, — None.
    """
    nxt = pattern[index + 1 : index + 2]
    end = index + 2
    if not nxt:
        return None
    if nxt in _WORD_EDGES:
        return r"\b", end
    if basic and nxt in _BASIC_OPERATORS:
        return nxt, end
    if nxt in _SHARED_ESCAPES:
        return pattern[index:end], end
    if nxt.isalnum():
        return None
    return re.escape(nxt), end


def _bracket(pattern: str, index: int) -> tuple[str, int] | None:
    r"""Скобочное выражение ``[...]``: обратная косая в нём — литерал (в Python — нет).

    Классы POSIX (``[[:space:]]``) и прочие ``[`` внутри — None.
    """
    end = _bracket_end(pattern, index)
    if end is None:
        return None
    inner = pattern[index + 1 : end]
    if "[" in inner or any(pair in inner for pair in _SET_OPERATIONS):
        return None
    escaped = inner.replace("\\", r"\\")
    return f"[{escaped}]", end + 1


def _bracket_end(pattern: str, index: int) -> int | None:
    """Позиция ``]``, закрывающей скобку на ``index``; None — не закрыта.

    ``]`` сразу после ``[`` или ``[^`` — литерал, а не конец.
    """
    pos = index + 1
    if pattern.startswith("^", pos):
        pos += 1
    if pattern.startswith("]", pos):
        pos += 1
    end = pattern.find("]", pos)
    return None if end < 0 else end


def _backtracks(regex: str) -> bool:
    """Повтор группы, внутри которой есть повтор или выбор (``(a+)+``, ``(a|aa)*``).

    На таких образцах поиск с возвратами бывает экспоненциальным. Проверка грубая: она
    отказывает и части безопасных образцов (``(ab|cd)+``) — им останется команда.
    """
    frames = [False]  # в открытых группах уже был повтор или выбор
    atoms = _ATOM.findall(regex)
    for pos, atom in enumerate(atoms):
        if atom.startswith("("):
            frames.append(False)
        elif atom == ")":
            following = "".join(atoms[pos + 1 : pos + 2])
            if not _close_group(frames, repeated=following in _REPEATS):
                return True
        elif atom in _MARKS:
            frames[-1] = True
    return False


def _close_group(frames: list[bool], *, repeated: bool) -> bool:
    """Закрывает группу; False — опасный повтор группы ``repeated`` или лишняя скобка.

    Лишняя скобка — образец не скомпилируется. Повтор или выбор внутри закрытой группы
    остаётся повтором и для внешней.
    """
    if len(frames) == 1:
        return False
    inner = frames.pop()
    frames[-1] = frames[-1] or inner
    return not (inner and repeated)
