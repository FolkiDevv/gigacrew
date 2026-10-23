"""Индикатор ожидания: ракета пролетает через строку, оставляя мерцающее пламя.

Кадры генерируются один раз на палитру из генератора с фиксированным seed:
анимация выглядит «живой» (след пульсирует, символы и цвета мерцают, за ракетой
остаются гаснущие искры), но одинакова при каждом запуске.
"""

from __future__ import annotations

import random
from functools import cache

from rich.text import Text

from gigacrew.ui.theme import MINT, Palette

WIDTH = 8  # ширина полосы полёта, в колонках
INTERVAL = 0.11  # длительность кадра, с

# Клетка — (символ, роль цвета в палитре).
type _Cell = tuple[str, str]
# Искра — (позиция, возраст).
type _Spark = tuple[int, int]

_ROCKET = ("➤", "brand")
# Варианты пламени по удалённости от сопла: ближе — горячее.
_FLAME: tuple[tuple[_Cell, ...], ...] = (
    (("≈", "flame"), ("≈", "warn"), ("=", "flame")),
    (("~", "warn"), ("≈", "danger"), ("=", "danger")),
    (("-", "danger"), ("~", "dim")),
    (("·", "dim"), ("-", "dim"), (" ", "dim")),
)
_SPARK = (("·", "warn"), ("˙", "dim"))  # искра на месте: вспыхнула → гаснет
_SPARK_CHANCE = 0.45
_BLANK = (" ", "")
# Пролёт по клетке за кадр (с вылетом за край полосы), затем два кадра догорают искры.
_FLIGHT = (*range(WIDTH + 4), None, None)


def _blank_cells() -> list[_Cell]:
    """Пустая полоса полёта."""
    return [_BLANK for _ in range(WIDTH)]


def _draw_sparks(cells: list[_Cell], sparks: list[_Spark]) -> None:
    """Рисует искры, попавшие в полосу."""
    for pos, age in sparks:
        if 0 <= pos < WIDTH:
            cells[pos] = _SPARK[age]


def _draw_rocket(
    cells: list[_Cell],
    sparks: list[_Spark],
    rnd: random.Random,
    nose: int,
) -> None:
    """Рисует ракету с пламенем в колонке ``nose``; за пламенем может вспыхнуть искра."""
    length = rnd.randint(2, 4)
    for distance in range(1, length + 1):
        if 0 <= nose - distance < WIDTH:
            cells[nose - distance] = rnd.choice(_FLAME[min(distance, len(_FLAME)) - 1])
    if rnd.random() < _SPARK_CHANCE:
        sparks.append((nose - length - 1, -1))
    if nose < WIDTH:
        cells[nose] = _ROCKET


def _flight_cells(rnd: random.Random) -> list[list[_Cell]]:
    """Клетки кадров одного пролёта ракеты."""
    flight: list[list[_Cell]] = []
    sparks: list[_Spark] = []
    for nose in _FLIGHT:
        cells = _blank_cells()
        sparks = [(pos, age + 1) for pos, age in sparks if age + 1 < len(_SPARK)]
        _draw_sparks(cells, sparks)
        if nose is not None:
            _draw_rocket(cells, sparks, rnd, nose)
        flight.append(cells)
    return flight


def _frame_text(palette: Palette, cells: list[_Cell]) -> Text:
    """Кадр из клеток в цветах палитры."""
    return Text.assemble(*((ch, getattr(palette, role) if role else "") for ch, role in cells))


def _build_frames(palette: Palette, seed: int = 11, passes: int = 3) -> tuple[Text, ...]:
    # Анимация, а не криптография: нужен воспроизводимый генератор с фиксированным seed.
    rnd = random.Random(seed)  # noqa: S311
    all_cells: list[list[_Cell]] = []
    for _ in range(passes):
        all_cells.extend(_flight_cells(rnd))
        all_cells.append(_blank_cells())  # пауза между пролётами
    return tuple(_frame_text(palette, cells) for cells in all_cells)


@cache
def frames(palette: Palette = MINT) -> tuple[Text, ...]:
    """Кадры анимации в цветах палитры."""
    return _build_frames(palette)


FRAMES = frames(MINT)


def rocket_frame(elapsed: float, palette: Palette = MINT) -> Text:
    """Кадр анимации для момента ``elapsed`` секунд от её начала."""
    shown = frames(palette)
    return shown[int(elapsed / INTERVAL) % len(shown)]
