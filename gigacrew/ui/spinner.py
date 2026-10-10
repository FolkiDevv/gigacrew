"""Индикатор ожидания: ракета пролетает через строку, оставляя мерцающее пламя.

Кадры генерируются один раз при импорте из генератора с фиксированным seed:
анимация выглядит «живой» (след пульсирует, символы и цвета мерцают, за ракетой
остаются гаснущие искры), но одинакова при каждом запуске.
"""

from __future__ import annotations

import random

from rich.text import Text

from gigacrew.ui.theme import BRAND, DANGER, FLAME, MUTED, WARN

WIDTH = 8  # ширина полосы полёта, в колонках
INTERVAL = 0.11  # длительность кадра, с

_ROCKET = ("➤", BRAND)
# Варианты пламени по удалённости от сопла: ближе — горячее.
_FLAME: tuple[tuple[tuple[str, str], ...], ...] = (
    (("≈", FLAME), ("≈", WARN), ("=", FLAME)),
    (("~", WARN), ("≈", DANGER), ("=", DANGER)),
    (("-", DANGER), ("~", MUTED)),
    (("·", MUTED), ("-", MUTED), (" ", MUTED)),
)
_SPARK = (("·", WARN), ("˙", MUTED))  # искра на месте: вспыхнула → гаснет
_SPARK_CHANCE = 0.45
_BLANK = (" ", "")


def _build_frames(seed: int = 11, passes: int = 3) -> tuple[Text, ...]:
    rnd = random.Random(seed)
    frames: list[list[tuple[str, str]]] = []
    for _ in range(passes):
        sparks: list[tuple[int, int]] = []  # (позиция, возраст)
        # Пролёт по клетке за кадр, затем два кадра догорают искры.
        for x in [*range(WIDTH + 4), None, None]:
            cells = [_BLANK] * WIDTH
            sparks = [(pos, age + 1) for pos, age in sparks if age + 1 < len(_SPARK)]
            for pos, age in sparks:
                if 0 <= pos < WIDTH:
                    cells[pos] = _SPARK[age]
            if x is not None:
                length = rnd.randint(2, 4)
                for i in range(1, length + 1):
                    if 0 <= x - i < WIDTH:
                        cells[x - i] = rnd.choice(_FLAME[min(i, len(_FLAME)) - 1])
                if rnd.random() < _SPARK_CHANCE:
                    sparks.append((x - length - 1, -1))
                if x < WIDTH:
                    cells[x] = _ROCKET
            frames.append(cells)
        frames.append([_BLANK] * WIDTH)  # пауза между пролётами
    return tuple(Text.assemble(*cells) for cells in frames)


FRAMES = _build_frames()


def rocket_frame(elapsed: float) -> Text:
    """Кадр анимации для момента ``elapsed`` секунд от её начала."""
    return FRAMES[int(elapsed / INTERVAL) % len(FRAMES)]
