"""Цвета терминала для приложений prompt_toolkit: палитра и глубина цвета.

rich сам понижает точные цвета, если терминал не сообщает о поддержке 24 бит
(смотрит на ``COLORTERM`` и ``TERM``). prompt_toolkit по умолчанию рисует в 256
цветах в любом терминале — меню вопросов и строка ввода выходили бы другого оттенка,
чем лента. Поэтому глубина цвета и палитра для prompt_toolkit берутся из той же
оценки терминала, что у rich.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from prompt_toolkit.output import ColorDepth
from prompt_toolkit.styles import Style
from rich.console import Console

from gigacrew.ui.theme import palette_for, prompt_styles

_DEPTHS: Mapping[str, ColorDepth] = MappingProxyType({
    "truecolor": ColorDepth.TRUE_COLOR,
    "256": ColorDepth.DEPTH_8_BIT,
    "standard": ColorDepth.DEPTH_4_BIT,
    "windows": ColorDepth.DEPTH_4_BIT,
})


def color_system() -> str | None:
    """Цветовая система терминала на stdout — как её определяет rich (None — не терминал)."""
    return Console().color_system


def prompt_style() -> Style:
    """Стили prompt_toolkit в палитре, подходящей терминалу."""
    return Style.from_dict(prompt_styles(palette_for(color_system())))


def prompt_color_depth(*, no_color: bool) -> ColorDepth | None:
    """Глубина цвета для prompt_toolkit: та же, что у rich.

    None — решает prompt_toolkit (не терминал); ``PROMPT_TOOLKIT_COLOR_DEPTH``
    и ``NO_COLOR`` по-прежнему главнее.
    """
    if no_color:
        return ColorDepth.MONOCHROME
    env = ColorDepth.from_env()
    if env is not None:
        return env
    return _DEPTHS.get(color_system() or "")
