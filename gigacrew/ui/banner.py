"""Арт стартового экрана: пиксельная ракета и крупное название."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from rich.text import Text

from gigacrew.ui.theme import MINT, Palette

# Пиксель-арт ракеты: сетка 29×29, пиксель — два одинаковых символа (█ ▓ ▒ ░ — обводка,
# тёмная обводка, штриховка, светлое). Ракета летит вправо вверх, выхлоп — слева внизу.
_ART: tuple[str, ...] = (
    "                                                          ",
    "                                        ████████▓▓████    ",
    "                                  ██████████        ▓▓    ",
    "                              ▓▓██▓▓██              ▓▓    ",
    "                          ██▓▓▓▓                  ████    ",
    "                        ▓▓████      ▓▓██          ██      ",
    "                      ██▓▓        ▓▓██████      ████      ",
    "                    ▓▓▓▓        ████░░░░████    ██▓▓      ",
    "          ██████████▓▓          ████░░░░████  ██▓▓        ",
    "        ██▓▓    ▒▒██              ████████    ▓▓██        ",
    "        ▓▓    ▒▒██                  ████    ▓▓▓▓          ",
    "      ████  ▒▒██          ▓▓██            ▓▓██            ",
    "      ██▒▒  ▒▒▓▓  ░░    ▓▓▓▓▒▒██          ████    ░░      ",
    "    ██▓▓▒▒▒▒██        ████▒▒██▒▒        ████              ",
    "    ██████████▓▓    ▓▓▓▓▒▒▓▓▒▒        ████                ",
    "            ░░    ████▒▒██▒▒        ██▓▓                  ",
    "      ░░░░        ██▒▒██▒▒        ▓▓██                    ",
    "    ▒▒▒▒░░        ▒▒██▒▒        ██▒▒██                    ",
    "  ░░▒▒▒▒▒▒░░                ▓▓▓▓  ▒▒██                    ",
    "        ▒▒░░          ░░████▒▒    ▒▒▓▓                    ",
    "      ▒▒▒▒▒▒░░    ░░  ▒▒██▒▒      ▒▒██                    ",
    "    ▒▒▒▒▒▒▒▒░░░░▒▒░░  ▒▒██▒▒▒▒▓▓████                      ",
    "  ░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░▒▒██▓▓████                          ",
    "  ▒▒      ▒▒▒▒▒▒░░▒▒░░  ▓▓▓▓                              ",
    "        ▒▒▒▒▒▒  ▒▒▒▒                                      ",
    "        ▒▒░░    ▒▒░░                                      ",
    "                                                          ",
    "                                                          ",
    "                        ░░            ░░░░              ░░",
)
# Строки сетки, которые рисуются: без пустых сверху и снизу, звёзды — через одну пустую.
_ROCKET_ROWS_END = 26
_STARS_ROW = 28
_ROWS = (*range(1, _ROCKET_ROWS_END), _STARS_ROW)
# Выхлоп: со строки 15 — у левого края до колонки 7, с 18-й — шире, до колонки 10.
_EXHAUST_FIRST_ROW = 15
_EXHAUST_WIDE_ROW = 18
_EXHAUST_NARROW_LAST_COL = 7
_EXHAUST_WIDE_LAST_COL = 10
# Иллюминатор: строки 7–8, колонки 18–19.
_WINDOW_FIRST_ROW = 7
_WINDOW_LAST_ROW = 8
_WINDOW_FIRST_COL = 18
_WINDOW_LAST_COL = 19
_EMPTY = " "


def _pixels() -> list[tuple[int, str]]:
    """(номер строки сетки, пиксели строки — по символу на пиксель)."""
    return [(row, _ART[row][::2]) for row in _ROWS]


def _is_exhaust(row: int, col: int) -> bool:
    """Пиксель выхлопа: левый нижний угол сетки."""
    wide = row >= _EXHAUST_WIDE_ROW and col <= _EXHAUST_WIDE_LAST_COL
    narrow = _EXHAUST_FIRST_ROW <= row < _EXHAUST_WIDE_ROW and col <= _EXHAUST_NARROW_LAST_COL
    return wide or narrow


def _is_window(row: int, col: int) -> bool:
    """Пиксель иллюминатора."""
    in_rows = _WINDOW_FIRST_ROW <= row <= _WINDOW_LAST_ROW
    return in_rows and _WINDOW_FIRST_COL <= col <= _WINDOW_LAST_COL


def _color(palette: Palette, row: int, col: int, ch: str) -> str:
    """Цвет пикселя: корпус — знак палитры, выхлоп — тёплый, иллюминатор — светлый."""
    if ch == "█":
        return palette.brand
    if ch == "▓":
        return palette.rocket_deep
    exhaust = _is_exhaust(row, col)
    if ch == "▒":
        return palette.warn if exhaust else palette.rocket_shade
    if _is_window(row, col):
        return palette.fg if palette.soft == palette.brand else palette.soft
    return palette.danger if exhaust else palette.dim  # искры выхлопа и звёзды


def _pixel_color(palette: Palette, row: int, col: int, ch: str) -> str | None:
    """Цвет пикселя; None — пустой пиксель."""
    return None if ch == _EMPTY else _color(palette, row, col, ch)


def _half_block(upper: str | None, lower: str | None, *, mono: bool) -> tuple[str, str | None]:
    """Символ клетки и его стиль по цветам верхнего и нижнего пикселя."""
    if upper is None and lower is None:
        return _EMPTY, None
    if lower is None:
        return "▀", upper
    if upper is None:
        return "▄", lower
    if upper == lower or mono:
        return "█", upper
    return "▀", f"{upper} on {lower}"


def _append_cells(
    art: Text,
    palette: Palette,
    top_row: tuple[int, str],
    bottom_row: tuple[int, str],
    *,
    mono: bool,
) -> None:
    """Дописывает в ``art`` строку клеток из двух строк пикселей."""
    top_index, top = top_row
    bottom_index, bottom = bottom_row
    for col, (top_pixel, bottom_pixel) in enumerate(zip(top, bottom, strict=True)):
        symbol, style = _half_block(
            _pixel_color(palette, top_index, col, top_pixel),
            _pixel_color(palette, bottom_index, col, bottom_pixel),
            mono=mono,
        )
        art.append(symbol, style=style)


def rocket(p: Palette = MINT, *, mono: bool = False) -> Text:
    """Ракета полублоками ▀▄: в клетке терминала два пикселя друг над другом.

    Поэтому пиксели квадратные, а ракета — 29×13 клеток. ``mono`` — без цвета: два
    пикселя разного цвета в одной клетке иначе слились бы в верхний.
    """
    rows = _pixels()
    art = Text()
    for index in range(0, len(rows), 2):
        if index:
            art.append("\n")
        _append_cells(art, p, rows[index], rows[index + 1], mono=mono)
    return art


# Контурный шрифт (в духе figlet «future»): три строки на букву.
_FONT: Mapping[str, tuple[str, str, str]] = MappingProxyType({
    "g": ("┏━╸", "┃╺┓", "┗━┛"),
    "i": ("╻", "┃", "╹"),
    "a": ("┏━┓", "┣━┫", "╹ ╹"),
    "c": ("┏━╸", "┃  ", "┗━╸"),
    "r": ("┏━┓", "┣┳┛", "╹┗╸"),
    "e": ("┏━╸", "┣╸ ", "┗━╸"),
    "w": ("╻ ╻", "┃╻┃", "┗┻┛"),
})


def title(p: Palette = MINT) -> Text:
    """Название «gigacrew» крупными буквами: «giga» цветом знака, «crew» — основным."""
    word = (("giga", f"bold {p.brand}"), ("crew", f"bold {p.fg}"))
    text = Text()
    for row in range(3):
        if row:
            text.append("\n")
        for part, style in word:
            text.append("".join(_FONT[ch][row] for ch in part), style=style)
    return text
