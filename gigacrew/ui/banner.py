"""Арт стартового экрана: пиксельная ракета и крупное название."""

from __future__ import annotations

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
_ROWS = (*range(1, 26), 28)


def _pixels() -> list[tuple[int, str]]:
    """(номер строки сетки, пиксели строки — по символу на пиксель)."""
    return [(r, _ART[r][::2]) for r in _ROWS]


def _color(p: Palette, row: int, col: int, ch: str) -> str:
    """Цвет пикселя: корпус — знак палитры, выхлоп — тёплый, иллюминатор — светлый."""
    if ch == "█":
        return p.brand
    if ch == "▓":
        return p.rocket_deep
    exhaust = (row >= 18 and col <= 10) or (15 <= row <= 17 and col <= 7)
    if ch == "▒":
        return p.warn if exhaust else p.rocket_shade
    if 7 <= row <= 8 and 18 <= col <= 19:  # иллюминатор
        return p.soft if p.soft != p.brand else p.fg
    return p.danger if exhaust else p.dim  # искры выхлопа и звёзды


def rocket(p: Palette = MINT, *, mono: bool = False) -> Text:
    """Ракета полублоками ▀▄: в клетке терминала два пикселя друг над другом, поэтому
    пиксели квадратные, а ракета — 29×13 клеток. ``mono`` — без цвета: два пикселя
    разного цвета в одной клетке иначе слились бы в верхний."""
    rows = _pixels()
    art = Text()
    for i in range(0, len(rows), 2):
        if i:
            art.append("\n")
        (r_top, top), (r_bottom, bottom) = rows[i], rows[i + 1]
        for col, (a, b) in enumerate(zip(top, bottom, strict=True)):
            upper = _color(p, r_top, col, a) if a != " " else None
            lower = _color(p, r_bottom, col, b) if b != " " else None
            if upper is None and lower is None:
                art.append(" ")
            elif lower is None:
                art.append("▀", style=upper)
            elif upper is None:
                art.append("▄", style=lower)
            elif upper == lower or mono:
                art.append("█", style=upper)
            else:
                art.append("▀", style=f"{upper} on {lower}")
    return art


# Контурный шрифт (в духе figlet «future»): три строки на букву.
_FONT: dict[str, tuple[str, str, str]] = {
    "g": ("┏━╸", "┃╺┓", "┗━┛"),
    "i": ("╻", "┃", "╹"),
    "a": ("┏━┓", "┣━┫", "╹ ╹"),
    "c": ("┏━╸", "┃  ", "┗━╸"),
    "r": ("┏━┓", "┣┳┛", "╹┗╸"),
    "e": ("┏━╸", "┣╸ ", "┗━╸"),
    "w": ("╻ ╻", "┃╻┃", "┗┻┛"),
}


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
