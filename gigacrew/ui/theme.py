"""Визуальный язык gigacrew: палитра и иконки.

Отдельный модуль, чтобы одни и те же цвета использовали вывод rich и стили
prompt_toolkit (строка ввода, меню вопросов, выбор чата). Библиотеки здесь не
импортируются: палитры — обычные данные, объекты строят потребители.

У каждого цвета палитры одна роль, роли перечислены от заметного к тихому: цвет
получает то, что ждёт пользователя или пошло не так, служебное приглушено. Точные
цвета рассчитаны на тёмный фон; основной текст — цвет терминала по умолчанию, он
читается на любом фоне. В терминале на 16 цветов пастель превратилась бы в белый
(rich подбирает ближайший цвет ANSI), поэтому там — запасная палитра из цветов
ANSI темы терминала (:func:`palette_for`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from types import MappingProxyType


@dataclass(frozen=True)
class Palette:
    """Цвета по ролям в формате стилей rich: ``#RRGGBB``, имя цвета ANSI или ``dim``."""

    warn: str  # внимание: вопрос пользователю, идущая работа, режим «авто», пламя
    danger: str  # ошибка, опасная команда, удалённые строки
    fg: str  # содержание: ответ модели, файлы и команды шагов
    brand: str  # знак: ✦ ❯ ●, «giga», ракета, выбранный пункт, заголовки Markdown
    soft: str  # тихий акцент: инлайн-код и ссылки в ответе модели, иллюминатор
    ok: str  # успех, добавленные строки
    muted: str  # служебное: итоги вызовов, цели группы, подписи
    dim: str  # фон: статистика хода, рецензент, подсказки клавиш, рамки, звёзды
    flame: str  # ядро пламени ракеты-индикатора
    rocket_deep: str  # ракета стартового экрана: тёмная обводка
    rocket_shade: str  # ракета стартового экрана: штриховка корпуса


MINT = Palette(
    warn="#F2CC8F",
    danger="#F4978E",
    fg="default",
    brand="#8EDDB0",
    soft="#BFEBD2",
    ok="#8EDDB0",
    # Контраст к тёмным темам (WCAG): служебное — от 4.6, фон — от 2.7 (Nord) до 3.7.
    muted="#979F99",
    dim="#6D7670",
    flame="#FBE3B0",
    rocket_deep="#5DA67F",
    rocket_shade="#3F7358",
)
# Терминал на 16 цветов: цвета из его темы, служебное — атрибутом dim.
MINT_ANSI = Palette(
    warn="yellow",
    danger="red",
    fg="default",
    brand="green",
    soft="green",
    ok="green",
    muted="dim",
    dim="bright_black",
    flame="yellow",
    rocket_deep="green",
    rocket_shade="green",
)


def palette_for(color_system: str | None) -> Palette:
    """Палитра для цветовой системы терминала — как её называет rich.

    Системы: ``truecolor``, ``256``, ``standard``, ``windows``; None — без цвета.
    """
    return MINT_ANSI if color_system in {"standard", "windows"} else MINT


ICON_TOOL = "●"
ICON_OK = "✔"
ICON_FAIL = "✘"
ICON_BRAND = "✦"
ICON_PENDING = "◌"  # шаг ещё выполняется
ICON_BAR = "▌"  # полоса слева у всего, что ждёт ответа пользователя
ICON_DANGER = "⚠"
SPINNER = "✦"
# Значки режимов разрешений (ключ — PermissionMode.value).
MODE_ICONS: Mapping[str, str] = MappingProxyType({"auto": "⏵⏵", "plan": "⏸"})


def markdown_styles(palette: Palette) -> dict[str, str]:
    """Стили Markdown-ответа модели (имена стилей rich: markdown.*)."""
    return {
        "markdown.h1": f"bold underline {palette.brand}",
        "markdown.h2": f"bold {palette.brand}",
        "markdown.h3": "bold",
        "markdown.h4": "bold italic",
        "markdown.h5": "italic",
        "markdown.h6": "italic",
        "markdown.code": palette.soft,  # inline-код — без фона, читается в любой теме
        "markdown.link": palette.soft,
        "markdown.link_url": f"underline {palette.soft}",
        "markdown.item.bullet": f"bold {palette.brand}",
        "markdown.item.number": f"bold {palette.brand}",
        "markdown.block_quote": f"italic {palette.muted}",
        "markdown.hr": palette.dim,
    }


def _pt(color: str) -> str:
    """Цвет rich — в цвет prompt_toolkit: имена ANSI с приставкой ``ansi``.

    Цвет терминала — явно (``ansidefault``): пустой стиль унаследовал бы цвет из
    стилей prompt_toolkit по умолчанию, у меню дополнения это чёрный текст.
    """
    if color.startswith("#"):
        return color
    if color == "default":
        return "ansidefault"
    if color == "dim":
        return "ansidefault dim"
    name = color.replace("_", "")
    return f"ansi{name}"


def _prompt_palette(palette: Palette) -> Palette:
    """Та же палитра, все цвета которой переведены в цвета prompt_toolkit."""
    return replace(
        palette,
        **{field.name: _pt(getattr(palette, field.name)) for field in fields(palette)},
    )


def prompt_styles(palette: Palette) -> dict[str, str]:
    """Стили строки ввода, меню вопросов и выбора чата (классы prompt_toolkit)."""
    colors = _prompt_palette(palette)
    ansi = palette is MINT_ANSI
    # Меню дополнения — на своём тёмном фоне, поэтому и текст в нём свой, светлый:
    # цвет терминала по умолчанию в светлой теме был бы тёмным на тёмном.
    if ansi:
        menu, text, meta = "bg:ansiblack", "ansiwhite", "ansiwhite"
        scrollbar, thumb = "bg:ansibrightblack", "bg:ansiwhite"
    else:
        menu, text, meta = "bg:#232826", "#E3E7E4", colors.muted
        scrollbar, thumb = "bg:#2F3532", f"bg:{colors.dim}"
    current_fg = "ansiblack" if ansi else "#111111"
    current = f"bg:{colors.brand} {current_fg}"
    return {
        "prompt": f"bold {colors.brand}",
        "continuation": colors.dim,
        "placeholder": colors.dim,
        "auto-suggestion": colors.dim,
        # По умолчанию у toolbar стиль reverse — цвета текста стали бы фоном.
        "bottom-toolbar": f"noreverse {colors.muted}",
        "toolbar.model": f"bold nodim {colors.fg}",
        "toolbar.ok": colors.ok,
        "toolbar.warn": colors.warn,
        "toolbar.danger": f"bold {colors.danger}",
        "toolbar.mode.auto": f"bold {colors.warn}",
        "toolbar.mode.plan": f"bold {colors.brand}",
        # меню вопросов агента (ask_user) и выбор чата
        "question.bar": colors.warn,
        "question.mark": f"bold {colors.warn}",
        "question.text": f"bold {colors.fg}",
        "question.tag": colors.muted,
        "question.counter": f"bold {colors.brand}",
        "choice": colors.fg,
        "choice.selected": f"bold {colors.brand}",
        "choice.number": colors.muted,
        "choice.number.selected": colors.brand,
        "choice.box": colors.dim,
        "choice.box.checked": f"bold {colors.brand}",
        "choice.checked": f"bold {colors.fg}",
        "choice.description": colors.muted,
        "choice.description.faint": colors.dim,
        "choice.hint": colors.dim,
        "choice.error": colors.warn,
        "choice.custom": f"bold {colors.fg}",
        "completion-menu": f"{menu} {text}",
        "completion-menu.completion": f"{menu} {text}",
        "completion-menu.completion.current": current,
        "completion-menu.meta.completion": f"{menu} {meta}",
        "completion-menu.meta.completion.current": current,
        "completion-menu.multi-column-meta": f"{menu} {meta}",
        "scrollbar.background": scrollbar,
        "scrollbar.button": thumb,
    }
