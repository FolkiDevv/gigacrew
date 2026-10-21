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

from dataclasses import dataclass


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
    """Палитра для цветовой системы терминала — как её называет rich
    (``truecolor``, ``256``, ``standard``, ``windows``; None — без цвета)."""
    return MINT_ANSI if color_system in ("standard", "windows") else MINT


ICON_TOOL = "●"
ICON_OK = "✔"
ICON_FAIL = "✘"
ICON_BRAND = "✦"
ICON_PENDING = "◌"  # шаг ещё выполняется
ICON_BAR = "▌"  # полоса слева у всего, что ждёт ответа пользователя
ICON_DANGER = "⚠"
SPINNER = "✦"
# Значки режимов разрешений (ключ — PermissionMode.value).
MODE_ICONS: dict[str, str] = {"auto": "⏵⏵", "plan": "⏸"}


def markdown_styles(p: Palette) -> dict[str, str]:
    """Стили Markdown-ответа модели (имена стилей rich: markdown.*)."""
    return {
        "markdown.h1": f"bold underline {p.brand}",
        "markdown.h2": f"bold {p.brand}",
        "markdown.h3": "bold",
        "markdown.h4": "bold italic",
        "markdown.h5": "italic",
        "markdown.h6": "italic",
        "markdown.code": p.soft,  # inline-код — без фона, читается в любой теме
        "markdown.link": p.soft,
        "markdown.link_url": f"underline {p.soft}",
        "markdown.item.bullet": f"bold {p.brand}",
        "markdown.item.number": f"bold {p.brand}",
        "markdown.block_quote": f"italic {p.muted}",
        "markdown.hr": p.dim,
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
    return "ansi" + color.replace("_", "")


def prompt_styles(p: Palette) -> dict[str, str]:
    """Стили строки ввода, меню вопросов и выбора чата (классы prompt_toolkit)."""
    c = _pt
    ansi = p is MINT_ANSI
    # Меню дополнения — на своём тёмном фоне, поэтому и текст в нём свой, светлый:
    # цвет терминала по умолчанию в светлой теме был бы тёмным на тёмном.
    if ansi:
        menu, text, meta = "bg:ansiblack", "ansiwhite", "ansiwhite"
        scrollbar, thumb = "bg:ansibrightblack", "bg:ansiwhite"
    else:
        menu, text, meta = "bg:#232826", "#E3E7E4", c(p.muted)
        scrollbar, thumb = "bg:#2F3532", f"bg:{c(p.dim)}"
    current = f"bg:{c(p.brand)} {'ansiblack' if ansi else '#111111'}"
    return {
        "prompt": f"bold {c(p.brand)}",
        "continuation": c(p.dim),
        "placeholder": c(p.dim),
        "auto-suggestion": c(p.dim),
        # По умолчанию у toolbar стиль reverse — цвета текста стали бы фоном.
        "bottom-toolbar": f"noreverse {c(p.muted)}",
        "toolbar.model": f"bold nodim {c(p.fg)}",
        "toolbar.ok": c(p.ok),
        "toolbar.warn": c(p.warn),
        "toolbar.danger": f"bold {c(p.danger)}",
        "toolbar.mode.auto": f"bold {c(p.warn)}",
        "toolbar.mode.plan": f"bold {c(p.brand)}",
        # меню вопросов агента (ask_user) и выбор чата
        "question.bar": c(p.warn),
        "question.mark": f"bold {c(p.warn)}",
        "question.text": f"bold {c(p.fg)}",
        "question.tag": c(p.muted),
        "question.counter": f"bold {c(p.brand)}",
        "choice": c(p.fg),
        "choice.selected": f"bold {c(p.brand)}",
        "choice.number": c(p.muted),
        "choice.number.selected": c(p.brand),
        "choice.box": c(p.dim),
        "choice.box.checked": f"bold {c(p.brand)}",
        "choice.checked": f"bold {c(p.fg)}",
        "choice.description": c(p.muted),
        "choice.description.faint": c(p.dim),
        "choice.hint": c(p.dim),
        "choice.error": c(p.warn),
        "choice.custom": f"bold {c(p.fg)}",
        "completion-menu": f"{menu} {text}",
        "completion-menu.completion": f"{menu} {text}",
        "completion-menu.completion.current": current,
        "completion-menu.meta.completion": f"{menu} {meta}",
        "completion-menu.meta.completion.current": current,
        "completion-menu.multi-column-meta": f"{menu} {meta}",
        "scrollbar.background": scrollbar,
        "scrollbar.button": thumb,
    }
