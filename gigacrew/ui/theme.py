"""Визуальный язык gigacrew: палитра и иконки.

Отдельный модуль, чтобы одни и те же цвета использовали вывод rich и стили
prompt_toolkit (строка ввода, автодополнение, статус-строка). Библиотеки здесь не
импортируются: стили — обычные словари, объекты строят потребители.
"""

from __future__ import annotations

BRAND = "#A78BFA"  # фиолетовый — бренд
BRAND_SOFT = "#C4B5FD"  # светло-фиолетовый — звёзды на стартовом экране
ACCENT = "#22D3EE"  # бирюзовый — акценты/пути
OK = "#34D399"  # зелёный — успех
WARN = "#FBBF24"  # жёлтый — предупреждение/подтверждение
DANGER = "#F87171"  # красный — ошибки/опасность
MUTED = "#7C7C8A"  # серый — второстепенное
USER = "#93C5FD"  # голубой — пользователь
FLAME = "#FDE68A"  # светло-жёлтый — ядро пламени в индикаторе ожидания

ICON_TOOL = "●"
ICON_OK = "✔"
ICON_FAIL = "✘"
ICON_BRAND = "✦"
ICON_ARROW = "↳"
ICON_SUBSTEP = "○"  # шаг суб-агента под вызовом task
SPINNER = "✦"
# Значки режимов разрешений (ключ — PermissionMode.value); ручной — без значка.
MODE_ICONS: dict[str, str] = {"edits": "⏵⏵", "plan": "⏸"}

# Стили Markdown-ответа модели (имена стилей rich: markdown.*).
MARKDOWN_STYLES: dict[str, str] = {
    "markdown.h1": f"bold underline {BRAND}",
    "markdown.h2": f"bold {BRAND}",
    "markdown.h3": "bold",
    "markdown.h4": "bold italic",
    "markdown.h5": "italic",
    "markdown.h6": "italic",
    "markdown.code": f"bold {ACCENT}",  # inline-код — без фона, читается в любой теме
    "markdown.link": ACCENT,
    "markdown.link_url": f"underline {ACCENT}",
    "markdown.item.bullet": f"bold {BRAND}",
    "markdown.item.number": f"bold {BRAND}",
    "markdown.block_quote": f"italic {MUTED}",
    "markdown.hr": MUTED,
}

# Стили строки ввода (классы prompt_toolkit).
PROMPT_STYLES: dict[str, str] = {
    "prompt": f"bold {BRAND}",
    "continuation": MUTED,
    "placeholder": MUTED,
    "auto-suggestion": MUTED,
    # По умолчанию у toolbar стиль reverse — цвета текста стали бы фоном.
    "bottom-toolbar": f"noreverse {MUTED}",
    "toolbar.model": f"bold {ACCENT}",
    "toolbar.ok": OK,
    "toolbar.warn": WARN,
    "toolbar.danger": f"bold {DANGER}",
    "toolbar.mode.manual": MUTED,
    "toolbar.mode.edits": f"bold {WARN}",
    "toolbar.mode.plan": f"bold {BRAND}",
    # меню вопросов агента (ask_user)
    "question.counter": f"bold {BRAND}",
    "question.header": f"bold {ACCENT}",
    "question.text": "bold",
    "choice": "",
    "choice.selected": f"bold {ACCENT}",
    "choice.description": MUTED,
    "choice.hint": MUTED,
    "choice.error": WARN,
    "choice.custom": "underline",
    "completion-menu": "bg:#26262e #d4d4d8",
    "completion-menu.completion.current": f"bg:{BRAND} #111111",
    "completion-menu.meta.completion": f"bg:#26262e {MUTED}",
    "completion-menu.meta.completion.current": f"bg:{BRAND} #111111",
}
