"""Markdown в стиле gigacrew.

Отличия от ``rich.markdown.Markdown``: заголовки выровнены по левому краю и без
рамок (в rich 13 h1 рисовался в панели, в rich 15 — по центру; в ленте чата это
выглядит чужеродно), код подсвечивается ANSI-темой терминала — как диффы.
"""

from __future__ import annotations

from typing import Any, ClassVar, override

from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import Heading, MarkdownElement
from rich.markdown import Markdown as RichMarkdown

CODE_THEME = "ansi_dark"

# Таблица rich: тег разметки -> класс элемента Markdown.
type _Elements = dict[str, type[MarkdownElement]]


class _LeftHeading(Heading):
    @override
    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        text = self.text.copy()
        text.justify = "left"
        yield text


class Markdown(RichMarkdown):
    """Markdown rich с заголовками слева и ANSI-темой подсветки кода."""

    elements: ClassVar[_Elements] = {
        **RichMarkdown.elements,
        "heading_open": _LeftHeading,
    }

    def __init__(self, markup: str, **kwargs: Any) -> None:
        """Создаёт Markdown; тема кода по умолчанию — ``CODE_THEME``."""
        kwargs.setdefault("code_theme", CODE_THEME)
        super().__init__(markup, **kwargs)
