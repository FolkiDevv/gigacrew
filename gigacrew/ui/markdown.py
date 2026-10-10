"""Markdown в стиле gigacrew.

Отличия от ``rich.markdown.Markdown``: заголовки выровнены по левому краю и без
рамок (в rich 13 h1 рисовался в панели, в rich 15 — по центру; в ленте чата это
выглядит чужеродно), код подсвечивается ANSI-темой терминала — как диффы.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import Heading
from rich.markdown import Markdown as RichMarkdown

CODE_THEME = "ansi_dark"


class _LeftHeading(Heading):
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        text = self.text.copy()
        text.justify = "left"
        yield text


class Markdown(RichMarkdown):
    elements = {**RichMarkdown.elements, "heading_open": _LeftHeading}

    def __init__(self, markup: str, **kwargs: Any) -> None:
        kwargs.setdefault("code_theme", CODE_THEME)
        super().__init__(markup, **kwargs)
