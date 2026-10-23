"""Потоковый Markdown: поток по кускам печатает ровно то же, что рендер целиком."""

from __future__ import annotations

import io
import random
from collections.abc import Callable
from types import MappingProxyType

import pytest
from rich.console import Console as RichConsole

from gigacrew.ui.markdown_stream import (
    Lines,
    MarkdownStream,
    block_start_lines,
    complete_prefix_length,
)

DOCS = MappingProxyType({
    "headings_and_lists": (
        "# Заголовок\n\nАбзац с **жирным** и `кодом`.\n\n## Подзаголовок\n"
        "- раз\n- два\n  - вложенный\n- три\n\nИтог.\n"
    ),
    "hr_and_quote": "Текст\n\n---\n\n> цитата\n> продолжение\n\nпосле цитаты\n",
    "table": (
        "Сравнение:\n\n| Колонка | Значение |\n|---|---|\n| a | 1 |\n"
        "| длинная строка | 22222 |\n\nПосле таблицы.\n"
    ),
    "fence_with_hash": (
        "Пример:\n\n```python\n# не заголовок\ndef f():\n\n    return 1\n```\n\nКонец.\n"
    ),
    "setext_and_ordered": (
        "Заголовок\n=========\n\n1. первый\n2. второй\n\n3. третий (loose)\n\nТекст.\n"
    ),
    "unclosed_fence": "Код:\n\n```\nprint(1)\n",
    "no_trailing_newline": "Абзац один.\n\nАбзац два без перевода строки",
    "loose_list_continuation": "- пункт\n\n  продолжение пункта\n\n- второй\n\nконец",
    "code_blocks_table_nested_list": (
        "Шаги:\n\n```py\nx = 1\n```\n\n```bash\nls\n```\n\n1. a\n2. b\n\n"
        "| a | b |\n|---|---|\n| 1 | 2 |\n\n- x\n  - y\n    - z\n\nКонец.\n"
    ),
    "heading_then_list": "## H\n- a\n- b\n### H3\ntext\n",
    "quote_with_list_and_hr": "> q1\n>\n> - item\n\npara\n\n***\n\nend\n",
    "table_after_table": "| a |\n|---|\n| 1 |\n\n| b |\n|---|\n| 2 |\n\n---\n# H\n",
})


def _console(width: int = 60) -> RichConsole:
    return RichConsole(file=io.StringIO(), width=width, color_system="truecolor")


def _streamed(doc: str, chunks: list[str]) -> Lines:
    stream = MarkdownStream(_console())
    lines: Lines = []
    for chunk in chunks:
        lines += stream.feed(chunk)
        assert stream.text  # хвост можно отрисовать на любом шаге
        stream.tail_lines()
    return lines + stream.finish()


def _random_chunks(doc: str, seed: int) -> list[str]:
    # Случайное деление на куски — данные теста, а не криптография; seed делает его повторимым.
    rnd = random.Random(seed)  # noqa: S311
    chunks: list[str] = []
    start = 0
    while start < len(doc):
        size = rnd.randint(1, 12)
        chunks.append(doc[start : start + size])
        start += size
    return chunks


def _recording[ReturnT](
    sizes: list[int],
    func: Callable[[str], ReturnT],
) -> Callable[[str], ReturnT]:
    """Обёртка над ``func``, записывающая в ``sizes`` длину каждого входа."""

    def recorded(source: str) -> ReturnT:
        sizes.append(len(source))
        return func(source)

    return recorded


@pytest.mark.parametrize("name", sorted(DOCS))
def test_streaming_equals_full_render(name: str) -> None:
    doc = DOCS[name]
    full = MarkdownStream(_console()).render(doc)
    assert _streamed(doc, list(doc)) == full  # посимвольно
    for seed in range(3):
        assert _streamed(doc, _random_chunks(doc, seed)) == full


def test_partial_line_is_not_committed() -> None:
    # "#" на новой строке — заголовок, "#тег" — продолжение абзаца
    assert complete_prefix_length("абзац\n#") == 0
    assert complete_prefix_length("абзац\n#тег\n") == 0
    assert complete_prefix_length("абзац\n# Заголовок\n") == len("абзац\n")


def test_unclosed_fence_stays_in_tail() -> None:
    stream = MarkdownStream(_console())
    committed = stream.feed("Текст\n\n```\nкод\n\nещё код\n")
    plain = ["".join(segment.text for segment in line) for line in committed]
    assert any("Текст" in line for line in plain)
    assert not any("код" in line for line in plain)
    tail = ["".join(segment.text for segment in line) for line in stream.tail_lines()]
    assert any("ещё код" in line for line in tail)


def test_tail_is_cached_and_separated() -> None:
    stream = MarkdownStream(_console())
    assert stream.feed("Первый\n\nВторой\n")  # первый абзац уже окончателен
    tail = stream.tail_lines()
    assert tail is stream.tail_lines()  # без изменений — без перерендера
    assert tail[0] == []
    assert "Второй" in "".join(segment.text for segment in tail[1])


def test_empty_answer() -> None:
    stream = MarkdownStream(_console())
    stream.feed("  \n")
    assert stream.finish() == []


def test_each_block_is_rendered_once_with_one_block_of_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Новый блок не перерисовывает весь напечатанный ответ (раньше — O(n²))."""
    doc = "".join(f"Абзац {number}.\n\n" for number in range(40))
    stream = MarkdownStream(_console())
    sizes: list[int] = []
    monkeypatch.setattr(stream, "render", _recording(sizes, stream.render))
    for ch in doc:
        stream.feed(ch)
    assert sizes
    assert max(sizes) < 40


_FENCE_DOCS = MappingProxyType({
    "longer_fence_around_shorter": "Код:\n\n````md\n```\ninner\n```\n````\n\nпосле\n",
    "tilde_fence": "~~~\ncode ```\n~~~~\n\nпосле\n",
    "indented_fence": "  ```\n  x\n   ```\nпосле\n",
    "fence_closed_with_trailing_spaces": "```\na\n```   \n# H\n",
    "not_a_closing_fence": "```\n``` x\n````\nпосле\n",
})


@pytest.mark.parametrize("name", sorted(_FENCE_DOCS))
def test_streaming_code_fences_equals_full_render(name: str) -> None:
    doc = _FENCE_DOCS[name]
    full = MarkdownStream(_console()).render(doc)
    assert _streamed(doc, list(doc)) == full
    for seed in range(3):
        assert _streamed(doc, _random_chunks(doc, seed)) == full


def _parsed_chars(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    sizes: list[int] = []
    monkeypatch.setattr(
        "gigacrew.ui.markdown_stream.block_start_lines",
        _recording(sizes, block_start_lines),
    )
    return sizes


def test_long_answer_is_parsed_in_linear_time(monkeypatch: pytest.MonkeyPatch) -> None:
    # Каждый перевод строки разбирал весь ответ заново — O(n²) на длинных ответах.
    sizes = _parsed_chars(monkeypatch)
    doc = "".join(f"Абзац номер {number}: немного текста.\n\n" for number in range(300))
    stream = MarkdownStream(_console())
    for line in doc.splitlines(keepends=True):
        stream.feed(line)
    stream.finish()
    assert sum(sizes) < 5 * len(doc)


def test_long_open_code_block_is_not_reparsed_and_tail_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sizes = _parsed_chars(monkeypatch)
    body = "".join(f"print({number})\n" for number in range(1000))
    stream = MarkdownStream(_console())
    stream.feed("Код:\n\n```python\n")
    for source_line in body.splitlines(keepends=True):
        stream.feed(source_line)
    assert sum(sizes) < 5 * len(body)  # внутри блока кода новых блоков быть не может
    rendered: list[int] = []
    monkeypatch.setattr(stream, "render", _recording(rendered, stream.render))
    tail = ["".join(segment.text for segment in line).strip() for line in stream.tail_lines()]
    tail = [line for line in tail if line]
    assert tail[-2:] == ["print(998)", "print(999)"]
    assert max(rendered) < len(body) // 4  # во временной области — только конец блока
    lines = stream.feed("```\n\nКонец.\n") + stream.finish()
    plain = ["".join(segment.text for segment in line) for line in lines]
    assert any("print(0)" in line for line in plain)  # окончательно — весь блок
