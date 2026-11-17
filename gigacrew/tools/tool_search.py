"""Отложенные инструменты и ``tool_search``: схемы — модели по мере надобности.

Инструментов MCP-серверов бывают десятки, и схемы всех ушли бы в каждый запрос, съедая
окно модели. Такие инструменты отложены (:attr:`~gigacrew.tools.base.Tool.deferred`): в
системном промпте — только их имена, а схему модель загружает инструментом ``tool_search``
(по ключевым словам или ``select:имя1,имя2``). Загруженный инструмент остаётся в запросах
до конца чата.

:class:`DeferredTools` — какие отложенные инструменты агента уже загружены. Продолженный
чат и сжатие истории загружают их заново по той части истории, которую видит модель
(:meth:`DeferredTools.restore`): вызванные и найденные прежними ``tool_search``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from typing import Any, override

from pydantic import BaseModel, Field

from gigacrew.llm.types import FunctionCall, Message, ToolSpec
from gigacrew.tools.base import Tool, ToolContext, ToolRegistry, ToolResult

__all__ = [
    "TOOL_SEARCH",
    "DeferredTools",
    "ToolSearchParams",
    "ToolSearchTool",
    "find_tools",
    "not_loaded_note",
]

TOOL_SEARCH = "tool_search"
SELECT_PREFIX = "select:"
DEFAULT_RESULTS = 5
MAX_RESULTS = 20
_DESCRIPTION_LIMIT = 300  # символов описания найденного инструмента в ответе
_NAME_HIT = 10  # слово запроса — слово имени инструмента
_NAME_PART = 5  # часть имени
_TEXT_HIT = 2  # встречается в описании
_PREFIX_MIN = 3  # слово запроса короче — только целым словом, длиннее — и началом слова
_WORD = re.compile(r"[^\W_]+")

# Заметки для модели — на английском, как и промпты (пользователь их не видит).
_LOADED = (
    "Loaded {count} tool(s). Their schemas are in your tool list from your next step on — "
    "call them then, not in this step:\n{tools}"
)
_NOTHING = (
    'No deferred tool matches "{query}". The deferred tools are listed in the system '
    'prompt: try other keywords or "select:<name>" with an exact name.'
)
_GUIDANCE = (
    "Deferred tools: the tools listed below exist, but their schemas are not in your tool "
    "list. To use one, first load it with tool_search: query "
    '"select:<name>[,<name>...]" for exact names, or keywords to search names and '
    "descriptions (load several related tools at once). A loaded tool stays callable for "
    "the rest of the chat. Do not call a deferred tool you have not loaded."
)
_NOT_LOADED = (
    "Tool {name} is deferred: its schema was not loaded, so the call was not made. It is "
    "loaded from your next step on — call it again then, with arguments that follow this "
    "schema:\n{schema}"
)


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _matches(term: str, words: set[str]) -> bool:
    """Слово запроса — одно из ``words`` или (не короче :data:`_PREFIX_MIN`) начало одного из них.

    Подстрока не годится: короткие слова («a», «id») нашлись бы почти везде.
    """
    if term in words:
        return True
    return len(term) >= _PREFIX_MIN and any(word.startswith(term) for word in words)


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit - 1].rstrip()
    return f"{head}…"


def _selected(tools: Sequence[Tool[Any]], names: str) -> list[Tool[Any]]:
    """Инструменты по точным именам через запятую (порядок запроса, без повторов)."""
    by_name = {tool.name: tool for tool in tools}
    wanted = dict.fromkeys(name.strip() for name in names.split(","))
    return [by_name[name] for name in wanted if name in by_name]


def _score(tool: Tool[Any], terms: Sequence[str], required: Sequence[str]) -> int:
    """Насколько инструмент подходит запросу (0 — не подходит).

    Слово запроса в имени весит больше, чем в описании; ``required`` (слова с ``+``)
    обязаны быть в имени.
    """
    name = tool.name.lower()
    if any(word not in name for word in required):
        return 0
    name_words = set(_words(name))
    text_words = set(_words(tool.description))
    total = 0
    for term in (*required, *terms):
        if term in name_words:
            total += _NAME_HIT
        elif _matches(term, name_words):
            total += _NAME_PART
        if _matches(term, text_words):
            total += _TEXT_HIT
    return total


def find_tools(tools: Sequence[Tool[Any]], query: str, limit: int) -> list[Tool[Any]]:
    """Инструменты из ``tools`` по запросу модели — не больше ``limit``.

    ``select:a,b`` — точные имена; точное имя одним словом — этот инструмент; иначе
    ключевые слова по имени и описанию (``+слово`` — обязательно в имени), лучшие — первыми.
    """
    query = query.strip()
    if query.lower().startswith(SELECT_PREFIX):
        return _selected(tools, query[len(SELECT_PREFIX) :])
    exact = _selected(tools, query)
    if exact:
        return exact
    required: list[str] = []
    terms: list[str] = []
    for token in query.split():
        bucket = required if token.startswith("+") else terms
        # Однобуквенные слова («a», «и») нашлись бы почти в любом описании.
        bucket.extend(word for word in _words(token) if len(word) > 1)
    scored = [(_score(tool, terms, required), index) for index, tool in enumerate(tools)]
    ranked = sorted((pair for pair in scored if pair[0] > 0), key=lambda pair: -pair[0])
    return [tools[index] for _, index in ranked[:limit]]


def not_loaded_note(tool: Tool[Any]) -> str:
    """Заметка модели: вызван отложенный инструмент, схемы которого она не видела."""
    schema = json.dumps(tool.input_schema(), ensure_ascii=False)
    return _NOT_LOADED.format(name=tool.name, schema=schema)


class DeferredTools:
    """Отложенные инструменты реестра агента и какие из них загружены (см. описание модуля).

    ``enabled`` — откладывать ли (``GIGACREW_MCP_DEFER``). Без ``tool_search`` в реестре
    (суб-агент, которому он не дан) загрузить их нечем. В обоих случаях отложенных нет:
    схемы всех инструментов уходят модели сразу, а ``tool_search`` ей не виден — как и
    когда откладывать нечего.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        loaded: Iterable[str] = (),
        *,
        enabled: bool = True,
    ) -> None:
        """``loaded`` — уже загруженные (суб-агенту — загруженные основным агентом)."""
        self._registry = registry
        self._enabled = enabled
        self._loaded: set[str] = set(loaded)
        # Схемы, отправленные модели в запросе текущего шага (см. step_specs).
        self._offered: frozenset[str] = frozenset()

    @property
    def loaded(self) -> frozenset[str]:
        """Имена загруженных инструментов."""
        return frozenset(self._loaded)

    def available(self) -> list[Tool[Any]]:
        """Все отложенные инструменты реестра (загруженные тоже), в порядке регистрации."""
        if not self._enabled or TOOL_SEARCH not in self._registry:
            return []
        return [tool for tool in self._registry if tool.deferred]

    def pending(self, name: str) -> bool:
        """Инструмент отложен, и его схемы не было в запросе, на который отвечает модель.

        Загруженный в этом же шаге (``tool_search`` или отклонённый вызов раньше в том же
        ответе) — тоже: модель писала аргументы, не видя схемы.
        """
        if name in self._offered:
            return False
        return any(tool.name == name for tool in self.available())

    def load(self, names: Iterable[str]) -> None:
        """Загрузить отложенные инструменты: их схемы уйдут модели со следующим запросом.

        Имена, которые не отложены, пропускаются.
        """
        deferred = {tool.name for tool in self.available()}
        self._loaded.update(name for name in names if name in deferred)

    def specs(self) -> list[ToolSpec]:
        """Спецификации для модели: обычные инструменты и загруженные отложенные."""
        deferred = {tool.name for tool in self.available()}
        hidden = deferred - self._loaded if deferred else {TOOL_SEARCH}
        return [tool.spec() for tool in self._registry if tool.name not in hidden]

    def step_specs(self) -> list[ToolSpec]:
        """:meth:`specs` для запроса нового шага: по ним проверяются вызовы из ответа."""
        specs = self.specs()
        self._offered = frozenset(spec.name for spec in specs)
        return specs

    def guidance(self) -> str:
        """Указание модели об отложенных инструментах ("" — их нет).

        Перечислены все отложенные, и загруженные тоже: промпт от загрузки не меняется.
        """
        names = [tool.name for tool in self.available()]
        if not names:
            return ""
        listed = "\n".join(names)
        return f"{_GUIDANCE}\n{listed}"

    def search(self, query: str, limit: int = DEFAULT_RESULTS) -> list[Tool[Any]]:
        """Найти отложенные инструменты по запросу и загрузить их."""
        found = find_tools(self.available(), query, limit)
        self.load(tool.name for tool in found)
        return found

    def restore(self, messages: Iterable[Message]) -> None:
        """Загружены только инструменты из истории ``messages`` (новый чат, сжатие).

        Вызванные моделью и найденные её вызовами ``tool_search``. ``messages`` — то, что
        видит модель (после границы сжатия): схемы из свёрнутого начала диалога не
        возвращаются, модель загрузит нужные снова. Схемы текущего шага (проверка вызовов)
        не меняются: запрос шага уже ушёл с ними.
        """
        self._loaded.clear()
        for message in messages:
            for call in message.function_calls or ():
                self._restore_call(call)

    def _restore_call(self, call: FunctionCall) -> None:
        if call.name != TOOL_SEARCH:
            self.load((call.name,))
            return
        try:
            params = ToolSearchParams.model_validate(call.arguments or {})
        except ValueError:
            return
        self.search(params.query, params.max_results)


# Без docstring: он ушёл бы в JSON-схему параметров, которую видит модель.
class ToolSearchParams(BaseModel):  # noqa: D101
    query: str = Field(
        description=(
            'Either "select:<name>[,<name>...]" to load deferred tools by exact name, or '
            "keywords to search their names and descriptions (prefix a word with + to "
            "require it in the tool name)"
        ),
    )
    max_results: int = Field(
        default=DEFAULT_RESULTS,
        ge=1,
        le=MAX_RESULTS,
        description="Maximum number of tools to load for a keyword search",
    )


class ToolSearchTool(Tool[ToolSearchParams]):
    """Загрузить схемы отложенных инструментов (см. описание модуля)."""

    name = TOOL_SEARCH
    description = (
        "Load deferred tools (listed by name in the system prompt) so that you can call "
        "them. Until loaded, their schemas are not in your tool list. Found tools become "
        "callable from your next step on and stay available for the rest of the chat. Load "
        "several related tools in one call."
    )
    Params = ToolSearchParams
    read_only = True

    @override
    def describe(self, params: ToolSearchParams) -> str:
        return params.query

    @override
    def run(self, params: ToolSearchParams, ctx: ToolContext) -> ToolResult:
        """Найти и загрузить; нашлось — имена и описания найденных."""
        deferred = ctx.deferred
        found = [] if deferred is None else deferred.search(params.query, params.max_results)
        if not found:
            return ToolResult(_NOTHING.format(query=params.query), summary="ничего не найдено")
        listed = "\n".join(
            f"- {tool.name}: {_clip(tool.description, _DESCRIPTION_LIMIT)}" for tool in found
        )
        names = ", ".join(tool.name for tool in found)
        return ToolResult(
            _LOADED.format(count=len(found), tools=listed),
            summary=f"загружены: {names}",
        )
