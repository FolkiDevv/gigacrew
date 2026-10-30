"""Тестовые двойники: провайдер с заготовленными ходами, записывающий UI, поддельный jj."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Sequence
from contextlib import ExitStack
from itertools import starmap
from pathlib import Path
from typing import Any, override

import pytest
from pydantic import BaseModel

from gigacrew.agent import AgentEvents, ToolCallInfo, TurnStats
from gigacrew.agent.events import (
    Approval,
    CompactResult,
    ConfirmResult,
    NoticeLevel,
    Review,
    SubagentInfo,
)
from gigacrew.llm.base import LLMProvider, ModelInfo, PromptTooLongError
from gigacrew.llm.types import AssistantTurn, FunctionCall, Message, ToolSpec, Usage
from gigacrew.permissions import ToolKind
from gigacrew.security import RiskLevel
from gigacrew.tools.base import Display, Tool, ToolContext, ToolResult
from gigacrew.tools.questions import Answer, Question

# Аргументы вызова инструмента — как их присылает модель.
type _Arguments = dict[str, Any]


class ScriptedProvider(LLMProvider):
    """Провайдер-заглушка: выдаёт заранее заданную последовательность ходов.

    Элемент сценария — AssistantTurn либо исключение (будет брошено). Каждый
    запрос агента записывается в ``requests`` (сообщения + параметры вызова).

    Запросы без инструментов (``tools=None``) — суммаризация при сжатии контекста и
    модель-рецензент команд: у них своя очередь ``summaries`` (по умолчанию ответ
    «сводка» — рецензент его не разберёт и передаст решение пользователю) и журнал
    ``summary_requests``, чтобы такие запросы не съедали ходы сценария.
    """

    def __init__(
        self,
        turns: Iterable[AssistantTurn | BaseException] = (),
        *,
        summaries: Iterable[AssistantTurn | BaseException] = (),
    ) -> None:
        self._turns = list(turns)
        self.summaries = list(summaries)
        self.requests: list[dict[str, Any]] = []
        self.summary_requests: list[dict[str, Any]] = []
        self._streaming = False  # идёт stream(): запрос пишется с ``streamed=True``

    @property
    def calls(self) -> int:
        return len(self.requests)

    @property
    @override
    def model(self) -> str:
        return "scripted"

    @override
    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> AssistantTurn:
        request = {"messages": list(messages), **kwargs, "streamed": self._streaming}
        if tools is None:
            self.summary_requests.append(request)
            return _next(self.summaries, "сводка")
        self.requests.append({**request, "tools": tools})
        return _next(self._turns, "конец")

    @override
    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        *,
        on_delta: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> AssistantTurn:
        # через complete(): подклассы тестов переопределяют только его
        self._streaming = True
        with ExitStack() as cleanup:
            # флаг снимается и тогда, когда complete() бросает исключение
            cleanup.callback(self._stop_streaming)
            turn = self.complete(messages, tools, **kwargs)
        if on_delta and turn.message.content:
            on_delta(turn.message.content)
        return turn

    def _stop_streaming(self) -> None:
        self._streaming = False


def _next(queue: list[AssistantTurn | BaseException], default: str) -> AssistantTurn:
    if not queue:
        return text_turn(default)
    scripted = queue.pop(0)
    if isinstance(scripted, BaseException):
        raise scripted
    return scripted


class WindowProvider(ScriptedProvider):
    """Провайдер с окном ``window`` токенов для замера (токен ≈ ``chars_per_token`` символа).

    Отказ — :class:`PromptTooLongError` со статусом ``status``; ``hint`` — текст отказа
    называет лимит (как сервер модели). Каждая проба пишется в ``measured`` (модель, токены).
    """

    supports_measure = True

    # Двойник настраивается именованными параметрами — по одному на каждый режим сервера.
    def __init__(  # noqa: WPS211
        self,
        window: int,
        turns: Iterable[AssistantTurn | BaseException] = (),
        *,
        chars_per_token: float = 3.3,
        hint: bool = False,
        status: int | None = 422,
        error: BaseException | None = None,
        overhead: int = 5,
        truncate: bool = False,
        max_chars: int | None = None,
    ) -> None:
        super().__init__(turns)
        self._window = window
        self._chars_per_token = chars_per_token
        self._hint = hint
        self._status = status
        self._error = error
        self._overhead = overhead  # служебные токены сообщения
        self._truncate = truncate  # сервер молча обрезает запрос до окна, а не отказывает
        self.measured: list[tuple[str | None, int]] = []
        self.sent_chars: list[int] = []  # длина текста каждой пробы
        self._max_chars = max_chars  # проба длиннее — AssertionError

    @override
    def measure_prompt(self, text: str, *, model: str | None = None) -> int:
        if self._error is not None:
            raise self._error
        length = len(text)
        self.sent_chars.append(length)
        if self._max_chars is not None and length > self._max_chars:
            # тест падает сразу, а не на таймауте, пока замер строит гигабайтные пробы
            raise AssertionError(f"проба в {length} символов — замер пошёл вразнос")
        tokens = int(length / self._chars_per_token) + self._overhead
        if self._truncate:
            tokens = min(tokens, self._window - 1)
        self.measured.append((model, tokens))
        requested = tokens + 1
        if requested > self._window:
            detail = (
                f"This model's maximum context length is {self._window} tokens. "
                f"However, you requested {requested} tokens"
                if self._hint
                else "Request too large"
            )
            raise PromptTooLongError(self._status, detail)
        return tokens


class CatalogueProvider(WindowProvider):
    """Провайдер с каталогом моделей, как бэкенд GigaCode.

    ``connect()`` (токен и каталог) считается и может бросать заготовленные ошибки,
    модель по умолчанию — первая в каталоге.
    """

    def __init__(
        self,
        turns: Iterable[AssistantTurn | BaseException] = (),
        *,
        models: Iterable[tuple[str, int]] = (("cat-a", 64_000), ("cat-b", 32_000)),
        connect_errors: Iterable[BaseException] = (),
    ) -> None:
        super().__init__(16_384, turns)
        self._catalogue = list(starmap(ModelInfo, models))
        self._connect_errors = list(connect_errors)
        self.connects = 0

    @property
    @override
    def model(self) -> str:
        return self._catalogue[0].id

    @override
    def connect(self) -> None:
        self.connects += 1
        if self._connect_errors:
            raise self._connect_errors.pop(0)

    @override
    def list_models(self) -> list[str]:
        return [model_info.id for model_info in self._catalogue]

    @override
    def known_models(self) -> list[ModelInfo]:
        return list(self._catalogue)


def text_turn(text: str, usage: Usage | None = None) -> AssistantTurn:
    return AssistantTurn(message=Message(role="assistant", content=text), usage=usage or Usage())


def tool_turn(name: str, args: _Arguments, usage: Usage | None = None) -> AssistantTurn:
    return AssistantTurn(
        message=Message(
            role="assistant",
            content="",
            function_calls=[FunctionCall(name=name, arguments=args)],
        ),
        finish_reason="function_call",
        usage=usage or Usage(),
    )


def cut_turn(text: str = "", reason: str = "length", usage: Usage | None = None) -> AssistantTurn:
    """Ответ, оборванный до конца (лимит длины, поток без окончания): вызов в нём потерян."""
    return AssistantTurn(
        message=Message(role="assistant", content=text),
        finish_reason=reason,
        usage=usage or Usage(),
    )


def tools_turn(calls: Sequence[tuple[str, _Arguments]], text: str = "") -> AssistantTurn:
    """Ход с несколькими вызовами сразу (id — c1, c2, …)."""
    return AssistantTurn(
        message=Message(
            role="assistant",
            content=text,
            function_calls=[
                FunctionCall(name=name, arguments=args, id=f"c{index}")
                for index, (name, args) in enumerate(calls, 1)
            ],
            state_id="S",
        ),
        finish_reason="function_call",
    )


# Двойник реализует весь интерфейс AgentEvents, а журналы событий — его API для тестов.
class RecordingEvents(AgentEvents):  # noqa: WPS214, WPS230
    """UI-двойник: запоминает события, на подтверждение отвечает ``confirm_answer``."""

    def __init__(
        self,
        *,
        confirm_answer: bool | Approval | ConfirmResult = True,
        answers: list[Answer] | None = None,
    ) -> None:
        self._confirm_answer = confirm_answer
        self._user_answers = answers  # ответы на ask_user (None — отказ)
        self.questions: list[list[Question]] = []
        self.events: list[tuple[str, Any]] = []
        self.confirms: list[tuple[ToolCallInfo, Display | None, bool]] = []
        self.tool_results: list[tuple[ToolCallInfo, ToolResult, bool]] = []
        self.stats: list[TurnStats] = []
        self.notices: list[tuple[str, str]] = []
        self.compactions: list[CompactResult | None] = []  # по каждому сжатию
        self.reviews: list[Review | None] = []  # по каждой проверке рецензента

    @override
    def on_stream_start(self) -> None:
        self.events.append(("stream_start", None))

    @override
    def on_stream_delta(self, text: str) -> None:
        self.events.append(("delta", text))

    @override
    def on_stream_end(self) -> None:
        self.events.append(("stream_end", None))

    @override
    def on_assistant_text(self, text: str) -> None:
        self.events.append(("text", text))

    @override
    def on_tool_call(self, call: ToolCallInfo) -> None:
        self.events.append(("tool_call", call))

    @override
    def confirm(
        self,
        call: ToolCallInfo,
        preview: Display | None,
        *,
        dangerous: bool,
    ) -> bool | Approval | ConfirmResult:
        self.confirms.append((call, preview, dangerous))
        return self._confirm_answer

    @override
    def on_review_start(self, call: ToolCallInfo) -> None:
        self.events.append(("review_start", call))

    @override
    def on_review_end(self, call: ToolCallInfo, review: Review | None) -> None:
        self.events.append(("review_end", review))
        self.reviews.append(review)

    @override
    def on_tool_start(self, call: ToolCallInfo) -> None:
        self.events.append(("tool_start", call))

    @override
    def on_tool_end(self, call: ToolCallInfo) -> None:
        self.events.append(("tool_end", call))

    @override
    def on_tool_result(
        self,
        call: ToolCallInfo,
        tool_result: ToolResult,
        *,
        previewed: bool,
    ) -> None:
        self.events.append(("tool_result", call))
        self.tool_results.append((call, tool_result, previewed))

    @override
    def ask_user(self, questions: Sequence[Question]) -> list[Answer] | None:
        self.questions.append(list(questions))
        return self._user_answers

    @override
    def on_notice(self, text: str, *, level: NoticeLevel = "info") -> None:
        self.notices.append((level, text))

    @override
    def on_compact_start(self, *, auto: bool) -> None:
        self.events.append(("compact_start", auto))

    @override
    def on_compact_end(self, compacted: CompactResult | None) -> None:
        self.events.append(("compact_end", compacted))
        self.compactions.append(compacted)

    @override
    def on_turn_end(self, stats: TurnStats) -> None:
        self.stats.append(stats)

    @override
    def on_wrap_up(self, reason: str) -> None:
        self.events.append(("wrap_up", reason))

    @override
    def on_hook_start(self, event: str, command: str) -> None:
        self.events.append(("hook_start", (event, command)))

    @override
    def on_hook_end(self) -> None:
        self.events.append(("hook_end", None))

    @override
    def on_subagent_start(self, subagent: SubagentInfo) -> None:
        self.events.append(("subagent_start", subagent))

    @override
    def on_subagent_activity(
        self,
        subagent: SubagentInfo,
        activity: str,
        *,
        wrapping_up: bool = False,
    ) -> None:
        self.events.append(("subagent_activity", (activity, wrapping_up)))

    @override
    def on_subagent_tool_call(self, subagent: SubagentInfo, call: ToolCallInfo) -> None:
        self.events.append(("subagent_tool_call", call))

    @override
    def on_subagent_tool_result(
        self,
        subagent: SubagentInfo,
        call: ToolCallInfo,
        tool_result: ToolResult,
        *,
        previewed: bool,
    ) -> None:
        self.events.append(("subagent_tool_result", (call, tool_result)))

    @override
    def on_subagent_end(self, subagent: SubagentInfo, stats: TurnStats | None) -> None:
        self.events.append(("subagent_end", stats))


class FakeParams(BaseModel):
    path: str = "x"


class FakeTool(Tool[FakeParams]):
    """Настраиваемый WRITE-инструмент для тестов агентного цикла.

    ``kind`` — вид для режимов разрешений.
    """

    name = "fake_write"
    description = "тестовый инструмент"
    # Имя схемы параметров задано базовым классом Tool.
    Params = FakeParams  # noqa: WPS115

    def __init__(
        self,
        *,
        preview: Callable[[], Display | None] | None = None,
        run: Callable[[], ToolResult] | None = None,
        kind: ToolKind = ToolKind.OTHER,
    ) -> None:
        self.kind = kind
        self._preview = preview or (lambda: Display("+x", kind="diff", title="x"))
        self._run = run or (lambda: ToolResult(content="ok", summary="ok"))
        self.runs = 0

    @override
    def risk(self, params: FakeParams, ctx: ToolContext) -> RiskLevel:
        return RiskLevel.WRITE

    @override
    def preview(self, params: FakeParams, ctx: ToolContext) -> Display | None:
        return self._preview()

    @override
    def run(self, params: FakeParams, ctx: ToolContext) -> ToolResult:
        self.runs += 1
        return self._run()


class FakeJJ:
    """Поддельный jj первым в PATH: на любой вызов печатает заданный ответ.

    Классификатор спрашивает у jj настройки репозитория (``jj config list``): настоящего
    jj в CI нет, а у разработчика он отвечал бы по своим настройкам.
    """

    def __init__(self, bin_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._bin_dir = bin_dir
        path = os.environ.get("PATH", "")
        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{path}")
        self.answer()

    def answer(self, output: str = "", code: int = 0) -> None:
        """Что напечатает jj и с каким кодом завершится."""
        (self._bin_dir / "jj.out").write_text(output, encoding="utf-8")
        script = self._bin_dir / "jj"
        body = f'#!/bin/sh\ncat "$(dirname "$0")/jj.out"\nexit {code}\n'
        script.write_text(body, encoding="utf-8")
        script.chmod(0o755)

    def config(self, *records: tuple[str, str, str]) -> None:
        """Ответ на ``jj config list``: записи (источник, имя, значение в записи TOML)."""
        self.answer("".join(f"{source}\0{name}\0{setting}\0" for source, name, setting in records))
