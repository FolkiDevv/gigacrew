"""Тестовые двойники: провайдер с заготовленными ходами и записывающий UI."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from pydantic import BaseModel

from gigacrew.agent.events import AgentEvents, ToolCallInfo, TurnStats
from gigacrew.llm.base import LLMProvider, ModelInfo, PromptTooLong
from gigacrew.llm.types import AssistantTurn, FunctionCall, Message, Usage
from gigacrew.permissions import ToolKind
from gigacrew.tools.base import Display, Tool, ToolContext, ToolResult


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
    ):
        self._turns = list(turns)
        self.summaries = list(summaries)
        self.requests: list[dict[str, Any]] = []
        self.summary_requests: list[dict[str, Any]] = []

    @property
    def calls(self) -> int:
        return len(self.requests)

    @property
    def model(self) -> str:
        return "scripted"

    def complete(self, messages, tools=None, **kwargs) -> AssistantTurn:
        if tools is None:
            self.summary_requests.append({"messages": list(messages), **kwargs})
            return _next(self.summaries, "сводка")
        self.requests.append({"messages": list(messages), "tools": tools, **kwargs})
        return _next(self._turns, "конец")

    def stream(self, messages, tools=None, *, on_delta=None, **kwargs) -> AssistantTurn:
        turn = self.complete(messages, tools, **kwargs)
        if on_delta and turn.message.content:
            on_delta(turn.message.content)
        return turn


def _next(queue: list, default: str) -> AssistantTurn:
    if not queue:
        return text_turn(default)
    item = queue.pop(0)
    if isinstance(item, BaseException):
        raise item
    return item


class WindowProvider(ScriptedProvider):
    """Провайдер с окном ``window`` токенов для замера (токен ≈ ``chars_per_token`` символа).

    Отказ — :class:`PromptTooLong` со статусом ``status``; ``hint`` — текст отказа
    называет лимит (как сервер модели). Каждая проба пишется в ``measured`` (модель, токены).
    """

    supports_measure = True

    def __init__(
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
    ):
        super().__init__(turns)
        self.window = window
        self.chars_per_token = chars_per_token
        self.hint = hint
        self.status = status
        self.error = error
        self.overhead = overhead  # служебные токены сообщения
        self.truncate = truncate  # сервер молча обрезает запрос до окна, а не отказывает
        self.measured: list[tuple[str | None, int]] = []
        self.sent_chars: list[int] = []  # длина текста каждой пробы
        self.max_chars = max_chars  # проба длиннее — AssertionError

    def measure_prompt(self, text: str, *, model: str | None = None) -> int:
        if self.error is not None:
            raise self.error
        self.sent_chars.append(len(text))
        if self.max_chars is not None and len(text) > self.max_chars:
            # тест падает сразу, а не на таймауте, пока замер строит гигабайтные пробы
            raise AssertionError(f"проба в {len(text)} символов — замер пошёл вразнос")
        tokens = int(len(text) / self.chars_per_token) + self.overhead
        if self.truncate:
            tokens = min(tokens, self.window - 1)
        self.measured.append((model, tokens))
        if tokens + 1 > self.window:
            detail = (
                f"This model's maximum context length is {self.window} tokens. "
                f"However, you requested {tokens + 1} tokens"
                if self.hint
                else "Request too large"
            )
            raise PromptTooLong(self.status, detail)
        return tokens


class CatalogueProvider(WindowProvider):
    """Провайдер с каталогом моделей, как бэкенд GigaCode: ``connect()`` (токен и каталог)
    считается и может бросать заготовленные ошибки, модель по умолчанию — первая в каталоге.
    """

    def __init__(
        self,
        turns: Iterable[AssistantTurn | BaseException] = (),
        *,
        models: Iterable[tuple[str, int]] = (("cat-a", 64_000), ("cat-b", 32_000)),
        connect_errors: Iterable[BaseException] = (),
    ):
        super().__init__(16_384, turns)
        self.catalogue = [ModelInfo(name, window) for name, window in models]
        self.connect_errors = list(connect_errors)
        self.connects = 0

    @property
    def model(self) -> str:
        return self.catalogue[0].id

    def connect(self) -> None:
        self.connects += 1
        if self.connect_errors:
            raise self.connect_errors.pop(0)

    def list_models(self) -> list[str]:
        return [m.id for m in self.catalogue]

    def known_models(self) -> list[ModelInfo]:
        return list(self.catalogue)


def text_turn(text: str, usage: Usage | None = None) -> AssistantTurn:
    return AssistantTurn(message=Message(role="assistant", content=text), usage=usage or Usage())


def tool_turn(name: str, args: dict, usage: Usage | None = None) -> AssistantTurn:
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


def tools_turn(calls: list[tuple[str, dict]], text: str = "") -> AssistantTurn:
    """Ход с несколькими вызовами сразу (id — c1, c2, …)."""
    return AssistantTurn(
        message=Message(
            role="assistant",
            content=text,
            function_calls=[
                FunctionCall(name=name, arguments=args, id=f"c{i}")
                for i, (name, args) in enumerate(calls, 1)
            ],
            state_id="S",
        ),
        finish_reason="function_call",
    )


class RecordingEvents(AgentEvents):
    """UI-двойник: запоминает события, на подтверждение отвечает ``confirm_answer``."""

    def __init__(self, confirm_answer: bool = True, answers: list | None = None):
        self.confirm_answer = confirm_answer
        self.answers = answers  # ответы на ask_user (None — отказ)
        self.questions: list = []
        self.events: list[tuple[str, Any]] = []
        self.confirms: list[tuple[ToolCallInfo, Display | None, bool]] = []
        self.results: list[tuple[ToolCallInfo, ToolResult, bool]] = []
        self.stats: list[TurnStats] = []
        self.notices: list[tuple[str, str]] = []
        self.compactions: list = []  # CompactResult | None по каждому сжатию
        self.reviews: list = []  # Review | None по каждой проверке рецензента

    def on_stream_start(self) -> None:
        self.events.append(("stream_start", None))

    def on_stream_delta(self, text: str) -> None:
        self.events.append(("delta", text))

    def on_stream_end(self) -> None:
        self.events.append(("stream_end", None))

    def on_assistant_text(self, text: str) -> None:
        self.events.append(("text", text))

    def on_tool_call(self, call: ToolCallInfo) -> None:
        self.events.append(("tool_call", call))

    def confirm(self, call, preview, *, dangerous):
        self.confirms.append((call, preview, dangerous))
        return self.confirm_answer

    def on_review_start(self, call: ToolCallInfo) -> None:
        self.events.append(("review_start", call))

    def on_review_end(self, call, review) -> None:
        self.events.append(("review_end", review))
        self.reviews.append(review)

    def on_tool_start(self, call: ToolCallInfo) -> None:
        self.events.append(("tool_start", call))

    def on_tool_end(self, call: ToolCallInfo) -> None:
        self.events.append(("tool_end", call))

    def on_tool_result(self, call, result, *, previewed):
        self.events.append(("tool_result", call))
        self.results.append((call, result, previewed))

    def ask_user(self, questions):
        self.questions.append(list(questions))
        return self.answers

    def on_notice(self, text, *, level="info"):
        self.notices.append((level, text))

    def on_compact_start(self, *, auto):
        self.events.append(("compact_start", auto))

    def on_compact_end(self, result):
        self.events.append(("compact_end", result))
        self.compactions.append(result)

    def on_turn_end(self, stats: TurnStats) -> None:
        self.stats.append(stats)

    def on_wrap_up(self, reason):
        self.events.append(("wrap_up", reason))

    def on_hook_start(self, event, command):
        self.events.append(("hook_start", (event, command)))

    def on_hook_end(self):
        self.events.append(("hook_end", None))

    def on_subagent_start(self, info):
        self.events.append(("subagent_start", info))

    def on_subagent_activity(self, info, activity, *, wrapping_up=False):
        self.events.append(("subagent_activity", (activity, wrapping_up)))

    def on_subagent_tool_call(self, info, call):
        self.events.append(("subagent_tool_call", call))

    def on_subagent_tool_result(self, info, call, result, *, previewed):
        self.events.append(("subagent_tool_result", (call, result)))

    def on_subagent_end(self, info, stats):
        self.events.append(("subagent_end", stats))


class FakeParams(BaseModel):
    path: str = "x"


class FakeTool(Tool):
    """Настраиваемый WRITE-инструмент для тестов агентного цикла (``kind`` — вид для
    режимов разрешений)."""

    name = "fake_write"
    description = "тестовый инструмент"
    Params = FakeParams

    def __init__(
        self,
        *,
        preview: Callable[[], Display | None] | None = None,
        run: Callable[[], ToolResult] | None = None,
        kind: ToolKind = ToolKind.OTHER,
    ):
        self.kind = kind
        self._preview = preview or (lambda: Display("+x", kind="diff", title="x"))
        self._run = run or (lambda: ToolResult(content="ok", summary="ok"))
        self.runs = 0

    def risk(self, params, ctx: ToolContext):
        from gigacrew.security import RiskLevel

        return RiskLevel.WRITE

    def preview(self, params, ctx):
        return self._preview()

    def run(self, params, ctx):
        self.runs += 1
        return self._run()
