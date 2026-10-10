"""Протокол событий агента — единственная связь ядра с пользовательским интерфейсом.

Агент не знает, как именно отображается его работа: он сообщает о событиях
(начало стрима, вызов инструмента, запрос подтверждения, итог хода), а UI
решает, как их показать. Это позволяет заменить терминальный интерфейс (live-
рендеринг Markdown, статус-строка, TUI), не трогая агентный цикл, и тестировать
цикл без rich.

:class:`AgentEvents` — базовый класс с пустыми реализациями: интерфейс
переопределяет только нужные методы. ``confirm`` по умолчанию возвращает
``False`` — интерфейс, «забывший» его реализовать, ничего не подтвердит молча.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from gigacrew.permissions import ToolKind
from gigacrew.tools.base import Display, ToolResult
from gigacrew.tools.questions import Answer, Question, QuestionsUnavailable

NoticeLevel = Literal["info", "warn", "error"]


@dataclass(frozen=True)
class ToolCallInfo:
    """Вызов инструмента в том виде, в каком его показывают пользователю."""

    name: str
    summary: str = ""
    kind: ToolKind = ToolKind.OTHER  # правка файлов, команда или прочее (для формулировок)


class Approval(Enum):
    """Ответ на подтверждение операции."""

    NO = "no"
    YES = "yes"
    # Да, и не спрашивать до конца сессии: для правок файлов — режим авто-правок,
    # для прочего — этот же вызов (те же аргументы).
    ALWAYS = "always"


@dataclass
class TurnStats:
    """Итоги одного хода агента."""

    steps: int = 0
    tool_calls: int = 0
    # Сумма по всем обращениям хода — столько токенов оплачено.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Размер контекста после последнего обращения (промпт + ответ).
    context_tokens: int = 0
    # Почему ход прерван ограничителем (None — модель завершила ход сама).
    stop_reason: str | None = None
    # Длительность хода, секунды.
    duration_s: float = 0.0
    # Успешных изменяющих вызовов за ход (правки, изменяющие команды).
    changes: int = 0

    @property
    def billed_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class CompactResult:
    """Итог сжатия контекста (оценки размера запроса в токенах)."""

    before_tokens: int
    after_tokens: int
    messages: int  # сколько сообщений свёрнуто в краткое содержание
    auto: bool = False  # запущено автоматически (контекст подошёл к порогу)


@dataclass(frozen=True)
class SubagentInfo:
    """Запущенный суб-агент: тип (``explore``) и краткое описание задачи для UI."""

    name: str
    description: str = ""


class AgentEvents:
    """Получатель событий агента. Все методы — no-op; ``confirm`` → False."""

    # --- ответ модели ---
    def on_stream_start(self) -> None:
        """Запрос к модели отправлен; ждём ответ.

        Вызывается в обоих режимах: в потоковом дальше приходят ``on_stream_delta``,
        в непотоковом после ``on_stream_end`` приходит ``on_assistant_text``.
        """

    def on_stream_delta(self, text: str) -> None:
        """Очередной кусок текста модели (потоковый режим)."""

    def on_stream_end(self) -> None:
        """Ответ модели получен или запрос прерван (вызывается всегда)."""

    def on_assistant_text(self, text: str) -> None:
        """Ответ модели целиком (непотоковый режим)."""

    # --- инструменты ---
    def on_tool_call(self, call: ToolCallInfo) -> None:
        """Модель вызвала инструмент."""

    def confirm(
        self, call: ToolCallInfo, preview: Display | None, *, dangerous: bool
    ) -> bool | Approval:
        """Спросить пользователя, выполнять ли изменяющую/опасную операцию.

        ``True``/``False`` — да/нет; :attr:`Approval.ALWAYS` — да и не спрашивать
        такое до конца сессии (для опасных операций не предлагается).
        """
        return False

    def on_tool_start(self, call: ToolCallInfo) -> None:
        """Инструмент запущен (после подтверждения, если оно требовалось)."""

    def on_tool_end(self, call: ToolCallInfo) -> None:
        """Выполнение закончилось — успешно, с ошибкой или прервано.

        Парный к ``on_tool_start``, вызывается всегда и раньше ``on_tool_result``.
        """

    def on_tool_result(self, call: ToolCallInfo, result: ToolResult, *, previewed: bool) -> None:
        """Результат инструмента. ``previewed`` — превью уже показано при подтверждении."""

    def ask_user(self, questions: Sequence[Question]) -> list[Answer] | None:
        """Задать вопросы (инструмент ask_user). None — пользователь отказался отвечать.

        Интерфейс без поддержки вопросов сообщает, что спросить некого, — модель
        получит ошибку и решит сама, а не примет выдуманный ответ.
        """
        raise QuestionsUnavailable()

    # --- сжатие контекста ---
    def on_compact_start(self, *, auto: bool) -> None:
        """Началось сжатие истории (``auto`` — по порогу, иначе по команде)."""

    def on_compact_end(self, result: CompactResult | None) -> None:
        """Сжатие закончилось (вызывается всегда). None — не удалось или прервано."""

    # --- суб-агенты ---
    # Суб-агент работает внутри вызова инструмента ``task`` (между ``on_tool_start`` и
    # ``on_tool_end``). Его собственные события не приходят как события основного
    # агента: текст суб-агента пользователю не печатается, а инструменты и ожидание
    # сообщаются хуками ниже. Подтверждения суб-агента приходят обычным ``confirm``.
    def on_subagent_start(self, info: SubagentInfo) -> None:
        """Суб-агент запущен."""

    def on_subagent_activity(
        self, info: SubagentInfo, activity: str, *, wrapping_up: bool = False
    ) -> None:
        """Чем занят суб-агент: «думаю», имя инструмента, «сжимаю контекст».

        ``wrapping_up`` — суб-агент остановлен и подводит итог (повторная остановка
        оборвёт его без отчёта).
        """

    def on_subagent_tool_call(self, info: SubagentInfo, call: ToolCallInfo) -> None:
        """Суб-агент вызвал инструмент."""

    def on_subagent_tool_result(
        self, info: SubagentInfo, call: ToolCallInfo, result: ToolResult, *, previewed: bool
    ) -> None:
        """Результат инструмента суб-агента."""

    def on_subagent_end(self, info: SubagentInfo, stats: TurnStats | None) -> None:
        """Суб-агент закончил (вызывается всегда). None — прерван исключением."""

    # --- служебное ---
    def on_wrap_up(self, reason: str) -> None:
        """Ход остановлен (ограничитель или пользователь) — модель подводит итог."""

    def on_notice(self, text: str, *, level: NoticeLevel = "info") -> None:
        """Сообщение агента пользователю (лимиты, остановки, предупреждения)."""

    def on_turn_end(self, stats: TurnStats) -> None:
        """Ход завершён."""
