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
from gigacrew.tools.questions import Answer, Question, QuestionsUnavailableError

NoticeLevel = Literal["info", "warn", "error"]


@dataclass(frozen=True)
class ToolCallInfo:
    """Вызов инструмента в том виде, в каком его показывают пользователю.

    Для подтверждения агент дополняет: ``reason`` — почему спрашивает (вердикт
    классификатора команд), ``remember`` — что запомнит ответ «всегда» (пусто — формулировка
    интерфейса по умолчанию), ``history`` — как пользователь отвечал на такое раньше,
    ``purpose`` — зачем вызов, со слов модели (показывается как её слова, не как факт),
    ``review`` — что сказала модель-рецензент (почему передала решение пользователю).
    """

    name: str
    summary: str = ""
    kind: ToolKind = ToolKind.OTHER  # правка файлов, команда или прочее (для формулировок)
    reason: str = ""
    remember: str = ""
    history: str = ""
    purpose: str = ""  # зачем вызов — со слов модели (параметр description)
    review: str = ""  # мнение модели-рецензента (режим «авто»)
    read_only: bool = False  # инструмент только смотрит проект (Tool.read_only)


class Approval(Enum):
    """Ответ на подтверждение операции."""

    # Член перечисления рядом с YES и ALWAYS, а не булево значение: не спутать.
    NO = "no"  # noqa: WPS110
    YES = "yes"
    # Да, и не спрашивать: для команд — сохранённое для проекта разрешение, для прочего —
    # этот же вызов до конца сессии.
    ALWAYS = "always"


@dataclass(frozen=True)
class ConfirmResult:
    """Ответ на подтверждение с подробностями.

    ``feedback`` — что пользователь написал вместо «да/нет» (это отказ; текст получит
    модель). ``answered`` — False, если ответа не было (Ctrl+C, конец ввода, нет
    терминала): такой отказ не считается выбором пользователя.
    """

    approval: Approval
    feedback: str = ""
    answered: bool = True


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
    duration_s: float = float(0)
    # Успешных изменяющих вызовов за ход (правки, изменяющие команды).
    changes: int = 0
    # Повторов запроса после ответа, оборванного до конца (finish_reason length/incomplete).
    cut_retries: int = 0
    # Самопроверок: ход в режиме плана кончился текстом без плана и вопроса — harness
    # переспросил модель, не собиралась ли она продолжить.
    self_checks: int = 0

    @property
    def billed_tokens(self) -> int:
        """Оплаченные токены хода: промпт и ответ по всем обращениям."""
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class CompactResult:
    """Итог сжатия контекста (оценки размера запроса в токенах)."""

    before_tokens: int
    after_tokens: int
    messages: int  # сколько сообщений свёрнуто в краткое содержание
    auto: bool = False  # запущено автоматически (контекст подошёл к порогу)


@dataclass(frozen=True)
class Review:
    """Ответ модели-рецензента на изменяющую команду (режим «авто»).

    ``allowed`` — команду можно выполнить без вопроса; иначе решает пользователь.
    ``failed`` — рецензент не дал ответа (ошибка API, неразборчивый ответ): ``reason`` —
    описание сбоя.
    """

    model: str
    allowed: bool
    reason: str = ""
    failed: bool = False

    @property
    def approval(self) -> str:
        """Строка о разрешении для пользователя: какая модель разрешила и почему."""
        text = f"рецензент ({self.model}) разрешил"
        return f"{text}: {self.reason}" if self.reason else text


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
        self,
        call: ToolCallInfo,
        preview: Display | None,
        *,
        dangerous: bool,
    ) -> bool | Approval | ConfirmResult:
        """Спросить пользователя, выполнять ли изменяющую/опасную операцию.

        ``True``/``False`` — да/нет; :attr:`Approval.ALWAYS` — да и не спрашивать
        такое снова (что именно запомнится — ``call.remember``; для опасных операций не
        предлагается, а если пришёл — значит «да» один раз); :class:`ConfirmResult` — то же
        с текстом пользователя. По умолчанию — «нет»: интерфейс без вопросов не разрешает
        ничего.
        """
        return False

    def on_review_start(self, call: ToolCallInfo) -> None:
        """Модель-рецензент проверяет команду (режим «авто»)."""

    def on_review_end(self, call: ToolCallInfo, review: Review | None) -> None:
        """Проверка закончилась (вызывается всегда).

        None — прервана; при отказе или сбое дальше придёт ``confirm`` с ``call.review``.
        """

    def on_tool_start(self, call: ToolCallInfo) -> None:
        """Инструмент запущен (после подтверждения, если оно требовалось)."""

    def on_tool_end(self, call: ToolCallInfo) -> None:
        """Выполнение закончилось — успешно, с ошибкой или прервано.

        Парный к ``on_tool_start``, вызывается всегда и раньше ``on_tool_result``.
        """

    def on_tool_result(
        self,
        call: ToolCallInfo,
        tool_result: ToolResult,
        *,
        previewed: bool,
    ) -> None:
        """Результат инструмента. ``previewed`` — превью уже показано при подтверждении."""

    def ask_user(self, questions: Sequence[Question]) -> list[Answer] | None:
        """Задать вопросы (инструмент ask_user). None — пользователь отказался отвечать.

        Интерфейс без поддержки вопросов сообщает, что спросить некого, — модель
        получит ошибку и решит сама, а не примет выдуманный ответ.
        """
        raise QuestionsUnavailableError

    # --- сжатие контекста ---
    def on_compact_start(self, *, auto: bool) -> None:
        """Началось сжатие истории (``auto`` — по порогу, иначе по команде)."""

    def on_compact_end(self, compacted: CompactResult | None) -> None:
        """Сжатие закончилось (вызывается всегда). None — не удалось или прервано."""

    # --- суб-агенты ---
    # Суб-агент работает внутри вызова инструмента ``task`` (между ``on_tool_start`` и
    # ``on_tool_end``). Его собственные события не приходят как события основного
    # агента: текст суб-агента пользователю не печатается, а инструменты и ожидание
    # сообщаются хуками ниже. Подтверждения суб-агента приходят обычным ``confirm``.
    def on_subagent_start(self, subagent: SubagentInfo) -> None:
        """Суб-агент запущен."""

    def on_subagent_activity(
        self,
        subagent: SubagentInfo,
        activity: str,
        *,
        wrapping_up: bool = False,
    ) -> None:
        """Чем занят суб-агент: «думаю», имя инструмента, «сжимаю контекст».

        ``wrapping_up`` — суб-агент остановлен и подводит итог (повторная остановка
        оборвёт его без отчёта).
        """

    def on_subagent_tool_call(self, subagent: SubagentInfo, call: ToolCallInfo) -> None:
        """Суб-агент вызвал инструмент."""

    def on_subagent_tool_result(
        self,
        subagent: SubagentInfo,
        call: ToolCallInfo,
        tool_result: ToolResult,
        *,
        previewed: bool,
    ) -> None:
        """Результат инструмента суб-агента."""

    def on_subagent_end(self, subagent: SubagentInfo, stats: TurnStats | None) -> None:
        """Суб-агент закончил (вызывается всегда). None — прерван исключением."""

    # --- хуки пользователя ---
    def on_hook_start(self, event: str, command: str) -> None:
        """Запущен долгий хук (Stop, SessionStart…) — пользователь видит, чего ждёт ход.

        Хуки PreToolUse/PostToolUse/Notification идут на каждом вызове и сюда не
        сообщаются; их результаты и сбои — через :meth:`on_notice`.
        """

    def on_hook_end(self) -> None:
        """Хук из :meth:`on_hook_start` завершился (вызывается всегда)."""

    # --- служебное ---
    def on_wrap_up(self, reason: str) -> None:
        """Ход остановлен (ограничитель или пользователь) — модель подводит итог."""

    def on_notice(self, text: str, *, level: NoticeLevel = "info") -> None:
        """Сообщение агента пользователю (лимиты, остановки, предупреждения)."""

    def on_turn_end(self, stats: TurnStats) -> None:
        """Ход завершён."""
