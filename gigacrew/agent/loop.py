"""Агентный цикл: план → действие → наблюдение → коррекция.

Один «ход» (run_turn) принимает запрос пользователя и крутит цикл:
обращение к модели → если модель просит инструмент, выполняем его (с
подтверждением для изменяющих/опасных операций) и возвращаем результат
обратно в модель → повторяем, пока модель не выдаст финальный текстовый
ответ либо не сработает ограничитель.

Ядро не зависит от UI: всё, что видит пользователь, передаётся через
:class:`~gigacrew.agent.events.AgentEvents`.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, BinaryIO, Final

from pydantic import BaseModel

try:
    import fcntl
except ImportError:  # Windows: файл плана не блокируется
    _HAS_FLOCK = False
else:
    _HAS_FLOCK = True

from gigacrew.agent.chat_store import ChatStore
from gigacrew.agent.compaction import (
    CompactionError,
    CompactionPlan,
    plan_compaction,
    summarize,
)
from gigacrew.agent.context_window import (
    CHARS_PER_TOKEN,
    DEFAULT_CONTEXT_WINDOW,
    MIN_HISTORY_TOKENS,
    budget_for_window,
    estimate_specs_tokens,
    estimate_text_tokens,
    estimate_tokens,
    fit_history,
)
from gigacrew.agent.conversation import HOOK_CONTEXT_HEADER, Conversation, Summary
from gigacrew.agent.decisions import DecisionMemory, history_key
from gigacrew.agent.events import (
    AgentEvents,
    Approval,
    CompactResult,
    ConfirmResult,
    Review,
    SubagentInfo,
    ToolCallInfo,
    TurnStats,
)
from gigacrew.agent.guard import (
    MODEL_STOP_REASONS,
    CallCheck,
    LoopGuard,
    StopReason,
    ToolOutcome,
    call_key,
)
from gigacrew.agent.hooks import HookResult, HookRunner, PromptBlockedError
from gigacrew.agent.prompts import (
    SYSTEM_PROMPT,
    build_project_context,
    compose_system_prompt,
    mode_prompt,
    nested_instructions_prompt,
    session_context_prompt,
    subagent_system_prompt,
    summary_prompt,
)
from gigacrew.agent.reviewer import (
    MAX_REVIEW_COMMAND,
    Reviewer,
    recent_requests,
    review_command,
)

# Запуск суб-агента и его отчёты для основного агента — всё из одного модуля.
from gigacrew.agent.subagents import (  # noqa: WPS235
    MAX_SUBAGENTS_PER_TURN,
    READ_ONLY_BLOCKED_NOTE,
    READ_ONLY_SHELL_NOTE,
    SUBAGENT_PLAN_BLOCKED_NOTE,
    SUBAGENT_STOP_NOTE,
    SUBAGENT_USER_STOP_NOTE,
    SubagentEvents,
    aborted_report,
    interrupted_note,
    model_error_report,
    subagent_registry,
    subagent_report,
)
from gigacrew.config import Config
from gigacrew.llm.base import LLMError, LLMProvider
from gigacrew.llm.model_windows import ModelWindows
from gigacrew.llm.types import AssistantTurn, FunctionCall, Message, ToolChoice, ToolSpec, Usage
from gigacrew.permissions import (
    Decision,
    PermissionMode,
    ToolKind,
    apply_hook,
    decide,
    next_mode,
)
from gigacrew.project.instructions import NestedInstructions
from gigacrew.project.settings import HookEvent
from gigacrew.project.workspace import Workspace
from gigacrew.safety import CommandLevel, Verdict
from gigacrew.security import RiskLevel, resolve_in_root
from gigacrew.tools.base import PlanFile, Tool, ToolContext, ToolError, ToolRegistry, ToolResult
from gigacrew.tools.fs import write_atomic
from gigacrew.tools.plan import PlanReview, review_plan
from gigacrew.tools.process import quote_reminder_tags
from gigacrew.tools.questions import Answer, Question
from gigacrew.tools.task import SUBAGENTS, SubagentSpec

# Заметки для модели — на английском, как и промпты (пользователь их не видит).
REJECTED_NOTE = (
    "The user REJECTED this operation. Do not repeat it; suggest an alternative or "
    "clarify the plan."
)
FEEDBACK_NOTE = "\nThe user said instead: {feedback}"
REJECTED_AGAIN_NOTE = (
    "The user already REJECTED this operation in this turn, so it was not offered again. "
    "Do not repeat it; suggest an alternative or clarify the plan."
)
LOOP_STOP_NOTE = (
    "Not executed: the same call repeated without any changes. The turn was stopped as a loop."
)
SKIPPED_NOTE = "Not executed: the turn was stopped before this call."
STOP_SUMMARY_NOTE = (
    "# Turn stopped\nReason: {reason}.\nDo not call tools anymore. Briefly tell the user "
    "what has been done, what did not work and why, what remains and what you suggest next."
)
INTERRUPTED_ANSWER_NOTE = "\n\n(ответ прерван пользователем)"
BROKEN_ANSWER_NOTE = "\n\n(ответ оборван из-за ошибки)"
PLAN_BLOCKED_NOTE = (
    "Not executed: plan mode is on, changes are forbidden. Do not try to make them: "
    "describe the needed changes in the plan. The only file you may write is the plan "
    "file {path} (write_file, fixes — edit_file); then hand it to the user with "
    "exit_plan_mode."
)
# Каждому запросу основного агента в режиме плана: правила режима в системном промпте
# модели теряют к концу длинного хода и пишут план текстом ответа.
PLAN_REMINDER_NOTE = (
    "Plan mode is on: research and plan, change nothing in the project. A plan goes only "
    "to the plan file {path} (write_file, fixes — edit_file) and is handed to the user "
    "with exit_plan_mode, not written in the reply text. If the user only asked a "
    "question, just answer it."
)
AUTO_REVIEW_NOTE = (
    "You ended your reply without calling exit_plan_mode, so gigacrew showed the plan file "
    "to the user itself. The user's decision:\n\n{result}"
)
SELF_CHECK_NOTE = (
    "# Turn ended in plan mode\nYou ended your turn in plan mode without handing over a "
    "plan (exit_plan_mode) or asking the user (ask_user). The user approves a plan only "
    "from the plan file, not from the reply text. If the user asked for a change — even "
    "if your reply already describes the plan — or you meant to continue, write the plan "
    "to {path} with write_file now and call exit_plan_mode. If the user only asked a "
    "question and your answer is complete, reply with one word: OK"
)
OUTPUT_CUT_NOTE = (
    "# Reply cut off\nYour previous reply was cut off before it finished (finish_reason: "
    "{reason}), and any tool call in it was lost. Do it again in smaller pieces: write a "
    "long file as an outline with write_file and add the sections with edit_file; keep a "
    "text answer shorter."
)
HOOK_BLOCKED_NOTE = (
    "Not executed: the user's PreToolUse hook blocked this call. Reason:\n{reason}\n"
    "Do not repeat the same call; change the approach or ask the user."
)
HOOK_FEEDBACK_NOTE = "The user's {event} hook reported:\n{text}"
STOP_HOOK_NOTE = (
    "# Turn not finished\nThe user's {event} hook did not let you finish yet. Reason:\n"
    "{reason}\nAddress it, then give your final answer."
)
PLAN_MOVED_NOTE = (
    "Not executed: another gigacrew session continues this chat and writes its plan file "
    "{old}. Your plan is now in {new} (as you last saw it); make the change there."
)
OUTPUT_LIMIT_MESSAGE = (
    "Прервано: ответ модели {count} раза подряд оборвался, не дойдя до конца "
    "(finish_reason={reason}). Задача может быть не завершена."
)

# Ответ без вызовов с такой причиной завершения оборван, а не закончен моделью: вызов в нём
# мог потеряться. Он не считается финальным — запрос повторяется (не больше MAX_CUT_RETRIES
# раз за ход).
CUT_REASONS = frozenset(("length", "incomplete"))
MAX_CUT_RETRIES = 2

# Сколько раз за ход хуки Stop/SubagentStop могут вернуть модель к работе: хук, который
# никогда не отпускает, не должен крутить ход до предела шагов.
MAX_STOP_HOOK_BLOCKS = 5


# Сжатие контекста — доли бюджета истории (бюджет запроса за вычетом системного
# промпта и схем инструментов).
COMPACT_KEEP_SHARE = 0.25  # свежие сообщения, остающиеся дословно
COMPACT_MIN_SHARE = 0.2  # меньше — сжимать не стоит запроса к модели
SUMMARY_SHARE = 0.15  # предел краткого содержания
MAX_SUMMARY_CHARS = 12_000
MIN_SUMMARY_CHARS = 1_000

# Калибровка оценки токенов по фактическому usage: реальные токены / оценка.
MIN_TOKEN_SCALE = 0.5
MAX_TOKEN_SCALE = 1.5
# Размер запроса по usage меньше этой доли оценки неправдоподобен (оценка ~3 символа на
# токен, токенизаторы дают 2–5): API не учёл часть запроса — закешированную или всю.
MIN_PLAUSIBLE_SCALE = 0.25

# Похожие вызовы суб-агента (та же цель): на N-м — предупреждение, затем остановка.
SUBAGENT_MAX_SIMILAR = 5
USER_STOP_REASON = "Пользователь остановил суб-агента досрочно."

# Уровень предупреждений пользователю (AgentEvents.on_notice).
_WARN: Final = "warn"
# Права файла блокировки плана: только владелец.
_LOCK_FILE_MODE = 0o600
# Сколько символов текста ошибки попадает в пояснение рецензента и в журнал решений.
_MAX_ERROR_CHARS = 200
# Длительность в журнале решений для вызова, который не выполнялся (заблокирован).
_NO_DURATION = float(0)


def _with_note(content: str, note: str | None) -> str:
    return f"{content}\n\n{note}" if note else content


def _tool_text(content: str, note: str | None = None) -> str:
    """Результат вызова для истории: тег заметок в нём — цитата, затем заметка harness'а.

    Содержимое файла, найденные строки, текст ошибки (цитата из файла) не должны выдать
    себя за ``<system-reminder>``, который системный промпт велит выполнять.
    """
    return _with_note(quote_reminder_tags(content), note)


def _reminder(note: str) -> str:
    """Заметка harness'а для одного запроса — в теге, который описан в системном промпте."""
    return f"<system-reminder>\n{note}\n</system-reminder>"


def _join_notes(*notes: str | None) -> str:
    return "\n\n".join(note for note in notes if note)


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0] if text.strip() else ""


@dataclass(frozen=True)
class _Journal:
    id: str


@dataclass
class _TurnLoop:
    """Состояние цикла одного хода (:meth:`Agent._loop`)."""

    reviews: int  # показы плана до хода
    compact_failed: bool = False  # не удалось — до конца хода не пытаемся снова
    checked: bool = False  # самопроверка — не больше одной за ход
    cuts: int = 0  # оборванных ответов за ход
    held: int = 0  # сколько раз хуки Stop вернули модель к работе
    note: str = ""  # заметка к следующему запросу (повтор оборванного ответа)


def _confirm_result(*, answer: bool | Approval | ConfirmResult) -> ConfirmResult:
    if isinstance(answer, ConfirmResult):
        return answer
    if isinstance(answer, bool):
        return ConfirmResult(Approval.YES if answer else Approval.NO)
    return ConfirmResult(answer)


def _decision_name(answer: ConfirmResult) -> str:
    if not answer.answered:
        return "no_answer"
    return {Approval.YES: "yes", Approval.ALWAYS: "always", Approval.NO: "no"}[answer.approval]


def _auto_source(
    risk: RiskLevel,
    kind: ToolKind,
    mode: PermissionMode,
    *,
    auto_approve: bool,
) -> str:
    """Почему вызов выполняется без вопроса (для журнала)."""
    if risk < RiskLevel.WRITE:
        return "read"
    if kind is ToolKind.EDIT and mode is PermissionMode.AUTO:
        return "edits_mode"
    return "yes_flag" if auto_approve else "auto"


def _request_tokens(usage: Usage, estimate: int) -> int | None:
    """Размер отправленного запроса по ``usage`` (``estimate`` — его оценка).

    API может не сообщить размер запроса или не включить в ``prompt_tokens`` часть,
    взятую из кеша (``precached_prompt_tokens``), — тогда контекст «падает» до
    сотен токенов, меньше одного системного промпта. Размер намного меньше оценки
    неправдоподобен: пробуется сумма с кешем, иначе None (размер неизвестен).
    """
    floor = estimate * MIN_PLAUSIBLE_SCALE
    for size in (usage.prompt_tokens, usage.prompt_tokens + usage.precached_prompt_tokens):
        if size > 0 and size >= floor:
            return size
    return None


def _history_tokens(conversation: Conversation) -> int:
    """История, которую видит модель: краткое содержание + сообщения после него."""
    size = estimate_tokens(conversation.context_messages())
    if conversation.summary is not None:
        size += estimate_text_tokens(summary_prompt(conversation.summary.text))
    return size


def _add_usage(stats: TurnStats, usage: Usage) -> None:
    """Учесть расход обращения в статистике хода."""
    stats.prompt_tokens += usage.prompt_tokens
    stats.completion_tokens += usage.completion_tokens


def _describe(tool: Tool[Any], params: BaseModel) -> str:
    try:
        return tool.describe(params)
    except Exception:
        return ""


class Agent:
    """Агент: ведёт диалог с моделью и выполняет вызовы инструментов, которые она просит."""

    # Параметры конструктора — публичный API (их передают по имени CLI и тесты).
    def __init__(  # noqa: WPS211
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        config: Config,
        events: AgentEvents | None = None,
        *,
        workspace: Workspace | None = None,
        conversation: Conversation | None = None,
        windows: ModelWindows | None = None,
        parent: Agent | None = None,
        subagent: SubagentSpec | None = None,
        chat_id: str | None = None,
        decisions: DecisionMemory | None = None,
        reviewer: Reviewer | None = None,
        hooks: HookRunner | None = None,
    ):
        """Конструктор не обращается к файловой системе и сети.

        ``windows`` — замеренные окна моделей (по умолчанию пусто: окно
        :data:`~gigacrew.agent.context_window.DEFAULT_CONTEXT_WINDOW`).

        ``parent``/``subagent`` — это суб-агент: режим разрешений, разрешения «всегда»,
        калибровка токенов, инструкции подкаталогов и контекст проекта — общие с
        основным агентом, расход токенов учитывается и у него; запускать суб-агентов
        и задавать вопросы пользователю суб-агент не может.

        ``chat_id`` — id чата: по нему назван файл плана (по умолчанию — новый id).

        ``decisions`` — память решений о командах: разрешения «всегда» и журнал (по
        умолчанию — только в памяти, до конца сессии); у суб-агента — общая с основным.

        ``reviewer`` — модель-рецензент изменяющих команд в режиме «авто» (по умолчанию —
        из ``config.reviewer_model`` без проверки); у суб-агента — общая с основным.

        ``hooks`` — хуки пользователя (по умолчанию нет); у суб-агента — основного агента.
        """
        self._provider = provider
        self._registry = registry
        self._cfg = config
        # `is None`, а не `or`: пустой Conversation ложен (__len__ == 0).
        self._events = AgentEvents() if events is None else events
        if workspace is None:
            workspace = Workspace(config.project_root, untrusted=config.untrusted_project)
        self._workspace = workspace
        self._parent = parent
        self._subagent = subagent
        main = parent is None  # основной агент, а не суб-агент
        # Файл плана чата — только у основного агента: суб-агент план не пишет.
        self._chat_id = chat_id or ChatStore.new_id()
        self._plan = PlanFile(self._plan_path(self._chat_id)) if main else None
        # Файл плана пишет одна сессия чата: блокировка — у той, что начала его писать;
        # _plan_seen — план, каким эта сессия видела его при продолжении чата.
        self._plan_lock: BinaryIO | None = None  # закрытие (и сборка мусора) снимает flock
        self._plan_seen = self._read_plan_file() if chat_id else ""
        if parent is not None:
            hooks = parent.hooks
        self._hooks = HookRunner() if hooks is None else hooks
        # Контекст хуков SessionStart: в системном сообщении до reset().
        self._session_context = ""
        self._ctx = ToolContext(
            workspace=self._workspace,
            # Вопрос пользователю — повод для хука Notification (суб-агент не спрашивает).
            ask_user=self._ask_user if main else self._events.ask_user,
            get_mode=lambda: self.mode,
            set_mode=self.set_mode,
            semantic=config.ty,
            run_subagent=self._run_subagent if main else None,
            plan=self._plan,
            classify_commands=config.safe_commands,
        )
        self._conversation = Conversation() if conversation is None else conversation
        self._model_name = config.model if parent is None else parent.model
        # Режим разрешений — состояние сессии, как модель: reset() его не меняет.
        # Может смениться посреди хода (Shift+Tab из потока клавиш) — читается при
        # каждом вызове инструмента и каждом обращении к модели.
        self._permission_mode = config.mode
        self._windows = ModelWindows() if windows is None else windows
        # Контекст проекта (дерево, инструкции, карта): строится лениво, сбрасывается в
        # reset(); суб-агент берёт контекст основного агента.
        self._project_context: str | None = None
        self._billed_tokens = 0  # потрачено за сессию (reset() не сбрасывает)
        # Размер контекста последнего запроса (реальные токены; None — берётся из
        # last_usage диалога) и модель, в чьих токенах он и last_usage посчитаны.
        self._context: int | None = None
        self._context_model = self._model_name
        # Оценка контекста, пока настоящий размер неизвестен: (состояние, оценка).
        self._context_estimate: tuple[tuple[object, ...], int] | None = None
        self._last_turn: TurnStats | None = None
        if parent is None:
            # Инструкции подкаталогов, с которыми агент работал в этом диалоге.
            self._nested = NestedInstructions(self._workspace.root)
            # Реальные токены / оценка по модели: оценка ~3 символа на токен грубая, а
            # бюджеты считаются в её единицах (калибруется по prompt_tokens ответов).
            self._token_scale: dict[str, float] = {}
            # Вызовы, которые пользователь разрешил «всегда» (до конца сессии; reset() не
            # сбрасывает — как и режим).
            self._session_allowed: set[str] = set()
            self._decisions = DecisionMemory() if decisions is None else decisions
            self._reviewer = Reviewer(config.reviewer_model) if reviewer is None else reviewer
        else:
            self._share_session(parent)
        self._last_estimate = 0  # оценка последнего отправленного запроса
        self._turn_stats: TurnStats | None = None  # статистика идущего хода
        # Суб-агенты: запущенный сейчас (для остановки из потока клавиш) и число
        # запусков за ход.
        self._stop_lock = threading.Lock()
        self._child: Agent | None = None
        self._subagent_runs = 0
        # Остановка самого суб-агента пользователем: 0 — нет, 1 — подвести итог,
        # 2 — оборвать. ``_wrapping`` — уже подводит итог (следующая остановка обрывает).
        self._stop_level = 0
        self._wrapping = False

    # ------------------------------------------------------------------ #
    @property
    def conversation(self) -> Conversation:
        """История текущего диалога."""
        return self._conversation

    @property
    def workspace(self) -> Workspace:
        """Рабочее пространство проекта (корень и служебные каталоги)."""
        return self._workspace

    @property
    def model(self) -> str:
        """Имя текущей модели (суб-агент начинает с модели основного агента)."""
        return self._model_name

    @property
    def decisions(self) -> DecisionMemory:
        """Память решений о командах (разрешения «всегда», журнал)."""
        return self._decisions

    @property
    def reviewer(self) -> Reviewer:
        """Модель-рецензент изменяющих команд (режим «авто»)."""
        return self._reviewer

    @property
    def config(self) -> Config:
        """Настройки агента."""
        return self._cfg

    @property
    def mode(self) -> PermissionMode:
        """Режим разрешений; у суб-агента — режим основного агента."""
        return self._permission_mode if self._parent is None else self._parent.mode

    @property
    def hooks(self) -> HookRunner:
        """Хуки пользователя; у суб-агента — хуки основного агента."""
        return self._hooks

    @hooks.setter
    def hooks(self, runner: HookRunner) -> None:
        """Заменить хуки (``/hooks reload``); действует со следующего события."""
        self._hooks = runner

    @property
    def context_tokens(self) -> int:
        """Размер контекста в токенах текущей модели (0 — диалог пуст).

        Берётся по последнему обращению к модели. Оценка — пока настоящий размер
        неизвестен: после сжатия, после смены модели (у другой модели свой токенизатор),
        если сохранённый usage неправдоподобен. Настоящий размер станет известен со
        следующим запросом.
        """
        conversation = self._conversation
        if not len(conversation):
            return 0
        if self._context_model == self._model_name:
            if self._context is not None:
                return self._context
            usage = conversation.last_usage
            if usage is not None:  # продолженный чат
                request = min(self._context_estimate_units(), self._estimate_budget())
                prompt = _request_tokens(usage, request)
                if prompt is not None:
                    return prompt + usage.completion_tokens
        return self._to_real(self._context_estimate_units())

    @property
    def provider(self) -> LLMProvider:
        """Провайдер модели."""
        return self._provider

    @property
    def windows(self) -> ModelWindows:
        """Замеренные окна контекста моделей."""
        return self._windows

    @property
    def context_window(self) -> int | None:
        """Окно текущей модели — замеренное или сообщённое API (None — неизвестно)."""
        return self._windows.get(self._model_name)

    @property
    def context_budget(self) -> int:
        """Бюджет запроса в реальных токенах: от окна модели или явный из конфига.

        Явный ``GIGACREW_CONTEXT_TOKENS`` задан в единицах оценки — переводится в
        реальные токены, как и размер контекста в статус-строке.
        """
        if self._cfg.context_budget_tokens is not None:
            return self._to_real(self._cfg.context_budget_tokens)
        return budget_for_window(self.context_window or DEFAULT_CONTEXT_WINDOW)

    @property
    def token_scale(self) -> float:
        """Реальные токены на единицу оценки для текущей модели (1.0 — ещё не известно)."""
        return self._token_scale.get(self._model_name, 1.0)

    @property
    def last_turn(self) -> TurnStats | None:
        """Итоги последнего хода (None — ход прерван исключением или ещё не было)."""
        return self._last_turn

    @property
    def billed_tokens(self) -> int:
        """Токены, оплаченные за сессию (сумма по всем обращениям, включая прерванные ходы)."""
        return self._billed_tokens

    def set_model(self, name: str) -> None:
        """Сменить модель (действует со следующего обращения)."""
        name = name.strip()
        if not name:
            raise ValueError("Имя модели не может быть пустым")
        self._model_name = name

    @property
    def chat_id(self) -> str:
        """Id чата (сохранённого — тот же, что у файла чата): по нему назван файл плана."""
        return self._chat_id

    @chat_id.setter
    def chat_id(self, new_id: str) -> None:
        self._chat_id = new_id
        if self._plan is not None:
            self._release_plan()
            self._plan.path = self._plan_path(new_id)
            self._plan.dirty = False
            self._plan_seen = self._read_plan_file()

    def continue_as(self, chat_id: str) -> None:
        """Диалог сохранён новым чатом ``chat_id``: план сессии переезжает в его файл плана.

        Так бывает, когда прежний файл чата изменила другая сессия. В отличие от смены
        ``chat_id`` (другой чат — другой план) план не теряется: и непоказанный
        (``dirty``), и уже показанный.
        """
        plan = self._plan
        self._chat_id = chat_id
        if plan is None:
            return
        # Свой (записанный этой сессией) план — с диска; чужой — каким его видела сессия.
        text = self._plan_seen if self._plan_lock is None else self._read_plan_file()
        self._release_plan()
        plan.path = self._plan_path(chat_id)
        self._plan_seen = text
        self._seed_plan(text)

    @property
    def plan_path(self) -> Path | None:
        """Файл плана чата (None — у суб-агента)."""
        return None if self._plan is None else self._plan.path

    def set_mode(self, mode: PermissionMode) -> None:
        """Сменить режим разрешений (действует со следующего вызова инструмента).

        У суб-агента режим — основного агента.
        """
        if self._parent is not None:
            self._parent.set_mode(mode)
            return
        self._permission_mode = PermissionMode(mode)

    def cycle_mode(self) -> PermissionMode:
        """Следующий режим по кругу (Shift+Tab); возвращает новый режим."""
        self.set_mode(next_mode(self.mode))
        return self.mode

    def request_subagent_stop(self) -> bool:
        """Остановить работающего суб-агента (Esc; вызывается из потока клавиш).

        Первый раз — суб-агент подведёт итог по сделанному, повторно (или когда он
        уже подводит итог) — будет оборван без отчёта. Само прерывание текущего
        запроса или команды — ``KeyboardInterrupt`` в основном потоке — посылает
        вызывающий. False — суб-агент не работает: прерывать нужно ход целиком.
        """
        with self._stop_lock:
            if self._child is None:
                return False
            child: Agent = self._child
            child._stop_level = 2 if child._wrapping else child._stop_level + 1
            return True

    def reset(
        self,
        conversation: Conversation | None = None,
        *,
        chat_id: str | None = None,
    ) -> None:
        """Начать новый диалог или продолжить сохранённый (``conversation``).

        ``chat_id`` — id чата (продолженного — его id: агент найдёт его план); по
        умолчанию — новый. Контекст проекта будет собран заново; потраченные токены
        сессии не сбрасываются.
        """
        self.chat_id = chat_id or ChatStore.new_id()
        self._conversation = Conversation() if conversation is None else conversation
        self._project_context = None
        self._session_context = ""  # новый чат — хуки SessionStart запустит вызывающий
        self._nested.reset()
        # usage продолженного чата — в токенах его модели: она становится текущей до reset().
        self._context = None
        self._context_model = self._model_name

    # ------------------------------------------------------------------ #
    def start_session(self, source: str) -> None:
        """Хуки SessionStart (``source``: ``startup``, ``resume``, ``clear``).

        Их контекст уходит в системное сообщение — переживает обрезку и сжатие истории —
        до следующего :meth:`reset`.
        """
        outcome = self._hook(
            HookEvent.SESSION_START,
            source,
            source=source,
            model=self._model_name,
        )
        self._session_context = outcome.context_text

    def end_session(self, reason: str) -> None:
        """Хуки SessionEnd: только их побочные действия.

        ``reason``: ``prompt_input_exit``, ``other``, ``clear``, ``resume``.
        """
        self._hook(HookEvent.SESSION_END, reason, reason=reason)

    # ------------------------------------------------------------------ #
    def run_turn(self, user_input: str) -> str:
        """Обрабатывает один запрос пользователя до финального ответа.

        При любом прерывании (Ctrl+C, ошибка LLM) история приводится в
        согласованное состояние, исключение пробрасывается дальше. Запрос, отклонённый
        хуком UserPromptSubmit, — :class:`~gigacrew.agent.hooks.PromptBlockedError`, история
        не меняется.
        """
        self._last_turn = None
        try:
            return self._run_turn(user_input)
        except BaseException:
            self._conversation.repair()
            raise

    # ------------------------------------------------------------------ #
    def project_context(self) -> str:
        """Контекст проекта для системного промпта (строится один раз за диалог)."""
        if self._parent is not None:
            return self._parent.project_context()
        if self._project_context is None:
            self._project_context = build_project_context(self._workspace)
        return self._project_context

    def system_prompt(self) -> str:
        """Системный промпт агента (у суб-агента — свой) с контекстом проекта."""
        if self._subagent is not None:
            return subagent_system_prompt(self._subagent.name, self.project_context())
        return compose_system_prompt(SYSTEM_PROMPT, self.project_context())

    # ------------------------------------------------------------------ #
    def compact(self, instructions: str = "") -> CompactResult | None:
        """Сжать весь диалог в краткое содержание (команда ``/compact``).

        ``instructions`` — пожелания пользователя к резюме. None — сжимать нечего.
        Ошибка модели — :class:`~gigacrew.llm.base.LLMError`; диалог тогда не меняется.
        """
        specs = self._registry.specs()
        return self._compact(self._history_budget(specs), specs, instructions=instructions)

    def call_info(self, call: FunctionCall) -> ToolCallInfo:
        """Вызов из истории (продолженный чат) — как его показывает интерфейс во время хода.

        Описание, вид, только ли смотрит проект. Неизвестный инструмент — только имя,
        неверные аргументы — без описания.
        """
        tool = self._registry.get(call.name)
        if tool is None:
            return ToolCallInfo(call.name)
        try:
            params = tool.parse(call.arguments)
        except Exception:
            return ToolCallInfo(call.name, kind=tool.kind, read_only=tool.read_only)
        return ToolCallInfo(
            call.name,
            _describe(tool, params),
            tool.kind,
            read_only=tool.read_only,
        )

    def _share_session(self, parent: Agent) -> None:
        """Суб-агент: общее с основным агентом состояние сессии.

        Инструкции подкаталогов, калибровка токенов, разрешения «всегда», память
        решений и рецензент — одни на сессию.
        """
        self._nested = parent._nested
        self._token_scale = parent._token_scale
        self._session_allowed = parent._session_allowed
        self._decisions = parent._decisions
        self._reviewer = parent._reviewer

    def _context_estimate_units(self) -> int:
        """Оценка контекста (системное сообщение, схемы, история) в единицах оценки.

        Кешируется по состоянию диалога: свойство ``context_tokens`` читается при
        каждой перерисовке статус-строки.
        """
        conversation = self._conversation
        key = (
            id(conversation),
            len(conversation),
            conversation.summary,
            self._model_name,
            self.mode,
            len(self._nested.files),
            self._session_context,
        )
        if self._context_estimate is None or self._context_estimate[0] != key:
            specs = self._registry.specs()
            size = self._overhead_tokens(specs) + _history_tokens(conversation)
            self._context_estimate = (key, size)
        return self._context_estimate[1]

    def _estimate_budget(self) -> int:
        """Бюджет запроса в единицах оценки, в которых меряется история.

        Бюджет от окна модели — в реальных токенах: делится на калибровку, иначе
        грубая оценка (~3 символа на токен) срабатывала бы раньше времени и часть
        окна пропадала. Явный ``GIGACREW_CONTEXT_TOKENS`` уже задан в единицах оценки.
        """
        if self._cfg.context_budget_tokens is not None:
            return self._cfg.context_budget_tokens
        return int(self.context_budget / self.token_scale)

    def _to_real(self, estimate: int) -> int:
        return int(estimate * self.token_scale)

    def _calibrate(self, prompt_tokens: int) -> None:
        """Уточнить калибровку по фактическому размеру отправленного запроса.

        Рост принимается сразу (осторожность), снижение — наполовину: один запрос с
        необычным текстом не должен резко раздвинуть бюджет.
        """
        if prompt_tokens <= 0 or self._last_estimate <= 0:
            return
        ratio = min(max(prompt_tokens / self._last_estimate, MIN_TOKEN_SCALE), MAX_TOKEN_SCALE)
        old = self._token_scale.get(self._model_name)
        if old is not None and ratio < old:
            ratio = (old + ratio) / 2
        self._token_scale[self._model_name] = ratio

    def _read_plan_file(self) -> str:
        """Текст файла плана ("" — нет, не читается или это ссылка)."""
        plan = self._plan
        if plan is None or not plan.is_target(plan.path):
            return ""
        try:
            return plan.path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return ""

    def _seed_plan(self, text: str) -> None:
        """Записать ``text`` в новый файл плана и занять его (пустой план — не писать)."""
        plan = self._plan
        if plan is None or not text:
            return
        try:
            self._workspace.ensure_data_dir()
            plan.path.parent.mkdir(parents=True, exist_ok=True)
            if self._lock_plan() and plan.is_target(plan.path):
                write_atomic(plan.path, text.encode())
        except OSError as exc:
            self._events.on_notice(
                f"план не перенесён в {self._plan_display()}: {exc}",
                level=_WARN,
            )

    def _lock_plan(self) -> bool:
        """Занять файл плана для этой сессии; False — его уже пишет другая сессия.

        Блокировка — ``flock`` на соседнем файле (сам план заменяется атомарно, у него
        меняется inode); снимается при смене чата или с завершением процесса. Нечем
        блокировать (Windows, файловая система без блокировок) — True: защиты нет, но и
        запись не запрещается.
        """
        plan = self._plan
        if plan is None or self._plan_lock is not None or not _HAS_FLOCK:
            return True
        stem = plan.path.stem
        lock = plan.path.with_name(f".{stem}.lock")
        try:
            fd = self._open_plan_lock(plan, lock)
        except OSError:
            return True
        if fd is None:  # ссылка — запись плана и так откажет
            return True
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        except OSError:
            os.close(fd)
            return True
        self._plan_lock = os.fdopen(fd, "r+b", buffering=0)
        return True

    def _open_plan_lock(self, plan: PlanFile, lock: Path) -> int | None:
        """Открыть файл блокировки ``lock`` плана; None — файл плана — ссылка."""
        self._workspace.ensure_data_dir()
        lock.parent.mkdir(parents=True, exist_ok=True)
        if not plan.is_target(plan.path):
            return None
        return os.open(
            lock,
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
            _LOCK_FILE_MODE,
        )

    def _release_plan(self) -> None:
        if self._plan_lock is not None:
            self._plan_lock.close()
            self._plan_lock = None

    def _own_plan(self, tool: Tool[Any], params: BaseModel) -> str | None:
        """Перед записью плана: файл плана чата пишет одна сессия.

        Один чат могут продолжить две сессии (два терминала с ``gigacrew -c``), а файл
        плана назван по id чата — они затирали бы планы друг друга. Если файл уже занят
        другой сессией, план этой переезжает в новый файл (с планом, каким она его
        видела). Возвращает заметку для модели, если вызов писал в прежний файл.
        """
        plan = self._plan
        if plan is None or self._plan_lock is not None:
            return None
        old = plan.path
        targets = tool.name == "exit_plan_mode" or self._touches_plan(plan, tool, params)
        if not targets or self._lock_plan():
            return None
        old_display = self._plan_display()
        plan.path = self._plan_path(ChatStore.new_id())
        self._seed_plan(self._plan_seen)
        self._lock_plan()
        new_display = self._plan_display()
        self._events.on_notice(
            f"файл плана {old_display} пишет другая сессия gigacrew этого чата — "
            f"план этой сессии теперь в {new_display}",
            level=_WARN,
        )
        if tool.name == "exit_plan_mode" or plan.path == old:
            return None
        return PLAN_MOVED_NOTE.format(old=old_display, new=new_display)

    def _touches_plan(self, plan: PlanFile, tool: Tool[Any], params: BaseModel) -> bool:
        """Работает ли вызов с файлом плана ``plan`` (пути — не строки — пропускаются)."""
        # Тип элементов шире объявленного: инструмент может вернуть не только строки.
        paths: Sequence[object] = tool.paths(params)
        return any(
            plan.is_target(resolve_in_root(self._workspace.root, path))
            for path in paths
            if isinstance(path, str)
        )

    @property
    def _session_chat_id(self) -> str:
        """Id чата для хуков и журнала решений: у суб-агента своего чата нет — основного."""
        return self._chat_id if self._parent is None else self._parent.chat_id

    @property
    def _is_subagent(self) -> bool:
        """Это суб-агент: у него есть основной агент."""
        return self._parent is not None

    def _plan_path(self, chat_id: str) -> Path:
        return self._workspace.plans_dir / f"{chat_id}.md"

    def _hook(self, event: HookEvent, target: str = "", **fields: Any) -> HookResult:
        """Запустить хуки события; нет подходящих — пустой результат без процессов.

        Общие поля stdin — как у Claude Code; id чата у суб-агента — основного агента.
        """
        if not self._hooks.has(event, target):
            return HookResult()
        chat_id = self._session_chat_id
        payload = {
            "session_id": chat_id,
            "transcript_path": str(self._workspace.chats_dir / f"{chat_id}.json"),
            "cwd": str(self._workspace.root),
            "hook_event_name": event.value,
            "permission_mode": self.mode.value,
            **fields,
        }
        return self._hooks.run(
            event,
            payload,
            cwd=self._workspace.root,
            events=self._events,
            target=target,
        )

    def _notify(self, kind: str, message: str) -> None:
        """Хуки Notification: gigacrew ждёт пользователя."""
        self._hook(HookEvent.NOTIFICATION, kind, message=message, notification_type=kind)

    def _ask_user(self, questions: Sequence[Question]) -> list[Answer] | None:
        if questions:
            first = questions[0]
            topic = first.header or first.text
            self._notify("question", f"gigacrew задаёт вопрос: {topic}")
        return self._events.ask_user(questions)

    def _prompt_hooks(self, prompt: str) -> str:
        """Хуки UserPromptSubmit: запрос с их контекстом или :class:`PromptBlockedError`."""
        outcome = self._hook(HookEvent.USER_PROMPT_SUBMIT, prompt=prompt)
        if outcome.blocked:
            raise PromptBlockedError(outcome.reason)
        if not outcome.context:
            return prompt
        context = _reminder(f"{HOOK_CONTEXT_HEADER}\n{outcome.context_text}")
        return f"{prompt}\n\n{context}"

    def _run_turn(self, user_input: str) -> str:
        if self._parent is None:  # задачу суб-агента пишет модель, а не пользователь
            user_input = self._prompt_hooks(user_input)
        # Рассуждения прошлых ходов модели больше не нужны (GIGACREW_REASONING=turn).
        self._conversation.forget_reasoning()
        self._conversation.add_user(user_input)
        specs = self._registry.specs()
        guard = self._new_guard()
        stats = TurnStats()
        self._turn_stats = stats
        self._subagent_runs = 0
        self._wrapping = False
        started = time.monotonic()
        try:
            final_text = self._loop(specs, guard, stats)
        except KeyboardInterrupt:
            if not self._take_soft_stop():
                raise
            final_text = self._user_stop(specs, stats)
        finally:
            self._turn_stats = None
            stats.changes = guard.changes

        stats.duration_s = time.monotonic() - started
        self._last_turn = stats
        self._events.on_turn_end(stats)
        return final_text

    def _new_guard(self) -> LoopGuard:
        """Ограничители хода; у суб-агента — ещё предел контекста, времени и похожие вызовы.

        Предел контекста — доля окна модели (не замерено — окно по умолчанию), но не
        больше бюджета запроса: сверх него старые сообщения уже отбрасывались бы.
        """
        cfg = self._cfg
        if self._parent is None:
            return LoopGuard(
                max_steps=cfg.max_steps,
                max_failures=cfg.max_tool_failures,
                max_repeats=cfg.max_tool_repeats,
            )
        return LoopGuard(
            max_steps=cfg.max_steps,  # у суб-агента уже subagent_max_steps
            max_failures=cfg.max_tool_failures,
            max_repeats=cfg.max_tool_repeats,
            max_similar=SUBAGENT_MAX_SIMILAR,
            max_context=min(
                int((self.context_window or DEFAULT_CONTEXT_WINDOW) * cfg.subagent_context_share),
                self._to_real(self._estimate_budget()),
            ),
            time_limit=cfg.subagent_timeout,
        )

    def _take_soft_stop(self) -> bool:
        """Прерывание — мягкая остановка суб-агента пользователем (подвести итог)?"""
        return self._is_subagent and self._stop_level == 1 and not self._wrapping

    def _user_stop(self, specs: list[ToolSpec], stats: TurnStats) -> str:
        """Суб-агент остановлен пользователем: итог по сделанному одним запросом.

        Повторная остановка во время итога (``KeyboardInterrupt``) пробрасывается —
        суб-агент обрывается без отчёта.
        """
        self._conversation.repair()
        stats.stop_reason = "user_stop"
        self._events.on_notice("остановлен пользователем — подвожу итог", level=_WARN)
        return self._wrap_up(
            specs,
            USER_STOP_REASON,
            MODEL_STOP_REASONS["user_stop"],
            stats,
            user_stop=True,
        )

    def _loop(self, specs: list[ToolSpec], guard: LoopGuard, stats: TurnStats) -> str:
        """Шаги хода до финального ответа модели или остановки ограничителем.

        Ход в режиме плана, закончившийся текстом, harness проверяет по структуре, не по
        тексту ответа: записанный, но не отправленный план показывает пользователю сам
        (:meth:`_auto_review`), а если план в ходе не показывался — переспрашивает модель
        (:meth:`_self_check`).
        """
        state = _TurnLoop(reviews=0 if self._plan is None else self._plan.reviews)
        while True:
            final_text = self._step(specs, guard, stats, state)
            if final_text is not None:
                return final_text

    def _step(
        self,
        specs: list[ToolSpec],
        guard: LoopGuard,
        stats: TurnStats,
        state: _TurnLoop,
    ) -> str | None:
        """Один шаг хода: запрос к модели и его исход.

        Возвращает финальный текст хода; None — ход продолжается следующим шагом.
        """
        context = self._subagent_context(specs, stats)
        stop = guard.before_step(context)
        if stop is not None:
            return self._stopped(specs, stop, stats)
        stats.steps = guard.steps
        if self._cfg.auto_compact and not state.compact_failed:
            state.compact_failed = not self._auto_compact(specs, stats)
        failures_note = self._failures_note(guard)
        note = _join_notes(
            guard.pressure(context),
            failures_note,
            self._plan_reminder(state.reviews),
            state.note,
        )
        turn = self._next_turn(specs, note=note)
        state.note = ""
        self._account(turn.usage, stats)
        if not turn.wants_tool and turn.finish_reason not in CUT_REASONS:
            if self._plan_unsubmitted():
                return self._submit_plan(turn, state)
            if not state.checked and self._self_check_due(state.reviews):
                state.checked = True
                followup = self._self_check(specs, turn, stats, failures_note)
                if followup is None:
                    return self._finish(turn.message.content, state)
                turn = followup  # с вызовами или оборван — дальше как обычно
        msg = turn.message
        # Повтор: ответ текстом не идёт в историю — запрос не должен кончаться репликой
        # ассистента, — а заметка дописывается к хвосту следующего запроса.
        if not turn.wants_tool and turn.finish_reason in CUT_REASONS:
            return self._retry_cut(specs, turn, stats, state)
        self._conversation.add_assistant(msg, turn.usage)

        if not turn.wants_tool:
            return self._finish(msg.content, state)
        stop = self._run_calls(msg.function_calls, guard, stats)
        if stop is not None:
            return self._stopped(specs, stop, stats)
        return None

    def _failures_note(self, guard: LoopGuard) -> str:
        """Предупреждение о серии неудач: пользователю — сразу, модели — заметка ("" — нет)."""
        warning = guard.failure_warning()
        if warning is None:
            return ""
        if warning.notice:
            self._events.on_notice(warning.notice, level=_WARN)
        return warning.model

    def _submit_plan(self, turn: AssistantTurn, state: _TurnLoop) -> str | None:
        """Ответ текстом, а план записан, но не отправлен: показать план пользователю.

        Возвращает финальный текст хода; None — ход продолжается.
        """
        self._conversation.add_assistant(turn.message, turn.usage)
        review = self._auto_review()
        if review is not None and review.continues:
            decision = AUTO_REVIEW_NOTE.format(result=review.content)
            self._conversation.add_user(_reminder(decision), meta=True)
            return None
        # План отклонён (Esc) — пользователь остановил ход: без хуков.
        if review is not None and review.outcome == "declined":
            return turn.message.content
        shown = "" if review is None else review.content
        pending = AUTO_REVIEW_NOTE.format(result=shown) if shown else ""
        return self._finish(turn.message.content, state, pending)

    def _retry_cut(
        self,
        specs: list[ToolSpec],
        turn: AssistantTurn,
        stats: TurnStats,
        state: _TurnLoop,
    ) -> str | None:
        """Ответ оборван: повторить запрос с заметкой, а после предела — остановить ход.

        Возвращает финальный текст хода; None — запрос повторяется.
        """
        self._forget_reply(turn, stats)
        if state.cuts >= MAX_CUT_RETRIES:
            message = OUTPUT_LIMIT_MESSAGE.format(
                count=state.cuts + 1,
                reason=turn.finish_reason,
            )
            return self._stopped(specs, StopReason("output_limit", message), stats)
        state.cuts += 1
        stats.cut_retries += 1
        state.note = OUTPUT_CUT_NOTE.format(reason=turn.finish_reason)
        self._events.on_notice(
            f"ответ модели оборван (finish_reason={turn.finish_reason}) — "
            f"повторяю запрос ({state.cuts}/{MAX_CUT_RETRIES})",
            level=_WARN,
        )
        return None

    def _finish(self, text: str, state: _TurnLoop, pending: str = "") -> str | None:
        """Закончить ход ответом ``text``, если хуки Stop не вернули модель к работе.

        Возвращает ``text``; None — хук вернул модель к работе, ход продолжается.
        """
        if not self._hold_turn(text, state.held, pending):
            return text
        state.held += 1
        return None

    def _hold_turn(self, text: str, held: int, pending: str = "") -> bool:
        """Хуки Stop (у суб-агента — SubagentStop) перед концом хода.

        True — хук вернул модель к работе, причина и ``pending`` (служебная заметка,
        которая иначе не понадобилась бы) записаны в историю служебным сообщением.

        Ответ ``text`` уже в истории, поэтому прерывание хука (Ctrl+C, Esc) не теряет
        его: ход заканчивается этим ответом. Только жёсткая остановка суб-агента
        пробрасывается.
        """
        if self._parent is None:
            event, target, extra = HookEvent.STOP, "", {}
        else:
            name = "" if self._subagent is None else self._subagent.name
            event, target, extra = HookEvent.SUBAGENT_STOP, name, {"agent_type": name}
        if not self._hooks.has(event, target):
            return False
        try:
            outcome = self._hook(
                event,
                target,
                stop_hook_active=held > 0,
                last_assistant_message=text,
                **extra,
            )
        except KeyboardInterrupt:
            if self._is_subagent and not self._take_soft_stop():
                raise
            self._events.on_notice(f"хук {event} прерван — ход завершён", level=_WARN)
            return False
        if not outcome.blocked:
            return False
        if held >= MAX_STOP_HOOK_BLOCKS:
            self._events.on_notice(
                f"хук {event} снова не дал закончить ход — ход завершён "
                f"(уже {MAX_STOP_HOOK_BLOCKS} раз подряд)",
                level=_WARN,
            )
            return False
        reason = _first_line(outcome.reason)
        self._events.on_notice(f"хук {event} вернул агента к работе: {reason}")
        stop_note = STOP_HOOK_NOTE.format(event=event, reason=outcome.reason)
        self._conversation.add_user(_reminder(_join_notes(pending, stop_note)), meta=True)
        return True

    def _active_plan(self) -> PlanFile | None:
        """Файл плана, если это основной агент в режиме плана (иначе None)."""
        if self._is_subagent or self.mode is not PermissionMode.PLAN:
            return None
        return self._plan

    def _plan_reminder(self, reviews: int) -> str:
        """Напоминание о режиме плана для запроса основного агента ("" — не план).

        После показа плана в этом ходе (``reviews`` — показы до хода) не нужно: что делать
        дальше, модели сказал исход показа (доработать, закончить, ждать).
        """
        plan = self._active_plan()
        if plan is None or plan.reviews != reviews:
            return ""
        return PLAN_REMINDER_NOTE.format(path=self._plan_display())

    def _plan_unsubmitted(self) -> bool:
        """Режим плана, а план менялся после того, как его показали пользователю."""
        plan = self._active_plan()
        return plan is not None and plan.dirty

    def _auto_review(self) -> PlanReview | None:
        """Показать пользователю план, который модель записала, но не отправила.

        Исход — как у ``exit_plan_mode``; None — показать нечего (файл плана пуст или не
        читается).
        """
        plan = self._plan
        if plan is None:
            return None
        try:
            review = review_plan(
                self._ctx,
                plan,
                on_show=lambda: self._events.on_notice(
                    "план записан, но не отправлен на одобрение — показываю его",
                ),
            )
        except ToolError:
            plan.dirty = False
            self._events.on_notice("план не показан: файл плана пуст или не читается", level=_WARN)
            return None
        summary = review.summary
        if review.outcome == "no_approver":
            summary = f"{summary} — план сохранён в {self._plan_display()}"
        self._events.on_notice(summary)
        return review

    def _self_check_due(self, reviews: int) -> bool:
        """Нужна ли самопроверка: основной агент в режиме плана, план не менялся.

        И в этом ходе план не показывался (``reviews`` — число показов до хода).
        """
        plan = self._active_plan()
        return plan is not None and not plan.dirty and plan.reviews == reviews

    def _self_check(
        self,
        specs: list[ToolSpec],
        turn: AssistantTurn,
        stats: TurnStats,
        note: str = "",
    ) -> AssistantTurn | None:
        """Ход в режиме плана кончился текстом — без плана и без вопроса: переспросить.

        Модель могла пообещать действие («сейчас запишу план») и остановиться, а могла
        просто ответить на вопрос; по тексту их не отличить — спрашивается сама модель.
        Ответ ``turn`` остаётся в истории, за ним — служебное сообщение с вопросом, и
        ответ на него не печатается. С вызовами или оборванный ответ возвращается — ход
        идёт дальше как обычно. Только текст — служебное сообщение убирается, финальным
        остаётся ``turn`` (None); так же при сбое запроса. ``note`` — заметка к запросу
        (предупреждение о серии неудач: вызовы ответа выполнятся).
        """
        stats.self_checks += 1
        conversation = self._conversation
        conversation.add_assistant(turn.message, turn.usage)
        mark, context = len(conversation), stats.context_tokens
        question = SELF_CHECK_NOTE.format(path=self._plan_display())
        conversation.add_user(_reminder(question), meta=True)
        try:
            check = self._next_turn(specs, note=note, quiet=True)
        except LLMError as exc:
            conversation.truncate(mark)
            self._events.on_notice(f"самопроверка плана не удалась: {exc}", level=_WARN)
            return None
        except BaseException:
            conversation.truncate(mark)
            raise
        self._account(check.usage, stats)
        if check.wants_tool or check.finish_reason in CUT_REASONS:
            return check
        conversation.truncate(mark)
        stats.context_tokens = context
        if self._context_model == self._model_name:
            self._context = context
        return None

    def _run_calls(
        self,
        calls: list[FunctionCall],
        guard: LoopGuard,
        stats: TurnStats,
    ) -> StopReason | None:
        """Выполняет вызовы из ответа модели по очереди.

        На каждый вызов в историю идёт ровно один результат (иначе API отвечает 422):
        если ограничитель остановил ход посреди пачки, оставшиеся вызовы закрываются
        заглушкой.
        """
        for call in calls:
            check = guard.before_tool(call)
            if check.stop is not None:
                # Вызов не выполняется; история остаётся согласованной.
                self._conversation.add_function_result(call.name, LOOP_STOP_NOTE)
                self._conversation.repair(SKIPPED_NOTE)
                return check.stop
            if check.notice:
                self._events.on_notice(check.notice, level=_WARN)
            stats.tool_calls += 1
            outcome = self._execute_tool_call(call, check)
            stop = guard.after_tool(call, outcome)
            if stop is not None:
                self._conversation.repair(SKIPPED_NOTE)
                return stop
        return None

    def _subagent_context(self, specs: list[ToolSpec], stats: TurnStats) -> int:
        """Размер контекста следующего запроса суб-агента в токенах (у основного — 0).

        Оценка всей истории без обрезки окном (с калибровкой по ответам), но не меньше
        контекста последнего ответа: результаты инструментов после него — тоже контекст.
        """
        if self._parent is None:
            return 0
        estimate = self._overhead_tokens(specs) + _history_tokens(self._conversation)
        return max(self._to_real(estimate), stats.context_tokens)

    def _stopped(self, specs: list[ToolSpec], stop: StopReason, stats: TurnStats) -> str:
        stats.stop_reason = stop.kind
        self._events.on_notice(stop.message, level="error")
        text = self._wrap_up(specs, stop.message, MODEL_STOP_REASONS[stop.kind], stats)
        if text:
            return text
        # Суб-агенту без итога — пустой ответ: отчёт сам назовёт причину остановки.
        return "" if self._is_subagent else stop.message

    def _forget_reply(self, turn: AssistantTurn, stats: TurnStats) -> None:
        """Ответ не идёт в историю: его вывод не занимает контекст следующего запроса.

        Иначе оборванный длинный ответ раздул бы счётчик, и суб-агент ложно упёрся бы в
        предел контекста.
        """
        stats.context_tokens = max(stats.context_tokens - turn.usage.completion_tokens, 0)
        if self._context_model == self._model_name:
            self._context = stats.context_tokens

    def _account(self, usage: Usage, stats: TurnStats) -> None:
        """Учесть расход обращения: статистика хода, сессия, размер контекста, калибровка.

        Неправдоподобный размер запроса (:func:`_request_tokens`) заменяется оценкой
        и калибровку не сдвигает.
        """
        _add_usage(stats, usage)
        self._bill(usage)
        prompt = _request_tokens(usage, self._last_estimate)
        if prompt is None:
            prompt = self._to_real(self._last_estimate)
        else:
            self._calibrate(prompt)
        stats.context_tokens = prompt + usage.completion_tokens
        self._context = stats.context_tokens
        self._context_model = self._model_name

    def _bill(self, usage: Usage) -> None:
        """Оплаченные токены сессии; у суб-агента — ещё и у основного агента."""
        self._billed_tokens += usage.prompt_tokens + usage.completion_tokens
        if self._parent is not None:
            parent: Agent = self._parent
            parent._bill_child(usage)

    def _bill_child(self, usage: Usage) -> None:
        """Расход суб-агента: в сессию и в статистику идущего хода (не в контекст)."""
        self._billed_tokens += usage.prompt_tokens + usage.completion_tokens
        if self._turn_stats is not None:
            _add_usage(self._turn_stats, usage)

    def _wrap_up(
        self,
        specs: list[ToolSpec],
        reason: str,
        model_reason: str,
        stats: TurnStats,
        *,
        user_stop: bool = False,
    ) -> str:
        """Итог модели после остановки ограничителем: что сделано и что дальше.

        Инструменты в запросе остаются (иначе API не примет историю с вызовами), но
        вызывать их модели запрещено (``tool_choice="none"``); вызов, если он всё же
        пришёл, не выполняется — в историю идёт только текст. Сбой обращения к модели
        не мешает: остаётся сообщение ограничителя.

        Просьба подвести итог — заметка к последнему сообщению запроса, а не отдельное
        сообщение пользователя: окно истории закрепляет последний запрос пользователя, и
        такая просьба вытеснила бы из обрезанной истории саму задачу. ``user_stop`` —
        суб-агент остановлен пользователем (Esc). ``reason`` — причина для пользователя,
        ``model_reason`` — для модели.
        """
        self._wrapping = True
        self._events.on_wrap_up(reason)
        if user_stop:
            note = SUBAGENT_USER_STOP_NOTE
        elif self._is_subagent:
            note = SUBAGENT_STOP_NOTE.format(reason=model_reason)
        else:
            note = STOP_SUMMARY_NOTE.format(reason=model_reason)
        try:
            turn = self._next_turn(specs, note=note, tool_choice="none")
        except LLMError:
            return ""
        self._account(turn.usage, stats)
        text = turn.message.content.strip()
        if text:
            self._conversation.add_assistant(
                Message(role="assistant", content=turn.message.content),
                turn.usage,
            )
        return text

    def _system_text(self) -> str:
        """Системное сообщение без краткого содержания.

        Промпт, контекст хуков SessionStart, инструкции подкаталогов, правила режима.
        """
        parts = [
            self.system_prompt(),
            session_context_prompt(self._session_context),
            nested_instructions_prompt(self._nested.files, untrusted=self._workspace.untrusted),
            mode_prompt(
                self.mode,
                subagent=self._is_subagent,
                read_only=self._subagent is not None and self._subagent.read_only,
                plan_path=self._plan_display(),
            ),
        ]
        return "\n\n".join(part for part in parts if part)

    def _plan_display(self) -> str:
        """Путь файла плана для промпта — относительно корня проекта."""
        if self._plan is None:
            return ""
        try:
            return self._plan.path.relative_to(self._workspace.root).as_posix()
        except ValueError:
            return str(self._plan.path)

    def _overhead_tokens(self, specs: list[ToolSpec]) -> int:
        """Системное сообщение и схемы инструментов (уходят в каждом запросе)."""
        system = Message(role="system", content=self._system_text())
        return estimate_tokens([system]) + estimate_specs_tokens(specs)

    def _history_budget(self, specs: list[ToolSpec]) -> int:
        """Бюджет истории: бюджет запроса за вычетом системного сообщения и схем."""
        return max(self._estimate_budget() - self._overhead_tokens(specs), MIN_HISTORY_TOKENS)

    def _build_request(self, specs: list[ToolSpec], note: str = "") -> list[Message]:
        """Сообщения для модели: системный промпт + история в пределах бюджета.

        Единственная точка сборки запроса. Краткое содержание сжатого начала диалога
        дописывается к системному сообщению (API принимает одно системное
        сообщение — первым); история — сообщения после него, а если и они не
        укладываются в бюджет, старые отбрасываются (:func:`fit_history`).

        ``note`` — заметка только для этого запроса (итог остановки, бюджет, напоминание).
        Она дописывается в конец последнего сообщения — запроса пользователя или
        результата вызова, — то есть туда, где модель её видит, а не в начало контекста;
        дописывается к копии, диалог не меняется. Если хвост истории другой — к
        системному сообщению.
        """
        system_text = self._system_text()
        summary = self._conversation.summary
        if summary is not None:
            system_text = f"{system_text}\n\n{summary_prompt(summary.text)}"
        reminder = _reminder(note) if note else ""
        reminder_tokens = estimate_text_tokens(reminder) if reminder else 0
        # Схемы инструментов и заметка уходят в запросе и занимают то же окно.
        used = (
            estimate_tokens([Message(role="system", content=system_text)])
            + estimate_specs_tokens(specs)
            + reminder_tokens
        )
        budget = max(self._estimate_budget() - used, MIN_HISTORY_TOKENS)
        history = fit_history(self._conversation.context_messages(), budget)
        self._last_estimate = used + estimate_tokens(history)
        if reminder:
            if history and history[-1].role in {"user", "function"}:
                last = history[-1]
                history[-1] = last.model_copy(
                    update={"content": _with_note(last.content, reminder)},
                )
            else:
                system_text = f"{system_text}\n\n{reminder}"
        return [Message(role="system", content=system_text), *history]

    def _auto_compact(self, specs: list[ToolSpec], stats: TurnStats) -> bool:
        """Сжать историю, если она подошла к порогу. False — сжать не удалось."""
        budget = self._history_budget(specs)
        if _history_tokens(self._conversation) < self._cfg.compact_threshold * budget:
            return True
        count = partial(_add_usage, stats)
        try:
            self._compact(budget, specs, auto=True, on_usage=count)
        except LLMError as exc:
            self._events.on_notice(
                f"не удалось сжать контекст: {exc} — старые сообщения будут отбрасываться",
                level=_WARN,
            )
            return False
        return True

    def _compact(
        self,
        budget: int,
        specs: list[ToolSpec],
        *,
        auto: bool = False,
        instructions: str = "",
        on_usage: Callable[[Usage], None] | None = None,
    ) -> CompactResult | None:
        """Свернуть начало диалога в краткое содержание (``budget`` — бюджет истории).

        Автоматически (``auto``) свежие сообщения остаются дословно, а сжатие
        пропускается, если сворачивать почти нечего; по команде сворачивается всё.
        Диалог меняется только после успешного ответа модели.
        """
        conversation = self._conversation
        if auto:
            compaction = plan_compaction(
                conversation,
                keep_tokens=int(budget * COMPACT_KEEP_SHARE),
                min_tokens=int(budget * COMPACT_MIN_SHARE),
            )
        else:
            compaction = plan_compaction(conversation, keep_tokens=0)
        if compaction is None:
            return None
        trigger = "auto" if auto else "manual"
        self._hook(
            HookEvent.PRE_COMPACT,
            trigger,
            trigger=trigger,
            custom_instructions=instructions,
        )
        overhead = self._overhead_tokens(specs)
        before = _history_tokens(conversation)
        max_chars = int(budget * SUMMARY_SHARE) * CHARS_PER_TOKEN
        compacted: CompactResult | None = None
        self._events.on_compact_start(auto=auto)
        try:
            compacted = self._summarize_history(
                compaction,
                before=before,
                overhead=overhead,
                max_chars=max_chars,
                instructions=instructions,
                on_usage=on_usage,
                auto=auto,
            )
        except BaseException:
            self._events.on_compact_end(compacted)
            raise
        self._events.on_compact_end(compacted)
        return compacted

    def _summarize_history(
        self,
        compaction: CompactionPlan,
        *,
        before: int,
        overhead: int,
        max_chars: int,
        instructions: str,
        on_usage: Callable[[Usage], None] | None,
        auto: bool,
    ) -> CompactResult:
        """Заменить начало диалога (``compaction``) кратким содержанием от модели.

        ``before`` — размер истории до сжатия, ``overhead`` — системное сообщение и схемы
        (в единицах оценки). Краткое содержание не короче самой истории —
        :class:`~gigacrew.agent.compaction.CompactionError`, диалог не меняется.
        """
        conversation = self._conversation
        previous = None if conversation.summary is None else conversation.summary.text
        text = summarize(
            self._provider,
            compaction.messages,
            model=self._model_name,
            temperature=self._cfg.temperature,
            input_budget=self._estimate_budget(),
            max_chars=min(max(max_chars, MIN_SUMMARY_CHARS), MAX_SUMMARY_CHARS),
            previous=previous,
            instructions=instructions,
            on_usage=partial(self._bill_compaction, on_usage),
        )
        summary = Summary(text=text, upto=compaction.upto)
        after = _history_tokens(Conversation(conversation.messages, summary=summary))
        if after >= before:
            raise CompactionError("краткое содержание вышло не короче самой истории")
        conversation.set_summary(summary)
        self._context = self._to_real(overhead + after)
        self._context_model = self._model_name
        return CompactResult(
            before_tokens=self._to_real(overhead + before),
            after_tokens=self._context,
            messages=len(compaction.messages),
            auto=auto,
        )

    def _bill_compaction(self, on_usage: Callable[[Usage], None] | None, usage: Usage) -> None:
        """Расход запроса сжатия: в сессию и вызывающему (``on_usage``)."""
        self._bill(usage)
        if on_usage is not None:
            on_usage(usage)

    def _next_turn(
        self,
        specs: list[ToolSpec],
        note: str = "",
        *,
        tool_choice: ToolChoice = "auto",
        quiet: bool = False,
    ) -> AssistantTurn:
        """Один проход модели с выводом текста (потоковым или цельным).

        ``note`` — заметка только для этого запроса (см. :meth:`_build_request`);
        ``tool_choice="none"`` — ответ только текстом (итог после остановки); ``quiet`` —
        текст показывается, только если в ответе есть вызовы, и целиком после ответа
        (самопроверка). Оборванный поток (Esc, сбой сети) оставляет в истории основного
        агента уже показанный текст с пометкой — модель знает, что её ответ прерван, а
        следующий запрос не идёт двумя репликами пользователя подряд. У суб-агента текст
        не показывается и не сохраняется.
        """
        messages = self._build_request(specs, note)
        streamed: list[str] = []
        on_delta = partial(self._stream_delta, streamed, quiet=quiet)
        self._events.on_stream_start()
        try:
            if self._cfg.stream:
                turn = self._provider.stream(
                    messages,
                    tools=specs,
                    model=self._model_name,
                    temperature=self._cfg.temperature,
                    tool_choice=tool_choice,
                    on_delta=on_delta,
                    on_notice=self._provider_notice,
                )
            else:
                turn = self._provider.complete(
                    messages,
                    tools=specs,
                    model=self._model_name,
                    temperature=self._cfg.temperature,
                    tool_choice=tool_choice,
                    on_notice=self._provider_notice,
                )
        except BaseException as exc:
            partial_text = "".join(streamed).rstrip()
            # Суб-агенту оборванный текст не нужен: пользователь его не видел, а запрос
            # итога после остановки (Esc) не должен кончаться репликой ассистента. Тихий
            # ответ пользователь тоже не видел.
            if partial_text.strip() and not self._is_subagent and not quiet:
                tail = (
                    INTERRUPTED_ANSWER_NOTE
                    if isinstance(exc, KeyboardInterrupt)
                    else (BROKEN_ANSWER_NOTE)
                )
                self._conversation.add_assistant(
                    Message(role="assistant", content=partial_text + tail),
                )
            raise
        finally:
            self._events.on_stream_end()

        shown = self._cfg.stream and not quiet  # текст уже показан потоком
        if not shown and turn.message.content.strip() and (turn.wants_tool or not quiet):
            self._events.on_assistant_text(turn.message.content)
        if self._cfg.reasoning == "off" and turn.message.reasoning:
            # Не возвращаются модели — и не хранятся: размер истории равен отправляемому.
            message = turn.message.model_copy(update={"reasoning": ""})
            turn = turn.model_copy(update={"message": message})
        return turn

    def _stream_delta(self, streamed: list[str], text: str, *, quiet: bool) -> None:
        """Кусок потокового ответа: в ``streamed`` и, если ответ не тихий, — на экран."""
        streamed.append(text)
        if not quiet:
            self._events.on_stream_delta(text)

    def _provider_notice(self, text: str) -> None:
        """Сообщение провайдера пользователю.

        Пауза при лимите запросов, отказ шлюза принять рассуждения.
        """
        self._events.on_notice(text, level=_WARN)

    # ------------------------------------------------------------------ #
    # Порядок проверок разрешения — граница безопасности: шаги видны подряд, в одном месте.
    def _execute_tool_call(  # noqa: C901, WPS210, WPS212, WPS213, WPS231
        self,
        function_call: FunctionCall,
        check: CallCheck,
    ) -> ToolOutcome:
        """Выполняет запрошенный моделью инструмент.

        ``check.warning`` (предупреждение ограничителя) дописывается к результату,
        который получит модель.
        """
        name = function_call.name
        tool = self._registry.get(name)
        note = check.warning

        if tool is None:
            unknown = ToolCallInfo(name, "(неизвестный инструмент)")
            self._events.on_tool_call(unknown)
            return self._fail(unknown, f"инструмент '{name}' не существует.", note=note)

        # 1) Валидация параметров
        try:
            params = tool.parse(function_call.arguments)
        except Exception as exc:  # ошибка схемы — возвращаем модели
            invalid = ToolCallInfo(name, "(неверные аргументы)")
            self._events.on_tool_call(invalid)
            return self._fail(invalid, f"валидация аргументов: {exc}", note=note)

        call = ToolCallInfo(name, _describe(tool, params), tool.kind, read_only=tool.read_only)
        self._events.on_tool_call(call)
        key = call_key(function_call)

        # Хуки PreToolUse: запретить вызов или решить, спрашивать ли пользователя.
        hook_fields = {
            "tool_name": name,
            "tool_input": function_call.arguments,
            "tool_use_id": function_call.id,
        }
        pre = self._hook(HookEvent.PRE_TOOL_USE, name, **hook_fields)
        if pre.blocked:
            self._fail(
                call,
                f"заблокировано хуком: {pre.reason}",
                model_text=HOOK_BLOCKED_NOTE.format(reason=pre.reason),
                note=note,
            )
            # Запрет хука — не застревание модели: серия ошибок не растёт.
            return ToolOutcome(ok=False, soft=True)

        # 2) Разрешение по режиму: выполнить, спросить или заблокировать
        previewed = False
        risk = RiskLevel.SAFE
        verdict: Verdict | None = None
        review: Review | None = None
        review_note = ""
        journal: _Journal | None = None
        hook_asks = pre.decision == "ask"
        # Шаги решения зависят друг от друга, а ошибка любого из них — отказ в вызове.
        try:  # noqa: WPS229
            assessment = tool.assess(params, self._ctx)
            risk, verdict = assessment.risk, assessment.verdict
            kind = assessment.kind or tool.kind
            read_only = self._subagent is not None and self._subagent.read_only
            decision = decide(
                self.mode,
                risk,
                kind,
                auto_approve=self._cfg.auto_approve,
                yes_all=self._cfg.yes_all,
                read_only=read_only,
            )
            source = _auto_source(risk, kind, self.mode, auto_approve=self._cfg.auto_approve)
            level = None if verdict is None else verdict.level
            if (
                decision in {Decision.ASK, Decision.ALLOW}  # -y не снимает запрет плана
                and self.mode is PermissionMode.PLAN
                and level is not None
                and level >= CommandLevel.WRITE
            ):
                decision = Decision.BLOCK  # план: изменяющая команда, а не проверка
            unverified = verdict is not None and level is None
            if decision is Decision.ALLOW and source == "yes_flag" and unverified:
                # -y не ручается за команду, которую классификатор не проверил: в плане
                # решает пользователь, в «авто» — как без -y (рецензент, затем пользователь).
                # Выполнять всё без вопроса обещает только --yes-all — и только в «авто».
                if self.mode is PermissionMode.PLAN:
                    decision = Decision.ASK
                elif not self._cfg.yes_all:
                    decision = Decision.REVIEW
            # Хук — после уточнений по вердикту: allow не снимает запрет режима плана
            # и, как -y, не ручается за непроверенную команду.
            hooked = apply_hook(decision, risk, pre.decision)
            if hooked is Decision.ALLOW and decision is not Decision.ALLOW:
                if not unverified:
                    decision, source = hooked, "hook"
            else:
                decision = hooked
            asking = (Decision.ASK, Decision.REVIEW)
            # Хук требует спросить — сохранённые разрешения и «всегда» не действуют.
            if decision in asking and verdict is not None and not hook_asks:
                rule = self._decisions.match(verdict, plan=self.mode is PermissionMode.PLAN)
                if rule is not None:
                    decision, source = Decision.ALLOW, f"rule:{rule.id}"
                    self._events.on_notice(f"разрешено сохранённым правилом: {rule.describe()}")
            # «Всегда» до конца сессии — не для опасного.
            remembered = key in self._session_allowed and risk < RiskLevel.DANGEROUS
            if (
                decision in asking
                and remembered
                and not hook_asks
                # план, как и сохранённые правила, не разрешает непроверенную команду
                and not (self.mode is PermissionMode.PLAN and unverified)
            ):
                decision, source = Decision.ALLOW, "session"  # «всегда» до конца сессии
            if decision is Decision.REVIEW:
                decision, review, review_note = self._review_command(call, params, verdict, check)
                if decision is Decision.ALLOW:
                    source = "reviewer"
            if decision is Decision.BLOCK:
                journal = self._journal_start(call, params, verdict, risk, "blocked", "")
                self._journal_end(journal, blocked=True)
                if read_only and verdict is not None:
                    # Непроверенная команда у исследователя: «нельзя здесь». В серию неудач
                    # идёт — иначе модель без конца перебирает варианты отклонённой команды.
                    return self._fail(
                        call,
                        "заблокировано: агент только читает",
                        model_text=READ_ONLY_SHELL_NOTE.format(
                            reason=assessment.reason or verdict.reason,
                        ),
                        note=note,
                    )
                if read_only:
                    error, model_text = "заблокировано: агент только читает", READ_ONLY_BLOCKED_NOTE
                elif self._is_subagent:
                    error, model_text = (
                        "заблокировано: режим планирования",
                        (SUBAGENT_PLAN_BLOCKED_NOTE),
                    )
                else:
                    error = "заблокировано: режим планирования"
                    model_text = PLAN_BLOCKED_NOTE.format(path=self._plan_display())
                return self._fail(call, error, model_text=model_text, note=note)
            if decision is Decision.ASK:
                if check.rejected_before:
                    # Тот же вызов уже отклонён в этом ходе — не переспрашиваем.
                    self._fail(
                        call,
                        "уже отклонено пользователем",
                        model_text=REJECTED_AGAIN_NOTE,
                        note=note,
                    )
                    return ToolOutcome(ok=False, rejected=True)
                # Превью заодно проверяет выполнимость: если оно падает, операция
                # не запускается и подтверждение не запрашивается.
                preview = tool.preview(params, self._ctx)
                dangerous = risk >= RiskLevel.DANGEROUS
                reason = assessment.reason
                if hook_asks:
                    hook_reason = "хук PreToolUse требует подтверждения"
                    reason = f"{reason}; {hook_reason}" if reason else hook_reason
                asked = replace(
                    call,
                    kind=kind,
                    reason=reason,
                    remember="" if dangerous else self._remember_label(kind, verdict),
                    history=self._history_hint(verdict),
                    purpose=assessment.purpose,
                    review=review_note,
                )
                self._notify(
                    "permission_prompt",
                    f"gigacrew ждёт подтверждения: {name} {call.summary}".strip(),
                )
                answer = _confirm_result(
                    answer=self._events.confirm(asked, preview, dangerous=dangerous),
                )
                if answer.approval is Approval.ALWAYS and (hook_asks or dangerous):
                    # «Всегда» не запомнить — значит, «да» один раз: хук спрашивает каждый
                    # раз, опасное не запоминается (интерфейс «всегда» для него не предлагает).
                    answer = replace(answer, approval=Approval.YES)
                if verdict is not None:
                    journal = self._journal_start(
                        call,
                        params,
                        verdict,
                        risk,
                        _decision_name(answer),
                        answer.feedback,
                        source="user",
                        review=review,
                    )
                if answer.approval is Approval.ALWAYS:
                    self._allow_always(asked, key, verdict)
                elif answer.approval is not Approval.YES:
                    model_text = REJECTED_NOTE
                    if answer.feedback:
                        model_text += FEEDBACK_NOTE.format(feedback=answer.feedback)
                    self._fail(call, "отклонено пользователем", model_text=model_text, note=note)
                    return ToolOutcome(ok=False, rejected=True)
                previewed = preview is not None
            else:
                journal = self._journal_start(
                    call,
                    params,
                    verdict,
                    risk,
                    "auto",
                    "",
                    source=source,
                    review=review,
                )
        except ToolError as exc:
            return self._fail(call, str(exc), note=note)
        except Exception as exc:  # неожиданная ошибка — не роняем агента
            return self._fail(call, f"внутренняя ошибка при подготовке: {exc}", note=note)

        try:
            moved = self._own_plan(tool, params)
        except Exception:  # путь вне корня и т. п. — ошибку сообщит сам инструмент
            moved = None
        if moved is not None:
            return self._fail(call, "файл плана пишет другая сессия", model_text=moved, note=note)

        # 3) Выполнение
        tool_result: ToolResult | None = None
        error = ""
        plan_writes = 0 if self._plan is None else self._plan.writes
        started = time.monotonic()
        journaled = False
        self._events.on_tool_start(call)
        # Обработчик прерывания должен знать, на каком шаге оно пришло: шаги — в одном try.
        try:  # noqa: WPS229
            # Ошибка инструмента — результат для модели, прерывание — наружу.
            try:  # noqa: WPS505
                tool_result = tool.run(params, self._ctx)
            except ToolError as exc:
                error = str(exc)
            except Exception as exc:  # неожиданная ошибка — не роняем агента
                error = f"внутренняя ошибка выполнения: {exc}"
            finally:
                self._events.on_tool_end(call)
            self._journal_end(
                journal,
                tool_result=tool_result,
                error=error,
                duration=time.monotonic() - started,
            )
            journaled = True
            if tool_result is None:
                return self._fail(call, error or "инструмент не вернул результат", note=note)

            if tool_result.ok and risk >= RiskLevel.WRITE:
                tool_result.changed = True  # UI и журнал суб-агента видят изменение
            self._events.on_tool_result(call, tool_result, previewed=previewed)
            feedback = self._post_tool_hooks(name, hook_fields, tool_result)
        except BaseException:
            # Ctrl+C/Esc во время работы инструмента — в журнале «прервано». Пришёл уже
            # после неё (показ результата, хук PostToolUse) — действие совершено: модель
            # должна знать настоящий результат, а не «прервано, результата нет».
            if not journaled:
                self._journal_end(
                    journal,
                    tool_result=tool_result,
                    interrupted=tool_result is None,
                    duration=time.monotonic() - started,
                )
            if tool_result is not None:
                self._conversation.add_function_result(
                    name,
                    _tool_text(tool_result.as_function_content(), note),
                )
            raise
        self._conversation.add_function_result(
            name,
            _tool_text(tool_result.as_function_content(), _join_notes(feedback, note)),
        )
        if tool_result.ok:
            self._attach_instructions(tool, params)
        # Правка плана — изменение для ограничителя: показ исправленного плана — не повтор.
        replanned = self._plan is not None and self._plan.writes != plan_writes
        return ToolOutcome(
            ok=tool_result.ok,
            changed=tool_result.changed or replanned,
            soft=tool_result.soft,
        )

    def _post_tool_hooks(
        self,
        name: str,
        fields: dict[str, Any],
        tool_result: ToolResult,
    ) -> str:
        """Хуки PostToolUse: их отзыв для модели (дописывается к результату) или ""."""
        hook = self._hook(
            HookEvent.POST_TOOL_USE,
            name,
            **fields,
            tool_response={"ok": tool_result.ok, "content": tool_result.content},
        )
        parts = list(hook.context)
        if hook.blocked:
            parts.insert(0, hook.reason)
            self._events.on_notice(
                f"хук PostToolUse ({name}): {_first_line(hook.reason)}",
                level=_WARN,
            )
        if not parts:
            return ""
        text = "\n\n".join(parts)
        return _reminder(HOOK_FEEDBACK_NOTE.format(event=HookEvent.POST_TOOL_USE, text=text))

    def _run_subagent(self, name: str, description: str, prompt: str) -> ToolResult:
        """Запустить суб-агента ``name`` с задачей ``prompt`` (инструмент ``task``).

        Суб-агент работает в своей истории; основному агенту возвращается отчёт. Esc
        во время его работы — подвести итог, повторный Esc — оборвать (ход основного
        агента продолжается в обоих случаях); Ctrl+C останавливает весь ход.
        """
        spec = SUBAGENTS.get(name)
        if spec is None:
            available = ", ".join(SUBAGENTS)
            raise ToolError(f"неизвестный суб-агент {name!r}; доступны: {available}.")
        if self._subagent_runs >= MAX_SUBAGENTS_PER_TURN:
            raise ToolError(
                f"лимит суб-агентов в этом ходе исчерпан ({MAX_SUBAGENTS_PER_TURN}) — "
                "продолжай сам.",
            )
        self._subagent_runs += 1
        subagent_info = SubagentInfo(spec.name, description)
        events = SubagentEvents(self._events, subagent_info)
        child = Agent(
            self._provider,
            subagent_registry(self._registry, spec),
            # Без сжатия: предел суб-агента — заполнение контекста, у него он подводит итог.
            replace(self._cfg, max_steps=self._cfg.subagent_max_steps, auto_compact=False),
            events,
            workspace=self._workspace,
            windows=self._windows,
            parent=self,
            subagent=spec,
        )
        self._events.on_subagent_start(subagent_info)
        with self._stop_lock:
            self._child = child
        try:
            text = child.run_turn(prompt)
        except LLMError as exc:
            # Сделанное до сбоя не теряется: основной агент не повторит его вслепую.
            return model_error_report(spec, exc, events.trail)
        except KeyboardInterrupt:
            if child._stop_level >= 2:  # повторный Esc: оборвать суб-агента, ход идёт дальше
                return aborted_report(spec, events.trail)
            # Ctrl+C: ход останавливается, но модель узнает, что успел сделать суб-агент.
            # Ответ — тому вызову (task или explore), что его запустил: он первый без ответа.
            pending = self._conversation.pending_calls()
            self._conversation.add_function_result(
                pending[0].name if pending else "task",
                interrupted_note(spec, events.trail),
            )
            raise
        finally:
            with self._stop_lock:
                self._child = None
            self._events.on_subagent_end(subagent_info, child.last_turn)
        return subagent_report(spec, text, child.last_turn, events.trail)

    def _allow_always(self, call: ToolCallInfo, key: str, verdict: Verdict | None = None) -> None:
        """«Да, и не спрашивать»: команды — сохранённые разрешения, прочее — до конца сессии.

        Для команды сохраняются разрешения на её части, которые спрашивались; любой
        другой вызов с этими аргументами разрешается до конца сессии.
        """
        if verdict is not None:
            try:
                added = self._decisions.approve(verdict)
            except OSError as exc:
                added = []
                self._events.on_notice(f"не удалось сохранить разрешение: {exc}", level=_WARN)
            if added:
                where = (
                    "сохранено для проекта"
                    if self._decisions.directory is not None and self._decisions.remember
                    else "до конца сессии"
                )
                rules = "; ".join(rule.describe() for rule in added)
                self._events.on_notice(f"разрешено без вопросов ({where}): {rules}")
                return
        self._session_allowed.add(key)
        self._events.on_notice(
            f"{call.name} с этими аргументами больше не требует подтверждения до конца сессии",
        )

    def _review_command(
        self,
        call: ToolCallInfo,
        params: BaseModel,
        verdict: Verdict | None,
        check: CallCheck,
    ) -> tuple[Decision, Review | None, str]:
        """Режим «авто»: спросить модель-рецензента о команде.

        ALLOW — выполнить без вопроса; ASK — решает пользователь, третье значение —
        пояснение для вопроса. Рецензент не спрашивается, если пользователь уже отклонял
        такую команду, команда слишком длинная или рецензент отключён. Сбой рецензента —
        вопрос пользователю; сбои подряд отключают рецензента. Ctrl+C/Esc прерывают ход.
        """
        reviewer = self._reviewer
        command = str(getattr(params, "command", ""))
        if verdict is None or check.rejected_before:
            return Decision.ASK, None, ""  # повтор отклонённого — ответит ветка вопроса
        if not reviewer.enabled:
            return Decision.ASK, None, f"рецензент отключён: {reviewer.disabled}"
        key = history_key(verdict)
        allowed, rejected = self._decisions.history(key)
        if self._decisions.rejected_in_session(key) or rejected > allowed:
            return Decision.ASK, None, "рецензент не спрашивался: такую команду уже отклоняли"
        if len(command) > MAX_REVIEW_COMMAND:
            return (
                Decision.ASK,
                None,
                f"рецензент не спрашивался: команда длиннее {MAX_REVIEW_COMMAND} символов",
            )
        model = reviewer.model_for(self._model_name)
        review: Review | None = None
        self._events.on_review_start(call)
        try:
            review = review_command(
                self._provider,
                command=command,
                verdict=verdict,
                requests=self._user_requests(),
                model=model,
                timeout=self._cfg.reviewer_timeout,
                on_usage=self._bill_side,
            )
        except Exception as exc:  # сбой рецензента — решает пользователь
            failure = str(exc) or type(exc).__name__
            error = " ".join(failure.split())[:_MAX_ERROR_CHARS]
            review = Review(model, allowed=False, reason=error, failed=True)
        finally:
            self._events.on_review_end(call, review)
        if review.failed:
            if reviewer.failed(review.reason):
                self._events.on_notice(reviewer.warning, level=_WARN)
            return Decision.ASK, review, f"рецензент не ответил: {review.reason}"
        reviewer.succeeded()
        if review.allowed:
            return Decision.ALLOW, review, ""
        reason = review.reason or "решать вам"
        return Decision.ASK, review, f"рецензент ({model}): {reason}"

    def _user_requests(self) -> list[str]:
        """Последние запросы пользователя — из диалога основного агента.

        У суб-агента первое сообщение — поручение модели, а не слова пользователя.
        """
        root = self
        while root._parent is not None:
            root = root._parent
        return recent_requests(root.conversation.messages)

    def _bill_side(self, usage: Usage) -> None:
        """Расход побочного запроса (рецензент): в сессию и статистику хода, не в контекст."""
        self._bill(usage)
        if not self._is_subagent and self._turn_stats is not None:
            _add_usage(self._turn_stats, usage)

    def _remember_label(self, kind: ToolKind, verdict: Verdict | None) -> str:
        """Что запомнит ответ «всегда» (пусто — формулировка интерфейса по умолчанию)."""
        if kind is ToolKind.EDIT or verdict is None:
            return ""
        return self._decisions.offer(verdict)

    def _history_hint(self, verdict: Verdict | None) -> str:
        if verdict is None:
            return ""
        allowed, rejected = self._decisions.history(history_key(verdict))
        if not allowed and not rejected:
            return ""
        parts: list[str] = []
        if allowed:
            parts.append(f"разрешали {allowed}")
        if rejected:
            parts.append(f"отклоняли {rejected}")
        counts = ", ".join(parts)
        hint = f"раньше: {counts}"
        if allowed >= 2 and not rejected:
            hint = f"{hint} — «a» запомнит выбор"
        return hint

    # Запись журнала собирается из всех сведений о решении: каждое — отдельный аргумент.
    def _journal_start(  # noqa: WPS211
        self,
        call: ToolCallInfo,
        params: BaseModel,
        verdict: Verdict | None,
        risk: RiskLevel,
        decision: str,
        feedback: str,
        *,
        source: str = "",
        review: Review | None = None,
    ) -> _Journal | None:
        """Записать решение о команде в журнал (что, где, почему, кто решил).

        Не команда (``verdict`` None) — в журнал не пишется.
        """
        if verdict is None:
            return None
        extra: dict[str, object] = {}
        if review is not None:
            extra["review"] = {
                "model": review.model,
                "allowed": review.allowed,
                "reason": review.reason,
                "failed": review.failed,
            }
        try:
            entry_id = self._decisions.record({
                "tool": call.name,
                "command": getattr(params, "command", call.summary),
                "description": getattr(params, "description", ""),
                "cwd": verdict.cwd,
                "key": history_key(verdict)[1],
                "verdict": {
                    "level": None if verdict.level is None else verdict.level.name,
                    "reason": verdict.reason,
                    "signatures": list(verdict.signatures),
                },
                "risk": risk.name,
                "mode": self.mode.value,
                "agent": "main" if self._subagent is None else self._subagent.name,
                "chat": self._session_chat_id,
                "model": self._model_name,
                "decision": decision,
                "source": source or "auto",
                "feedback": feedback,
                **extra,
            })
        except Exception:  # журнал — не повод прерывать работу
            return None
        return _Journal(entry_id)

    def _journal_end(
        self,
        journal: _Journal | None,
        *,
        tool_result: ToolResult | None = None,
        error: str = "",
        duration: float = _NO_DURATION,
        interrupted: bool = False,
        blocked: bool = False,
    ) -> None:
        if journal is None:
            return
        fields: dict[str, object] = {"duration_s": round(duration, 2)}
        if blocked:
            fields["blocked"] = True
        elif interrupted:
            fields["interrupted"] = True
        elif tool_result is None:
            fields.update(ok=False, error=error[:_MAX_ERROR_CHARS])
        else:
            fields.update(
                ok=tool_result.ok,
                soft=tool_result.soft,
                exit_code=tool_result.exit_code,
            )
        with contextlib.suppress(Exception):
            self._decisions.outcome(journal.id, **fields)

    def _attach_instructions(self, tool: Tool[Any], params: BaseModel) -> None:
        """Подключить инструкции подкаталогов, которых коснулся вызов.

        Они уходят в системном сообщении следующих запросов — поэтому переживают
        обрезку и сжатие истории и не дублируются.
        """
        try:
            found = self._nested.add_paths(tool.paths(params))
        except Exception:  # инструкции не повод прерывать работу
            return
        if found:
            names = ", ".join(doc.name for doc in found)
            self._events.on_notice(f"подключены инструкции подкаталога: {names}")

    def _fail(
        self,
        call: ToolCallInfo,
        error: str,
        *,
        model_text: str | None = None,
        note: str | None = None,
    ) -> ToolOutcome:
        """Неудачный вызов: показать пользователю и сообщить модели."""
        summary = error.splitlines()[0] if error else "ошибка"
        failure = ToolResult(content=error, ok=False, summary=summary)
        self._events.on_tool_result(call, failure, previewed=False)
        content = model_text or f"ОШИБКА: {error}"
        self._conversation.add_function_result(call.name, _tool_text(content, note))
        return ToolOutcome(ok=False)
