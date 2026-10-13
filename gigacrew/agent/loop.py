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

import threading
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from pydantic import BaseModel

from gigacrew.agent.chat_store import ChatStore
from gigacrew.agent.compaction import CompactionError, plan_compaction, summarize
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
from gigacrew.agent.conversation import Conversation, Summary
from gigacrew.agent.events import (
    AgentEvents,
    Approval,
    CompactResult,
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
from gigacrew.agent.prompts import (
    SYSTEM_PROMPT,
    build_project_context,
    compose_system_prompt,
    mode_prompt,
    nested_instructions_prompt,
    subagent_system_prompt,
    summary_prompt,
)
from gigacrew.agent.subagents import (
    MAX_SUBAGENTS_PER_TURN,
    READ_ONLY_BLOCKED_NOTE,
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
from gigacrew.permissions import Decision, PermissionMode, ToolKind, decide, next_mode
from gigacrew.project.instructions import NestedInstructions
from gigacrew.project.workspace import Workspace
from gigacrew.security import RiskLevel
from gigacrew.tools.base import PlanFile, Tool, ToolContext, ToolError, ToolRegistry, ToolResult
from gigacrew.tools.task import SUBAGENTS, SubagentSpec

# Заметки для модели — на английском, как и промпты (пользователь их не видит).
REJECTED_NOTE = (
    "The user REJECTED this operation. Do not repeat it; suggest an alternative or "
    "clarify the plan."
)
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
    "describe the needed changes in the plan file and hand it to the user with "
    "exit_plan_mode."
)
PLAN_NOT_SUBMITTED_NOTE = (
    "# Plan not submitted\nYou changed the plan file but ended the turn without handing "
    "it to the user. Call exit_plan_mode now (without arguments); do not repeat the plan "
    "as text. If you need the user's answer first, ask with ask_user."
)
OUTPUT_CUT_NOTE = (
    "# Reply cut off\nYour previous reply was cut off before it finished (finish_reason: "
    "{reason}), and any tool call in it was lost. Do it again in smaller pieces: write a "
    "long file as an outline with write_file and add the sections with edit_file; keep a "
    "text answer shorter."
)
OUTPUT_LIMIT_MESSAGE = (
    "Прервано: ответ модели {count} раза подряд оборвался, не дойдя до конца "
    "(finish_reason={reason}). Задача может быть не завершена."
)

# Ответ без вызовов с такой причиной завершения оборван, а не закончен моделью: вызов в нём
# мог потеряться. Он не считается финальным — запрос повторяется (не больше MAX_CUT_RETRIES
# раз за ход).
CUT_REASONS = frozenset({"length", "incomplete"})
MAX_CUT_RETRIES = 2


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


def _with_note(content: str, note: str | None) -> str:
    return f"{content}\n\n{note}" if note else content


def _reminder(note: str) -> str:
    """Заметка harness'а для одного запроса — в теге, который описан в системном промпте."""
    return f"<system-reminder>\n{note}\n</system-reminder>"


class Agent:
    def __init__(
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
    ):
        """Конструктор не обращается к файловой системе и сети.

        ``windows`` — замеренные окна моделей (по умолчанию пусто: окно
        :data:`~gigacrew.agent.context_window.DEFAULT_CONTEXT_WINDOW`).

        ``parent``/``subagent`` — это суб-агент: режим разрешений, разрешения «всегда»,
        калибровка токенов, инструкции подкаталогов и контекст проекта — общие с
        основным агентом, расход токенов учитывается и у него; запускать суб-агентов
        и задавать вопросы пользователю суб-агент не может.

        ``chat_id`` — id чата: по нему назван файл плана (по умолчанию — новый id).
        """
        self._provider = provider
        self._registry = registry
        self._cfg = config
        # `is None`, а не `or`: пустой Conversation ложен (__len__ == 0).
        self._events = AgentEvents() if events is None else events
        self._workspace = Workspace(config.project_root) if workspace is None else workspace
        self._parent = parent
        self._subagent = subagent
        # Файл плана чата — только у основного агента: суб-агент план не пишет.
        self._chat_id = chat_id or ChatStore.new_id()
        self._plan = None if parent is not None else PlanFile(self._plan_path(self._chat_id))
        self._ctx = ToolContext(
            workspace=self._workspace,
            ask_user=self._events.ask_user,
            get_mode=lambda: self.mode,
            set_mode=self.set_mode,
            semantic=config.ty,
            run_subagent=None if parent is not None else self._run_subagent,
            plan=self._plan,
        )
        self._conversation = Conversation() if conversation is None else conversation
        self._model = config.model if parent is None else parent.model
        # Режим разрешений — состояние сессии, как модель: reset() его не меняет.
        # Может смениться посреди хода (Shift+Tab из потока клавиш) — читается при
        # каждом вызове инструмента и каждом обращении к модели.
        self._mode = config.mode
        self._windows = ModelWindows() if windows is None else windows
        # Контекст проекта (дерево, инструкции, карта): строится лениво, сбрасывается в
        # reset(); суб-агент берёт контекст основного агента.
        self._project_context: str | None = None
        # Инструкции подкаталогов, с которыми агент работал в этом диалоге.
        self._nested = (
            NestedInstructions(self._workspace.root) if parent is None else parent._nested
        )
        self._billed_tokens = 0  # потрачено за сессию (reset() не сбрасывает)
        # Размер контекста последнего запроса (реальные токены; None — берётся из
        # last_usage диалога) и модель, в чьих токенах он и last_usage посчитаны.
        self._context: int | None = None
        self._context_model = self._model
        # Оценка контекста, пока настоящий размер неизвестен: (состояние, оценка).
        self._context_estimate: tuple[tuple[object, ...], int] | None = None
        self._last_turn: TurnStats | None = None
        # Реальные токены / оценка по модели: оценка ~3 символа на токен грубая, а
        # бюджеты считаются в её единицах (калибруется по prompt_tokens ответов).
        self._token_scale: dict[str, float] = {} if parent is None else parent._token_scale
        # Вызовы, которые пользователь разрешил «всегда» (до конца сессии; reset() не
        # сбрасывает — как и режим).
        self._session_allowed: set[str] = set() if parent is None else parent._session_allowed
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
        return self._conversation

    @property
    def workspace(self) -> Workspace:
        return self._workspace

    @property
    def model(self) -> str:
        return self._model

    @property
    def config(self) -> Config:
        return self._cfg

    @property
    def mode(self) -> PermissionMode:
        return self._mode if self._parent is None else self._parent.mode

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
        if self._context_model == self._model:
            if self._context is not None:
                return self._context
            usage = conversation.last_usage
            if usage is not None:  # продолженный чат
                request = min(self._context_estimate_units(), self._estimate_budget())
                prompt = self._request_tokens(usage, request)
                if prompt is not None:
                    return prompt + usage.completion_tokens
        return self._to_real(self._context_estimate_units())

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
            self._model,
            self.mode,
            len(self._nested.files),
        )
        if self._context_estimate is None or self._context_estimate[0] != key:
            specs = self._registry.specs()
            size = self._overhead_tokens(specs) + self._history_tokens(conversation)
            self._context_estimate = (key, size)
        return self._context_estimate[1]

    @property
    def provider(self) -> LLMProvider:
        return self._provider

    @property
    def windows(self) -> ModelWindows:
        return self._windows

    @property
    def context_window(self) -> int | None:
        """Замеренное окно текущей модели (None — не замерено)."""
        return self._windows.get(self._model)

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
        return self._token_scale.get(self._model, 1.0)

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

    @staticmethod
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

    def _calibrate(self, prompt_tokens: int) -> None:
        """Уточнить калибровку по фактическому размеру отправленного запроса.

        Рост принимается сразу (осторожность), снижение — наполовину: один запрос с
        необычным текстом не должен резко раздвинуть бюджет.
        """
        if prompt_tokens <= 0 or self._last_estimate <= 0:
            return
        ratio = min(max(prompt_tokens / self._last_estimate, MIN_TOKEN_SCALE), MAX_TOKEN_SCALE)
        old = self._token_scale.get(self._model)
        if old is not None and ratio < old:
            ratio = (old + ratio) / 2
        self._token_scale[self._model] = ratio

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
        self._model = name

    @property
    def chat_id(self) -> str:
        """Id чата (сохранённого — тот же, что у файла чата): по нему назван файл плана."""
        return self._chat_id

    @chat_id.setter
    def chat_id(self, value: str) -> None:
        self._chat_id = value
        if self._plan is not None:
            self._plan.path = self._plan_path(value)
            self._plan.dirty = False

    def _plan_path(self, chat_id: str) -> Path:
        return self._workspace.plans_dir / f"{chat_id}.md"

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
        self._mode = PermissionMode(mode)

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
            child = self._child
            if child is None:
                return False
            child._stop_level = 2 if child._wrapping else child._stop_level + 1
            return True

    def reset(
        self, conversation: Conversation | None = None, *, chat_id: str | None = None
    ) -> None:
        """Начать новый диалог или продолжить сохранённый (``conversation``).

        ``chat_id`` — id чата (продолженного — его id: агент найдёт его план); по
        умолчанию — новый. Контекст проекта будет собран заново; потраченные токены
        сессии не сбрасываются.
        """
        self.chat_id = chat_id or ChatStore.new_id()
        self._conversation = Conversation() if conversation is None else conversation
        self._project_context = None
        self._nested.reset()
        # usage продолженного чата — в токенах его модели: она становится текущей до reset().
        self._context = None
        self._context_model = self._model

    # ------------------------------------------------------------------ #
    def run_turn(self, user_input: str) -> str:
        """Обрабатывает один запрос пользователя до финального ответа.

        При любом прерывании (Ctrl+C, ошибка LLM) история приводится в
        согласованное состояние, исключение пробрасывается дальше.
        """
        self._last_turn = None
        try:
            return self._run_turn(user_input)
        except BaseException:
            self._conversation.repair()
            raise

    def _run_turn(self, user_input: str) -> str:
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
        return self._parent is not None and self._stop_level == 1 and not self._wrapping

    def _user_stop(self, specs: list[ToolSpec], stats: TurnStats) -> str:
        """Суб-агент остановлен пользователем: итог по сделанному одним запросом.

        Повторная остановка во время итога (``KeyboardInterrupt``) пробрасывается —
        суб-агент обрывается без отчёта.
        """
        self._conversation.repair()
        stats.stop_reason = "user_stop"
        self._events.on_notice("остановлен пользователем — подвожу итог", level="warn")
        return self._wrap_up(
            specs, USER_STOP_REASON, MODEL_STOP_REASONS["user_stop"], stats, user_stop=True
        )

    def _loop(self, specs: list[ToolSpec], guard: LoopGuard, stats: TurnStats) -> str:
        """Шаги хода до финального ответа модели или остановки ограничителем."""
        compact_failed = False  # не удалось — до конца хода не пытаемся снова
        plan_reminded = False  # напоминание отправить план — не больше одного за ход
        cuts = 0  # оборванных ответов за ход
        note = ""
        while True:
            context = self._subagent_context(specs, stats)
            stop = guard.before_step(context)
            if stop is None:
                stats.steps = guard.steps
                if self._cfg.auto_compact and not compact_failed:
                    compact_failed = not self._auto_compact(specs, stats)
                notes = [guard.pressure(context) or "", note]
                turn = self._next_turn(specs, note="\n\n".join(n for n in notes if n))
                note = ""
                msg = turn.message
                self._account(turn.usage, stats)
                # Повторы ниже: ответ текстом не идёт в историю — запрос не должен кончаться
                # репликой ассистента, — а заметка дописывается к хвосту следующего запроса.
                if not turn.wants_tool and turn.finish_reason in CUT_REASONS:
                    self._forget_reply(turn, stats)
                    if cuts >= MAX_CUT_RETRIES:
                        message = OUTPUT_LIMIT_MESSAGE.format(
                            count=cuts + 1, reason=turn.finish_reason
                        )
                        return self._stopped(specs, StopReason("output_limit", message), stats)
                    cuts += 1
                    stats.cut_retries += 1
                    note = OUTPUT_CUT_NOTE.format(reason=turn.finish_reason)
                    self._events.on_notice(
                        f"ответ модели оборван (finish_reason={turn.finish_reason}) — "
                        f"повторяю запрос ({cuts}/{MAX_CUT_RETRIES})",
                        level="warn",
                    )
                    continue
                if not turn.wants_tool and not plan_reminded and self._plan_unsubmitted():
                    self._forget_reply(turn, stats)
                    plan_reminded = True
                    note = PLAN_NOT_SUBMITTED_NOTE
                    self._events.on_notice(
                        "план изменён, но не отправлен на одобрение — напоминаю модели",
                        level="warn",
                    )
                    continue
                self._conversation.add_assistant(msg, turn.usage)

                if not turn.wants_tool:
                    return msg.content
                stop = self._run_calls(msg.function_calls, guard, stats)
            if stop is not None:
                return self._stopped(specs, stop, stats)

    def _plan_unsubmitted(self) -> bool:
        """Режим плана, а план менялся после того, как его показали пользователю."""
        return (
            self._plan is not None
            and self._plan.dirty
            and self._parent is None
            and self.mode is PermissionMode.PLAN
        )

    def _run_calls(
        self, calls: list[FunctionCall], guard: LoopGuard, stats: TurnStats
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
            if check.warning:
                self._events.on_notice(
                    f"Модель {check.repeats}-й раз повторяет вызов {call.name} "
                    "без изменений — предупреждена; следующий повтор остановит ход.",
                    level="warn",
                )
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
        estimate = self._overhead_tokens(specs) + self._history_tokens(self._conversation)
        return max(self._to_real(estimate), stats.context_tokens)

    def _stopped(self, specs: list[ToolSpec], stop: StopReason, stats: TurnStats) -> str:
        stats.stop_reason = stop.kind
        self._events.on_notice(stop.message, level="error")
        text = self._wrap_up(specs, stop.message, MODEL_STOP_REASONS[stop.kind], stats)
        # Суб-агенту без итога — пустой ответ: отчёт сам назовёт причину остановки.
        return text or ("" if self._parent is not None else stop.message)

    def _forget_reply(self, turn: AssistantTurn, stats: TurnStats) -> None:
        """Ответ не идёт в историю: его вывод не занимает контекст следующего запроса.

        Иначе оборванный длинный ответ раздул бы счётчик, и суб-агент ложно упёрся бы в
        предел контекста.
        """
        stats.context_tokens = max(stats.context_tokens - turn.usage.completion_tokens, 0)
        if self._context_model == self._model:
            self._context = stats.context_tokens

    def _account(self, usage: Usage, stats: TurnStats) -> None:
        """Учесть расход обращения: статистика хода, сессия, размер контекста, калибровка.

        Неправдоподобный размер запроса (:meth:`_request_tokens`) заменяется оценкой
        и калибровку не сдвигает.
        """
        stats.prompt_tokens += usage.prompt_tokens
        stats.completion_tokens += usage.completion_tokens
        self._bill(usage)
        prompt = self._request_tokens(usage, self._last_estimate)
        if prompt is None:
            prompt = self._to_real(self._last_estimate)
        else:
            self._calibrate(prompt)
        stats.context_tokens = prompt + usage.completion_tokens
        self._context, self._context_model = stats.context_tokens, self._model

    def _bill(self, usage: Usage) -> None:
        """Оплаченные токены сессии; у суб-агента — ещё и у основного агента."""
        self._billed_tokens += usage.prompt_tokens + usage.completion_tokens
        if self._parent is not None:
            self._parent._bill_child(usage)

    def _bill_child(self, usage: Usage) -> None:
        """Расход суб-агента: в сессию и в статистику идущего хода (не в контекст)."""
        self._billed_tokens += usage.prompt_tokens + usage.completion_tokens
        if self._turn_stats is not None:
            self._turn_stats.prompt_tokens += usage.prompt_tokens
            self._turn_stats.completion_tokens += usage.completion_tokens

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
        elif self._parent is not None:
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
                Message(role="assistant", content=turn.message.content), turn.usage
            )
        return text

    # ------------------------------------------------------------------ #
    def project_context(self) -> str:
        """Контекст проекта для системного промпта (строится один раз за диалог)."""
        if self._parent is not None:
            return self._parent.project_context()
        if self._project_context is None:
            self._project_context = build_project_context(self._workspace)
        return self._project_context

    def system_prompt(self) -> str:
        if self._subagent is not None:
            return subagent_system_prompt(self._subagent.name, self.project_context())
        return compose_system_prompt(SYSTEM_PROMPT, self.project_context())

    def _system_text(self) -> str:
        """Системное сообщение без краткого содержания: промпт, инструкции
        подкаталогов, правила режима."""
        parts = [
            self.system_prompt(),
            nested_instructions_prompt(self._nested.files),
            mode_prompt(
                self.mode, subagent=self._parent is not None, plan_path=self._plan_display()
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

    @staticmethod
    def _history_tokens(conversation: Conversation) -> int:
        """История, которую видит модель: краткое содержание + сообщения после него."""
        size = estimate_tokens(conversation.context_messages())
        if conversation.summary is not None:
            size += estimate_text_tokens(summary_prompt(conversation.summary.text))
        return size

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
        # Схемы инструментов и заметка уходят в запросе и занимают то же окно.
        used = (
            estimate_tokens([Message(role="system", content=system_text)])
            + estimate_specs_tokens(specs)
            + (estimate_text_tokens(reminder) if reminder else 0)
        )
        budget = max(self._estimate_budget() - used, MIN_HISTORY_TOKENS)
        history = fit_history(self._conversation.context_messages(), budget)
        self._last_estimate = used + estimate_tokens(history)
        if reminder:
            if history and history[-1].role in ("user", "function"):
                last = history[-1]
                history[-1] = last.model_copy(
                    update={"content": _with_note(last.content, reminder)}
                )
            else:
                system_text = f"{system_text}\n\n{reminder}"
        return [Message(role="system", content=system_text), *history]

    # ------------------------------------------------------------------ #
    def compact(self, instructions: str = "") -> CompactResult | None:
        """Сжать весь диалог в краткое содержание (команда ``/compact``).

        ``instructions`` — пожелания пользователя к резюме. None — сжимать нечего.
        Ошибка модели — :class:`~gigacrew.llm.base.LLMError`; диалог тогда не меняется.
        """
        specs = self._registry.specs()
        return self._compact(self._history_budget(specs), specs, instructions=instructions)

    def _auto_compact(self, specs: list[ToolSpec], stats: TurnStats) -> bool:
        """Сжать историю, если она подошла к порогу. False — сжать не удалось."""
        budget = self._history_budget(specs)
        if self._history_tokens(self._conversation) < self._cfg.compact_threshold * budget:
            return True

        def count(usage: Usage) -> None:
            stats.prompt_tokens += usage.prompt_tokens
            stats.completion_tokens += usage.completion_tokens

        try:
            self._compact(budget, specs, auto=True, on_usage=count)
        except LLMError as e:
            self._events.on_notice(
                f"не удалось сжать контекст: {e} — старые сообщения будут отбрасываться",
                level="warn",
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
            plan = plan_compaction(
                conversation,
                keep_tokens=int(budget * COMPACT_KEEP_SHARE),
                min_tokens=int(budget * COMPACT_MIN_SHARE),
            )
        else:
            plan = plan_compaction(conversation, keep_tokens=0)
        if plan is None:
            return None

        def count(usage: Usage) -> None:
            self._bill(usage)
            if on_usage is not None:
                on_usage(usage)

        overhead = self._overhead_tokens(specs)
        before = self._history_tokens(conversation)
        max_chars = int(budget * SUMMARY_SHARE) * CHARS_PER_TOKEN
        result: CompactResult | None = None
        self._events.on_compact_start(auto=auto)
        try:
            previous = conversation.summary.text if conversation.summary is not None else None
            text = summarize(
                self._provider,
                plan.messages,
                model=self._model,
                temperature=self._cfg.temperature,
                input_budget=self._estimate_budget(),
                max_chars=min(max(max_chars, MIN_SUMMARY_CHARS), MAX_SUMMARY_CHARS),
                previous=previous,
                instructions=instructions,
                on_usage=count,
            )
            summary = Summary(text=text, upto=plan.upto)
            after = self._history_tokens(Conversation(conversation.messages, summary=summary))
            if after >= before:
                raise CompactionError("краткое содержание вышло не короче самой истории")
            conversation.set_summary(summary)
            self._context, self._context_model = self._to_real(overhead + after), self._model
            result = CompactResult(
                before_tokens=self._to_real(overhead + before),
                after_tokens=self._context,
                messages=len(plan.messages),
                auto=auto,
            )
        finally:
            self._events.on_compact_end(result)
        return result

    def _next_turn(
        self, specs: list[ToolSpec], note: str = "", *, tool_choice: ToolChoice = "auto"
    ) -> AssistantTurn:
        """Один проход модели с выводом текста (потоковым или цельным).

        ``note`` — заметка только для этого запроса (см. :meth:`_build_request`);
        ``tool_choice="none"`` — ответ только текстом (итог после остановки).
        Оборванный поток (Esc, сбой сети) оставляет в истории основного агента уже
        показанный текст с пометкой — модель знает, что её ответ прерван, а следующий
        запрос не идёт двумя репликами пользователя подряд. У суб-агента текст не
        показывается и не сохраняется.
        """
        messages = self._build_request(specs, note)
        streamed: list[str] = []

        def on_delta(text: str) -> None:
            streamed.append(text)
            self._events.on_stream_delta(text)

        self._events.on_stream_start()
        try:
            if self._cfg.stream:
                return self._provider.stream(
                    messages,
                    tools=specs,
                    model=self._model,
                    temperature=self._cfg.temperature,
                    tool_choice=tool_choice,
                    on_delta=on_delta,
                )
            turn = self._provider.complete(
                messages,
                tools=specs,
                model=self._model,
                temperature=self._cfg.temperature,
                tool_choice=tool_choice,
            )
        except BaseException as e:
            partial = "".join(streamed).rstrip()
            # Суб-агенту оборванный текст не нужен: пользователь его не видел, а запрос
            # итога после остановки (Esc) не должен кончаться репликой ассистента.
            if partial.strip() and self._parent is None:
                tail = (
                    INTERRUPTED_ANSWER_NOTE
                    if isinstance(e, KeyboardInterrupt)
                    else (BROKEN_ANSWER_NOTE)
                )
                self._conversation.add_assistant(Message(role="assistant", content=partial + tail))
            raise
        finally:
            self._events.on_stream_end()

        if turn.message.content.strip():
            self._events.on_assistant_text(turn.message.content)
        return turn

    # ------------------------------------------------------------------ #
    def _execute_tool_call(self, function_call: FunctionCall, check: CallCheck) -> ToolOutcome:
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
        except Exception as e:  # ошибка схемы — возвращаем модели
            invalid = ToolCallInfo(name, "(неверные аргументы)")
            self._events.on_tool_call(invalid)
            return self._fail(invalid, f"валидация аргументов: {e}", note=note)

        call = ToolCallInfo(name, self._describe(tool, params), tool.kind)
        self._events.on_tool_call(call)
        key = call_key(function_call)

        # 2) Разрешение по режиму: выполнить, спросить или заблокировать
        previewed = False
        risk = RiskLevel.SAFE
        try:
            risk = tool.risk(params, self._ctx)
            read_only = self._subagent is not None and self._subagent.read_only
            decision = decide(
                self.mode,
                risk,
                tool.kind,
                auto_approve=self._cfg.auto_approve,
                yes_all=self._cfg.yes_all,
                read_only=read_only,
            )
            if decision is Decision.ASK and key in self._session_allowed:
                decision = Decision.ALLOW  # пользователь разрешил этот вызов «всегда»
            if decision is Decision.BLOCK:
                if read_only:
                    error, model_text = "заблокировано: агент только читает", READ_ONLY_BLOCKED_NOTE
                elif self._parent is not None:
                    error, model_text = (
                        "заблокировано: режим планирования",
                        (SUBAGENT_PLAN_BLOCKED_NOTE),
                    )
                else:
                    error, model_text = "заблокировано: режим планирования", PLAN_BLOCKED_NOTE
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
                answer = self._events.confirm(call, preview, dangerous=dangerous)
                if isinstance(answer, bool):
                    answer = Approval.YES if answer else Approval.NO
                if answer is Approval.ALWAYS and not dangerous:
                    self._allow_always(call, key)
                elif answer is not Approval.YES:
                    self._fail(call, "отклонено пользователем", model_text=REJECTED_NOTE, note=note)
                    return ToolOutcome(ok=False, rejected=True)
                previewed = preview is not None
        except ToolError as e:
            return self._fail(call, str(e), note=note)
        except Exception as e:  # неожиданная ошибка — не роняем агента
            return self._fail(call, f"внутренняя ошибка при подготовке: {e}", note=note)

        # 3) Выполнение
        result: ToolResult | None = None
        error = ""
        self._events.on_tool_start(call)
        try:
            result = tool.run(params, self._ctx)
        except ToolError as e:
            error = str(e)
        except Exception as e:  # неожиданная ошибка — не роняем агента
            error = f"внутренняя ошибка выполнения: {e}"
        finally:
            self._events.on_tool_end(call)
        if result is None:
            return self._fail(call, error or "инструмент не вернул результат", note=note)

        if result.ok and risk >= RiskLevel.WRITE:
            result.changed = True  # UI и журнал суб-агента видят изменение
        self._events.on_tool_result(call, result, previewed=previewed)
        self._conversation.add_function_result(name, _with_note(result.as_function_content(), note))
        if result.ok:
            self._attach_instructions(tool, params)
        return ToolOutcome(ok=result.ok, changed=result.changed, soft=result.soft)

    def _run_subagent(self, name: str, description: str, prompt: str) -> ToolResult:
        """Запустить суб-агента ``name`` с задачей ``prompt`` (инструмент ``task``).

        Суб-агент работает в своей истории; основному агенту возвращается отчёт. Esc
        во время его работы — подвести итог, повторный Esc — оборвать (ход основного
        агента продолжается в обоих случаях); Ctrl+C останавливает весь ход.
        """
        spec = SUBAGENTS.get(name)
        if spec is None:
            raise ToolError(f"неизвестный суб-агент {name!r}; доступны: {', '.join(SUBAGENTS)}.")
        if self._subagent_runs >= MAX_SUBAGENTS_PER_TURN:
            raise ToolError(
                f"лимит суб-агентов в этом ходе исчерпан ({MAX_SUBAGENTS_PER_TURN}) — "
                "продолжай сам."
            )
        self._subagent_runs += 1
        info = SubagentInfo(spec.name, description)
        events = SubagentEvents(self._events, info)
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
        self._events.on_subagent_start(info)
        with self._stop_lock:
            self._child = child
        try:
            text = child.run_turn(prompt)
        except LLMError as e:
            # Сделанное до сбоя не теряется: основной агент не повторит его вслепую.
            return model_error_report(spec, e, events.trail)
        except KeyboardInterrupt:
            if child._stop_level >= 2:  # повторный Esc: оборвать суб-агента, ход идёт дальше
                return aborted_report(spec, events.trail)
            # Ctrl+C: ход останавливается, но модель узнает, что успел сделать суб-агент.
            self._conversation.add_function_result("task", interrupted_note(spec, events.trail))
            raise
        finally:
            with self._stop_lock:
                self._child = None
            self._events.on_subagent_end(info, child.last_turn)
        return subagent_report(spec, text, child.last_turn, events.trail)

    def _allow_always(self, call: ToolCallInfo, key: str) -> None:
        """«Да, и не спрашивать»: правки — режим авто-правок, прочее — этот вызов."""
        if call.kind is ToolKind.EDIT:
            if self.mode is PermissionMode.MANUAL:
                self.set_mode(PermissionMode.ACCEPT_EDITS)
                self._events.on_notice(
                    "режим «авто-правки»: дальнейшие правки файлов — без вопросов "
                    "(Shift+Tab — сменить)"
                )
            return
        self._session_allowed.add(key)
        self._events.on_notice(
            f"{call.name} с этими аргументами больше не требует подтверждения до конца сессии"
        )

    def _attach_instructions(self, tool: Tool, params: BaseModel) -> None:
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
        result = ToolResult(content=error, ok=False, summary=summary)
        self._events.on_tool_result(call, result, previewed=False)
        content = model_text or f"ОШИБКА: {error}"
        self._conversation.add_function_result(call.name, _with_note(content, note))
        return ToolOutcome(ok=False)

    @staticmethod
    def _describe(tool: Tool, params: BaseModel) -> str:
        try:
            return tool.describe(params)
        except Exception:
            return ""
