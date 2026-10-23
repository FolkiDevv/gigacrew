"""Хуки пользователя: команды, которые gigacrew запускает на событиях агента.

Команды хуков задаются в настройках (:mod:`gigacrew.project.settings`). Хук получает
JSON в stdin и отвечает кодом выхода и (необязательно) JSON в stdout — как в Claude Code:

* 0 — успех; stdout-объект разбирается как JSON (решение, контекст для модели),
  простой текст становится контекстом только у UserPromptSubmit и SessionStart;
* 2 — блокировка: действие не выполняется (или ход продолжается — у Stop), причина
  берётся из stderr; у PostToolUse это замечание модели, а у событий, где блокировать
  нечего (SessionStart, SessionEnd, PreCompact, Notification), — предупреждение;
* прочие коды, таймаут, ошибка запуска — предупреждение пользователю, работа идёт дальше.

Ядро не зависит от UI: предупреждения и индикатор долгих хуков идут через
:class:`~gigacrew.agent.events.AgentEvents`. Ctrl+C во время хука пробрасывается — группа
процесса хука убивается (:func:`~gigacrew.tools.process.run_process`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from gigacrew.agent.events import AgentEvents
from gigacrew.project.settings import HookCommand, HookEvent, HookSettings
from gigacrew.tools.process import (
    ProcessResult,
    quote_reminder_tags,
    run_process,
    truncate_middle,
)

BLOCK_EXIT_CODE = 2
# Сколько символов вывода хука уходит модели (контекст, причина блокировки).
HOOK_OUTPUT_LIMIT = 10_000
# Простой текст в stdout — контекст для модели только у этих событий: иначе вывод вроде
# «1 file reformatted» уходил бы модели после каждой правки.
CONTEXT_EVENTS = frozenset((HookEvent.USER_PROMPT_SUBMIT, HookEvent.SESSION_START))
# Здесь блокировка (код 2, decision "block", deny) действует — и первая останавливает
# остальные хуки события. У PostToolUse это замечание модели: идут все хуки (линтер не
# отменяет форматтер), замечания складываются. У остальных событий блокировать нечего:
# код 2 — предупреждение, как прочие коды, остальные хуки идут.
BLOCKING_EVENTS = frozenset((
    HookEvent.PRE_REQUEST,
    HookEvent.PRE_TOOL_USE,
    HookEvent.USER_PROMPT_SUBMIT,
    HookEvent.STOP,
    HookEvent.SUBAGENT_STOP,
))
# Эти хуки идут на каждом вызове инструмента — без индикатора, иначе он мигал бы.
QUIET_EVENTS = frozenset((
    HookEvent.PRE_TOOL_USE,
    HookEvent.POST_TOOL_USE,
    HookEvent.NOTIFICATION,
    HookEvent.PRE_REQUEST,
))
NO_REASON = "хук не пояснил причину"
# Сколько символов первой строки вывода хука попадает в предупреждение.
_FIRST_LINE_LIMIT = 200


@dataclass
class HookResult:
    """Итог всех хуков события."""

    blocked: bool = False  # код 2, decision "block" или permissionDecision "deny"
    reason: str = ""  # причина блокировки (для модели и пользователя)
    decision: Literal["allow", "ask"] | None = None  # PreToolUse: решение о подтверждении
    context: list[str] = field(default_factory=list)  # дополнительный контекст для модели

    @property
    def context_text(self) -> str:
        """Контекст от всех хуков одним текстом."""
        return "\n\n".join(self.context)


class PromptBlockedError(Exception):
    """Хук UserPromptSubmit отклонил запрос пользователя: ход не начинался."""

    def __init__(self, reason: str):
        """Отказ с причиной ``reason`` — её видит пользователь."""
        super().__init__(reason)
        self.reason = reason


class HookRunner:
    """Запуск хуков события и сведение их ответов в один :class:`HookResult`.

    Хуки события идут по очереди (порядок — как в настройках); первая блокировка
    останавливает остальные там, где она действует (:data:`BLOCKING_EVENTS`).
    ``deny`` важнее ``ask``, ``ask`` — важнее ``allow``; контексты и замечания
    PostToolUse склеиваются.
    """

    def __init__(self, settings: HookSettings | None = None):
        """Хуки из настроек ``settings`` (по умолчанию — пустых)."""
        self.settings = HookSettings() if settings is None else settings

    def has(self, event: HookEvent, target: str = "") -> bool:
        """Есть ли хуки события ``event`` для ``target``."""
        return self.settings.has(event, target)

    def run(
        self,
        event: HookEvent,
        payload: dict[str, Any],
        *,
        cwd: Path,
        events: AgentEvents,
        target: str = "",
    ) -> HookResult:
        """Запустить хуки события с данными ``payload`` в stdin и свести их ответы."""
        outcome = HookResult()
        commands = self.settings.commands(event, target)
        if not commands:
            return outcome
        stdin_text = json.dumps(payload, ensure_ascii=True, default=str)
        env = {"GIGACREW_PROJECT_DIR": str(cwd), "CLAUDE_PROJECT_DIR": str(cwd)}
        for hook in commands:
            proc = _run_hook(event, hook, stdin_text, cwd, env, events)
            if proc is not None:
                self._apply(event, hook, proc, outcome, events)
            if outcome.blocked and event in BLOCKING_EVENTS:
                break
        return outcome

    def _apply(
        self,
        event: HookEvent,
        hook: HookCommand,
        proc: ProcessResult,
        outcome: HookResult,
        events: AgentEvents,
    ) -> None:
        blockable = event in BLOCKING_EVENTS or event is HookEvent.POST_TOOL_USE
        if proc.returncode == BLOCK_EXIT_CODE and blockable:
            _block(event, hook, _clip(proc.stderr.strip()) or NO_REASON, outcome, events)
            return
        if proc.returncode != 0:
            detail = _first_line(proc.stderr) or _first_line(proc.stdout)
            suffix = f": {detail}" if detail else ""
            events.on_notice(
                f"хук {event} ({short_command(hook.command)}) завершился с кодом "
                f"{proc.returncode}{suffix}",
                level="warn",
            )
            return
        out = proc.stdout.strip()
        if not out:
            return
        answer = _json_object(out)
        if answer is None:
            if out.startswith("{"):
                _warn_unparsed(event, hook, outcome, events)
            if event in CONTEXT_EVENTS:
                outcome.context.append(_clip(out))
            return
        self._apply_json(event, hook, answer, outcome, events)

    def _apply_json(
        self,
        event: HookEvent,
        hook: HookCommand,
        answer: dict[str, Any],
        outcome: HookResult,
        events: AgentEvents,
    ) -> None:
        message = _text(answer.get("systemMessage"))
        if message:
            events.on_notice(f"хук {event}: {message}")
        specific = answer.get("hookSpecificOutput")
        if not isinstance(specific, dict):
            specific = {}
        decision = answer.get("decision")
        reason = _text(answer.get("reason"))
        if event is HookEvent.PRE_TOOL_USE:
            _apply_permission(event, hook, specific, decision, reason, outcome, events)
        elif decision == "block":
            _block(event, hook, _clip(reason) or NO_REASON, outcome, events)
        context = _text(specific.get("additionalContext"))
        if context:
            outcome.context.append(_clip(context))


def _run_hook(
    event: HookEvent,
    hook: HookCommand,
    stdin_text: str,
    cwd: Path,
    env: dict[str, str],
    events: AgentEvents,
) -> ProcessResult | None:
    """Запустить один хук; не запустился или не уложился в таймаут — предупреждение и None."""
    show = event not in QUIET_EVENTS
    if show:
        events.on_hook_start(event.value, hook.command)
    try:
        # Хук — команда оболочки из настроек пользователя, как в Claude Code.
        proc = run_process(  # noqa: S604
            hook.command,
            cwd=cwd,
            timeout=hook.timeout,
            shell=True,
            input=stdin_text,
            extra_env=env,
        )
    except (OSError, ValueError) as error:  # ValueError: NUL или суррогат в команде
        events.on_notice(
            f"хук {event} не запустился ({short_command(hook.command)}): {error}",
            level="warn",
        )
        return None
    finally:
        if show:
            events.on_hook_end()
    if proc.timed_out:
        events.on_notice(
            f"хук {event} не уложился в {hook.timeout} с и остановлен "
            f"({short_command(hook.command)})",
            level="warn",
        )
        return None
    return proc


def _warn_unparsed(
    event: HookEvent,
    hook: HookCommand,
    outcome: HookResult,
    events: AgentEvents,
) -> None:
    """Stdout похож на JSON, но не разбирается (лишняя строка, обрезка).

    Решение хука потеряно. Молча продолжить — значит, разрешить то, что хук, возможно,
    запретил: предупреждаем, а вызов инструмента — подтверждает пользователь.
    """
    note = "вывод похож на JSON, но не разобран — ответ хука пропущен"
    if event is HookEvent.PRE_TOOL_USE:
        note = "вывод похож на JSON, но не разобран — вызов подтвердите сами"
        outcome.decision = "ask"
    events.on_notice(f"хук {event} ({short_command(hook.command)}): {note}", level="warn")


def _block(
    event: HookEvent,
    hook: HookCommand,
    reason: str,
    outcome: HookResult,
    events: AgentEvents,
) -> None:
    """Блокировка по ответу хука — там, где ей есть что блокировать (:data:`BLOCKING_EVENTS`).

    У остальных событий — предупреждение пользователю.
    """
    if event in BLOCKING_EVENTS:
        outcome.blocked = True
        outcome.reason = reason
    elif event is HookEvent.POST_TOOL_USE:  # замечания всех хуков — модели
        outcome.blocked = True
        outcome.reason = _clip(f"{outcome.reason}\n\n{reason}" if outcome.reason else reason)
    else:
        events.on_notice(
            f"хук {event} ({short_command(hook.command)}): блокировать нечего — "
            f"{_first_line(reason)}",
            level="warn",
        )


def _apply_permission(
    event: HookEvent,
    hook: HookCommand,
    specific: dict[str, Any],
    decision: object,
    reason: str,
    outcome: HookResult,
    events: AgentEvents,
) -> None:
    """Решение хука PreToolUse о вызове инструмента: запретить, спросить или разрешить."""
    permission = specific.get("permissionDecision")
    if permission is None:  # прежний формат: decision approve/block
        legacy: dict[object, str] = {"approve": "allow", "block": "deny"}
        permission = legacy.get(decision)
    if permission == "deny":
        deny_reason = _text(specific.get("permissionDecisionReason")) or reason
        _block(event, hook, _clip(deny_reason) or NO_REASON, outcome, events)
    elif permission == "ask":
        outcome.decision = "ask"
    elif permission == "allow" and outcome.decision is None:
        outcome.decision = "allow"


def short_command(command: str, limit: int = 60) -> str:
    """Команда хука для сообщений: первая строка, не длиннее ``limit``."""
    line = command.strip().splitlines()[0] if command.strip() else ""
    if len(line) <= limit:
        return line
    head = line[: limit - 1]
    return f"{head}…"


def _clip(text: str) -> str:
    return truncate_middle(text, HOOK_OUTPUT_LIMIT, head=HOOK_OUTPUT_LIMIT // 3)


def _first_line(text: str) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return lines[0].strip()[:_FIRST_LINE_LIMIT] if lines else ""


def _text(raw: object) -> str:
    # Строку из JSON run_process не цитировал: «\u003csystem-reminder>» — тег после разбора.
    return quote_reminder_tags(raw.strip()) if isinstance(raw, str) else ""


def _json_object(text: str) -> dict[str, Any] | None:
    if not text.startswith("{"):
        return None
    try:
        parsed = json.loads(text)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None
