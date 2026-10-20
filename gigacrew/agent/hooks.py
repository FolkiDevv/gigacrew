"""Хуки пользователя: команды из настроек (:mod:`gigacrew.project.settings`), которые
gigacrew запускает на событиях агента.

Хук получает JSON в stdin и отвечает кодом выхода и (необязательно) JSON в stdout —
как в Claude Code:

* 0 — успех; stdout-объект разбирается как JSON (решение, контекст для модели),
  простой текст становится контекстом только у UserPromptSubmit и SessionStart;
* 2 — блокировка: действие не выполняется (или ход продолжается — у Stop), причина
  берётся из stderr;
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
from gigacrew.tools.process import ProcessResult, run_process, truncate_middle

BLOCK_EXIT_CODE = 2
# Сколько символов вывода хука уходит модели (контекст, причина блокировки).
HOOK_OUTPUT_LIMIT = 10_000
# Простой текст в stdout — контекст для модели только у этих событий: иначе вывод вроде
# «1 file reformatted» уходил бы модели после каждой правки.
CONTEXT_EVENTS = frozenset({HookEvent.USER_PROMPT_SUBMIT, HookEvent.SESSION_START})
# Эти хуки идут на каждом вызове инструмента — без индикатора, иначе он мигал бы.
QUIET_EVENTS = frozenset({HookEvent.PRE_TOOL_USE, HookEvent.POST_TOOL_USE, HookEvent.NOTIFICATION})
NO_REASON = "хук не пояснил причину"


@dataclass
class HookResult:
    """Итог всех хуков события."""

    blocked: bool = False  # код 2, decision "block" или permissionDecision "deny"
    reason: str = ""  # причина блокировки (для модели и пользователя)
    decision: Literal["allow", "ask"] | None = None  # PreToolUse: решение о подтверждении
    context: list[str] = field(default_factory=list)  # дополнительный контекст для модели

    @property
    def context_text(self) -> str:
        return "\n\n".join(self.context)


class PromptBlocked(Exception):
    """Хук UserPromptSubmit отклонил запрос пользователя: ход не начинался."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class HookRunner:
    """Запуск хуков события и сведение их ответов в один :class:`HookResult`.

    Хуки события идут по очереди (порядок — как в настройках); первая блокировка
    останавливает остальные. ``deny`` важнее ``ask``, ``ask`` — важнее ``allow``;
    контексты склеиваются.
    """

    def __init__(self, settings: HookSettings | None = None):
        self.settings = HookSettings() if settings is None else settings

    def has(self, event: HookEvent, target: str = "") -> bool:
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
        result = HookResult()
        commands = self.settings.commands(event, target)
        if not commands:
            return result
        data = json.dumps(payload, ensure_ascii=True, default=str)
        env = {"GIGACREW_PROJECT_DIR": str(cwd), "CLAUDE_PROJECT_DIR": str(cwd)}
        for hook in commands:
            proc = self._run_one(event, hook, data, cwd, env, events)
            if proc is not None:
                self._apply(event, hook, proc, result, events)
            if result.blocked:
                break
        return result

    @staticmethod
    def _run_one(
        event: HookEvent,
        hook: HookCommand,
        data: str,
        cwd: Path,
        env: dict[str, str],
        events: AgentEvents,
    ) -> ProcessResult | None:
        show = event not in QUIET_EVENTS
        if show:
            events.on_hook_start(event.value, hook.command)
        try:
            proc = run_process(
                hook.command, cwd=cwd, timeout=hook.timeout, shell=True, input=data, extra_env=env
            )
        except OSError as e:
            events.on_notice(
                f"хук {event} не запустился ({short_command(hook.command)}): {e}", level="warn"
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

    def _apply(
        self,
        event: HookEvent,
        hook: HookCommand,
        proc: ProcessResult,
        result: HookResult,
        events: AgentEvents,
    ) -> None:
        if proc.returncode == BLOCK_EXIT_CODE:
            result.blocked = True
            result.reason = _clip(proc.stderr.strip()) or NO_REASON
            return
        if proc.returncode != 0:
            detail = _first_line(proc.stderr) or _first_line(proc.stdout)
            events.on_notice(
                f"хук {event} ({short_command(hook.command)}) завершился с кодом "
                f"{proc.returncode}" + (f": {detail}" if detail else ""),
                level="warn",
            )
            return
        out = proc.stdout.strip()
        if not out:
            return
        data = _json_object(out)
        if data is None:
            if event in CONTEXT_EVENTS:
                result.context.append(_clip(out))
            return
        self._apply_json(event, data, result, events)

    @staticmethod
    def _apply_json(
        event: HookEvent, data: dict[str, Any], result: HookResult, events: AgentEvents
    ) -> None:
        message = _text(data.get("systemMessage"))
        if message:
            events.on_notice(f"хук {event}: {message}")
        specific = data.get("hookSpecificOutput")
        if not isinstance(specific, dict):
            specific = {}
        decision = data.get("decision")
        reason = _text(data.get("reason"))
        if event is HookEvent.PRE_TOOL_USE:
            permission = specific.get("permissionDecision")
            if permission is None:  # прежний формат: decision approve/block
                permission = {"approve": "allow", "block": "deny"}.get(decision)
            if permission == "deny":
                result.blocked = True
                result.reason = _text(specific.get("permissionDecisionReason")) or reason
                result.reason = _clip(result.reason) or NO_REASON
            elif permission == "ask":
                result.decision = "ask"
            elif permission == "allow" and result.decision is None:
                result.decision = "allow"
        elif decision == "block":
            result.blocked = True
            result.reason = _clip(reason) or NO_REASON
        context = _text(specific.get("additionalContext"))
        if context:
            result.context.append(_clip(context))


def short_command(command: str, limit: int = 60) -> str:
    """Команда хука для сообщений: первая строка, не длиннее ``limit``."""
    line = command.strip().splitlines()[0] if command.strip() else ""
    return line if len(line) <= limit else f"{line[: limit - 1]}…"


def _clip(text: str) -> str:
    return truncate_middle(text, HOOK_OUTPUT_LIMIT, head=HOOK_OUTPUT_LIMIT // 3)


def _first_line(text: str) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return lines[0].strip()[:200] if lines else ""


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _json_object(text: str) -> dict[str, Any] | None:
    if not text.startswith("{"):
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None
