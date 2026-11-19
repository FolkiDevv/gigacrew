"""Подтверждение операции через ACP: варианты ``session/request_permission`` и разбор ответа.

Варианты — по порядку: «да» (``allow_once``; T3 на «Разрешить» выбирает первый вариант
этого вида), «до конца хода» (тоже ``allow_once``; только если агент назвал, что он покроет),
«всегда» (``allow_always``) и «нет» (``reject_once``). Опасное «до конца хода» и «всегда» не
предлагается, «всегда» — и когда его нельзя запомнить (хук PreToolUse спрашивает каждый раз).
История ответов на такую команду — в названии «да» (для клиентов, которые его показывают);
«да» всегда разрешает один раз. Ответ, который не выбирает ни один из предложенных
вариантов, — отказ без ответа пользователя.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Final

from gigacrew.agent.events import Approval, ConfirmResult, ToolCallInfo
from gigacrew.agent.loop import KEY_REMEMBER_HINT

__all__ = ["ALLOW_ALWAYS", "ALLOW_ONCE", "ALLOW_TURN", "REJECT_ONCE", "confirm_result", "options"]

ALLOW_ONCE: Final = "allow_once"
ALLOW_TURN: Final = "allow_turn"
ALLOW_ALWAYS: Final = "allow_always"
REJECT_ONCE: Final = "reject_once"

_APPROVALS: Final = MappingProxyType({
    ALLOW_ONCE: Approval.YES,
    ALLOW_TURN: Approval.TURN,
    ALLOW_ALWAYS: Approval.ALWAYS,
    REJECT_ONCE: Approval.NO,
})
_NO_ANSWER: Final = ConfirmResult(Approval.NO, answered=False)
# Что запомнит «всегда», если агент не назвал правило.
_DEFAULT_ALWAYS: Final = "не спрашивать этот вызов до конца сессии"

type Json = dict[str, Any]


def options(call: ToolCallInfo, *, dangerous: bool) -> list[Json]:
    """Варианты ответа на подтверждение вызова ``call`` по порядку."""
    # История ответов — при «да», как подсказка CLI; само «да» ничего не запоминает.
    # Подсказка про клавишу терминала «a» в T3 бессмысленна: вместо неё — «Всегда».
    history = call.history.removesuffix(f" — {KEY_REMEMBER_HINT}")
    once = f"Да ({history})" if history else "Да"
    offered = [_option(ALLOW_ONCE, once, ALLOW_ONCE)]
    if not dangerous:
        if call.turn_scope:
            name = f"Да, до конца хода: {call.turn_scope}"
            offered.append(_option(ALLOW_TURN, name, ALLOW_ONCE))
        if call.can_remember:
            remember = call.remember or _DEFAULT_ALWAYS
            offered.append(_option(ALLOW_ALWAYS, f"Всегда: {remember}", ALLOW_ALWAYS))
    offered.append(_option(REJECT_ONCE, "Нет", REJECT_ONCE))
    return offered


def confirm_result(response: Any, offered: list[Json]) -> ConfirmResult:
    """Решение по ответу клиента; не выбран ни один из ``offered`` — отказ без ответа."""
    outcome = response.get("outcome") if isinstance(response, dict) else None
    if not isinstance(outcome, dict) or outcome.get("outcome") != "selected":
        return _NO_ANSWER
    option_id = outcome.get("optionId")
    if not isinstance(option_id, str):
        return _NO_ANSWER
    if not any(option["optionId"] == option_id for option in offered):
        return _NO_ANSWER
    return ConfirmResult(_APPROVALS[option_id])


def _option(option_id: str, name: str, kind: str) -> Json:
    return {"optionId": option_id, "name": name, "kind": kind}
