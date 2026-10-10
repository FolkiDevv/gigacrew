"""Режимы разрешений: решение по вызову, переключение, разбор имени."""

from __future__ import annotations

import pytest

from gigacrew.permissions import (
    MODE_CYCLE,
    Decision,
    PermissionMode,
    ToolKind,
    decide,
    next_mode,
    parse_mode,
)
from gigacrew.security import RiskLevel

MANUAL, EDITS, PLAN = PermissionMode.MANUAL, PermissionMode.ACCEPT_EDITS, PermissionMode.PLAN
ALLOW, ASK, BLOCK = Decision.ALLOW, Decision.ASK, Decision.BLOCK


@pytest.mark.parametrize("mode", list(PermissionMode))
@pytest.mark.parametrize("kind", list(ToolKind))
@pytest.mark.parametrize("auto_approve", [False, True])
def test_reading_is_always_allowed(mode, kind, auto_approve):
    assert decide(mode, RiskLevel.SAFE, kind, auto_approve=auto_approve) is ALLOW


@pytest.mark.parametrize(
    ("mode", "risk", "kind", "expected"),
    [
        # ручной: всё изменяющее — с вопросом
        (MANUAL, RiskLevel.WRITE, ToolKind.EDIT, ASK),
        (MANUAL, RiskLevel.WRITE, ToolKind.COMMAND, ASK),
        (MANUAL, RiskLevel.DANGEROUS, ToolKind.COMMAND, ASK),
        (MANUAL, RiskLevel.WRITE, ToolKind.OTHER, ASK),
        # авто-правки: правки файлов — без вопроса, остальное — с вопросом
        (EDITS, RiskLevel.WRITE, ToolKind.EDIT, ALLOW),
        (EDITS, RiskLevel.WRITE, ToolKind.COMMAND, ASK),
        (EDITS, RiskLevel.DANGEROUS, ToolKind.COMMAND, ASK),
        (EDITS, RiskLevel.WRITE, ToolKind.OTHER, ASK),
        # план: правки и изменяющий git запрещены, команды — с вопросом
        (PLAN, RiskLevel.WRITE, ToolKind.EDIT, BLOCK),
        (PLAN, RiskLevel.WRITE, ToolKind.OTHER, BLOCK),
        (PLAN, RiskLevel.WRITE, ToolKind.COMMAND, ASK),
        (PLAN, RiskLevel.DANGEROUS, ToolKind.COMMAND, ASK),
    ],
)
def test_decision_matrix(mode, risk, kind, expected):
    assert decide(mode, risk, kind, auto_approve=False) is expected


@pytest.mark.parametrize(
    ("mode", "kind", "expected"),
    [
        (MANUAL, ToolKind.EDIT, ALLOW),
        (MANUAL, ToolKind.COMMAND, ALLOW),
        (EDITS, ToolKind.COMMAND, ALLOW),
        # -y не снимает запрет режима плана
        (PLAN, ToolKind.EDIT, BLOCK),
        (PLAN, ToolKind.OTHER, BLOCK),
        (PLAN, ToolKind.COMMAND, ALLOW),
    ],
)
def test_auto_approve(mode, kind, expected):
    assert decide(mode, RiskLevel.WRITE, kind, auto_approve=True) is expected
    # опасное без вопроса — только с --yes-all
    assert decide(mode, RiskLevel.DANGEROUS, kind, auto_approve=True, yes_all=True) is expected


@pytest.mark.parametrize("mode", [MANUAL, EDITS, PLAN])
def test_auto_approve_still_asks_for_dangerous(mode):
    assert decide(mode, RiskLevel.DANGEROUS, ToolKind.COMMAND, auto_approve=True) is ASK


def test_cycle_order():
    assert MODE_CYCLE == (MANUAL, EDITS, PLAN)
    assert [next_mode(m) for m in MODE_CYCLE] == [EDITS, PLAN, MANUAL]


@pytest.mark.parametrize(("text", "mode"), [("manual", MANUAL), (" Edits ", EDITS), ("PLAN", PLAN)])
def test_parse_mode(text, mode):
    assert parse_mode(text) is mode


@pytest.mark.parametrize("text", ["", "auto", "план"])
def test_parse_mode_rejects_unknown(text):
    with pytest.raises(ValueError, match="manual, edits, plan"):
        parse_mode(text)


def test_labels():
    assert [m.label for m in MODE_CYCLE] == ["ручной", "авто-правки", "план"]
    assert all(m.description for m in MODE_CYCLE)


@pytest.mark.parametrize("mode", list(PermissionMode))
@pytest.mark.parametrize("kind", list(ToolKind))
@pytest.mark.parametrize("risk", [RiskLevel.WRITE, RiskLevel.DANGEROUS])
@pytest.mark.parametrize(
    "flags", [{}, {"auto_approve": True}, {"auto_approve": True, "yes_all": True}]
)
def test_read_only_blocks_any_change(mode, kind, risk, flags):
    flags = {"auto_approve": False, **flags}
    assert decide(mode, risk, kind, read_only=True, **flags) is BLOCK
    assert decide(mode, RiskLevel.SAFE, kind, read_only=True, **flags) is ALLOW
