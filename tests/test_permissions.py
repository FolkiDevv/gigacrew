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

AUTO, PLAN = PermissionMode.AUTO, PermissionMode.PLAN
ALLOW, REVIEW, ASK, BLOCK = Decision.ALLOW, Decision.REVIEW, Decision.ASK, Decision.BLOCK


@pytest.mark.parametrize("mode", list(PermissionMode))
@pytest.mark.parametrize("kind", list(ToolKind))
@pytest.mark.parametrize("auto_approve", [False, True])
def test_reading_is_always_allowed(
    mode: PermissionMode,
    kind: ToolKind,
    *,
    auto_approve: bool,
) -> None:
    assert decide(mode, RiskLevel.SAFE, kind, auto_approve=auto_approve) is ALLOW


@pytest.mark.parametrize(
    ("mode", "risk", "kind", "expected"),
    [
        # авто: правки файлов — без вопроса, изменяющие команды — рецензенту,
        # опасные команды и прочее изменяющее — с вопросом
        (AUTO, RiskLevel.WRITE, ToolKind.EDIT, ALLOW),
        (AUTO, RiskLevel.WRITE, ToolKind.COMMAND, REVIEW),
        (AUTO, RiskLevel.DANGEROUS, ToolKind.COMMAND, ASK),
        (AUTO, RiskLevel.WRITE, ToolKind.OTHER, ASK),
        (AUTO, RiskLevel.DANGEROUS, ToolKind.OTHER, ASK),
        # план: правки и изменяющий git запрещены, команды — с вопросом
        (PLAN, RiskLevel.WRITE, ToolKind.EDIT, BLOCK),
        (PLAN, RiskLevel.WRITE, ToolKind.OTHER, BLOCK),
        (PLAN, RiskLevel.WRITE, ToolKind.COMMAND, ASK),
        (PLAN, RiskLevel.DANGEROUS, ToolKind.COMMAND, ASK),
    ],
)
def test_decision_matrix(
    mode: PermissionMode,
    risk: RiskLevel,
    kind: ToolKind,
    expected: Decision,
) -> None:
    assert decide(mode, risk, kind, auto_approve=False) is expected


@pytest.mark.parametrize(
    ("mode", "kind", "expected"),
    [
        (AUTO, ToolKind.EDIT, ALLOW),
        (AUTO, ToolKind.COMMAND, ALLOW),
        (AUTO, ToolKind.OTHER, ALLOW),
        # -y не снимает запрет режима плана
        (PLAN, ToolKind.EDIT, BLOCK),
        (PLAN, ToolKind.OTHER, BLOCK),
        (PLAN, ToolKind.COMMAND, ALLOW),
    ],
)
def test_auto_approve(mode: PermissionMode, kind: ToolKind, expected: Decision) -> None:
    assert decide(mode, RiskLevel.WRITE, kind, auto_approve=True) is expected
    # опасное без вопроса — только с --yes-all
    assert decide(mode, RiskLevel.DANGEROUS, kind, auto_approve=True, yes_all=True) is expected


@pytest.mark.parametrize("mode", [AUTO, PLAN])
def test_auto_approve_still_asks_for_dangerous(mode: PermissionMode) -> None:
    assert decide(mode, RiskLevel.DANGEROUS, ToolKind.COMMAND, auto_approve=True) is ASK


def test_cycle_order() -> None:
    assert MODE_CYCLE == (AUTO, PLAN)
    assert [next_mode(mode) for mode in MODE_CYCLE] == [PLAN, AUTO]


@pytest.mark.parametrize(
    ("text", "mode"),
    # manual и edits — прежние режимы, их заменил «авто»
    [("auto", AUTO), ("PLAN", PLAN), ("manual", AUTO), (" Edits ", AUTO)],
)
def test_parse_mode(text: str, mode: PermissionMode) -> None:
    assert parse_mode(text) is mode


@pytest.mark.parametrize("text", ["", "yolo", "план"])
def test_parse_mode_rejects_unknown(text: str) -> None:
    with pytest.raises(ValueError, match="auto, plan"):
        parse_mode(text)


def test_labels() -> None:
    assert [mode.label for mode in MODE_CYCLE] == ["авто", "план"]
    assert all(mode.description for mode in MODE_CYCLE)


@pytest.mark.parametrize("mode", list(PermissionMode))
@pytest.mark.parametrize("kind", list(ToolKind))
@pytest.mark.parametrize("risk", [RiskLevel.WRITE, RiskLevel.DANGEROUS])
@pytest.mark.parametrize(
    "flags",
    [{}, {"auto_approve": True}, {"auto_approve": True, "yes_all": True}],
)
def test_read_only_blocks_any_change(
    mode: PermissionMode,
    kind: ToolKind,
    risk: RiskLevel,
    flags: dict[str, bool],
) -> None:
    auto_approve = flags.get("auto_approve", False)
    yes_all = flags.get("yes_all", False)
    assert (
        decide(mode, risk, kind, read_only=True, auto_approve=auto_approve, yes_all=yes_all)
        is BLOCK
    )
    assert (
        decide(
            mode,
            RiskLevel.SAFE,
            kind,
            read_only=True,
            auto_approve=auto_approve,
            yes_all=yes_all,
        )
        is ALLOW
    )
