"""Инструмент exit_plan_mode: одобрение плана, замечания, отказ, нет UI."""

from __future__ import annotations

import pytest

from gigacrew.permissions import PermissionMode
from gigacrew.project.workspace import Workspace
from gigacrew.tools.base import ToolContext, ToolError
from gigacrew.tools.plan import (
    APPROVE_EDITS,
    APPROVE_MANUAL,
    DECLINED_NOTE,
    NO_APPROVER_NOTE,
    ExitPlanModeTool,
)
from gigacrew.tools.questions import Answer, QuestionsUnavailable

PLAN = "## План\n1. Поправить `a.py`\n2. Прогнать тесты"


class _Session:
    """Режим агента и ответы пользователя для контекста инструмента."""

    def __init__(self, answers=None, *, mode=PermissionMode.PLAN, ask=True):
        self.mode = mode
        self.answers = answers
        self.asked: list = []
        self.ask = ask

    def ask_user(self, questions):
        self.asked.append(list(questions))
        if isinstance(self.answers, BaseException):
            raise self.answers
        return self.answers

    def set_mode(self, mode):
        self.mode = mode

    def context(self, tmp_path) -> ToolContext:
        return ToolContext(
            Workspace(tmp_path),
            ask_user=self.ask_user if self.ask else None,
            get_mode=lambda: self.mode,
            set_mode=self.set_mode,
        )


def _run(tmp_path, session: _Session, plan: str = PLAN):
    tool = ExitPlanModeTool()
    return tool.run(tool.parse({"plan": plan}), session.context(tmp_path))


@pytest.mark.parametrize(
    ("label", "mode"),
    [(APPROVE_EDITS, PermissionMode.ACCEPT_EDITS), (APPROVE_MANUAL, PermissionMode.MANUAL)],
)
def test_approval_switches_mode(tmp_path, label, mode):
    session = _Session([Answer((label,))])
    result = _run(tmp_path, session)
    assert session.mode is mode
    assert result.ok and "одобрил план" in result.content and mode.label in result.content
    (question,) = session.asked[0]
    assert question.body == PLAN and question.header == "План"
    assert [o.label for o in question.options] == [APPROVE_EDITS, APPROVE_MANUAL]


def test_feedback_keeps_planning(tmp_path):
    session = _Session([Answer(custom="  без новых зависимостей ")])
    result = _run(tmp_path, session)
    assert session.mode is PermissionMode.PLAN
    assert result.summary == "план на доработку"
    assert "«без новых зависимостей»" in result.content and "exit_plan_mode" in result.content


def test_decline_keeps_planning(tmp_path):
    session = _Session(None)
    result = _run(tmp_path, session)
    assert session.mode is PermissionMode.PLAN
    assert result.content == DECLINED_NOTE and result.summary == "план не одобрен"


@pytest.mark.parametrize("session", [_Session(QuestionsUnavailable()), _Session(ask=False)])
def test_nobody_to_approve(tmp_path, session):
    result = _run(tmp_path, session)
    assert session.mode is PermissionMode.PLAN
    assert result.ok and result.content == NO_APPROVER_NOTE


def test_outside_plan_mode_is_an_error(tmp_path):
    session = _Session([Answer((APPROVE_EDITS,))], mode=PermissionMode.MANUAL)
    with pytest.raises(ToolError, match="не активен"):
        _run(tmp_path, session)
    assert session.asked == []


def test_without_agent_is_an_error(tmp_path):
    tool = ExitPlanModeTool()
    with pytest.raises(ToolError, match="не активен"):
        tool.run(tool.parse({"plan": PLAN}), ToolContext(Workspace(tmp_path)))


def test_empty_plan_is_an_error(tmp_path):
    session = _Session([Answer((APPROVE_EDITS,))])
    with pytest.raises(ToolError, match="Пустой план"):
        _run(tmp_path, session, plan="  \n ")
    assert session.asked == []


def test_describe_uses_first_line():
    tool = ExitPlanModeTool()
    assert tool.describe(tool.parse({"plan": "\n## Рефакторинг конфига\n1. ..."})) == (
        "Рефакторинг конфига"
    )
