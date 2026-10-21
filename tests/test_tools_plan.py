"""Инструмент exit_plan_mode: план из файла, одобрение, замечания, отказ, нет UI."""

from __future__ import annotations

import pytest

from gigacrew.permissions import PermissionMode
from gigacrew.project.workspace import Workspace
from gigacrew.tools.base import PlanFile, ToolContext, ToolError
from gigacrew.tools.plan import (
    APPROVE,
    DECLINED_NOTE,
    NO_APPROVER_NOTE,
    ExitPlanModeTool,
    review_plan,
)
from gigacrew.tools.questions import Answer, QuestionsUnavailable

PLAN = "## План\n1. Поправить `a.py`\n2. Прогнать тесты"
PLAN_REL = ".gigacode/plans/chat1.md"


class _Session:
    """Режим агента и ответы пользователя для контекста инструмента."""

    def __init__(self, answers=None, *, mode=PermissionMode.PLAN, ask=True):
        self.mode = mode
        self.answers = answers
        self.asked: list = []
        self.ask = ask
        self.plan: PlanFile | None = None

    def ask_user(self, questions):
        self.asked.append(list(questions))
        if isinstance(self.answers, BaseException):
            raise self.answers
        return self.answers

    def set_mode(self, mode):
        self.mode = mode

    def context(self, tmp_path) -> ToolContext:
        workspace = Workspace(tmp_path)
        self.plan = PlanFile(workspace.plans_dir / "chat1.md", dirty=True)
        return ToolContext(
            workspace,
            ask_user=self.ask_user if self.ask else None,
            get_mode=lambda: self.mode,
            set_mode=self.set_mode,
            plan=self.plan,
        )


def _write_plan(tmp_path, plan: str = PLAN) -> None:
    path = tmp_path / PLAN_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan, encoding="utf-8")


def _run(tmp_path, session: _Session, plan: str | None = PLAN):
    if plan is not None:
        _write_plan(tmp_path, plan)
    tool = ExitPlanModeTool()
    return tool.run(tool.parse({}), session.context(tmp_path))


def test_approval_switches_mode(tmp_path):
    mode = PermissionMode.AUTO
    session = _Session([Answer((APPROVE,))])
    result = _run(tmp_path, session)
    assert session.mode is mode
    assert result.ok and "одобрил план" in result.content and mode.label in result.content
    # одобренный план (в итоговой редакции) — в результате: модель выполняет его
    assert PLAN in result.content and PLAN_REL in result.content
    (question,) = session.asked[0]
    assert question.body.startswith(PLAN) and PLAN_REL in question.body
    assert question.header == "План"
    assert [o.label for o in question.options] == [APPROVE]
    assert not session.plan.dirty


def test_feedback_keeps_planning(tmp_path):
    session = _Session([Answer(custom="  без новых зависимостей ")])
    result = _run(tmp_path, session)
    assert session.mode is PermissionMode.PLAN
    assert result.summary == "план на доработку"
    assert "«без новых зависимостей»" in result.content and "exit_plan_mode" in result.content
    # правки — точечно в файле, а не новым планом целиком
    assert "edit_file" in result.content and PLAN_REL in result.content
    assert not session.plan.dirty  # пользователь эту версию видел


def test_decline_keeps_planning(tmp_path):
    session = _Session(None)
    result = _run(tmp_path, session)
    assert session.mode is PermissionMode.PLAN
    assert result.content == DECLINED_NOTE and result.summary == "план не одобрен"


@pytest.mark.parametrize("session", [_Session(QuestionsUnavailable()), _Session(ask=False)])
def test_nobody_to_approve(tmp_path, session):
    result = _run(tmp_path, session)
    assert session.mode is PermissionMode.PLAN
    assert result.ok and result.content == NO_APPROVER_NOTE.format(path=PLAN_REL)
    # отдать план дальше нельзя — он «отправлен»: harness не напоминает о нём снова
    assert not session.plan.dirty and session.plan.reviews == 1


@pytest.mark.parametrize(
    ("answers", "outcome", "continues"),
    [
        ([Answer((APPROVE,))], "approved", True),
        ([Answer(custom="ещё тесты")], "revise", True),
        (None, "declined", False),
        (QuestionsUnavailable(), "no_approver", False),
    ],
)
def test_review_outcomes(tmp_path, answers, outcome, continues):
    _write_plan(tmp_path)
    session = _Session(answers)
    shown: list[int] = []
    review = review_plan(
        session.context(tmp_path), session.plan, on_show=lambda: shown.append(len(session.asked))
    )
    assert review.outcome == outcome and review.continues is continues
    assert shown == [0]  # перед вопросом пользователю
    assert session.plan.reviews == 1 and not session.plan.dirty


@pytest.mark.parametrize("plan", [None, "  \n "])
def test_review_of_missing_plan_shows_nothing(tmp_path, plan):
    if plan is not None:
        _write_plan(tmp_path, plan)
    session = _Session([Answer((APPROVE,))])
    shown: list[bool] = []
    context = session.context(tmp_path)
    with pytest.raises(ToolError):
        review_plan(context, session.plan, on_show=lambda: shown.append(True))
    assert shown == [] and session.asked == [] and session.plan.reviews == 0


def test_outside_plan_mode_is_an_error(tmp_path):
    session = _Session([Answer((APPROVE,))], mode=PermissionMode.AUTO)
    with pytest.raises(ToolError, match="не активен"):
        _run(tmp_path, session)
    assert session.asked == []


def test_without_agent_is_an_error(tmp_path):
    tool = ExitPlanModeTool()
    with pytest.raises(ToolError, match="не активен"):
        tool.run(tool.parse({}), ToolContext(Workspace(tmp_path)))


@pytest.mark.parametrize("plan", [None, "  \n "])
def test_missing_or_empty_plan_file_is_an_error(tmp_path, plan):
    session = _Session([Answer((APPROVE,))])
    with pytest.raises(ToolError, match="План не записан") as e:
        _run(tmp_path, session, plan=plan)
    assert PLAN_REL in str(e.value) and "write_file" in str(e.value)
    assert session.asked == []


def test_plan_argument_is_ignored(tmp_path):
    """Старая привычка модели передавать план в аргументе — не ошибка: план из файла."""
    _write_plan(tmp_path)
    session = _Session([Answer((APPROVE,))])
    tool = ExitPlanModeTool()
    tool.run(tool.parse({"plan": "другой план"}), session.context(tmp_path))
    (question,) = session.asked[0]
    assert question.body.startswith(PLAN)


def test_schema_has_no_parameters():
    params = ExitPlanModeTool().spec().parameters
    assert params == {"type": "object", "properties": {}}
