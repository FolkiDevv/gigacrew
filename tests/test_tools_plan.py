"""Инструмент exit_plan_mode: план из файла, одобрение, замечания, отказ, нет UI."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from gigacrew.permissions import PermissionMode
from gigacrew.project.workspace import Workspace
from gigacrew.tools.base import PlanFile, ToolContext, ToolError, ToolResult
from gigacrew.tools.plan import (
    APPROVE,
    DECLINED_NOTE,
    NO_APPROVER_NOTE,
    ExitPlanModeTool,
    review_plan,
)
from gigacrew.tools.questions import Answer, Question, QuestionsUnavailableError

PLAN = "## План\n1. Поправить `a.py`\n2. Прогнать тесты"
PLAN_REL = ".gigacode/plans/chat1.md"

# Ответы пользователя: по ответу на вопрос, None — отказ, исключение — спросить некого.
_Answers = list[Answer] | BaseException | None


class _Session:
    """Режим агента и ответы пользователя для контекста инструмента."""

    def __init__(
        self,
        answers: _Answers = None,
        *,
        mode: PermissionMode = PermissionMode.PLAN,
        ask: bool = True,
    ) -> None:
        self.mode = mode
        self.answers = answers
        self.asked: list[list[Question]] = []
        self.ask = ask
        self._plan: PlanFile | None = None

    @property
    def plan(self) -> PlanFile:
        """Файл плана из последнего :meth:`context`."""
        assert self._plan is not None
        return self._plan

    def ask_user(self, questions: Sequence[Question]) -> list[Answer] | None:
        self.asked.append(list(questions))
        if isinstance(self.answers, BaseException):
            raise self.answers
        return self.answers

    def switch_mode(self, mode: PermissionMode) -> None:
        self.mode = mode

    def context(self, tmp_path: Path) -> ToolContext:
        workspace = Workspace(tmp_path)
        self._plan = PlanFile(workspace.plans_dir / "chat1.md", dirty=True)
        return ToolContext(
            workspace,
            ask_user=self.ask_user if self.ask else None,
            get_mode=lambda: self.mode,
            set_mode=self.switch_mode,
            plan=self._plan,
        )


def _write_plan(tmp_path: Path, plan: str = PLAN) -> None:
    path = tmp_path / PLAN_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan, encoding="utf-8")


def _run(tmp_path: Path, session: _Session, plan: str | None = PLAN) -> ToolResult:
    if plan is not None:
        _write_plan(tmp_path, plan)
    tool = ExitPlanModeTool()
    return tool.run(tool.parse({}), session.context(tmp_path))


def test_approval_switches_mode(tmp_path: Path) -> None:
    mode = PermissionMode.AUTO
    session = _Session([Answer((APPROVE,))])
    tool_result = _run(tmp_path, session)
    assert session.mode is mode
    assert tool_result.ok
    assert "одобрил план" in tool_result.content
    assert mode.label in tool_result.content
    # одобренный план (в итоговой редакции) — в результате: модель выполняет его
    assert PLAN in tool_result.content
    assert PLAN_REL in tool_result.content
    questions = session.asked[0]
    assert len(questions) == 1
    question = questions[0]
    # план — в рамке, путь к файлу — в её подписи; свой ответ — замечания к плану
    assert question.body == PLAN
    assert question.source == PLAN_REL
    assert question.custom_label == "Замечания…"
    assert question.header == "План"
    assert [option.label for option in question.options] == [APPROVE]
    assert not session.plan.dirty


def test_feedback_keeps_planning(tmp_path: Path) -> None:
    session = _Session([Answer(custom="  без новых зависимостей ")])
    tool_result = _run(tmp_path, session)
    assert session.mode is PermissionMode.PLAN
    assert tool_result.summary == "план на доработку"
    assert "«без новых зависимостей»" in tool_result.content
    assert "exit_plan_mode" in tool_result.content
    # правки — точечно в файле, а не новым планом целиком
    assert "edit_file" in tool_result.content
    assert PLAN_REL in tool_result.content
    assert not session.plan.dirty  # пользователь эту версию видел


def test_decline_keeps_planning(tmp_path: Path) -> None:
    session = _Session(None)
    tool_result = _run(tmp_path, session)
    assert session.mode is PermissionMode.PLAN
    assert tool_result.content == DECLINED_NOTE
    assert tool_result.summary == "план не одобрен"


@pytest.mark.parametrize("session", [_Session(QuestionsUnavailableError()), _Session(ask=False)])
def test_nobody_to_approve(tmp_path: Path, session: _Session) -> None:
    tool_result = _run(tmp_path, session)
    assert session.mode is PermissionMode.PLAN
    assert tool_result.ok
    assert tool_result.content == NO_APPROVER_NOTE.format(path=PLAN_REL)
    # отдать план дальше нельзя — он «отправлен»: harness не напоминает о нём снова
    assert not session.plan.dirty
    assert session.plan.reviews == 1


@pytest.mark.parametrize(
    ("answers", "outcome", "continues"),
    [
        ([Answer((APPROVE,))], "approved", True),
        ([Answer(custom="ещё тесты")], "revise", True),
        (None, "declined", False),
        (QuestionsUnavailableError(), "no_approver", False),
    ],
)
def test_review_outcomes(
    tmp_path: Path,
    answers: _Answers,
    outcome: str,
    *,
    continues: bool,
) -> None:
    _write_plan(tmp_path)
    session = _Session(answers)
    shown: list[int] = []
    review = review_plan(
        session.context(tmp_path),
        session.plan,
        on_show=lambda: shown.append(len(session.asked)),
    )
    assert review.outcome == outcome
    assert review.continues is continues
    assert shown == [0]  # перед вопросом пользователю
    assert session.plan.reviews == 1
    assert not session.plan.dirty


@pytest.mark.parametrize("plan", [None, "  \n "])
def test_review_of_missing_plan_shows_nothing(tmp_path: Path, plan: str | None) -> None:
    if plan is not None:
        _write_plan(tmp_path, plan)
    session = _Session([Answer((APPROVE,))])
    shown: list[bool] = []
    context = session.context(tmp_path)
    with pytest.raises(ToolError):
        review_plan(context, session.plan, on_show=lambda: shown.append(True))
    assert shown == []
    assert session.asked == []
    assert session.plan.reviews == 0


def test_outside_plan_mode_is_an_error(tmp_path: Path) -> None:
    session = _Session([Answer((APPROVE,))], mode=PermissionMode.AUTO)
    with pytest.raises(ToolError, match="не активен"):
        _run(tmp_path, session)
    assert session.asked == []


def test_without_agent_is_an_error(tmp_path: Path) -> None:
    tool = ExitPlanModeTool()
    with pytest.raises(ToolError, match="не активен"):
        tool.run(tool.parse({}), ToolContext(Workspace(tmp_path)))


@pytest.mark.parametrize("plan", [None, "  \n "])
def test_missing_or_empty_plan_file_is_an_error(tmp_path: Path, plan: str | None) -> None:
    session = _Session([Answer((APPROVE,))])
    with pytest.raises(ToolError, match="План не записан") as exc_info:
        _run(tmp_path, session, plan=plan)
    assert PLAN_REL in str(exc_info.value)
    assert "write_file" in str(exc_info.value)
    assert session.asked == []


def test_plan_argument_is_ignored(tmp_path: Path) -> None:
    """Старая привычка модели передавать план в аргументе — не ошибка: план из файла."""
    _write_plan(tmp_path)
    session = _Session([Answer((APPROVE,))])
    tool = ExitPlanModeTool()
    tool.run(tool.parse({"plan": "другой план"}), session.context(tmp_path))
    questions = session.asked[0]
    assert len(questions) == 1
    assert questions[0].body.startswith(PLAN)


def test_plan_argument_fills_empty_plan_file(tmp_path: Path) -> None:
    """План только в аргументе (как у других агентов), файла нет — файлом становится он."""
    session = _Session([Answer((APPROVE,))])
    context = session.context(tmp_path)
    session.plan.dirty = False
    tool = ExitPlanModeTool()
    tool_result = tool.run(tool.parse({"plan": PLAN}), context)
    assert (tmp_path / PLAN_REL).read_text(encoding="utf-8") == PLAN
    questions = session.asked[0]
    assert len(questions) == 1
    assert questions[0].body == PLAN
    assert tool_result.ok
    assert session.mode is PermissionMode.AUTO
    assert session.plan.writes == 1


def test_plan_argument_does_not_replace_shown_plan(tmp_path: Path) -> None:
    """Файл с прошлого показа не менялся, в аргументе — другой текст: попросить поправить файл.

    Старый план не показывать и не стирать его пересказом.
    """
    _write_plan(tmp_path)
    session = _Session([Answer((APPROVE,))])
    context = session.context(tmp_path)
    session.plan.dirty = False
    tool = ExitPlanModeTool()
    with pytest.raises(ToolError, match="не менялся") as exc_info:
        tool.run(tool.parse({"plan": "план в двух словах"}), context)
    assert PLAN_REL in str(exc_info.value)
    assert "edit_file" in str(exc_info.value)
    assert (tmp_path / PLAN_REL).read_text(encoding="utf-8") == PLAN
    assert session.asked == []
    tool.run(tool.parse({"plan": f"{PLAN}\n"}), context)  # тот же план — просто показ
    assert session.asked[0][0].body == PLAN


def test_schema_has_no_parameters() -> None:
    params = ExitPlanModeTool().spec().parameters
    assert params == {"type": "object", "properties": {}}
