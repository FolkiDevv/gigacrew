"""Инструмент ``exit_plan_mode``: готовый план — пользователю на одобрение.

В режиме планирования агент только исследует код. Закончив, он вызывает этот
инструмент с планом: пользователь видит план и выбирает, выполнять ли его и в
каком режиме. После одобрения режим меняется, и агент приступает к выполнению в
том же ходе; свой ответ пользователя — замечания, план дорабатывается.
"""

from __future__ import annotations

from dataclasses import replace

from pydantic import BaseModel, Field

from gigacrew.permissions import PermissionMode
from gigacrew.tools.base import Tool, ToolContext, ToolError, ToolResult
from gigacrew.tools.questions import Question, QuestionOption, QuestionsUnavailable

APPROVE_EDITS = "Да, правки без вопросов"
APPROVE_MANUAL = "Да, с подтверждением"
_APPROVALS = {
    APPROVE_EDITS: PermissionMode.ACCEPT_EDITS,
    APPROVE_MANUAL: PermissionMode.MANUAL,
}
PLAN_QUESTION = Question(
    text="Выполнить этот план?",
    header="План",
    options=(
        QuestionOption(
            APPROVE_EDITS,
            f"Режим «{PermissionMode.ACCEPT_EDITS.label}»: "
            f"{PermissionMode.ACCEPT_EDITS.description}",
        ),
        QuestionOption(
            APPROVE_MANUAL,
            f"Режим «{PermissionMode.MANUAL.label}»: {PermissionMode.MANUAL.description}",
        ),
    ),
)

NOT_PLANNING_NOTE = "Режим планирования не активен — одобрение плана не требуется. Выполняй задачу."
DECLINED_NOTE = (
    "Пользователь пока не одобрил план. Режим планирования сохраняется: ничего не "
    "меняй, кратко заверши ответ и дождись указаний пользователя."
)
NO_APPROVER_NOTE = (
    "Одобрить план некому (неинтерактивный режим). Ничего не меняй: выведи план "
    "итоговым ответом и заверши работу."
)


class ExitPlanModeParams(BaseModel):
    plan: str = Field(
        description=(
            "The plan in Markdown, in the user's language: the goal, the steps (which "
            "files and what to change in them), how to check the result. Brief and concrete."
        )
    )


class ExitPlanModeTool(Tool):
    name = "exit_plan_mode"
    description = (
        "Plan mode only: hands the finished plan to the user for approval. Call it when "
        "the research is done and the plan is ready. If the user approves the plan, the "
        "mode changes and you start the implementation right away; if the user sends "
        "remarks, refine the plan and call it again."
    )
    Params = ExitPlanModeParams

    def describe(self, params: ExitPlanModeParams) -> str:
        for line in params.plan.splitlines():
            line = line.strip().lstrip("#").strip()
            if line:
                return line[:70]
        return ""

    def run(self, params: ExitPlanModeParams, ctx: ToolContext) -> ToolResult:
        plan = params.plan.strip()
        if not plan:
            raise ToolError("Пустой план: передай его в параметре plan.")
        if ctx.get_mode is None or ctx.set_mode is None or ctx.get_mode() != PermissionMode.PLAN:
            raise ToolError(NOT_PLANNING_NOTE)
        if ctx.ask_user is None:
            return ToolResult(content=NO_APPROVER_NOTE, summary="одобрить план некому")
        try:
            answers = ctx.ask_user([replace(PLAN_QUESTION, body=plan)])
        except QuestionsUnavailable:
            return ToolResult(content=NO_APPROVER_NOTE, summary="одобрить план некому")
        if not answers:
            return ToolResult(content=DECLINED_NOTE, summary="план не одобрен")

        answer = answers[0]
        mode = _APPROVALS.get(answer.selected[0]) if answer.selected else None
        if mode is None:
            feedback = answer.custom.strip()
            return ToolResult(
                content=(
                    f"Пользователь не одобрил план и просит доработать: «{feedback}». "
                    "Режим планирования сохраняется: учти замечания и снова вызови "
                    "exit_plan_mode с исправленным планом."
                ),
                summary="план на доработку",
            )
        ctx.set_mode(mode)
        return ToolResult(
            content=(
                f"Пользователь одобрил план. Режим теперь «{mode.label}» "
                f"({mode.description}). Приступай к выполнению плана по шагам."
            ),
            summary=f"план одобрен · режим «{mode.label}»",
        )
