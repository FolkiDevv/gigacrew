"""Инструмент ``exit_plan_mode``: готовый план — пользователю на одобрение.

В режиме планирования агент только исследует код и пишет план в файл плана чата
(``.gigacrew/plans/<id>.md``, обычными write_file/edit_file). Закончив, он
вызывает этот инструмент без аргументов: пользователь видит план из файла и
выбирает, выполнять ли его и в каком режиме. После одобрения режим меняется, и
агент приступает к выполнению в том же ходе; свой ответ пользователя — замечания,
агент точечно правит файл и отправляет план снова. План не передаётся в
параметрах: иначе модель переписывала бы его целиком после каждого замечания.

Сам показ — :func:`review_plan`: ею же агентный цикл показывает записанный план, если
модель закончила ход текстом, не вызвав инструмент.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

from pydantic import BaseModel

from gigacrew.permissions import PermissionMode
from gigacrew.tools.base import PlanFile, Tool, ToolContext, ToolError, ToolResult
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
    "Одобрить план некому (неинтерактивный режим). Ничего не меняй: план сохранён в "
    "{path} — кратко перескажи его суть, назови этот файл и заверши работу."
)
REVISE_NOTE = (
    "Пользователь не одобрил план и просит доработать: «{feedback}». Режим планирования "
    "сохраняется. Учти замечания: поправь в файле плана {path} только нужные места "
    "инструментом edit_file (НЕ переписывай план целиком и не пересказывай его текстом), "
    "затем снова вызови exit_plan_mode без аргументов."
)
APPROVED_NOTE = (
    "Пользователь одобрил план. Режим теперь «{label}» ({description}). Приступай к "
    "выполнению плана по шагам.\n\nОдобренный план ({path}):\n\n{plan}"
)

# Исход показа плана: одобрен, на доработке, не одобрен (Esc), одобрить некому.
ReviewOutcome = Literal["approved", "revise", "declined", "no_approver"]


@dataclass(frozen=True)
class PlanReview:
    """Итог показа плана: ``content`` — для модели, ``summary`` — строка для пользователя."""

    outcome: ReviewOutcome
    content: str
    summary: str

    @property
    def continues(self) -> bool:
        """Работа в ходе продолжается: план выполняется или дорабатывается."""
        return self.outcome in ("approved", "revise")


# Без параметров: план читается из файла плана (docstring ушёл бы в схему).
class ExitPlanModeParams(BaseModel):
    pass


def _plan_path(ctx: ToolContext, plan: PlanFile) -> str:
    try:
        return plan.path.relative_to(ctx.root).as_posix()
    except ValueError:
        return str(plan.path)


def _read_plan(ctx: ToolContext, plan: PlanFile) -> str:
    path = _plan_path(ctx, plan)
    try:
        text = plan.path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        text = ""
    except (OSError, UnicodeDecodeError) as e:
        raise ToolError(f"Не удалось прочитать файл плана {path}: {e}") from None
    if not text:
        raise ToolError(
            f"План не записан: сначала запиши его в {path} инструментом write_file, "
            "затем снова вызови exit_plan_mode (без аргументов)."
        )
    return text


def review_plan(
    ctx: ToolContext, plan_file: PlanFile, *, on_show: Callable[[], None] | None = None
) -> PlanReview:
    """Показать план из файла пользователю и узнать решение.

    Одобрение меняет режим (``ctx.set_mode``); пустой или нечитаемый файл плана —
    :class:`ToolError`. После показа ``dirty`` сбрасывается — и когда одобрить некому:
    отдать план дальше всё равно нельзя, а напоминание отправить его гоняло бы модель
    по кругу. ``on_show`` вызывается перед вопросом пользователю (план прочитан).
    """
    set_mode = ctx.set_mode
    if set_mode is None:  # агента нет — режим менять некому
        raise ToolError(NOT_PLANNING_NOTE)
    path = _plan_path(ctx, plan_file)
    plan = _read_plan(ctx, plan_file)
    plan_file.reviews += 1
    no_approver = PlanReview(
        "no_approver", NO_APPROVER_NOTE.format(path=path), "одобрить план некому"
    )
    if ctx.ask_user is None:
        plan_file.dirty = False
        return no_approver
    body = f"{plan}\n\n---\nФайл плана: `{path}`"
    if on_show is not None:
        on_show()
    try:
        answers = ctx.ask_user([replace(PLAN_QUESTION, body=body)])
    except QuestionsUnavailable:
        plan_file.dirty = False
        return no_approver
    plan_file.dirty = False  # пользователь видел эту версию плана
    if not answers:
        return PlanReview("declined", DECLINED_NOTE, "план не одобрен")

    answer = answers[0]
    mode = _APPROVALS.get(answer.selected[0]) if answer.selected else None
    if mode is None:
        feedback = answer.custom.strip()
        return PlanReview(
            "revise", REVISE_NOTE.format(feedback=feedback, path=path), "план на доработку"
        )
    set_mode(mode)
    return PlanReview(
        "approved",
        APPROVED_NOTE.format(label=mode.label, description=mode.description, path=path, plan=plan),
        f"план одобрен · режим «{mode.label}»",
    )


class ExitPlanModeTool(Tool):
    name = "exit_plan_mode"
    description = (
        "Plan mode only: hands the plan to the user for approval. First write the plan "
        "to the plan file named in the plan mode rules (write_file), then call this "
        "tool WITHOUT arguments — it shows the file to the user. If the user approves "
        "the plan, the mode changes and you start the implementation right away; if the "
        "user sends remarks, fix the plan file with edit_file and call it again."
    )
    Params = ExitPlanModeParams

    def run(self, params: ExitPlanModeParams, ctx: ToolContext) -> ToolResult:
        plan_file = ctx.plan
        if plan_file is None or ctx.get_mode is None or ctx.get_mode() != PermissionMode.PLAN:
            raise ToolError(NOT_PLANNING_NOTE)
        review = review_plan(ctx, plan_file)
        return ToolResult(content=review.content, summary=review.summary)
