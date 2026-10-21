"""Инструмент ``exit_plan_mode``: готовый план — пользователю на одобрение.

В режиме планирования агент только исследует код и пишет план в файл плана чата
(``.gigacode/plans/<id>.md``, обычными write_file/edit_file). Закончив, он
вызывает этот инструмент без аргументов: пользователь видит план из файла и
выбирает, выполнять ли его и в каком режиме. После одобрения режим меняется, и
агент приступает к выполнению в том же ходе; свой ответ пользователя — замечания,
агент точечно правит файл и отправляет план снова. План не передаётся в
параметрах: иначе модель переписывала бы его целиком после каждого замечания. Если
модель по привычке других агентов всё же передала план (``plan``), а файл плана пуст,
текст из аргумента записывается в файл и показывается.

Сам показ — :func:`review_plan`: ею же агентный цикл показывает записанный план, если
модель закончила ход текстом, не вызвав инструмент.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Literal

from pydantic import BaseModel, PrivateAttr

from gigacrew.permissions import PermissionMode
from gigacrew.tools.base import PlanFile, Tool, ToolContext, ToolError, ToolResult
from gigacrew.tools.fs import WriteFileParams, WriteFileTool
from gigacrew.tools.questions import Question, QuestionOption, QuestionsUnavailable

APPROVE = "Да, выполнять"
_APPROVALS = {APPROVE: PermissionMode.AUTO}
PLAN_QUESTION = Question(
    text="Выполнить этот план?",
    header="План",
    options=(
        QuestionOption(
            APPROVE, f"Режим «{PermissionMode.AUTO.label}»: {PermissionMode.AUTO.description}"
        ),
    ),
    custom_label="Замечания…",
    custom_hint="Напишите, что поправить — модель исправит план и покажет снова",
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
PLAN_ARGUMENT_NOTE = (
    "exit_plan_mode показывает файл плана {path}, а не аргумент plan; файл с прошлого "
    "показа не менялся. Внеси изменения в файл (edit_file) и снова вызови "
    "exit_plan_mode без аргументов."
)
PLAN_LINK_NOTE = (
    "Файл плана {path} или папка планов — символическая ссылка: план из аргумента не "
    "записан. Сообщи пользователю, что ссылку нужно убрать."
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


# Без параметров: план читается из файла плана (docstring ушёл бы в схему). План из
# аргумента ``plan`` — запасной путь, в схеме его нет.
class ExitPlanModeParams(BaseModel):
    _plan: str = PrivateAttr("")


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


def _take_plan_argument(ctx: ToolContext, plan: PlanFile, text: str) -> None:
    """План в аргументе ``plan``: в пустой файл плана — записать (как write_file, с теми же
    проверками пути). Файл с другим планом, не менявшийся с прошлого показа, — ошибка:
    показ старого плана сбил бы пользователя, а перезапись могла бы стереть план
    кратким пересказом. Записанный в этом цикле файл важнее аргумента.

    Файл плана или папка планов — ссылка: не записывать. Запись идёт мимо проверки
    риска write_file, а ссылка увела бы её без вопроса в файл проекта.
    """
    if plan.dirty:
        return
    path = _plan_path(ctx, plan)
    if not plan.is_target(plan.path):
        raise ToolError(PLAN_LINK_NOTE.format(path=path))
    try:
        current = plan.path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        current = ""
    except (OSError, UnicodeDecodeError) as e:
        raise ToolError(f"Не удалось прочитать файл плана {path}: {e}") from None
    if not current:
        WriteFileTool().run(WriteFileParams(path=path, content=text), ctx)
    elif current != text.strip():
        raise ToolError(PLAN_ARGUMENT_NOTE.format(path=path))


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
    if on_show is not None:
        on_show()
    try:
        answers = ctx.ask_user([replace(PLAN_QUESTION, body=plan, source=path)])
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

    def parse(self, arguments: dict[str, Any]) -> ExitPlanModeParams:
        params = ExitPlanModeParams.model_validate(arguments or {})
        plan = (arguments or {}).get("plan")
        if isinstance(plan, str):
            params._plan = plan
        return params

    def run(self, params: ExitPlanModeParams, ctx: ToolContext) -> ToolResult:
        plan_file = ctx.plan
        if plan_file is None or ctx.get_mode is None or ctx.get_mode() != PermissionMode.PLAN:
            raise ToolError(NOT_PLANNING_NOTE)
        if params._plan.strip():
            _take_plan_argument(ctx, plan_file, params._plan)
        review = review_plan(ctx, plan_file)
        return ToolResult(content=review.content, summary=review.summary)
