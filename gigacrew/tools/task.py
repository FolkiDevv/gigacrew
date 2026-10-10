"""Инструмент ``task``: подзадача — суб-агенту со своим контекстом.

Основной агент поручает самостоятельную подзадачу (например, широкое исследование
кодовой базы) суб-агенту. Суб-агент работает в отдельной истории со своим набором
инструментов и возвращает только итоговый отчёт — контекст основного агента не
засоряется промежуточными результатами.

Здесь — описания встроенных агентов (данные, общие для инструмента и ядра) и сам
инструмент. Запускает суб-агента ядро (:mod:`gigacrew.agent.subagents`,
``Agent._run_subagent``) — инструмент получает его через ``ToolContext.run_subagent``.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from gigacrew.tools.base import Tool, ToolContext, ToolError, ToolResult


@dataclass(frozen=True)
class SubagentSpec:
    """Встроенный суб-агент.

    ``tools`` — разрешённые инструменты (None — все, кроме :data:`SUBAGENT_EXCLUDED`).
    ``read_only`` — любые изменения заблокированы (``permissions.decide``), в каком бы
    режиме ни работал пользователь.
    """

    name: str
    description: str
    tools: frozenset[str] | None = None
    read_only: bool = False


# Суб-агенту недоступны: запуск суб-агентов (без вложенности), вопросы пользователю и
# одобрение плана — пользователь говорит только с основным агентом.
SUBAGENT_EXCLUDED: frozenset[str] = frozenset({"task", "ask_user", "exit_plan_mode"})

EXPLORE = SubagentSpec(
    name="explore",
    description=(
        "исследование кодовой базы, только чтение: поиск, навигация по коду, чтение "
        "файлов, git status/diff/log. Возвращает выжимку с путями и номерами строк"
    ),
    tools=frozenset(
        {
            "read_file",
            "list_dir",
            "find_files",
            "search_content",
            "find_symbol",
            "find_references",
            "goto_definition",
            "call_hierarchy",
            "file_outline",
            "repo_map",
            "git",  # изменяющие подкоманды блокирует read_only
        }
    ),
    read_only=True,
)
GENERAL = SubagentSpec(
    name="general",
    description=(
        "самостоятельная многошаговая подзадача со всеми инструментами (правки файлов, "
        "команды — с подтверждением по режиму пользователя). Возвращает отчёт о сделанном"
    ),
)

SUBAGENTS: dict[str, SubagentSpec] = {spec.name: spec for spec in (EXPLORE, GENERAL)}


def allows_tool(spec: SubagentSpec, name: str) -> bool:
    """Доступен ли инструмент ``name`` суб-агенту ``spec``."""
    if name in SUBAGENT_EXCLUDED:
        return False
    return spec.tools is None or name in spec.tools


def _agents_list() -> str:
    return "; ".join(f"{spec.name} — {spec.description}" for spec in SUBAGENTS.values())


class TaskParams(BaseModel):
    agent: str = Field(description=f"Тип суб-агента: {', '.join(SUBAGENTS)}.")
    prompt: str = Field(
        description=(
            "Полная постановка задачи. Суб-агент НЕ видит этот диалог: опиши цель, "
            "известные пути, имена и факты, ограничения и что именно вернуть в отчёте."
        )
    )
    description: str = Field(
        "", description="Кратко, 3–5 слов: что поручено (показывается пользователю)."
    )


class TaskTool(Tool):
    name = "task"
    description = (
        "Запускает суб-агента со своим контекстом для самостоятельной подзадачи и "
        "возвращает его итоговый отчёт; промежуточные шаги суб-агента в твой контекст "
        f"не попадают. Агенты: {_agents_list()}. Когда использовать: широкое "
        "исследование, когда неизвестно, где искать, и придётся просмотреть много "
        "файлов, — explore; самостоятельная многошаговая подзадача — general. "
        "Точечные вопросы (известен файл или символ) решай сам — так быстрее. "
        "Суб-агент не видит диалог: в prompt передай всё нужное. Отчёт видишь только "
        "ты — перескажи пользователю главное. Если пользователь остановил суб-агента, "
        "не запускай ту же задачу снова без его просьбы."
    )
    Params = TaskParams

    def describe(self, params: TaskParams) -> str:
        return f"{params.agent.strip()} · {task_title(params.description, params.prompt)}"[:70]

    def run(self, params: TaskParams, ctx: ToolContext) -> ToolResult:
        agent = params.agent.strip().lower()
        if agent not in SUBAGENTS:
            raise ToolError(
                f"неизвестный суб-агент {params.agent.strip()!r}; доступны: {', '.join(SUBAGENTS)}."
            )
        prompt = params.prompt.strip()
        if not prompt:
            raise ToolError("Пустая задача: опиши её в параметре prompt.")
        if ctx.run_subagent is None:
            raise ToolError("суб-агенты здесь недоступны — выполни задачу сам.")
        return ctx.run_subagent(agent, task_title(params.description, prompt), prompt)


def task_title(description: str, prompt: str) -> str:
    """Краткое описание задачи для UI: ``description`` или первая строка ``prompt``."""
    title = description.strip()
    if not title:
        title = next((line.strip() for line in prompt.splitlines() if line.strip()), "")
    return title[:60]
