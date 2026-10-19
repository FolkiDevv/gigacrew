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
        "codebase research, read-only: search, code navigation, reading files, git "
        "status/diff/log. Returns a digest with paths and line numbers"
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
            "skill",
            "git",  # изменяющие подкоманды блокирует read_only
        }
    ),
    read_only=True,
)
GENERAL = SubagentSpec(
    name="general",
    description=(
        "a self-contained multi-step subtask with all tools (file edits and commands — "
        "with confirmation according to the user's mode). Returns a report on what was done"
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
    agent: str = Field(description=f"Subagent type: {', '.join(SUBAGENTS)}.")
    prompt: str = Field(
        description=(
            "A complete brief — the subagent does NOT see this conversation: the goal and "
            "the concrete question(s); known starting points (paths, names, strings, "
            "errors); what is out of scope; what to return and when to stop."
        )
    )
    description: str = Field(
        "",
        description="3–5 words in the user's language: what is delegated (shown to the user).",
    )


class TaskTool(Tool):
    name = "task"
    description = (
        "Runs a subagent with its own context for a self-contained subtask and returns "
        "its final report; the subagent's intermediate steps do not get into your "
        f"context. Agents: {_agents_list()}. The subagent has a limited number of steps "
        "and a limited context: give it ONE narrow question or subtask (one subsystem, "
        "one flow) and split broad research into several such tasks sent in one "
        "response. Pinpoint questions (a known file or symbol) are faster to answer "
        "yourself. A report of a subagent stopped by a limit is partial but useful: do "
        "not rerun the same task — narrow it to what remains. If the user stopped a "
        "subagent, do not run the same task again unless asked."
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
