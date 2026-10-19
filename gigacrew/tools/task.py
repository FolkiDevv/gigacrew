"""Инструмент ``task``: подзадача — суб-агенту со своим контекстом.

Основной агент поручает самостоятельную подзадачу (например, широкое исследование
кодовой базы) суб-агенту. Суб-агент работает в отдельной истории со своим набором
инструментов и возвращает только итоговый отчёт — контекст основного агента не
засоряется промежуточными результатами.

Исследование кода — отдельный инструмент ``explore``: один вопрос вместо брифа и типа
агента. Модель охотнее берёт инструмент «ответь на вопрос о коде», чем «поручи подзадачу».

Здесь — описания встроенных агентов (данные, общие для инструментов и ядра) и сами
инструменты. Запускает суб-агента ядро (:mod:`gigacrew.agent.subagents`,
``Agent._run_subagent``) — инструмент получает его через ``ToolContext.run_subagent``.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import AliasChoices, BaseModel, Field

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
SUBAGENT_EXCLUDED: frozenset[str] = frozenset({"task", "explore", "ask_user", "exit_plan_mode"})

EXPLORE = SubagentSpec(
    name="explore",
    description=(
        "codebase research, read-only: search, code navigation, reading files, git "
        "status/diff/log. Returns a digest with paths and line numbers. For a research "
        "question prefer the explore tool"
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
    # Прежнее имя параметра — agent: так его пишут старые чаты и модели по привычке.
    subagent_type: str = Field(
        validation_alias=AliasChoices("subagent_type", "agent"),
        description=f"Subagent type: {' or '.join(SUBAGENTS)}.",
    )
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


class ExploreParams(BaseModel):
    question: str = Field(
        description=(
            "What to find out about the code, as one full question, e.g. 'How is a "
            "request authenticated: where are credentials checked and where is the "
            "session created?'"
        )
    )
    hints: str = Field(
        "",
        description=(
            "Optional: what you already know (paths, names, strings, errors) and what is "
            "out of scope."
        ),
    )


class ExploreTool(Tool):
    name = "explore"
    description = (
        "Investigates the codebase to answer a question and returns a short report with "
        "path:line references. A read-only subagent does the searching and reading in its "
        "own context, so none of it fills yours. Use it instead of searching yourself for "
        "any question that is not a pinpoint lookup: how something works, where something "
        "is handled or configured, which code a change will touch and what depends on it, "
        "how it is tested, an overview. Several independent questions — several explore "
        "calls in one response. Not for a known file, symbol or string: use read_file, "
        "find_symbol or search_content directly."
    )
    Params = ExploreParams

    def describe(self, params: ExploreParams) -> str:
        return " ".join(params.question.split())[:70]

    def run(self, params: ExploreParams, ctx: ToolContext) -> ToolResult:
        question = params.question.strip()
        if not question:
            raise ToolError("Пустой вопрос: опиши в параметре question, что выяснить.")
        if ctx.run_subagent is None:
            raise ToolError("суб-агенты здесь недоступны — выясни сам.")
        hints = params.hints.strip()
        brief = f"{question}\n\nAlready known / out of scope: {hints}" if hints else question
        return ctx.run_subagent(EXPLORE.name, task_title("", question), brief)


class TaskTool(Tool):
    name = "task"
    description = (
        "Hands a self-contained subtask to a subagent that works in its own context and "
        "returns only its final report. For research questions use the explore tool; "
        "task is mainly for a general subagent: a change that needs no decisions from the "
        "user (a fix with its test, a refactoring inside one module, one of several "
        f"independent subtasks — one subagent each, in one response). Subagent types: "
        f"{_agents_list()}. A subagent has limited steps and context: give it ONE narrow "
        "subtask and a complete brief — it does not see this conversation. Do not delegate "
        "what needs the user's decisions, and do not let it edit files you are changing "
        "yourself. A report of a subagent stopped by a limit is partial but useful: do not "
        "rerun the same task — narrow it to what remains. If the user stopped a subagent, "
        "do not run the same task again unless asked."
    )
    Params = TaskParams

    def describe(self, params: TaskParams) -> str:
        kind = params.subagent_type.strip()
        return f"{kind} · {task_title(params.description, params.prompt)}"[:70]

    def run(self, params: TaskParams, ctx: ToolContext) -> ToolResult:
        agent = params.subagent_type.strip().lower()
        if agent not in SUBAGENTS:
            raise ToolError(
                f"неизвестный суб-агент {params.subagent_type.strip()!r}; "
                f"доступны: {', '.join(SUBAGENTS)}."
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
