"""Базовые абстракции для инструментов агента.

Каждый инструмент:
  * объявляет имя, описание и pydantic-модель параметров (Params);
  * из модели автоматически строится JSON-schema для function calling;
  * объявляет уровень риска (для модели прав);
  * кратко описывает вызов для UI (``describe``) и называет пути проекта, с
    которыми работает (``paths`` — по ним подключаются инструкции подкаталогов);
  * умеет (опционально) показать превью (например, дифф) перед выполнением;
  * выполняется методом ``run`` и возвращает ``ToolResult``.

Как показывать результат, решает сам инструмент (через :class:`Display`), а не
агентный цикл — поэтому новый инструмент не требует правок в ядре и UI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from gigacrew.errors import ToolError
from gigacrew.llm.types import ToolSpec
from gigacrew.permissions import PermissionMode, ToolKind
from gigacrew.project.workspace import Workspace
from gigacrew.safety.levels import Verdict
from gigacrew.security import RiskLevel
from gigacrew.tools.questions import AskUser

__all__ = [
    "Assessment",
    "Display",
    "PlanFile",
    "Tool",
    "ToolContext",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "build_default_registry",
]


@dataclass
class PlanFile:
    """Файл плана текущего чата (режим планирования): ``.gigacode/plans/<id>.md``.

    Агент пишет его обычными write_file/edit_file — единственный файл в
    ``.gigacode/``, доступный для записи. ``dirty`` — план менялся после того, как
    его в последний раз отдали на одобрение (:func:`~gigacrew.tools.plan.review_plan`);
    ``reviews`` — сколько раз отдавали: по нему агент видит, показан ли план в этом ходе.
    """

    path: Path
    dirty: bool = False
    reviews: int = 0

    def is_target(self, path: Path) -> bool:
        """Указывает ли (уже разрешённый) путь на этот файл плана.

        Файл плана или папка планов — символическая ссылка: не план. Иначе ссылка
        увела бы запись без вопроса в ``.git/config`` или в файл чата.
        """
        if self.path.is_symlink() or self.path.parent.is_symlink():
            return False
        return Path(path).resolve() == self.path.resolve()


@dataclass(frozen=True)
class ToolContext:
    """Контекст выполнения, доступный инструменту.

    Намеренно не содержит Config: инструментам не нужны (и не должны быть
    доступны) реквизиты API. ``ask_user`` — способ задать вопрос пользователю
    (None — спросить некого). ``get_mode``/``set_mode`` — текущий режим разрешений
    агента (None — агента нет, например в тестах инструмента). ``semantic`` —
    разрешена ли точная навигация через ty (``GIGACREW_TY``). ``run_subagent`` —
    запуск суб-агента (инструмент ``task``): ``(агент, описание, задача) → отчёт``;
    None — запускать некому (в том числе у самого суб-агента: без вложенности).
    ``plan`` — файл плана чата (None — плана нет: суб-агент, тесты инструментов).
    ``classify_commands`` — команды, которые классификатор признал «только чтение»,
    выполняются без вопроса (``GIGACREW_SAFE_COMMANDS``).
    """

    workspace: Workspace
    ask_user: AskUser | None = None
    get_mode: Callable[[], PermissionMode] | None = None
    set_mode: Callable[[PermissionMode], None] | None = None
    semantic: bool = True
    run_subagent: Callable[[str, str, str], ToolResult] | None = None
    plan: PlanFile | None = None
    classify_commands: bool = True

    @property
    def root(self) -> Path:
        return self.workspace.root


@dataclass(frozen=True)
class Display:
    """Дополнительный вывод для пользователя (не уходит в модель).

    ``kind="diff"`` — unified diff (подсвечивается как дифф), ``"text"`` — обычный
    вывод (команды, git). ``title`` — заголовок блока.
    """

    text: str
    kind: Literal["diff", "text"] = "text"
    title: str = ""


@dataclass(frozen=True)
class Assessment:
    """Оценка вызова перед выполнением: уровень риска и подробности для вопроса.

    ``reason`` — почему (показывается пользователю при подтверждении); ``kind`` — вид
    вызова для режимов, если он зависит от аргументов (команда, которая только правит
    файлы проекта, — как правка: в режиме «авто» не спрашивается); ``verdict`` —
    оценка команды оболочки (для памяти решений и журнала).
    """

    risk: RiskLevel
    reason: str = ""
    kind: ToolKind | None = None
    verdict: Verdict | None = None
    purpose: str = ""  # зачем вызов, со слов модели


@dataclass
class ToolResult:
    """Результат выполнения инструмента."""

    content: str  # текст, который вернётся модели
    ok: bool = True
    summary: str = ""  # краткая строка для UI
    display: Display | None = None  # доп. вывод для пользователя (дифф/листинг)
    # Неуспех — обычный исход работы (команда вернула ненулевой код), а не сбой
    # инструмента: не приближает остановку хода по серии ошибок.
    soft: bool = False
    # Вызов изменил проект. Агентный цикл ставит сам для успешных вызовов с риском
    # ≥ WRITE; инструмент с риском ниже ставит, если изменения сделал не он сам
    # (суб-агент правил файлы): для ограничителя зацикливания это изменение.
    changed: bool = False
    exit_code: int | None = None  # код возврата команды (для журнала решений)

    def as_function_content(self) -> str:
        if self.ok:
            return self.content
        return f"ОШИБКА: {self.content}"


def _normalize_property(prop: dict[str, Any], defs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Приводит описание свойства к формату, который принимает API.

    API не понимает union-типы (``anyOf``/``oneOf``), которые pydantic
    генерирует для ``Optional[...]``. Сворачиваем их в одиночный ``type``,
    отбрасывая ветку ``null`` (необязательность отражается отсутствием в
    ``required``). Также убираем служебные ключи (title/default). Ссылки на
    вложенные модели (``$ref`` в ``$defs``) подставляются на место.
    """
    defs = defs or {}
    prop = dict(prop)
    ref = prop.pop("$ref", None)
    if ref:
        prop = {**defs[ref.rsplit("/", 1)[-1]], **prop}  # описание поля важнее описания модели
    prop.pop("title", None)
    prop.pop("default", None)

    variants = prop.pop("anyOf", None) or prop.pop("oneOf", None)
    if variants:
        non_null = [v for v in variants if v.get("type") != "null"]
        chosen = non_null[0] if non_null else variants[0]
        merged = {k: v for k, v in prop.items()}
        merged.update(chosen)
        merged.pop("title", None)
        prop = _normalize_property(merged, defs) if "$ref" in merged else merged

    # Рекурсивно нормализуем элементы массива и свойства вложенных объектов
    if prop.get("type") == "array" and isinstance(prop.get("items"), dict):
        prop["items"] = _normalize_property(prop["items"], defs)
    if prop.get("type") == "object" and isinstance(prop.get("properties"), dict):
        prop["properties"] = {
            k: _normalize_property(v, defs) for k, v in prop["properties"].items()
        }
    return prop


def _clean_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON-schema параметров в формате function calling API."""
    schema = model.model_json_schema()
    schema.pop("title", None)
    defs = schema.pop("$defs", None) or {}
    props = schema.get("properties", {})
    schema["properties"] = {k: _normalize_property(v, defs) for k, v in props.items()}
    schema.setdefault("type", "object")
    return schema


class Tool[P: BaseModel](ABC):
    name: str = ""
    description: str = ""
    Params: type[BaseModel] = BaseModel
    # Вид инструмента для режимов разрешений (permissions.decide): правки файлов
    # (EDIT) применяются без вопроса в режиме «авто», команды (COMMAND) в нём смотрит
    # модель-рецензент и они разрешены в режиме плана; остальное изменяющее в плане
    # заблокировано.
    kind: ToolKind = ToolKind.OTHER

    # ------------------------------------------------------------------ #
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=_clean_schema(self.Params),
        )

    def parse(self, arguments: dict[str, Any]) -> BaseModel:
        return self.Params.model_validate(arguments or {})

    def describe(self, params: BaseModel) -> str:
        """Краткое описание вызова для UI (путь, команда, шаблон...).

        Реализация по умолчанию берёт самое информативное из типичных полей;
        инструменты с особыми параметрами переопределяют метод.
        """
        d = params.model_dump()
        if "path" in d and d["path"] not in (None, "", "."):
            return str(d["path"])
        if "command" in d:
            return str(d["command"])[:70]
        if "pattern" in d:
            return f"/{d['pattern']}/" if "glob" in d else str(d["pattern"])
        if "path" in d:
            return str(d["path"])
        return ""

    def paths(self, params: BaseModel) -> tuple[str, ...]:
        """Пути проекта, с которыми работает вызов (для инструкций подкаталогов).

        По умолчанию — поле ``path``, если оно есть (файл или каталог).
        """
        path = getattr(params, "path", None)
        return (path,) if isinstance(path, str) and path.strip() else ()

    def risk(self, params: BaseModel, ctx: ToolContext) -> RiskLevel:  # noqa: ARG002
        """Уровень риска по умолчанию. Переопределяется инструментами."""
        return RiskLevel.SAFE

    def assess(self, params: BaseModel, ctx: ToolContext) -> Assessment:
        """Оценка вызова для решения «выполнить / спросить / заблокировать».

        По умолчанию — только уровень риска (:meth:`risk`); инструменты с подробной
        оценкой (run_shell) переопределяют.
        """
        return Assessment(self.risk(params, ctx))

    def preview(self, params: BaseModel, ctx: ToolContext) -> Display | None:  # noqa: ARG002
        """Что показать перед подтверждением (дифф, команда). None — нечего показывать.

        Вызывается только для операций, требующих подтверждения. Ошибка превью
        (ToolError) означает, что операция невыполнима: она не будет запущена.
        """
        return None

    @abstractmethod
    def run(self, params: BaseModel, ctx: ToolContext) -> ToolResult: ...


class ToolRegistry:
    """Реестр инструментов: хранение, спецификации, диспетчеризация вызовов."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError("Инструмент без имени")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values()]

    def subset(self, keep: Callable[[Tool], bool]) -> ToolRegistry:
        """Новый реестр из инструментов, для которых ``keep`` истинно (порядок сохраняется)."""
        reg = ToolRegistry()
        for tool in self._tools.values():
            if keep(tool):
                reg.register(tool)
        return reg


def build_default_registry() -> ToolRegistry:
    """Собирает реестр со всеми штатными инструментами."""
    # Импорт здесь, чтобы избежать циклов.
    from gigacrew.tools.ask_user import AskUserTool
    from gigacrew.tools.fs import (
        EditFileTool,
        FindFilesTool,
        ListDirTool,
        ReadFileTool,
        WriteFileTool,
    )
    from gigacrew.tools.git import GitTool
    from gigacrew.tools.index import FileOutlineTool, FindSymbolTool, RepoMapTool
    from gigacrew.tools.navigation import (
        CallHierarchyTool,
        FindReferencesTool,
        GotoDefinitionTool,
    )
    from gigacrew.tools.plan import ExitPlanModeTool
    from gigacrew.tools.search import SearchContentTool
    from gigacrew.tools.shell import RunShellTool
    from gigacrew.tools.skill import SkillTool
    from gigacrew.tools.task import ExploreTool, TaskTool

    reg = ToolRegistry()
    for tool in (
        # Первым: исследование через суб-агента — выбор по умолчанию для вопросов о коде.
        ExploreTool(),
        ReadFileTool(),
        WriteFileTool(),
        EditFileTool(),
        ListDirTool(),
        FindFilesTool(),
        SearchContentTool(),
        FindSymbolTool(),
        FindReferencesTool(),
        GotoDefinitionTool(),
        CallHierarchyTool(),
        FileOutlineTool(),
        RepoMapTool(),
        RunShellTool(),
        GitTool(),
        SkillTool(),
        AskUserTool(),
        ExitPlanModeTool(),
        TaskTool(),
    ):
        reg.register(tool)
    return reg
