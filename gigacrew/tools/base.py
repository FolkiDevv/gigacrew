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
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

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

# Сколько символов команды показывает в UI описание вызова по умолчанию.
_DESCRIBE_LIMIT = 70
# Нет такого поля в параметрах (None — тоже значение поля).
_MISSING: Final = object()


@dataclass
class PlanFile:
    """Файл плана текущего чата (режим планирования): ``.gigacode/plans/<id>.md``.

    Агент пишет его обычными write_file/edit_file — единственный файл в
    ``.gigacode/``, доступный для записи. ``dirty`` — план менялся после того, как
    его в последний раз отдали на одобрение (:func:`~gigacrew.tools.plan.review_plan`);
    ``reviews`` — сколько раз отдавали: по нему агент видит, показан ли план в этом ходе;
    ``writes`` — сколько раз его записывали (для ограничителя повторов: план изменился).
    """

    path: Path
    dirty: bool = False
    reviews: int = 0
    writes: int = 0

    def is_target(self, path: Path) -> bool:
        """Указывает ли (уже разрешённый) путь на этот файл плана.

        Файл плана, папка планов или служебная папка над ней — символическая ссылка: не
        план. Иначе ссылка увела бы запись без вопроса в ``.git/config``, в файл чата или в
        каталог проекта.
        """
        guarded = (self.path, self.path.parent, self.path.parent.parent)
        if any(guarded_path.is_symlink() for guarded_path in guarded):
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
        """Корень проекта (``workspace.root``)."""
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
        """Текст результата для модели: при неуспехе — с пометкой «ОШИБКА»."""
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
        non_null = [variant for variant in variants if variant.get("type") != "null"]
        chosen = non_null[0] if non_null else variants[0]
        merged = dict(prop.items())
        merged.update(chosen)
        merged.pop("title", None)
        prop = _normalize_property(merged, defs) if "$ref" in merged else merged

    # Рекурсивно нормализуем элементы массива и свойства вложенных объектов
    if prop.get("type") == "array" and isinstance(prop.get("items"), dict):
        prop["items"] = _normalize_property(prop["items"], defs)
    if prop.get("type") == "object" and isinstance(prop.get("properties"), dict):
        prop["properties"] = {
            field_name: _normalize_property(field_schema, defs)
            for field_name, field_schema in prop["properties"].items()
        }
    return prop


def _clean_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON-schema параметров в формате function calling API."""
    schema = model.model_json_schema()
    schema.pop("title", None)
    # Docstring модели параметров — для разработчика, а не для модели: у инструмента
    # своё описание (Tool.description).
    schema.pop("description", None)
    defs = schema.pop("$defs", None) or {}
    props = schema.get("properties", {})
    schema["properties"] = {
        field_name: _normalize_property(field_schema, defs)
        for field_name, field_schema in props.items()
    }
    schema.setdefault("type", "object")
    return schema


class Tool[ParamsT: BaseModel](ABC):
    """Инструмент агента с моделью параметров ``ParamsT`` (см. описание модуля)."""

    name: str = ""
    description: str = ""
    # Модель параметров (класс pydantic), а не константа: имя — часть API инструментов.
    Params: type[ParamsT]  # noqa: WPS115
    # Вид инструмента для режимов разрешений (permissions.decide): правки файлов
    # (EDIT) применяются без вопроса в режиме «авто», команды (COMMAND) в нём смотрит
    # модель-рецензент и они разрешены в режиме плана; остальное изменяющее в плане
    # заблокировано.
    kind: ToolKind = ToolKind.OTHER
    # Только смотрит проект (читает, ищет): интерфейс сворачивает подряд идущие такие
    # вызовы в одну строку.
    read_only: bool = False

    # ------------------------------------------------------------------ #
    def spec(self) -> ToolSpec:
        """Спецификация инструмента для function calling API."""
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=_clean_schema(self.Params),
        )

    def parse(self, arguments: dict[str, Any]) -> ParamsT:
        """Проверяет аргументы вызова от модели и строит из них параметры."""
        return self.Params.model_validate(arguments or {})

    def describe(self, params: ParamsT) -> str:
        """Краткое описание вызова для UI (путь, команда, шаблон...).

        Реализация по умолчанию берёт самое информативное из типичных полей;
        инструменты с особыми параметрами переопределяют метод.
        """
        fields = params.model_dump()
        path = fields.get("path", _MISSING)
        if path is not _MISSING and path not in {None, "", "."}:
            return str(path)
        command = fields.get("command", _MISSING)
        if command is not _MISSING:
            return str(command)[:_DESCRIBE_LIMIT]
        pattern = fields.get("pattern", _MISSING)
        if pattern is not _MISSING:
            return f"/{pattern}/" if "glob" in fields else str(pattern)
        if path is not _MISSING:
            return str(path)
        return ""

    def paths(self, params: ParamsT) -> tuple[str, ...]:
        """Пути проекта, с которыми работает вызов (для инструкций подкаталогов).

        По умолчанию — поле ``path``, если оно есть (файл или каталог).
        """
        path = getattr(params, "path", None)
        return (path,) if isinstance(path, str) and path.strip() else ()

    def risk(self, params: ParamsT, ctx: ToolContext) -> RiskLevel:
        """Уровень риска по умолчанию. Переопределяется инструментами."""
        return RiskLevel.SAFE

    def assess(self, params: ParamsT, ctx: ToolContext) -> Assessment:
        """Оценка вызова для решения «выполнить / спросить / заблокировать».

        По умолчанию — только уровень риска (:meth:`risk`); инструменты с подробной
        оценкой (run_shell) переопределяют.
        """
        return Assessment(self.risk(params, ctx))

    # Необязательный хук: по умолчанию превью нет (возвращает None), а не абстрактный метод.
    def preview(self, params: ParamsT, ctx: ToolContext) -> Display | None:  # noqa: B027
        """Что показать перед подтверждением (дифф, команда). None — нечего показывать.

        Вызывается только для операций, требующих подтверждения. Ошибка превью
        (ToolError) означает, что операция невыполнима: она не будет запущена.
        """

    @abstractmethod
    def run(self, params: ParamsT, ctx: ToolContext) -> ToolResult:
        """Выполняет вызов и возвращает результат (ошибка — :class:`ToolError`)."""


class ToolRegistry:
    """Реестр инструментов: хранение, спецификации, диспетчеризация вызовов."""

    def __init__(self) -> None:
        """Создаёт пустой реестр."""
        self._tools: dict[str, Tool[Any]] = {}

    def register(self, tool: Tool[Any]) -> None:
        """Добавляет инструмент (с тем же именем — заменяет). Без имени — ValueError."""
        if not tool.name:
            raise ValueError("Инструмент без имени")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool[Any] | None:
        """Инструмент по имени; None — такого нет."""
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        """Есть ли инструмент с таким именем."""
        return name in self._tools

    def __iter__(self) -> Iterator[Tool[Any]]:
        """Инструменты в порядке регистрации."""
        return iter(self._tools.values())

    def specs(self) -> list[ToolSpec]:
        """Спецификации всех инструментов (в порядке регистрации)."""
        return [tool.spec() for tool in self._tools.values()]

    def subset(self, keep: Callable[[Tool[Any]], bool]) -> ToolRegistry:
        """Новый реестр из инструментов, для которых ``keep`` истинно (порядок сохраняется)."""
        reg = ToolRegistry()
        for tool in self._tools.values():
            if keep(tool):
                reg.register(tool)
        return reg


def build_default_registry() -> ToolRegistry:
    """Собирает реестр со всеми штатными инструментами."""
    # Импорт здесь, чтобы избежать циклов: модули инструментов сами импортируют base.
    from gigacrew.tools import (  # noqa: PLC0415
        ask_user,
        fs,
        git,
        index,
        navigation,
        plan,
        search,
        shell,
    )

    # Тоже отложенный импорт: цикл через base.
    from gigacrew.tools.skill import SkillTool  # noqa: PLC0415

    # Тоже отложенный импорт: цикл через base.
    from gigacrew.tools.task import ExploreTool, TaskTool  # noqa: PLC0415

    reg = ToolRegistry()
    for tool in (
        # Первым: исследование через суб-агента — выбор по умолчанию для вопросов о коде.
        ExploreTool(),
        fs.ReadFileTool(),
        fs.WriteFileTool(),
        fs.EditFileTool(),
        fs.ListDirTool(),
        fs.FindFilesTool(),
        search.SearchContentTool(),
        index.FindSymbolTool(),
        navigation.FindReferencesTool(),
        navigation.GotoDefinitionTool(),
        navigation.CallHierarchyTool(),
        index.FileOutlineTool(),
        index.RepoMapTool(),
        shell.RunShellTool(),
        git.GitTool(),
        SkillTool(),
        ask_user.AskUserTool(),
        plan.ExitPlanModeTool(),
        TaskTool(),
    ):
        reg.register(tool)
    return reg
