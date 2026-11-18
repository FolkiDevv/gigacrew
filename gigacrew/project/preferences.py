"""Настройки пользователя в JSON: ``~/.gigacode/gigacrew.json`` (меню ``/settings``).

Отдельный файл, а не ``~/.gigacode/settings.json``: тот общий с официальным CLI GigaCode, а
этот читает и пишет только gigacrew. Здесь — поведение агента (модель, разрешения,
контекст, суб-агенты, сеть, интерфейс); подключение и секреты (адреса API, ключ,
сертификаты) остаются в переменных окружения и ``.env``.

Формат — разделы с настройками; в файле только отличия от значений по умолчанию::

    {
        "version": 1,
        "model": {"default": "GigaChat-2-Max", "temperature": 0.3},
        "permissions": {"defaultMode": "plan"},
    }

У каждой настройки есть и переменная окружения (:attr:`SettingSpec.env`): переменные
окружения и ``.env`` главнее файла — так прежние ``.env`` работают, а CI задаёт своё.
Ошибка в файле не мешает запуску: настройка берёт значение по умолчанию, вызывающий
получает предупреждение.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

from gigacrew.permissions import MODE_CYCLE, parse_mode
from gigacrew.project.workspace import user_data_dir

__all__ = [
    "INVALID_SETTING",
    "PREFERENCES_FILE_NAME",
    "PREFERENCES_HINT",
    "SECTIONS",
    "SETTINGS",
    "Preferences",
    "Setting",
    "SettingKind",
    "SettingSpec",
    "load_preferences",
    "preferences_file",
    "store_setting",
    "write_text_atomic",
]

PREFERENCES_FILE_NAME = "gigacrew.json"
PREFERENCES_HINT = f"~/.gigacode/{PREFERENCES_FILE_NAME}"
FORMAT_VERSION = 1
_VERSION_KEY = "version"

# Значение настройки: флаг, число, строка; None — не задано (модель, бюджет контекста).
Setting = bool | int | float | str | None

_SECONDS = "с"
_PERCENT = "%"
_TOKENS = "ток."
_ON = "вкл"
_OFF = "выкл"
_TRUE_WORDS = frozenset(("1", "true", "yes", "on"))
_FALSE_WORDS = frozenset(("0", "false", "no", "off"))
_SHARE_MIN = 10
_SHARE_MAX = 95
_SHARE_DEFAULT = 80
_TEMPERATURE_DEFAULT = 0.2
_TEMPERATURE_MAX = 2.0
_CONTEXT_MIN = 4_000
_STEPS_DEFAULT = 50
_STEPS_EXTENSION = 25
_SUBAGENT_STEPS = 25
_REQUEST_WAIT = 120  # с: таймаут запросов, ожидание при 429
_LONG_WAIT = 600  # с: ответ модели, вложенная задача
_QUEUE_MAX = 16

# Неверное значение настройки или содержимое файла (текст ошибки — для пользователя).
INVALID_SETTING = (TypeError, ValueError)
# Нечитаемый файл; RecursionError — глубоко вложенный JSON ([[[…]]]).
_READ_ERRORS = (OSError, UnicodeDecodeError, ValueError, RecursionError)


class SettingKind(StrEnum):
    """Тип настройки: как она разбирается, показывается и меняется в меню."""

    FLAG = "flag"  # вкл/выкл
    INTEGER = "integer"
    NUMBER = "number"  # дробное
    CHOICE = "choice"  # одно из ``choices``
    MODE = "mode"  # режим разрешений (принимает и прежние имена: manual, edits)
    MODEL = "model"  # имя модели; None — не задана


@dataclass(frozen=True)
class SettingSpec:
    """Описание настройки: ключ в JSON, переменная окружения, поле Config, тип и границы."""

    key: str  # "раздел.имя" в gigacrew.json
    env: str  # переменная окружения (и строка .env), которая главнее файла
    field: str  # поле :class:`gigacrew.config.Config`
    label: str  # подпись в меню
    hint: str  # пояснение под меню
    kind: SettingKind
    default: Setting
    minimum: int | float | None = None
    maximum: int | float | None = None
    choices: tuple[str, ...] = ()
    unit: str = ""  # «с», «%», «ток.» — при показе
    empty: str = "не задана"  # как показать None
    restart: bool = False  # действует со следующего запуска

    @property
    def section(self) -> str:
        """Раздел JSON (``model`` у ``model.temperature``)."""
        return self.key.partition(".")[0]

    @property
    def name(self) -> str:
        """Имя внутри раздела (``temperature`` у ``model.temperature``)."""
        return self.key.partition(".")[2]

    @property
    def percent(self) -> bool:
        """Доля в процентах: в Config — дробь от 0 до 1."""
        return self.unit == _PERCENT

    def parse(self, raw: str) -> Setting:
        """Значение из текста (переменная окружения, ввод в меню); пусто — по умолчанию.

        Ошибка — :class:`ValueError` с текстом для пользователя (без имени настройки).
        """
        text = raw.strip()
        if not text:
            return self.default
        if self.kind is SettingKind.FLAG:
            return _parse_flag(text, raw)
        if self.kind is SettingKind.INTEGER:
            return self._bounded(_parse_integer(text, raw))
        if self.kind is SettingKind.NUMBER:
            return self._bounded(_parse_number(text, raw))
        if self.kind is SettingKind.MODE:
            return parse_mode(text).value
        if self.kind is SettingKind.CHOICE:
            return self._choice(text.lower())
        return text

    def check(self, raw: object) -> Setting:
        """Значение из JSON; неверный тип или границы — :class:`ValueError`."""
        if raw is None and self.default is None:
            return None
        if self.kind is SettingKind.FLAG:
            if isinstance(raw, bool):
                return raw
            raise ValueError(f"ожидается true или false, получено {raw!r}")
        if self.kind in {SettingKind.INTEGER, SettingKind.NUMBER}:
            return self._check_number(raw)
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"ожидается непустая строка, получено {raw!r}")
        return self.parse(raw)

    def show(self, setting: Setting) -> str:
        """Значение для показа: «вкл», «120 с», «80 %», «не задана»."""
        if setting is None:
            return self.empty
        if isinstance(setting, bool):
            return _ON if setting else _OFF
        if self.unit:
            return f"{setting} {self.unit}"
        return str(setting)

    def text(self, setting: Setting) -> str:
        """Значение для правки в строке ввода (None — пусто)."""
        if setting is None:
            return ""
        if isinstance(setting, bool):
            return "1" if setting else "0"
        return str(setting)

    def _check_number(self, raw: object) -> Setting:
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            raise TypeError(f"ожидается число, получено {raw!r}")
        if self.kind is SettingKind.INTEGER:
            if not isinstance(raw, int):
                raise TypeError(f"ожидается целое число, получено {raw!r}")
            return self._bounded(raw)
        return self._bounded(float(raw))

    def _bounded(self, number: int | float) -> int | float:
        if self.kind is SettingKind.NUMBER:
            low = float(self.minimum or 0)
            high = float(self.maximum or 0)
            if not low <= number <= high:
                raise ValueError(
                    f"значение должно быть в диапазоне [{low}; {high}], получено {number}",
                )
            return number
        if self.minimum is not None and number < self.minimum:
            raise ValueError(f"значение должно быть не меньше {self.minimum}, получено {number}")
        if self.maximum is not None and number > self.maximum:
            raise ValueError(f"значение должно быть не больше {self.maximum}, получено {number}")
        return number

    def _choice(self, text: str) -> str:
        if text not in self.choices:
            allowed = ", ".join(self.choices)
            raise ValueError(f"ожидается одно из {allowed}, получено {text!r}")
        return text


def _parse_flag(text: str, raw: str) -> bool:
    norm = text.lower()
    if norm in _TRUE_WORDS:
        return True
    if norm in _FALSE_WORDS:
        return False
    # Опечатка не должна молча выключать настройку.
    raise ValueError(f"ожидается 1/0, true/false, yes/no или on/off, получено {raw!r}")


def _parse_integer(text: str, raw: str) -> int:
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"ожидается целое число, получено {raw!r}") from exc


def _parse_number(text: str, raw: str) -> float:
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"ожидается число, получено {raw!r}") from exc


SECTIONS: Mapping[str, str] = MappingProxyType({
    "model": "Модель",
    "permissions": "Разрешения",
    "context": "Контекст",
    "agent": "Агент",
    "subagents": "Суб-агенты",
    "network": "Сеть",
    "tools": "Инструменты",
    "history": "История",
    "ui": "Интерфейс",
})

SETTINGS: tuple[SettingSpec, ...] = (
    SettingSpec(
        "model.default",
        "GIGACREW_MODEL",
        "model",
        "Модель по умолчанию",
        "Модель чата при запуске (-m перекрывает; в работе — /model). Не задана — у "
        "бэкенда GigaCode его модель по умолчанию.",
        SettingKind.MODEL,
        None,
        restart=True,
    ),
    SettingSpec(
        "model.temperature",
        "GIGACREW_TEMPERATURE",
        "temperature",
        "Температура",
        "Температура генерации, 0–2: чем выше, тем разнообразнее ответы.",
        SettingKind.NUMBER,
        _TEMPERATURE_DEFAULT,
        minimum=float(0),
        maximum=_TEMPERATURE_MAX,
    ),
    SettingSpec(
        "model.reasoning",
        "GIGACREW_REASONING",
        "reasoning",
        "Рассуждения модели",
        "turn — возвращать модели её рассуждения до конца хода (запрос растёт); off — нет.",
        SettingKind.CHOICE,
        "turn",
        choices=("turn", "off"),
    ),
    SettingSpec(
        "permissions.defaultMode",
        "GIGACREW_MODE",
        "mode",
        "Режим при запуске",
        "auto — правки без вопросов, изменяющие команды проверяет рецензент; plan — только "
        "исследование и план. В работе — Shift+Tab или /mode.",
        SettingKind.MODE,
        "auto",
        choices=tuple(mode.value for mode in MODE_CYCLE),
        restart=True,
    ),
    SettingSpec(
        "permissions.safeCommands",
        "GIGACREW_SAFE_COMMANDS",
        "safe_commands",
        "Безопасные команды",
        "ls, grep, git log, python без записи — без подтверждения; выкл — каждая команда "
        "с подтверждением.",
        SettingKind.FLAG,
        default=True,
    ),
    SettingSpec(
        "permissions.rememberApprovals",
        "GIGACREW_REMEMBER_APPROVALS",
        "remember_approvals",
        "Запоминать «всегда»",
        "Ответ «a» на вопрос о команде сохраняется для проекта (/permissions); выкл — "
        "только до конца сессии.",
        SettingKind.FLAG,
        default=True,
        restart=True,
    ),
    SettingSpec(
        "permissions.reviewerModel",
        "GIGACREW_REVIEWER_MODEL",
        "reviewer_model",
        "Модель-рецензент",
        "Проверяет изменяющие команды в режиме auto. Не задана — модель чата.",
        SettingKind.MODEL,
        None,
        empty="модель чата",
    ),
    SettingSpec(
        "permissions.reviewerTimeout",
        "GIGACREW_REVIEWER_TIMEOUT",
        "reviewer_timeout",
        "Ожидание рецензента",
        "Не ответил за это время — команду подтверждаете вы.",
        SettingKind.INTEGER,
        60,
        minimum=1,
        unit=_SECONDS,
    ),
    SettingSpec(
        "context.budgetTokens",
        "GIGACREW_CONTEXT_TOKENS",
        "context_budget_tokens",
        "Бюджет контекста",
        "Явный бюджет запроса в токенах (не меньше 4000). Не задан — от замеренного окна модели.",
        SettingKind.INTEGER,
        None,
        minimum=_CONTEXT_MIN,
        unit=_TOKENS,
        empty="по окну модели",
    ),
    SettingSpec(
        "context.autoCompact",
        "GIGACREW_AUTO_COMPACT",
        "auto_compact",
        "Автосжатие истории",
        "Начало истории сворачивается в краткое содержание; выкл — только /compact.",
        SettingKind.FLAG,
        default=True,
    ),
    SettingSpec(
        "context.compactThreshold",
        "GIGACREW_COMPACT_THRESHOLD",
        "compact_threshold",
        "Порог сжатия",
        "Сколько процентов бюджета истории занято, когда история сжимается (10–95).",
        SettingKind.INTEGER,
        _SHARE_DEFAULT,
        minimum=_SHARE_MIN,
        maximum=_SHARE_MAX,
        unit=_PERCENT,
    ),
    SettingSpec(
        "context.autoMeasure",
        "GIGACREW_AUTO_MEASURE",
        "auto_measure",
        "Предлагать замер окна",
        "Окно незамеренной модели предлагается замерить (пробы оплачиваются); выкл — только "
        "gigacrew --test-context.",
        SettingKind.FLAG,
        default=True,
    ),
    SettingSpec(
        "agent.maxSteps",
        "GIGACREW_MAX_STEPS",
        "max_steps",
        "Шагов за ход",
        "Предохранитель: сколько вызовов модели агент делает за один ход.",
        SettingKind.INTEGER,
        _STEPS_DEFAULT,
        minimum=1,
    ),
    SettingSpec(
        "agent.maxStepsExtension",
        "GIGACREW_MAX_STEPS_EXTENSION",
        "max_steps_extension",
        "Продление шагов",
        "На сколько шагов лимит продлевается один раз за ход, если агент в режиме auto "
        "что-то изменил; 0 — не продлевать.",
        SettingKind.INTEGER,
        _STEPS_EXTENSION,
        minimum=0,
    ),
    SettingSpec(
        "subagents.maxSteps",
        "GIGACREW_SUBAGENT_MAX_STEPS",
        "subagent_max_steps",
        "Шагов суб-агента",
        "Сколько шагов у одной вложенной задачи (task); на исчерпании — итог.",
        SettingKind.INTEGER,
        _SUBAGENT_STEPS,
        minimum=1,
    ),
    SettingSpec(
        "subagents.contextShare",
        "GIGACREW_SUBAGENT_CONTEXT",
        "subagent_context_share",
        "Контекст суб-агента",
        "Сколько процентов окна модели может занять контекст суб-агента (10–95).",
        SettingKind.INTEGER,
        _SHARE_DEFAULT,
        minimum=_SHARE_MIN,
        maximum=_SHARE_MAX,
        unit=_PERCENT,
    ),
    SettingSpec(
        "subagents.timeout",
        "GIGACREW_SUBAGENT_TIMEOUT",
        "subagent_timeout",
        "Время суб-агента",
        "Предел времени одной вложенной задачи (не меньше 10 с).",
        SettingKind.INTEGER,
        _LONG_WAIT,
        minimum=10,
        unit=_SECONDS,
    ),
    SettingSpec(
        "subagents.background",
        "GIGACREW_BACKGROUND_SUBAGENTS",
        "background_subagents",
        "Фоновых одновременно",
        "Сколько фоновых суб-агентов работают одновременно (1–8).",
        SettingKind.INTEGER,
        2,
        minimum=1,
        maximum=8,
        restart=True,
    ),
    SettingSpec(
        "subagents.queue",
        "GIGACREW_BACKGROUND_QUEUE",
        "background_queue",
        "Очередь фоновых",
        "Сколько фоновых суб-агентов могут ждать в очереди (0–16); сверх — отказ.",
        SettingKind.INTEGER,
        4,
        minimum=0,
        maximum=_QUEUE_MAX,
        restart=True,
    ),
    SettingSpec(
        "network.timeout",
        "GIGACREW_TIMEOUT",
        "timeout",
        "Таймаут запросов",
        "Подключение и короткие запросы к API.",
        SettingKind.INTEGER,
        _REQUEST_WAIT,
        minimum=1,
        unit=_SECONDS,
        restart=True,
    ),
    SettingSpec(
        "network.readTimeout",
        "GIGACREW_READ_TIMEOUT",
        "read_timeout",
        "Ожидание ответа модели",
        "Пока модель пишет длинный план или файл, данные не идут; не меньше таймаута запросов.",
        SettingKind.INTEGER,
        _LONG_WAIT,
        minimum=1,
        unit=_SECONDS,
        restart=True,
    ),
    SettingSpec(
        "network.rateLimitWait",
        "GIGACREW_RATE_LIMIT_WAIT",
        "rate_limit_wait",
        "Ожидание при 429",
        "Сколько ждать на один запрос, если шлюз ответил «лимит запросов»; 0 — сразу ошибка.",
        SettingKind.INTEGER,
        _REQUEST_WAIT,
        minimum=0,
        unit=_SECONDS,
        restart=True,
    ),
    SettingSpec(
        "tools.mcpDefer",
        "GIGACREW_MCP_DEFER",
        "mcp_defer",
        "Отложенные MCP-инструменты",
        "Модель видит только имена MCP-инструментов и загружает схемы через tool_search; "
        "выкл — все схемы в каждом запросе.",
        SettingKind.FLAG,
        default=True,
        restart=True,
    ),
    SettingSpec(
        "tools.ty",
        "GIGACREW_TY",
        "ty",
        "Навигация через ty",
        "Точная навигация по Python-коду (LSP-сервер ty); выкл — только индекс проекта.",
        SettingKind.FLAG,
        default=True,
        restart=True,
    ),
    SettingSpec(
        "history.saveChats",
        "GIGACREW_SAVE_CHATS",
        "save_chats",
        "Сохранять чаты",
        "Чаты и журнал решений о командах — в ~/.gigacode/projects/ (продолжение — /resume).",
        SettingKind.FLAG,
        default=True,
        restart=True,
    ),
    SettingSpec(
        "ui.stream",
        "GIGACREW_STREAM",
        "stream",
        "Потоковый вывод",
        "Ответ печатается по мере генерации; выкл — целиком в конце (как --no-stream).",
        SettingKind.FLAG,
        default=True,
    ),
    SettingSpec(
        "ui.color",
        "GIGACREW_COLOR",
        "color",
        "Цвет",
        "Цветной вывод; выкл — как --no-color (переменная NO_COLOR главнее).",
        SettingKind.FLAG,
        default=True,
        restart=True,
    ),
)

_BY_SECTION: Mapping[str, Mapping[str, SettingSpec]] = MappingProxyType({
    section: {spec.name: spec for spec in SETTINGS if spec.section == section}
    for section in SECTIONS
})


def preferences_file() -> Path:
    """``~/.gigacode/gigacrew.json`` — настройки пользователя для всех проектов."""
    return user_data_dir() / PREFERENCES_FILE_NAME


@dataclass(frozen=True)
class Preferences:
    """Настройки из файла: проверенные значения по ключу и предупреждения об ошибках."""

    settings: Mapping[str, Setting]  # ключ "раздел.имя" → значение
    warnings: tuple[str, ...] = ()


def load_preferences(path: Path | None = None) -> Preferences:
    """Читает файл настроек (по умолчанию :func:`preferences_file`).

    Нет файла — пусто. Нечитаемый файл, неизвестная или неверная настройка —
    предупреждение; такая настройка берёт значение по умолчанию.
    """
    path = path or preferences_file()
    try:
        raw = _read_raw(path)
    except INVALID_SETTING as error:
        return Preferences({}, (str(error),))
    settings: dict[str, Setting] = {}
    warnings: list[str] = []
    for section, entries in raw.items():
        if section == _VERSION_KEY:
            continue
        known = _BY_SECTION.get(section)
        if known is None or not isinstance(entries, dict):
            warnings.append(f"{path}: неизвестный раздел «{section}» пропущен")
            continue
        for name, raw_setting in entries.items():
            problem = _load_one(known.get(name), f"{section}.{name}", raw_setting, settings)
            if problem:
                warnings.append(f"{path}: {problem}")
    return Preferences(settings, tuple(warnings))


def _load_one(
    spec: SettingSpec | None,
    key: str,
    raw: object,
    settings: dict[str, Setting],
) -> str | None:
    """Проверяет настройку и кладёт её в ``settings``; ошибка — текст предупреждения."""
    if spec is None:
        return f"неизвестная настройка «{key}» пропущена"
    try:
        settings[key] = spec.check(raw)
    except INVALID_SETTING as error:
        return f"{key}: {error} — значение по умолчанию"
    return None


def store_setting(spec: SettingSpec, setting: Setting, path: Path | None = None) -> None:
    """Записывает настройку в файл; значение по умолчанию — убрать её из файла.

    Остальное содержимое файла сохраняется. Нечитаемый файл не перезаписывается —
    :data:`INVALID_SETTING`; ошибка записи — :class:`OSError`.
    """
    path = path or preferences_file()
    raw = _read_raw(path)
    section = raw.get(spec.section)
    entries: dict[str, object] = dict(section) if isinstance(section, dict) else {}
    if setting == spec.default:
        entries.pop(spec.name, None)
    else:
        entries[spec.name] = setting
    if entries:
        raw[spec.section] = entries
    else:
        raw.pop(spec.section, None)
    raw.pop(_VERSION_KEY, None)
    ordered = {_VERSION_KEY: FORMAT_VERSION, **raw}  # версия формата — первой строкой
    write_text_atomic(path, f"{json.dumps(ordered, ensure_ascii=False, indent=2)}\n")


def _read_raw(path: Path) -> dict[str, object]:
    """Содержимое файла как словарь; нет файла — пустой. Ошибка — :data:`INVALID_SETTING`."""
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except _READ_ERRORS as error:
        raise ValueError(f"не удалось прочитать {path}: {error}") from error
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: ожидается объект JSON с разделами настроек")
    return raw


def write_text_atomic(path: Path, text: str) -> None:
    """Запись через временный файл рядом: прерванная запись не портит настройки.

    Символическая ссылка (файл из репозитория dotfiles) остаётся ссылкой: пишется её цель.
    """
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out_file:
            out_file.write(text)
        Path(temp_name).replace(path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
