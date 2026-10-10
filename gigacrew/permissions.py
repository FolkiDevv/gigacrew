"""Режимы разрешений: что агенту можно делать без вопроса, что — с подтверждением,
а что запрещено.

Режим выбирает пользователь (Shift+Tab, ``/mode``, ``--mode``, ``GIGACREW_MODE``):

* **ручной** (``manual``) — каждая изменяющая операция подтверждается;
* **авто-правки** (``edits``) — правки файлов применяются без вопроса, команды
  подтверждаются;
* **план** (``plan``) — агент исследует и составляет план: правки файлов и
  изменяющий git заблокированы, команды shell — с подтверждением. План
  передаётся пользователю инструментом ``exit_plan_mode``.

Суб-агент «только для чтения» (``explore``) не меняет ничего ни в каком режиме и
ни с какими флагами.

Решение по конкретному вызову принимает :func:`decide` — единственное место, где
сводятся вместе режим, уровень риска операции, вид инструмента, ``-y``,
``--yes-all`` и «только чтение».
"""

from __future__ import annotations

from enum import Enum, StrEnum

from gigacrew.security import RiskLevel


class PermissionMode(StrEnum):
    MANUAL = "manual"
    ACCEPT_EDITS = "edits"
    PLAN = "plan"

    @property
    def label(self) -> str:
        return _LABELS[self]

    @property
    def description(self) -> str:
        return _DESCRIPTIONS[self]


_LABELS = {
    PermissionMode.MANUAL: "ручной",
    PermissionMode.ACCEPT_EDITS: "авто-правки",
    PermissionMode.PLAN: "план",
}
_DESCRIPTIONS = {
    PermissionMode.MANUAL: "каждая правка и команда — с подтверждением",
    PermissionMode.ACCEPT_EDITS: "правки файлов без вопросов, команды — с подтверждением",
    PermissionMode.PLAN: "только исследование и план: правки заблокированы, "
    "команды — с подтверждением",
}

# Порядок переключения по Shift+Tab.
MODE_CYCLE: tuple[PermissionMode, ...] = (
    PermissionMode.MANUAL,
    PermissionMode.ACCEPT_EDITS,
    PermissionMode.PLAN,
)


def next_mode(mode: PermissionMode) -> PermissionMode:
    """Следующий режим по кругу (Shift+Tab)."""
    return MODE_CYCLE[(MODE_CYCLE.index(mode) + 1) % len(MODE_CYCLE)]


def parse_mode(text: str) -> PermissionMode:
    """Режим по имени (``manual``/``edits``/``plan``, без учёта регистра)."""
    value = text.strip().lower()
    for mode in PermissionMode:
        if value == mode.value:
            return mode
    names = ", ".join(mode.value for mode in PermissionMode)
    raise ValueError(f"неизвестный режим {text.strip()!r}; допустимые: {names}")


class ToolKind(StrEnum):
    """Вид инструмента — от него зависит поведение в режимах."""

    OTHER = "other"
    EDIT = "edit"  # правка файлов: write_file, edit_file
    COMMAND = "command"  # произвольные команды: run_shell


class Decision(Enum):
    ALLOW = "allow"  # выполнить без вопроса
    ASK = "ask"  # спросить пользователя
    BLOCK = "block"  # не выполнять (режим планирования)


def decide(
    mode: PermissionMode,
    risk: RiskLevel,
    kind: ToolKind,
    *,
    auto_approve: bool,
    yes_all: bool = False,
    read_only: bool = False,
) -> Decision:
    """Как поступить с вызовом инструмента.

    Чтение разрешено всегда. В режиме плана изменения запрещены даже с ``-y``;
    исключение — команды shell: они нужны для исследования (тесты, просмотр) и
    подтверждаются, как в ручном режиме. ``-y`` (``auto_approve``) снимает вопросы,
    кроме опасных операций (``rm -rf``, ``git reset --hard``…) — их без вопроса
    выполняет только явный ``--yes-all``. ``read_only`` (суб-агент для исследования)
    блокирует любые изменения, включая команды shell, независимо от режима и флагов.
    """
    if risk < RiskLevel.WRITE:
        return Decision.ALLOW
    if read_only:
        return Decision.BLOCK
    if mode is PermissionMode.PLAN and kind is not ToolKind.COMMAND:
        return Decision.BLOCK
    if auto_approve and (risk < RiskLevel.DANGEROUS or yes_all):
        return Decision.ALLOW
    if mode is PermissionMode.ACCEPT_EDITS and kind is ToolKind.EDIT:
        return Decision.ALLOW
    return Decision.ASK
