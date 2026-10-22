"""Режимы разрешений: что агенту можно делать без вопроса, что — с подтверждением,
а что запрещено.

Режим выбирает пользователь (Shift+Tab, ``/mode``, ``--mode``, ``GIGACREW_MODE``):

* **авто** (``auto``, по умолчанию) — правки файлов (в том числе командами, которые
  только правят файлы проекта: ``sed -i``, форматтеры) применяются без вопроса;
  изменяющие команды сначала смотрит модель-рецензент (:mod:`gigacrew.agent.reviewer`):
  она разрешает команду или передаёт её пользователю. Опасные команды и остальные
  изменяющие операции подтверждает пользователь;
* **план** (``plan``) — агент исследует и составляет план: правки файлов, изменяющий
  git и команды, которые что-то меняют, заблокированы; проверки (тесты, сборка) и
  непроверенные команды — с подтверждением. План агент
  пишет в файл плана чата (``.gigacode/plans/<id>.md`` — его запись безопасна в
  любом режиме) и передаёт пользователю инструментом ``exit_plan_mode``.

Прежние имена ``manual`` и ``edits`` принимаются как синонимы ``auto``.

Команды, которые классификатор (:mod:`gigacrew.safety`) признал «только чтение»,
имеют риск SAFE и выполняются без вопроса в любом режиме. Суб-агент «только для
чтения» (``explore``) не меняет ничего ни в каком режиме и ни с какими флагами.

Решение по конкретному вызову принимает :func:`decide` — единственное место, где
сводятся вместе режим, уровень риска операции, вид инструмента, ``-y``,
``--yes-all``, «только чтение» и решение хука PreToolUse.
"""

from __future__ import annotations

from enum import Enum, StrEnum
from typing import Literal

from gigacrew.security import RiskLevel


class PermissionMode(StrEnum):
    AUTO = "auto"
    PLAN = "plan"

    @property
    def label(self) -> str:
        return _LABELS[self]

    @property
    def description(self) -> str:
        return _DESCRIPTIONS[self]


_LABELS = {
    PermissionMode.AUTO: "авто",
    PermissionMode.PLAN: "план",
}
_DESCRIPTIONS = {
    PermissionMode.AUTO: "правки файлов без вопросов; команды проверяет модель-рецензент, "
    "сомнительные и опасные — с подтверждением",
    PermissionMode.PLAN: "только исследование и план: правки и изменяющие команды "
    "заблокированы, проверки — с подтверждением",
}
# Прежние режимы «ручной» и «авто-правки» заменены режимом «авто».
_ALIASES = {"manual": PermissionMode.AUTO, "edits": PermissionMode.AUTO}

# Порядок переключения по Shift+Tab.
MODE_CYCLE: tuple[PermissionMode, ...] = (PermissionMode.AUTO, PermissionMode.PLAN)


def next_mode(mode: PermissionMode) -> PermissionMode:
    """Следующий режим по кругу (Shift+Tab)."""
    return MODE_CYCLE[(MODE_CYCLE.index(mode) + 1) % len(MODE_CYCLE)]


def parse_mode(text: str) -> PermissionMode:
    """Режим по имени (``auto``/``plan``, без учёта регистра; ``manual``/``edits`` —
    прежние имена ``auto``)."""
    value = text.strip().lower()
    for mode in PermissionMode:
        if value == mode.value:
            return mode
    if value in _ALIASES:
        return _ALIASES[value]
    names = ", ".join(mode.value for mode in PermissionMode)
    raise ValueError(f"неизвестный режим {text.strip()!r}; допустимые: {names}")


class ToolKind(StrEnum):
    """Вид инструмента — от него зависит поведение в режимах."""

    OTHER = "other"
    EDIT = "edit"  # правка файлов: write_file, edit_file
    COMMAND = "command"  # произвольные команды: run_shell


class Decision(Enum):
    ALLOW = "allow"  # выполнить без вопроса
    REVIEW = "review"  # спросить модель-рецензента (она разрешит или спросит пользователя)
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
    hook: Literal["allow", "ask"] | None = None,
) -> Decision:
    """Как поступить с вызовом инструмента.

    Чтение разрешено всегда. В режиме плана изменения запрещены даже с ``-y``;
    исключение — команды shell: они нужны для исследования (тесты, просмотр) и
    подтверждаются. ``-y`` (``auto_approve``) снимает вопросы, кроме опасных операций
    (``rm -rf``, ``git reset --hard``…) — их без вопроса выполняет только явный
    ``--yes-all``. В режиме «авто» правки выполняются без вопроса, а изменяющие (но не
    опасные) команды идут к модели-рецензенту. ``read_only`` (суб-агент для
    исследования) блокирует любые изменения, включая команды shell, независимо от
    режима и флагов.

    ``hook`` — решение хука PreToolUse: ``"ask"`` требует подтверждения пользователя
    даже там, где вызов выполнился бы без вопроса или решался бы рецензентом;
    ``"allow"`` снимает вопрос и рецензента, как ``-y``, — опасные операции всё равно
    спрашиваются. Запрет режима (BLOCK) хук не снимает.
    """
    decision = _decide(
        mode, risk, kind, auto_approve=auto_approve, yes_all=yes_all, read_only=read_only
    )
    return apply_hook(decision, risk, hook)


def apply_hook(
    decision: Decision, risk: RiskLevel, hook: Literal["allow", "ask"] | None
) -> Decision:
    """Решение с учётом хука PreToolUse (см. :func:`decide`). Отдельно — для агента: он
    сначала уточняет решение по вердикту классификатора (изменяющая команда в режиме
    плана — запрет; за команду, которую классификатор не проверил, ``-y`` не ручается),
    и только потом применяет хук, чтобы ``allow`` не снял запрет. За непроверенную
    команду не ручается и ``allow``."""
    if hook == "ask" and decision in (Decision.ALLOW, Decision.REVIEW):
        return Decision.ASK
    if (
        hook == "allow"
        and decision in (Decision.ASK, Decision.REVIEW)
        and risk < RiskLevel.DANGEROUS
    ):
        return Decision.ALLOW
    return decision


def _decide(
    mode: PermissionMode,
    risk: RiskLevel,
    kind: ToolKind,
    *,
    auto_approve: bool,
    yes_all: bool,
    read_only: bool,
) -> Decision:
    if risk < RiskLevel.WRITE:
        return Decision.ALLOW
    if read_only:
        return Decision.BLOCK
    if mode is PermissionMode.PLAN and kind is not ToolKind.COMMAND:
        return Decision.BLOCK
    if auto_approve and (risk < RiskLevel.DANGEROUS or yes_all):
        return Decision.ALLOW
    if mode is PermissionMode.AUTO and kind is ToolKind.EDIT:
        return Decision.ALLOW
    if mode is PermissionMode.AUTO and kind is ToolKind.COMMAND and risk < RiskLevel.DANGEROUS:
        return Decision.REVIEW
    return Decision.ASK
