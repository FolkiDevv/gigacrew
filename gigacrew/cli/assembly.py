"""Сборка агента из конфигурации и провайдера: общая часть CLI и ACP-сервера.

Сообщения не выводятся сами: предупреждения уходят в ``warn``, ошибка выбора модели —
исключение :class:`ModelChoiceError`. Так ими пользуются и консоль CLI, и ACP-сервер,
у которого терминала нет.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from gigacrew.agent.decisions import DecisionMemory
from gigacrew.agent.events import AgentEvents
from gigacrew.agent.hooks import HookRunner
from gigacrew.agent.loop import Agent
from gigacrew.agent.reviewer import check_reviewer
from gigacrew.config import Config
from gigacrew.llm.base import LLMProvider
from gigacrew.llm.model_windows import ModelWindows
from gigacrew.tools.base import ToolRegistry

__all__ = [
    "NOT_IN_CATALOGUE",
    "ModelChoiceError",
    "build_agent",
    "load_windows",
    "select_model",
]

NOT_IN_CATALOGUE = "модели {model} нет в каталоге бэкенда; доступны: {available}"

type Warn = Callable[[str], None]


class ModelChoiceError(Exception):
    """Модель не выбрать: она не задана или её нет в каталоге бэкенда."""


def load_windows(
    config: Config,
    provider: LLMProvider,
    warn: Warn,
) -> tuple[ModelWindows, list[str] | None]:
    """Окна контекста моделей и id моделей каталога бэкенда (None — каталога нет).

    Вызывается после ``provider.connect()``: каталог приходит при подключении. Окна из
    каталога сразу записываются в замеры.
    """
    catalogue = provider.known_models()
    known = None if catalogue is None else [entry.id for entry in catalogue]
    windows, warning = ModelWindows.load(base_url=config.base_url)
    if warning:
        warn(warning)
    if catalogue is not None:
        windows.report({entry.id: entry.context_window for entry in catalogue})
    return windows, known


def select_model(
    config: Config,
    provider: LLMProvider,
    known: list[str] | None,
    *,
    how_to_set: str = "GIGACREW_MODEL",
) -> Config:
    """Конфигурация с моделью из настроек или моделью бэкенда по умолчанию.

    ``known`` — модели каталога бэкенда (None — каталога нет, модель любая).
    ``how_to_set`` — чем задать модель (для сообщения об ошибке). Модели нет в каталоге
    или она не задана — :class:`ModelChoiceError`.
    """
    if known is not None:
        if not config.model:  # модель по умолчанию — бэкенда
            config = dataclasses.replace(config, model=provider.model)
        elif config.model not in known:
            raise ModelChoiceError(
                NOT_IN_CATALOGUE.format(model=config.model, available=", ".join(known)),
            )
    if not config.model:
        raise ModelChoiceError(f"не задана модель: укажите {how_to_set}")
    return config


def build_agent(
    config: Config,
    provider: LLMProvider,
    events: AgentEvents,
    *,
    windows: ModelWindows,
    hooks: HookRunner,
    registry: ToolRegistry,
    warn: Warn,
) -> Agent:
    """Агент с журналом решений о командах и моделью-рецензентом.

    ``config`` — уже с выбранной моделью (:func:`select_model`); ``registry`` — инструменты
    агента.
    """
    decisions, warning = DecisionMemory.load(
        config.project_root,
        remember=config.remember_approvals,
        log=config.save_chats,
    )
    if warning:
        warn(warning)
    reviewer, warning = check_reviewer(provider, config.reviewer_model)
    if warning:
        warn(warning)
    return Agent(
        provider,
        registry,
        config,
        events,
        windows=windows,
        decisions=decisions,
        reviewer=reviewer,
        hooks=hooks,
    )
