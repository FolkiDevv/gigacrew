"""Режим и модель сессии ACP: ``modes``, ``configOptions`` и методы, которые их меняют.

Режимы ACP — :class:`~gigacrew.permissions.PermissionMode` (``session/set_mode``). Модель —
параметр ``configOptions`` с id ``model`` (категория ``model``, вид ``select``):
варианты — каталог бэкенда, без каталога — настроенная и текущая модели, а выбрать можно
любую, как в CLI. В ``configOptions`` нет параметра категории ``mode``: найдя его, T3
переключал бы режим через ``session/set_config_option``, а не ``session/set_mode``.
"""

from __future__ import annotations

from typing import Any, Final, Literal

from pydantic import BaseModel, Field

from gigacrew.agent.loop import Agent
from gigacrew.cli.acp.rpc import INVALID_PARAMS, RpcError
from gigacrew.cli.assembly import NOT_IN_CATALOGUE
from gigacrew.cli.commands import catalogue_ids
from gigacrew.permissions import PermissionMode

__all__ = [
    "SetConfigOptionParams",
    "SetModeParams",
    "SetModelParams",
    "choose_model",
    "config_options",
    "session_setup",
]

# id параметра сессии с выбором модели.
_MODEL: Final = "model"

type Message = dict[str, Any]


class SetModeParams(BaseModel):
    """Параметры ``session/set_mode``: режим — значение :class:`PermissionMode`."""

    session_id: str = Field(alias="sessionId")
    mode_id: PermissionMode = Field(alias="modeId")


class SetConfigOptionParams(BaseModel):
    """Параметры ``session/set_config_option``: параметр есть один — модель."""

    session_id: str = Field(alias="sessionId")
    config_id: Literal["model"] = Field(alias="configId")
    choice: str = Field(alias="value", min_length=1)


class SetModelParams(BaseModel):
    """Параметры ``session/set_model``."""

    session_id: str = Field(alias="sessionId")
    model_id: str = Field(alias="modelId", min_length=1)


def session_setup(agent: Agent) -> Message:
    """Поля ``modes`` и ``configOptions`` ответа ``session/new``."""
    return {
        "modes": {
            "currentModeId": agent.mode.value,
            "availableModes": [
                {"id": mode.value, "name": mode.label, "description": mode.description}
                for mode in PermissionMode
            ],
        },
        "configOptions": config_options(agent),
    }


def config_options(agent: Agent) -> list[Message]:
    """Параметры сессии (``configOptions``): выбор модели."""
    # Модель читается один раз: её могут сменить параллельно, а текущая — среди вариантов.
    model = agent.model
    return [
        {
            "id": _MODEL,
            "name": "Модель",
            "category": _MODEL,
            "type": "select",
            "currentValue": model,
            "options": [{"value": choice, "name": choice} for choice in _models(agent, model)],
        },
    ]


def choose_model(agent: Agent, model: str) -> None:
    """Модель агента со следующего обращения; модели нет в каталоге — «Invalid params»."""
    catalogue = catalogue_ids(agent.provider)
    if catalogue is not None and model not in catalogue:
        problem = NOT_IN_CATALOGUE.format(model=model, available=", ".join(catalogue))
        raise RpcError(INVALID_PARAMS, problem)
    try:
        agent.set_model(model)
    except ValueError as error:
        raise RpcError(INVALID_PARAMS, str(error)) from None


def _models(agent: Agent, current: str) -> list[str]:
    """Варианты модели: каталог бэкенда, без него — настроенная и текущая ``current``."""
    catalogue = catalogue_ids(agent.provider)
    if catalogue is not None:
        return catalogue
    return list(dict.fromkeys((agent.config.model, current)))
