"""Чат сессии ACP: новый или продолженный сохранённый, его сохранение и история.

Id сессии — id чата хранилища проекта (:class:`~gigacrew.agent.chat_store.ChatStore`).
Чат сохраняется после каждого хода, как в REPL; продолженный чат переигрывается клиенту
уведомлениями (:mod:`gigacrew.cli.acp.history`).
"""

from __future__ import annotations

import logging

from gigacrew.agent.chat_store import ChatRecorder, ChatStore, ChatStoreError, SavedChat
from gigacrew.agent.loop import Agent
from gigacrew.cli.acp.events import AcpEvents
from gigacrew.cli.acp.history import history_updates
from gigacrew.cli.acp.rpc import INTERNAL_ERROR, RESOURCE_NOT_FOUND, RpcError
from gigacrew.cli.commands import resume_chat, store_chat
from gigacrew.project.workspace import Workspace

__all__ = ["autosave", "replay_history", "saved_chat", "start_chat"]

_log = logging.getLogger(__name__)


def saved_chat(workspace: Workspace, chat_id: str) -> SavedChat:
    """Сохранённый чат проекта: нет его (или ``chat_id`` — не id чата) — «Resource not found».

    Файл чата не читается или повреждён — «Internal error».
    """
    store = ChatStore(workspace)
    try:
        found = store.exists(chat_id)
    except ChatStoreError as error:
        raise RpcError(RESOURCE_NOT_FOUND, str(error)) from None
    if not found:
        raise RpcError(RESOURCE_NOT_FOUND, f"чат {chat_id} не найден")
    try:
        return store.load(chat_id)
    except ChatStoreError as error:
        raise RpcError(INTERNAL_ERROR, str(error)) from None


def start_chat(
    agent: Agent,
    events: AcpEvents,
    saved: SavedChat | None,
    *,
    enabled: bool,
) -> ChatRecorder:
    """Чат сессии — новый или продолженный ``saved``; хуки SessionStart.

    ``enabled`` — сохранять ли чат (``GIGACREW_SAVE_CHATS``). Модель продолженного чата
    становится текущей, если она есть в каталоге бэкенда.
    """
    chats = ChatRecorder(ChatStore(agent.workspace), enabled=enabled)
    if saved is None:
        agent.chat_id = chats.chat_id  # файл плана назван по id чата
        source = "startup"
    else:
        skipped = resume_chat(agent, chats, saved)
        if skipped:
            _warn(f"модели чата {skipped} нет в каталоге бэкенда — модель: {agent.model}")
        source = "resume"
    events.bind(chats.chat_id)
    agent.start_session(source)
    return chats


def autosave(chats: ChatRecorder, agent: Agent) -> None:
    """Сохраняет чат после хода; предупреждения — в журнал."""
    warning = store_chat(chats, agent)
    if warning:
        _warn(warning)


def replay_history(agent: Agent, events: AcpEvents) -> None:
    """Переигрывает клиенту историю чата агента."""
    messages = agent.conversation.messages
    events.report_history(history_updates(messages, agent.call_info))


def _warn(text: str) -> None:
    _log.warning("%s", text)
