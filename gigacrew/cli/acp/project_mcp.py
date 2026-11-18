"""MCP-серверы проекта сессии ACP: запускаются один раз — сразу или при первом ходе.

Сессию, загруженную лишь ради истории (снимок другого треда), клиент может и не закрыть:
её серверы запускаются, только когда она начинает работать.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from importlib import metadata

from gigacrew.cli.mcp import connect_trusted_mcp
from gigacrew.project.workspace import Workspace
from gigacrew.tools.base import ToolRegistry
from gigacrew.tools.mcp import McpHub

__all__ = ["ProjectMcp"]


class ProjectMcp:
    """Подтверждённые раньше MCP-серверы проекта; их инструменты — в реестре агента."""

    def __init__(
        self,
        workspace: Workspace,
        registry: ToolRegistry,
        warn: Callable[[str], None],
    ) -> None:
        """Серверы проекта ``workspace``; :meth:`start` регистрирует их в ``registry``."""
        self._workspace = workspace
        self._registry = registry
        self._warn = warn
        self._lock = threading.Lock()
        self._hub: McpHub | None = None
        self._closed = False

    def start(self) -> None:
        """Запускает серверы, если они ещё не запущены и не закрыты."""
        with self._lock:
            if self._hub is None and not self._closed:
                self._hub = connect_trusted_mcp(
                    self._workspace,
                    self._registry,
                    warn=self._warn,
                    version=metadata.version("gigacrew"),
                )

    def close(self) -> None:
        """Останавливает запущенные серверы; запустить их потом уже нельзя."""
        with self._lock:
            self._closed = True
            hub = self._hub
            self._hub = None
        if hub is not None:
            hub.close()
