"""Раскрытие ``${VAR}`` и ``${VAR:-по умолчанию}`` в записях MCP-серверов.

Окружение передаёт вызывающий — без секретов gigacrew (``security.subprocess_env``).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from functools import partial

from gigacrew.tools.mcp_transport import McpError

__all__ = ["expand_all", "expand_env"]

_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(text: str, env: Mapping[str, str]) -> tuple[str, list[str]]:
    """``${VAR}`` и ``${VAR:-по умолчанию}`` из ``env``; второе — переменные, которых нет.

    Пустая переменная с ``:-`` даёт значение по умолчанию (как в оболочке).
    """
    missing: list[str] = []
    expanded = _ENV_REF_RE.sub(partial(_substitute, env, missing), text)
    return expanded, missing


def expand_all(texts: list[str], env: Mapping[str, str]) -> list[str]:
    """Строки с раскрытыми ``${VAR}``; переменной нет и нет значения по умолчанию — McpError."""
    expanded: list[str] = []
    missing: list[str] = []
    for text in texts:
        substituted, absent = expand_env(text, env)
        expanded.append(substituted)
        missing.extend(absent)
    if missing:
        names = ", ".join(sorted(set(missing)))
        raise McpError(f"не заданы переменные окружения: {names}")
    return expanded


def _substitute(env: Mapping[str, str], missing: list[str], match: re.Match[str]) -> str:
    """Значение ссылки ``${VAR}``/``${VAR:-…}``; переменной нет — она в ``missing``."""
    name, default = match.group(1), match.group(2)
    found = env.get(name, "")
    if found:
        return found
    if default is not None:
        return default
    if name not in env:
        missing.append(name)
    return ""
