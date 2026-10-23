"""Модель безопасности gigacrew.

Две основные гарантии:
  1. **Песочница по пути** — все файловые операции ограничены корнем проекта;
     попытки выйти за его пределы (../, абсолютные пути наружу, симлинки)
     отклоняются.
  2. **Классификация риска операции** — каждый вызов инструмента имеет уровень
     риска (SAFE / WRITE / DANGEROUS). Мутирующие и потенциально опасные
     операции требуют подтверждения пользователя (если не включён auto_approve).
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from enum import IntEnum
from pathlib import Path

from gigacrew.errors import SandboxError

__all__ = [
    "GIT_SAFE_CONFIG",
    "SECRET_ENV_PREFIXES",
    "SHELL_ENV_DROP",
    "RiskLevel",
    "SandboxError",
    "classify_shell_command",
    "resolve_in_root",
    "subprocess_env",
]


class RiskLevel(IntEnum):
    """Уровень риска операции: от неё зависит, нужно ли подтверждение."""

    SAFE = 0  # чтение, поиск — без подтверждения
    WRITE = 1  # изменение ФС (запись/редактирование/коммит)
    DANGEROUS = 2  # потенциально разрушительные команды


def resolve_in_root(root: Path, path: str | Path) -> Path:
    """Приводит ``path`` к абсолютному и проверяет, что он внутри ``root``.

    Возвращает разрешённый Path. Бросает SandboxError, если путь ведёт наружу.
    """
    root = root.resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    # resolve() раскрывает .. и симлинки; strict=False — файла может ещё не быть.
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise SandboxError(
            f"Путь '{path}' выходит за пределы корня проекта ({root}). Операция запрещена.",
        )
    return resolved


# Шаблоны заведомо опасных shell-конструкций.
_DANGEROUS_PATTERNS = (
    # rm с флагом -r/-R/-f (в любой позиции и комбинации) или --recursive/--force
    r"\brm\s+(?:\S+\s+)*?-(?:[a-zA-Z]*[rRf][a-zA-Z]*|-recursive|-force)\b",
    r"\bsudo\b",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r">\s*/dev/sd",
    r":\(\)\s*\{",  # fork-бомба
    r"\bchmod\s+-R\b",
    r"\bchown\s+-R\b",
    r"\bgit\s+push\b.*--force",
    r"\bgit\s+push\b.*\s-[a-zA-Z]*f[a-zA-Z]*\b",  # -f, -fu…
    r"\bgit\s+push\b.*\s\+\S",  # +refspec — тоже принудительный push
    r"\bgit\s+branch\b.*\s(?-i:-D)\b",
    r"\bgit\s+stash\s+(?:drop|clear)\b",
    r"\bfind\b.*\s-delete\b",
    r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+clean\b",
    r"\bshutdown\b|\breboot\b",
    r"\bcurl\b|\bwget\b",  # сетевые загрузки запрещены политикой
    r"\bnc\b|\bnetcat\b",
)
_DANGEROUS_RE = re.compile("|".join(_DANGEROUS_PATTERNS), re.IGNORECASE)


def classify_shell_command(command: str) -> RiskLevel:
    """Оценивает риск shell-команды."""
    if _DANGEROUS_RE.search(command):
        return RiskLevel.DANGEROUS
    return RiskLevel.WRITE  # любая shell-команда может что-то менять


# Переменные с этими префиксами не передаются дочерним процессам.
SECRET_ENV_PREFIXES = ("GIGACREW_",)

# Переменные, меняющие поведение оболочки команды: CDPATH уводит `cd src` из проекта,
# ENV и BASH_ENV — файлы, которые sh/bash исполнят перед командой.
SHELL_ENV_DROP = ("CDPATH", "ENV", "BASH_ENV")
# Настройки git для всех команд агента: неявный bare-репозиторий (файлы HEAD, config,
# objects/ в каталоге проекта) не распознаётся, fsmonitor-хук не запускается.
GIT_SAFE_CONFIG = (("safe.bareRepository", "explicit"), ("core.fsmonitor", "false"))


def subprocess_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Окружение для дочернего процесса: без секретов и без интерактива."""
    source = os.environ if environ is None else environ
    env = {
        name: setting
        for name, setting in source.items()
        if not name.upper().startswith(SECRET_ENV_PREFIXES) and name not in SHELL_ENV_DROP
    }
    env["GIT_TERMINAL_PROMPT"] = "0"  # не спрашивать логин/пароль
    env["GIT_EDITOR"] = "true"  # не открывать редактор (commit без -m)
    env["GIT_PAGER"] = "cat"
    env["PAGER"] = "cat"
    try:
        count = int(env.get("GIT_CONFIG_COUNT", "0"))
    except ValueError:
        count = 0
    for config_key, config_setting in GIT_SAFE_CONFIG:
        env[f"GIT_CONFIG_KEY_{count}"] = config_key
        env[f"GIT_CONFIG_VALUE_{count}"] = config_setting
        count += 1
    env["GIT_CONFIG_COUNT"] = str(count)
    return env
