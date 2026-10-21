"""Настройки репозитория, при которых «читающие» команды git запускают программы.

``git diff``/``log -p``/``show`` вызывают внешний diff и textconv-фильтры, ``git status``
и ``git diff`` — fsmonitor и clean-фильтры, ``log --show-signature`` — gpg. Всё это
задаётся в настройках репозитория (``.git/config`` и подключённых через include).
Клон их не приносит, но их может записать изменяющая команда — поэтому чтение git
считается чтением, только пока таких настроек нет. Глобальные настройки пользователя —
его собственные, их не проверяем.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

__all__ = ["repo_config_runs_programs"]

_TIMEOUT = 5
_RUNS_PROGRAM = re.compile(
    r"^(core\.fsmonitor|diff\.external|diff\..+\.(textconv|command)|"
    r"filter\..+\.(clean|smudge|process)|gpg(\..+)?\.program)$",
    re.IGNORECASE,
)
_FALSE = frozenset({"", "false", "no", "off", "0"})
_LOCAL_SCOPES = frozenset({"local", "worktree", "command"})


def repo_config_runs_programs(cwd: Path) -> bool | None:
    """Есть ли в настройках репозитория (вокруг ``cwd``) программы, которые git запустит
    при чтении. None — не удалось выяснить (git не запустился)."""
    from gigacrew.tools.process import subprocess_env

    try:
        res = subprocess.run(
            ["git", "config", "--list", "--includes", "--show-scope", "-z"],
            cwd=str(cwd),
            env=subprocess_env(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0:
        return None
    # -z --show-scope: «область\0ключ\nзначение\0» подряд
    tokens = res.stdout.decode("utf-8", errors="replace").split("\0")
    for scope, entry in zip(tokens[0::2], tokens[1::2], strict=False):
        if scope not in _LOCAL_SCOPES:
            continue
        key, _, value = entry.partition("\n")
        if _RUNS_PROGRAM.match(key.strip()):
            if key.strip().lower() == "core.fsmonitor" and value.strip().lower() in _FALSE:
                continue
            return True
    return False
