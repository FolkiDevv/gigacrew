"""Настройки репозитория, при которых «читающие» команды git запускают программы.

``git diff``/``log -p``/``show`` вызывают внешний diff и textconv-фильтры, ``git status``
и ``git diff`` — fsmonitor и clean-фильтры, ``log --show-signature`` — gpg. Всё это
задаётся в настройках репозитория (``.git/config`` и подключённых через include) и его
подмодулей (``.git/modules/<имя>/config``: ``status`` и ``diff`` заходят и в них).
Клон их не приносит, но их может записать изменяющая команда — поэтому чтение git
считается чтением, только пока таких настроек нет. Глобальные настройки пользователя —
его собственные, их не проверяем.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from gigacrew.security import subprocess_env

__all__ = ["repo_config_runs_programs", "repo_toplevel"]

_TIMEOUT = 5
_MAX_DEPTH = 8  # вложенность подмодулей
_RUNS_PROGRAM = re.compile(
    r"^(core\.fsmonitor|diff\.external|diff\..+\.(textconv|command)|"
    r"filter\..+\.(clean|smudge|process)|gpg(\..+)?\.program)$",
    re.IGNORECASE,
)
_DISABLED = frozenset(("", "false", "no", "off", "0"))
_LOCAL_SCOPES = frozenset(("local", "worktree", "command"))


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[bytes] | None:
    """Служебный запуск git в ``cwd``; None — git не запустился."""
    try:
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            env=subprocess_env(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def repo_toplevel(cwd: Path) -> Path | None:
    """Корень рабочего дерева репозитория, который git найдёт из ``cwd``.

    None — ``cwd`` не в рабочем дереве (или git не запустился).
    """
    res = _git(cwd, "rev-parse", "--show-toplevel")
    if res is None or res.returncode != 0:
        return None
    top = res.stdout.decode("utf-8", errors="replace").strip()
    return Path(top).resolve() if top else None


def repo_config_runs_programs(cwd: Path) -> bool | None:
    """Есть ли программы, которые git запустит при чтении.

    Ищет их в настройках репозитория (вокруг ``cwd``) и его подмодулей. None — не удалось
    выяснить (git не запустился).
    """
    return _runs_programs(cwd, depth=0)


def _runs_programs(cwd: Path, *, depth: int) -> bool | None:
    found = _config_runs_programs(cwd)
    if found is not False:
        return found
    # `git status` и `git diff` заходят в каждый подмодуль с рабочим деревом и применяют
    # фильтры из его собственных настроек (.git/modules/<имя>/config).
    submodules = _submodules(cwd)
    if submodules is None or (submodules and depth >= _MAX_DEPTH):
        return None
    for path in submodules:
        found = _runs_programs(path, depth=depth + 1)
        if found is not False:
            return found
    return False


def _submodules(cwd: Path) -> list[Path] | None:
    """Рабочие деревья подмодулей репозитория (вокруг ``cwd``), в которые зайдёт git.

    ``[]`` — подмодулей нет или ``cwd`` не в рабочем дереве; None — не удалось выяснить.
    """
    top = repo_toplevel(cwd)
    if top is None:
        return []
    res = _git(top, "ls-files", "--stage", "-z")
    if res is None or res.returncode != 0:
        return None
    out = []
    for entry in res.stdout.decode("utf-8", errors="replace").split("\0"):
        meta, _, path = entry.partition("\t")
        if not meta.startswith("160000 "):  # gitlink — коммит подмодуля
            continue
        work_tree = top / path
        # без .git подмодуль не инициализирован — git в него не заходит
        if not work_tree.is_symlink() and (work_tree / ".git").exists():
            out.append(work_tree)
    return out


def _config_runs_programs(cwd: Path) -> bool | None:
    res = _git(cwd, "config", "--list", "--includes", "--show-scope", "-z")
    if res is None or res.returncode != 0:
        return None
    # -z --show-scope: «область\0ключ\nзначение\0» подряд
    tokens = res.stdout.decode("utf-8", errors="replace").split("\0")
    for scope, entry in zip(tokens[::2], tokens[1::2], strict=False):
        if scope not in _LOCAL_SCOPES:
            continue
        key, _, setting = entry.partition("\n")
        if _RUNS_PROGRAM.match(key.strip()):
            if key.strip().lower() == "core.fsmonitor" and setting.strip().lower() in _DISABLED:
                continue
            return True
    return False
