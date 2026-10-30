"""Настройки репозитория jj, при которых «читающие» команды jj запускают программы.

``jj diff``/``show``/``log -p`` выводят разницу внешней программой, если её задают
``ui.diff-formatter`` или ``merge-tools.<имя>`` (в том числе для программы, которую
назвали в настройках пользователя). Снимок рабочей копии, который jj делает почти перед
каждой командой, подписывает коммит программой из ``signing.backends.<имя>.program``.
Псевдонимы ``st``, ``b``, ``ci`` и ``desc`` — тоже настройки (``status``, ``bookmark``…),
их можно переопределить на ``util exec``. Всё это задают настройки репозитория и рабочей
копии: с jj 0.37 они лежат в каталоге настроек пользователя (раньше — в ``.jj/``). Клон их
не приносит, но их может записать изменяющая команда (``jj config set --repo``) — поэтому
чтение jj считается чтением, только пока таких настроек нет. Где лежат файлы, знает сам
jj: настройки читаются его командой ``jj config list`` вместе с источником каждого
значения. Настройки пользователя — его собственные, их не проверяем.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from gigacrew.security import subprocess_env

__all__ = ["repo_config_runs_programs"]

_TIMEOUT = 5
# Источник, имя и значение (запись TOML) каждой действующей настройки, через \0.
_TEMPLATE = r'source ++ "\0" ++ name ++ "\0" ++ value ++ "\0"'
_FIELDS = 3
_LOCAL_SOURCES = frozenset(("repo", "workspace"))
_RUNS_PROGRAM = re.compile(
    r"^(ui\.diff-formatter|ui\.diff\.tool|merge-tools\..+|signing\.backends\..+\.program|"
    r"aliases\.(b|ci|desc|st))$",
)


def repo_config_runs_programs(cwd: Path) -> bool | None:
    """Есть ли программы, которые jj запустит при чтении.

    Ищет их в настройках репозитория и рабочей копии вокруг ``cwd``. None — не удалось
    выяснить (jj не ответил или слишком старый для ``source`` в шаблоне). jj не
    установлен — False: команда jj не запустится, а с ней и программы из настроек.
    """
    env = subprocess_env()
    if shutil.which("jj", path=env.get("PATH")) is None:
        return False
    command = ["jj", "--ignore-working-copy", "--no-pager", "--color=never"]
    try:
        res = subprocess.run(
            [*command, "config", "list", "--template", _TEMPLATE],
            cwd=str(cwd),
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0:
        return None
    fields = res.stdout.decode("utf-8", errors="replace").split("\0")
    records = zip(
        fields[::_FIELDS],
        fields[1::_FIELDS],
        fields[2::_FIELDS],
        strict=False,
    )
    return any(
        source.strip() in _LOCAL_SOURCES and _runs_program(name.strip(), setting.strip())
        for source, name, setting in records
    )


def _runs_program(name: str, setting: str) -> bool:
    if not _RUNS_PROGRAM.match(name):
        return False
    # ":git", ":color-words"… — встроенные форматы, без внешней программы
    return not (name == "ui.diff-formatter" and setting.startswith(('":', "':")))
