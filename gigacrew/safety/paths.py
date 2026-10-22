"""Где окажется путь из команды: внутри проекта, снаружи, среди секретов или неизвестно.

Путь считается относительно рабочего каталога команды (``cwd`` от корня). Симлинки
раскрываются, как в песочнице файловых инструментов (:func:`gigacrew.security.resolve_in_root`):
ссылка внутри проекта на ``~/.ssh`` — снаружи. Шаблон (``*.py``) раскрывается тем же
способом, что и в оболочке: скрытые файлы ``*`` не совпадают, а сегмент вида ``.*`` в
dash совпадает и с ``..`` — такой шаблон считается выходом наружу.
"""

from __future__ import annotations

import fnmatch
import glob
import os
import re
from enum import Enum
from pathlib import Path

from gigacrew.project import files
from gigacrew.project.files import is_secret_path
from gigacrew.project.workspace import Workspace
from gigacrew.safety.bash import Word

__all__ = [
    "Scope",
    "has_escaping_links",
    "has_secrets_under",
    "is_data_path",
    "is_git_internal",
    "is_secret_name",
    "scope",
    "scope_of_path",
]

# Сколько совпадений шаблона проверять; больше — не знаем, что там.
MAX_GLOB_MATCHES = 2000
SINKS = frozenset({"/dev/null", "/dev/stdout", "/dev/stderr", "/dev/stdin", "/dev/tty"})

# Файлы с ключами и учётными данными (имя или путь). .env и .env.* — is_secret_path.
_SECRET_NAME = re.compile(
    r"^(\.netrc|_netrc|\.npmrc|\.pypirc|\.git-credentials|\.pgpass|\.my\.cnf|\.htpasswd|"
    r"id_(rsa|dsa|ecdsa|ed25519)(_sk)?|credentials(\.json|\.ya?ml)?|kubeconfig|"
    r"secrets?(\.(ya?ml|json|toml|ini|env|txt|cfg|conf))?|.*\.secrets?|"
    r".*_history|.*\.(pem|key|p12|pfx|jks|keystore|kdbx|ppk|gpg|asc))$",
    re.IGNORECASE,
)
_SECRET_DIRS = frozenset({".ssh", ".aws", ".gnupg", ".kube", ".docker", ".azure", ".gcloud"})
_PROC_ENV = re.compile(r"^/proc/[^/]+/(environ|cmdline|mem|maps)$")


class Scope(Enum):
    INSIDE = "inside"
    OUTSIDE = "outside"
    SECRET = "secret"  # внутри проекта, но с ключами/паролями (.env, *.pem)
    SECRET_OUTSIDE = "secret-outside"  # ~/.ssh, ~/.aws, /proc/*/environ
    SINK = "sink"  # /dev/null и стандартные потоки
    DYNAMIC = "dynamic"  # значение известно только при выполнении


def is_secret_name(name: str) -> bool:
    return bool(_SECRET_NAME.match(name)) and not name.casefold().endswith(".pub")


def _is_secret(path: Path) -> bool:
    if is_secret_path(path) or is_secret_name(path.name):
        return True
    return any(part in _SECRET_DIRS for part in path.parts)


def is_git_internal(path: Path, root: Path) -> bool:
    """Путь в служебном каталоге git: хуки и настройки оттуда запускают программы."""
    return files.is_git_internal(root, path)


def is_data_path(path: Path, root: Path) -> bool:
    """Путь в ``.gigacode/`` — служебные данные gigacrew (чаты, планы, скиллы)."""
    return Workspace(root).is_data_path(path)


def scope_of_path(path: Path, root: Path) -> Scope:
    """Область абсолютного пути (симлинки раскрываются)."""
    text = path.as_posix()
    if text in SINKS:
        return Scope.SINK
    if _PROC_ENV.match(text):
        return Scope.SECRET_OUTSIDE
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError):
        return Scope.DYNAMIC
    if resolved != root and root not in resolved.parents:
        return Scope.SECRET_OUTSIDE if _is_secret(resolved) or _is_secret(path) else Scope.OUTSIDE
    if _is_secret(path) or _is_secret(resolved):
        return Scope.SECRET
    return Scope.INSIDE


def _resolve(root: Path, cwd: str, value: str) -> Scope:
    if value.startswith("~"):
        lexical = Path(os.path.expanduser(value))
        return Scope.SECRET_OUTSIDE if _is_secret(lexical) else Scope.OUTSIDE
    if value in SINKS:
        return Scope.SINK
    return scope_of_path(Path(value) if value.startswith("/") else root / cwd / value, root)


def _worst(scopes: list[Scope]) -> Scope:
    order = [
        Scope.SECRET_OUTSIDE,
        Scope.OUTSIDE,
        Scope.DYNAMIC,
        Scope.SECRET,
        Scope.SINK,
        Scope.INSIDE,
    ]
    for s in order:
        if s in scopes:
            return s
    return Scope.INSIDE


def scope(word: Word, *, root: Path, cwd: str = ".") -> Scope:
    """Область пути ``word`` для команды, запущенной в ``root/cwd``."""
    if not word.static:
        return Scope.DYNAMIC
    value = word.value
    if not word.glob:
        return _resolve(root, cwd, value)
    # Шаблон: сегмент, начинающийся с точки, в dash совпадает с «..».
    if any(seg.startswith(".") and re.search(r"[*?\[]", seg) for seg in value.split("/")):
        return Scope.OUTSIDE
    if value.startswith(("~", "/")):
        base_scope = _resolve(root, cwd, value.split("*")[0].rsplit("/", 1)[0] or "/")
        if base_scope not in (Scope.INSIDE,):
            return Scope.OUTSIDE if base_scope != Scope.SECRET_OUTSIDE else base_scope
    try:
        matches = glob.glob(value, root_dir=root / cwd)
    except (OSError, ValueError):
        return Scope.DYNAMIC
    if len(matches) > MAX_GLOB_MATCHES:
        return Scope.DYNAMIC
    if not matches:  # без совпадений шаблон остаётся словом как есть
        return _resolve(root, cwd, value)
    return _worst([_resolve(root, cwd, m) for m in matches])


def _walk(directory: Path, limit: int):
    """Обход дерева целиком, как у ``grep -r`` и ``find``: служебные каталоги (build,
    node_modules, .venv…) они не пропускают, значит, и проверка не пропускает. Отдаёт
    (каталог, подкаталоги, файлы); None — дерево больше ``limit`` записей."""
    seen = 0
    for current, dirs, names in os.walk(directory):
        seen += len(names) + len(dirs)
        if seen > limit:
            yield None
            return
        yield Path(current), dirs, names


def has_secrets_under(
    directory: Path, root: Path, *, name_pattern: str | None = None, limit: int = 20000
) -> bool:
    """Есть ли файлы с секретами в дереве каталога (рекурсивное чтение их покажет).

    ``name_pattern`` — шаблон имён, которые прочитает команда (``*.py``): секреты с
    другими именами не в счёт. Слишком большое дерево — считаем, что есть.
    """
    if not directory.is_dir():
        return _is_secret(directory)
    for entry in _walk(directory, limit):
        if entry is None:
            return True
        current, dirs, names = entry
        if any(d in _SECRET_DIRS for d in dirs):
            return True
        if ".git" in dirs and _git_config_has_credentials(current / ".git" / "config"):
            return True
        for name in names:
            if name_pattern is not None and not fnmatch.fnmatch(name, name_pattern):
                continue
            if is_secret_name(name) or is_secret_path(current / name):
                return True
    return False


_URL_CREDENTIALS = re.compile(r"://[^/\s:@]+:[^/\s@]+@")


def _git_config_has_credentials(path: Path) -> bool:
    """В адресах удалённых репозиториев записан пароль или токен (``https://u:t@host``)."""
    try:
        return bool(_URL_CREDENTIALS.search(path.read_text(encoding="utf-8", errors="replace")))
    except OSError:
        return False


def has_escaping_links(
    directory: Path, root: Path, *, name_pattern: str | None = None, limit: int = 20000
) -> bool:
    """Есть ли в дереве символические ссылки за пределы проекта (чтение по ним — наружу)."""
    if not directory.is_dir():
        return False
    for entry in _walk(directory, limit):
        if entry is None:
            return True
        current, dirs, names = entry
        for name in (*dirs, *names):
            path = current / name
            if not path.is_symlink():
                continue
            if (
                name_pattern is not None
                and name in names
                and not fnmatch.fnmatch(name, name_pattern)
            ):
                continue
            if scope_of_path(path, root) in (Scope.OUTSIDE, Scope.SECRET_OUTSIDE, Scope.DYNAMIC):
                return True
    return False
