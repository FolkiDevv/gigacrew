"""Файлы с секретами и область пути: внутри проекта, снаружи, среди секретов.

Правило одно для всех: классификатор команд (:mod:`gigacrew.safety.paths` — ``cat .env``
там — непроверенная команда), файловые инструменты, поиск, навигация и индекс проекта
отказываются показывать одни и те же файлы. Модуль — в слое проекта: индексу он нужен
без классификатора команд.
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path

from gigacrew.project.files import is_secret_path

__all__ = [
    "SECRET_DIRS",
    "SINKS",
    "Scope",
    "is_secret_entry",
    "is_secret_location",
    "is_secret_name",
    "is_secret_target",
    "scope_of_path",
]

SINKS = frozenset(("/dev/null", "/dev/stdout", "/dev/stderr", "/dev/stdin", "/dev/tty"))

# Файлы с ключами и учётными данными (имя или путь). .env и .env.* — is_secret_path.
_SECRET_NAME = re.compile(
    r"^(\.netrc|_netrc|\.npmrc|\.pypirc|\.git-credentials|\.pgpass|\.my\.cnf|\.htpasswd|"
    r"id_(rsa|dsa|ecdsa|ed25519)(_sk)?|credentials(\.json|\.ya?ml)?|kubeconfig|"
    r"secrets?(\.(ya?ml|json|toml|ini|env|txt|cfg|conf))?|.*\.secrets?|"
    r".*_history|.*\.(pem|key|p12|pfx|jks|keystore|kdbx|ppk|gpg|asc))$",
    re.IGNORECASE,
)
SECRET_DIRS = frozenset((".ssh", ".aws", ".gnupg", ".kube", ".docker", ".azure", ".gcloud"))
_PROC_ENV = re.compile(r"^/proc/[^/]+/(environ|cmdline|mem|maps)$")
# Известные места учётных данных: пароли системы, ключи хоста ssh, токены gh и gcloud.
_SECRET_PATH = re.compile(
    r"^/etc/(g?shadow|sudoers)-?$|^/etc/ssh/ssh_host_[^/]*_key$|"
    r"(^|/)\.config/(gh|hub)/hosts\.ya?ml$|(^|/)\.config/gcloud/",
)


class Scope(Enum):
    """Область пути: проект, снаружи, секреты, стандартный поток или неизвестно до запуска."""

    INSIDE = "inside"
    OUTSIDE = "outside"
    # Внутри проекта, но с ключами/паролями (.env, *.pem). S105 — ложное срабатывание:
    # это имя области, а не пароль.
    SECRET = "secret"  # noqa: S105
    # ~/.ssh, ~/.aws, /proc/*/environ. S105 — то же ложное срабатывание на имя области.
    SECRET_OUTSIDE = "secret-outside"  # noqa: S105
    SINK = "sink"  # /dev/null и стандартные потоки
    DYNAMIC = "dynamic"  # значение известно только при выполнении


def is_secret_name(name: str) -> bool:
    """Имя файла с ключами или учётными данными (открытый ключ ``*.pub`` — не секрет)."""
    return bool(_SECRET_NAME.match(name)) and not name.casefold().endswith(".pub")


def is_secret_location(path: Path) -> bool:
    """Путь к файлу с секретами — по имени, пути или каталогу (``.ssh/``, ``.aws/``…)."""
    if is_secret_path(path) or is_secret_name(path.name) or _SECRET_PATH.search(path.as_posix()):
        return True
    return any(part in SECRET_DIRS for part in path.parts)


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
        return (
            Scope.SECRET_OUTSIDE
            if is_secret_location(resolved) or is_secret_location(path)
            else Scope.OUTSIDE
        )
    if is_secret_location(path) or is_secret_location(resolved):
        return Scope.SECRET
    return Scope.INSIDE


_SECRET_SCOPES = (Scope.SECRET, Scope.SECRET_OUTSIDE, Scope.DYNAMIC)


def is_secret_target(root: Path, path: Path) -> bool:
    """Файл с ключами или учётными данными — по тому же правилу, что у классификатора команд.

    У классификатора ``cat .env`` — непроверенная команда. Секреты: ``.env*``, ``*.pem``,
    ``id_rsa``, ``.npmrc``, ``credentials.json``, всё в ``.ssh/``, ``.aws/``… Симлинк
    проверяется и по цели; путь, который не раскрыть, считается секретом. Так файлы с
    секретами отказываются показывать все инструменты: чтение, поиск, навигация, индекс.
    """
    return scope_of_path(path, root) in _SECRET_SCOPES


def is_secret_entry(root: Path, path: Path) -> bool:
    """:func:`is_secret_target` для файла из обхода каталогов проекта.

    Обход не заходит в ссылки на каталоги, поэтому раскрывать путь нужно только у самой
    ссылки: иначе проверка по имени. Так в разы быстрее — индекс проверяет каждый файл
    проекта при каждом обновлении.
    """
    return is_secret_location(path) or (path.is_symlink() and is_secret_target(root, path))
