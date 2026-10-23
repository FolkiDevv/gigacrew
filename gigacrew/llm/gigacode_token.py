"""Токен бэкенда GigaCode из файла официального CLI GigaCode.

gigacrew сам не входит в бэкенд: токен выдаёт официальный CLI при авторизации и
кладёт в ``$GIGACODE_HOME/gigacode_token.json`` (по умолчанию
``~/.gigacode/gigacode_token.json``). gigacrew только читает этот файл — каждый раз,
когда токен нужен заново, так что повторный вход в GigaCode подхватывается без
перезапуска. Нет файла, токен истёк или бэкенд его не принял — :class:`ApiError` с
просьбой пройти авторизацию в GigaCode.

Формат файла::

    {"version": 1, "token": {"accessToken": "…", "expiresAtUnixMs": 1.7e12,
        "configEndpoint": "https://…/api/v2/config", "authBundle": …, "resourceUrl": …}}

Нужны ``accessToken`` и срок ``expiresAtUnixMs`` (unix-время в миллисекундах; нет —
токен действует, пока бэкенд не ответит 401). ``configEndpoint`` — адрес конфигурации
бэкенда, для которого выдан токен: не совпадает с адресом из ``GIGACREW_API_URL`` —
ошибка, токен на чужой бэкенд не уходит. ``authBundle`` и ``resourceUrl`` не нужны.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gigacrew.llm.transport import ApiError

# Имя файла с токеном, а не сам пароль.
TOKEN_FILE = "gigacode_token.json"  # noqa: S105
FORMAT_VERSION = 1
EXPIRY_MARGIN = 60.0  # с: токен считается истёкшим заранее
_TOKEN = re.compile(r"[\x21-\x7e]+")  # печатный ASCII без пробелов


@dataclass(frozen=True)
class Token:
    """Токен бэкенда; ``expires_at`` — unix-время истечения (None — действует до 401)."""

    access_token: str
    expires_at: float | None = None
    config_endpoint: str | None = None  # для какого бэкенда выдан (None — не указано)

    def expired(self, now: float | None = None) -> bool:
        """Истёк ли (за :data:`EXPIRY_MARGIN` до срока — уже да)."""
        if self.expires_at is None:
            return False
        moment = time.time() if now is None else now
        return moment >= self.expires_at - EXPIRY_MARGIN


def default_home() -> Path:
    """Каталог официального CLI по умолчанию."""
    return Path.home() / ".gigacode"


def token_path(home: Path | None = None) -> Path:
    """Путь к файлу токена в каталоге ``home`` (None — каталог по умолчанию)."""
    directory = default_home() if home is None else home
    return directory / TOKEN_FILE


def login_hint(path: Path) -> str:
    """Что сделать пользователю, когда токена нет или он не годится."""
    return f"пройдите авторизацию в GigaCode — gigacrew возьмёт новый токен из {path}"


def _token_entry(document: Any) -> Mapping[str, Any]:
    """Запись ``token`` из содержимого файла; некорректное — ``ValueError`` с причиной."""
    if not isinstance(document, Mapping):
        # ValueError, а не TypeError: load_token превращает его в просьбу авторизоваться.
        raise ValueError("не JSON-объект")  # noqa: TRY004
    if document.get("version") != FORMAT_VERSION:
        raise ValueError(f"неизвестная версия формата: {document.get('version')!r}")
    entry = document.get("token")
    if not isinstance(entry, Mapping):
        # ValueError, а не TypeError: load_token превращает его в просьбу авторизоваться.
        raise ValueError("нет записи token")  # noqa: TRY004
    return entry


def _expires_ms(entry: Mapping[str, Any]) -> float | None:
    """Срок ``expiresAtUnixMs`` записи (None — не указан); некорректный — ``ValueError``."""
    expires_ms = entry.get("expiresAtUnixMs")
    if expires_ms is not None and (
        isinstance(expires_ms, bool)
        or not isinstance(expires_ms, int | float)
        or not math.isfinite(expires_ms)
    ):
        raise ValueError(f"некорректный expiresAtUnixMs: {expires_ms!r}")
    return expires_ms


def _config_endpoint(entry: Mapping[str, Any]) -> str | None:
    """``configEndpoint`` записи (None — не указан); некорректный — ``ValueError``."""
    endpoint = entry.get("configEndpoint")
    if endpoint is not None and (not isinstance(endpoint, str) or not endpoint.strip()):
        raise ValueError(f"некорректный configEndpoint: {endpoint!r}")
    return endpoint


def parse_token(document: Any) -> Token:
    """Токен из содержимого файла; некорректный — ``ValueError`` с причиной."""
    entry = _token_entry(document)
    token = entry.get("accessToken")
    if not isinstance(token, str) or not _TOKEN.fullmatch(token):
        raise ValueError("нет корректного accessToken")
    expires_ms = _expires_ms(entry)
    endpoint = _config_endpoint(entry)
    return Token(
        token,
        None if expires_ms is None else expires_ms / 1000,
        None if endpoint is None else endpoint.strip(),
    )


def _read_token(path: Path) -> Token:
    """Токен из файла CLI; нет, не читается или некорректен — :class:`ApiError`."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ApiError(f"нет токена GigaCode: {login_hint(path)}") from None
    except OSError as exc:
        raise ApiError(f"не удалось прочитать {path}: {exc}") from exc
    try:
        return parse_token(json.loads(text))
    except ValueError as exc:
        raise ApiError(f"{path}: {exc} — {login_hint(path)}") from None


def load_token(path: Path, config_url: str) -> Token:
    """Действующий токен из файла CLI для бэкенда ``config_url``.

    Нет, не читается, истёк или выдан другому бэкенду — :class:`ApiError`.
    """
    token = _read_token(path)
    if token.config_endpoint is not None and (
        token.config_endpoint.rstrip("/") != config_url.rstrip("/")
    ):
        raise ApiError(
            f"токен GigaCode в {path} выдан для {token.config_endpoint}, а GIGACREW_API_URL "
            f"указывает на {config_url} — исправьте GIGACREW_API_URL или пройдите авторизацию "
            "в GigaCode на нужном бэкенде",
        )
    if token.expired():
        raise ApiError(f"токен GigaCode истёк: {login_hint(path)}")
    return token
