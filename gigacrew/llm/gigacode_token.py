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

TOKEN_FILE = "gigacode_token.json"
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
        return (time.time() if now is None else now) >= self.expires_at - EXPIRY_MARGIN


def default_home() -> Path:
    """Каталог официального CLI по умолчанию."""
    return Path.home() / ".gigacode"


def token_path(home: Path | None = None) -> Path:
    return (default_home() if home is None else home) / TOKEN_FILE


def login_hint(path: Path) -> str:
    """Что сделать пользователю, когда токена нет или он не годится."""
    return f"пройдите авторизацию в GigaCode — gigacrew возьмёт новый токен из {path}"


def parse_token(data: Any) -> Token:
    """Токен из содержимого файла; некорректный — ``ValueError`` с причиной."""
    if not isinstance(data, Mapping):
        raise ValueError("не JSON-объект")
    if data.get("version") != FORMAT_VERSION:
        raise ValueError(f"неизвестная версия формата: {data.get('version')!r}")
    entry = data.get("token")
    if not isinstance(entry, Mapping):
        raise ValueError("нет записи token")
    token = entry.get("accessToken")
    if not isinstance(token, str) or not _TOKEN.fullmatch(token):
        raise ValueError("нет корректного accessToken")
    expires_ms = entry.get("expiresAtUnixMs")
    if expires_ms is not None and (
        isinstance(expires_ms, bool)
        or not isinstance(expires_ms, int | float)
        or not math.isfinite(expires_ms)
    ):
        raise ValueError(f"некорректный expiresAtUnixMs: {expires_ms!r}")
    endpoint = entry.get("configEndpoint")
    if endpoint is not None and (not isinstance(endpoint, str) or not endpoint.strip()):
        raise ValueError(f"некорректный configEndpoint: {endpoint!r}")
    return Token(
        token,
        None if expires_ms is None else expires_ms / 1000,
        endpoint.strip() if endpoint is not None else None,
    )


def load_token(path: Path, config_url: str) -> Token:
    """Действующий токен из файла CLI для бэкенда ``config_url``; нет, не читается,
    истёк или выдан другому бэкенду — :class:`ApiError`."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ApiError(f"нет токена GigaCode: {login_hint(path)}") from None
    except OSError as e:
        raise ApiError(f"не удалось прочитать {path}: {e}") from e
    try:
        token = parse_token(json.loads(text))
    except ValueError as e:
        raise ApiError(f"{path}: {e} — {login_hint(path)}") from None
    if token.config_endpoint is not None and (
        token.config_endpoint.rstrip("/") != config_url.rstrip("/")
    ):
        raise ApiError(
            f"токен GigaCode в {path} выдан для {token.config_endpoint}, а GIGACREW_API_URL "
            f"указывает на {config_url} — исправьте GIGACREW_API_URL или пройдите авторизацию "
            "в GigaCode на нужном бэкенде"
        )
    if token.expired():
        raise ApiError(f"токен GigaCode истёк: {login_hint(path)}")
    return token
