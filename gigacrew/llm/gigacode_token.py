"""Токен бэкенда GigaCode из файла официального CLI GigaCode.

gigacrew сам не входит в бэкенд: токен выдаёт официальный CLI при авторизации и
кладёт в ``$GIGACODE_HOME/gigacode_token.json`` (по умолчанию
``~/.gigacode/gigacode_token.json``). gigacrew только читает этот файл — каждый раз,
когда токен нужен заново, так что повторный вход в GigaCode подхватывается без
перезапуска. Нет файла, токен истёк или бэкенд его не принял — :class:`ApiError` с
просьбой пройти авторизацию в GigaCode.

Формат читается терпимо: токен — ``access_token`` (или ``accessToken``/``token``) на
верхнем уровне либо в записи ``tokens[<адрес конфигурации>]``; срок — ``expires_at``
(``expiresAt``, ``expiry_date``, ``expiry``): unix-время в секундах или миллисекундах
либо дата ISO 8601. Срока нет — токен действует, пока бэкенд не ответит 401.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gigacrew.llm.transport import ApiError

TOKEN_FILE = "gigacode_token.json"
EXPIRY_MARGIN = 60.0  # с: токен считается истёкшим заранее
_TOKEN_KEYS = ("access_token", "accessToken", "token")
_EXPIRY_KEYS = ("expires_at", "expiresAt", "expiry_date", "expiry")
_MS_THRESHOLD = 1e11  # больше — это миллисекунды (1e11 с — 5138 год)
_TOKEN = re.compile(r"[\x21-\x7e]+")  # печатный ASCII без пробелов


@dataclass(frozen=True)
class Token:
    """Токен бэкенда; ``expires_at`` — unix-время истечения (None — действует до 401)."""

    access_token: str
    expires_at: float | None = None

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


def _expiry(value: Any) -> float | None:
    """Unix-время в секундах из числа (секунды или миллисекунды) или даты ISO 8601."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip()
        try:
            value = float(text)
        except ValueError:
            try:
                moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                return None
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=UTC)
            return moment.timestamp()
    if not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return value / 1000 if value > _MS_THRESHOLD else float(value)


def _entry(data: Mapping[str, Any], config_url: str) -> Mapping[str, Any] | None:
    if any(key in data for key in _TOKEN_KEYS):
        return data
    tokens = data.get("tokens")
    entry = tokens.get(config_url) if isinstance(tokens, Mapping) else None
    return entry if isinstance(entry, Mapping) else None


def parse_token(data: Any, config_url: str) -> Token | None:
    """Токен из содержимого файла; не нашёлся или некорректен — None."""
    entry = _entry(data, config_url) if isinstance(data, Mapping) else None
    if entry is None:
        return None
    token = next((entry[k] for k in _TOKEN_KEYS if k in entry), None)
    if not isinstance(token, str) or not _TOKEN.fullmatch(token.strip()):
        return None
    raw = next((entry[k] for k in _EXPIRY_KEYS if entry.get(k) is not None), None)
    expires_at = _expiry(raw)
    if raw is not None and expires_at is None:
        return None  # срок есть, но не разобран — не рискуем просроченным токеном
    return Token(token.strip(), expires_at)


def load_token(path: Path, config_url: str) -> Token:
    """Действующий токен из файла CLI; нет, не читается или истёк — :class:`ApiError`."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ApiError(f"нет токена GigaCode: {login_hint(path)}") from None
    except OSError as e:
        raise ApiError(f"не удалось прочитать {path}: {e}") from e
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    token = parse_token(data, config_url)
    if token is None:
        raise ApiError(f"в {path} нет корректного токена GigaCode: {login_hint(path)}")
    if token.expired():
        raise ApiError(f"токен GigaCode истёк: {login_hint(path)}")
    return token
