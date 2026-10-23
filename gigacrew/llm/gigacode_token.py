"""Токен бэкенда GigaCode из файла официального CLI GigaCode.

gigacrew сам не входит в бэкенд: токен выдаёт официальный CLI при авторизации и
кладёт в ``$GIGACODE_HOME/gigacode_token.json`` (по умолчанию
``~/.gigacode/gigacode_token.json``). gigacrew только читает этот файл — каждый раз,
когда токен нужен заново, так что повторный вход в GigaCode подхватывается без
перезапуска. Нет файла, токен истёк или бэкенд его не принял — :class:`ApiError` с
просьбой пройти авторизацию в GigaCode.

Формат файла::

    {"token": "…", "expiryDate": 1792744173568}

Нужен ``token``; ``expiryDate`` — срок (unix-время в миллисекундах; нет — токен
действует, пока бэкенд не ответит 401). Прочие поля игнорируются.
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
EXPIRY_MARGIN = 60.0  # с: токен считается истёкшим заранее
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


def _expires_ms(document: Mapping[str, Any]) -> float | None:
    """Срок ``expiryDate`` (None — не указан); некорректный — ``ValueError``."""
    expires_ms = document.get("expiryDate")
    if expires_ms is not None and (
        isinstance(expires_ms, bool)
        or not isinstance(expires_ms, int | float)
        or not math.isfinite(expires_ms)
    ):
        raise ValueError(f"некорректный expiryDate: {expires_ms!r}")
    return expires_ms


def parse_token(document: Any) -> Token:
    """Токен из содержимого файла; некорректный — ``ValueError`` с причиной."""
    if not isinstance(document, Mapping):
        # ValueError, а не TypeError: load_token превращает его в просьбу авторизоваться.
        raise ValueError("не JSON-объект")  # noqa: TRY004
    token = document.get("token")
    if not isinstance(token, str) or not _TOKEN.fullmatch(token):
        raise ValueError("нет корректного token")
    expires_ms = _expires_ms(document)
    return Token(token, None if expires_ms is None else expires_ms / 1000)


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


def load_token(path: Path) -> Token:
    """Действующий токен из файла CLI; нет, не читается или истёк — :class:`ApiError`."""
    token = _read_token(path)
    if token.expired():
        raise ApiError(f"токен GigaCode истёк: {login_hint(path)}")
    return token
