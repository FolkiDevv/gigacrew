"""Вход в бэкенд GigaCode через браузер и хранение его токена.

Контракт (как у официального CLI):
  1. ``GET <конфигурация>`` без действующего токена → 401 с заголовками
     ``X-Authorization-Url`` (куда опрашивать), ``X-Signin-Gitverse`` (страница входа) и
     ``Retry-After`` (пауза между опросами);
  2. пользователь входит в браузере, а клиент опрашивает ``POST X-Authorization-Url`` с
     телом устройства (:func:`login_body`): ``401 {"error": "authorization_pending"}`` —
     ждать, ``200 {"access_token", "expires_in"}`` — вход выполнен.

Refresh-токена нет: истёк — вход повторяется. Имя клиента (``plugin_name``) задаёт
``GIGACREW_PLUGIN_NAME``; токен привязан к имени, под которым получен.
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import secrets
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from gigacrew import __version__
from gigacrew.llm.transport import ApiError, error_body

FORMAT_VERSION = 1
TOKEN_FILE = "gigacode_token.json"
EXPIRY_MARGIN = 60.0  # с: токен считается истёкшим заранее
LOGIN_BUDGET = 600.0  # с: сколько ждать подтверждения входа
DEFAULT_PAUSE = 10.0  # с между опросами, если бэкенд не прислал Retry-After
MIN_PAUSE, MAX_PAUSE = 1.0, 60.0
SLOW_DOWN_STEP = 5.0  # ответ slow_down — опрашивать реже
_PENDING = ("authorization_pending", "slow_down")
_TOKEN = re.compile(r"[\x21-\x7e]+")  # печатный ASCII без пробелов
_DEVICE_ID = re.compile(r"[0-9a-fA-F]{64}")


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

    @classmethod
    def from_login(cls, data: Mapping[str, Any], *, now: float | None = None) -> Token:
        """Токен из ответа на вход: ``access_token`` и ``expires_in`` (секунды)."""
        token = data.get("access_token")
        if not isinstance(token, str) or not _TOKEN.fullmatch(token):
            raise ApiError(f"GigaCode не вернул access_token при входе: {str(data)[:200]}")
        lifetime = _seconds(data.get("expires_in"))
        if lifetime is None or lifetime <= 0:
            return cls(token)
        return cls(token, (time.time() if now is None else now) + lifetime)


def _seconds(value: Any) -> float | None:
    """Конечное число секунд (число или строка) либо None."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# ------------------------------------------------------------------ #
# Хранилище токенов
# ------------------------------------------------------------------ #
def default_token_path() -> Path:
    return Path.home() / ".gigacrew" / TOKEN_FILE


class TokenStore:
    """Токены бэкендов в ``~/.gigacrew/gigacode_token.json`` (0600, запись атомарная).

    Ключ — адрес конфигурации бэкенда: токен одного бэкенда не уходит на другой. В записи —
    имя клиента, под которым токен получен: после смены ``GIGACREW_PLUGIN_NAME`` он не
    берётся. Формат::

        {"version": 1, "tokens": {"<адрес>": {"access_token": "…", "expires_at": 1.7e9,
            "plugin_name": "…"}}}

    Битый или нечитаемый файл — как пустой. Ошибки записи — ``OSError``.
    """

    def __init__(self, path: Path | None = None):
        self._path = path

    @property
    def path(self) -> Path:
        """Путь к файлу (по умолчанию определяется в момент обращения)."""
        return default_token_path() if self._path is None else self._path

    def load(self, url: str, plugin_name: str) -> Token | None:
        entry = self._read().get(url)
        if not isinstance(entry, dict) or entry.get("plugin_name") != plugin_name:
            return None
        token = entry.get("access_token")
        if not isinstance(token, str) or not _TOKEN.fullmatch(token):
            return None
        expires = entry.get("expires_at")
        if expires is None:
            return Token(token)
        expires_at = _seconds(expires)
        return None if expires_at is None else Token(token, expires_at)

    def save(self, url: str, plugin_name: str, token: Token) -> None:
        entries = self._read()
        entries[url] = {
            "access_token": token.access_token,
            "expires_at": token.expires_at,
            "plugin_name": plugin_name,
        }
        self._write(entries)

    def forget(self, url: str) -> None:
        entries = self._read()
        if entries.pop(url, None) is not None:
            self._write(entries)

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(data, dict) or data.get("version") != FORMAT_VERSION:
            return {}
        tokens = data.get("tokens")
        return dict(tokens) if isinstance(tokens, dict) else {}

    def _write(self, entries: dict[str, Any]) -> None:
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"version": FORMAT_VERSION, "tokens": dict(sorted(entries.items()))}
        # mkstemp создаёт файл с правами 0600 — токен не читают другие пользователи.
        fd, tmp = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


# ------------------------------------------------------------------ #
# Устройство и тело входа
# ------------------------------------------------------------------ #
def default_home() -> Path:
    """Каталог официального CLI по умолчанию."""
    return Path.home() / ".gigacode"


def device_id(home: Path) -> str:
    """Постоянный id устройства официального CLI: ``<home>/device_id`` (64 hex).

    Нет файла — создаётся так же, как это делает CLI (64 hex, ``O_EXCL``, 0600);
    существующий файл никогда не перезаписывается. Не читается или неверный формат —
    :class:`ApiError` с путём.
    """
    path = home / "device_id"
    try:
        value = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        value = secrets.token_hex(32)
        try:
            home.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            value = _read_device_id(path)  # создан одновременно с нами — берём его
        except OSError as e:
            raise ApiError(f"не удалось создать {path}: {e}") from e
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(value)
            return value
    except OSError as e:
        raise ApiError(f"не удалось прочитать {path}: {e}") from e
    if not _DEVICE_ID.fullmatch(value):
        raise ApiError(
            f"{path}: неверный id устройства (ожидается 64 шестнадцатеричных символа) — "
            "файл не изменён; исправьте или удалите его"
        )
    return value


def _read_device_id(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as e:
        raise ApiError(f"не удалось прочитать {path}: {e}") from e


def _env_name() -> str:
    if sys.platform == "darwin":
        return "mac"
    if sys.platform in ("win32", "cygwin"):
        return "win"
    return "linux" if sys.platform.startswith("linux") else "other"


def login_body(*, device_code: str, plugin_name: str) -> dict[str, str]:
    """Тело опроса входа. ``plugin_name`` — имя клиента из ``GIGACREW_PLUGIN_NAME``."""
    return {
        "device_code": device_code,
        "env_name": _env_name(),
        "plugin_version": __version__,
        "plugin_name": plugin_name,
        "env_version": platform.python_version(),
        "login": os.environ.get("USER") or os.environ.get("USERNAME") or "unknown-user",
        "locale": "ru",
    }


# ------------------------------------------------------------------ #
# Челлендж и опрос
# ------------------------------------------------------------------ #
def safe_url(value: str | None, base: str) -> str | None:
    """Абсолютная ссылка от бэкенда, которую можно открыть и опрашивать, иначе None.

    Только https (http — если сам адрес конфигурации ``base`` http), без пробелов и
    управляющих символов.
    """
    url = (value or "").strip()
    if not url or any(ch.isspace() or not ch.isprintable() for ch in url):
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    allowed = ("https", "http") if urlsplit(base).scheme == "http" else ("https",)
    if parts.scheme not in allowed or not parts.hostname:
        return None
    return url


def _pause(value: str | None, default: float) -> float:
    seconds = _seconds(value)
    if seconds is None:
        return default
    return min(max(seconds, MIN_PAUSE), MAX_PAUSE)


@dataclass(frozen=True)
class Challenge:
    """401 на конфигурации: куда опрашивать, страница входа и пауза между опросами."""

    authorization_url: str
    signin_url: str | None
    retry_after: float
    config_url: str

    @classmethod
    def from_response(cls, resp: httpx.Response, config_url: str) -> Challenge:
        url = safe_url(resp.headers.get("X-Authorization-Url"), config_url)
        if url is None:
            raise ApiError(
                f"GigaCode требует вход ({resp.status_code}), но не прислал корректный "
                f"X-Authorization-Url: {error_body(resp)}"
            )
        return cls(
            authorization_url=url,
            signin_url=safe_url(resp.headers.get("X-Signin-Gitverse"), config_url),
            retry_after=_pause(resp.headers.get("Retry-After"), DEFAULT_PAUSE),
            config_url=config_url,
        )


def _error_code(resp: httpx.Response) -> str | None:
    try:
        data = resp.json()
    except ValueError:
        return None
    error = data.get("error") if isinstance(data, dict) else None
    return error if isinstance(error, str) else None


class DeviceLogin:
    """Опрос адреса входа, пока пользователь входит в браузере.

    ``post(url, body)`` — запрос опроса; ``on_link(url)`` — показать ссылку на страницу
    входа (и открыть браузер). Первый опрос — сразу: бэкенд может пустить без браузера
    или прислать ссылку для этого устройства. Ссылка показывается один раз — из ответа
    опроса, иначе из челленджа; пришла из опроса позже — ещё раз. Ожидание ограничено
    ``budget`` секундами (по сумме пауз и по часам); Ctrl+C — ``KeyboardInterrupt``.
    """

    def __init__(
        self,
        post: Callable[[str, dict[str, str]], httpx.Response],
        *,
        on_link: Callable[[str], None] | None,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
        budget: float = LOGIN_BUDGET,
    ):
        self._post = post
        self._on_link = on_link
        self._sleep = sleep or time.sleep
        self._clock = clock or time.monotonic
        self._budget = budget

    def run(self, challenge: Challenge, body: dict[str, str]) -> Token:
        client = body.get("plugin_name", "")
        shown: str | None = None  # показанная ссылка
        from_poll = False  # она пришла в ответе опроса
        pause = challenge.retry_after
        waited = 0.0
        started = self._clock()
        while True:
            resp = self._post(challenge.authorization_url, body)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                except ValueError:
                    data = None
                if not isinstance(data, dict):
                    raise ApiError(f"GigaCode вернул при входе не-JSON: {error_body(resp)}")
                return Token.from_login(data)
            code = _error_code(resp)
            link = safe_url(resp.headers.get("X-Signin-Gitverse"), challenge.config_url)
            if resp.status_code not in (400, 401) or (code not in _PENDING and link is None):
                raise ApiError(
                    f"GigaCode отклонил вход клиента «{client}» ({resp.status_code}): "
                    f"{error_body(resp)} — проверьте GIGACREW_PLUGIN_NAME"
                )
            if shown is None or (link is not None and not from_poll and link != shown):
                url = link or challenge.signin_url
                if url is None:
                    raise ApiError("GigaCode требует вход, но не прислал ссылку на страницу входа")
                if self._on_link is None:
                    raise ApiError(f"нужен вход в GigaCode — откройте в браузере: {url}")
                self._on_link(url)
                shown, from_poll = url, link is not None
            pause = _pause(resp.headers.get("Retry-After"), pause)
            if code == "slow_down":
                pause = min(pause + SLOW_DOWN_STEP, MAX_PAUSE)
            if waited + pause > self._budget or self._clock() - started + pause > self._budget:
                minutes = int(self._budget // 60)
                raise ApiError(
                    f"вход в GigaCode не подтверждён за {minutes} мин — повторите запрос"
                )
            self._sleep(pause)
            waited += pause
