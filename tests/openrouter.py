"""Адаптер OpenRouter для live-тестов и eval: контракт :class:`LLMProvider` на чужих моделях.

gigacrew работает с API v2 и бэкендом GigaCode; чтобы гонять те же live-тесты
(``pytest -m live``) и поведенческие сценарии (``make eval``) на моделях OpenRouter, тесты
берут провайдера через :func:`open_live_provider`:

* ``GIGACREW_TEST_PROVIDER`` — ``api`` (по умолчанию: провайдер API v2 из настроек
  gigacrew) или ``openrouter``;
* ``GIGACREW_OPENROUTER_API_KEY`` (иначе ``OPENROUTER_API_KEY``) — ключ OpenRouter. Имя с
  ``GIGACREW_`` надёжнее: такие переменные не доходят до команд, которые агент запускает
  в eval (:func:`gigacrew.security.subprocess_env`);
* ``GIGACREW_TEST_MODEL`` и ``GIGACREW_EVAL_MODELS`` — id моделей OpenRouter
  (``qwen/qwen3-coder``, ``z-ai/glm-4.6``…).

Формат — OpenAI Chat Completions (:mod:`gigacrew.llm.openai_chat`, тот же кодек, что у
бэкенда GigaCode); сеть, повторы, поток и замер окна — :class:`HttpProvider`. Каталог
``/models`` (:meth:`OpenRouterProvider.connect`) сообщает окна моделей, и eval не
обрезает историю до окна незамеренной модели (32K). Настройки агента (температура,
рассуждения, сжатие…) — из окружения, как у gigacrew; реквизиты и TLS-настройки API v2
(ключ, клиентский сертификат, свой корневой сертификат) в OpenRouter не уходят.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Final, override

import httpx

from gigacrew.config import Config
from gigacrew.llm import openai_chat
from gigacrew.llm.api import ApiProvider
from gigacrew.llm.base import LLMProvider, ModelInfo
from gigacrew.llm.transport import ApiError, HttpProvider, error_body
from gigacrew.llm.types import AssistantTurn, Message, ToolChoice, ToolSpec

__all__ = [
    "API",
    "KEY_ENVS",
    "OPENROUTER",
    "OPENROUTER_URL",
    "PROVIDER_ENV",
    "LiveUnavailableError",
    "OpenRouterProvider",
    "live_provider_kind",
    "open_live_provider",
    "openrouter_config",
    "openrouter_key",
    "parse_catalogue",
]

REPO_ROOT: Final = Path(__file__).resolve().parent.parent
OPENROUTER_URL: Final = "https://openrouter.ai/api/v1"
PROVIDER_ENV: Final = "GIGACREW_TEST_PROVIDER"
KEY_ENVS: Final = ("GIGACREW_OPENROUTER_API_KEY", "OPENROUTER_API_KEY")
# Провайдеры live-тестов: API v2 из настроек gigacrew и OpenRouter.
API: Final = "api"
OPENROUTER: Final = "openrouter"
# Настройки подключения к API v2 и бэкенду GigaCode. OpenRouter они не нужны, а
# некорректные (GIGACREW_CA_BUNDLE без файла, опечатка в GIGACREW_PROVIDER) сорвали бы
# загрузку настроек агента — при загрузке для OpenRouter они пустые.
_GATEWAY_ENVS: Final = (
    "GIGACREW_PROVIDER",
    "GIGACREW_API_URL",
    "GIGACREW_ACCESS_KEY",
    "GIGACREW_AUTH_URL",
    "GIGACREW_SCOPE",
    "GIGACREW_CERT",
    "GIGACREW_KEY",
    "GIGACREW_CA_BUNDLE",
    "GIGACREW_VERIFY_SSL",
    "GIGACREW_USER_AGENT",
)
# Сколько символов некорректного ответа показывать в ошибке.
_SHOWN_CHARS: Final = 200


class LiveUnavailableError(RuntimeError):
    """Провайдер live-тестов не настроен (нет ключа, адреса): тест пропускается."""


def live_provider_kind(env: Mapping[str, str]) -> str:
    """Провайдер live-тестов из ``GIGACREW_TEST_PROVIDER``; опечатка — ``ValueError``."""
    kind = (env.get(PROVIDER_ENV) or API).strip().lower()
    if kind not in {API, OPENROUTER}:
        raise ValueError(f"{PROVIDER_ENV}={kind}: ожидается {API} или {OPENROUTER}")
    return kind


def openrouter_key(env: Mapping[str, str]) -> str:
    """Ключ OpenRouter из :data:`KEY_ENVS`; не задан — :class:`LiveUnavailableError`."""
    for name in KEY_ENVS:
        key = env.get(name, "").strip()
        if key:
            return key
    raise LiveUnavailableError(f"не задан {KEY_ENVS[0]}")


def openrouter_config(config: Config) -> Config:
    """Настройки агента из ``config`` с адресом OpenRouter, без реквизитов API v2.

    Ключ, клиентский сертификат и корневой сертификат шлюза OpenRouter не нужны и уходить
    туда не должны; проверка TLS — по хранилищу ОС, всегда.
    """
    return replace(
        config,
        provider=API,
        base_url=OPENROUTER_URL,
        access_key=None,
        cert=None,
        key=None,
        auth_url=None,
        scope=None,
        ca_bundle=None,
        verify_ssl=True,
        user_agent=None,
    )


def open_live_provider(model: str, env: Mapping[str, str]) -> tuple[Config, LLMProvider]:
    """Конфиг и провайдер live-тестов для ``model`` по ``GIGACREW_TEST_PROVIDER``.

    Настройки — окружение и ``.env`` репозитория (доверенный); для OpenRouter — без настроек
    подключения к API v2 (:data:`_GATEWAY_ENVS`). Провайдер не настроен —
    :class:`LiveUnavailableError` с причиной для ``pytest.skip``.
    """
    if live_provider_kind(env) == OPENROUTER:
        key = openrouter_key(env)
        blank = dict.fromkeys(_GATEWAY_ENVS, "")
        loaded = Config.load(
            project_root=REPO_ROOT,
            trusted_dotenv=True,
            model=model,
            environ={**os.environ, **blank},
        )
        router_config = openrouter_config(loaded)
        return router_config, OpenRouterProvider(router_config, key)
    config = Config.load(project_root=REPO_ROOT, trusted_dotenv=True, model=model)
    if config.provider != API:
        raise LiveUnavailableError(
            f"live-тесты — для API v2 (GIGACREW_PROVIDER={API}) "
            f"или OpenRouter ({PROVIDER_ENV}={OPENROUTER})",
        )
    try:
        config.require_credentials()
    except RuntimeError as error:
        raise LiveUnavailableError(str(error)) from error
    return config, ApiProvider(config)


def _window(raw: Any) -> int | None:
    if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
        return raw
    return None


def parse_catalogue(body: Mapping[str, Any]) -> list[ModelInfo]:
    """Модели из ответа ``/models``: ``data[].id`` и окно ``context_length``.

    Записи без id пропускаются; ответ без списка ``data`` — :class:`ApiError`.
    """
    entries = body.get("data")
    if not isinstance(entries, list):
        shown = str(body)[:_SHOWN_CHARS]
        raise ApiError(f"OpenRouter вернул каталог без списка data: {shown}")
    models: list[ModelInfo] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        model_id = entry.get("id")
        if isinstance(model_id, str) and model_id.strip():
            models.append(ModelInfo(model_id, _window(entry.get("context_length"))))
    return models


class OpenRouterProvider(HttpProvider):
    """OpenRouter: ключ — Bearer, формат — OpenAI Chat Completions, каталог — ``/models``.

    Конструктор не обращается к сети; каталог (окна моделей) загружает :meth:`connect`.
    """

    def __init__(
        self,
        config: Config,
        api_key: str,
        *,
        transport: httpx.BaseTransport | None = None,
    ):
        """Провайдер по ``config`` (:func:`openrouter_config`) с ключом ``api_key``.

        ``transport`` — подмена HTTP-транспорта (``httpx.MockTransport`` в тестах).
        """
        super().__init__(config, transport=transport)
        self._base_url = config.base_url
        self._api_key = api_key
        self._catalogue: list[ModelInfo] | None = None

    @override
    def connect(self) -> None:
        """Каталог моделей, если ещё не загружен; ошибка — :class:`ApiError`."""
        if self._catalogue is None:
            self._catalogue = self._load_catalogue()

    @override
    def list_models(self) -> list[str]:
        """Модели каталога (без сети); до :meth:`connect` — только текущая."""
        if self._catalogue is None:
            return super().list_models()
        return [entry.id for entry in self._catalogue]

    @override
    def known_models(self) -> list[ModelInfo] | None:
        """Модели каталога с окнами контекста (до :meth:`connect` — None)."""
        return None if self._catalogue is None else list(self._catalogue)

    def _load_catalogue(self) -> list[ModelInfo]:
        url = f"{self._base_url}/models"
        resp = self._request_with_retry(lambda: self._client.get(url, headers=self._auth_headers()))
        if not resp.is_success:
            raise ApiError(
                f"OpenRouter не отдал каталог моделей ({resp.status_code}): {error_body(resp)}",
            )
        return parse_catalogue(self._json(resp))

    # ------------------------------------------------------------------ #
    # Хуки HttpProvider
    # ------------------------------------------------------------------ #
    @override
    def _chat_url(self) -> str:
        return f"{self._base_url}/chat/completions"

    @override
    def _auth_headers(self, *, force: bool = False) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    @override
    def _payload(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None,
        *,
        model: str | None,
        temperature: float,
        stream: bool = False,
        tool_choice: ToolChoice = "auto",
        reasoning: bool = True,
    ) -> dict[str, Any]:
        return openai_chat.chat_payload(
            messages,
            tools,
            model=model or self.model,
            temperature=temperature,
            stream=stream,
            tool_choice=tool_choice,
            reasoning=reasoning,
        )

    @override
    def _has_reasoning(self, payload: Mapping[str, Any]) -> bool:
        return openai_chat.has_reasoning(payload)

    @override
    def _parse_response(self, body: dict[str, Any]) -> AssistantTurn:
        return openai_chat.parse_response(body)

    @override
    def _consume_sse(self, resp: httpx.Response, emit: Callable[[str], None]) -> AssistantTurn:
        content_type = resp.headers.get("content-type", "")
        return openai_chat.consume_stream(resp.iter_lines(), emit, content_type=content_type)

    @override
    def _probe_payload(self, text: str, model: str) -> dict[str, Any]:
        return openai_chat.probe_payload(text, model, stream=False)
