"""Провайдер бэкенда GigaCode: токен официального CLI, каталог моделей, Chat Completions.

Адрес — ``GIGACREW_API_URL``: ``https://host`` → конфигурация ``…/api/v2/config``; адрес,
который кончается на ``/api/vN``, — ``+ /config``; на ``/config`` — как есть. Конфигурация
(``GET``) — это и проверка токена, и каталог моделей: id, окно контекста, параметры
генерации (``ai_api``), модель по умолчанию и адрес чата.

Токен (:mod:`gigacrew.llm.gigacode_token`) gigacrew не получает сам, а читает из файла
официального CLI GigaCode: ``$GIGACODE_HOME/gigacode_token.json`` (по умолчанию
``~/.gigacode/gigacode_token.json``). Токена нет, он истёк или бэкенд ответил 401 — ошибка
с просьбой пройти авторизацию в GigaCode. User-Agent всех запросов —
``GIGACREW_USER_AGENT`` (значения по умолчанию нет).

Чат — OpenAI Chat Completions (:mod:`gigacrew.llm.openai_chat`); каждый запрос идёт
потоком (бэкенд рассчитан на поток), кроме моделей с ``ai_api.stream: false``.
``max_tokens`` не передаётся: ответ ограничивает бюджет агента.
"""

from __future__ import annotations

import math
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx

from gigacrew.config import Config
from gigacrew.llm import openai_chat
from gigacrew.llm.base import ModelInfo
from gigacrew.llm.gigacode_token import Token, load_token, login_hint, token_path
from gigacrew.llm.transport import ApiError, HttpProvider, error_body
from gigacrew.llm.types import AssistantTurn, Message, ToolChoice, ToolSpec, Usage

# Окно модели, если каталог его не называет (так считает и официальный CLI).
DEFAULT_CONTEXT_WINDOW = 262_144
# Параметры генерации из ``ai_api`` модели, которые уходят в запрос. Температуру задаёт
# gigacrew (GIGACREW_TEMPERATURE), max_tokens — бюджет агента; n и max_lines не нужны.
SAMPLING_KEYS = (
    "top_p",
    "top_k",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "seed",
)

_API_VERSION = re.compile(r"/api/v\d+$")


def config_endpoint(base_url: str) -> str:
    """Адрес конфигурации бэкенда по ``GIGACREW_API_URL``."""
    url = base_url.strip().rstrip("/")
    if url.endswith("/config"):
        return url
    if _API_VERSION.search(url):
        return f"{url}/config"
    return f"{url}/api/v2/config"


@dataclass(frozen=True)
class CatalogueModel:
    """Модель каталога: окно контекста, параметры генерации, отвечает ли потоком."""

    id: str
    context_window: int = DEFAULT_CONTEXT_WINDOW
    sampling: Mapping[str, int | float] = field(default_factory=dict)
    streams: bool = True


@dataclass(frozen=True)
class Catalogue:
    models: tuple[CatalogueModel, ...]
    default_model: str
    chat_url: str

    def get(self, model: str) -> CatalogueModel | None:
        return next((m for m in self.models if m.id == model), None)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _window(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return value if math.isfinite(value) else None


def _parse_model(raw: Any) -> CatalogueModel:
    model_id = raw.get("id") if isinstance(raw, Mapping) else None
    if (
        not isinstance(model_id, str)
        or not model_id.strip()
        or any(not ch.isprintable() for ch in model_id)
    ):
        raise ApiError(f"в каталоге GigaCode модель без корректного id: {str(raw)[:200]}")
    window = _window(_mapping(raw.get("plugin_side")).get("context_window_size")) or _window(
        raw.get("context_window_size")
    )
    options = _mapping(raw.get("ai_api"))
    sampling = {
        key: value for key in SAMPLING_KEYS if (value := _number(options.get(key))) is not None
    }
    return CatalogueModel(
        id=model_id,
        context_window=window or DEFAULT_CONTEXT_WINDOW,
        sampling=sampling,
        streams=options.get("stream") is not False,
    )


def _chat_endpoint(value: Any, config_url: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return config_url.removesuffix("/config") + "/chat/completions"
    url = urljoin(config_url, value.strip())  # относительный — от адреса конфигурации
    parts = urlsplit(url)
    if parts.scheme not in ("https", "http") or not parts.hostname:
        raise ApiError(f"некорректный адрес чата в конфигурации GigaCode: {value[:200]}")
    if parts.scheme == "http" and urlsplit(config_url).scheme == "https":
        # Конфигурация по https, а чат — нет: токен ушёл бы открытым текстом.
        raise ApiError(
            "адрес чата в конфигурации GigaCode — http при https-адресе конфигурации, "
            f"токен ушёл бы открытым текстом: {value[:200]}"
        )
    return url


def parse_catalogue(data: Mapping[str, Any], config_url: str) -> Catalogue:
    """Каталог моделей из конфигурации бэкенда; некорректный — :class:`ApiError`.

    Модели — из ``chat.models``, иначе ``agent.models``, иначе ``models``; окно —
    ``plugin_side.context_window_size`` или ``context_window_size``; модель по умолчанию —
    ``chat.default_model``, затем ``default_model`` (только из каталога), иначе первая.
    Адрес чата — ``chat.url`` или ``chatEndpoint``, иначе ``…/chat/completions`` рядом с
    конфигурацией. Незнакомые поля не мешают.
    """
    chat = _mapping(data.get("chat"))
    agent = _mapping(data.get("agent"))
    raw = next(
        (
            models
            for models in (chat.get("models"), agent.get("models"), data.get("models"))
            if isinstance(models, list) and models
        ),
        None,
    )
    if raw is None:
        raise ApiError("бэкенд GigaCode не вернул ни одной модели в конфигурации")
    models: list[CatalogueModel] = []
    for item in raw:
        model = _parse_model(item)
        if any(m.id == model.id for m in models):
            raise ApiError(f"в каталоге GigaCode модель {model.id} повторяется")
        models.append(model)
    ids = {m.id for m in models}
    default = next(
        (
            d
            for d in (chat.get("default_model"), data.get("default_model"))
            if isinstance(d, str) and d in ids
        ),
        models[0].id,
    )
    url = _chat_endpoint(chat.get("url") or data.get("chatEndpoint"), config_url)
    return Catalogue(models=tuple(models), default_model=default, chat_url=url)


class GigaCodeProvider(HttpProvider):
    """Бэкенд GigaCode. Конструктор не обращается к сети и файлам.

    Токен — из файла официального CLI (:mod:`gigacrew.llm.gigacode_token`); сам
    gigacrew не входит: токена нет, он истёк или бэкенд ответил 401 — ошибка с просьбой
    пройти авторизацию в GigaCode. ``token_file`` — подмена пути к файлу (тесты).
    """

    def __init__(
        self,
        config: Config,
        *,
        transport: httpx.BaseTransport | None = None,
        token_file: Path | None = None,
    ):
        if config.provider != "gigacode":
            raise RuntimeError(
                f"GIGACREW_PROVIDER={config.provider}: этот провайдер — для бэкенда GigaCode"
            )
        config.require_credentials()  # адрес и User-Agent заданы
        super().__init__(
            config, transport=transport, headers={"User-Agent": config.user_agent or ""}
        )
        self._config_url = config_endpoint(config.base_url)
        self._token_file = token_file if token_file is not None else token_path(config.gigacode_dir)
        self._session_id = str(uuid.uuid4())
        self._token: Token | None = None
        self._catalogue: Catalogue | None = None

    @property
    def config_url(self) -> str:
        return self._config_url

    @property
    def token_file(self) -> Path:
        return self._token_file

    @property
    def model(self) -> str:
        if self._cfg.model:
            return self._cfg.model
        return self._catalogue.default_model if self._catalogue is not None else ""

    # ------------------------------------------------------------------ #
    # Подключение, каталог, токен
    # ------------------------------------------------------------------ #
    def connect(self) -> None:
        """Действующий токен из файла CLI и каталог моделей; токена нет или истёк —
        :class:`ApiError` с просьбой пройти авторизацию в GigaCode."""
        self._token = load_token(self._token_file, self._config_url)
        if self._catalogue is None:
            self._load_config()

    def list_models(self) -> list[str]:
        """Модели каталога (без сети; до :meth:`connect` — пусто)."""
        return [m.id for m in self._catalogue.models] if self._catalogue is not None else []

    def known_models(self) -> list[ModelInfo] | None:
        if self._catalogue is None:
            return None
        return [ModelInfo(m.id, m.context_window) for m in self._catalogue.models]

    def _current_token(self) -> Token:
        """Действующий токен: в памяти, а истёк или его нет — заново из файла."""
        if self._token is None or self._token.expired():
            self._token = load_token(self._token_file, self._config_url)
        return self._token

    def _reload_rejected(self, rejected: Token) -> Token:
        """Бэкенд ответил 401 на ``rejected``: файл мог обновиться — перечитать. Там тот
        же токен — :class:`ApiError` с просьбой пройти авторизацию заново."""
        self._token = None
        token = load_token(self._token_file, self._config_url)
        if token.access_token == rejected.access_token:
            raise ApiError(f"GigaCode не принял токен (401): {login_hint(self._token_file)}")
        self._token = token
        return token

    def _get_config(self, token: Token) -> httpx.Response:
        headers = {"Accept": "*/*", "Authorization": f"Bearer {token.access_token}"}
        return self._request_with_retry(lambda: self._client.get(self._config_url, headers=headers))

    def _load_config(self) -> None:
        """``GET`` конфигурации: каталог моделей; 401 — перечитать токен и повторить раз."""
        token = self._current_token()
        resp = self._get_config(token)
        if resp.status_code == 401:
            resp = self._get_config(self._reload_rejected(token))
            if resp.status_code == 401:
                self._token = None
                raise ApiError(f"GigaCode не принял токен (401): {login_hint(self._token_file)}")
        if not resp.is_success:
            raise ApiError(
                f"Не удалось получить конфигурацию GigaCode ({resp.status_code}): "
                f"{error_body(resp)}"
            )
        self._catalogue = parse_catalogue(self._json(resp), self._config_url)

    def _ensure_catalogue(self) -> Catalogue:
        """Каталог для запроса; не загружен (запрос без :meth:`connect`) — загрузить."""
        if self._catalogue is None:
            self._load_config()
        assert self._catalogue is not None
        return self._catalogue

    # ------------------------------------------------------------------ #
    # Хуки HttpProvider
    # ------------------------------------------------------------------ #
    def _chat_url(self) -> str:
        return self._ensure_catalogue().chat_url

    def _auth_headers(self, *, force: bool = False) -> dict[str, str]:
        """``Authorization`` с действующим токеном. ``force`` (чат ответил 401) —
        перечитать файл: там тот же токен — ошибка с просьбой пройти авторизацию."""
        token = self._current_token()
        if force:
            token = self._reload_rejected(token)
        return {"Authorization": f"Bearer {token.access_token}"}

    @property
    def _refreshable(self) -> bool:
        return True

    def _chat_headers(self, *, stream: bool) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Session-Id": self._session_id,
        }

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
        catalogue = self._ensure_catalogue()
        name = model or self.model
        info = catalogue.get(name)
        return openai_chat.chat_payload(
            messages,
            tools,
            model=name,
            temperature=temperature,
            stream=stream and (info is None or info.streams),
            tool_choice=tool_choice,
            reasoning=reasoning,
            options=info.sampling if info is not None else None,
        )

    def _has_reasoning(self, payload: Mapping[str, Any]) -> bool:
        return openai_chat.has_reasoning(payload)

    def _parse_response(self, data: dict[str, Any]) -> AssistantTurn:
        return openai_chat.parse_response(data)

    def _consume_sse(self, resp: httpx.Response, emit: Callable[[str], None]) -> AssistantTurn:
        content_type = resp.headers.get("content-type", "")
        return openai_chat.consume_stream(resp.iter_lines(), emit, content_type=content_type)

    def _probe_payload(self, text: str, model: str) -> dict[str, Any]:
        info = self._ensure_catalogue().get(model)
        return openai_chat.probe_payload(text, model, stream=info is None or info.streams)

    def _probe_usage(self, resp: httpx.Response) -> Usage:
        content_type = resp.headers.get("content-type", "")
        turn = openai_chat.consume_stream(
            resp.text.splitlines(), lambda _s: None, content_type=content_type
        )
        return turn.usage

    # ------------------------------------------------------------------ #
    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        *,
        model: str | None = None,
        temperature: float = 0.2,
        tool_choice: ToolChoice = "auto",
        on_delta: Callable[[str], None] | None = None,
        on_notice: Callable[[str], None] | None = None,
    ) -> AssistantTurn:
        self._ensure_catalogue()  # модель по умолчанию — из каталога
        return super().stream(
            messages,
            tools,
            model=model,
            temperature=temperature,
            tool_choice=tool_choice,
            on_delta=on_delta,
            on_notice=on_notice,
        )

    def measure_prompt(self, text: str, *, model: str | None = None) -> int:
        self._ensure_catalogue()
        return super().measure_prompt(text, model=model)

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        *,
        model: str | None = None,
        temperature: float = 0.2,
        tool_choice: ToolChoice = "auto",
        on_notice: Callable[[str], None] | None = None,
    ) -> AssistantTurn:
        """Через поток: бэкенд рассчитан на него, а ответ без потока разбирается тем же
        кодом."""
        return self.stream(
            messages,
            tools,
            model=model,
            temperature=temperature,
            tool_choice=tool_choice,
            on_notice=on_notice,
        )
