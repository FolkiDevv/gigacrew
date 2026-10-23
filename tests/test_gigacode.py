"""Офлайн-тесты провайдера бэкенда GigaCode на подменном транспорте.

Транспорт — httpx.MockTransport; проверяются каталог моделей, токен из файла
официального CLI, запросы чата.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, override

import httpx
import pytest
from fakes import RecordingEvents

from gigacrew.agent.loop import Agent
from gigacrew.config import Config
from gigacrew.llm.base import ModelInfo
from gigacrew.llm.gigacode import (
    DEFAULT_CONTEXT_WINDOW,
    GigaCodeProvider,
    config_endpoint,
    parse_catalogue,
)
from gigacrew.llm.gigacode_token import Token, load_token, parse_token
from gigacrew.llm.transport import ApiError
from gigacrew.llm.types import FunctionCall, Message, ToolSpec
from gigacrew.tools.base import build_default_registry

BASE = "https://code.test"
CONFIG = f"{BASE}/api/v2/config"
CHAT = f"{BASE}/api/v2/chat/completions"
AGENT = "demo-agent/1.0 (test)"
_CODER_TOP_P = 0.9  # top_p модели coder из каталога
_TEMPERATURE = 0.4  # температура запроса в тесте заголовков и тела

CATALOGUE: Final[Mapping[str, Any]] = MappingProxyType({
    "default_model": "coder",
    "features": ["serve"],
    "agent": {
        "models": [
            {
                "id": "coder",
                "description": "Coder",
                "plugin_side": {"context_window_size": 131_072},
                "ai_api": {
                    "max_tokens": 32000,
                    "temperature": 0.3,
                    "top_p": _CODER_TOP_P,
                    "stream": True,
                },
            },
            {"id": "fast", "ai_api": {"stream": False}},
        ],
    },
    "chat": {"url": CHAT, "models": []},
})

TOOL = ToolSpec(
    name="read_file",
    description="Читает файл",
    parameters={"type": "object", "properties": {"path": {"type": "string"}}},
)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    return sleeps


def _sse_payload(chunk: str | dict[str, Any]) -> str:
    if isinstance(chunk, str):
        return chunk
    return json.dumps(chunk, ensure_ascii=False)


def _sse(*chunks: str | dict[str, Any]) -> bytes:
    payloads = (_sse_payload(chunk) for chunk in chunks)
    return "".join(f"data: {payload}\n\n" for payload in payloads).encode()


def _text(text: str, *, prompt_tokens: int = 10) -> bytes:
    return _sse(
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": text}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 2}},
        "[DONE]",
    )


def _tool_call(name: str, args: dict[str, Any], call_id: str = "c1") -> bytes:
    call = {
        "index": 0,
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }
    return _sse(
        {"choices": [{"index": 0, "delta": {"tool_calls": [call]}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        "[DONE]",
    )


class Backend:
    """Бэкенд GigaCode: конфигурация и чат принимают только токены из ``valid``."""

    def __init__(
        self,
        *,
        chat: Iterable[bytes | httpx.Response] = (),
        catalogue: Mapping[str, Any] | None = None,
        valid: Iterable[str] = ("tok-1",),
    ) -> None:
        self.catalogue = catalogue or CATALOGUE
        self.valid: set[str] = set(valid)
        self.chat_rejects: set[str] = set()  # токены, которые не принимает чат
        self.requests: list[httpx.Request] = []
        self.chat_bodies: list[dict[str, Any]] = []
        self.chat = list(chat)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)
        token = request.headers.get("Authorization", "").removeprefix("Bearer ")
        if url == CONFIG and request.method == "GET":
            if token in self.valid:
                return httpx.Response(200, json=dict(self.catalogue))
            return httpx.Response(401, json={"error": "unauthorized"})
        if url == CHAT:
            if token not in self.valid or token in self.chat_rejects:
                return httpx.Response(401, json={"error": "unauthorized"})
            self.chat_bodies.append(json.loads(request.content))
            reply = self.chat.pop(0) if self.chat else _text("ok")
            if isinstance(reply, httpx.Response):
                return reply
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=reply)
        return httpx.Response(404, text=f"нет {url}")

    def count(self, url: str) -> int:
        return sum(str(request.url) == url for request in self.requests)


def token_file(
    token: str = "tok-1",
    expires_ms: float | None = None,
    endpoint: str = CONFIG,
) -> dict[str, Any]:
    """Содержимое файла токена официального CLI."""
    entry: dict[str, Any] = {
        "accessToken": token,
        "authBundle": {"x": 1},
        "configEndpoint": endpoint,
    }
    if expires_ms is not None:
        entry["expiresAtUnixMs"] = expires_ms
    return {"version": 1, "token": entry}


def write_token(
    tmp_path: Path,
    token: str = "tok-1",
    expires_at: float | None = None,
    **kw: str,
) -> Path:
    """Файл токена официального CLI в ``<GIGACODE_HOME>/gigacode_token.json``."""
    home = tmp_path / "cli-home"
    home.mkdir(exist_ok=True)
    expires_ms = None if expires_at is None else expires_at * 1000
    path = home / "gigacode_token.json"
    path.write_text(json.dumps(token_file(token, expires_ms, **kw)), encoding="utf-8")
    return path


def _config(tmp_path: Path, **kw: Any) -> Config:
    settings: dict[str, Any] = {
        "provider": "gigacode",
        "base_url": BASE,
        "user_agent": AGENT,
        "gigacode_home": str(tmp_path / "cli-home"),
        "project_root": tmp_path,
    }
    settings.update(kw)
    return Config(**settings)


def _provider(
    backend: Backend,
    tmp_path: Path,
    *,
    token: str | None = "tok-1",
    **kw: Any,
) -> GigaCodeProvider:
    if token is not None:
        write_token(tmp_path, token)
    return GigaCodeProvider(_config(tmp_path, **kw), transport=httpx.MockTransport(backend.handler))


def _msgs() -> list[Message]:
    return [Message(role="user", content="привет")]


# ------------------------------- адрес ------------------------------- #
@pytest.mark.parametrize(
    ("base", "url"),
    [
        ("https://h", "https://h/api/v2/config"),
        ("https://h/", "https://h/api/v2/config"),
        ("https://h/api/v3", "https://h/api/v3/config"),
        ("https://h/api/v2/config", "https://h/api/v2/config"),
        ("https://h/prefix", "https://h/prefix/api/v2/config"),
    ],
)
def test_config_endpoint(base: str, url: str) -> None:
    assert config_endpoint(base) == url


# ------------------------------ каталог ------------------------------ #
def test_catalogue_from_the_contract_example() -> None:
    catalogue = parse_catalogue(CATALOGUE, CONFIG)
    assert [model.id for model in catalogue.models] == ["coder", "fast"]
    coder, fast = catalogue.models
    assert coder.context_window == 131_072
    assert fast.context_window == DEFAULT_CONTEXT_WINDOW
    assert dict(coder.sampling) == {"top_p": _CODER_TOP_P}  # temperature и max_tokens — свои
    assert coder.streams
    assert not fast.streams
    assert catalogue.default_model == "coder"
    assert catalogue.chat_url == CHAT


def test_catalogue_sources_in_order() -> None:
    def ids(catalogue: Mapping[str, Any]) -> list[str]:
        return [model.id for model in parse_catalogue(catalogue, CONFIG).models]

    chat_and_agent = {"chat": {"models": [{"id": "a"}]}, "agent": {"models": [{"id": "b"}]}}
    assert ids(chat_and_agent) == ["a"]
    assert ids({"chat": {"models": []}, "agent": {"models": [{"id": "b"}]}}) == ["b"]
    assert ids({"models": [{"id": "c", "context_window_size": 8192}]}) == ["c"]
    assert (
        parse_catalogue({"models": [{"id": "c", "context_window_size": 8192}]}, CONFIG)
        .models[0]
        .context_window
        == 8192
    )


def test_model_id_from_the_model_key() -> None:
    """Реальный бэкенд называет id модели model; id по контракту — важнее."""
    raw_model = {
        "model": "Qwen3.6-35B",
        "description": "Qwen3.6-35B-262k",
        "default_system_message": "",
        "ai_api": {
            "frequency_penalty": 0,
            "max_tokens": 32768,
            "n": 1,
            "presence_penalty": 0,
            "stream": True,
            "temperature": 0.6,
        },
    }
    catalogue = parse_catalogue(
        {"agent": {"models": [raw_model]}, "default_model": "Qwen3.6-35B"},
        CONFIG,
    )
    assert len(catalogue.models) == 1
    model = catalogue.models[0]
    assert model.id == catalogue.default_model == "Qwen3.6-35B"
    assert model.context_window == DEFAULT_CONTEXT_WINDOW
    assert model.streams
    assert dict(model.sampling) == {"frequency_penalty": 0, "presence_penalty": 0}
    assert parse_catalogue({"models": [{"id": "a", "model": "b"}]}, CONFIG).models[0].id == "a"


@pytest.mark.parametrize(
    ("catalogue", "message"),
    [
        ({}, "ни одной модели"),
        ({"models": [{"model": ""}]}, "id или model"),
        ({"models": [{"description": "без id"}]}, "id или model"),
        ({"models": []}, "ни одной модели"),
        ({"models": [{"id": "a"}, {"id": "a"}]}, "повторяется"),
        ({"models": [{"id": ""}]}, "id"),
        ({"models": [{"id": "a\nb"}]}, "id"),
        ({"models": ["a"]}, "id"),
        ({"models": [{"id": "a"}], "chat": {"url": "ftp://h/chat"}}, "адрес чата"),
        (  # конфигурация по https, а чат — по http: токен ушёл бы открытым текстом
            {"models": [{"id": "a"}], "chat": {"url": "http://code.test/api/v2/chat/completions"}},
            "открытым текстом",
        ),
        ({"models": [{"id": "a"}], "chatEndpoint": "http://llm.test/v1/c"}, "открытым текстом"),
    ],
)
def test_bad_catalogues(catalogue: dict[str, Any], message: str) -> None:
    with pytest.raises(ApiError, match=message):
        parse_catalogue(catalogue, CONFIG)


def test_catalogue_default_model_and_chat_url() -> None:
    models = [{"id": "a"}, {"id": "b"}]
    pick = parse_catalogue(
        {"models": models, "chat": {"default_model": "b"}, "default_model": "a"},
        CONFIG,
    )
    assert pick.default_model == "b"
    assert parse_catalogue({"models": models, "default_model": "b"}, CONFIG).default_model == "b"
    assert parse_catalogue({"models": models, "default_model": "x"}, CONFIG).default_model == "a"
    assert parse_catalogue({"models": models, "default_model": ["b"]}, CONFIG).default_model == "a"
    assert parse_catalogue({"models": models}, CONFIG).chat_url == CHAT
    relative = parse_catalogue({"models": models, "chatEndpoint": "chat/completions"}, CONFIG)
    assert relative.chat_url == f"{BASE}/api/v2/chat/completions"
    other = parse_catalogue({"models": models, "chat": {"url": "https://llm.test/v1/c"}}, CONFIG)
    assert other.chat_url == "https://llm.test/v1/c"
    # конфигурация по http (локальный стенд) — и чат по http можно
    local = "http://127.0.0.1:8000/api/v2/config"
    plain = parse_catalogue({"models": models, "chat": {"url": "http://127.0.0.1:8000/c"}}, local)
    assert plain.chat_url == "http://127.0.0.1:8000/c"


def test_catalogue_sampling_keeps_only_finite_numbers() -> None:
    model = {
        "id": "m",
        "ai_api": {
            "top_p": 0.5,
            "top_k": 40,
            "seed": 7,
            "frequency_penalty": float("nan"),
            "presence_penalty": True,
            "repetition_penalty": "1.1",
            "n": 2,
            "max_lines": 10,
            "max_tokens": 100,
            "temperature": 1.0,
        },
    }
    models = parse_catalogue({"models": [model]}, CONFIG).models
    assert len(models) == 1
    parsed = models[0]
    assert dict(parsed.sampling) == {"top_p": 0.5, "top_k": 40, "seed": 7}


# ------------------------------ токен ------------------------------ #
def test_parse_token_of_the_cli_format() -> None:
    stored = token_file(expires_ms=1_900_000_000_000)
    entry = stored["token"]
    entry["resourceUrl"] = "https://res.test"
    assert parse_token(stored) == Token("tok-1", 1_900_000_000, CONFIG)
    entry.pop("expiresAtUnixMs")
    entry.pop("configEndpoint")
    assert parse_token(stored) == Token("tok-1")  # срока нет — до 401


@pytest.mark.parametrize(
    ("stored", "reason"),
    [
        ([], "не JSON-объект"),
        ({"token": {"accessToken": "t"}}, "версия"),
        ({"version": 2, "token": {"accessToken": "t"}}, "версия"),
        ({"version": 1}, "нет записи token"),
        ({"version": 1, "token": {"access_token": "t"}}, "accessToken"),
        ({"version": 1, "token": {"accessToken": "с пробелом"}}, "accessToken"),
        ({"version": 1, "token": {"accessToken": 1}}, "accessToken"),
        ({"version": 1, "token": {"accessToken": "t", "expiresAtUnixMs": "1"}}, "expires"),
        ({"version": 1, "token": {"accessToken": "t", "expiresAtUnixMs": True}}, "expires"),
        ({"version": 1, "token": {"accessToken": "t", "configEndpoint": ""}}, "configEndpoint"),
    ],
)
def test_parse_token_rejects(stored: object, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        parse_token(stored)


def test_load_token_errors_ask_to_log_in_to_gigacode(tmp_path: Path) -> None:
    path = tmp_path / "gigacode_token.json"
    with pytest.raises(ApiError, match=r"нет токена GigaCode.*авторизацию в GigaCode"):
        load_token(path, CONFIG)
    path.write_text("{битый", encoding="utf-8")
    with pytest.raises(ApiError, match="авторизацию в GigaCode"):
        load_token(path, CONFIG)
    path.write_text(json.dumps(token_file("t", (time.time() + 30) * 1000)))
    with pytest.raises(ApiError, match=r"истёк.*авторизацию в GigaCode"):  # запас — минута
        load_token(path, CONFIG)
    path.write_text(json.dumps(token_file("t", (time.time() + 3600) * 1000)))
    assert load_token(path, CONFIG).access_token == "t"
    assert load_token(path, f"{CONFIG}/").access_token == "t"


def test_unreadable_token_file_is_an_api_error(tmp_path: Path) -> None:
    """Файл не читается (каталог, нет прав) — ApiError, а не OSError мимо REPL."""
    path = tmp_path / "gigacode_token.json"
    path.mkdir()
    with pytest.raises(ApiError, match="не удалось прочитать"):
        load_token(path, CONFIG)


def test_token_of_another_backend_is_not_sent(tmp_path: Path) -> None:
    backend = Backend()
    write_token(tmp_path, endpoint="https://other.test/api/v2/config")
    provider = GigaCodeProvider(_config(tmp_path), transport=httpx.MockTransport(backend.handler))
    with pytest.raises(ApiError, match=r"выдан для https://other.test.*GIGACREW_API_URL"):
        provider.connect()
    assert not backend.requests


def test_constructor_does_no_io(tmp_path: Path) -> None:
    backend = Backend()
    provider = _provider(backend, tmp_path, token=None)
    assert not backend.requests
    assert not (tmp_path / "cli-home").exists()
    assert provider.token_file == tmp_path / "cli-home" / "gigacode_token.json"
    assert not provider.list_models()
    assert provider.known_models() is None
    assert not provider.model
    provider.close()


def test_token_file_defaults_to_the_cli_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    provider = GigaCodeProvider(_config(tmp_path, gigacode_home=None))
    assert provider.token_file == tmp_path / ".gigacode" / "gigacode_token.json"
    provider.close()


def test_constructor_requires_settings(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="GIGACREW_USER_AGENT"):
        _provider(Backend(), tmp_path, user_agent=None)
    with pytest.raises(RuntimeError, match="GIGACREW_API_URL"):
        _provider(Backend(), tmp_path, base_url="")
    with pytest.raises(RuntimeError, match="GIGACREW_PROVIDER"):
        _provider(Backend(), tmp_path, provider="api")


def test_connect_reads_the_token_and_the_catalogue(tmp_path: Path) -> None:
    backend = Backend()
    provider = _provider(backend, tmp_path)
    provider.connect()
    assert backend.requests[0].headers["Authorization"] == "Bearer tok-1"
    assert provider.model == "coder"
    assert provider.list_models() == ["coder", "fast"]
    assert provider.known_models() == [
        ModelInfo("coder", 131_072),
        ModelInfo("fast", DEFAULT_CONTEXT_WINDOW),
    ]
    provider.connect()  # каталог есть — без сети
    assert backend.count(CONFIG) == 1
    provider.close()


def test_connect_without_a_token_asks_to_log_in(tmp_path: Path) -> None:
    backend = Backend()
    provider = _provider(backend, tmp_path, token=None)
    with pytest.raises(ApiError, match="авторизацию в GigaCode"):
        provider.connect()
    assert not backend.requests
    write_token(tmp_path, "tok-1")  # пользователь вошёл в GigaCode
    provider.connect()
    assert provider.model == "coder"


def test_expired_token_asks_to_log_in_and_a_new_one_is_picked_up(tmp_path: Path) -> None:
    backend = Backend()
    provider = _provider(backend, tmp_path)
    provider.connect()
    write_token(tmp_path, "tok-1", expires_at=time.time() - 10)
    with pytest.raises(ApiError, match=r"истёк.*авторизацию в GigaCode"):
        provider.connect()
    backend.valid.add("tok-2")
    write_token(tmp_path, "tok-2", expires_at=time.time() + 3600)
    provider.connect()
    assert provider.stream(_msgs()).message.content == "ok"
    assert backend.requests[-1].headers["Authorization"] == "Bearer tok-2"


def test_token_expiring_during_a_session_is_reread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = Backend(valid=("tok-1", "tok-2"))
    write_token(tmp_path, "tok-1", expires_at=time.time() + 3600)
    provider = GigaCodeProvider(_config(tmp_path), transport=httpx.MockTransport(backend.handler))
    provider.connect()
    write_token(tmp_path, "tok-2", expires_at=time.time() + 7200)
    later = time.time() + 3600
    monkeypatch.setattr(time, "time", lambda: later)
    assert provider.stream(_msgs()).message.content == "ok"
    assert backend.requests[-1].headers["Authorization"] == "Bearer tok-2"


def test_token_rejected_by_config_asks_to_log_in(tmp_path: Path) -> None:
    backend = Backend(valid=())
    provider = _provider(backend, tmp_path, token="revoked")
    with pytest.raises(ApiError, match=r"не принял токен \(401\).*авторизацию в GigaCode"):
        provider.connect()
    assert backend.count(CONFIG) == 1  # тот же токен второй раз не отправляется


def test_config_errors(tmp_path: Path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="forbidden")

    write_token(tmp_path)
    provider = GigaCodeProvider(_config(tmp_path), transport=httpx.MockTransport(handler))
    with pytest.raises(ApiError, match=r"конфигурацию GigaCode \(403\)"):
        provider.connect()


def test_chat_http401_rereads_the_token_file(tmp_path: Path) -> None:
    backend = Backend()
    provider = _provider(backend, tmp_path)
    provider.connect()
    assert provider.stream(_msgs()).message.content == "ok"
    backend.valid = {"tok-2"}  # бэкенд отозвал tok-1, пользователь вошёл заново
    write_token(tmp_path, "tok-2")
    assert provider.stream(_msgs()).message.content == "ok"
    assert backend.requests[-1].headers["Authorization"] == "Bearer tok-2"
    backend.valid = {"tok-3"}
    write_token(tmp_path, "tok-3")
    assert provider.complete(_msgs()).message.content == "ok"  # и без потока — так же


def test_chat_http401_with_the_same_token_asks_to_log_in(tmp_path: Path) -> None:
    backend = Backend()
    provider = _provider(backend, tmp_path)
    provider.connect()
    backend.chat_rejects.add("tok-1")  # конфигурацию даёт, а чат не пускает
    with pytest.raises(ApiError, match=r"не принял токен.*авторизацию в GigaCode"):
        provider.stream(_msgs())
    assert backend.count(CHAT) == 1


@pytest.mark.usefixtures("_no_sleep")
def test_chat_errors_and_retries(tmp_path: Path) -> None:
    backend = Backend(
        chat=[
            httpx.Response(529, json={"error": "overloaded"}),
            _text("после перегрузки"),
            httpx.Response(403, text="нет доступа к модели"),
        ],
    )
    provider = _provider(backend, tmp_path)
    provider.connect()
    assert provider.stream(_msgs()).message.content == "после перегрузки"
    with pytest.raises(ApiError, match="403"):
        provider.stream(_msgs())


def test_chat_request_headers_and_body(tmp_path: Path) -> None:
    backend = Backend()
    provider = _provider(backend, tmp_path)
    provider.connect()
    deltas: list[str] = []
    provider.stream(_msgs(), [TOOL], temperature=_TEMPERATURE, on_delta=deltas.append)
    provider.stream(_msgs())
    chat = [request for request in backend.requests if str(request.url) == CHAT]
    assert chat[0].headers["Accept"] == "application/json"
    assert chat[0].headers["Content-Type"] == "application/json"
    assert chat[0].headers["User-Agent"] == AGENT
    assert chat[0].headers["X-Session-Id"] == chat[1].headers["X-Session-Id"]
    body = backend.chat_bodies[0]
    assert body["model"] == "coder"
    assert body["temperature"] == _TEMPERATURE
    assert body["top_p"] == _CODER_TOP_P
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["tool_choice"] == "auto"
    assert body["tools"][0]["function"]["name"] == "read_file"
    assert "max_tokens" not in body
    assert body["messages"] == [{"role": "user", "content": "привет"}]
    assert deltas == ["ok"]
    # User-Agent из GIGACREW_USER_AGENT — у всех запросов к бэкенду
    assert {request.headers["User-Agent"] for request in backend.requests} == {AGENT}
    assert {str(request.url) for request in backend.requests} == {CONFIG, CHAT}


def test_complete_goes_through_the_stream(tmp_path: Path) -> None:
    backend = Backend()
    provider = _provider(backend, tmp_path, model="coder")
    provider.connect()
    turn = provider.complete(_msgs(), [TOOL])
    assert turn.message.content == "ok"
    assert turn.usage.prompt_tokens == 10
    assert backend.chat_bodies[0]["stream"] is True


class _BrokenStream(httpx.SyncByteStream):
    """Тело ответа: кусок текста, потом обрыв соединения."""

    @override
    def __iter__(self) -> Iterator[bytes]:
        yield _sse({"choices": [{"index": 0, "delta": {"content": "Сводка, часть 1"}}]})
        raise httpx.ReadError("connection reset")


def _broken() -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        stream=_BrokenStream(),
    )


def test_complete_is_retried_after_a_drop_mid_answer(tmp_path: Path) -> None:
    """complete() (рецензент, сжатие) ничего не показывает — обрыв повторяется.

    Обрыв после первого куска — обычный транзиентный сбой, повтор не дублирует вывод.
    """
    backend = Backend(chat=[_broken(), _text("сводка целиком")])
    provider = _provider(backend, tmp_path)
    provider.connect()
    assert provider.complete(_msgs()).message.content == "сводка целиком"
    assert backend.count(CHAT) == 2


def test_stream_shown_text_is_not_retried_after_a_drop(tmp_path: Path) -> None:
    backend = Backend(chat=[_broken(), _text("второй раз")])
    provider = _provider(backend, tmp_path)
    provider.connect()
    deltas: list[str] = []
    with pytest.raises(ApiError, match="Сетевой сбой во время потоковой передачи"):
        provider.stream(_msgs(), on_delta=deltas.append)
    assert deltas == ["Сводка, часть 1"]
    assert backend.count(CHAT) == 1


class _CutAfterEnd(httpx.SyncByteStream):
    """Ответ дошёл до конца (вызов, finish_reason, usage), а соединение закрыто нечисто."""

    @override
    def __iter__(self) -> Iterator[bytes]:
        yield _tool_call("run_shell", {"command": "pytest"}).removesuffix(b"data: [DONE]\n\n")
        yield _sse({"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 3}})
        raise httpx.RemoteProtocolError("peer closed connection without sending complete body")


def test_unclean_close_after_the_end_keeps_the_answer(tmp_path: Path) -> None:
    """Обрыв после окончания ответа — ответ целиком.

    Иначе показанный текст стал бы ошибкой, а не показанный ответ ушёл бы повтором
    и был бы оплачен дважды.
    """
    cut = httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=_CutAfterEnd())
    backend = Backend(chat=[cut])
    provider = _provider(backend, tmp_path)
    provider.connect()
    turn = provider.stream(_msgs(), [TOOL])
    assert turn.finish_reason == "function_call"
    assert turn.usage.prompt_tokens == 12
    assert turn.message.function_calls == [
        FunctionCall(name="run_shell", arguments={"command": "pytest"}, id="c1"),
    ]
    assert backend.count(CHAT) == 1


def test_model_without_stream_gets_a_whole_answer(tmp_path: Path) -> None:
    whole = httpx.Response(
        200,
        json={
            "choices": [{"index": 0, "message": {"content": "целиком"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 1},
        },
    )
    backend = Backend(chat=[whole])
    provider = _provider(backend, tmp_path)
    provider.connect()
    deltas: list[str] = []
    turn = provider.stream(_msgs(), model="fast", on_delta=deltas.append)
    assert turn.message.content == "целиком"
    assert deltas == ["целиком"]
    assert backend.chat_bodies[0]["stream"] is False
    assert "stream_options" not in backend.chat_bodies[0]


def test_rejected_reasoning_is_dropped(tmp_path: Path) -> None:
    history = [
        Message(role="user", content="что в a?"),
        Message(
            role="assistant",
            reasoning="прочитаю a",
            function_calls=[FunctionCall(name="read_file", arguments={"path": "a"}, id="c1")],
        ),
        Message(role="function", name="read_file", content="x", call_id="c1"),
    ]
    backend = Backend(chat=[httpx.Response(400, json={"error": "unknown field"}), _text("ok")])
    provider = _provider(backend, tmp_path)
    provider.connect()
    notices: list[str] = []
    assert provider.stream(history, on_notice=notices.append).message.content == "ok"
    assert "reasoning_content" not in backend.chat_bodies[-1]["messages"][1]
    assert "не принял рассуждения" in notices[0]


def test_measure_prompt_reads_usage_from_the_stream(tmp_path: Path) -> None:
    backend = Backend(chat=[_text(".", prompt_tokens=1234)])
    provider = _provider(backend, tmp_path)
    provider.connect()
    assert provider.supports_measure
    assert provider.measure_prompt("текст") == 1234
    body = backend.chat_bodies[0]
    assert body["max_tokens"] == 1
    assert body["model"] == "coder"
    assert body["stream"] is True


def test_probe_of_a_model_without_stream_is_not_streamed(tmp_path: Path) -> None:
    """Модель с ``ai_api.stream: false`` и пробу получает целым ответом, как и чат."""
    whole = httpx.Response(
        200,
        json={
            "choices": [{"index": 0, "message": {"content": "."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 77, "completion_tokens": 1},
        },
    )
    backend = Backend(chat=[whole])
    provider = _provider(backend, tmp_path)
    provider.connect()
    assert provider.measure_prompt("x", model="fast") == 77
    body = backend.chat_bodies[0]
    assert body["stream"] is False
    assert "stream_options" not in body


def test_request_without_connect_loads_the_catalogue(tmp_path: Path) -> None:
    backend = Backend()
    provider = _provider(backend, tmp_path)
    assert provider.stream(_msgs()).message.content == "ok"
    assert backend.chat_bodies[0]["model"] == "coder"


def test_agent_turn_with_a_tool_call(tmp_path: Path) -> None:
    """Ход агента через настоящий провайдер: вызов read_file, затем ответ."""
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    backend = Backend(chat=[_tool_call("read_file", {"path": "a.txt"}), _text("там hello")])
    provider = _provider(backend, tmp_path, stream=True)
    provider.connect()
    config = _config(tmp_path, model=provider.model, stream=True)
    events = RecordingEvents()
    answer = Agent(provider, build_default_registry(), config, events).run_turn("что в a.txt?")
    assert answer == "там hello"
    second = backend.chat_bodies[1]["messages"]
    assert second[-2]["tool_calls"][0]["id"] == "c1"
    assert second[-1]["role"] == "tool"
    assert second[-1]["tool_call_id"] == "c1"
    assert "hello" in second[-1]["content"]
    provider.close()


# --------------------------------------------------------------------------- #
# request_hook (хуки пользователя PreRequest)


class RecordingHook:
    """request_hook: запоминает (вид, причина), может обновить токен или отказать."""

    def __init__(
        self,
        tmp_path: Path,
        *,
        tokens: Iterable[str] = (),
        block: str | None = None,
    ) -> None:
        self.tmp_path = tmp_path
        self.calls: list[tuple[str, str]] = []
        self.tokens = list(tokens)  # какой токен записать в файл на очередном вызове
        self.block = block

    def __call__(self, kind: str, reason: str) -> str | None:
        self.calls.append((kind, reason))
        if self.tokens:
            write_token(self.tmp_path, self.tokens.pop(0))
        return self.block


def test_hook_runs_before_config_chat_and_probe(tmp_path: Path) -> None:
    backend = Backend()
    provider = _provider(backend, tmp_path)
    hook = RecordingHook(tmp_path)
    provider.request_hook = hook
    provider.connect()
    provider.stream(_msgs())
    provider.complete(_msgs())
    provider.measure_prompt("проба")
    assert hook.calls == [
        ("config", "request"),
        ("chat", "request"),
        ("chat", "request"),
        ("probe", "request"),
    ]


def test_hook_refreshed_token_is_sent_although_cached_one_is_valid(tmp_path: Path) -> None:
    backend = Backend(valid=("tok-1", "tok-2"))
    provider = _provider(backend, tmp_path)
    provider.connect()  # tok-1 в памяти, не истёк
    provider.request_hook = RecordingHook(tmp_path, tokens=["tok-2"])
    provider.stream(_msgs())
    assert backend.requests[-1].headers["Authorization"] == "Bearer tok-2"


def test_hook_runs_again_before_retry_after_http401(tmp_path: Path) -> None:
    backend = Backend(valid=("tok-1", "tok-2"))
    provider = _provider(backend, tmp_path)
    provider.connect()
    backend.chat_rejects.add("tok-1")
    # первый вызов файл не меняет, второй (перед повтором после 401) кладёт новый токен
    hook = RecordingHook(tmp_path, tokens=["tok-1", "tok-2"])
    provider.request_hook = hook
    assert provider.stream(_msgs()).message.content == "ok"
    assert hook.calls == [("chat", "request"), ("chat", "unauthorized")]
    assert backend.requests[-1].headers["Authorization"] == "Bearer tok-2"


def test_hook_on_config_http401(tmp_path: Path) -> None:
    backend = Backend(valid=("tok-2",))
    provider = _provider(backend, tmp_path)  # в файле tok-1, бэкенд его не принимает
    hook = RecordingHook(tmp_path, tokens=["tok-1", "tok-2"])
    provider.request_hook = hook
    provider.connect()
    assert hook.calls == [("config", "request"), ("config", "unauthorized")]
    assert provider.list_models() == ["coder", "fast"]


def test_config_http401_with_a_new_token_still_asks_to_log_in(tmp_path: Path) -> None:
    """Хук положил новый токен, но и его конфигурация не приняла — просьба войти заново.

    Именно просьба войти, а не «Не удалось получить конфигурацию (401)».
    """
    backend = Backend(valid=())
    provider = _provider(backend, tmp_path)
    provider.request_hook = RecordingHook(tmp_path, tokens=["tok-1", "tok-2"])
    with pytest.raises(ApiError, match=r"не принял токен \(401\).*авторизацию в GigaCode"):
        provider.connect()
    assert backend.count(CONFIG) == 2


def test_hook_block_sends_nothing(tmp_path: Path) -> None:
    backend = Backend()
    provider = _provider(backend, tmp_path)
    provider.request_hook = RecordingHook(tmp_path, block="VPN не подключён")
    with pytest.raises(ApiError, match="остановлен хуком PreRequest: VPN не подключён"):
        provider.connect()
    assert not backend.requests


@pytest.mark.usefixtures("_no_sleep")
def test_hook_not_called_on_transient_retries(tmp_path: Path) -> None:
    backend = Backend(chat=[httpx.Response(503, json={"error": "busy"}), _text("ok")])
    provider = _provider(backend, tmp_path)
    provider.connect()
    hook = RecordingHook(tmp_path)
    provider.request_hook = hook
    assert provider.stream(_msgs()).message.content == "ok"
    assert hook.calls == [("chat", "request")]
    assert backend.count(CHAT) == 2


def test_connect_with_hook_leaves_expired_token_to_the_request(tmp_path: Path) -> None:
    backend = Backend(valid=("tok-1", "tok-2"))
    provider = _provider(backend, tmp_path)
    provider.connect()
    write_token(tmp_path, "tok-1", expires_at=int(time.time()) - 3600)  # истёк
    provider.request_hook = RecordingHook(tmp_path, tokens=["tok-2"])
    provider.connect()  # перед ходом в REPL: без хука упало бы «токен истёк»
    assert provider.stream(_msgs()).message.content == "ok"
    assert backend.requests[-1].headers["Authorization"] == "Bearer tok-2"
