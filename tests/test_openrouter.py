"""Адаптер OpenRouter для live-тестов (``tests/openrouter.py``) без сети: httpx.MockTransport."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import httpx
import pytest
from openrouter import (
    OPENROUTER_URL,
    LiveUnavailableError,
    OpenRouterProvider,
    live_provider_kind,
    open_live_provider,
    openrouter_config,
)

from gigacrew.config import Config
from gigacrew.llm.base import ModelInfo
from gigacrew.llm.types import Message, ToolSpec

_KEY = "sk-or-test"
_MODEL = "vendor/model"
_CATALOGUE = MappingProxyType({
    "data": [
        {"id": _MODEL, "context_length": 131072},
        {"id": "vendor/unknown-window"},
        {"name": "без id"},
        "мусор",
    ],
})
_TOOL = ToolSpec(
    name="get_weather",
    description="Погода в городе",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
)


def _sse(*chunks: dict[str, Any]) -> bytes:
    events = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
    return "".join([*events, "data: [DONE]\n\n"]).encode()


class _Server:
    """OpenRouter: каталог и поток чата; запоминает запросы."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=dict(_CATALOGUE))
        function = {"name": "get_weather", "arguments": '{"city": "Сочи"}'}
        call = {"index": 0, "id": "call_1", "type": "function", "function": function}
        stream = _sse(
            {"choices": [{"delta": {"reasoning": "думаю"}}]},
            {"choices": [{"delta": {"content": "Сейчас "}}]},
            {"choices": [{"delta": {"tool_calls": [call]}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 5}},
        )
        return httpx.Response(200, content=stream, headers={"content-type": "text/event-stream"})


@pytest.fixture
def server() -> _Server:
    return _Server()


@pytest.fixture
def provider(project: Path, server: _Server) -> OpenRouterProvider:
    config = openrouter_config(Config(project_root=project, model=_MODEL))
    return OpenRouterProvider(config, _KEY, transport=httpx.MockTransport(server))


def test_stream_turn_with_tool_call(provider: OpenRouterProvider, server: _Server) -> None:
    """Чат — Chat Completions на OpenRouter с ключом Bearer; поток собирается в ход."""
    deltas: list[str] = []
    turn = provider.stream(
        [Message(role="user", content="Погода в Сочи?")],
        tools=[_TOOL],
        on_delta=deltas.append,
    )
    request = server.requests[-1]
    assert str(request.url) == f"{OPENROUTER_URL}/chat/completions"
    assert request.headers["Authorization"] == f"Bearer {_KEY}"
    body = json.loads(request.content)
    assert body["model"] == _MODEL
    assert body["stream"] is True
    assert body["tools"][0]["function"]["name"] == "get_weather"
    assert turn.finish_reason == "function_call"
    assert turn.message.content == "Сейчас "
    assert turn.message.reasoning == "думаю"
    assert deltas == ["Сейчас "]
    calls = turn.message.function_calls
    assert [(call.id, call.name, call.arguments) for call in calls] == [
        ("call_1", "get_weather", {"city": "Сочи"}),
    ]
    assert turn.usage.prompt_tokens == 12


def test_catalogue_gives_windows(provider: OpenRouterProvider, server: _Server) -> None:
    """Каталог — после connect; записи без id пропускаются, окно может быть неизвестно."""
    assert provider.known_models() is None
    assert provider.list_models() == [_MODEL]
    provider.connect()
    provider.connect()  # каталог загружается один раз
    assert len(server.requests) == 1
    assert server.requests[0].headers["Authorization"] == f"Bearer {_KEY}"
    assert provider.known_models() == [
        ModelInfo(_MODEL, 131072),
        ModelInfo("vendor/unknown-window", None),
    ]
    assert provider.list_models() == [_MODEL, "vendor/unknown-window"]


def test_config_drops_api_credentials(project: Path) -> None:
    """Ключ, сертификаты и CA шлюза API v2 в OpenRouter не уходят; TLS проверяется."""
    config = Config(
        project_root=project,
        model=_MODEL,
        access_key="secret",
        cert="client.pem",
        key="client.key",
        auth_url="https://auth.test",
        ca_bundle="root.pem",
        verify_ssl=False,
        temperature=0.7,
    )
    router = openrouter_config(config)
    assert router.base_url == OPENROUTER_URL
    assert router.auth_mode == "none"
    assert (router.access_key, router.cert, router.key, router.ca_bundle) == (None,) * 4
    assert router.verify_ssl
    assert router.temperature == pytest.approx(0.7)


def test_live_provider_selection() -> None:
    """GIGACREW_TEST_PROVIDER выбирает провайдера; без ключа OpenRouter — пропуск."""
    assert live_provider_kind({}) == "api"
    assert live_provider_kind({"GIGACREW_TEST_PROVIDER": " OpenRouter "}) == "openrouter"
    with pytest.raises(ValueError, match="openruoter"):
        live_provider_kind({"GIGACREW_TEST_PROVIDER": "openruoter"})
    with pytest.raises(LiveUnavailableError, match="GIGACREW_OPENROUTER_API_KEY"):
        open_live_provider(_MODEL, {"GIGACREW_TEST_PROVIDER": "openrouter"})
    env = {"GIGACREW_TEST_PROVIDER": "openrouter", "OPENROUTER_API_KEY": _KEY}
    config, provider = open_live_provider(_MODEL, env)
    provider.close()
    assert isinstance(provider, OpenRouterProvider)
    assert provider.model == _MODEL
    assert config.base_url == OPENROUTER_URL
