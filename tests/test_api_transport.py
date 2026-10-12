"""Офлайн-тесты сетевого слоя API на подменном транспорте (httpx.MockTransport).

Проверяют повторы, обработку обрывов и превращение любых ошибок транспорта в
ApiError (иначе они роняют REPL).
"""

from __future__ import annotations

import json
import time

import httpx
import pytest

from gigacrew.config import Config
from gigacrew.llm import api
from gigacrew.llm.api import ApiError, ApiProvider
from gigacrew.llm.base import LLMError, PromptTooLong
from gigacrew.llm.types import Message, Usage

AUTH_URL = "https://auth.test/oauth"
BASE_URL = "https://api.test/v1"


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(api.time, "sleep", lambda _s: None)


def _provider(handler) -> ApiProvider:
    cfg = Config(access_key="dummy", auth_url=AUTH_URL, base_url=BASE_URL, model="M-default")
    return ApiProvider(cfg, transport=httpx.MockTransport(handler))


def _auth_response():
    return httpx.Response(
        200, json={"access_token": "t", "expires_at": int((time.time() + 3600) * 1000)}
    )


def _event(data: dict, name: str = "response.message.delta") -> str:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _text(text: str) -> dict:
    """Кусок потока / ответ v2 с текстом ассистента."""
    return {"messages": [{"role": "assistant", "content": [{"text": text}]}]}


class _BrokenStream(httpx.SyncByteStream):
    """Тело ответа: сначала отдаёт кусок, потом рвёт соединение."""

    def __init__(self, first: bytes):
        self._first = first

    def __iter__(self):
        yield self._first
        raise httpx.ReadError("connection reset")


def _msgs():
    return [Message(role="user", content="привет")]


def test_api_error_is_llm_error():
    assert issubclass(ApiError, LLMError)


def test_stream_broken_after_first_chunk_is_not_retried():
    chat_calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            return _auth_response()
        chat_calls.append(request)
        first = _event(_text("При")).encode()
        return httpx.Response(200, stream=_BrokenStream(first))

    deltas: list[str] = []
    p = _provider(handler)
    with pytest.raises(ApiError):
        p.stream(_msgs(), on_delta=deltas.append)
    assert deltas == ["При"]  # текст не продублирован
    assert len(chat_calls) == 1  # повтора не было
    p.close()


def test_stream_retries_connect_error_before_output():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            return _auth_response()
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise httpx.ConnectError("boom", request=request)
        body = _event({**_text("ok"), "finish_reason": "stop"}, "response.message.done")
        return httpx.Response(200, content=body.encode())

    deltas: list[str] = []
    p = _provider(handler)
    turn = p.stream(_msgs(), on_delta=deltas.append)
    assert turn.message.content == "ok"
    assert deltas == ["ok"]
    assert attempts["n"] == 2
    p.close()


@pytest.mark.parametrize("exc", [httpx.ProxyError, httpx.LocalProtocolError])
def test_non_retryable_transport_errors_wrapped(exc):
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            return _auth_response()
        raise exc("nope")

    p = _provider(handler)
    with pytest.raises(ApiError):
        p.complete(_msgs())
    with pytest.raises(ApiError):
        p.stream(_msgs())
    p.close()


def test_non_json_response_wrapped():
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            return _auth_response()
        return httpx.Response(200, text="<html>gateway</html>")

    p = _provider(handler)
    with pytest.raises(ApiError):
        p.complete(_msgs())
    p.close()


def test_auth_without_token_wrapped():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": "no"})

    p = _provider(handler)
    with pytest.raises(ApiError):
        p.complete(_msgs())
    p.close()


def test_list_models_http_error_wrapped():
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            return _auth_response()
        return httpx.Response(403, text="forbidden")

    p = _provider(handler)
    with pytest.raises(ApiError):
        p.list_models()
    p.close()


def test_model_override_per_call():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            return _auth_response()
        seen.append((str(request.url), json.loads(request.content)["model"]))
        return httpx.Response(
            200,
            json={
                **_text("x"),
                "finish_reason": "stop",
                "usage": {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9},
            },
        )

    p = _provider(handler)
    turn = p.complete(_msgs())
    p.complete(_msgs(), model="M-other")
    chat = "https://api.test/v2/chat/completions"
    assert seen == [(chat, "M-default"), (chat, "M-other")]
    assert turn.usage == Usage(prompt_tokens=7, completion_tokens=2, total_tokens=9)
    p.close()


def test_usage_from_raw_is_tolerant():
    assert Usage.from_raw(None) == Usage()
    assert Usage.from_raw("garbage") == Usage()  # type: ignore[arg-type]
    u = Usage.from_raw({"prompt_tokens": "5", "completion_tokens": None, "total_tokens": "x"})
    assert (u.prompt_tokens, u.completion_tokens, u.total_tokens) == (5, 0, 0)


# ------------------------------ замер окна ------------------------------ #
def _chat_reply(prompt_tokens: int) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            **_text("."),
            "usage": {"input_tokens": prompt_tokens, "output_tokens": 1},
        },
    )


def test_measure_prompt_sends_one_token_request():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            return _auth_response()
        bodies.append(json.loads(request.content))
        return _chat_reply(1234)

    p = _provider(handler)
    assert p.supports_measure
    assert p.measure_prompt("текст", model="M-probe") == 1234
    (body,) = bodies
    assert body["model_options"]["max_tokens"] == 1 and body["model"] == "M-probe"
    assert body["messages"] == [{"role": "user", "content": [{"text": "текст"}]}]
    assert "tools" not in body
    p.close()


@pytest.mark.parametrize("status", [400, 413, 422, 500, 504])
def test_measure_prompt_oversize_statuses(status):
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            return _auth_response()
        return httpx.Response(status, text="maximum context length is 32768 tokens")

    p = _provider(handler)
    with pytest.raises(PromptTooLong) as e:
        p.measure_prompt("x" * 10)
    assert e.value.status == status and "32768" in e.value.detail
    p.close()


def test_measure_prompt_timeout_is_oversize_but_other_errors_are_not():
    def timeout(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            return _auth_response()
        raise httpx.ReadTimeout("timeout", request=request)

    p = _provider(timeout)
    with pytest.raises(PromptTooLong) as e:
        p.measure_prompt("x")
    assert e.value.status is None
    p.close()

    def missing(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            return _auth_response()
        return httpx.Response(404, text="model not found")

    p = _provider(missing)
    with pytest.raises(ApiError) as e:
        p.measure_prompt("x")
    assert not isinstance(e.value, PromptTooLong)
    p.close()

    def no_usage(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            return _auth_response()
        return httpx.Response(200, json={"messages": []})

    p = _provider(no_usage)
    with pytest.raises(ApiError, match="usage"):
        p.measure_prompt("x")
    p.close()


def test_measure_prompt_refreshes_expired_token():
    auth_calls = []
    statuses = iter([401, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            auth_calls.append(1)
            return _auth_response()
        return _chat_reply(10) if next(statuses) == 200 else httpx.Response(401)

    p = _provider(handler)
    assert p.measure_prompt("x") == 10
    assert len(auth_calls) == 2
    p.close()


def test_list_models_keeps_only_chat_models():
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            return _auth_response()
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "Alpha-2-Max", "type": "chat"},
                    {"id": "Embeddings", "type": "embedder"},
                    {"id": "untyped"},
                    {"id": "Beta-72B", "type": "chat"},
                    "мусор",
                ]
            },
        )

    p = _provider(handler)
    assert p.list_models() == ["Alpha-2-Max", "Beta-72B"]
    p.close()


def test_measure_prompt_token_errors_are_not_oversize():
    """Таймаут при получении токена — не «запрос не помещается»: проба не отправлялась."""

    def auth_timeout(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            raise httpx.ReadTimeout("auth timeout", request=request)
        return _chat_reply(10)

    p = _provider(auth_timeout)
    with pytest.raises(ApiError) as e:
        p.measure_prompt("x")
    assert not isinstance(e.value, PromptTooLong)
    p.close()

    auth_calls = []

    def refresh_timeout(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            auth_calls.append(1)
            if len(auth_calls) > 1:
                raise httpx.ReadTimeout("auth timeout", request=request)
            return _auth_response()
        return httpx.Response(401)

    p = _provider(refresh_timeout)
    with pytest.raises(ApiError) as e:
        p.measure_prompt("x")
    assert not isinstance(e.value, PromptTooLong)
    p.close()


@pytest.mark.parametrize("status", [502, 503])
def test_measure_prompt_unavailable_gateway_is_not_oversize(status):
    """Временная недоступность — не «не помещается»: иначе окно занизилось бы навсегда."""

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            return _auth_response()
        return httpx.Response(status, text="<html>Bad Gateway</html>")

    p = _provider(handler)
    with pytest.raises(ApiError) as e:
        p.measure_prompt("x" * 10)
    assert not isinstance(e.value, PromptTooLong) and "временно недоступен" in str(e.value)
    p.close()


def test_error_body_is_clipped():
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            return _auth_response()
        return httpx.Response(400, text="<html>" + "x" * 50_000 + "</html>")

    p = _provider(handler)
    with pytest.raises(ApiError) as e:
        p.complete(_msgs())
    assert len(str(e.value)) < 700 and str(e.value).endswith("…")
    p.close()


def test_negative_retry_after_does_not_crash():
    resp = httpx.Response(429, headers={"Retry-After": "-5"})
    assert ApiProvider._retry_after(resp, 1.0) == 0.0
