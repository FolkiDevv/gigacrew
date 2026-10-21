"""Офлайн-тесты сетевого слоя API на подменном транспорте (httpx.MockTransport).

Проверяют повторы, обработку обрывов и превращение любых ошибок транспорта в
ApiError (иначе они роняют REPL).
"""

from __future__ import annotations

import json
import ssl
import time

import httpx
import pytest

from gigacrew.config import Config
from gigacrew.llm import api
from gigacrew.llm.api import ApiError, ApiProvider
from gigacrew.llm.base import LLMError, PromptTooLong
from gigacrew.llm.types import FunctionCall, Message, Usage

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


class _SilentStream(httpx.SyncByteStream):
    """Тело ответа: отдаёт кусок, потом модель молчит дольше ожидания чтения."""

    def __init__(self, first: bytes):
        self._first = first

    def __iter__(self):
        yield self._first
        raise httpx.ReadTimeout("The read operation timed out")


def _done(text: str) -> bytes:
    return _event({**_text(text), "finish_reason": "stop"}, "response.message.done").encode()


def _timeouts(handler, config: Config) -> list[tuple[str, float]]:
    """Таймаут чтения каждого запроса: токен, цельный ответ, поток и замер окна."""
    seen: list[tuple[str, float]] = []

    def recording(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            kind = "auth"
        else:
            kind = "stream" if json.loads(request.content).get("stream") else "chat"
        seen.append((kind, request.extensions["timeout"]["read"]))
        return handler(request)

    p = ApiProvider(config, transport=httpx.MockTransport(recording))
    p.complete(_msgs())
    p.stream(_msgs())
    p.measure_prompt("x")
    p.close()
    return seen


def _answer(request: httpx.Request) -> httpx.Response:
    if str(request.url) == AUTH_URL:
        return _auth_response()
    body = json.loads(request.content)
    if body.get("stream"):
        return httpx.Response(200, content=_done("ok"))
    if body["model_options"].get("max_tokens") == 1:  # замер окна
        return _chat_reply(10)
    return httpx.Response(200, json={**_text("ok"), "finish_reason": "stop"})


def test_chat_waits_for_the_model_longer_than_other_requests():
    """Вызовы инструментов приходят в конце ответа: чат ждёт read_timeout, а токен и
    замер окна — обычный timeout (долгая проба и есть «не помещается»)."""
    config = Config(access_key="dummy", auth_url=AUTH_URL, base_url=BASE_URL, model="M")
    assert _timeouts(_answer, config) == [
        ("auth", 120),
        ("chat", 600),
        ("stream", 600),
        ("chat", 120),  # замер окна
    ]


def test_longer_general_timeout_is_not_shortened_for_chat():
    config = Config(
        access_key="dummy", auth_url=AUTH_URL, base_url=BASE_URL, model="M", timeout=900
    )
    reads = [read for kind, read in _timeouts(_answer, config) if kind != "auth"]
    assert reads == [900, 900, 900]


def test_model_silent_after_shown_text_names_the_setting():
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            return _auth_response()
        return httpx.Response(
            200, stream=_SilentStream(_event(_text("Сейчас запишу план")).encode())
        )

    deltas: list[str] = []
    p = _provider(handler)
    with pytest.raises(ApiError, match="GIGACREW_READ_TIMEOUT") as e:
        p.stream(_msgs(), on_delta=deltas.append)
    assert "дольше 600 с" in str(e.value)
    assert deltas == ["Сейчас запишу план"]  # текст не продублирован повтором
    p.close()


@pytest.mark.parametrize("streaming", [False, True])
def test_silent_model_is_retried_only_once(streaming):
    """Молчание дольше ожидания чтения — не обрыв сети: пять попыток по 10 минут не нужны."""
    chat_calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            return _auth_response()
        chat_calls.append(request)
        raise httpx.ReadTimeout("The read operation timed out", request=request)

    p = _provider(handler)
    with pytest.raises(ApiError, match="GIGACREW_READ_TIMEOUT"):
        (p.stream if streaming else p.complete)(_msgs())
    assert len(chat_calls) == 2
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


@pytest.mark.parametrize("streaming", [False, True])
def test_rejected_server_certificate_is_not_retried(streaming):
    """Сертификат сервера не прошёл проверку — повтор не поможет: сразу ошибка с подсказкой."""
    chat_calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            return _auth_response()
        chat_calls.append(request)
        cause = ssl.SSLCertVerificationError(1, "certificate verify failed")
        cause.verify_message = "unable to get local issuer certificate"
        raise httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED]", request=request) from cause

    p = _provider(handler)
    with pytest.raises(ApiError, match="GIGACREW_CA_BUNDLE") as e:
        (p.stream if streaming else p.complete)(_msgs())
    assert "unable to get local issuer certificate" in str(e.value)
    assert len(chat_calls) == 1
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


@pytest.mark.parametrize(
    "raw",
    [
        {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 4}},
        {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 4}},
        {"prompt_tokens": 10, "cached_tokens": 4},
        {"prompt_tokens": 10, "precached_prompt_tokens": 4, "cached_tokens": 9},
    ],
)
def test_usage_cached_tokens_in_every_layout(raw):
    usage = Usage.from_raw(raw)
    assert (usage.prompt_tokens, usage.precached_prompt_tokens) == (10, 4)


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


@pytest.mark.parametrize("status", [502, 503, 529])
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


@pytest.mark.parametrize("streaming", [False, True], ids=["complete", "stream"])
def test_overloaded_model_server_is_retried(streaming):
    """529 — перегрузка сервера модели: повторяется, как 503."""
    calls = {"chat": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            return _auth_response()
        calls["chat"] += 1
        if calls["chat"] == 1:
            return httpx.Response(529, json={"message": "overloaded"})
        if streaming:
            return httpx.Response(200, content=_done("ok"))
        return httpx.Response(200, json={**_text("ok"), "finish_reason": "stop"})

    p = _provider(handler)
    assert (p.stream if streaming else p.complete)(_msgs()).message.content == "ok"
    assert calls["chat"] == 2
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


# ------------------------------ лимит запросов (429) ------------------------------ #
def _limited(times: int, *, stream: bool = False, retry_after: str | None = None):
    """Шлюз отвечает 429 ``times`` раз подряд, потом — нормально."""
    calls = {"chat": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            return _auth_response()
        calls["chat"] += 1
        if calls["chat"] <= times:
            headers = {"Retry-After": retry_after} if retry_after is not None else {}
            return httpx.Response(429, headers=headers, json={"message": "Too many requests"})
        if stream:
            body = _event({**_text("ok"), "finish_reason": "stop"}, "response.message.done")
            return httpx.Response(200, content=body.encode())
        return httpx.Response(200, json={**_text("ok"), "finish_reason": "stop"})

    return handler, calls


def _sleeps(monkeypatch) -> list[float]:
    pauses: list[float] = []
    monkeypatch.setattr(api.time, "sleep", pauses.append)
    return pauses


@pytest.mark.parametrize("streaming", [False, True], ids=["complete", "stream"])
def test_rate_limit_waits_longer_than_other_retries(monkeypatch, streaming):
    pauses = _sleeps(monkeypatch)
    handler, calls = _limited(6, stream=streaming)  # больше обычных 5 попыток
    p = _provider(handler)
    waits: list[str] = []
    call = p.stream if streaming else p.complete
    assert call(_msgs(), on_notice=waits.append).message.content == "ok"
    assert calls["chat"] == 7
    assert pauses == [5.0, 10.0, 20.0, 30.0, 30.0, 25.0]  # до бюджета 120 с
    assert len(waits) == 6 and "лимит запросов шлюза (429) — жду 5 с" in waits[0]
    p.close()


@pytest.mark.parametrize("streaming", [False, True], ids=["complete", "stream"])
def test_rate_limit_gives_up_when_budget_is_spent(monkeypatch, streaming):
    pauses = _sleeps(monkeypatch)
    handler, _ = _limited(100, stream=streaming)
    p = _provider(handler)
    call = p.stream if streaming else p.complete
    with pytest.raises(ApiError, match="429"):
        call(_msgs())
    assert sum(pauses) == 120.0
    p.close()


@pytest.mark.parametrize(("header", "pause"), [("12", 12.0), ("0", 1.0), ("600", 30.0)])
def test_rate_limit_respects_retry_after(monkeypatch, header, pause):
    pauses = _sleeps(monkeypatch)
    handler, _ = _limited(1, retry_after=header)
    p = _provider(handler)
    p.complete(_msgs())
    assert pauses == [pause]  # не меньше секунды и не дольше 30 с
    p.close()


def test_rate_limit_wait_can_be_turned_off(monkeypatch):
    pauses = _sleeps(monkeypatch)
    handler, calls = _limited(1)
    cfg = Config(
        access_key="dummy",
        auth_url=AUTH_URL,
        base_url=BASE_URL,
        model="M-default",
        rate_limit_wait=0,
    )
    p = ApiProvider(cfg, transport=httpx.MockTransport(handler))
    with pytest.raises(ApiError, match="429"):
        p.complete(_msgs())
    assert pauses == [] and calls["chat"] == 1
    p.close()


# ------------------------------ рассуждения модели ------------------------------ #
def _bodies_of(handler):
    """Обработчик, запоминающий тела запросов чата."""
    bodies: list[dict] = []

    def recording(request: httpx.Request) -> httpx.Response:
        if str(request.url) != AUTH_URL:
            bodies.append(json.loads(request.content))
        return handler(request)

    return recording, bodies


def _roles(body: dict) -> list[str]:
    return [m["role"] for m in body["messages"]]


def _with_reasoning_history():
    call = FunctionCall(name="read_file", arguments={"path": "a"}, id="c1")
    return [
        Message(role="user", content="что в a?"),
        Message(role="assistant", function_calls=[call], state_id="S", reasoning="прочитаю a"),
        Message(role="function", name="read_file", content="x", call_id="c1"),
    ]


def _rejecting(*, always: bool = False):
    """Шлюз не принимает рассуждения в запросе (422); ``always`` — и без них."""

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            return _auth_response()
        body = json.loads(request.content)
        if always or "reasoning" in _roles(body):
            return httpx.Response(422, json={"message": "unknown role: reasoning"})
        if body.get("stream"):
            done = _event({**_text("ok"), "finish_reason": "stop"}, "response.message.done")
            return httpx.Response(200, content=done.encode())
        return httpx.Response(200, json={**_text("ok"), "finish_reason": "stop"})

    return handler


@pytest.mark.parametrize("streaming", [False, True], ids=["complete", "stream"])
def test_rejected_reasoning_is_dropped_for_the_model(streaming):
    handler, bodies = _bodies_of(_rejecting())
    p = _provider(handler)
    notices: list[str] = []
    call = p.stream if streaming else p.complete
    assert call(_with_reasoning_history(), on_notice=notices.append).message.content == "ok"
    assert [_roles(b).count("reasoning") for b in bodies] == [1, 0]  # повтор — без них
    assert len(notices) == 1 and "не принял рассуждения модели M-default" in notices[0]
    # дальше эта модель — сразу без рассуждений, без лишнего запроса и уведомления
    call(_with_reasoning_history(), on_notice=notices.append)
    assert len(bodies) == 3 and "reasoning" not in _roles(bodies[2]) and len(notices) == 1
    p.close()


@pytest.mark.parametrize("streaming", [False, True], ids=["complete", "stream"])
def test_other_rejection_is_not_blamed_on_reasoning(streaming):
    handler, bodies = _bodies_of(_rejecting(always=True))
    p = _provider(handler)
    call = p.stream if streaming else p.complete
    with pytest.raises(ApiError, match="422"):
        call(_with_reasoning_history())
    with pytest.raises(ApiError, match="422"):
        call(_msgs())  # без рассуждений — и повторять нечего
    # отказ не из-за рассуждений: модель их не теряет
    assert [_roles(b).count("reasoning") for b in bodies] == [1, 0, 0]
    with pytest.raises(ApiError):
        call(_with_reasoning_history())
    assert _roles(bodies[3]).count("reasoning") == 1
    p.close()


def _agent_requests(tmp_path, reasoning: str, *, thoughts: bool) -> list[dict]:
    """Ход агента через настоящий клиент API: чтение файла, затем ответ; тела запросов."""
    from fakes import RecordingEvents

    from gigacrew.agent.loop import Agent
    from gigacrew.tools.base import build_default_registry

    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    replies = iter(
        [
            {
                "messages": [
                    {
                        "role": "assistant",
                        "tools_state_id": "S",
                        "content": [
                            {
                                "function_call": {
                                    "id": "c1",
                                    "name": "read_file",
                                    "arguments": {"path": "a.txt"},
                                }
                            }
                        ],
                    }
                ],
                "finish_reason": "function_call",
            },
            {**_text("там hello"), "finish_reason": "stop"},
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            return _auth_response()
        data = next(replies)
        if thoughts:
            data = {**data, "messages": [_reasoning_part(), *data["messages"]]}
        return httpx.Response(200, json=data)

    recording, bodies = _bodies_of(handler)
    cfg = Config(
        access_key="dummy",
        auth_url=AUTH_URL,
        base_url=BASE_URL,
        model="M-default",
        project_root=tmp_path,
        stream=False,
        reasoning=reasoning,
    )
    provider = ApiProvider(cfg, transport=httpx.MockTransport(recording))
    Agent(provider, build_default_registry(), cfg, RecordingEvents()).run_turn("что в a.txt?")
    provider.close()
    return bodies


def _reasoning_part() -> dict:
    return {"role": "reasoning", "content": [{"text": "надо прочитать a.txt"}]}


def test_model_without_reasoning_gets_the_same_requests_in_turn_mode(tmp_path):
    """Модели без рассуждений режим turn не касается: запросы те же, что с off."""
    turn = _agent_requests(tmp_path, "turn", thoughts=False)
    off = _agent_requests(tmp_path, "off", thoughts=False)
    assert turn == off and len(turn) == 2
    assert all("reasoning" not in _roles(body) for body in turn)


def test_model_with_reasoning_gets_it_back_within_the_turn(tmp_path):
    turn = _agent_requests(tmp_path, "turn", thoughts=True)
    off = _agent_requests(tmp_path, "off", thoughts=True)
    assert _roles(turn[1]) == ["system", "user", "reasoning", "assistant", "tool"]
    assert turn[1]["messages"][2] == _reasoning_part()
    assert "reasoning" not in _roles(off[1])
