"""Офлайн-тесты сетевого слоя API на подменном транспорте (httpx.MockTransport).

Проверяют повторы, обработку обрывов и превращение любых ошибок транспорта в
ApiError (иначе они роняют REPL).
"""

from __future__ import annotations

import json
import math
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


class _CutAfterDone(httpx.SyncByteStream):
    """Событие окончания ответа пришло, потом соединение закрыто нечисто."""

    def __iter__(self):
        yield _event(_text("Запускаю тесты")).encode()
        done = {
            "messages": [
                {
                    "role": "assistant",
                    "tool_state_id": "S",
                    "content": [
                        {
                            "function_call": {
                                "id": "c1",
                                "name": "run_shell",
                                "arguments": {"command": "pytest"},
                            }
                        }
                    ],
                }
            ],
            "finish_reason": "function_call",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        yield _event(done, "response.message.done").encode()
        raise httpx.RemoteProtocolError("peer closed connection without sending complete body")


@pytest.mark.parametrize("shown", [False, True])
def test_unclean_close_after_done_keeps_the_answer(shown):
    """Обрыв после окончания ответа — ответ целиком, без ошибки и без повтора."""
    chat_calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            return _auth_response()
        chat_calls.append(request)
        return httpx.Response(200, stream=_CutAfterDone())

    deltas: list[str] = []
    p = _provider(handler)
    turn = p.stream(_msgs(), on_delta=deltas.append if shown else None)
    assert turn.finish_reason == "function_call" and turn.message.state_id == "S"
    assert turn.message.function_calls == [
        FunctionCall(name="run_shell", arguments={"command": "pytest"}, id="c1")
    ]
    assert turn.usage.prompt_tokens == 10
    assert deltas == (["Запускаю тесты"] if shown else [])
    assert len(chat_calls) == 1
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
    chat_calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            return _auth_response()
        chat_calls.append(request)
        raise exc("nope")

    p = _provider(handler)
    with pytest.raises(ApiError, match="Ошибка соединения"):
        p.complete(_msgs())
    assert len(chat_calls) == 1  # не повторяется
    with pytest.raises(ApiError, match="Ошибка соединения"):
        p.stream(_msgs())
    assert len(chat_calls) == 2
    p.close()


@pytest.mark.parametrize("streaming", [False, True], ids=["complete", "stream"])
def test_result_with_nan_does_not_break_the_chat(streaming):
    """Результат вызова — JSON-объект с NaN (файл, записанный json.dump): запрос уходит,
    результат — текстом, а не ValueError до отправки на каждом следующем ходе."""
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            return _auth_response()
        bodies.append(json.loads(request.content))
        if streaming:
            return httpx.Response(200, content=_done("ok"))
        return httpx.Response(200, json={**_text("ok"), "finish_reason": "stop"})

    history = [
        Message(role="user", content="покажи метрики"),
        Message(
            role="assistant",
            function_calls=[FunctionCall(name="skill", arguments={}, id="c1")],
            state_id="S",
        ),
        Message(role="function", name="skill", content='{"loss": NaN}', call_id="c1"),
    ]
    p = _provider(handler)
    assert (p.stream if streaming else p.complete)(history).message.content == "ok"
    (body,) = bodies
    result = body["messages"][-1]["content"][0]["function_result"]["result"]
    assert result == {"result": '{"loss": NaN}'}
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
    """Ответ OAuth без токена — ошибка до чата: иначе ушёл бы «Bearer None»."""
    chat_calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            return httpx.Response(200, json={"error": "no"})
        chat_calls.append(request)
        return httpx.Response(200, json={**_text("ok"), "finish_reason": "stop"})

    p = _provider(handler)
    with pytest.raises(ApiError, match="access_token"):
        p.complete(_msgs())
    assert chat_calls == []
    p.close()


# ------------------------------ секреты в ошибках ------------------------------ #
SECRET_KEY = "U0VDUkVULUtFWQ=="


def _secret_provider(handler, access_key: str = SECRET_KEY) -> ApiProvider:
    cfg = Config(access_key=access_key, auth_url=AUTH_URL, base_url=BASE_URL, model="M")
    return ApiProvider(cfg, transport=httpx.MockTransport(handler))


def _h11_like(request: httpx.Request):
    """Как h11 отвергает недопустимое значение заголовка: значение — в тексте ошибки."""
    raise httpx.LocalProtocolError(
        f"Illegal header value {request.headers['Authorization'].encode()!r}"
    )


@pytest.mark.parametrize("streaming", [False, True], ids=["complete", "stream"])
def test_rejected_header_does_not_leak_the_token(streaming):
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            return httpx.Response(200, json={"access_token": "SECRET-TOKEN"})
        _h11_like(request)

    p = _secret_provider(handler)
    with pytest.raises(ApiError, match="Ошибка соединения") as e:
        (p.stream if streaming else p.complete)(_msgs())
    assert "SECRET" not in str(e.value)
    p.close()


def test_rejected_auth_header_does_not_leak_the_key():
    p = _secret_provider(_h11_like)
    with pytest.raises(ApiError, match="Ошибка соединения") as e:
        p.complete(_msgs())
    assert SECRET_KEY not in str(e.value)
    p.close()


@pytest.mark.parametrize("tail", [" ", "\r", "\r\n", "\n"])
def test_access_key_whitespace_is_trimmed(tail):
    """Ключ из файла с CRLF (``$(cat key.txt)``) или с пробелом в .env — без хвоста."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            seen.append(request.headers["Authorization"])
            return _auth_response()
        return httpx.Response(200, json={**_text("ok"), "finish_reason": "stop"})

    p = _secret_provider(handler, access_key=SECRET_KEY + tail)
    assert p.complete(_msgs()).message.content == "ok"
    assert seen == [f"Basic {SECRET_KEY}"]
    p.close()


@pytest.mark.parametrize("key", ["U0VD\x00UkVU", "U0VD UkVU", "ключ-U0VD"])
def test_malformed_access_key_is_refused_without_echo(key):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _auth_response()

    p = _secret_provider(handler, access_key=key)
    with pytest.raises(ApiError, match="GIGACREW_ACCESS_KEY") as e:
        p.complete(_msgs())
    assert "U0VD" not in str(e.value) and requests == []
    p.close()


def test_oauth_reply_without_access_token_is_not_echoed():
    """Токен под другим именем поля не печатается: в ошибке — только имена полей."""

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            return httpx.Response(200, json={"accessToken": "SECRET-TOKEN", "expires": 1})
        return httpx.Response(200, json={**_text("ok"), "finish_reason": "stop"})

    p = _secret_provider(handler)
    with pytest.raises(ApiError, match="access_token") as e:
        p.complete(_msgs())
    assert "SECRET" not in str(e.value)
    assert "accessToken" in str(e.value)  # имя поля подскажет, в чём дело
    p.close()


@pytest.mark.parametrize("token", [42, ["t"], {"value": "SECRET-TOKEN"}, "SECRET TOKEN\r"])
def test_malformed_access_token_is_refused_without_echo(token):
    chat: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            return httpx.Response(200, json={"access_token": token})
        chat.append(request)
        return httpx.Response(200, json={**_text("ok"), "finish_reason": "stop"})

    p = _secret_provider(handler)
    with pytest.raises(ApiError, match="access_token") as e:
        p.complete(_msgs())
    assert "SECRET" not in str(e.value) and chat == []
    p.close()


@pytest.mark.parametrize(
    "reply",
    [
        httpx.Response(403, text="forbidden"),
        # JSON-тело с данными: статус всё равно ошибка, а не пустой или чужой список
        httpx.Response(403, json={"data": [{"id": "x", "type": "chat"}]}),
    ],
    ids=["text", "json"],
)
def test_list_models_http_error_wrapped(reply):
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            return _auth_response()
        return reply

    p = _provider(handler)
    with pytest.raises(ApiError, match=r"список моделей \(403\)"):
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


# ------------------------------ авторизация и повторы ------------------------------ #
CHAT_URL = "https://api.test/v2/chat/completions"
REQUEST_CAP = 8  # больше запросов чата за вызов — бесконечный цикл: тест падает, а не висит


class _Server:
    """Подменный шлюз: OAuth и чат по отдельным обработчикам, запросы — в ``auth``/``chat``."""

    def __init__(self, chat=None, auth=None):
        self.chat_reply = chat or _ok_for
        self.auth_reply = auth or (lambda _r: _auth_response())
        self.auth: list[httpx.Request] = []
        self.chat: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            self.auth.append(request)
            return self.auth_reply(request)
        self.chat.append(request)
        if len(self.chat) > REQUEST_CAP:
            raise AssertionError(f"больше {REQUEST_CAP} запросов чата — цикл повторов")
        return self.chat_reply(request)


def _ok_for(request: httpx.Request) -> httpx.Response:
    """Успешный ответ на любой запрос чата: целый, поток или замер окна."""
    body = json.loads(request.content)
    if body.get("stream"):
        return httpx.Response(200, content=_done("ok"))
    if body["model_options"].get("max_tokens") == 1:
        return _chat_reply(10)
    return httpx.Response(200, json={**_text("ok"), "finish_reason": "stop"})


def _then_ok(times: int, reply):
    """Обработчик чата: ``times`` раз ответ ``reply()``, потом — успешный."""
    left = {"n": times}

    def handler(request: httpx.Request) -> httpx.Response:
        if left["n"] > 0:
            left["n"] -= 1
            return reply()
        return _ok_for(request)

    return handler


def _call_of(p: ApiProvider, kind: str):
    return {
        "complete": lambda: p.complete(_msgs()),
        "stream": lambda: p.stream(_msgs()),
        "measure": lambda: p.measure_prompt("x"),
    }[kind]


@pytest.mark.parametrize("kind", ["complete", "stream", "measure"])
def test_persistent_401_refreshes_the_token_exactly_once(kind):
    """Отозванный ключ: токен обновляется один раз, потом — «API вернул 401» с телом,
    а не бесконечный цикл обновлений или «Сетевая ошибка … : None»."""
    server = _Server(chat=lambda _r: httpx.Response(401, text="token revoked"))
    p = _provider(server)
    with pytest.raises(ApiError, match="401: token revoked"):
        _call_of(p, kind)()
    assert (len(server.auth), len(server.chat)) == (2, 2)
    p.close()


@pytest.mark.parametrize("streaming", [False, True], ids=["complete", "stream"])
def test_oauth_and_chat_wire_format(streaming):
    """Запрос токена — POST с Basic-ключом, scope в теле формы и RqUID; чат — Bearer."""
    server = _Server()
    cfg = Config(
        access_key="dummy", auth_url=AUTH_URL, base_url=BASE_URL, model="M", scope="MY_SCOPE"
    )
    p = ApiProvider(cfg, transport=httpx.MockTransport(server))
    (p.stream if streaming else p.complete)(_msgs())
    (auth,) = server.auth
    assert auth.method == "POST"
    assert auth.headers["Authorization"] == "Basic dummy"
    assert auth.headers["Content-Type"] == "application/x-www-form-urlencoded"
    assert auth.headers["Accept"] == "application/json"
    assert auth.headers.get("RqUID")
    assert auth.content == b"scope=MY_SCOPE"
    (chat,) = server.chat
    assert str(chat.url) == CHAT_URL
    assert chat.headers["Authorization"] == "Bearer t"
    p.close()


def test_oauth_without_scope_sends_an_empty_form():
    server = _Server()
    p = _provider(server)
    p.complete(_msgs())
    assert server.auth[0].content == b""
    p.close()


def _mtls_provider(monkeypatch, handler) -> ApiProvider:
    monkeypatch.setattr(Config, "build_ssl_verify", lambda self: True)  # без файлов PEM
    cfg = Config(cert="c.pem", key="k.pem", base_url=BASE_URL, model="M")
    assert cfg.auth_mode == "mtls"
    return ApiProvider(cfg, transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("kind", ["complete", "stream", "measure"])
def test_mtls_sends_no_oauth_request_and_no_authorization(monkeypatch, kind):
    """mTLS: клиента авторизует сертификат — ни запроса токена, ни заголовка Authorization."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok_for(request)

    p = _mtls_provider(monkeypatch, handler)
    _call_of(p, kind)()
    assert [str(r.url) for r in seen] == [CHAT_URL]
    assert "authorization" not in seen[0].headers
    p.close()


@pytest.mark.parametrize("kind", ["complete", "stream"])
def test_mtls_401_is_not_refreshed(monkeypatch, kind):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(401, text="certificate not accepted")

    p = _mtls_provider(monkeypatch, handler)
    with pytest.raises(ApiError, match="401: certificate not accepted"):
        _call_of(p, kind)()
    assert len(seen) == 1
    p.close()


def test_token_is_refreshed_before_it_expires():
    """Токен, истекающий в ближайшие 30 с, обновляется заранее, а не ценой 401."""
    soon = _Server(auth=lambda _r: _auth_response_expiring(10))
    p = _provider(soon)
    p.complete(_msgs())
    p.complete(_msgs())
    assert len(soon.auth) == 2 and len(soon.chat) == 2
    p.close()

    later = _Server()
    p = _provider(later)
    p.complete(_msgs())
    p.complete(_msgs())
    assert len(later.auth) == 1  # кэшируется до истечения
    p.close()


def _auth_response_expiring(seconds: float) -> httpx.Response:
    return httpx.Response(
        200, json={"access_token": "t", "expires_at": int((time.time() + seconds) * 1000)}
    )


def test_token_endpoint_errors():
    """Ошибка выдачи токена — с её статусом и телом, чат не вызывается."""
    server = _Server(auth=lambda _r: httpx.Response(401, text="bad key"))
    p = _provider(server)
    with pytest.raises(ApiError, match=r"токен API \(401\): bad key"):
        p.complete(_msgs())
    assert server.chat == []
    p.close()

    # срок токена не разобрался — живёт 25 минут, а не истекает сразу
    odd = _Server(
        auth=lambda _r: httpx.Response(200, json={"access_token": "t", "expires_at": "soon"})
    )
    p = _provider(odd)
    p.complete(_msgs())
    p.complete(_msgs())
    assert len(odd.auth) == 1
    p.close()


@pytest.mark.parametrize("status", [500, 502, 503, 504, 529])
@pytest.mark.parametrize("streaming", [False, True], ids=["complete", "stream"])
def test_transient_server_errors_are_retried(monkeypatch, status, streaming):
    pauses = _sleeps(monkeypatch)
    server = _Server(chat=_then_ok(1, lambda: httpx.Response(status, text="busy")))
    p = _provider(server)
    assert (p.stream if streaming else p.complete)(_msgs()).message.content == "ok"
    assert len(server.chat) == 2 and pauses == [1.0]
    p.close()


@pytest.mark.parametrize("streaming", [False, True], ids=["complete", "stream"])
def test_server_error_backoff_is_exponential(monkeypatch, streaming):
    pauses = _sleeps(monkeypatch)
    server = _Server(chat=_then_ok(4, lambda: httpx.Response(503, text="busy")))
    p = _provider(server)
    assert (p.stream if streaming else p.complete)(_msgs()).message.content == "ok"
    assert pauses == [1.0, 2.0, 4.0, 8.0] and len(server.chat) == 5
    p.close()


@pytest.mark.parametrize("streaming", [False, True], ids=["complete", "stream"])
def test_persistent_server_error_reports_status_and_body(monkeypatch, streaming):
    """Пять попыток, затем ошибка со статусом и телом шлюза, а не «Сетевая ошибка: None»."""
    pauses = _sleeps(monkeypatch)
    server = _Server(chat=lambda _r: httpx.Response(503, text="busy-body"))
    p = _provider(server)
    with pytest.raises(ApiError, match="503: busy-body"):
        (p.stream if streaming else p.complete)(_msgs())
    assert len(server.chat) == 5 and pauses == [1.0, 2.0, 4.0, 8.0]
    p.close()


@pytest.mark.parametrize("streaming", [False, True], ids=["complete", "stream"])
def test_server_error_retry_after_is_capped(monkeypatch, streaming):
    """Retry-After: 600 на 5xx — не десять минут на повтор (у 429 свой потолок)."""
    pauses = _sleeps(monkeypatch)
    busy = {"Retry-After": "600"}
    server = _Server(chat=_then_ok(1, lambda: httpx.Response(503, headers=busy, text="x")))
    p = _provider(server)
    (p.stream if streaming else p.complete)(_msgs())
    assert pauses == [30.0]
    p.close()


@pytest.mark.parametrize("streaming", [False, True], ids=["complete", "stream"])
def test_persistent_connection_failure_gives_up_after_five_attempts(monkeypatch, streaming):
    pauses = _sleeps(monkeypatch)

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    server = _Server(chat=refuse)
    p = _provider(server)
    with pytest.raises(ApiError, match="после 5 попыток.*connection refused"):
        (p.stream if streaming else p.complete)(_msgs())
    assert len(server.chat) == 5 and pauses == [1.0, 2.0, 4.0, 8.0]
    p.close()


@pytest.mark.parametrize("target", ["chat", "auth", "models"])
def test_json_that_is_not_an_object_is_an_api_error(target):
    """Шлюз ответил JSON-массивом — ApiError, а не AttributeError мимо LLMError."""
    array = httpx.Response(200, json=[1, 2])
    server = _Server(
        chat=lambda _r: array if target != "auth" else _ok_for(_r),
        auth=(lambda _r: array) if target == "auth" else None,
    )
    p = _provider(server)
    with pytest.raises(ApiError, match="Неожиданный формат"):
        p.list_models() if target == "models" else p.complete(_msgs())
    p.close()


def test_usage_from_raw_is_tolerant():
    assert Usage.from_raw(None) == Usage()
    assert Usage.from_raw("garbage") == Usage()  # type: ignore[arg-type]
    u = Usage.from_raw({"prompt_tokens": "5", "completion_tokens": None, "total_tokens": "x"})
    assert (u.prompt_tokens, u.completion_tokens, u.total_tokens) == (5, 0, 0)
    # отрицательные числа от сломанного шлюза не попадают в счёт контекста и бюджета
    assert Usage.from_raw({"prompt_tokens": -5, "output_tokens": -1}) == Usage()


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


def _refuse(exc):
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc("no route", request=request)

    return handler


@pytest.mark.parametrize(
    "chat",
    [
        _refuse(httpx.ConnectError),
        _refuse(httpx.ConnectTimeout),
        lambda _r: httpx.Response(401, text="unauthorized"),
        lambda _r: httpx.Response(403, text="нет доступа к модели"),
        lambda _r: httpx.Response(429, text="too many requests"),
    ],
    ids=["connect-error", "connect-timeout", "401", "403", "429"],
)
def test_measure_prompt_errors_that_are_not_about_size(chat):
    """Ошибки установки соединения, авторизация и лимит частоты — не «не помещается»:
    иначе окно занизилось бы и сохранилось в ~/.gigacode/models.json."""
    cfg = Config(
        access_key="dummy", auth_url=AUTH_URL, base_url=BASE_URL, model="M", rate_limit_wait=0
    )
    p = ApiProvider(cfg, transport=httpx.MockTransport(_Server(chat=chat)))
    with pytest.raises(ApiError) as e:
        p.measure_prompt("x")
    assert not isinstance(e.value, PromptTooLong)
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


@pytest.mark.parametrize("header", ["nan", "NaN", "inf", "-inf", "1e400"])
@pytest.mark.parametrize("status", [429, 503])
@pytest.mark.parametrize("streaming", [False, True], ids=["complete", "stream"])
def test_non_finite_retry_after_is_ignored(monkeypatch, header, status, streaming):
    """Retry-After: nan — не ValueError из time.sleep, а обычная пауза по умолчанию."""
    pauses: list[float] = []

    def strict_sleep(seconds: float) -> None:
        if not math.isfinite(seconds):  # как настоящий time.sleep
            raise ValueError("Invalid value NaN (not a number)")
        pauses.append(seconds)

    calls = {"chat": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            return _auth_response()
        calls["chat"] += 1
        if calls["chat"] == 1:
            return httpx.Response(status, headers={"Retry-After": header}, text="busy")
        if streaming:
            return httpx.Response(200, content=_done("ok"))
        return httpx.Response(200, json={**_text("ok"), "finish_reason": "stop"})

    p = _provider(handler)
    monkeypatch.setattr(api.time, "sleep", strict_sleep)
    assert (p.stream if streaming else p.complete)(_msgs()).message.content == "ok"
    assert pauses == [5.0 if status == 429 else 1.0]  # пауза по умолчанию
    assert calls["chat"] == 2
    p.close()


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
