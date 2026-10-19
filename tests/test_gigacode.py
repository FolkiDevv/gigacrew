"""Офлайн-тесты провайдера бэкенда GigaCode на подменном транспорте (httpx.MockTransport):
каталог моделей, вход через браузер, токены, запросы чата."""

from __future__ import annotations

import json
import threading
import time

import httpx
import pytest

from gigacrew.config import Config
from gigacrew.llm.base import ModelInfo
from gigacrew.llm.gigacode import (
    DEFAULT_CONTEXT_WINDOW,
    GigaCodeProvider,
    config_endpoint,
    parse_catalogue,
)
from gigacrew.llm.gigacode_auth import Token, TokenStore
from gigacrew.llm.transport import ApiError
from gigacrew.llm.types import FunctionCall, Message, ToolSpec

BASE = "https://code.test"
CONFIG = f"{BASE}/api/v2/config"
POLL = f"{BASE}/api/v2/auth/poll"
SIGNIN = "https://gitverse.test/signin"
CHAT = f"{BASE}/api/v2/chat/completions"
AGENT = "demo-agent/1.0 (test)"

CATALOGUE = {
    "default_model": "coder",
    "features": ["serve"],
    "agent": {
        "models": [
            {
                "id": "coder",
                "description": "Coder",
                "plugin_side": {"context_window_size": 131_072},
                "ai_api": {"max_tokens": 32000, "temperature": 0.3, "top_p": 0.9, "stream": True},
            },
            {"id": "fast", "ai_api": {"stream": False}},
        ]
    },
    "chat": {"url": CHAT, "models": []},
}

TOOL = ToolSpec(
    name="read_file",
    description="Читает файл",
    parameters={"type": "object", "properties": {"path": {"type": "string"}}},
)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    return sleeps


def _sse(*chunks) -> bytes:
    return "".join(
        f"data: {c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)}\n\n"
        for c in chunks
    ).encode()


def _text(text: str, *, prompt_tokens: int = 10) -> bytes:
    return _sse(
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": text}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 2}},
        "[DONE]",
    )


def _tool_call(name: str, args: dict, call_id: str = "c1") -> bytes:
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
    """Бэкенд GigaCode: конфигурация с 401-челленджем, опрос входа, чат."""

    def __init__(self, *, pending: int = 1, chat=(), catalogue=None):
        self.pending = pending  # ответов «ждём подтверждения» на каждый вход
        self.catalogue = catalogue or CATALOGUE
        self.valid: set[str] = set()
        self.chat_rejects: set[str] = set()  # токены, которые не принимает чат
        self.issued = 0
        self.polls = 0
        self.logins: list[dict] = []
        self.requests: list[httpx.Request] = []
        self.chat_bodies: list[dict] = []
        self.chat = list(chat)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)
        token = request.headers.get("Authorization", "").removeprefix("Bearer ")
        if url == CONFIG and request.method == "GET":
            if token in self.valid:
                return httpx.Response(200, json=self.catalogue)
            return httpx.Response(
                401,
                headers={
                    "X-Authorization-Url": POLL,
                    "X-Signin-Gitverse": SIGNIN,
                    "Retry-After": "2",
                },
            )
        if url == POLL:
            self.logins.append(json.loads(request.content))
            if self.polls < self.pending:
                self.polls += 1
                return httpx.Response(401, json={"error": "authorization_pending"})
            self.polls = 0
            self.issued += 1
            new = f"tok-{self.issued}"
            self.valid.add(new)
            return httpx.Response(200, json={"access_token": new, "expires_in": 3600})
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
        return sum(str(r.url) == url for r in self.requests)


def _config(tmp_path, **kw) -> Config:
    values = {
        "provider": "gigacode",
        "base_url": BASE,
        "plugin_name": "demo",
        "user_agent": AGENT,
        "gigacode_home": str(tmp_path / "cli-home"),
        "project_root": tmp_path,
    }
    values.update(kw)
    return Config(**values)


def _provider(backend: Backend, tmp_path, *, links=None, notices=None, store=None, **kw):
    return GigaCodeProvider(
        _config(tmp_path, **kw),
        on_login=(links.append if links is not None else lambda _url: None),
        on_notice=(notices.append if notices is not None else None),
        transport=httpx.MockTransport(backend.handler),
        token_store=store if store is not None else TokenStore(tmp_path / "tokens.json"),
    )


def _msgs():
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
def test_config_endpoint(base, url):
    assert config_endpoint(base) == url


# ------------------------------ каталог ------------------------------ #
def test_catalogue_from_the_contract_example():
    catalogue = parse_catalogue(CATALOGUE, CONFIG)
    assert [m.id for m in catalogue.models] == ["coder", "fast"]
    coder, fast = catalogue.models
    assert coder.context_window == 131_072 and fast.context_window == DEFAULT_CONTEXT_WINDOW
    assert dict(coder.sampling) == {"top_p": 0.9}  # temperature и max_tokens — свои
    assert coder.streams and not fast.streams
    assert catalogue.default_model == "coder" and catalogue.chat_url == CHAT


def test_catalogue_sources_in_order():
    def ids(data):
        return [m.id for m in parse_catalogue(data, CONFIG).models]

    assert ids({"chat": {"models": [{"id": "a"}]}, "agent": {"models": [{"id": "b"}]}}) == ["a"]
    assert ids({"chat": {"models": []}, "agent": {"models": [{"id": "b"}]}}) == ["b"]
    assert ids({"models": [{"id": "c", "context_window_size": 8192}]}) == ["c"]
    assert (
        parse_catalogue({"models": [{"id": "c", "context_window_size": 8192}]}, CONFIG)
        .models[0]
        .context_window
        == 8192
    )


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({}, "ни одной модели"),
        ({"models": []}, "ни одной модели"),
        ({"models": [{"id": "a"}, {"id": "a"}]}, "повторяется"),
        ({"models": [{"id": ""}]}, "id"),
        ({"models": [{"id": "a\nb"}]}, "id"),
        ({"models": ["a"]}, "id"),
        ({"models": [{"id": "a"}], "chat": {"url": "ftp://h/chat"}}, "адрес чата"),
    ],
)
def test_bad_catalogues(data, message):
    with pytest.raises(ApiError, match=message):
        parse_catalogue(data, CONFIG)


def test_catalogue_default_model_and_chat_url():
    models = [{"id": "a"}, {"id": "b"}]
    pick = parse_catalogue(
        {"models": models, "chat": {"default_model": "b"}, "default_model": "a"}, CONFIG
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


def test_catalogue_sampling_keeps_only_finite_numbers():
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
    (parsed,) = parse_catalogue({"models": [model]}, CONFIG).models
    assert dict(parsed.sampling) == {"top_p": 0.5, "top_k": 40, "seed": 7}


# ------------------------------ вход ------------------------------ #
def test_constructor_does_no_io(tmp_path):
    backend = Backend()
    provider = _provider(backend, tmp_path)
    assert backend.requests == [] and not (tmp_path / "tokens.json").exists()
    assert not (tmp_path / "cli-home").exists()
    assert provider.list_models() == [] and provider.known_models() is None
    assert provider.model == ""
    provider.close()


def test_constructor_requires_the_client_name(tmp_path):
    with pytest.raises(RuntimeError, match="GIGACREW_PLUGIN_NAME"):
        _provider(Backend(), tmp_path, plugin_name=None)
    with pytest.raises(RuntimeError, match="GIGACREW_USER_AGENT"):
        _provider(Backend(), tmp_path, user_agent=None)
    with pytest.raises(RuntimeError, match="GIGACREW_API_URL"):
        _provider(Backend(), tmp_path, base_url="")
    with pytest.raises(RuntimeError, match="GIGACREW_PROVIDER"):
        _provider(Backend(), tmp_path, provider="api")


def test_connect_logs_in_through_the_browser(tmp_path):
    backend = Backend(pending=2)
    links: list[str] = []
    provider = _provider(backend, tmp_path, links=links, plugin_name="demo-a")
    provider.connect()
    assert links == [SIGNIN]
    assert len(backend.logins) == 3 and {b["plugin_name"] for b in backend.logins} == {"demo-a"}
    device = (tmp_path / "cli-home" / "device_id").read_text(encoding="utf-8")
    assert backend.logins[0]["device_code"] == device
    assert provider.model == "coder"
    assert provider.list_models() == ["coder", "fast"]
    assert provider.known_models() == [
        ModelInfo("coder", 131_072),
        ModelInfo("fast", DEFAULT_CONTEXT_WINDOW),
    ]
    stored = json.loads((tmp_path / "tokens.json").read_text(encoding="utf-8"))
    assert stored["tokens"][CONFIG]["access_token"] == "tok-1"
    assert stored["tokens"][CONFIG]["plugin_name"] == "demo-a"
    provider.connect()  # токен действует, каталог есть — без сети
    assert backend.count(CONFIG) == 2
    provider.close()


def test_next_run_takes_the_token_from_the_file(tmp_path):
    backend = Backend()
    _provider(backend, tmp_path).connect()
    links: list[str] = []
    again = _provider(backend, tmp_path, links=links)
    again.connect()
    assert links == [] and len(backend.logins) == 2  # второго входа не было
    assert again.stream(_msgs()).message.content == "ok"
    assert backend.requests[-1].headers["Authorization"] == "Bearer tok-1"


def test_changed_client_name_logs_in_again(tmp_path):
    backend = Backend()
    _provider(backend, tmp_path, plugin_name="demo-a").connect()
    links: list[str] = []
    renamed = _provider(backend, tmp_path, links=links, plugin_name="demo-b")
    renamed.connect()
    assert links == [SIGNIN] and backend.logins[-1]["plugin_name"] == "demo-b"
    assert TokenStore(tmp_path / "tokens.json").load(CONFIG, "demo-b") == Token(
        "tok-2", pytest.approx(time.time() + 3600, abs=60)
    )


def test_expired_token_logs_in_again(tmp_path):
    backend = Backend()
    backend.valid.add("old")
    store = TokenStore(tmp_path / "tokens.json")
    store.save(CONFIG, "demo", Token("old", time.time() + 30))  # истекает через 30 с
    links: list[str] = []
    _provider(backend, tmp_path, links=links, store=store).connect()
    assert links == [SIGNIN] and store.load(CONFIG, "demo").access_token == "tok-1"


def test_token_rejected_by_config_is_forgotten(tmp_path):
    backend = Backend()
    store = TokenStore(tmp_path / "tokens.json")
    store.save(CONFIG, "demo", Token("revoked"))
    links: list[str] = []
    _provider(backend, tmp_path, links=links, store=store).connect()
    assert links == [SIGNIN] and store.load(CONFIG, "demo").access_token == "tok-1"


def test_token_rejected_right_after_login(tmp_path):
    backend = Backend()

    def handler(request):
        resp = backend.handler(request)
        if str(request.url) == CONFIG and resp.status_code == 200:
            return httpx.Response(401, text="no")  # и новый токен не принят
        return resp

    provider = GigaCodeProvider(
        _config(tmp_path),
        on_login=lambda _u: None,
        transport=httpx.MockTransport(handler),
        token_store=TokenStore(tmp_path / "tokens.json"),
    )
    with pytest.raises(ApiError, match="сразу после входа"):
        provider.connect()
    assert len(backend.logins) == 2  # один вход
    assert TokenStore(tmp_path / "tokens.json").load(CONFIG, "demo") is None


def test_rejected_login_names_the_client(tmp_path):
    backend = Backend()

    def handler(request):
        if str(request.url) == POLL:
            return httpx.Response(403, json={"error": "unknown plugin"})
        return backend.handler(request)

    provider = GigaCodeProvider(
        _config(tmp_path, plugin_name="demo-x"),
        on_login=lambda _u: None,
        transport=httpx.MockTransport(handler),
        token_store=TokenStore(tmp_path / "tokens.json"),
    )
    with pytest.raises(ApiError, match="«demo-x».*GIGACREW_PLUGIN_NAME"):
        provider.connect()


def test_config_errors(tmp_path):
    def handler(_request):
        return httpx.Response(403, text="forbidden")

    provider = GigaCodeProvider(_config(tmp_path), transport=httpx.MockTransport(handler))
    with pytest.raises(ApiError, match="конфигурацию GigaCode \\(403\\)"):
        provider.connect()


def test_chat_401_logs_in_once_more(tmp_path):
    backend = Backend()
    links: list[str] = []
    provider = _provider(backend, tmp_path, links=links)
    provider.connect()
    assert provider.stream(_msgs()).message.content == "ok"  # токен работает с чатом
    backend.valid.discard("tok-1")  # бэкенд его отозвал
    assert provider.stream(_msgs()).message.content == "ok"
    assert links == [SIGNIN, SIGNIN]
    assert backend.requests[-1].headers["Authorization"] == "Bearer tok-2"
    backend.valid.discard("tok-2")
    assert provider.complete(_msgs()).message.content == "ok"  # и без потока — так же
    assert len(links) == 3


def test_chat_401_on_a_fresh_token_does_not_log_in_again(tmp_path):
    backend = Backend()
    links: list[str] = []
    provider = _provider(backend, tmp_path, links=links)
    provider.connect()
    backend.chat_rejects.add("tok-1")  # конфигурацию даёт, а чат не пускает
    with pytest.raises(ApiError, match="повторный вход не поможет.*GIGACREW_PLUGIN_NAME"):
        provider.stream(_msgs())
    assert links == [SIGNIN]


def test_chat_errors_and_retries(tmp_path, _no_sleep):
    backend = Backend(
        chat=[
            httpx.Response(529, json={"error": "overloaded"}),
            _text("после перегрузки"),
            httpx.Response(403, text="нет доступа к модели"),
        ]
    )
    provider = _provider(backend, tmp_path)
    provider.connect()
    assert provider.stream(_msgs()).message.content == "после перегрузки"
    with pytest.raises(ApiError, match="403"):
        provider.stream(_msgs())


def test_chat_request_headers_and_body(tmp_path):
    backend = Backend()
    provider = _provider(backend, tmp_path)
    provider.connect()
    deltas: list[str] = []
    provider.stream(_msgs(), [TOOL], temperature=0.4, on_delta=deltas.append)
    provider.stream(_msgs())
    chat = [r for r in backend.requests if str(r.url) == CHAT]
    assert chat[0].headers["Accept"] == "application/json"
    assert chat[0].headers["Content-Type"] == "application/json"
    assert chat[0].headers["User-Agent"] == AGENT
    assert chat[0].headers["X-Session-Id"] == chat[1].headers["X-Session-Id"]
    body = backend.chat_bodies[0]
    assert body["model"] == "coder" and body["temperature"] == 0.4 and body["top_p"] == 0.9
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["tool_choice"] == "auto" and body["tools"][0]["function"]["name"] == "read_file"
    assert "max_tokens" not in body
    assert body["messages"] == [{"role": "user", "content": "привет"}]
    assert deltas == ["ok"]
    # User-Agent из GIGACREW_USER_AGENT — у всех запросов к бэкенду, и у входа тоже
    assert {r.headers["User-Agent"] for r in backend.requests} == {AGENT}
    assert {str(r.url) for r in backend.requests} == {CONFIG, POLL, CHAT}


def test_complete_goes_through_the_stream(tmp_path):
    backend = Backend()
    provider = _provider(backend, tmp_path, model="coder")
    provider.connect()
    turn = provider.complete(_msgs(), [TOOL])
    assert turn.message.content == "ok" and turn.usage.prompt_tokens == 10
    assert backend.chat_bodies[0]["stream"] is True


def test_model_without_stream_gets_a_whole_answer(tmp_path):
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
    assert turn.message.content == "целиком" and deltas == ["целиком"]
    assert backend.chat_bodies[0]["stream"] is False
    assert "stream_options" not in backend.chat_bodies[0]


def test_rejected_reasoning_is_dropped(tmp_path):
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


def test_measure_prompt_reads_usage_from_the_stream(tmp_path):
    backend = Backend(chat=[_text(".", prompt_tokens=1234)])
    provider = _provider(backend, tmp_path)
    provider.connect()
    assert provider.supports_measure and provider.measure_prompt("текст") == 1234
    body = backend.chat_bodies[0]
    assert body["max_tokens"] == 1 and body["model"] == "coder" and body["stream"] is True


def test_cancelled_login_is_not_restarted_until_connect(tmp_path, monkeypatch):
    backend = Backend(pending=5)
    provider = _provider(backend, tmp_path)

    def interrupted(_seconds):
        raise KeyboardInterrupt  # так приходят Ctrl+C и Esc

    monkeypatch.setattr(time, "sleep", interrupted)
    with pytest.raises(KeyboardInterrupt):
        provider.connect()
    polls = len(backend.logins)
    with pytest.raises(ApiError, match="отменён"):
        provider.stream(_msgs())  # например, итог суб-агента после Esc
    assert len(backend.logins) == polls
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    provider.connect()  # следующий запрос пользователя — вход снова
    assert provider.model == "coder"


def test_login_from_a_background_thread_is_refused(tmp_path):
    backend = Backend()
    provider = _provider(backend, tmp_path)
    errors: list[BaseException] = []

    def run():
        try:
            provider.connect()
        except BaseException as e:  # noqa: BLE001 — проверяем тип ниже
            errors.append(e)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(5)
    assert isinstance(errors[0], ApiError) and backend.logins == []


def test_token_save_failure_is_reported_and_work_goes_on(tmp_path):
    class ReadOnly(TokenStore):
        def save(self, url, plugin_name, token):
            raise OSError("только для чтения")

    backend = Backend()
    notices: list[str] = []
    provider = _provider(backend, tmp_path, notices=notices, store=ReadOnly(tmp_path / "t.json"))
    provider.connect()
    assert "не сохранён" in notices[0] and "только для чтения" in notices[0]
    assert provider.stream(_msgs()).message.content == "ok"


def test_request_without_connect_loads_the_catalogue(tmp_path):
    backend = Backend()
    provider = _provider(backend, tmp_path)
    assert provider.stream(_msgs()).message.content == "ok"
    assert backend.chat_bodies[0]["model"] == "coder"


def test_agent_turn_with_a_tool_call(tmp_path):
    """Ход агента через настоящий провайдер: вызов read_file, затем ответ."""
    from fakes import RecordingEvents

    from gigacrew.agent.loop import Agent
    from gigacrew.tools.base import build_default_registry

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
    assert second[-1]["role"] == "tool" and second[-1]["tool_call_id"] == "c1"
    assert "hello" in second[-1]["content"]
    provider.close()
