"""Офлайн-тесты входа в бэкенд GigaCode: токен, хранилище, id устройства, опрос входа."""

from __future__ import annotations

import json
import os
import platform
import stat

import httpx
import pytest

from gigacrew import __version__
from gigacrew.llm.gigacode_auth import (
    EXPIRY_MARGIN,
    Challenge,
    DeviceLogin,
    Token,
    TokenStore,
    default_token_path,
    device_id,
    login_body,
    safe_url,
)
from gigacrew.llm.transport import ApiError

CONFIG = "https://code.test/api/v2/config"
POLL = "https://code.test/api/v2/auth/poll"
SIGNIN = "https://gitverse.test/signin?d=1"


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


# ------------------------------- токен ------------------------------- #
def test_token_expiry_has_a_margin():
    token = Token("t", expires_at=1_000.0)
    assert not token.expired(now=1_000.0 - EXPIRY_MARGIN - 1)
    assert token.expired(now=1_000.0 - EXPIRY_MARGIN)
    assert not Token("t").expired(now=10**12)  # без срока — до первого 401


@pytest.mark.parametrize(
    ("expires_in", "expected"),
    [(3600, 4600.0), ("120", 1120.0), (None, None), ("скоро", None), (0, None), (True, None)],
)
def test_token_from_login(expires_in, expected):
    data = {"access_token": "abc.DEF-123", "expires_in": expires_in}
    assert Token.from_login(data, now=1000.0) == Token("abc.DEF-123", expected)


@pytest.mark.parametrize("value", ["", "с пробелом", "кириллица", None, 42])
def test_token_must_be_printable_ascii(value):
    with pytest.raises(ApiError, match="access_token"):
        Token.from_login({"access_token": value})


# ----------------------------- хранилище ----------------------------- #
def test_store_round_trip_is_private_and_atomic(tmp_path):
    path = tmp_path / ".gigacrew" / "gigacode_token.json"
    store = TokenStore(path)
    assert store.load(CONFIG, "demo") is None and not path.exists()
    store.save(CONFIG, "demo", Token("t1", 2e9))
    store.save("https://other.test/api/v2/config", "demo", Token("t2"))
    assert _mode(path) == 0o600
    assert [p.name for p in path.parent.iterdir()] == ["gigacode_token.json"]  # без .tmp
    assert store.load(CONFIG, "demo") == Token("t1", 2e9)
    assert TokenStore(path).load("https://other.test/api/v2/config", "demo") == Token("t2")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 1 and data["tokens"][CONFIG]["plugin_name"] == "demo"
    store.forget(CONFIG)
    assert store.load(CONFIG, "demo") is None
    assert store.load("https://other.test/api/v2/config", "demo") == Token("t2")


def test_store_token_of_another_client_name_is_not_used(tmp_path):
    store = TokenStore(tmp_path / "t.json")
    store.save(CONFIG, "demo-a", Token("t1"))
    assert store.load(CONFIG, "demo-b") is None
    assert store.load(CONFIG, "demo-a") == Token("t1")


def _file(tokens) -> str:
    return json.dumps({"version": 1, "tokens": tokens})


@pytest.mark.parametrize(
    "text",
    [
        "{битый",
        '{"version": 2, "tokens": {}}',
        '{"version": 1, "tokens": []}',
        _file({CONFIG: {"access_token": "a b", "plugin_name": "demo"}}),
        _file({CONFIG: {"access_token": "a", "expires_at": "x", "plugin_name": "demo"}}),
    ],
)
def test_store_broken_file_is_empty_and_overwritten(tmp_path, text):
    path = tmp_path / "t.json"
    path.write_text(text, encoding="utf-8")
    store = TokenStore(path)
    assert store.load(CONFIG, "demo") is None
    store.save(CONFIG, "demo", Token("new"))
    assert store.load(CONFIG, "demo") == Token("new")


def test_store_default_path_is_resolved_on_use(_isolated_home, tmp_path, monkeypatch):
    store = TokenStore()
    assert (
        store.path == default_token_path() == _isolated_home / ".gigacrew" / "gigacode_token.json"
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    assert store.path == tmp_path / ".gigacrew" / "gigacode_token.json"


# --------------------------- id устройства --------------------------- #
def test_device_id_is_created_once_like_the_official_cli(tmp_path):
    home = tmp_path / ".gigacode"
    first = device_id(home)
    assert len(first) == 64 and int(first, 16) >= 0
    assert _mode(home / "device_id") == 0o600 and _mode(home) == 0o700
    assert device_id(home) == first  # существующий — читается, не создаётся заново


def test_existing_device_id_is_never_overwritten(tmp_path):
    home = tmp_path / "cli"
    home.mkdir()
    value = "AB" * 32
    (home / "device_id").write_text(value + "\n", encoding="utf-8")
    assert device_id(home) == value
    (home / "device_id").write_text("не hex", encoding="utf-8")
    with pytest.raises(ApiError, match="device_id"):
        device_id(home)
    assert (home / "device_id").read_text(encoding="utf-8") == "не hex"


# ----------------------------- тело входа ----------------------------- #
def test_login_body_uses_the_configured_client_name(monkeypatch):
    monkeypatch.setenv("USER", "ivan")
    body = login_body(device_code="d" * 64, plugin_name="demo-client")
    assert body == {
        "device_code": "d" * 64,
        "env_name": body["env_name"],
        "plugin_version": __version__,
        "plugin_name": "demo-client",
        "env_version": platform.python_version(),
        "login": "ivan",
        "locale": "ru",
    }
    assert body["env_name"] in ("mac", "win", "linux", "other")
    monkeypatch.delenv("USER")
    monkeypatch.setenv("USERNAME", "petr")
    assert login_body(device_code="d", plugin_name="x")["login"] == "petr"
    monkeypatch.delenv("USERNAME")
    assert login_body(device_code="d", plugin_name="x")["login"] == "unknown-user"


# ------------------------------ ссылки ------------------------------ #
@pytest.mark.parametrize(
    ("value", "base", "ok"),
    [
        (SIGNIN, CONFIG, True),
        ("http://code.test/x", CONFIG, False),  # к https-бэкенду — только https
        ("http://127.0.0.1:8000/x", "http://127.0.0.1:8000/api/v2/config", True),
        ("https://code.test/a b", CONFIG, False),
        ("https://code.test/\x1b[31m", CONFIG, False),
        ("/relative", CONFIG, False),
        ("javascript:alert(1)", CONFIG, False),
        ("https://", CONFIG, False),
        (None, CONFIG, False),
    ],
)
def test_safe_url(value, base, ok):
    assert (safe_url(value, base) is not None) is ok


def test_challenge_from_response():
    resp = httpx.Response(
        401, headers={"X-Authorization-Url": POLL, "X-Signin-Gitverse": SIGNIN, "Retry-After": "3"}
    )
    challenge = Challenge.from_response(resp, CONFIG)
    assert (challenge.authorization_url, challenge.signin_url, challenge.retry_after) == (
        POLL,
        SIGNIN,
        3.0,
    )
    clamped = httpx.Response(401, headers={"X-Authorization-Url": POLL, "Retry-After": "900"})
    assert Challenge.from_response(clamped, CONFIG).retry_after == 60.0
    assert Challenge.from_response(clamped, CONFIG).signin_url is None
    with pytest.raises(ApiError, match="X-Authorization-Url"):
        Challenge.from_response(httpx.Response(401, text="нужен вход"), CONFIG)


# ------------------------------- опрос ------------------------------- #
def _challenge(signin: str | None = SIGNIN, retry: float = 3.0) -> Challenge:
    return Challenge(POLL, signin, retry, CONFIG)


def _pending(code: str = "authorization_pending", status: int = 401, **headers):
    return httpx.Response(status, json={"error": code}, headers=headers)


def _granted(token: str = "tok") -> httpx.Response:
    return httpx.Response(200, json={"access_token": token, "expires_in": 3600})


class _Poll:
    def __init__(self, *responses: httpx.Response | BaseException):
        self.responses = list(responses)
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url: str, body: dict) -> httpx.Response:
        self.calls.append((url, body))
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _login(poll, *, links=None, sleeps=None, **kw) -> DeviceLogin:
    return DeviceLogin(
        poll,
        on_link=(links.append if links is not None else lambda _url: None),
        sleep=(sleeps.append if sleeps is not None else lambda _s: None),
        clock=lambda: 0.0,
        **kw,
    )


BODY = {"device_code": "d" * 64, "plugin_name": "demo"}


def test_pending_then_granted():
    poll = _Poll(_pending(), _pending(), _granted("abc"))
    links, sleeps = [], []
    token = _login(poll, links=links, sleeps=sleeps).run(_challenge(), BODY)
    assert token.access_token == "abc" and token.expires_at is not None
    assert links == [SIGNIN]  # один раз
    assert sleeps == [3.0, 3.0]
    assert [url for url, _ in poll.calls] == [POLL] * 3 and poll.calls[0][1] == BODY


def test_granted_at_once_shows_no_link():
    links = []
    assert _login(_Poll(_granted()), links=links).run(_challenge(), BODY).access_token == "tok"
    assert links == []


def test_link_from_poll_is_preferred_and_shown_once():
    device_link = "https://gitverse.test/signin?device=42"
    poll = _Poll(
        _pending(**{"X-Signin-Gitverse": device_link}),
        _pending(**{"X-Signin-Gitverse": "https://gitverse.test/other"}),
        _granted(),
    )
    links = []
    _login(poll, links=links).run(_challenge(), BODY)
    assert links == [device_link]


def test_link_from_challenge_is_replaced_by_a_later_poll_link():
    device_link = "https://gitverse.test/signin?device=42"
    poll = _Poll(_pending(), _pending(**{"X-Signin-Gitverse": device_link}), _pending(), _granted())
    links = []
    _login(poll, links=links).run(_challenge(), BODY)
    assert links == [SIGNIN, device_link]


def test_slow_down_retry_after_and_400_pending():
    poll = _Poll(
        _pending("slow_down"),
        _pending(status=400, **{"Retry-After": "7"}),
        _pending(**{"Retry-After": "0"}),
        _granted(),
    )
    sleeps = []
    _login(poll, sleeps=sleeps).run(_challenge(), BODY)
    assert sleeps == [8.0, 7.0, 1.0]


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(403, json={"error": "plugin_not_allowed"}),
        httpx.Response(401, json={"error": "invalid_client"}),
        httpx.Response(500, text="oops"),
    ],
)
def test_rejection_names_the_client(response):
    with pytest.raises(ApiError) as e:
        _login(_Poll(response)).run(_challenge(), BODY)
    assert "«demo»" in str(e.value) and "GIGACREW_PLUGIN_NAME" in str(e.value)
    assert str(response.status_code) in str(e.value)


def test_budget_is_limited_by_the_sum_of_pauses():
    poll = _Poll(*[_pending()] * 1000)
    sleeps = []
    with pytest.raises(ApiError, match="не подтверждён за 10 мин"):
        _login(poll, sleeps=sleeps).run(_challenge(retry=60.0), BODY)
    assert sum(sleeps) <= 600 and len(sleeps) == 10


def test_budget_is_limited_by_the_clock():
    ticks = iter([0.0, 0.0, 599.0])
    login = DeviceLogin(
        _Poll(_pending(), _pending()),
        on_link=lambda _u: None,
        sleep=lambda _s: None,
        clock=lambda: next(ticks),
    )
    with pytest.raises(ApiError, match="не подтверждён"):
        login.run(_challenge(), BODY)


def test_ctrl_c_while_waiting_propagates():
    def interrupted(_seconds):
        raise KeyboardInterrupt

    login = DeviceLogin(_Poll(_pending()), on_link=lambda _u: None, sleep=interrupted)
    with pytest.raises(KeyboardInterrupt):
        login.run(_challenge(), BODY)


def test_without_a_link_or_a_way_to_show_it():
    with pytest.raises(ApiError, match="ссылку"):
        _login(_Poll(_pending())).run(_challenge(signin=None), BODY)
    login = DeviceLogin(_Poll(_pending()), on_link=None, sleep=lambda _s: None)
    with pytest.raises(ApiError, match="откройте в браузере: " + SIGNIN.replace("?", r"\?")):
        login.run(_challenge(), BODY)


def test_unsafe_links_are_not_shown():
    poll = _Poll(_pending(**{"X-Signin-Gitverse": "http://evil.test/"}), _granted())
    links = []
    _login(poll, links=links).run(_challenge(), BODY)
    assert links == [SIGNIN]


def test_granted_with_a_bad_body():
    with pytest.raises(ApiError, match="не-JSON"):
        _login(_Poll(httpx.Response(200, text="<html>"))).run(_challenge(), BODY)
    with pytest.raises(ApiError, match="access_token"):
        _login(_Poll(httpx.Response(200, json={"token": "x"}))).run(_challenge(), BODY)
