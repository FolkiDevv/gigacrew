"""Тесты загрузки конфигурации: приоритет источников, отсутствие побочных эффектов."""

from __future__ import annotations

import os
import shutil
import ssl
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict, Unpack, cast

import certifi
import pytest
from fakes import ScriptedProvider, text_turn

from gigacrew.cli import main
from gigacrew.config import Config, ConfigError, load_environment, read_dotenv
from gigacrew.llm.base import LLMProvider
from gigacrew.permissions import PermissionMode


class _LoadOverrides(TypedDict, total=False):
    """Необязательные аргументы ``Config.load``, которые тесты передают поверх окружения."""

    save_chats: bool
    mode: PermissionMode


def _env_file(path: Path, text: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / ".env").write_text(text, encoding="utf-8")


def test_read_dotenv_parses_quotes_comments_export(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "# comment\nA=1\nexport B=\"two\"\nC='three'\nbroken line\n",
        encoding="utf-8",
    )
    assert read_dotenv(tmp_path / ".env") == {"A": "1", "B": "two", "C": "three"}
    assert not read_dotenv(tmp_path / "missing.env")


def test_read_dotenv_inline_comments_and_quotes(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "MODEL=Alpha-2-Max # быстрая\n"
        'Q="x # не комментарий" # а это — да\n'
        "S='it''s'\n"
        "URL=https://h/a#b\n"
        'OPEN="без закрывающей\n',
        encoding="utf-8",
    )
    assert read_dotenv(tmp_path / ".env") == {
        "MODEL": "Alpha-2-Max",
        "Q": "x # не комментарий",
        "S": "it",
        "URL": "https://h/a#b",
        "OPEN": '"без закрывающей',
    }


def test_read_dotenv_with_bom_keeps_first_setting(tmp_path: Path) -> None:
    # UTF-8 с BOM пишут PowerShell 5 и старый Блокнот: первая настройка не теряется
    (tmp_path / ".env").write_bytes("GIGACREW_VERIFY_SSL=1\nB=2\n".encode("utf-8-sig"))
    assert read_dotenv(tmp_path / ".env") == {"GIGACREW_VERIFY_SSL": "1", "B": "2"}
    assert Config.load(project_root=tmp_path, environ={}, cwd=tmp_path).verify_ssl is True


def test_undecodable_dotenv_is_a_config_error(tmp_path: Path) -> None:
    (tmp_path / ".env").write_bytes(b"\xff\xfe=1\n")
    with pytest.raises(ConfigError, match=r"\.env"):
        Config.load(project_root=tmp_path, environ={}, cwd=tmp_path)


def test_priority_environ_over_root_over_cwd(tmp_path: Path) -> None:
    root, cwd = tmp_path / "proj", tmp_path / "cwd"
    _env_file(root, "GIGACREW_MODEL=from-root\nGIGACREW_SCOPE=root-scope\n")
    _env_file(cwd, "GIGACREW_MODEL=from-cwd\nGIGACREW_SCOPE=cwd-scope\nGIGACREW_TIMEOUT=7\n")

    env = load_environment(root, environ={"GIGACREW_SCOPE": "env-scope"}, cwd=cwd)
    assert env["GIGACREW_MODEL"] == "from-root"  # root .env важнее cwd .env
    assert env["GIGACREW_SCOPE"] == "env-scope"  # окружение важнее файлов
    assert env["GIGACREW_TIMEOUT"] == "7"  # cwd .env — последний источник

    cfg = Config.load(project_root=root, environ={}, cwd=cwd, model="from-cli")
    assert cfg.model == "from-cli"  # аргумент CLI важнее всего
    assert cfg.timeout == 7


def test_load_does_not_touch_os_environ(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GIGACREW_ACCESS_KEY", raising=False)
    _env_file(tmp_path, "GIGACREW_ACCESS_KEY=secret-from-dotenv\n")
    cfg = Config.load(project_root=tmp_path, environ={}, cwd=tmp_path)
    assert cfg.access_key == "secret-from-dotenv"
    assert "GIGACREW_ACCESS_KEY" not in os.environ


@pytest.mark.parametrize(
    ("name", "raw"),
    [
        ("GIGACREW_TIMEOUT", "abc"),
        ("GIGACREW_TIMEOUT", "0"),
        ("GIGACREW_TEMPERATURE", "warm"),
        ("GIGACREW_TEMPERATURE", "5"),
        ("GIGACREW_MODE", "yolo"),
        ("GIGACREW_COMPACT_THRESHOLD", "5"),
        ("GIGACREW_COMPACT_THRESHOLD", "99"),
        ("GIGACREW_COMPACT_THRESHOLD", "0.8"),
        ("GIGACREW_SUBAGENT_MAX_STEPS", "0"),
        ("GIGACREW_SUBAGENT_CONTEXT", "5"),
        ("GIGACREW_SUBAGENT_CONTEXT", "99"),
        ("GIGACREW_SUBAGENT_TIMEOUT", "5"),
        ("GIGACREW_SUBAGENT_TIMEOUT", "long"),
        ("GIGACREW_READ_TIMEOUT", "0"),
        ("GIGACREW_READ_TIMEOUT", "long"),
        ("GIGACREW_PROVIDER", "openai"),
        ("GIGACREW_USER_AGENT", "агент/1.0"),
        ("GIGACREW_USER_AGENT", "agent\x1b[31m"),
    ],
)
def test_bad_values_raise_config_error(tmp_path: Path, name: str, raw: str) -> None:
    # ошибка называет ту переменную, что задана неверно
    with pytest.raises(ConfigError, match=name):
        Config.load(project_root=tmp_path, environ={name: raw}, cwd=tmp_path)


@pytest.mark.parametrize(
    ("env", "field", "expected"),
    [
        ({"GIGACREW_TEMPERATURE": "0"}, "temperature", float(0)),
        ({"GIGACREW_TEMPERATURE": "2"}, "temperature", 2.0),
        ({"GIGACREW_COMPACT_THRESHOLD": "10"}, "compact_threshold", 0.1),
        ({"GIGACREW_COMPACT_THRESHOLD": "95"}, "compact_threshold", 0.95),
        ({"GIGACREW_SUBAGENT_CONTEXT": "10"}, "subagent_context_share", 0.1),
        ({"GIGACREW_SUBAGENT_CONTEXT": "95"}, "subagent_context_share", 0.95),
        ({"GIGACREW_SUBAGENT_TIMEOUT": "10"}, "subagent_timeout", 10),
        ({"GIGACREW_TIMEOUT": "1"}, "timeout", 1),
        ({"GIGACREW_RATE_LIMIT_WAIT": "0"}, "rate_limit_wait", 0),
    ],
)
def test_bounds_are_inclusive(
    tmp_path: Path,
    env: dict[str, str],
    field: str,
    expected: float,
) -> None:
    cfg = Config.load(project_root=tmp_path, environ=env, cwd=tmp_path)
    assert getattr(cfg, field) == expected


def test_main_reports_config_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIGACREW_ACCESS_KEY", "dummy")
    monkeypatch.setenv("GIGACREW_TIMEOUT", "abc")
    assert main(["-C", str(tmp_path), "-p", "x", "--no-color"]) == 1
    assert "GIGACREW_TIMEOUT" in capsys.readouterr().out


def test_main_without_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("GIGACREW_ACCESS_KEY", "GIGACREW_CERT", "GIGACREW_KEY", "GIGACREW_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    assert main(["-C", str(tmp_path), "-p", "x", "--no-color"]) == 1
    assert "реквизиты" in capsys.readouterr().out


def test_endpoints_and_model_have_no_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Адреса API и модель задаёт только пользователь — без них клиент не стартует."""
    cfg = Config.load(project_root=tmp_path, environ={"GIGACREW_ACCESS_KEY": "k"}, cwd=tmp_path)
    assert (cfg.base_url, cfg.auth_url, cfg.model, cfg.scope) == ("", None, "", None)
    with pytest.raises(RuntimeError, match="GIGACREW_API_URL"):
        cfg.require_credentials()
    cfg = Config.load(
        project_root=tmp_path,
        environ={"GIGACREW_ACCESS_KEY": "k", "GIGACREW_API_URL": "https://api.test/v1/"},
        cwd=tmp_path,
    )
    assert cfg.base_url == "https://api.test/v1"
    with pytest.raises(RuntimeError, match="GIGACREW_AUTH_URL"):
        cfg.require_credentials()
    # mTLS адрес выдачи токена не нужен
    Config(cert="c.pem", key="c.key", base_url="https://api.test/v1").require_credentials()
    # половина пары — не mTLS: понятная ошибка, а не сбой при сборке TLS
    halves: tuple[tuple[str | None, str | None], ...] = (("c.pem", None), (None, "c.key"))
    for cert, key in halves:
        cfg = Config(base_url="https://api.test/v1", cert=cert, key=key)
        assert cfg.auth_mode == "none"
        with pytest.raises(RuntimeError, match="GIGACREW_CERT и GIGACREW_KEY"):
            cfg.require_credentials()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIGACREW_ACCESS_KEY", "dummy")
    monkeypatch.setenv("GIGACREW_AUTH_URL", "https://auth.test/oauth")
    monkeypatch.setenv("GIGACREW_API_URL", "https://api.test/v1")
    monkeypatch.delenv("GIGACREW_MODEL", raising=False)
    monkeypatch.delenv("GIGACREW_PROVIDER", raising=False)
    assert main(["-C", str(tmp_path), "-p", "x", "--no-color"]) == 1
    assert "GIGACREW_MODEL" in capsys.readouterr().out


def test_provider_choice(tmp_path: Path) -> None:
    def load(env: dict[str, str]) -> Config:
        return Config.load(project_root=tmp_path, environ=env, cwd=tmp_path)

    cfg = load({"GIGACREW_ACCESS_KEY": "k"})
    assert cfg.provider == "api"
    assert cfg.auth_mode == "oauth"
    cfg = load({"GIGACREW_PROVIDER": " GigaCode ", "GIGACREW_ACCESS_KEY": "k"})
    assert cfg.provider == "gigacode"
    assert cfg.auth_mode == "cli_token"
    assert cfg.verify_ssl  # проверка TLS — по умолчанию, как у официального CLI
    with pytest.raises(ConfigError, match="GIGACREW_PROVIDER"):
        load({"GIGACREW_PROVIDER": "openai"})


def test_gigacode_settings(tmp_path: Path) -> None:
    """Для GigaCode нужен адрес; каталог CLI с токеном — GIGACODE_HOME, по умолчанию нет."""

    def load(env: dict[str, str]) -> Config:
        return Config.load(project_root=tmp_path, environ=env, cwd=tmp_path)

    base = {"GIGACREW_PROVIDER": "gigacode", "GIGACREW_API_URL": "https://code.test"}
    assert load(base).gigacode_home is None
    assert load({**base, "GIGACODE_HOME": "/opt/gc"}).gigacode_home == "/opt/gc"
    with pytest.raises(RuntimeError, match="GIGACREW_API_URL"):
        load({"GIGACREW_PROVIDER": "gigacode", "GIGACREW_USER_AGENT": "x"}).require_credentials()


def test_gigacode_user_agent_has_no_default(tmp_path: Path) -> None:
    """User-Agent запросов к GigaCode тоже задаёт только пользователь."""

    def load(env: dict[str, str]) -> Config:
        return Config.load(project_root=tmp_path, environ=env, cwd=tmp_path)

    base = {
        "GIGACREW_PROVIDER": "gigacode",
        "GIGACREW_API_URL": "https://code.test",
    }
    cfg = load(base)
    assert cfg.user_agent is None
    with pytest.raises(RuntimeError, match="GIGACREW_USER_AGENT"):
        cfg.require_credentials()
    assert load({**base, "GIGACREW_USER_AGENT": "  "}).user_agent is None
    cfg = load({**base, "GIGACREW_USER_AGENT": " demo-cli/26.10 (darwin; x64) "})
    assert cfg.user_agent == "demo-cli/26.10 (darwin; x64)"
    cfg.require_credentials()  # ключ и сертификат не нужны
    # API v2 без него обходится
    Config(
        access_key="k",
        auth_url="https://a.test",
        base_url="https://api.test/v1",
    ).require_credentials()


def test_save_chats_flag_and_env(tmp_path: Path) -> None:
    def load(env: dict[str, str], **overrides: Unpack[_LoadOverrides]) -> bool:
        return Config.load(project_root=tmp_path, environ=env, cwd=tmp_path, **overrides).save_chats

    assert load({}) is True
    assert load({"GIGACREW_SAVE_CHATS": "0"}) is False
    assert load({"GIGACREW_SAVE_CHATS": "1"}, save_chats=False) is False  # --no-save сильнее


def test_command_safety_settings(tmp_path: Path) -> None:
    def load(env: dict[str, str]) -> tuple[bool, bool]:
        cfg = Config.load(project_root=tmp_path, environ=env, cwd=tmp_path)
        return cfg.safe_commands, cfg.remember_approvals

    assert load({}) == (True, True)
    assert load({"GIGACREW_SAFE_COMMANDS": "0", "GIGACREW_REMEMBER_APPROVALS": "0"}) == (
        False,
        False,
    )


def test_mode_flag_and_env(tmp_path: Path) -> None:
    def load(env: dict[str, str], **overrides: Unpack[_LoadOverrides]) -> PermissionMode:
        return Config.load(project_root=tmp_path, environ=env, cwd=tmp_path, **overrides).mode

    assert load({}) is PermissionMode.AUTO
    assert load({"GIGACREW_MODE": " Plan "}) is PermissionMode.PLAN
    # прежние режимы «ручной» и «авто-правки» — это «авто»
    assert load({"GIGACREW_MODE": "manual"}) is PermissionMode.AUTO
    assert load({"GIGACREW_MODE": "edits"}) is PermissionMode.AUTO
    # --mode сильнее окружения
    assert load({"GIGACREW_MODE": "plan"}, mode=PermissionMode.AUTO) is PermissionMode.AUTO
    with pytest.raises(ConfigError, match="GIGACREW_MODE"):
        load({"GIGACREW_MODE": "yolo"})


def test_reviewer_model(tmp_path: Path) -> None:
    def load(env: dict[str, str]) -> str | None:
        return Config.load(project_root=tmp_path, environ=env, cwd=tmp_path).reviewer_model

    assert load({}) is None
    assert load({"GIGACREW_REVIEWER_MODEL": "  "}) is None
    assert load({"GIGACREW_REVIEWER_MODEL": " Reviewer-Max "}) == "Reviewer-Max"


def test_compaction_settings(tmp_path: Path) -> None:
    def load(env: dict[str, str]) -> Config:
        return Config.load(project_root=tmp_path, environ=env, cwd=tmp_path)

    cfg = load({})
    assert cfg.auto_compact is True
    assert cfg.compact_threshold == pytest.approx(0.8)
    cfg = load({"GIGACREW_AUTO_COMPACT": "0", "GIGACREW_COMPACT_THRESHOLD": "65"})
    assert cfg.auto_compact is False
    assert cfg.compact_threshold == pytest.approx(0.65)


def test_subagent_limits(tmp_path: Path) -> None:
    def load(env: dict[str, str]) -> Config:
        return Config.load(project_root=tmp_path, environ=env, cwd=tmp_path)

    cfg = load({})
    assert (cfg.subagent_max_steps, cfg.subagent_context_share, cfg.subagent_timeout) == (
        25,
        0.8,
        600,
    )
    cfg = load(
        {
            "GIGACREW_SUBAGENT_MAX_STEPS": "10",
            "GIGACREW_SUBAGENT_CONTEXT": "60",
            "GIGACREW_SUBAGENT_TIMEOUT": "120",
        },
    )
    assert (cfg.subagent_max_steps, cfg.subagent_context_share, cfg.subagent_timeout) == (
        10,
        0.6,
        120,
    )


def test_read_timeout(tmp_path: Path) -> None:
    def load(env: dict[str, str]) -> Config:
        return Config.load(project_root=tmp_path, environ=env, cwd=tmp_path)

    assert (load({}).timeout, load({}).read_timeout) == (120, 600)
    assert load({"GIGACREW_READ_TIMEOUT": "900"}).read_timeout == 900


def test_reasoning_mode(tmp_path: Path) -> None:
    def load(env: dict[str, str]) -> Config:
        return Config.load(project_root=tmp_path, environ=env, cwd=tmp_path)

    assert load({}).reasoning == "turn"
    assert load({"GIGACREW_REASONING": " Off "}).reasoning == "off"
    with pytest.raises(ConfigError, match="GIGACREW_REASONING"):
        load({"GIGACREW_REASONING": "always"})


def test_rate_limit_wait(tmp_path: Path) -> None:
    def load(env: dict[str, str]) -> Config:
        return Config.load(project_root=tmp_path, environ=env, cwd=tmp_path)

    assert load({}).rate_limit_wait == 120
    assert load({"GIGACREW_RATE_LIMIT_WAIT": "300"}).rate_limit_wait == 300
    assert load({"GIGACREW_RATE_LIMIT_WAIT": "0"}).rate_limit_wait == 0  # не ждать
    with pytest.raises(ConfigError):
        load({"GIGACREW_RATE_LIMIT_WAIT": "-1"})


def test_yes_all_implies_auto_approve(tmp_path: Path) -> None:
    cfg = Config.load(project_root=tmp_path, yes_all=True, environ={}, cwd=tmp_path)
    assert cfg.auto_approve
    assert cfg.yes_all
    cfg = Config.load(project_root=tmp_path, auto_approve=True, environ={}, cwd=tmp_path)
    assert cfg.auto_approve
    assert not cfg.yes_all


def test_ca_bundle_enables_verification(tmp_path: Path) -> None:
    env = {"GIGACREW_ACCESS_KEY": "k", "GIGACREW_CA_BUNDLE": certifi.where()}
    cfg = Config.load(project_root=tmp_path, environ=env, cwd=tmp_path)
    assert cfg.verify_ssl
    assert cfg.ca_bundle == certifi.where()
    context = cfg.build_ssl_verify()
    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED
    # явный GIGACREW_VERIFY_SSL=0 важнее
    cfg = Config.load(
        project_root=tmp_path,
        environ={**env, "GIGACREW_VERIFY_SSL": "0"},
        cwd=tmp_path,
    )
    assert cfg.build_ssl_verify() is False
    with pytest.raises(ConfigError, match="GIGACREW_CA_BUNDLE"):
        Config.load(
            project_root=tmp_path,
            environ={"GIGACREW_CA_BUNDLE": str(tmp_path / "нет.pem")},
            cwd=tmp_path,
        )


@pytest.mark.parametrize(
    ("flag", "expected"),
    [("1", True), (" Yes ", True), ("ON", True), ("off", False), ("No", False)],
)
def test_boolean_settings(tmp_path: Path, flag: str, *, expected: bool) -> None:
    env = {"GIGACREW_PROVIDER": "gigacode", "GIGACREW_VERIFY_SSL": flag}
    assert Config.load(project_root=tmp_path, environ=env, cwd=tmp_path).verify_ssl is expected


@pytest.mark.parametrize("flag", ["ture", "enabled", "y"])
def test_invalid_boolean_is_an_error(tmp_path: Path, flag: str) -> None:
    """Опечатка не выключает проверку TLS молча — ошибка конфигурации."""
    env = {"GIGACREW_PROVIDER": "gigacode", "GIGACREW_VERIFY_SSL": flag}
    with pytest.raises(ConfigError, match="GIGACREW_VERIFY_SSL"):
        Config.load(project_root=tmp_path, environ=env, cwd=tmp_path)


def _self_signed(tmp_path: Path, cn: str = "test") -> tuple[Path, Path]:
    if shutil.which("openssl") is None:
        pytest.skip("нет openssl")
    tmp_path.mkdir(parents=True, exist_ok=True)
    cert, key = tmp_path / "c.pem", tmp_path / "k.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-subj",
            f"/CN={cn}",
            "-keyout",
            str(key),
            "-out",
            str(cert),
        ],
        check=True,
        capture_output=True,
    )
    return cert, key


@pytest.fixture
def _certifi_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    """Стандартные корни — из certifi, как у httpx.

    Не из SSL_CERT_FILE/SSL_CERT_DIR машины: корни из каталога грузятся лениво, и
    get_ca_certs() их не показал бы.
    """
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)


def _common_names(cfg: Config) -> set[str]:
    context = cfg.build_ssl_verify()
    assert isinstance(context, ssl.SSLContext)
    names: set[str] = set()
    for cert in context.get_ca_certs():
        # В typeshed subject — объединение типов; на деле — кортеж RDN из пар (ключ, значение).
        subject = cast("tuple[tuple[tuple[str, str], ...], ...]", cert["subject"])
        names.update(name for rdn in subject for key, name in rdn if key == "commonName")
    return names


@pytest.mark.usefixtures("_certifi_roots")
def test_gigacode_verifies_tls_by_default(tmp_path: Path) -> None:
    env = {"GIGACREW_PROVIDER": "gigacode"}
    cfg = Config.load(project_root=tmp_path, environ=env, cwd=tmp_path)
    assert cfg.verify_ssl  # токен официального CLI без проверки сервера не уходит
    context = cfg.build_ssl_verify()
    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname
    assert context.get_ca_certs()
    cfg = Config.load(
        project_root=tmp_path,
        environ={**env, "GIGACREW_VERIFY_SSL": "0"},
        cwd=tmp_path,
    )
    assert cfg.build_ssl_verify() is False  # явное значение важнее


@pytest.mark.usefixtures("_certifi_roots")
def test_gigacode_trusts_official_cli_certificates(tmp_path: Path, user_home: Path) -> None:
    certs = user_home / ".gigacode" / "certs"
    certs.mkdir(parents=True)
    _self_signed(tmp_path / "a", "bundled")[0].rename(certs / "gigacode-bundled-1.pem")
    _self_signed(tmp_path / "b", "extra")[0].rename(certs / "extra.crt")
    (certs / "README.txt").write_text("не сертификат — не читается")
    env = {"GIGACREW_PROVIDER": "gigacode"}

    cfg = Config.load(project_root=tmp_path, environ=env, cwd=tmp_path)
    assert {"bundled", "extra"} <= _common_names(cfg)

    own, _ = _self_signed(tmp_path / "c", "own")  # GIGACREW_CA_BUNDLE — в дополнение
    cfg = Config.load(
        project_root=tmp_path,
        environ={**env, "GIGACREW_CA_BUNDLE": str(own)},
        cwd=tmp_path,
    )
    names = _common_names(cfg)
    assert {"bundled", "extra", "own"} <= names
    assert len(names) > 3

    other = tmp_path / "other-home"
    (other / "certs").mkdir(parents=True)
    _self_signed(tmp_path / "d", "other")[0].rename(other / "certs" / "root.pem")
    cfg = Config.load(
        project_root=tmp_path,
        environ={**env, "GIGACODE_HOME": str(other)},
        cwd=tmp_path,
    )
    names = _common_names(cfg)
    assert "other" in names
    assert "bundled" not in names


def test_gigacode_broken_certificate_is_an_error(tmp_path: Path, user_home: Path) -> None:
    certs = user_home / ".gigacode" / "certs"
    certs.mkdir(parents=True)
    (certs / "broken.pem").write_text("-----BEGIN CERTIFICATE-----\nмусор\n")
    cfg = Config.load(
        project_root=tmp_path,
        environ={"GIGACREW_PROVIDER": "gigacode"},
        cwd=tmp_path,
    )
    with pytest.raises(RuntimeError, match=r"broken\.pem"):
        cfg.build_ssl_verify()


def test_mtls_verification_loads_trust_anchors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Голый SSLContext без корневых сертификатов не проверил бы ни один сервер."""
    cert, key = _self_signed(tmp_path)
    # Системные корни бывают только в capath (подгружаются лениво, счётчики пусты) —
    # подменяем хранилище по умолчанию, чтобы проверка не зависела от машины.
    monkeypatch.setenv("SSL_CERT_FILE", str(cert))
    base = {"GIGACREW_CERT": str(cert), "GIGACREW_KEY": str(key)}
    cfg = Config.load(
        project_root=tmp_path,
        environ={**base, "GIGACREW_VERIFY_SSL": "1"},
        cwd=tmp_path,
    )
    context = cfg.build_ssl_verify()
    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.cert_store_stats()["x509_ca"] > 0 or context.get_ca_certs() != []
    cfg = Config.load(
        project_root=tmp_path,
        environ={**base, "GIGACREW_CA_BUNDLE": str(cert)},
        cwd=tmp_path,
    )
    context = cfg.build_ssl_verify()
    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert len(context.get_ca_certs()) == 1


# ------------------------- проверка TLS у API v2 ------------------------- #
@pytest.mark.usefixtures("_certifi_roots")
def test_api_verifies_tls_by_default(tmp_path: Path) -> None:
    """Ключ OAuth и Bearer-токен не уходят на непроверенный сервер.

    Проверка TLS у API v2 включена и без GIGACREW_CA_BUNDLE — стандартными корнями.
    """
    assert Config().verify_ssl  # и у конфигурации, собранной вручную
    cfg = Config.load(project_root=tmp_path, environ={"GIGACREW_ACCESS_KEY": "k"}, cwd=tmp_path)
    assert cfg.verify_ssl
    context = cfg.build_ssl_verify()
    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname
    assert context.get_ca_certs()
    env = {"GIGACREW_ACCESS_KEY": "k", "GIGACREW_VERIFY_SSL": "0"}  # явный отказ — работает
    cfg = Config.load(project_root=tmp_path, environ=env, cwd=tmp_path)
    assert not cfg.verify_ssl
    assert cfg.build_ssl_verify() is False


@pytest.mark.usefixtures("_certifi_roots")
def test_mtls_verifies_server_by_default(tmp_path: Path) -> None:
    cert, key = _self_signed(tmp_path)
    env = {"GIGACREW_CERT": str(cert), "GIGACREW_KEY": str(key)}
    context = Config.load(project_root=tmp_path, environ=env, cwd=tmp_path).build_ssl_verify()
    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname
    assert context.get_ca_certs()
    env["GIGACREW_VERIFY_SSL"] = "0"
    context = Config.load(project_root=tmp_path, environ=env, cwd=tmp_path).build_ssl_verify()
    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_NONE


# ------------------- настройки безопасности из .env проекта ------------------- #
_SECURITY = (
    "GIGACREW_API_URL=https://collector.test/v1\n"
    "GIGACREW_AUTH_URL=https://collector.test/oauth\n"
    "GIGACREW_VERIFY_SSL=0\n"
    "GIGACREW_CA_BUNDLE={bundle}\n"
    "GIGACODE_HOME={home}\n"
    "GIGACREW_CERT=repo.pem\n"
    "GIGACREW_KEY=repo.key\n"
)


def test_project_dotenv_cannot_set_security_settings(tmp_path: Path) -> None:
    """Закоммиченный .env репозитория не меняет настройки безопасности.

    Он не перенаправит ключ на чужой адрес, не выключит проверку TLS и не подложит свои
    корни или токен; ключ доступа и прочие настройки — можно.
    """
    root, cwd = tmp_path / "repo", tmp_path / "repo" / "sub"
    _env_file(
        root,
        _SECURITY.format(bundle=certifi.where(), home=root)
        + "GIGACREW_ACCESS_KEY=k\nGIGACREW_MODEL=M\nGIGACREW_TIMEOUT=7\nGIGACREW_MODE=plan\n",
    )
    _env_file(cwd, "GIGACREW_AUTH_URL=https://other.test/oauth\nGIGACREW_TEMPERATURE=1\n")
    cfg = Config.load(project_root=root, environ={}, cwd=cwd)
    assert (cfg.base_url, cfg.auth_url, cfg.ca_bundle, cfg.gigacode_home) == ("", None, None, None)
    assert cfg.verify_ssl
    assert (cfg.cert, cfg.key) == (None, None)
    assert (cfg.access_key, cfg.model, cfg.timeout, cfg.mode.value) == ("k", "M", 7, "plan")
    assert cfg.temperature == pytest.approx(1.0)
    keys = {key for key, _ in cfg.ignored_env}
    assert keys == {
        "GIGACREW_API_URL",
        "GIGACREW_AUTH_URL",
        "GIGACREW_VERIFY_SSL",
        "GIGACREW_CA_BUNDLE",
        "GIGACODE_HOME",
        "GIGACREW_CERT",
        "GIGACREW_KEY",
    }
    assert ("GIGACREW_AUTH_URL", str(cwd.resolve() / ".env")) in cfg.ignored_env
    text = cfg.ignored_env_warning
    assert text is not None
    assert "GIGACREW_AUTH_URL" in text
    assert "~/.gigacode/.env" in text


def test_user_dotenv_and_environment_are_trusted(tmp_path: Path, user_home: Path) -> None:
    """Настройки безопасности задаются окружением или ~/.gigacode/.env.

    Файл пользователя важнее .env проекта, окружение — важнее всего.
    """
    root = tmp_path / "repo"
    _env_file(root, "GIGACREW_AUTH_URL=https://collector.test/oauth\nGIGACREW_MODEL=repo-M\n")
    _env_file(
        user_home / ".gigacode",
        "GIGACREW_AUTH_URL=https://auth.mine/oauth\nGIGACREW_VERIFY_SSL=0\n"
        "GIGACREW_MODEL=user-M\nGIGACREW_API_URL=https://api.mine/v1\n",
    )
    cfg = Config.load(project_root=root, environ={}, cwd=root)
    assert cfg.auth_url == "https://auth.mine/oauth"
    assert not cfg.verify_ssl
    assert cfg.base_url == "https://api.mine/v1"
    assert cfg.model == "user-M"  # файл пользователя важнее .env проекта и для обычных настроек
    assert [key for key, _ in cfg.ignored_env] == ["GIGACREW_AUTH_URL"]
    env = {"GIGACREW_AUTH_URL": "https://auth.env/oauth"}
    assert (
        Config.load(project_root=root, environ=env, cwd=root).auth_url == env["GIGACREW_AUTH_URL"]
    )
    # то же значение, что задано доверенно, — не повод для предупреждения
    _env_file(root, "GIGACREW_AUTH_URL=https://auth.mine/oauth\n")
    cfg = Config.load(project_root=root, environ={}, cwd=root)
    assert not cfg.ignored_env
    assert cfg.ignored_env_warning is None


def test_trusted_dotenv_reads_security_settings(tmp_path: Path) -> None:
    """Явный доверенный режим — для собственного .env разработчиков (live-тесты)."""
    _env_file(tmp_path, "GIGACREW_AUTH_URL=https://auth.dev/oauth\nGIGACREW_VERIFY_SSL=0\n")
    env = load_environment(tmp_path, environ={}, cwd=tmp_path, trusted_dotenv=True)
    assert env["GIGACREW_AUTH_URL"] == "https://auth.dev/oauth"
    assert "GIGACREW_AUTH_URL" not in load_environment(tmp_path, environ={}, cwd=tmp_path)
    cfg = Config.load(project_root=tmp_path, environ={}, cwd=tmp_path, trusted_dotenv=True)
    assert cfg.auth_url == "https://auth.dev/oauth"
    assert not cfg.verify_ssl
    assert not cfg.ignored_env


def test_main_warns_about_ignored_project_settings(
    tmp_path: Path,
    oneshot: Callable[[LLMProvider], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _env_file(tmp_path, "GIGACREW_AUTH_URL=https://collector.test/oauth\nGIGACREW_KEY=x\n")
    oneshot(ScriptedProvider([text_turn("ok")]))
    assert main(["-C", str(tmp_path), "-p", "x", "--no-color", "--no-stream"]) == 0
    out = " ".join(capsys.readouterr().out.split())
    assert "GIGACREW_AUTH_URL, GIGACREW_KEY" in out
    assert "~/.gigacode/.env" in out
