"""Тесты загрузки конфигурации: приоритет источников, отсутствие побочных эффектов."""

from __future__ import annotations

import os

import pytest

from gigacrew.cli import main
from gigacrew.config import Config, ConfigError, load_environment, read_dotenv


def _env_file(path, text):
    path.mkdir(parents=True, exist_ok=True)
    (path / ".env").write_text(text, encoding="utf-8")


def test_read_dotenv_parses_quotes_comments_export(tmp_path):
    (tmp_path / ".env").write_text(
        "# comment\nA=1\nexport B=\"two\"\nC='three'\nbroken line\n", encoding="utf-8"
    )
    assert read_dotenv(tmp_path / ".env") == {"A": "1", "B": "two", "C": "three"}
    assert read_dotenv(tmp_path / "missing.env") == {}


def test_read_dotenv_inline_comments_and_quotes(tmp_path):
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


def test_priority_environ_over_root_over_cwd(tmp_path):
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


def test_load_does_not_touch_os_environ(tmp_path, monkeypatch):
    monkeypatch.delenv("GIGACREW_ACCESS_KEY", raising=False)
    _env_file(tmp_path, "GIGACREW_ACCESS_KEY=secret-from-dotenv\n")
    cfg = Config.load(project_root=tmp_path, environ={}, cwd=tmp_path)
    assert cfg.access_key == "secret-from-dotenv"
    assert "GIGACREW_ACCESS_KEY" not in os.environ


@pytest.mark.parametrize(
    "env",
    [
        {"GIGACREW_TIMEOUT": "abc"},
        {"GIGACREW_TIMEOUT": "0"},
        {"GIGACREW_TEMPERATURE": "warm"},
        {"GIGACREW_TEMPERATURE": "5"},
        {"GIGACREW_MODE": "yolo"},
        {"GIGACREW_COMPACT_THRESHOLD": "5"},
        {"GIGACREW_COMPACT_THRESHOLD": "99"},
        {"GIGACREW_COMPACT_THRESHOLD": "0.8"},
        {"GIGACREW_SUBAGENT_MAX_STEPS": "0"},
        {"GIGACREW_SUBAGENT_CONTEXT": "5"},
        {"GIGACREW_SUBAGENT_CONTEXT": "99"},
        {"GIGACREW_SUBAGENT_TIMEOUT": "5"},
        {"GIGACREW_SUBAGENT_TIMEOUT": "long"},
        {"GIGACREW_READ_TIMEOUT": "0"},
        {"GIGACREW_READ_TIMEOUT": "long"},
    ],
)
def test_bad_values_raise_config_error(tmp_path, env):
    with pytest.raises(ConfigError):
        Config.load(project_root=tmp_path, environ=env, cwd=tmp_path)


def test_main_reports_config_error(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIGACREW_ACCESS_KEY", "dummy")
    monkeypatch.setenv("GIGACREW_TIMEOUT", "abc")
    assert main(["-C", str(tmp_path), "-p", "x", "--no-color"]) == 1
    assert "GIGACREW_TIMEOUT" in capsys.readouterr().out


def test_main_without_credentials(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    for name in ("GIGACREW_ACCESS_KEY", "GIGACREW_CERT", "GIGACREW_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert main(["-C", str(tmp_path), "-p", "x", "--no-color"]) == 1
    assert "реквизиты" in capsys.readouterr().out


def test_endpoints_and_model_have_no_defaults(tmp_path, monkeypatch, capsys):
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

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIGACREW_ACCESS_KEY", "dummy")
    monkeypatch.setenv("GIGACREW_AUTH_URL", "https://auth.test/oauth")
    monkeypatch.setenv("GIGACREW_API_URL", "https://api.test/v1")
    monkeypatch.delenv("GIGACREW_MODEL", raising=False)
    assert main(["-C", str(tmp_path), "-p", "x", "--no-color"]) == 1
    assert "GIGACREW_MODEL" in capsys.readouterr().out


def test_save_chats_flag_and_env(tmp_path):
    def load(env, **kw):
        return Config.load(project_root=tmp_path, environ=env, cwd=tmp_path, **kw).save_chats

    assert load({}) is True
    assert load({"GIGACREW_SAVE_CHATS": "0"}) is False
    assert load({"GIGACREW_SAVE_CHATS": "1"}, save_chats=False) is False  # --no-save сильнее


def test_command_safety_settings(tmp_path):
    def load(env):
        cfg = Config.load(project_root=tmp_path, environ=env, cwd=tmp_path)
        return cfg.safe_commands, cfg.remember_approvals

    assert load({}) == (True, True)
    assert load({"GIGACREW_SAFE_COMMANDS": "0", "GIGACREW_REMEMBER_APPROVALS": "0"}) == (
        False,
        False,
    )


def test_mode_flag_and_env(tmp_path):
    from gigacrew.permissions import PermissionMode

    def load(env, **kw):
        return Config.load(project_root=tmp_path, environ=env, cwd=tmp_path, **kw).mode

    assert load({}) is PermissionMode.MANUAL
    assert load({"GIGACREW_MODE": " Plan "}) is PermissionMode.PLAN
    # --mode сильнее окружения
    assert load({"GIGACREW_MODE": "plan"}, mode=PermissionMode.ACCEPT_EDITS) is (
        PermissionMode.ACCEPT_EDITS
    )
    with pytest.raises(ConfigError, match="GIGACREW_MODE"):
        load({"GIGACREW_MODE": "auto"})


def test_compaction_settings(tmp_path):
    def load(env):
        return Config.load(project_root=tmp_path, environ=env, cwd=tmp_path)

    cfg = load({})
    assert cfg.auto_compact is True and cfg.compact_threshold == 0.8
    cfg = load({"GIGACREW_AUTO_COMPACT": "0", "GIGACREW_COMPACT_THRESHOLD": "65"})
    assert cfg.auto_compact is False and cfg.compact_threshold == 0.65


def test_subagent_limits(tmp_path):
    def load(env):
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
        }
    )
    assert (cfg.subagent_max_steps, cfg.subagent_context_share, cfg.subagent_timeout) == (
        10,
        0.6,
        120,
    )


def test_read_timeout(tmp_path):
    def load(env):
        return Config.load(project_root=tmp_path, environ=env, cwd=tmp_path)

    assert (load({}).timeout, load({}).read_timeout) == (120, 600)
    assert load({"GIGACREW_READ_TIMEOUT": "900"}).read_timeout == 900


def test_reasoning_mode(tmp_path):
    def load(env):
        return Config.load(project_root=tmp_path, environ=env, cwd=tmp_path)

    assert load({}).reasoning == "turn"
    assert load({"GIGACREW_REASONING": " Off "}).reasoning == "off"
    with pytest.raises(ConfigError, match="GIGACREW_REASONING"):
        load({"GIGACREW_REASONING": "always"})


def test_rate_limit_wait(tmp_path):
    def load(env):
        return Config.load(project_root=tmp_path, environ=env, cwd=tmp_path)

    assert load({}).rate_limit_wait == 120
    assert load({"GIGACREW_RATE_LIMIT_WAIT": "300"}).rate_limit_wait == 300
    assert load({"GIGACREW_RATE_LIMIT_WAIT": "0"}).rate_limit_wait == 0  # не ждать
    with pytest.raises(ConfigError):
        load({"GIGACREW_RATE_LIMIT_WAIT": "-1"})


def test_yes_all_implies_auto_approve(tmp_path):
    cfg = Config.load(project_root=tmp_path, yes_all=True, environ={}, cwd=tmp_path)
    assert cfg.auto_approve and cfg.yes_all
    cfg = Config.load(project_root=tmp_path, auto_approve=True, environ={}, cwd=tmp_path)
    assert cfg.auto_approve and not cfg.yes_all


def test_ca_bundle_enables_verification(tmp_path):
    import ssl

    import certifi

    env = {"GIGACREW_ACCESS_KEY": "k", "GIGACREW_CA_BUNDLE": certifi.where()}
    cfg = Config.load(project_root=tmp_path, environ=env, cwd=tmp_path)
    assert cfg.verify_ssl and cfg.ca_bundle == certifi.where()
    context = cfg.build_ssl_verify()
    assert isinstance(context, ssl.SSLContext) and context.verify_mode == ssl.CERT_REQUIRED
    # явный GIGACREW_VERIFY_SSL=0 важнее
    cfg = Config.load(
        project_root=tmp_path, environ={**env, "GIGACREW_VERIFY_SSL": "0"}, cwd=tmp_path
    )
    assert cfg.build_ssl_verify() is False
    with pytest.raises(ConfigError, match="GIGACREW_CA_BUNDLE"):
        Config.load(
            project_root=tmp_path,
            environ={"GIGACREW_CA_BUNDLE": str(tmp_path / "нет.pem")},
            cwd=tmp_path,
        )


def _self_signed(tmp_path):
    import shutil
    import subprocess

    if shutil.which("openssl") is None:
        pytest.skip("нет openssl")
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
            "/CN=test",
            "-keyout",
            str(key),
            "-out",
            str(cert),
        ],
        check=True,
        capture_output=True,
    )
    return cert, key


def test_mtls_verification_loads_trust_anchors(tmp_path, monkeypatch):
    """Голый SSLContext без корневых сертификатов не проверил бы ни один сервер."""
    import ssl

    cert, key = _self_signed(tmp_path)
    # Системные корни бывают только в capath (подгружаются лениво, счётчики пусты) —
    # подменяем хранилище по умолчанию, чтобы проверка не зависела от машины.
    monkeypatch.setenv("SSL_CERT_FILE", str(cert))
    base = {"GIGACREW_CERT": str(cert), "GIGACREW_KEY": str(key)}
    cfg = Config.load(
        project_root=tmp_path, environ={**base, "GIGACREW_VERIFY_SSL": "1"}, cwd=tmp_path
    )
    context = cfg.build_ssl_verify()
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.cert_store_stats()["x509_ca"] > 0 or context.get_ca_certs() != []
    cfg = Config.load(
        project_root=tmp_path, environ={**base, "GIGACREW_CA_BUNDLE": str(cert)}, cwd=tmp_path
    )
    context = cfg.build_ssl_verify()
    assert context.verify_mode == ssl.CERT_REQUIRED and len(context.get_ca_certs()) == 1
