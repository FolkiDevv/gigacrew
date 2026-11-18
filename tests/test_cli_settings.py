"""Команда /settings: применение изменений к агенту, перекрытие переменными, перенос из .env."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fakes import CatalogueProvider, ScriptedProvider

from gigacrew.agent.loop import Agent
from gigacrew.cli.commands import CommandContext, default_commands
from gigacrew.cli.settings import SettingsSession, connection_lines, import_env
from gigacrew.config import Config
from gigacrew.project.preferences import preferences_file
from gigacrew.tools import build_default_registry
from gigacrew.ui.console import Console
from gigacrew.ui.settings_picker import SettingRow


def _agent(tmp_path: Path, provider: ScriptedProvider | None = None) -> Agent:
    config = Config(access_key="x", project_root=tmp_path, stream=False)
    return Agent(provider or ScriptedProvider(), build_default_registry(), config, Console())


def _session(agent: Agent, environ: dict[str, str] | None = None) -> SettingsSession:
    return SettingsSession(agent, ["A", "B"], environ=environ or {})


def _row(session: SettingsSession, key: str) -> SettingRow:
    return next(row for row in session.rows() if row.spec.key == key)


def _stored() -> object:
    return json.loads(preferences_file().read_text(encoding="utf-8"))


def test_live_setting_applies_to_agent(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    session = _session(agent)
    change = session.change(_row(session, "model.temperature"), 0.7)
    assert not change.failed
    assert "сразу" in change.message
    assert agent.config.temperature == pytest.approx(0.7)
    assert _stored() == {"model": {"temperature": 0.7}, "version": 1}
    assert _row(session, "model.temperature").stored


def test_restart_setting_is_saved_but_not_applied(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    session = _session(agent)
    change = session.change(_row(session, "network.timeout"), 30)
    assert "следующего запуска" in change.message
    assert agent.config.timeout == 120
    assert any("Таймаут запросов" in notice for notice in session.summary())


def test_overridden_setting_reports_source(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    session = _session(agent, {"GIGACREW_SAFE_COMMANDS": "0"})
    row = _row(session, "permissions.safeCommands")
    assert row.source == "переменная окружения"
    assert row.setting is False
    change = session.change(row, setting=True)
    assert "GIGACREW_SAFE_COMMANDS" in change.message
    assert change.row.setting is False  # действует переменная
    assert agent.config.safe_commands  # агент не тронут


def test_reviewer_model_rebuilds_reviewer(tmp_path: Path) -> None:
    agent = _agent(tmp_path, CatalogueProvider())
    session = SettingsSession(agent, None, environ={})  # список моделей не загружен
    change = session.change(_row(session, "permissions.reviewerModel"), "cat-b")
    assert agent.reviewer.model == "cat-b"
    assert not change.failed
    change = session.change(_row(session, "permissions.reviewerModel"), "нет-такой")
    assert change.failed
    assert agent.reviewer.model == "cat-b"


def test_unwritable_file_fails_softly(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    session = _session(agent)
    preferences_file().parent.mkdir(parents=True, exist_ok=True)
    preferences_file().write_text("{сломан", encoding="utf-8")
    change = session.change(_row(session, "model.temperature"), 0.7)
    assert change.failed
    assert "не сохранено" in change.message
    assert agent.config.temperature == pytest.approx(0.2)


def test_import_env_moves_behaviour_settings(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env_file = preferences_file().with_name(".env")
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text(
        "# мой файл\n"
        "GIGACREW_ACCESS_KEY=секрет\n"
        "export GIGACREW_MODE=plan\n"
        "GIGACREW_COMPACT_THRESHOLD=60  # меньше\n"
        "GIGACREW_TEMPERATURE=9\n",
        encoding="utf-8",
    )
    import_env(Console(no_color=True), tmp_path)
    out = capsys.readouterr().out
    assert "GIGACREW_TEMPERATURE" in out  # с ошибкой — остаётся
    assert env_file.read_text(encoding="utf-8") == (
        "# мой файл\nGIGACREW_ACCESS_KEY=секрет\nGIGACREW_TEMPERATURE=9\n"
    )
    assert _stored() == {
        "permissions": {"defaultMode": "plan"},
        "context": {"compactThreshold": 60},
        "version": 1,
    }


def test_import_env_without_settings(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import_env(Console(no_color=True), tmp_path)
    assert "нет настроек" in capsys.readouterr().out


def test_command_without_terminal_lists_settings(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent = _agent(tmp_path)
    commands = default_commands()
    ctx = CommandContext(agent=agent, ui=Console(no_color=True, width=200), commands=commands)
    assert commands.dispatch("/config", ctx)
    out = capsys.readouterr().out
    assert "Температура: 0.2" in out
    assert commands.dispatch("/settings what", ctx)
    assert "использование: /settings [import-env]" in capsys.readouterr().out


def test_connection_lines() -> None:
    lines = dict(
        connection_lines(Config(provider="gigacode", base_url="https://g", verify_ssl=False)),
    )
    assert lines["провайдер"] == "бэкенд GigaCode"
    assert lines["авторизация"] == "токен официального CLI GigaCode"
    assert "ВЫКЛЮЧЕНА" in lines["TLS"]
    assert lines["User-Agent"] == "не задан"


def test_import_env_ignores_empty_lines(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Пустая переменная (как в скопированном .env.example) не сбрасывает значение в файле."""
    env_file = preferences_file().with_name(".env")
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text("GIGACREW_TEMPERATURE=\n", encoding="utf-8")
    preferences_file().write_text('{"model": {"temperature": 0.7}}', encoding="utf-8")
    import_env(Console(no_color=True), tmp_path)
    assert "нет настроек" in capsys.readouterr().out
    assert _stored() == {"model": {"temperature": 0.7}}
    assert env_file.read_text(encoding="utf-8") == "GIGACREW_TEMPERATURE=\n"


def test_import_env_keeps_settings_of_project_env(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Перенос не меняет действующее значение: .env проекта главнее файла настроек."""
    env_file = preferences_file().with_name(".env")
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text("GIGACREW_TEMPERATURE=0.7\nGIGACREW_TY=0\n", encoding="utf-8")
    (tmp_path / ".env").write_text("GIGACREW_TEMPERATURE=0.4\n", encoding="utf-8")
    import_env(Console(no_color=True, width=200), tmp_path)
    assert "GIGACREW_TEMPERATURE остаётся" in capsys.readouterr().out
    assert env_file.read_text(encoding="utf-8") == "GIGACREW_TEMPERATURE=0.7\n"
    assert _stored() == {"version": 1, "tools": {"ty": False}}


def test_default_model_must_be_in_catalogue(tmp_path: Path) -> None:
    session = _session(_agent(tmp_path, CatalogueProvider()))
    row = _row(session, "model.default")
    change = session.change(row, "нет-такой")
    assert change.failed
    assert "доступны: cat-a, cat-b" in change.message
    assert not preferences_file().exists()
    assert session.change(row, "CAT-B").row.setting == "cat-b"  # регистр — как в каталоге


def test_model_without_catalogue_is_free(tmp_path: Path) -> None:
    """Без каталога бэкенда (API v2) модель не проверяется: запуск её тоже не проверяет."""
    session = _session(_agent(tmp_path))
    assert session.change(_row(session, "model.default"), "Новая").row.setting == "Новая"
