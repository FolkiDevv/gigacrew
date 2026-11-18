"""Настройки пользователя в ~/.gigacode/gigacrew.json: разбор, запись и место в Config."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import MappingProxyType

import pytest

from gigacrew.config import Config, ConfigError, env_sources, setting_field
from gigacrew.permissions import PermissionMode
from gigacrew.project.preferences import (
    SETTINGS,
    SettingKind,
    SettingSpec,
    load_preferences,
    preferences_file,
    store_setting,
)

_BY_KEY = MappingProxyType({spec.key: spec for spec in SETTINGS})
TEMPERATURE = _BY_KEY["model.temperature"]
COMPACT = _BY_KEY["context.compactThreshold"]
BUDGET = _BY_KEY["context.budgetTokens"]
MODE = _BY_KEY["permissions.defaultMode"]
SAFE = _BY_KEY["permissions.safeCommands"]
REVIEWER = _BY_KEY["permissions.reviewerModel"]


def _write(path: Path, content: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content), encoding="utf-8")
    return path


def _load(tmp_path: Path, environ: dict[str, str] | None = None) -> Config:
    return Config.load(project_root=tmp_path, cwd=tmp_path, environ=environ or {})


def test_registry_is_consistent() -> None:
    keys = [spec.key for spec in SETTINGS]
    envs = [spec.env for spec in SETTINGS]
    assert len(set(keys)) == len(keys)
    assert len(set(envs)) == len(envs)
    fields = {field.name for field in dataclasses.fields(Config)}
    assert {spec.field for spec in SETTINGS} <= fields


@pytest.mark.parametrize("spec", SETTINGS, ids=lambda spec: spec.key)
def test_config_defaults_match_registry(spec: SettingSpec) -> None:
    """Значения по умолчанию в Config и в реестре не расходятся."""
    assert getattr(Config(), spec.field) == setting_field(spec, spec.default)


@pytest.mark.parametrize(
    ("spec", "raw", "expected"),
    [
        (SAFE, "off", False),
        (SAFE, " Yes ", True),
        (TEMPERATURE, "0.7", 0.7),
        (COMPACT, "60", 60),
        (BUDGET, "", None),
        (MODE, "manual", "auto"),  # прежнее имя
        (_BY_KEY["model.reasoning"], "OFF", "off"),
        (REVIEWER, "  Big  ", "Big"),
    ],
)
def test_parse_text(spec: SettingSpec, raw: str, expected: object) -> None:
    assert spec.parse(raw) == expected


@pytest.mark.parametrize(
    ("spec", "raw", "message"),
    [
        (SAFE, "maybe", "ожидается 1/0"),
        (COMPACT, "5", "не меньше 10"),
        (COMPACT, "99", "не больше 95"),
        (COMPACT, "x", "целое число"),
        (TEMPERATURE, "3", "[0.0; 2.0]"),
        (_BY_KEY["model.reasoning"], "full", "turn, off"),
    ],
)
def test_parse_text_errors(spec: SettingSpec, raw: str, message: str) -> None:
    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        spec.parse(raw)


@pytest.mark.parametrize(
    ("spec", "raw"),
    [(SAFE, 1), (SAFE, "true"), (COMPACT, 60.5), (COMPACT, True), (TEMPERATURE, "0.3"), (MODE, 3)],
)
def test_check_json_rejects_wrong_types(spec: SettingSpec, raw: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        spec.check(raw)


def test_check_json_accepts_and_normalizes() -> None:
    assert TEMPERATURE.check(1) == pytest.approx(1.0)
    assert REVIEWER.check(None) is None
    assert BUDGET.check(None) is None
    assert MODE.check("PLAN") == "plan"


def test_show_and_text() -> None:
    assert SAFE.show(setting=True) == "вкл"
    assert COMPACT.show(80) == "80 %"
    assert REVIEWER.show(None) == "модель чата"
    assert BUDGET.show(None) == "по окну модели"
    assert SAFE.text(setting=False) == "0"
    assert not BUDGET.text(None)
    assert SettingKind.FLAG in {spec.kind for spec in SETTINGS}


def test_load_missing_file(tmp_path: Path) -> None:
    loaded = load_preferences(tmp_path / "нет.json")
    assert not loaded.settings
    assert not loaded.warnings


def test_load_reports_problems_and_keeps_good_settings(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "gigacrew.json",
        {
            "version": 1,
            "model": {"temperature": 0.5, "creativity": 3},
            "context": {"compactThreshold": 200},
            "colors": {"x": 1},
            "permissions": {"defaultMode": "plan"},
        },
    )
    loaded = load_preferences(path)
    assert loaded.settings == {"model.temperature": 0.5, "permissions.defaultMode": "plan"}
    joined = "\n".join(loaded.warnings)
    assert "model.creativity" in joined
    assert "context.compactThreshold" in joined
    assert "colors" in joined


@pytest.mark.parametrize("text", ["{не json", "[1, 2]"])
def test_load_unreadable_file(tmp_path: Path, text: str) -> None:
    path = tmp_path / "gigacrew.json"
    path.write_text(text, encoding="utf-8")
    loaded = load_preferences(path)
    assert not loaded.settings
    assert loaded.warnings


def test_store_keeps_only_differences(tmp_path: Path) -> None:
    path = _write(tmp_path / "gigacrew.json", {"model": {"temperature": 0.5}, "extra": True})
    store_setting(COMPACT, 60, path)
    store_setting(TEMPERATURE, 0.2, path)  # значение по умолчанию — убрать из файла
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored == {"context": {"compactThreshold": 60}, "extra": True, "version": 1}
    store_setting(COMPACT, 80, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"extra": True, "version": 1}


def test_store_creates_file_and_refuses_broken_one(tmp_path: Path) -> None:
    path = tmp_path / "новый" / "gigacrew.json"
    store_setting(SAFE, setting=False, path=path)
    assert json.loads(path.read_text(encoding="utf-8"))["permissions"] == {"safeCommands": False}
    path.write_text("{сломан", encoding="utf-8")
    with pytest.raises(ValueError, match="не удалось прочитать"):
        store_setting(SAFE, setting=True, path=path)
    assert path.read_text(encoding="utf-8") == "{сломан"  # не перезаписан
    assert not list(path.parent.glob("*.tmp"))


def test_config_reads_preferences(tmp_path: Path, user_home: Path) -> None:
    _write(
        preferences_file(),
        {
            "model": {"default": "Big", "temperature": 0.9},
            "permissions": {"defaultMode": "plan", "reviewerModel": "Small"},
            "context": {"compactThreshold": 50},
            "agent": {"maxSteps": 70},
            "ui": {"stream": False, "color": False},
        },
    )
    config = _load(tmp_path)
    assert config.model == "Big"
    assert config.temperature == pytest.approx(0.9)
    assert config.mode is PermissionMode.PLAN
    assert config.reviewer_model == "Small"
    assert config.compact_threshold == pytest.approx(0.5)
    assert config.max_steps == 70
    assert not config.stream
    assert not config.color
    assert not config.preference_warnings
    assert preferences_file().is_relative_to(user_home)


def test_env_and_dotenv_override_preferences(tmp_path: Path) -> None:
    _write(preferences_file(), {"model": {"temperature": 0.9}, "network": {"timeout": 30}})
    user_env = preferences_file().with_name(".env")
    user_env.write_text("GIGACREW_TIMEOUT=45\n", encoding="utf-8")
    config = _load(tmp_path, {"GIGACREW_TEMPERATURE": "0.1"})
    assert config.temperature == pytest.approx(0.1)
    assert config.timeout == 45


def test_flags_override_preferences(tmp_path: Path) -> None:
    _write(
        preferences_file(),
        {"model": {"default": "Big"}, "permissions": {"defaultMode": "plan"}},
    )
    config = Config.load(
        project_root=tmp_path,
        cwd=tmp_path,
        environ={},
        model="Flag",
        mode=PermissionMode.AUTO,
        stream=False,
    )
    assert config.model == "Flag"
    assert config.mode is PermissionMode.AUTO
    assert not config.stream


def test_bad_preference_warns_instead_of_failing(tmp_path: Path) -> None:
    _write(preferences_file(), {"subagents": {"background": 99}})
    config = _load(tmp_path)
    assert config.background_subagents == 2
    assert "subagents.background" in config.preference_warnings[0]


def test_bad_env_still_fails(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="GIGACREW_MAX_STEPS: значение должно быть не меньше 1"):
        _load(tmp_path, {"GIGACREW_MAX_STEPS": "0"})


def test_with_setting_converts(tmp_path: Path) -> None:
    config = Config(project_root=tmp_path)
    assert config.with_setting(COMPACT, 40).compact_threshold == pytest.approx(0.4)
    assert config.with_setting(MODE, "plan").mode is PermissionMode.PLAN
    assert not config.with_setting(_BY_KEY["model.default"], None).model


def test_env_sources(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("GIGACREW_TY=0\nGIGACREW_TIMEOUT=\n", encoding="utf-8")
    user_env = preferences_file().with_name(".env")
    user_env.parent.mkdir(parents=True, exist_ok=True)
    user_env.write_text("GIGACREW_TY=1\nGIGACREW_MODE=plan\n", encoding="utf-8")
    sources = env_sources(tmp_path, cwd=tmp_path, environ={"GIGACREW_MODE": "auto"})
    assert sources["GIGACREW_TY"] == "~/.gigacode/.env"
    assert sources["GIGACREW_MODE"] == "переменная окружения"
    assert "GIGACREW_TIMEOUT" not in sources  # пустое значение не в счёт
