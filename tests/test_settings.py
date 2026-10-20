"""Настройки хуков (.gigacode/settings*.json, ~/.gigacode/settings.json) и доверие к ним."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gigacrew.project.settings import (
    DEFAULT_HOOK_TIMEOUT,
    MAX_HOOK_TIMEOUT,
    HookCommand,
    HookEvent,
    HookGroup,
    HookSettings,
    TrustStore,
    load_settings,
    project_fingerprint,
    user_fingerprint,
)
from gigacrew.project.workspace import Workspace, user_settings_file


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")


def _hooks(event: str, command: str, matcher: str | None = None, **extra) -> dict:
    group: dict = {"hooks": [{"type": "command", "command": command, **extra}]}
    if matcher is not None:
        group["matcher"] = matcher
    return {"hooks": {event: [group]}}


def test_no_files_no_hooks(tmp_path):
    settings, warnings = load_settings(Workspace(tmp_path))
    assert settings.groups == () and warnings == []


def test_sources_merge_in_order(tmp_path):
    ws = Workspace(tmp_path)
    _write(user_settings_file(), _hooks("Stop", "echo user"))
    _write(ws.settings_file, _hooks("Stop", "echo project"))
    _write(ws.local_settings_file, _hooks("Stop", "echo local"))
    settings, warnings = load_settings(ws)
    assert warnings == []
    assert [g.source for g in settings.groups] == ["user", "project", "local"]
    assert [h.command for h in settings.commands(HookEvent.STOP)] == [
        "echo user",
        "echo project",
        "echo local",
    ]


def test_same_command_runs_once(tmp_path):
    ws = Workspace(tmp_path)
    _write(user_settings_file(), _hooks("Stop", "echo x"))
    _write(ws.settings_file, _hooks("Stop", "echo x"))
    settings, _ = load_settings(ws)
    assert len(settings.commands(HookEvent.STOP)) == 1


@pytest.mark.parametrize(
    ("matcher", "target", "expected"),
    [
        (None, "run_shell", True),
        ("", "run_shell", True),
        ("*", "run_shell", True),
        ("run_shell", "run_shell", True),
        ("run", "run_shell", False),  # целиком, а не подстрока
        ("write_file|edit_file", "edit_file", True),
        (".*_file", "read_file", True),
        ("write_file|edit_file", "read_file", False),
    ],
)
def test_matcher_fullmatch(tmp_path, matcher, target, expected):
    ws = Workspace(tmp_path)
    _write(ws.settings_file, _hooks("PreToolUse", "echo", matcher))
    settings, _ = load_settings(ws)
    assert settings.has(HookEvent.PRE_TOOL_USE, target) is expected


def test_matcher_ignored_for_stop_and_prompt():
    group = HookGroup(HookEvent.STOP, "nothing-matches", (HookCommand("echo"),))
    assert group.matches("")


def test_timeout_default_and_clamp(tmp_path):
    ws = Workspace(tmp_path)
    _write(
        ws.settings_file,
        {
            "hooks": {
                "Stop": [
                    {
                        "hooks": [
                            {"type": "command", "command": "a"},
                            {"type": "command", "command": "b", "timeout": 5},
                            {"type": "command", "command": "c", "timeout": 10_000},
                            {"type": "command", "command": "d", "timeout": "soon"},
                            {"type": "command", "command": "e", "timeout": float("inf")},
                            {"type": "command", "command": "f", "timeout": float("nan")},
                            {"type": "command", "command": "g", "timeout": 10**400},
                        ]
                    }
                ]
            }
        },
    )
    settings, warnings = load_settings(ws)
    timeouts = [h.timeout for h in settings.commands(HookEvent.STOP)]
    assert timeouts == [DEFAULT_HOOK_TIMEOUT, 5, MAX_HOOK_TIMEOUT] + [DEFAULT_HOOK_TIMEOUT] * 3 + [
        MAX_HOOK_TIMEOUT
    ]
    assert len(warnings) == 5


@pytest.mark.parametrize(
    ("content", "fragment"),
    [
        ("{not json", "не удалось прочитать"),
        ("[1, 2]", "ожидается JSON-объект"),
        ({"hooks": []}, "hooks — не объект"),
        ({"hooks": {"BeforeEverything": []}}, "неизвестное событие"),
        ({"hooks": {"Stop": {}}}, "ожидается список"),
        ({"hooks": {"Stop": ["echo"]}}, "ожидается объект"),
        (_hooks("PreToolUse", "echo", "(unclosed"), "регулярное выражение"),
        (
            {"hooks": {"Stop": [{"hooks": [{"type": "prompt", "prompt": "x"}]}]}},
            "не поддерживается",
        ),
        ({"hooks": {"Stop": [{"hooks": [{"type": "command", "command": " "}]}]}}, "command"),
        ({"hooks": {"Stop": [{"hooks": []}]}}, "непустой список"),
    ],
)
def test_bad_entries_warn_and_skip(tmp_path, content, fragment):
    ws = Workspace(tmp_path)
    _write(ws.settings_file, content)
    # Хорошая запись в другом файле не теряется из-за ошибки в этом.
    _write(ws.local_settings_file, _hooks("Stop", "echo ok"))
    settings, warnings = load_settings(ws)
    assert len(warnings) == 1 and fragment in warnings[0]
    assert str(ws.settings_file) in warnings[0]
    assert [h.command for h in settings.commands(HookEvent.STOP)] == ["echo ok"]


def test_disable_all_hooks(tmp_path):
    ws = Workspace(tmp_path)
    _write(ws.settings_file, _hooks("Stop", "echo"))
    _write(user_settings_file(), {"disableAllHooks": True})
    settings, _ = load_settings(ws)
    assert settings.disabled and not settings.has(HookEvent.STOP)


def test_without_project_keeps_user_hooks(tmp_path):
    ws = Workspace(tmp_path)
    _write(user_settings_file(), _hooks("Stop", "echo user"))
    _write(ws.settings_file, _hooks("Stop", "echo project"))
    settings, _ = load_settings(ws)
    assert [h.command for h in settings.without("project", "local").commands(HookEvent.STOP)] == [
        "echo user"
    ]


# --------------------------------------------------------------------------- #
# Доверие


def test_fingerprint_only_for_project_hooks(tmp_path):
    ws = Workspace(tmp_path)
    _write(user_settings_file(), _hooks("Stop", "echo user"))
    settings, _ = load_settings(ws)
    assert project_fingerprint(settings) == ""
    _write(ws.local_settings_file, _hooks("Stop", "echo local"))
    first = project_fingerprint(load_settings(ws)[0])
    assert len(first) == 64
    _write(ws.local_settings_file, _hooks("Stop", "echo changed"))
    assert project_fingerprint(load_settings(ws)[0]) != first


def test_fingerprint_ignores_user_hooks(tmp_path):
    ws = Workspace(tmp_path)
    _write(ws.settings_file, _hooks("Stop", "echo project"))
    before = project_fingerprint(load_settings(ws)[0])
    _write(user_settings_file(), _hooks("Stop", "echo user"))
    assert project_fingerprint(load_settings(ws)[0]) == before


def test_user_fingerprint_ignores_project_hooks(tmp_path):
    ws = Workspace(tmp_path)
    assert user_fingerprint(load_settings(ws)[0]) == ""
    _write(user_settings_file(), _hooks("Stop", "echo user"))
    first = user_fingerprint(load_settings(ws)[0])
    assert len(first) == 64
    _write(ws.settings_file, _hooks("Stop", "echo project"))
    assert user_fingerprint(load_settings(ws)[0]) == first
    _write(user_settings_file(), _hooks("Stop", "echo planted"))
    assert user_fingerprint(load_settings(ws)[0]) != first


def test_trust_store_user_and_projects_coexist(tmp_path):
    store = TrustStore(tmp_path / "trusted_hooks.json")
    store.trust(tmp_path, "proj")
    store.trust_user("usr")
    assert store.is_user_trusted("usr") and not store.is_user_trusted("other")
    assert store.is_trusted(tmp_path, "proj")  # trust_user не затёр проекты
    store.trust(tmp_path, "proj2")
    assert store.is_user_trusted("usr")  # и наоборот
    assert not TrustStore(store.path).is_user_trusted("")


def test_pre_connect_event_matches_provider(tmp_path):
    ws = Workspace(tmp_path)
    _write(ws.settings_file, _hooks("PreConnect", "echo", "gigacode"))
    settings, warnings = load_settings(ws)
    assert warnings == []
    assert settings.has(HookEvent.PRE_CONNECT, "gigacode")
    assert not settings.has(HookEvent.PRE_CONNECT, "api")


def test_trust_store_roundtrip(tmp_path):
    store = TrustStore(tmp_path / "home" / "trusted_hooks.json")
    root = tmp_path / "proj"
    root.mkdir()
    assert not store.is_trusted(root, "abc")
    store.trust(root, "abc")
    assert store.is_trusted(root, "abc")
    assert not store.is_trusted(root, "other")
    assert not store.is_trusted(tmp_path, "abc")
    assert not TrustStore(store.path).is_trusted(root, "")


def test_trust_store_default_path_and_broken_file(tmp_path):
    store = TrustStore()
    assert store.path == Path.home() / ".gigacode" / "trusted_hooks.json"
    _write(store.path, "{broken")
    assert not store.is_trusted(tmp_path, "abc")
    store.trust(tmp_path, "abc")  # битый файл перезаписывается
    assert store.is_trusted(tmp_path, "abc")


def test_empty_settings_has_nothing():
    assert not HookSettings().has(HookEvent.STOP)
