"""Настройки хуков (.gigacode/settings*.json, ~/.gigacode/settings.json) и доверие к ним."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gigacrew.project import settings as hook_settings
from gigacrew.project.settings import (
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


def _write(path: Path, payload: str | dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")


def _hooks(event: str, command: str, matcher: str | None = None, **extra: object) -> dict[str, Any]:
    group: dict[str, Any] = {"hooks": [{"type": "command", "command": command, **extra}]}
    if matcher is not None:
        group["matcher"] = matcher
    return {"hooks": {event: [group]}}


def test_no_files_no_hooks(tmp_path: Path) -> None:
    settings, warnings = load_settings(Workspace(tmp_path))
    assert not settings.groups
    assert not warnings


def test_sources_merge_in_order(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    _write(user_settings_file(), _hooks("Stop", "echo user"))
    _write(ws.settings_file, _hooks("Stop", "echo project"))
    _write(ws.local_settings_file, _hooks("Stop", "echo local"))
    settings, warnings = load_settings(ws)
    assert not warnings
    assert [group.source for group in settings.groups] == ["user", "project", "local"]
    assert [hook.command for hook in settings.commands(HookEvent.STOP)] == [
        "echo user",
        "echo project",
        "echo local",
    ]


def test_same_command_runs_once(tmp_path: Path) -> None:
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
def test_matcher_fullmatch(
    tmp_path: Path,
    matcher: str | None,
    target: str,
    *,
    expected: bool,
) -> None:
    ws = Workspace(tmp_path)
    _write(ws.settings_file, _hooks("PreToolUse", "echo", matcher))
    settings, _ = load_settings(ws)
    assert settings.has(HookEvent.PRE_TOOL_USE, target) is expected


def test_matcher_ignored_for_stop_and_prompt() -> None:
    group = HookGroup(HookEvent.STOP, "nothing-matches", (HookCommand("echo"),))
    assert group.matches("")


def test_timeout_default_and_clamp(tmp_path: Path) -> None:
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
                            {"type": "command", "command": "h", "timeout": 0},
                            {"type": "command", "command": "i", "timeout": -5},
                        ],
                    },
                ],
            },
        },
    )
    settings, warnings = load_settings(ws)
    timeouts = [hook.timeout for hook in settings.commands(HookEvent.STOP)]
    default_timeout = hook_settings.DEFAULT_HOOK_TIMEOUT
    min_timeout = hook_settings.MIN_HOOK_TIMEOUT
    max_timeout = hook_settings.MAX_HOOK_TIMEOUT
    assert timeouts == [
        default_timeout,  # a
        5,  # b
        max_timeout,  # c
        default_timeout,  # d
        default_timeout,  # e
        default_timeout,  # f
        max_timeout,  # g
        min_timeout,  # h
        min_timeout,  # i
    ]
    # Каждое предупреждение — о своей команде (c…i) и с подставленным значением.
    assert len(warnings) == 7
    for index, warning in enumerate(warnings, start=2):
        assert f"hooks[{index}]" in warning
    assert f"взято {min_timeout}" in warnings[-1]
    assert "timeout -5" in warnings[-1]


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
def test_bad_entries_warn_and_skip(
    tmp_path: Path,
    content: str | dict[str, Any],
    fragment: str,
) -> None:
    ws = Workspace(tmp_path)
    _write(ws.settings_file, content)
    # Хорошая запись в другом файле не теряется из-за ошибки в этом.
    _write(ws.local_settings_file, _hooks("Stop", "echo ok"))
    settings, warnings = load_settings(ws)
    assert len(warnings) == 1
    assert fragment in warnings[0]
    assert str(ws.settings_file) in warnings[0]
    assert [hook.command for hook in settings.commands(HookEvent.STOP)] == ["echo ok"]


@pytest.mark.parametrize("command", ["echo hi\u0000", "echo \ud800"], ids=["nul", "surrogate"])
def test_command_that_cannot_run_is_skipped(tmp_path: Path, command: str) -> None:
    ws = Workspace(tmp_path)
    raw_json = json.dumps(_hooks("Stop", command))  # \u0000 и \ud800 — как в файле
    _write(ws.settings_file, raw_json)
    _write(ws.local_settings_file, _hooks("Stop", "echo ok"))
    settings, warnings = load_settings(ws)
    assert [hook.command for hook in settings.commands(HookEvent.STOP)] == ["echo ok"]
    assert len(warnings) == 1
    assert "hooks[0]" in warnings[0]
    assert "command" in warnings[0]


def test_deeply_nested_settings_file_is_skipped(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    _write(ws.settings_file, "[" * 200_000 + "]" * 200_000)
    settings, warnings = load_settings(ws)
    assert settings == HookSettings()
    assert len(warnings) == 1
    assert "не удалось прочитать" in warnings[0]


def test_disable_all_hooks(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    _write(ws.settings_file, _hooks("Stop", "echo"))
    _write(user_settings_file(), {"disableAllHooks": True})
    settings, _ = load_settings(ws)
    assert settings.disabled
    assert not settings.has(HookEvent.STOP)


@pytest.mark.parametrize("file_kind", ["settings", "local"])
def test_project_disable_all_hooks_turns_off_only_project_hooks(
    tmp_path: Path,
    file_kind: str,
) -> None:
    # Файл проекта приходит с репозиторием: выключить хуки пользователя он не может.
    ws = Workspace(tmp_path)
    _write(user_settings_file(), _hooks("PreToolUse", "guard", "run_shell"))
    project = ws.settings_file if file_kind == "settings" else ws.local_settings_file
    _write(project, {"disableAllHooks": True, **_hooks("Stop", "echo project")})
    other = ws.settings_file if file_kind == "local" else ws.local_settings_file
    _write(other, _hooks("Stop", "x"))
    settings, warnings = load_settings(ws)
    assert not warnings
    assert not settings.disabled
    assert settings.project_disabled
    guards = settings.commands(HookEvent.PRE_TOOL_USE, "run_shell")
    assert [hook.command for hook in guards] == ["guard"]
    assert not settings.has(HookEvent.STOP)
    assert not settings.project_groups


@pytest.mark.parametrize("flag", ["true", 1, "yes"])
def test_disable_all_hooks_must_be_boolean(tmp_path: Path, flag: str | int) -> None:
    ws = Workspace(tmp_path)
    _write(user_settings_file(), {"disableAllHooks": flag, **_hooks("Stop", "echo")})
    settings, warnings = load_settings(ws)
    assert not settings.disabled
    assert settings.has(HookEvent.STOP)
    assert len(warnings) == 1
    assert "disableAllHooks" in warnings[0]


def test_untrusted_project_hooks_are_skipped(tmp_path: Path) -> None:
    ws = Workspace(tmp_path, untrusted=True)
    _write(user_settings_file(), _hooks("Stop", "echo user"))
    _write(ws.settings_file, _hooks("Stop", "echo project"))
    _write(ws.local_settings_file, _hooks("Stop", "echo local"))
    settings, _ = load_settings(ws)
    assert [hook.command for hook in settings.commands(HookEvent.STOP)] == ["echo user"]
    assert not settings.project_groups
    # Файлы проекта даже не читаются: битый файл не даёт предупреждения.
    ws.settings_file.write_text("{не JSON", encoding="utf-8")
    assert not load_settings(ws)[1]


def test_unhashable_disable_all_hooks_is_a_warning(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    _write(ws.settings_file, {"disableAllHooks": [], **_hooks("Stop", "echo")})
    _, warnings = load_settings(ws)
    assert len(warnings) == 1
    assert "disableAllHooks" in warnings[0]


def test_without_project_keeps_user_hooks(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    _write(user_settings_file(), _hooks("Stop", "echo user"))
    _write(ws.settings_file, _hooks("Stop", "echo project"))
    settings, _ = load_settings(ws)
    user_only = settings.without("project", "local")
    assert [hook.command for hook in user_only.commands(HookEvent.STOP)] == ["echo user"]


# --------------------------------------------------------------------------- #
# Доверие


def test_fingerprint_only_for_project_hooks(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    _write(user_settings_file(), _hooks("Stop", "echo user"))
    settings, _ = load_settings(ws)
    assert not project_fingerprint(settings)
    _write(ws.local_settings_file, _hooks("Stop", "echo local"))
    first = project_fingerprint(load_settings(ws)[0])
    assert len(first) == 64
    _write(ws.local_settings_file, _hooks("Stop", "echo changed"))
    assert project_fingerprint(load_settings(ws)[0]) != first


@pytest.mark.parametrize(
    ("file_kind", "changed"),
    [
        ("settings", _hooks("PreToolUse", "echo", ".*")),  # matcher шире
        ("settings", _hooks("PreRequest", "echo", "never")),  # другое событие
        ("settings", _hooks("PreToolUse", "echo", "never", timeout=5)),  # таймаут
        ("local", _hooks("PreToolUse", "echo", "never")),  # другой файл
    ],
)
def test_fingerprint_covers_matcher_event_timeout_and_file(
    tmp_path: Path,
    file_kind: str,
    changed: dict[str, Any],
) -> None:
    ws = Workspace(tmp_path)
    _write(ws.settings_file, _hooks("PreToolUse", "echo", "never"))
    before = project_fingerprint(load_settings(ws)[0])
    if file_kind == "local":
        ws.settings_file.unlink()
    _write(ws.settings_file if file_kind == "settings" else ws.local_settings_file, changed)
    assert project_fingerprint(load_settings(ws)[0]) not in {"", before}


def test_fingerprint_ignores_user_hooks(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    _write(ws.settings_file, _hooks("Stop", "echo project"))
    before = project_fingerprint(load_settings(ws)[0])
    _write(user_settings_file(), _hooks("Stop", "echo user"))
    assert project_fingerprint(load_settings(ws)[0]) == before


def test_user_fingerprint_ignores_project_hooks(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    assert not user_fingerprint(load_settings(ws)[0])
    _write(user_settings_file(), _hooks("Stop", "echo user"))
    first = user_fingerprint(load_settings(ws)[0])
    assert len(first) == 64
    _write(ws.settings_file, _hooks("Stop", "echo project"))
    assert user_fingerprint(load_settings(ws)[0]) == first
    _write(user_settings_file(), _hooks("Stop", "echo planted"))
    assert user_fingerprint(load_settings(ws)[0]) != first


def test_trust_store_user_and_projects_coexist(tmp_path: Path) -> None:
    store = TrustStore(tmp_path / "trusted_hooks.json")
    store.trust(tmp_path, "proj")
    store.trust_user("usr")
    assert store.is_user_trusted("usr")
    assert not store.is_user_trusted("other")
    assert store.is_trusted(tmp_path, "proj")  # trust_user не затёр проекты
    store.trust(tmp_path, "proj2")
    assert store.is_user_trusted("usr")  # и наоборот
    assert not TrustStore(store.path).is_user_trusted("")


def test_pre_request_event_matches_request_kind(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    _write(ws.settings_file, _hooks("PreRequest", "echo", "chat|probe"))
    settings, warnings = load_settings(ws)
    assert not warnings
    assert settings.has(HookEvent.PRE_REQUEST, "chat")
    assert not settings.has(HookEvent.PRE_REQUEST, "config")


def test_trust_store_roundtrip(tmp_path: Path) -> None:
    store = TrustStore(tmp_path / "home" / "trusted_hooks.json")
    root = tmp_path / "proj"
    root.mkdir()
    assert not store.is_trusted(root, "abc")
    store.trust(root, "abc")
    assert store.is_trusted(root, "abc")
    assert not store.is_trusted(root, "other")
    assert not store.is_trusted(tmp_path, "abc")
    assert not TrustStore(store.path).is_trusted(root, "")


def test_trust_store_default_path_and_broken_file(tmp_path: Path) -> None:
    store = TrustStore()
    assert store.path == Path.home() / ".gigacode" / "trusted_hooks.json"
    _write(store.path, "{broken")
    assert not store.is_trusted(tmp_path, "abc")
    store.trust(tmp_path, "abc")  # битый файл перезаписывается
    assert store.is_trusted(tmp_path, "abc")


def test_empty_settings_has_nothing() -> None:
    assert not HookSettings().has(HookEvent.STOP)
