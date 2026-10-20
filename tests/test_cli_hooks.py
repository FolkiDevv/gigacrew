"""Хуки в CLI: подтверждение хуков проекта, хуки сессии в REPL и -p, /hooks."""

from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path

import pytest
from fakes import ScriptedProvider, text_turn

import gigacrew.cli.app as app
from gigacrew.agent.chat_store import ChatRecorder, ChatStore
from gigacrew.agent.loop import Agent
from gigacrew.cli.commands import CommandContext, default_commands
from gigacrew.cli.hooks import load_hooks
from gigacrew.cli.repl import run_repl
from gigacrew.config import Config
from gigacrew.project.settings import HookEvent, TrustStore
from gigacrew.project.workspace import Workspace, user_settings_file
from gigacrew.tools.base import build_default_registry
from gigacrew.ui.console import Console

pytestmark = pytest.mark.skipif(os.name != "posix", reason="команды хуков — через /bin/sh")

EVENTS = ("SessionStart", "SessionEnd", "UserPromptSubmit", "Stop")


def _logging_hooks(tmp_path: Path, *, block_prompt: bool = False) -> dict:
    """Настройки: каждое событие пишет «событие:источник/причина» в events.log."""
    script = tmp_path / "hook.py"
    script.write_text(
        "import json, sys\n"
        "d = json.load(sys.stdin)\n"
        "tag = d.get('source') or d.get('reason') or ''\n"
        f"open({str(tmp_path / 'events.log')!r}, 'a').write(f\"{{d['hook_event_name']}}:{{tag}}\\n\")\n"
        + (
            "if d['hook_event_name'] == 'UserPromptSubmit':\n"
            "    print('секреты нельзя', file=sys.stderr); sys.exit(2)\n"
            if block_prompt
            else ""
        ),
        encoding="utf-8",
    )
    cmd = f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"
    group = [{"hooks": [{"type": "command", "command": cmd}]}]
    return {"hooks": {event: group for event in EVENTS}}


def _reader(*items):
    queue = list(items)

    def read():
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    return read


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _log(tmp_path: Path) -> list[str]:
    path = tmp_path / "events.log"
    return path.read_text().split() if path.exists() else []


def _main(tmp_path, *extra):
    return app.main(["-C", str(tmp_path), "-p", "привет", "--no-color", "--no-stream", *extra])


# ------------------------------- -p ------------------------------- #
def test_oneshot_skips_untrusted_project_hooks(tmp_path, oneshot, capsys):
    _write(Workspace(tmp_path).settings_file, _logging_hooks(tmp_path))
    oneshot(ScriptedProvider([text_turn("ок")]))
    assert _main(tmp_path) == 0
    assert "не подтверждены" in capsys.readouterr().out
    assert _log(tmp_path) == []


def test_oneshot_trust_hooks_flag(tmp_path, oneshot):
    _write(Workspace(tmp_path).settings_file, _logging_hooks(tmp_path))
    oneshot(ScriptedProvider([text_turn("ок")]))
    assert _main(tmp_path, "--trust-hooks") == 0
    assert _log(tmp_path) == [
        "SessionStart:startup",
        "UserPromptSubmit:",
        "Stop:",
        "SessionEnd:other",
    ]
    assert not TrustStore().path.exists()  # флаг не запоминает подтверждение


def test_oneshot_user_hooks_need_no_trust(tmp_path, oneshot):
    _write(user_settings_file(), _logging_hooks(tmp_path))
    oneshot(ScriptedProvider([text_turn("ок")]))
    assert _main(tmp_path) == 0
    assert _log(tmp_path)[0] == "SessionStart:startup"


def test_oneshot_prompt_blocked(tmp_path, oneshot, capsys):
    _write(user_settings_file(), _logging_hooks(tmp_path, block_prompt=True))
    provider = ScriptedProvider([text_turn("ок")])
    oneshot(provider)
    assert _main(tmp_path) == app.EXIT_STOPPED
    assert "запрос отклонён хуком UserPromptSubmit: секреты нельзя" in capsys.readouterr().out
    assert provider.calls == 0
    assert _log(tmp_path)[-1] == "SessionEnd:other"


# ----------------------------- доверие ----------------------------- #
def test_trust_prompt_remembers_answer(tmp_path, monkeypatch, capsys):
    ws = Workspace(tmp_path)
    _write(ws.settings_file, _logging_hooks(tmp_path))
    ui = Console(no_color=True)
    asked: list[str] = []
    monkeypatch.setattr(ui, "ask", lambda q, dangerous=False: asked.append(q) or True)
    runner = load_hooks(ws, ui, interactive=True)
    assert runner.has(HookEvent.STOP) and len(asked) == 1
    assert "hook.py" in capsys.readouterr().out  # пользователь видел команду
    load_hooks(ws, ui, interactive=True)
    assert len(asked) == 1  # подтверждено — больше не спрашиваем
    _write(
        ws.settings_file, {"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "true"}]}]}}
    )
    load_hooks(ws, ui, interactive=True)
    assert len(asked) == 2  # хуки изменились — спрашиваем снова


def test_trust_prompt_declined(tmp_path, monkeypatch):
    ws = Workspace(tmp_path)
    _write(ws.settings_file, _logging_hooks(tmp_path))
    _write(
        user_settings_file(),
        {"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "true"}]}]}},
    )
    ui = Console(no_color=True)
    monkeypatch.setattr(ui, "ask", lambda q, dangerous=False: False)
    runner = load_hooks(ws, ui, interactive=True)
    assert [h.command for h in runner.settings.commands(HookEvent.STOP)] == ["true"]
    assert not runner.has(HookEvent.SESSION_START)


# ------------------------------- REPL ------------------------------- #
@pytest.fixture
def hooked_repl(tmp_path):
    _write(user_settings_file(), _logging_hooks(tmp_path))
    cfg = Config(access_key="x", project_root=tmp_path, stream=False, auto_approve=True)
    ui = Console(no_color=True)
    hooks = load_hooks(Workspace(tmp_path), ui, interactive=False)
    provider = ScriptedProvider()
    agent = Agent(provider, build_default_registry(), cfg, ui, hooks=hooks)
    chats = ChatRecorder(ChatStore(agent.workspace))
    agent.chat_id = chats.chat_id
    return agent, ui, chats, provider


def test_repl_session_hooks(tmp_path, hooked_repl):
    agent, ui, chats, provider = hooked_repl
    read = _reader("привет", "/clear", "/exit")
    assert (
        run_repl(agent, ui, default_commands(), read_input=read, chats=chats, index_on_start=False)
        == 0
    )
    assert _log(tmp_path) == [
        "SessionStart:startup",
        "UserPromptSubmit:",
        "Stop:",
        "SessionEnd:clear",
        "SessionStart:clear",
        "SessionEnd:prompt_input_exit",
    ]


def test_repl_session_end_on_ctrl_d(tmp_path, hooked_repl):
    agent, ui, chats, _ = hooked_repl
    read = _reader(EOFError())
    run_repl(agent, ui, default_commands(), read_input=read, chats=chats, index_on_start=False)
    assert _log(tmp_path)[-1] == "SessionEnd:prompt_input_exit"


def test_repl_prompt_blocked_keeps_running(tmp_path, capsys):
    _write(user_settings_file(), _logging_hooks(tmp_path, block_prompt=True))
    cfg = Config(access_key="x", project_root=tmp_path, stream=False, auto_approve=True)
    ui = Console(no_color=True)
    provider = ScriptedProvider([text_turn("ответ")])
    hooks = load_hooks(Workspace(tmp_path), ui, interactive=False)
    agent = Agent(provider, build_default_registry(), cfg, ui, hooks=hooks)
    read = _reader("пароль 123", "/exit")
    assert run_repl(agent, ui, default_commands(), read_input=read, index_on_start=False) == 0
    assert "запрос отклонён хуком UserPromptSubmit" in capsys.readouterr().out
    assert provider.calls == 0 and len(agent.conversation) == 0


def test_hooks_command_lists_and_reloads(tmp_path, capsys):
    cfg = Config(access_key="x", project_root=tmp_path, stream=False, auto_approve=True)
    ui = Console(no_color=True)
    agent = Agent(ScriptedProvider(), build_default_registry(), cfg, ui)
    commands = default_commands()
    ctx = CommandContext(agent=agent, ui=ui, commands=commands)
    commands.dispatch("/hooks", ctx)
    assert "хуков нет" in capsys.readouterr().out
    _write(
        user_settings_file(),
        {
            "hooks": {
                "PreToolUse": [
                    {
                        "matcher": "run_shell",
                        "hooks": [{"type": "command", "command": "./check.sh"}],
                    }
                ]
            }
        },
    )
    commands.dispatch("/hooks reload", ctx)
    out = capsys.readouterr().out
    assert "перечитаны" in out and "PreToolUse" in out and "[run_shell] ./check.sh" in out
    assert agent.hooks.has(HookEvent.PRE_TOOL_USE, "run_shell")
    commands.dispatch("/hooks что-то", ctx)
    assert "использование: /hooks [reload]" in capsys.readouterr().out
