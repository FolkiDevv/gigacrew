"""Хуки в CLI: подтверждение хуков, PreConnect, хуки сессии в REPL и -p, /hooks."""

from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path

import pytest
from fakes import CatalogueProvider, ScriptedProvider, text_turn

import gigacrew.cli.app as app
from gigacrew.agent.chat_store import ChatRecorder, ChatStore
from gigacrew.agent.loop import Agent
from gigacrew.cli.commands import CommandContext, default_commands
from gigacrew.cli.hooks import load_hooks
from gigacrew.cli.repl import run_repl
from gigacrew.config import Config
from gigacrew.project.settings import HookEvent, TrustStore, load_settings, user_fingerprint
from gigacrew.project.workspace import Workspace, user_settings_file
from gigacrew.tools.base import build_default_registry
from gigacrew.ui.console import Console

pytestmark = pytest.mark.skipif(os.name != "posix", reason="команды хуков — через /bin/sh")

EVENTS = ("PreConnect", "SessionStart", "SessionEnd", "UserPromptSubmit", "Stop")


def _logging_hooks(
    tmp_path: Path, *, block_prompt: bool = False, block_connect: bool = False, matcher: str = ""
) -> dict:
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
        )
        + (
            "if d['hook_event_name'] == 'PreConnect':\n"
            "    print('нет токена', file=sys.stderr); sys.exit(2)\n"
            if block_connect
            else ""
        ),
        encoding="utf-8",
    )
    cmd = f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"
    group = [{"hooks": [{"type": "command", "command": cmd}]}]
    hooks = {event: group for event in EVENTS}
    hooks["PreConnect"] = [{"matcher": matcher, "hooks": [{"type": "command", "command": cmd}]}]
    return {"hooks": hooks}


def _trust_user_hooks(tmp_path: Path) -> None:
    """Подтвердить текущие хуки ~/.gigacode/settings.json (как ответ «да» раньше)."""
    settings, _ = load_settings(Workspace(tmp_path))
    TrustStore().trust_user(user_fingerprint(settings))


def _text(out: str) -> str:
    """Вывод в одну строку: Rich переносит длинные строки по ширине терминала."""
    return " ".join(out.split())


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
    assert "хуки проекта" in _text(capsys.readouterr().out)
    assert _log(tmp_path) == []


def test_oneshot_trust_hooks_flag(tmp_path, oneshot):
    _write(Workspace(tmp_path).settings_file, _logging_hooks(tmp_path))
    oneshot(ScriptedProvider([text_turn("ок")]))
    assert _main(tmp_path, "--trust-hooks") == 0
    assert _log(tmp_path) == [
        "PreConnect:",
        "SessionStart:startup",
        "UserPromptSubmit:",
        "Stop:",
        "SessionEnd:other",
    ]
    assert not TrustStore().path.exists()  # флаг не запоминает подтверждение


def test_oneshot_skips_untrusted_user_hooks(tmp_path, oneshot, capsys):
    _write(user_settings_file(), _logging_hooks(tmp_path))
    oneshot(ScriptedProvider([text_turn("ок")]))
    assert _main(tmp_path) == 0
    out = _text(capsys.readouterr().out)
    assert "хуки пользователя (~/.gigacode/settings.json) не подтверждены" in out
    assert _log(tmp_path) == []


def test_oneshot_trusted_user_hooks_run(tmp_path, oneshot):
    _write(user_settings_file(), _logging_hooks(tmp_path))
    _trust_user_hooks(tmp_path)
    oneshot(ScriptedProvider([text_turn("ок")]))
    assert _main(tmp_path) == 0
    assert _log(tmp_path)[:2] == ["PreConnect:", "SessionStart:startup"]


def test_oneshot_prompt_blocked(tmp_path, oneshot, capsys):
    _write(user_settings_file(), _logging_hooks(tmp_path, block_prompt=True))
    _trust_user_hooks(tmp_path)
    provider = ScriptedProvider([text_turn("ок")])
    oneshot(provider)
    assert _main(tmp_path) == app.EXIT_STOPPED
    assert "запрос отклонён хуком UserPromptSubmit: секреты нельзя" in capsys.readouterr().out
    assert provider.calls == 0
    assert _log(tmp_path)[-1] == "SessionEnd:other"


# ---------------------------- PreConnect ---------------------------- #
class ConnectProvider(CatalogueProvider):
    """Провайдер, который при подключении запоминает, что уже записано в лог хуков."""

    def __init__(self, log: Path):
        super().__init__([text_turn("ок")], models=[("M-default", 64_000)])
        self.log = log
        self.seen_at_connect: list[str] | None = None

    def connect(self) -> None:
        self.seen_at_connect = self.log.read_text().split() if self.log.exists() else []
        super().connect()


def test_pre_connect_runs_before_connect(tmp_path, oneshot):
    _write(Workspace(tmp_path).settings_file, _logging_hooks(tmp_path))
    provider = ConnectProvider(tmp_path / "events.log")
    oneshot(provider)
    assert _main(tmp_path, "--trust-hooks") == 0
    assert provider.seen_at_connect == ["PreConnect:"]


def test_pre_connect_block_stops_startup(tmp_path, oneshot, capsys):
    _write(Workspace(tmp_path).settings_file, _logging_hooks(tmp_path, block_connect=True))
    provider = ConnectProvider(tmp_path / "events.log")
    oneshot(provider)
    assert _main(tmp_path, "--trust-hooks") == app.EXIT_ERROR
    assert "запуск остановлен хуком PreConnect: нет токена" in _text(capsys.readouterr().out)
    assert provider.connects == 0
    assert _log(tmp_path) == ["PreConnect:"]  # ни SessionStart, ни хода


def test_pre_connect_matcher_is_provider(tmp_path, oneshot):
    _write(Workspace(tmp_path).settings_file, _logging_hooks(tmp_path, matcher="gigacode"))
    oneshot(ScriptedProvider([text_turn("ок")]))  # провайдер по умолчанию — api
    assert _main(tmp_path, "--trust-hooks") == 0
    assert "PreConnect:" not in _log(tmp_path)


def test_pre_connect_on_list_models(tmp_path, oneshot):
    _write(Workspace(tmp_path).settings_file, _logging_hooks(tmp_path))
    oneshot(ConnectProvider(tmp_path / "events.log"))
    code = app.main(["-C", str(tmp_path), "--list-models", "--no-color", "--trust-hooks"])
    assert code == 0
    assert _log(tmp_path) == ["PreConnect:"]


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


def test_user_hooks_confirmed_once_for_all_projects(tmp_path, monkeypatch, capsys):
    _write(user_settings_file(), _logging_hooks(tmp_path))
    ui = Console(no_color=True)
    asked: list[str] = []
    monkeypatch.setattr(ui, "ask", lambda q, dangerous=False: asked.append(q) or True)
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    assert load_hooks(Workspace(first), ui, interactive=True).has(HookEvent.STOP)
    assert asked == ["Разрешить хуки пользователя?"]
    assert "hook.py" in capsys.readouterr().out
    assert load_hooks(Workspace(second), ui, interactive=True).has(HookEvent.STOP)
    assert len(asked) == 1  # подтверждение пользовательских хуков — на все проекты
    _write(
        user_settings_file(),
        {"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "echo planted"}]}]}},
    )
    load_hooks(Workspace(second), ui, interactive=True)
    assert len(asked) == 2  # файл пользователя изменился — спрашиваем снова


def test_user_and_project_hooks_one_question(tmp_path, monkeypatch):
    ws = Workspace(tmp_path)
    _write(ws.settings_file, _logging_hooks(tmp_path))
    _write(
        user_settings_file(),
        {"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "true"}]}]}},
    )
    ui = Console(no_color=True)
    asked: list[str] = []
    monkeypatch.setattr(ui, "ask", lambda q, dangerous=False: asked.append(q) or True)
    load_hooks(ws, ui, interactive=True)
    assert asked == ["Разрешить хуки пользователя и проекта?"]
    load_hooks(ws, ui, interactive=True)
    assert len(asked) == 1


def test_trust_prompt_declined(tmp_path, monkeypatch):
    ws = Workspace(tmp_path)
    _write(ws.settings_file, _logging_hooks(tmp_path))
    _write(
        user_settings_file(),
        {"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "true"}]}]}},
    )
    _trust_user_hooks(tmp_path)  # хуки пользователя подтверждены раньше
    ui = Console(no_color=True)
    monkeypatch.setattr(ui, "ask", lambda q, dangerous=False: False)
    runner = load_hooks(ws, ui, interactive=True)
    assert [h.command for h in runner.settings.commands(HookEvent.STOP)] == ["true"]
    assert not runner.has(HookEvent.SESSION_START)


# ------------------------------- REPL ------------------------------- #
@pytest.fixture
def hooked_repl(tmp_path):
    _write(user_settings_file(), _logging_hooks(tmp_path))
    _trust_user_hooks(tmp_path)
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
    _trust_user_hooks(tmp_path)
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
    ctx = CommandContext(agent=agent, ui=ui, commands=commands, trust_hooks=True)
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
