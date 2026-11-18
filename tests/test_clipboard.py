"""Команда /copy и буфер обмена: исходный Markdown ответа, а не его отрисовка."""

from __future__ import annotations

import base64
import io
import shlex
import shutil
import sys
from pathlib import Path
from typing import override

import pytest
from fakes import ScriptedProvider

from gigacrew.agent.loop import Agent
from gigacrew.cli.clipboard import OSC52, clipboard_commands, copy_text
from gigacrew.cli.commands import CommandContext, default_commands
from gigacrew.config import Config
from gigacrew.llm.types import FunctionCall, Message
from gigacrew.tools.base import build_default_registry
from gigacrew.ui.console import Console

ANSWER = "## План\n\n- **шаг** `один`\n- шаг два\n\n```python\nprint(1)\n```"


class _Tty(io.StringIO):
    """Поток, который выдаёт себя за терминал (для OSC 52)."""

    @override
    def isatty(self) -> bool:
        return True


def _fake_tool(bin_dir: Path, name: str, *, code: int = 0) -> Path:
    """Утилита буфера обмена: пишет stdin в файл ``<name>.out`` и выходит с ``code``."""
    out = bin_dir / f"{name}.out"
    script = bin_dir / name
    cat = shutil.which("cat")  # PATH в тестах — только каталог поддельных утилит
    command = f"{shlex.quote(str(cat))} > {shlex.quote(str(out))}"
    script.write_text(f"#!/bin/sh\n{command}\nexit {code}\n", encoding="utf-8")
    script.chmod(0o755)
    return out


def test_clipboard_commands_follow_platform_and_display() -> None:
    assert clipboard_commands({}, "darwin") == [["pbcopy"]]
    assert clipboard_commands({}, "linux") == []
    wayland = clipboard_commands({"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}, "linux")
    assert [command[0] for command in wayland] == ["wl-copy", "xclip", "xsel"]
    assert "text/plain;charset=utf-8" in wayland[0]


def test_copy_text_pipes_to_tool_and_falls_back(tmp_path: Path) -> None:
    broken = _fake_tool(tmp_path, "xclip", code=1)
    working = _fake_tool(tmp_path, "xsel")
    terminal = _Tty()
    env = {"PATH": str(tmp_path), "DISPLAY": ":0"}
    assert copy_text(ANSWER, terminal=terminal, environ=env, platform="linux") == "xsel"
    assert broken.read_text(encoding="utf-8") == ANSWER  # xclip пробовали, он упал
    assert working.read_text(encoding="utf-8") == ANSWER
    assert not terminal.getvalue()  # утилита справилась — OSC 52 не нужен


def test_copy_text_uses_osc52_without_tools(tmp_path: Path) -> None:
    terminal = _Tty()
    env = {"PATH": str(tmp_path), "DISPLAY": ":0"}  # дисплей есть, утилит нет
    assert copy_text(ANSWER, terminal=terminal, environ=env, platform="linux") == OSC52
    sequence = terminal.getvalue()
    assert sequence.startswith("\x1b]52;c;")
    assert sequence.endswith("\x07")
    assert base64.b64decode(sequence[7:-1]).decode() == ANSWER


def test_copy_text_fails_without_tools_and_terminal(tmp_path: Path) -> None:
    pipe = io.StringIO()
    assert copy_text(ANSWER, terminal=pipe, environ={"PATH": str(tmp_path)}) is None
    assert not pipe.getvalue()


@pytest.fixture
def copy_ctx(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[CommandContext, Path]:
    """Контекст команд и файл, куда поддельный xclip кладёт скопированное."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    out = _fake_tool(bin_dir, "xclip")
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    cfg = Config(access_key="x", project_root=tmp_path, stream=False, auto_approve=True)
    ui = Console(no_color=True)
    agent = Agent(ScriptedProvider(), build_default_registry(), cfg, ui)
    return CommandContext(agent=agent, ui=ui, commands=default_commands()), out


def test_copy_puts_raw_markdown_of_answers(
    copy_ctx: tuple[CommandContext, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    ctx, out = copy_ctx
    conversation = ctx.agent.conversation
    conversation.add_user("сделай план")
    conversation.add_assistant(Message(role="assistant", content="Первый **ответ**"))
    conversation.add_user("подробнее")
    call = FunctionCall(name="read_file", arguments={"path": "a.py"})
    conversation.add_assistant(Message(role="assistant", function_calls=[call]))
    conversation.add_function_result("read_file", "x = 1")
    conversation.add_assistant(Message(role="assistant", content="\a"))  # пусто после очистки
    conversation.add_assistant(Message(role="assistant", content=f"\n{ANSWER}\n"))

    assert ctx.commands.dispatch("/copy", ctx) is True
    assert out.read_text(encoding="utf-8") == ANSWER  # разметка как есть, без отрисовки
    assert "ответ скопирован (xclip): 8 строк" in capsys.readouterr().out

    ctx.commands.dispatch("/copy 2", ctx)  # вызов инструмента без текста — не ответ
    assert out.read_text(encoding="utf-8") == "Первый **ответ**"
    assert "1 строка" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("arg", "message"),
    [
        ("", "ответов модели ещё нет"),
        ("0", "использование: /copy [N]"),
        ("два", "использование: /copy [N]"),
        ("1" * 5000, "использование: /copy [N]"),  # не доходит до int()
    ],
)
def test_copy_reports_bad_requests(
    copy_ctx: tuple[CommandContext, Path],
    capsys: pytest.CaptureFixture[str],
    arg: str,
    message: str,
) -> None:
    ctx, out = copy_ctx
    assert ctx.commands.dispatch(f"/copy {arg}", ctx) is True
    assert message in capsys.readouterr().out
    assert not out.exists()


def test_copy_beyond_answers(
    copy_ctx: tuple[CommandContext, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    ctx, out = copy_ctx
    ctx.agent.conversation.add_user("привет")
    ctx.agent.conversation.add_assistant(Message(role="assistant", content="здравствуйте"))
    ctx.commands.dispatch("/copy 3", ctx)
    assert "в чате 1 ответ модели" in capsys.readouterr().out
    assert not out.exists()


def test_copy_keeps_indented_code(
    copy_ctx: tuple[CommandContext, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    ctx, out = copy_ctx
    ctx.agent.conversation.add_user("код")
    content = "\n\n    print(1)\n    print(2)\n\n"
    ctx.agent.conversation.add_assistant(Message(role="assistant", content=content))
    ctx.commands.dispatch("/copy", ctx)
    assert out.read_text(encoding="utf-8") == "    print(1)\n    print(2)"
    assert "2 строки" in capsys.readouterr().out
