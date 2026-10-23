"""Классификатор команд оболочки: :func:`classify` (устройство — в :mod:`gigacrew.safety`)."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from pathlib import Path

from gigacrew.project import treesitter
from gigacrew.safety import bash, programs
from gigacrew.safety.levels import CommandLevel, Part, Verdict, worst

__all__ = ["MAX_COMMAND", "classify", "invisible_chars"]

# Длиннее — не разбираем (так делают и другие агенты: длинная команда — повод спросить).
MAX_COMMAND = 10_000

# Сколько звеньев конвейера (stdin) входит в ключ точного разрешения.
_MAX_STDIN_DEPTH = 8
# Перенос строки через обратную косую черту: оболочки понимают его по-разному.
_LINE_CONTINUATION = "\N{REVERSE SOLIDUS}\n"
# Запись в эти устройства — не запись в файлы.
_STD_STREAMS = frozenset(("/dev/null", "/dev/stdout", "/dev/stderr"))


def classify(
    command: str,
    *,
    root: Path,
    cwd: str = ".",
    environ: Mapping[str, str] | None = None,
) -> Verdict:
    """Оценить команду, которую ``/bin/sh -c`` выполнит в ``root/cwd``.

    ``environ`` — окружение дочернего процесса (по умолчанию — текущее): из него берутся
    PATH (где окажется программа) и PYTHONPATH.
    """
    root = Path(root).resolve()
    cwd = cwd or "."
    # Невидимые символы — до всех отказов «не знаю»: длинный хвост не должен их прятать.
    hidden = invisible_chars(command)
    if hidden:
        listed = ", ".join(hidden)
        return Verdict(
            CommandLevel.DANGER,
            f"в команде невидимые или управляющие символы ({listed}): "
            "на экране она выглядит не так, как выполнится",
            cwd,
            command.strip(),
        )
    if not _is_posix():
        return Verdict(None, "классификатор работает только с /bin/sh", cwd, command.strip())
    if len(command) > MAX_COMMAND:
        return Verdict(None, "слишком длинная команда", cwd, command.strip())
    env = os.environ if environ is None else environ
    return _classify(command, root=root, cwd=cwd, environ=env, depth=0)


# Управляющие символы (кроме табуляции и перевода строки), смена направления текста,
# символы нулевой ширины: с ними команда на экране выглядит не так, как выполнится.
_INVISIBLE = re.compile(
    r"[\x00-\x08\x0b-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff]",
)


def invisible_chars(text: str) -> list[str]:
    """Невидимые символы текста в виде ``U+202E`` (без повторов, по порядку)."""
    found: list[str] = []
    for ch in _INVISIBLE.findall(text):
        hex_code = format(ord(ch), "04X")
        code = f"U+{hex_code}"
        if code not in found:
            found.append(code)
    return found


def _is_posix() -> bool:
    """Команды выполняет /bin/sh (на Windows — cmd.exe, его синтаксис другой)."""
    return os.name == "posix"


# Разбор команды и сборка вердикта — граница безопасности: шаги видны подряд, в одном месте.
def _classify(  # noqa: WPS210
    command: str,
    *,
    root: Path,
    cwd: str,
    environ: Mapping[str, str],
    depth: int,
) -> Verdict:
    script = bash.parse(command, root=root, cwd=cwd)
    if script is None:
        return Verdict(None, "разбор команд оболочки недоступен", cwd, command.strip())
    if _LINE_CONTINUATION in _outside_heredocs(command):
        script.problems.append(r"перенос строки через \ (оболочки понимают по-разному)")

    nested = partial(_nested, root=root, environ=environ)
    base = programs.Env(root=root, cwd=script.cwd, environ=environ, depth=depth, nested=nested)
    parts: list[Part] = []
    writes: list[Part] = []
    for simple in script.commands:
        env = replace(base, cwd=simple.cwd, assigned=simple.assigned)
        part = replace(programs.classify_simple(simple, env), text=_text_of(simple))
        parts.append(part)
        for redirect in simple.redirects:
            writes.extend(_redirect_parts(env, redirect))
    # Присваивания вне команд (X=1; export X=1) действуют на всё дальше.
    for name, assigned in script.assigns:
        bad = programs.check_env(name, assigned, reading=True)
        if bad is not None:
            parts.append(
                Part(None, f"присваивание {name}: меняет окружение следующих команд")
                if bad.level is None
                else bad,
            )
    problems = [Part(None, problem) for problem in script.problems]
    if script.compound:
        # Условия и циклы разобраны, но «только чтение» для них не выносим: решает человек.
        problems.append(Part(None, "условия или циклы оболочки"))
    everything = [*parts, *writes, *problems]
    if not everything:
        return Verdict(
            CommandLevel.READ,
            "пустая команда",
            script.cwd,
            script.rest,
            composable=True,
        )
    level = worst(*(each.level for each in everything))
    reason = _reason(level, everything)
    # Часть, выполняемая не в script.cwd (`(cd X && make test)`), сверялась бы с
    # разрешениями другого каталога: память решений ищет их по Verdict.cwd.
    same_dir = all(each.cwd == script.cwd for each in script.commands)
    composable = not writes and not problems and bool(parts) and same_dir
    redirects_edit = all(write.edits for write in writes if write.level is CommandLevel.WRITE)
    return Verdict(level, reason, script.cwd, script.rest, tuple(parts), composable, redirects_edit)


def _redirect_parts(env: programs.Env, redirect: bash.Redirect) -> list[Part]:
    """Оценки перенаправления: переменные в теле heredoc, запись и чтение файлов."""
    found: list[Part] = []
    if redirect.body is not None and redirect.body_vars:
        # Тело heredoc/herestring программа получит на stdin и, как правило,
        # выведет: переменные в нём — как в аргументах echo.
        body = bash.Word(
            redirect.body,
            static=False,
            variables=redirect.body_vars,
            plain=redirect.body_plain,
        )
        bad = programs.check_vars(env, [body], "текст на stdin")
        if bad is not None:
            found.append(bad)
    if redirect.writes and redirect.target is not None:
        target = redirect.target
        if target.static and target.text in _STD_STREAMS:
            return found
        found.append(programs.check_write(env, [target], "перенаправление"))
    elif redirect.reads and redirect.target is not None:
        bad = programs.check_read(env, [redirect.target], "чтение через <")
        if bad is not None:
            found.append(bad)
    return found


def _nested(
    text: str,
    env: programs.Env,
    *,
    root: Path,
    environ: Mapping[str, str],
) -> Part:
    """Оценка вложенной команды оболочки (``sh -c "…"``, тело heredoc для bash)."""
    inner = _classify(text, root=root, cwd=env.cwd, environ=environ, depth=env.depth)
    if inner.level is CommandLevel.READ and not inner.composable:
        return Part(None, inner.reason)
    return Part(inner.level, f"вложенная команда: {inner.reason}")


def _text_of(cmd: bash.Command, depth: int = 0) -> str:
    """Простая команда как написана — ключ точного разрешения.

    Вместе с тем, что она получит на stdin (тело heredoc, предыдущее звено конвейера):
    ``python3 -`` с другим кодом — другая команда.
    """
    words = [f"{name}={word.raw}" for name, word in cmd.assigns]
    words += [word.raw for word in cmd.argv]
    for redirect in cmd.redirects:
        if redirect.body is not None:
            words.append(f"<<{redirect.body!r}")
        elif redirect.reads and redirect.target is not None:
            source = redirect.target.raw
            words.append(f"<{source}")
    if cmd.stdin is not None and depth < _MAX_STDIN_DEPTH:
        words.append(f"<|{_text_of(cmd.stdin, depth + 1)}")
    return " ".join(words)


def _reason(level: CommandLevel | None, parts: list[Part]) -> str:
    for part in parts:
        if part.level == level and part.reason:
            return part.reason
    return parts[0].reason if parts else ""


def _outside_heredocs(command: str) -> str:
    r"""Текст команды без тел heredoc (в них ``\\`` и перевод строки — обычные символы)."""
    tree = treesitter.parse_tree("shell", command)
    if tree is None:
        return command
    encoded = command.encode("utf-8", errors="surrogateescape")
    spans: list[tuple[int, int]] = []
    stack = [tree]
    while stack:
        node = stack.pop()
        if node.type in {"heredoc_body", "raw_string"}:
            spans.append((node.start_byte, node.end_byte))
            continue
        stack.extend(node.children)
    out = bytearray()
    pos = 0
    for start, end in sorted(spans):
        out += encoded[pos:start]
        pos = max(pos, end)
    out += encoded[pos:]
    return out.decode("utf-8", errors="surrogateescape")
