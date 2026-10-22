"""Классификатор команд оболочки по степени безопасности.

Цель — не спрашивать пользователя о командах, которые только читают проект, и сохранить
вопрос (с причиной) для всего остального. Работает локально, без сети и моделей:

1. разбор команды деревом tree-sitter-bash (:mod:`gigacrew.safety.bash`);
2. политика программ — по имени, подкоманде, флагам и путям (:mod:`gigacrew.safety.programs`),
   отказ по умолчанию: незнакомое — «не знаю»;
3. статический анализ Python-кода из ``python -c``, heredoc и скриптов
   (:mod:`gigacrew.safety.python`).

Итог — :class:`Verdict`: уровень (READ/EXEC/WRITE/DANGER) или «не знаю» с причиной,
рабочий каталог и команда без ведущих ``cd`` (ключ для памяти решений), оценки частей.

Классификатор выключен на Windows: там команды выполняет cmd.exe, а не ``/bin/sh``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from gigacrew.safety.levels import RULES_VERSION, CommandLevel, Part, Verdict, worst

__all__ = ["RULES_VERSION", "CommandLevel", "Part", "Verdict", "classify", "invisible_chars"]

# Длиннее — не разбираем (так делают и другие агенты: длинная команда — повод спросить).
MAX_COMMAND = 10_000


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
        return Verdict(
            CommandLevel.DANGER,
            f"в команде невидимые или управляющие символы ({', '.join(hidden)}): "
            "на экране она выглядит не так, как выполнится",
            cwd,
            command.strip(),
        )
    if not _is_posix():
        return Verdict(None, "классификатор работает только с /bin/sh", cwd, command.strip())
    if len(command) > MAX_COMMAND:
        return Verdict(None, "слишком длинная команда", cwd, command.strip())
    env = environ if environ is not None else os.environ
    return _classify(command, root=root, cwd=cwd, environ=env, depth=0)


# Управляющие символы (кроме табуляции и перевода строки), смена направления текста,
# символы нулевой ширины: с ними команда на экране выглядит не так, как выполнится.
_INVISIBLE = re.compile(
    "[\x00-\x08\x0b-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff]"
)


def invisible_chars(text: str) -> list[str]:
    """Невидимые символы текста в виде ``U+202E`` (без повторов, по порядку)."""
    found: list[str] = []
    for ch in _INVISIBLE.findall(text):
        code = f"U+{ord(ch):04X}"
        if code not in found:
            found.append(code)
    return found


def _is_posix() -> bool:
    """Команды выполняет /bin/sh (на Windows — cmd.exe, его синтаксис другой)."""
    return os.name == "posix"


def _classify(
    command: str, *, root: Path, cwd: str, environ: Mapping[str, str], depth: int
) -> Verdict:
    from gigacrew.safety import bash, programs

    script = bash.parse(command, root=root, cwd=cwd)
    if script is None:
        return Verdict(None, "разбор команд оболочки недоступен", cwd, command.strip())
    if "\\\n" in _outside_heredocs(command):
        script.problems.append("перенос строки через \\ (оболочки понимают по-разному)")

    def nested(text: str, env: programs.Env) -> Part:
        inner = _classify(text, root=root, cwd=env.cwd, environ=environ, depth=env.depth)
        if inner.level is CommandLevel.READ and not inner.composable:
            return Part(None, inner.reason)
        return Part(inner.level, f"вложенная команда: {inner.reason}")

    base = programs.Env(root=root, cwd=script.cwd, environ=environ, depth=depth, nested=nested)
    parts: list[Part] = []
    writes: list[Part] = []
    for cmd in script.commands:
        env = replace(base, cwd=cmd.cwd)
        part = replace(programs.classify_simple(cmd, env), text=_text_of(cmd))
        parts.append(part)
        for r in cmd.redirects:
            if r.writes and r.target is not None:
                if r.target.static and r.target.value in (
                    "/dev/null",
                    "/dev/stdout",
                    "/dev/stderr",
                ):
                    continue
                writes.append(programs._check_write(env, [r.target], "перенаправление"))
            elif r.reads:
                bad = programs._check_read(env, [r.target], "чтение через <")
                if bad is not None:
                    writes.append(bad)
    # Присваивания вне команд (X=1; export X=1) действуют на всё дальше.
    for name, value in script.assigns:
        bad = programs.check_env(name, value, reading=True)
        if bad is not None:
            parts.append(
                bad
                if bad.level is not None
                else Part(None, f"присваивание {name}: меняет окружение следующих команд")
            )
    problems = [Part(None, p) for p in script.problems]
    if script.compound:
        # Условия и циклы разобраны, но «только чтение» для них не выносим: решает человек.
        problems.append(Part(None, "условия или циклы оболочки"))
    everything = [*parts, *writes, *problems]
    if not everything:
        return Verdict(
            CommandLevel.READ, "пустая команда", script.cwd, script.rest, composable=True
        )
    level = worst(*(p.level for p in everything))
    reason = _reason(level, everything)
    composable = not writes and not problems and bool(parts)
    return Verdict(level, reason, script.cwd, script.rest, tuple(parts), composable)


def _text_of(cmd, depth: int = 0) -> str:
    """Простая команда как написана — ключ точного разрешения. Вместе с тем, что она
    получит на stdin (тело heredoc, предыдущее звено конвейера): ``python3 -`` с другим
    кодом — другая команда."""
    words = [f"{name}={value.raw}" for name, value in cmd.assigns]
    words += [w.raw for w in cmd.argv]
    for r in cmd.redirects:
        if r.body is not None:
            words.append(f"<<{r.body!r}")
        elif r.reads and r.target is not None:
            words.append(f"<{r.target.raw}")
    if cmd.stdin is not None and depth < 8:
        words.append(f"<|{_text_of(cmd.stdin, depth + 1)}")
    return " ".join(words)


def _reason(level: CommandLevel | None, parts: list[Part]) -> str:
    for p in parts:
        if p.level == level and p.reason:
            return p.reason
    return parts[0].reason if parts else ""


def _outside_heredocs(command: str) -> str:
    """Текст команды без тел heredoc (в них ``\\`` и перевод строки — обычные символы)."""
    from gigacrew.project import treesitter

    tree = treesitter.parse_tree("shell", command)
    if tree is None:
        return command
    data = command.encode("utf-8", errors="surrogateescape")
    spans: list[tuple[int, int]] = []
    stack = [tree]
    while stack:
        n = stack.pop()
        if n.type in ("heredoc_body", "raw_string"):
            spans.append((n.start_byte, n.end_byte))
            continue
        stack.extend(n.children)
    out = bytearray()
    pos = 0
    for start, end in sorted(spans):
        out += data[pos:start]
        pos = max(pos, end)
    out += data[pos:]
    return out.decode("utf-8", errors="surrogateescape")
