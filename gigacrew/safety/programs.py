"""Политика программ: что делает простая команда — по имени, подкоманде, флагам и путям.

Принцип — **отказ по умолчанию**: незнакомая программа, подкоманда или флаг, меняющий
поведение, дают «не знаю» (решает пользователь). Без вопроса выполняется только то, про
что точно известно, что оно лишь читает файлы внутри проекта. Знание об опасных флагах
«читающих» программ (``find -exec``, ``rg --pre``, ``sed e``, ``awk system()``,
``tar --to-command``, ``less +!``) — здесь же.

Уровни: READ — без вопроса; EXEC — запуск кода проекта или инструмента (тесты, сборка);
WRITE — изменение файлов и окружения (в т.ч. установка пакетов — сеть и чужой код);
DANGER — необратимое, вне проекта, секреты, привилегии, публикация.
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import shutil
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from gigacrew.safety.bash import Command, Word
from gigacrew.safety.levels import CommandLevel, Part, worst
from gigacrew.safety.paths import (
    Scope,
    has_escaping_links,
    has_secrets_under,
    is_data_path,
    is_git_internal,
    scope,
)

READ, EXEC, WRITE, DANGER = (
    CommandLevel.READ,
    CommandLevel.EXEC,
    CommandLevel.WRITE,
    CommandLevel.DANGER,
)

__all__ = ["Env", "classify_simple"]


@dataclass
class Env:
    """Контекст оценки: корень, рабочий каталог команды, окружение дочернего процесса."""

    root: Path
    cwd: str = "."
    environ: Mapping[str, str] = field(default_factory=dict)
    # Переменные, которым команда раньше сама присвоила постоянное значение
    # (bash.Command.assigned): это не секреты из окружения.
    assigned: frozenset[str] = frozenset()
    depth: int = 0  # вложенность sh -c
    # Оценка вложенной команды оболочки (sh -c "…", тело heredoc для bash).
    nested: Callable[[str, Env], Part] | None = None
    program_path: str | None = None  # явный путь к программе (.venv/bin/python)

    @property
    def dir(self) -> Path:
        return self.root / self.cwd


# --------------------------------------------------------------------------- #
# Конструкторы оценок
# --------------------------------------------------------------------------- #
def _read(reason: str) -> Part:
    return Part(READ, reason)


def _unknown(reason: str) -> Part:
    return Part(None, reason)


def _danger(reason: str) -> Part:
    return Part(DANGER, reason)


def _write(reason: str, *, edits: bool = False) -> Part:
    return Part(WRITE, reason, edits=edits)


def _exec(reason: str, signature: str | None) -> Part:
    return Part(EXEC, reason, signature=signature)


def _net(what: str) -> Part:
    return Part(WRITE, f"{what}: сеть и сторонний код (установка пакетов)")


def _combine(*parts: Part | None) -> Part:
    """Худшая из оценок (None — пропускается)."""
    real = [p for p in parts if p is not None]
    if not real:
        return _read("")
    level = worst(*(p.level for p in real))
    for p in real:
        if p.level == level:
            return p
    return real[0]


# --------------------------------------------------------------------------- #
# Аргументы
# --------------------------------------------------------------------------- #
def _split(args: list[Word]) -> tuple[list[Word], list[Word]]:
    """(флаги, позиционные); после ``--`` всё — позиционные."""
    flags: list[Word] = []
    pos: list[Word] = []
    rest = False
    for a in args:
        if rest:
            pos.append(a)
        elif a.static and a.value == "--":
            rest = True
        elif a.static and a.value.startswith("-") and a.value != "-":
            flags.append(a)
        else:
            pos.append(a)
    return flags, pos


def _flagset(flags: list[Word]) -> set[str]:
    """Имена флагов: ``--name`` (без ``=значения``) и буквы сгруппированных ``-abc``."""
    out: set[str] = set()
    for f in flags:
        v = f.value
        if v.startswith("--"):
            out.add(v.split("=", 1)[0])
        else:
            out.add(v)
            letters = v[1:]
            if letters.isalpha():
                out.update("-" + ch for ch in letters)
    return out


def _has(flags: set[str], *names: str) -> bool:
    return any(n in flags for n in names)


def _options(
    args: list[Word], valued: Collection[str], *, clusters: bool = True
) -> tuple[list[Word], list[tuple[str, Word]]]:
    """Разбор флагов со значением, как в getopt: (аргументы без них, [(флаг, значение)]).

    ``valued`` — все флаги программы, которые берут значение: ``--name=V``, ``--name V``,
    ``-n V``, ``-nV``. В группе коротких (``-uo FILE``, ``-uoFILE``) буквы до первой
    со значением — флаги без значения (остаются словом ``-u``), значение — остаток слова
    или следующее слово. После ``--`` флагов нет. ``clusters=False`` — для программ со
    своим разбором (xxd: ``-ps`` — один флаг): только точные слова и ``-nV``.
    """
    kept: list[Word] = []
    found: list[tuple[str, Word]] = []
    i = 0
    while i < len(args):
        a = args[i]
        v = a.value
        nxt = args[i + 1] if i + 1 < len(args) else None
        if not a.static or not v.startswith("-") or v == "-":
            kept.append(a)
        elif v == "--":
            kept.extend(args[i:])
            break
        elif v in valued:
            if nxt is not None:
                found.append((v, nxt))
            i += 2
            continue
        elif v.startswith("--"):
            name, eq, value = v.partition("=")
            if eq and name in valued:
                found.append((name, Word(value, a.raw, a.static, a.glob)))
            else:
                kept.append(a)
        else:
            letters = range(1, len(v) if clusters else 2)
            k = next((k for k in letters if "-" + v[k] in valued), None)
            if k is None:
                kept.append(a)
            else:
                if k > 1:
                    kept.append(Word(v[:k], a.raw, a.static, a.glob))
                if k + 1 < len(v):
                    found.append(("-" + v[k], Word(v[k + 1 :], a.raw, a.static, a.glob)))
                else:
                    if nxt is not None:
                        found.append(("-" + v[k], nxt))
                    i += 2
                    continue
        i += 1
    return kept, found


def _values(args: list[Word], *names: str, valued: Collection[str] = ()) -> list[Word]:
    """Значения флагов ``names``; ``valued`` — остальные флаги программы со значением
    (чтобы в ``-ko FILE`` не принять значение ``-k`` за значение ``-o``)."""
    _, found = _options(args, {*names, *valued})
    return [w for n, w in found if n in names]


def _without_values(args: list[Word], *names: str, clusters: bool = True) -> list[Word]:
    """Аргументы без флагов из ``names`` и их значений (``-n 5``, ``-qn5`` → ``-q``)."""
    return _options(args, names, clusters=clusters)[0]


def _subcommand(args: list[Word]) -> tuple[str, list[Word]]:
    """Подкоманда — первый позиционный аргумент; (подкоманда, аргументы после неё)."""
    for i, a in enumerate(args):
        if not a.static:
            return "", args
        if a.value.startswith("-") or a.value.startswith("+"):
            continue
        return a.value, args[i + 1 :]
    return "", []


_PATHLIKE = re.compile(r"^(/|~|\.\.?(/|$))|/")


def _looks_like_path(value: str) -> bool:
    return bool(_PATHLIKE.search(value))


def _abbrev(args: list[Word], *names: str, allow: tuple[str, ...] = ()) -> Word | None:
    """Длинный флаг — сокращение одного из ``names`` (``--outp`` для ``--output``).

    GNU getopt_long и parse-options git принимают однозначные сокращения, поэтому
    сверка по точному имени пропустила бы ``sort --outp=file``. Точное имя тоже
    возвращается — вызывающий решает, что с ним делать.
    """
    flags, _ = _split(args)
    for a in flags:
        if not a.value.startswith("--"):
            continue
        opt = a.value[2:].split("=", 1)[0]
        if not opt or f"--{opt}" in allow:  # полное имя другого флага (grep --include)
            continue
        for n in names:
            if n.lstrip("-").startswith(opt):
                return a
    return None


def _abbrev_only(args: list[Word], *names: str, allow: tuple[str, ...] = ()) -> Word | None:
    """Сокращённый (не полный) вариант одного из длинных флагов ``names``."""
    hit = _abbrev(args, *names, allow=allow)
    if hit is None:
        return None
    return None if hit.value.split("=", 1)[0] in names else hit


# --------------------------------------------------------------------------- #
# Проверка путей
# --------------------------------------------------------------------------- #
# Системные файлы со сведениями о машине — читать их безопасно.
_SYSTEM_INFO = frozenset(
    {
        "/etc/os-release",
        "/usr/lib/os-release",
        "/etc/lsb-release",
        "/etc/debian_version",
        "/etc/redhat-release",
        "/etc/alpine-release",
        "/etc/hostname",
        "/etc/timezone",
        "/proc/cpuinfo",
        "/proc/meminfo",
        "/proc/version",
        "/proc/loadavg",
        "/proc/uptime",
        "/proc/stat",
        "/etc/issue",
    }
)


def _check_read(env: Env, words: list[Word], what: str, *, content: bool = True) -> Part | None:
    """Чтение файлов: None — все пути внутри проекта и не секреты.

    ``content=False`` — читаются только имена и размеры (ls, du): секретный каталог
    снаружи — «не знаю», а не опасность.
    """
    found: list[Part] = []
    for w in words:
        if w.static and not w.glob and w.value in _SYSTEM_INFO:
            continue
        s = scope(w, root=env.root, cwd=env.cwd)
        if s is Scope.SECRET_OUTSIDE:
            if not content:
                found.append(_unknown(f"{what}: каталог с секретами вне проекта ({w.raw})"))
                continue
            return _danger(f"{what}: секреты вне проекта ({w.raw})")
        if s is Scope.SECRET:
            found.append(_unknown(f"{what}: файл с секретами ({w.raw})"))
        elif s is Scope.OUTSIDE:
            found.append(_unknown(f"{what}: путь вне проекта ({w.raw})"))
        elif s is Scope.DYNAMIC:
            found.append(_unknown(f"{what}: путь известен только при выполнении ({w.raw})"))
    return found[0] if found else None


def _check_write(env: Env, words: list[Word], what: str, *, edits: bool = True) -> Part:
    """Запись в файлы: WRITE внутри проекта, DANGER — снаружи или в служебные каталоги."""
    result: list[Part] = []
    for w in words:
        s = scope(w, root=env.root, cwd=env.cwd)
        if s in (Scope.OUTSIDE, Scope.SECRET_OUTSIDE):
            return _danger(f"{what}: запись вне проекта ({w.raw})")
        if s is Scope.SINK:
            if w.value == "/dev/null" and what in ("перенаправление",):
                continue
            return _danger(f"{what}: {w.raw}")
        if s is Scope.DYNAMIC:
            result.append(_unknown(f"{what}: путь известен только при выполнении ({w.raw})"))
            continue
        target = (env.dir / w.value) if not w.value.startswith("/") else Path(w.value)
        if is_git_internal(target.resolve(), env.root) or is_git_internal(target, env.root):
            return _danger(f"{what}: служебные файлы git ({w.raw}) — оттуда запускаются программы")
        if is_data_path(target.resolve(), env.root):
            return _danger(f"{what}: служебная папка gigacrew ({w.raw})")
        if s is Scope.SECRET:
            result.append(_write(f"{what}: файл с секретами ({w.raw})"))
            continue
        result.append(_write(what, edits=edits))
    if not words:  # куда пишет — не разобрали (`tree -o` без значения): не «правка»
        return _unknown(f"{what}: файл записи не найден в аргументах")
    if not result:
        return _write(what, edits=edits)
    return _combine(*result)


def _check_recursive(
    env: Env,
    dirs: list[Word],
    what: str,
    *,
    hidden: bool = True,
    follow: bool = False,
    pattern: str | None = None,
) -> Part | None:
    """Рекурсивное чтение каталогов: секреты и ссылки наружу в дереве — «не знаю»."""
    bad = _check_read(env, dirs, what)
    if bad is not None:
        return bad
    for w in dirs or [Word(".")]:
        d = (env.dir / w.value).resolve() if not w.glob else env.dir
        if hidden and has_secrets_under(d, env.root, name_pattern=pattern):
            return _unknown(f"{what}: в каталоге есть файлы с секретами")
        if has_escaping_links(d, env.root, name_pattern=pattern) and (follow or hidden):
            return _unknown(f"{what}: в каталоге есть ссылки за пределы проекта")
    return None


def _check_exec_args(env: Env, args: list[Word], what: str) -> Part | None:
    """Аргументы инструмента, исполняющего код: пути наружу и опасные флаги — «не знаю»."""
    for a in args:
        if not a.static:
            return _unknown(f"{what}: аргумент известен только при выполнении ({a.raw})")
        value = a.value
        if value.startswith("--basetemp"):
            return _unknown(f"{what}: --basetemp очищает каталог")
        candidates = [value]
        if "=" in value:
            candidates.append(value.split("=", 1)[1])
        for c in candidates:
            if c and _looks_like_path(c) and not c.startswith(("--", "http://", "https://")):
                s = scope(Word(c), root=env.root, cwd=env.cwd)
                if s is Scope.SECRET_OUTSIDE:
                    return _danger(f"{what}: секреты вне проекта ({c})")
                if s in (Scope.OUTSIDE, Scope.DYNAMIC, Scope.SECRET):
                    return _unknown(f"{what}: путь вне проекта в аргументах ({c})")
    return None


def _exec_tool(env: Env, args: list[Word], what: str, signature: str | None) -> Part:
    bad = _check_exec_args(env, args, what)
    if bad is not None:
        return bad
    if signature is not None and not _signature_args_ok(signature, args, env):
        signature = None  # аргументы вне проверенного набора — разрешать только точную команду
    return _exec(what, signature)


# Аргументы, с которыми разрешение «программа с любыми аргументами» остаётся безопасным:
# (флаги без значения, флаги со значением, что допустимо после «--»). Позиционные —
# только статичные пути внутри проекта. Флаги вроде pytest --basetemp/-p, go -exec,
# cargo --config, npm --script-shell сюда не входят: с ними — только точная команда.
_QUIET = frozenset({"-q", "--quiet", "-v", "--verbose"})
_LINT_FLAGS = _QUIET | {
    "--strict",
    "--noEmit",
    "--pretty",
    "--check",
    "--diff",
    "--statistics",
    "--show-source",
    "--show-error-codes",
    "--no-error-summary",
    "--show-fixes",
    "--no-fix",
    "--no-cache",
    "--exit-zero",
    "--color",
    "--no-color",
    "--warn-unused-ignores",
    "--incremental",
}
_LINT_VALUES = frozenset(
    {
        "--select",
        "--ignore",
        "--extend-select",
        "--extend-ignore",
        "--max-line-length",
        "--target-version",
        "-p",
        "--project",
        "--python-version",
        "--output-format",
    }
)
_SIGNATURE_ARGS: dict[str, tuple[frozenset[str], frozenset[str], frozenset[str] | None]] = {
    "pytest": (
        _QUIET
        | {
            "-qq",
            "-vv",
            "-vvv",
            "-x",
            "-s",
            "-l",
            "--lf",
            "--ff",
            "--nf",
            "--sw",
            "--last-failed",
            "--failed-first",
            "--new-first",
            "--stepwise",
            "--exitfirst",
            "--no-header",
            "--no-summary",
            "--showlocals",
            "--co",
            "--collect-only",
            "--disable-warnings",
            "-ra",
            "-rA",
            "-rf",
            "-rE",
            "-rs",
            "-rx",
            "-rX",
            "-rP",
            "--runxfail",
            "--strict-markers",
            "--color",
            "--no-cov",
        },
        frozenset(
            {
                "-k",
                "-m",
                "--maxfail",
                "--durations",
                "-n",
                "--tb",
                "-W",
                "--deselect",
                "--color",
                "-r",
            }
        ),
        None,
    ),
    "go test": (
        _QUIET | {"-race", "-short", "-cover", "-failfast", "-json", "-benchmem"},
        frozenset(
            {"-run", "-count", "-timeout", "-bench", "-p", "-tags", "-cpu", "-parallel", "-skip"}
        ),
        None,
    ),
    "go build": (_QUIET | {"-race"}, frozenset({"-tags", "-p"}), None),
    "go vet": (_QUIET, frozenset({"-tags"}), None),
    "cargo test": (
        _QUIET
        | {
            "--release",
            "--all",
            "--workspace",
            "--all-targets",
            "--all-features",
            "--lib",
            "--bins",
            "--tests",
            "--benches",
            "--examples",
            "--no-fail-fast",
            "--locked",
            "--offline",
            "--frozen",
            "--doc",
            "--no-default-features",
        },
        frozenset(
            {
                "-p",
                "--package",
                "--features",
                "-F",
                "--bin",
                "--test",
                "--example",
                "--bench",
                "-j",
                "--jobs",
            }
        ),
        frozenset(
            {
                "--nocapture",
                "--include-ignored",
                "--ignored",
                "--exact",
                "-q",
                "--quiet",
                "--show-output",
                "--test-threads",
            }
        ),
    ),
    "cargo clippy": (
        _QUIET
        | {
            "--release",
            "--all",
            "--workspace",
            "--all-targets",
            "--all-features",
            "--lib",
            "--bins",
            "--tests",
            "--no-deps",
            "--locked",
            "--offline",
        },
        frozenset({"-p", "--package", "--features", "-F", "-j"}),
        frozenset({"-D", "-W", "-A", "warnings"}),
    ),
    "make": (
        frozenset({"-k", "-s", "--silent", "--keep-going", "--no-print-directory"}),
        frozenset({"-j", "--jobs"}),
        None,
    ),
    "jest": (
        _QUIET
        | {
            "--ci",
            "--runInBand",
            "-i",
            "--silent",
            "--bail",
            "--passWithNoTests",
            "--watchAll=false",
            "--no-watchman",
            "--detectOpenHandles",
        },
        frozenset({"-t", "--testNamePattern"}),
        None,
    ),
    "vitest": (_QUIET | {"run", "--run", "--silent", "--bail"}, frozenset({"-t"}), None),
    "tox": (_QUIET | {"-p", "--parallel"}, frozenset({"-e"}), None),
    "nox": (_QUIET, frozenset({"-s", "--session", "-k"}), None),
}
for _tool in (
    "mypy",
    "pyright",
    "ty",
    "flake8",
    "pylint",
    "ruff check",
    "ruff format --check",
    "tsc",
    "eslint",
    "golangci-lint",
    "black",
    "isort",
    "prettier",
    "shellcheck",
    "hadolint",
    "yamllint",
    "cargo fmt --check",
    "cargo check",
    "cargo build",
    "go vet",
    "rubocop",
    "phpstan",
    "psalm",
    "ktlint",
    "swiftlint",
    "rspec",
    "phpunit",
    "mocha",
    "bandit",
):
    _SIGNATURE_ARGS.setdefault(_tool, (_LINT_FLAGS, _LINT_VALUES, None))
_SIGNATURE_ARGS["cargo build"] = _SIGNATURE_ARGS["cargo test"]
_SIGNATURE_ARGS["cargo check"] = _SIGNATURE_ARGS["cargo clippy"]


def _signature_args_ok(signature: str, args: list[Word], env: Env) -> bool:
    """Аргументы не выходят за проверенный набор для разрешения по сигнатуре."""
    spec = _SIGNATURE_ARGS.get(signature)
    if spec is None:
        base = signature.split(" ", 1)[0] if signature.startswith(("make ",)) else None
        spec = _SIGNATURE_ARGS.get(base) if base else None
    if spec is None:
        return not args
    plain, valued, after = spec
    i = 0
    rest = False
    while i < len(args):
        a = args[i]
        if not a.static:
            return False
        v = a.value
        if rest:
            if after is None or not (
                v in after or v.split("=", 1)[0] in after or not v.startswith("-")
            ):
                return False
            i += 1
            continue
        if v == "--":
            rest = True
            i += 1
            continue
        if v.startswith("-") or v in plain:
            name = v.split("=", 1)[0]
            if v in plain:
                i += 1
                continue
            if name in valued:
                i += 1 if "=" in v else 2
                continue
            if len(v) > 2 and not v.startswith("--") and v[:2] in valued:  # -j8, -kexpr
                i += 1
                continue
            return False
        # позиционный: путь внутри проекта (pytest: tests/test_x.py::test_y)
        path = v.split("::", 1)[0]
        if scope(Word(path), root=env.root, cwd=env.cwd) not in (Scope.INSIDE,):
            return False
        i += 1
    return True


# --------------------------------------------------------------------------- #
# Переменные и окружение
# --------------------------------------------------------------------------- #
# Раскрытие этих переменных в аргументах ничего секретного не показывает.
SAFE_VARS = frozenset(
    {
        "?",
        "#",
        "@",
        "*",
        "$",
        "!",
        "-",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "PWD",
        "OLDPWD",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "PATH",
        "LANG",
        "TERM",
        "VIRTUAL_ENV",
        "PYTHONPATH",
        "NODE_ENV",
        "GOPATH",
        "GOROOT",
        "CARGO_HOME",
        "JAVA_HOME",
        "TMPDIR",
        "HOSTNAME",
        "UID",
        "RANDOM",
        "LINENO",
        "SECONDS",
        "OSTYPE",
        "CI",
    }
)
# Переменные окружения, меняющие, какой код запустится (подмена программ, хуки, пейджеры).
_DANGER_ENV = re.compile(
    r"^(LD_\w+|DYLD_\w+|PATH|IFS|CDPATH|PAGER|GIT_\w+|EDITOR|VISUAL|BASH_ENV|ENV|SHELLOPTS|"
    r"BASHOPTS|PS4|PROMPT_COMMAND|HOME|XDG_CONFIG_HOME|NODE_OPTIONS|PERL5OPT|PERL5LIB|"
    r"RUBYOPT|RUBYLIB|PYTHONSTARTUP|PYTHONHOME|PYTHONINSPECT|LESSOPEN|LESSCLOSE|LESSKEY|"
    r"JAVA_TOOL_OPTIONS|_JAVA_OPTIONS|RIPGREP_CONFIG_PATH|RUSTC_WRAPPER|CC|CXX|LD|"
    r"MANPAGER|BROWSER|SSH_ASKPASS|SUDO_ASKPASS|GCONV_PATH|NODE_PATH|DENO_\w+|NPM_CONFIG_\w+|"
    r"npm_config_\w+|PIP_\w+|UV_\w+|POETRY_\w+|CARGO_\w+|GOFLAGS|GOPROXY|MAKEFLAGS|TAR_OPTIONS)$"
)
# Безобидные для чтения: локаль, цвета, режимы вывода.
_SAFE_ENV = re.compile(
    r"^(LANG|LANGUAGE|LC_\w+|TZ|TERM|COLUMNS|LINES|NO_COLOR|FORCE_COLOR|CLICOLOR\w*|"
    r"PYTHONUNBUFFERED|PYTHONDONTWRITEBYTECODE|PYTHONIOENCODING|PYTHONWARNINGS|PYTHONHASHSEED|"
    r"PYTHONUTF8|PYTHONPATH|CI|NODE_ENV|RUST_LOG|RUST_BACKTRACE|DEBUG|VERBOSE|GIT_PAGER|"
    r"PAGER|PIP_DISABLE_PIP_VERSION_CHECK)$"
)
_PAGERS = frozenset({"cat", "less", "more", ""})
_SECRETISH = re.compile(r"KEY|TOKEN|SECRET|PASS|CRED|AUTH|COOKIE|SESSION|PRIVATE", re.I)


def check_env(name: str, value: Word, *, reading: bool) -> Part | None:
    """Префиксное присваивание ``NAME=value cmd`` (или ``export``): None — безвредно."""
    if name in ("GIT_PAGER", "PAGER") and value.static and value.value in _PAGERS:
        return None
    if name == "PYTHONPATH":
        return None  # проверяется у интерпретатора Python
    if _DANGER_ENV.match(name):
        return _danger(f"переменная {name} меняет, какой код запустится")
    if reading and not _SAFE_ENV.match(name):
        return _unknown(f"переменная {name} может изменить поведение команды")
    return None


def _check_vars(env: Env, args: list[Word], what: str) -> Part | None:
    """Раскрытие переменных в аргументах, которые выводятся (``echo $X``): только известные."""
    for a in args:
        if a.static:
            continue
        if not a.vars:
            return _unknown(f"{what}: аргумент известен только при выполнении ({a.raw})")
        for name in a.vars:
            if name not in SAFE_VARS and not (name in env.assigned and name in a.plain):
                return _unknown(f"{what}: переменная ${name} (может быть секретом)")
    return None


# --------------------------------------------------------------------------- #
# Чтение: простые программы
# --------------------------------------------------------------------------- #
# Аргументы — пути к файлам, которые читаются (без побочных эффектов).
_READ_FILES = frozenset(
    {
        "cat",
        "head",
        "tail",
        "wc",
        "nl",
        "tac",
        "rev",
        "fold",
        "fmt",
        "column",
        "cut",
        "comm",
        "cmp",
        "diff",
        "md5sum",
        "sha1sum",
        "sha224sum",
        "sha256sum",
        "sha384sum",
        "sha512sum",
        "b2sum",
        "cksum",
        "sum",
        "shasum",
        "md5",
        "od",
        "hexdump",
        "strings",
        "base32",
        "zcat",
        "gzcat",
        "zipinfo",
        "ls",
        "dir",
        "vdir",
        "du",
        "stat",
        "tree",
        "bat",
        "batcat",
        "exa",
        "eza",
        "lsd",
        "paste",
        "join",
        "expand",
        "unexpand",
        "pr",
        "look",
        "tsort",
        "file",
        "readelf",
        "objdump",
        "nm",
        "size",
        "ldd",
    }
)
# Аргументы — не пути (или только проверка существования): ничего не читают.
_NO_FILES = frozenset(
    {
        "echo",
        "printf",
        "true",
        "false",
        ":",
        "sleep",
        "seq",
        "expr",
        "yes",
        "uname",
        "whoami",
        "id",
        "groups",
        "nproc",
        "free",
        "uptime",
        "locale",
        "getconf",
        "pwd",
        "which",
        "type",
        "cal",
        "tty",
        "logname",
        "arch",
        "w",
        "who",
        "pgrep",
        "basename",
        "dirname",
        "realpath",
        "readlink",
        "test",
        "[",
        "[[",
        "exit",
        "return",
        "wait",
        "times",
        "shift",
        "getopts",
        "read",
        "printenv",
        "df",
        "lscpu",
        "lsblk",
        "vm_stat",
        "sw_vers",
        "jobs",
        "factor",
        "numfmt",
        "tput",
        "dircolors",
        "date",
        "hostname",
        "dmesg",
        "lsof",
    }
)
_DASH_BUILTINS = frozenset(
    {
        "cd",
        "echo",
        "printf",
        "test",
        "[",
        "pwd",
        "true",
        "false",
        ":",
        "type",
        "command",
        "export",
        "set",
        "unset",
        "read",
        "shift",
        "exit",
        "return",
        "kill",
        "ulimit",
        "umask",
        "wait",
        "times",
        "trap",
        "alias",
        "unalias",
        "getopts",
        "hash",
        "jobs",
        "fg",
        "bg",
        "local",
        "eval",
        "exec",
        ".",
        "source",
        "readonly",
        "break",
        "continue",
    }
)


_TREE_VALUED = ("-o", "-L", "-P", "-I", "-H", "-T")
_DIFF_VALUED = (
    "-x",
    "--exclude",
    "-X",
    "--exclude-from",
    "-I",
    "--ignore-matching-lines",
    "-F",
    "--show-function-line",
    "-S",
    "--starting-file",
    "-U",
    "-C",
    "-W",
    "--width",
    "-D",
    "--ifdef",
    "--label",
    "--tabsize",
    "--horizon-lines",
    "--from-file",
    "--to-file",
)
# util-linux column (у BSD column — подмножество).
_COLUMN_VALUED = tuple(f"-{ch}" for ch in "CcEHilNnOopRrSsTW") + (
    "--columns",
    "--separator",
    "--output-separator",
    "--table-columns",
    "--table-name",
    "--table-order",
    "--table-right",
    "--table-hide",
    "--table-truncate",
    "--table-noextreme",
    "--table-wrap",
    "--tree",
    "--tree-id",
    "--tree-parent",
    "--output-width",
)


def _simple_read(env: Env, name: str, args: list[Word]) -> Part:
    flags, pos = _split(args)
    fl = _flagset(flags)
    if name == "tree":
        # tree берёт значения букв группы из следующих слов по порядку (`-Lo 2 out`), а не
        # из остатка слова: совпадает с getopt, только если буква со значением — последняя.
        for f in flags:
            v = f.value
            if not v.startswith("--") and any("-" + ch in _TREE_VALUED for ch in v[1:-1]):
                return _unknown(f"tree {v}: значения флагов в группе не разобрать")
        if _has(fl, "-o"):
            return _check_write(env, _values(args, "-o", valued=_TREE_VALUED), "tree -o")
        pos = _split(_without_values(args, *_TREE_VALUED))[1]
    if name == "file" and _has(fl, "-C", "--compile"):
        return _unknown("file -C пишет файл")
    if name in ("head", "tail"):
        pos = [
            p
            for p in _without_values(
                args, "-n", "-c", "--lines", "--bytes", "--pid", "-s", "--sleep-interval"
            )
            if not (p.static and (p.value.startswith("-") or p.value.lstrip("+-").isdigit()))
        ]
        flags = []
    if name == "cut":
        pos = _split(
            _without_values(args, "-d", "-f", "-b", "-c", "--delimiter", "--output-delimiter")
        )[1]
    if name in ("column",):
        # -t — таблица, без значения (раньше съедал имя файла: `column -t файл`).
        pos = _split(_without_values(args, *_COLUMN_VALUED))[1]
    # Флаги, значения которых — файлы: их читают так же, как позиционные.
    if _abbrev(args, "--files0-from"):
        return _unknown(f"{name} --files0-from: файлы по списку из файла")
    read_values: list[Word] = []
    if name == "du":
        rest, found = _options(
            args,
            (
                "-d",
                "--max-depth",
                "-t",
                "--threshold",
                "-B",
                "--block-size",
                "-X",
                "--exclude-from",
            ),
        )
        pos = _split(rest)[1]
        read_values = [w for n, w in found if n in ("-X", "--exclude-from")]
    if name == "diff":
        rest, found = _options(args, _DIFF_VALUED)
        flags, pos = _split(rest)
        read_values = [
            w for n, w in found if n in ("--from-file", "--to-file", "-X", "--exclude-from")
        ]
        if _has(_flagset(flags), "-r", "-R", "--recursive"):  # содержимое всех файлов дерева
            bad = _check_read(env, read_values, name) or _check_recursive(env, pos, "diff -r")
            return bad if bad is not None else _read("diff -r: сравнение каталогов")
    if name == "file":
        rest, found = _options(args, ("-m", "--magic-file", "-f", "--files-from", "-F"))
        if any(n in ("-f", "--files-from") for n, _ in found):
            return _unknown("file -f: файлы по списку из файла")
        pos = _split(rest)[1]
        read_values = [w for n, w in found if n in ("-m", "--magic-file")]
    if name in ("bat", "batcat") and _abbrev(args, "--pager"):
        return _unknown(f"{name} --pager: запускает программу")
    listing = name in ("ls", "dir", "vdir", "du", "stat", "tree", "exa", "eza", "lsd")
    bad = (
        _check_read(env, read_values, name)
        or _check_read(env, pos, name, content=not listing)
        or _check_vars(env, args, name)
    )
    if bad is not None:
        return bad
    return _read(f"{name}: чтение")


def _no_files(env: Env, name: str, args: list[Word]) -> Part:
    flags, pos = _split(args)
    fl = _flagset(flags)
    if name == "date":
        rest, found = _options(
            args, ("-d", "--date", "-f", "--file", "-r", "--reference", "-s", "--set")
        )
        if (
            any(n in ("-s", "--set") for n, _ in found)
            or _abbrev(args, "--set", "--file")
            or any(p.static and p.value[:1].isdigit() for p in _split(rest)[1])
        ):
            return _unknown("date может установить системное время")
        # date -f ФАЙЛ печатает строки, которые не разобрала, — содержимое файла
        bad = _check_read(env, [w for n, w in found if n in ("-f", "--file")], "date -f")
        if bad is not None:
            return bad
    if name == "dmesg":  # -F ФАЙЛ: вывод файла вместо буфера ядра
        bad = _check_read(env, _values(args, "-F", "--file"), "dmesg -F")
        if bad is not None:
            return bad
    if name == "dircolors":  # печатает разобранный файл (и его строки в ошибках)
        bad = _check_read(env, pos, name)
        if bad is not None:
            return bad
    if name == "printenv" and not pos:
        return _unknown("printenv без аргументов выводит всё окружение (там бывают секреты)")
    if name == "printenv":
        for p in pos:
            if p.value not in SAFE_VARS and _SECRETISH.search(p.value):
                return _unknown(f"printenv {p.value}: возможно, секрет")
    if name in ("kill",):
        return _write("kill: завершение процессов")
    if name == "hostname" and any(p.static and not p.value.startswith("-") for p in pos):
        return _unknown("hostname с аргументом меняет имя машины")
    if name == "dmesg" and _has(
        fl, "-c", "-C", "--clear", "--read-clear", "-n", "--console-level", "-D", "-E"
    ):
        return _unknown("dmesg: изменение буфера ядра")
    bad = _check_vars(env, args, name)
    return bad if bad is not None else _read(f"{name}: без побочных эффектов")


_PS_VALUE_FLAGS = frozenset(
    {
        "-o",
        "-O",
        "-p",
        "-u",
        "-U",
        "-g",
        "-G",
        "-C",
        "-t",
        "-q",
        "-s",
        "--format",
        "--pid",
        "--ppid",
        "--user",
        "--User",
        "--group",
        "--Group",
        "--sort",
        "--tty",
        "--sid",
        "--cols",
        "--columns",
        "--width",
        "--rows",
        "--lines",
        "-k",
        "--quick-pid",
    }
)


def _ps(args: list[Word]) -> Part:
    """ps: BSD-модификатор ``e`` выводит окружение процессов (и ключи gigacrew)."""
    expect_value = False
    for a in args:
        if not a.static:
            return _unknown("ps: аргумент известен только при выполнении")
        v = a.value
        if expect_value:
            expect_value = False
            if re.search(r"env|environ", v, re.I):
                return _unknown("ps: формат с окружением процессов")
            continue
        if v.startswith("--"):
            name = v.split("=", 1)[0]
            if re.search(r"env", v, re.I):
                return _unknown("ps: вывод окружения процессов")
            if name in _PS_VALUE_FLAGS and "=" not in v:
                expect_value = True
            continue
        if v.startswith("-"):
            if v in _PS_VALUE_FLAGS:
                expect_value = True
            elif len(v) > 2 and v[:2] in _PS_VALUE_FLAGS:
                if re.search(r"env|environ", v[2:], re.I):
                    return _unknown("ps: формат с окружением процессов")
            continue
        # BSD-стиль: буквы без дефиса (aux, axww); e/E — окружение
        if re.fullmatch(r"[a-zA-Z]+", v):
            if "e" in v or "E" in v:
                return _unknown("ps e: вывод окружения процессов (там ключи и токены)")
            continue
        return _unknown(f"ps {v}: аргумент не проверен")
    return _read("ps: список процессов")


# Путь, найденный find/fd: лежит в проверенных каталогах поиска и начинается с них
# («./x»), поэтому не похож на флаг.
_FOUND = Word(".", "{}")


# --------------------------------------------------------------------------- #
# Поиск: grep, rg, find, fd, xargs
# --------------------------------------------------------------------------- #
# Флаги grep со значением. --color — без: значение только через «=» (`--color=always`),
# иначе следующее слово — образец, и первый файл не проверялся бы.
_GREP_VALUED = (
    "-e",
    "--regexp",
    "-f",
    "--file",
    "-m",
    "--max-count",
    "-A",
    "-B",
    "-C",
    "--context",
    "--after-context",
    "--before-context",
    "--label",
    "-d",
    "--directories",
    "-D",
    "--devices",
    "--binary-files",
    "--include",
    "--exclude",
    "--exclude-dir",
    "--exclude-from",
    "--include-from",
)


def _grep(env: Env, name: str, args: list[Word]) -> Part:
    hit = _abbrev_only(
        args,
        "--file",
        "--exclude-from",
        "--include-from",
        "--regexp",
        allow=(
            "--include",
            "--exclude",
            "--exclude-dir",
            "--fixed-strings",
            "--files-with-matches",
            "--files-without-match",
            "--extended-regexp",
        ),
    )
    if hit is not None:
        return _unknown(f"{name} {hit.value}: сокращённый флаг")
    rest, found = _options(args, _GREP_VALUED)
    flags, pos = _split(rest)
    fl = _flagset(flags)
    has_pattern_flag = any(n in ("-e", "--regexp", "-f", "--file") for n, _ in found)
    paths = pos if has_pattern_flag else pos[1:]
    lists = [w for n, w in found if n in ("-f", "--file", "--exclude-from", "--include-from")]
    bad = _check_read(env, lists, name)
    if bad is not None:
        return bad
    recursive = _has(fl, "-r", "-R", "--recursive", "--dereference-recursive") or any(
        n in ("-d", "--directories") and w.value == "recurse" for n, w in found
    )
    if recursive:
        bad = _check_recursive(
            env, paths, f"{name} -r", follow=_has(fl, "-R", "--dereference-recursive")
        )
    else:
        bad = _check_read(env, paths, name)
    return bad if bad is not None else _read(f"{name}: поиск в файлах проекта")


def _rg(env: Env, name: str, args: list[Word]) -> Part:
    for a in args:
        if a.static and a.value.split("=", 1)[0] in (
            "--pre",
            "--pre-glob",
            "--search-zip",
            "-z",
            "--hostname-bin",
        ):
            return _unknown(f"{name} {a.value}: запускает внешние программы")
    value_flags = (
        "-e",
        "--regexp",
        "-f",
        "--file",
        "-g",
        "--glob",
        "--iglob",
        "-t",
        "--type",
        "-T",
        "--type-not",
        "-m",
        "--max-count",
        "-A",
        "-B",
        "-C",
        "--context",
        "--max-depth",
        "-d",
        "-M",
        "--max-columns",
        "-j",
        "--threads",
        "--type-add",
        "--sort",
        "--sortr",
        "-E",
        "--encoding",
        "--color",
        "--colors",
        "--ignore-file",
        "-r",
        "--replace",
        "--path-separator",
        "--engine",
        "--max-filesize",
        "--field-match-separator",
        "--field-context-separator",
        "--context-separator",
    )
    rest, found = _options(args, value_flags)
    flags, pos = _split(rest)
    fl = _flagset(flags)
    if _has(fl, "-z"):  # в группе: `-iz`
        return _unknown(f"{name} -z: запускает внешние программы")
    has_pattern = any(n in ("-e", "--regexp", "-f", "--file") for n, _ in found)
    listing = _has(fl, "--files", "--type-list")
    paths = pos if (has_pattern or listing) else pos[1:]
    files_from = [w for n, w in found if n in ("-f", "--file", "--ignore-file")]
    bad = _check_read(env, files_from, name)
    if bad is not None:
        return bad
    unrestricted = sum(f.value.count("u") for f in flags if re.fullmatch(r"-[a-zA-Z.]+", f.value))
    hidden = _has(fl, "--hidden", "-.") or unrestricted >= 2
    follow = _has(fl, "-L", "--follow")
    if listing:
        bad = _check_read(env, paths, name)
    else:
        bad = _check_recursive(env, paths, name, hidden=hidden, follow=follow)
    return bad if bad is not None else _read(f"{name}: поиск в файлах проекта")


_FIND_ACTIONS = frozenset({"-exec", "-execdir", "-ok", "-okdir"})
_FIND_WRITES = frozenset({"-fprint", "-fprint0", "-fprintf", "-fls"})


def _find(env: Env, name: str, args: list[Word], classify: Callable[[Command], Part]) -> Part:
    if any(a.static and a.value == "-files0-from" for a in args):
        return _unknown("find -files0-from: каталоги поиска по списку из файла")
    starts: list[Word] = []
    i = 0
    while i < len(args) and not (
        args[i].static and (args[i].value.startswith("-") or args[i].value in ("(", ")", "!", ","))
    ):
        starts.append(args[i])
        i += 1
    expr = args[i:]
    parts: list[Part] = []
    reads_contents = False
    j = 0
    while j < len(expr):
        w = expr[j]
        if not w.static:
            j += 1
            continue
        v = w.value
        if v == "-delete":
            return _danger("find -delete: удаление файлов")
        if v in _FIND_WRITES:
            target = expr[j + 1 : j + 2]
            parts.append(_check_write(env, target, f"find {v}"))
            j += 2
            continue
        if v in _FIND_ACTIONS:
            inner: list[Word] = []
            k = j + 1
            while k < len(expr) and not (expr[k].static and expr[k].value in (";", "+")):
                # {} — найденный путь: внутри каталогов поиска (они проверены ниже)
                inner.append(_FOUND if expr[k].value == "{}" else expr[k])
                k += 1
            if not inner:
                return _unknown("find -exec без команды")
            sub = classify(Command(argv=inner, cwd=env.cwd))
            if sub.level is CommandLevel.READ:
                reads_contents = True
            elif (
                inner[0].value == "rm"
                and sub.level is not CommandLevel.DANGER
                and all(not a.value.startswith("-") for a in inner[1:] if a.static)
            ):
                sub = _write("find -exec rm: удаление найденных файлов")
            parts.append(sub)
            j = k + 1
            continue
        j += 1
    bad = _check_read(env, starts, name)
    if bad is not None:
        return bad
    if reads_contents:
        rec = _check_recursive(env, starts, "find -exec")
        if rec is not None:
            return rec
    if any(p.level is None for p in parts):
        return next(p for p in parts if p.level is None)
    return _combine(_read("find: поиск файлов"), *parts)


_FD_VALUED = (
    "-e",
    "--extension",
    "-t",
    "--type",
    "-E",
    "--exclude",
    "-d",
    "--max-depth",
    "--min-depth",
    "-S",
    "--size",
    "--changed-within",
    "--changed-before",
    "-j",
    "--threads",
    "--color",
    "-c",
    "-C",
    "--base-directory",
    "--search-path",
)


def _fd_exec(args: list[Word]) -> int | None:
    """Индекс слова с -x/-X (дальше — команда); -1 — в группе после x есть ещё буквы."""
    i = 0
    while i < len(args):
        a = args[i]
        v = a.value
        if not a.static or not v.startswith("-") or v == "-":
            i += 1
            continue
        if v == "--":
            return None
        if v in ("-x", "--exec", "-X", "--exec-batch"):
            return i
        if v.startswith(("--exec=", "--exec-batch=")):
            return -1
        if v in _FD_VALUED:
            i += 2
            continue
        if not v.startswith("--"):
            for k, ch in enumerate(v[1:], 1):
                if ch in "xX":
                    return i if k == len(v) - 1 else -1
                if "-" + ch in _FD_VALUED:
                    i += 1 if k + 1 < len(v) else 2
                    break
            else:
                i += 1
            continue
        i += 1
    return None


def _under(base: Word, path: Word) -> Word:
    """Путь ``path`` для программы, перешедшей в каталог ``base``: относительный — от
    ``base``. Шаблон имён оболочка раскроет от каталога команды — такой путь «не знаю»."""
    if path.static and not path.glob and path.value.startswith(("/", "~")):
        return path
    joined = posixpath.join(base.value, path.value)
    return Word(joined, joined, static=base.static and path.static and not path.glob)


def _fd_dirs(args: list[Word]) -> list[Word]:
    """Каталоги поиска fd: позиционные после шаблона, --search-path и --base-directory (-C).
    Относительные пути поиска fd отсчитывает от --base-directory — так они и проверяются."""
    rest, found = _options(args, _FD_VALUED)
    _, pos = _split(rest)
    paths = pos[1:] + [w for n, w in found if n == "--search-path"]
    bases = [w for n, w in found if n in ("-C", "--base-directory")]
    if bases:
        paths = [_under(b, p) for b in bases for p in paths]
    return paths + bases


def _fd(env: Env, name: str, args: list[Word], classify: Callable[[Command], Part]) -> Part:
    at = _fd_exec(args)
    if at == -1:
        return _unknown(f"{name}: --exec в группе флагов не разобрать")
    if at is not None:
        # Каталоги поиска — до -x/-X: найденные там файлы и прочитает команда.
        dirs = _fd_dirs(args[:at])
        bad = _check_read(env, dirs, name)
        if bad is not None:
            return bad
        inner = [
            _FOUND if w.value == "{}" else w
            for w in args[at + 1 :]
            if not (w.static and w.value == ";")
        ]
        if not inner:
            return _unknown("fd --exec без команды")
        sub = classify(Command(argv=inner, cwd=env.cwd))
        if sub.level is not CommandLevel.READ:
            return sub if sub.level is not None else _unknown(f"fd --exec: {sub.reason}")
        rec = _check_recursive(env, dirs, "fd --exec")
        return rec if rec is not None else _read("fd --exec: чтение найденных файлов")
    bad = _check_read(env, _fd_dirs(args), name)
    return bad if bad is not None else _read(f"{name}: поиск файлов")


# Флаги xargs, которые не меняют, что и как запускается; остальные — «не знаю».
_XARGS_VALUES = frozenset(
    {
        "-I",
        "-L",
        "-n",
        "-P",
        "-d",
        "--replace",
        "--max-lines",
        "--max-args",
        "--max-procs",
        "--delimiter",
    }
)
_XARGS_FLAGS = frozenset({"-0", "--null", "-r", "--no-run-if-empty", "-t", "--verbose"})


def _xargs(env: Env, cmd: Command, args: list[Word], classify: Callable[[Command], Part]) -> Part:
    i = 0
    lists: list[Word] = []  # -a ФАЙЛ: аргументы из файла, а не со stdin
    while i < len(args) and args[i].static and args[i].value.startswith("-"):
        v = args[i].value
        name = v.split("=", 1)[0]
        if v in ("-a", "--arg-file") and i + 1 < len(args):
            lists.append(args[i + 1])
            i += 2
            continue
        if name == "--arg-file" or (v.startswith("-a") and len(v) > 2):
            lists.append(Word(v.split("=", 1)[1] if "=" in v else v[2:], v))
            i += 1
            continue
        if v in _XARGS_VALUES and i + 1 < len(args):
            i += 2
            continue
        if (
            name in _XARGS_VALUES
            and "=" in v
            or v in _XARGS_FLAGS
            or re.fullmatch(r"-[nLP]\d+|-I.+", v)
        ):
            i += 1
            continue
        return _unknown(f"xargs {v}: флаг не проверен")
    inner = args[i:] or [Word("echo")]
    bad = _check_read(env, lists, "xargs -a")
    if bad is not None:
        return bad
    if inner[0].value in ("echo", "printf") and len(inner) == 1:
        return _read("xargs echo")
    # Аргументы приходят со stdin. Если это пути из find/fd по проекту — они начинаются с
    # каталога поиска и не похожи на флаги; иначе известны только при выполнении.
    source = cmd.stdin
    from_find = (
        not lists
        and source is not None
        and source.name is not None
        and source.name.value in ("find", "fd", "fdfind")
        and classify(source).level is CommandLevel.READ
    )
    placeholder = _FOUND if from_find else Word("{}", "{}", static=False)
    sub = classify(Command(argv=[*inner, placeholder], cwd=env.cwd))
    if not from_find and inner[0].static and inner[0].value == "rm":
        # имена известны только при выполнении — как rm по шаблону
        return _danger("xargs rm: удаление файлов по списку, известному при выполнении")
    if sub.level is CommandLevel.READ:
        if not from_find:
            return _unknown("xargs: аргументы приходят со stdin (известны только при выполнении)")
        rec = _check_recursive(env, [], "xargs")
        return rec if rec is not None else _read("xargs: чтение найденных файлов проекта")
    if from_find and sub.level is not CommandLevel.DANGER:
        dynamic = classify(Command(argv=[*inner, Word("{}", "{}", static=False)], cwd=env.cwd))
        return dynamic if dynamic.level is None else sub
    return sub


# --------------------------------------------------------------------------- #
# Текстовые утилиты с программами внутри: sed, awk, jq, yq, sort, uniq, xxd, tee
# --------------------------------------------------------------------------- #
_SED_SAFE = re.compile(
    r"^\s*(\d+|\$|/[^/]*/)?(\s*,\s*(\d+|\$|/[^/]*/))?\s*"
    r"(p|d|q|=|s(.)(?:(?!\5).)*\5(?:(?!\5).)*\5[gipI0-9]*)\s*$"
)


def _sed_script_safe(script: str) -> bool:
    return all(_SED_SAFE.match(part) for part in re.split(r";|\n", script) if part.strip())


_SED_ADDR = r"^\s*(\d+|\$|/[^/]*/)?(\s*,\s*(\d+|\$|/[^/]*/))?\s*"


def _sed_runs_commands(script: str) -> bool:
    """Команда `e` (`1e id`) или флаг `e` у замены (`s/.*/id/e`) — запуск оболочки."""
    for part in re.split(r";|\n", script):
        if re.match(_SED_ADDR + r"e(\s|$)", part):
            return True
        m = re.match(_SED_ADDR + r"s(.)(?:(?!\4).)*\4(?:(?!\4).)*\4([a-zA-Z0-9]*)", part)
        if m is not None and "e" in m.group(5).split("w", 1)[0]:
            return True
    return False


def _sed(env: Env, args: list[Word]) -> Part:
    if (
        _abbrev(args, "--file")
        or _abbrev_only(args, "--expression", "--in-place")
        or any(
            a.static and re.fullmatch(r"-[a-zA-Z]*f.*", a.value) and not a.value.startswith("--")
            for a in args
        )
    ):
        return _unknown("sed -f: скрипт из файла")
    rest, found = _options(args, ("-e", "--expression", "-l", "--line-length"))
    scripts = [w for n, w in found if n in ("-e", "--expression")]
    flags, pos = _split(rest)
    fl = _flagset(flags)
    in_place = any(
        f.value.startswith(("-i", "--in-place")) or re.match(r"^-[a-zA-Z]*i", f.value)
        for f in flags
    )
    if not scripts:
        if not pos:
            return _unknown("sed без скрипта")
        scripts, pos = [pos[0]], pos[1:]
    for s in scripts:
        if s.static and _sed_runs_commands(s.value):
            return _danger("sed e: скрипт запускает команды оболочки")
        if not s.static or not _sed_script_safe(s.value):
            return _unknown("sed: скрипт с командами, которые не проверены (w, e, r…)")
    if in_place:
        if any(
            f.value.startswith("-i") and len(f.value) > 2 and not f.value[2:].isalpha()
            for f in flags
        ):
            pass  # -i.bak — суффикс резервной копии
        return _check_write(env, pos, "sed -i")
    _ = fl
    bad = _check_read(env, pos, "sed")
    return bad if bad is not None else _read("sed: вывод с заменами")


# ARGV/ARGC: программа подменила бы входные файлы (`ARGV[1] = ".env"`) мимо проверки путей.
_AWK_UNSAFE = re.compile(r"system|getline|close|fflush|ENVIRON|PROCINFO|ARGV|ARGC|@load|@include")
_AWK_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Запуск команд: system(), вывод в команду (`print | "sh"`), ввод из неё (`"id" | getline`).
_AWK_COMMANDS = re.compile(r'\bsystem\s*\(|\|&|(?<!\|)\|(?!\|)\s*"|"\s*\|(?!\|)')


def _awk_redirects(program: str) -> bool:
    """Выводит ли программа awk в файл или команду (``print > f``, ``print | "cmd"``).

    ``>`` — вывод только после print/printf вне скобок, иначе это сравнение
    (``NR>=10``, ``if ($1 > 5)``, ``print (a > b)``). ``|`` (кроме ``||``) — всегда
    вывод или ввод команды. С ``/`` вне строк регулярное выражение не отличить от
    деления без полного разбора — тогда любые ``>`` и ``|`` считаются выводом. ``/``
    внутри строки (``print "/etc"``) не мешает: первый ``/`` регулярного выражения или
    деления всегда вне строки — до него строки и комментарии разбираются как в awk.
    """
    depth = 0
    output: list[int] = []  # глубины скобок, где начались print/printf
    i, n = 0, len(program)
    while i < n:
        ch = program[i]
        if ch == '"':
            j = i + 1
            while j < n and program[j] != '"':
                j += 2 if program[j] == "\\" else 1
            if j >= n:
                return True  # строка не закрыта — не разобрать
            i = j + 1
            continue
        if ch == "#":  # комментарий до конца строки
            j = program.find("\n", i)
            i = n if j < 0 else j + 1
            continue
        if ch == "/":  # регулярное выражение или деление: дальше не разобрать
            return ">" in program or "|" in program
        name = _AWK_NAME.match(program, i)
        if name is not None:
            if name.group() in ("print", "printf"):
                output.append(depth)
            i = name.end()
            continue
        if ch == "|":
            if not program.startswith("||", i):
                return True
            i += 2
            continue
        if ch == ">" and output and output[-1] == depth:
            return True
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch in ";}":  # конец оператора: print закончился
            while output and output[-1] >= depth:
                output.pop()
        i += 1
    return False


# Опции awk, которые не меняют, откуда берётся программа, и ничего не пишут: разделитель
# полей и присваивания. Остальные — «не проверено»: -f/-e/-E/-i/-l (программа или
# расширения не из первого аргумента), -W (опции реализации: mawk `-W exec файл`, gawk
# `-W source=…`), -d/-o/-p (запись файлов), сгруппированные короткие (`-bf файл`) и
# сокращения длинных (`--as x=1` сдвинул бы, какой аргумент — программа).
_AWK_OPTION = re.compile(r"-[Fv].+|--(field-separator|assign)=.*")


def _awk(env: Env, args: list[Word]) -> Part:
    rest, found = _options(args, ("-F", "-v", "--assign", "--field-separator"))
    given = [w for n, w in found if n in ("-v", "--assign")]
    flags, pos = _split(rest)
    for flag in flags:
        if not _AWK_OPTION.fullmatch(flag.value):
            return _unknown(
                f"awk: опция {flag.value} — программа или расширения из файла, запись файлов"
            )
    if not pos:
        return _unknown("awk без программы")
    program = pos[0]
    if program.static and _AWK_COMMANDS.search(program.value):
        return _danger("awk: программа запускает команды оболочки")
    if not program.static or _AWK_UNSAFE.search(program.value) or _awk_redirects(program.value):
        return _unknown("awk: программа с вызовом команд, записью или окружением")
    operands = [p for p in pos[1:] if p.static and "=" in p.value]  # var=значение
    assigns = [*given, *operands]
    if any(_AWK_UNSAFE.search(a.value.partition("=")[0]) for a in assigns if a.static):
        return _unknown("awk: присваивание ARGV, ARGC или переменным окружения")
    files = [p for p in pos[1:] if p not in operands]
    bad = _check_read(env, files, "awk") or _check_vars(env, given, "awk")
    return bad if bad is not None else _read("awk: обработка текста")


def _jq(env: Env, name: str, args: list[Word]) -> Part:
    if name == "yq" and (
        _has(_flagset(_split(args)[0]), "-i", "--inplace")
        or any(a.static and a.value.startswith("--inplace") for a in args)
    ):
        _, pos = _split(args)
        return _check_write(env, pos[1:], "yq -i")
    # --arg/--argjson/--rawfile/--slurpfile берут два слова (имя и значение или файл):
    # иначе значение приняли бы за выражение, а настоящее выражение — за файл.
    kept: list[Word] = []
    files: list[Word] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a.static and a.value == "--":
            kept.extend(args[i:])
            break
        if a.static and a.value in ("--arg", "--argjson", "--rawfile", "--slurpfile"):
            if a.value in ("--rawfile", "--slurpfile"):
                files.extend(args[i + 2 : i + 3])
            i += 3
            continue
        kept.append(a)
        i += 1
    flags, pos = _split(_without_values(kept, "--indent", "-L", "--library-path"))
    fl = _flagset(flags)
    if _has(fl, "--args", "--jsonargs"):
        return _unknown(f"{name} --args")
    # -f: выражение — из файла, это первый позиционный (все позиционные — файлы).
    has_file_program = _has(fl, "-f", "--from-file")
    program = None if has_file_program else (pos[0] if pos else None)
    if program is not None and (
        not program.static
        or re.search(
            r"\$ENV|\benv\b|input_filename|\$__loc__|\bload\b|@sh|getpath|\binclude\b|"
            r"\bimport\b",
            program.value,
        )
    ):
        return _unknown(f"{name}: выражение читает окружение или файлы")
    paths = pos if has_file_program else pos[1:]
    bad = _check_read(env, files + paths, name)
    return bad if bad is not None else _read(f"{name}: обработка JSON")


def _sort(env: Env, args: list[Word]) -> Part:
    if _abbrev(args, "--compress-program", "--files0-from", "--random-source"):
        return _unknown("sort: флаг запускает программу или читает файлы по списку")
    if _abbrev_only(args, "--output"):
        return _unknown("sort: сокращённый --output")
    rest, found = _options(
        args,
        (
            "-o",
            "--output",
            "-k",
            "--key",
            "-t",
            "--field-separator",
            "-S",
            "--buffer-size",
            "-T",
            "--temporary-directory",
            "--parallel",
            "--batch-size",
            "--sort",
            "-y",
        ),
    )
    out = [w for n, w in found if n in ("-o", "--output")]
    _, pos = _split(rest)
    bad = _check_read(env, pos, "sort")
    if bad is not None:
        return bad
    if out:
        return _check_write(env, out, "sort -o")
    return _read("sort: сортировка")


def _uniq(env: Env, args: list[Word]) -> Part:
    rest = _without_values(args, "-f", "-s", "-w", "--skip-fields", "--skip-chars", "--check-chars")
    _, pos = _split(rest)
    bad = _check_read(env, pos[:1], "uniq")
    if bad is not None:
        return bad
    if len(pos) > 1:
        return _check_write(env, pos[1:2], "uniq: файл вывода")
    return _read("uniq")


def _xxd(env: Env, args: list[Word]) -> Part:
    # У xxd свой разбор: `-ps` — один флаг (-p), а не -p с -s и значением.
    rest = _without_values(
        args,
        "-c",
        "-g",
        "-l",
        "-o",
        "-s",
        "-n",
        "-C",
        "-cols",
        "-len",
        "-seek",
        "-name",
        clusters=False,
    )
    _, pos = _split(rest)
    bad = _check_read(env, pos[:1], "xxd")
    if bad is not None:
        return bad
    if len(pos) > 1:
        return _check_write(env, pos[1:2], "xxd: файл вывода")
    return _read("xxd")


def _base64(env: Env, args: list[Word]) -> Part:
    if _abbrev_only(args, "--output", "--input"):
        return _unknown("base64: сокращённый флаг")
    rest, found = _options(
        args, ("-o", "--output", "-w", "--wrap", "-b", "--break", "-i", "--input")
    )
    out = [w for n, w in found if n in ("-o", "--output")]
    _, pos = _split(rest)
    bad = _check_read(env, pos + [w for n, w in found if n in ("-i", "--input")], "base64")
    if bad is not None:
        return bad
    return _check_write(env, out, "base64 -o") if out else _read("base64")


def _tee(env: Env, args: list[Word]) -> Part:
    _, pos = _split(args)
    if not pos:
        return _read("tee: только вывод")
    return _check_write(env, pos, "tee")


def _less(env: Env, name: str, args: list[Word]) -> Part:
    for a in args:
        if not a.static:
            continue
        if a.value.startswith("+") and ("!" in a.value or "|" in a.value):
            return _danger(f"{name} {a.value}: запуск команды оболочки")
        if a.value.startswith("+") or a.value.split("=", 1)[0] in (
            "-o",
            "-O",
            "--log-file",
            "--LOG-FILE",
            "-k",
            "--lesskey-file",
            "--lesskey-src",
        ):
            return _unknown(f"{name} {a.value}: команда или запись файла")
    _, pos = _split(args)
    bad = _check_read(env, pos, name)
    return bad if bad is not None else _read(f"{name}: просмотр")


# --------------------------------------------------------------------------- #
# Файлы: запись, архивы
# --------------------------------------------------------------------------- #
def _rm(env: Env, args: list[Word]) -> Part:
    flags, pos = _split(args)
    fl = _flagset(flags)
    if _has(fl, "-r", "-R", "-f", "--recursive", "--force", "-d", "--dir"):
        return _danger("rm с -r/-f: удаление без возврата")
    if not pos:
        return _unknown("rm без файлов")
    for p in pos:
        if p.glob:
            return _danger(f"rm {p.raw}: удаление по шаблону")
    part = _check_write(env, pos, "rm: удаление файла", edits=False)
    return part


def _cp_mv(env: Env, name: str, args: list[Word]) -> Part:
    rest, found = _options(args, ("-t", "--target-directory", "-S", "--suffix"))
    target_dir = [w for n, w in found if n in ("-t", "--target-directory")]
    flags, pos = _split(rest)
    if not pos:
        return _unknown(f"{name} без файлов")
    sources, dest = (pos, target_dir) if target_dir else (pos[:-1], pos[-1:])
    if name == "cp":
        bad = _check_read(env, sources, name)
        if bad is not None:
            return (
                bad
                if bad.level is not None
                else _unknown(
                    f"cp: копирование файла вне проекта или с секретами в проект ({bad.reason})"
                )
            )
    else:
        moved = _check_write(env, sources, name)
        if moved.level is not CommandLevel.WRITE:
            return moved
    return _check_write(env, dest, name)


# Флаги со значением у каждой программы: общий список сдвигал бы позиционные (у touch
# -a и -m — без значения, а `touch -a /etc/x a.txt` проверил бы только a.txt).
_WRITE_VALUED: dict[str, tuple[str, ...]] = {
    "mkdir": ("-m", "--mode"),
    "mkfifo": ("-m", "--mode"),
    "touch": ("-r", "--reference", "-d", "--date", "-t"),
    "truncate": ("-s", "--size", "-r", "--reference"),
    "install": (
        "-m",
        "--mode",
        "-o",
        "--owner",
        "-g",
        "--group",
        "-S",
        "--suffix",
        "-t",
        "--target-directory",
    ),
    # -n (новый файл: `dos2unix -n ВХОД ВЫХОД`) — без значения: оба файла проверяются.
    "dos2unix": ("-c", "--convmode"),
    "unix2dos": ("-c", "--convmode"),
    "fallocate": ("-l", "--length", "-o", "--offset"),
    "split": (
        "-a",
        "--suffix-length",
        "-b",
        "--bytes",
        "-C",
        "--line-bytes",
        "-l",
        "--lines",
        "-n",
        "--number",
        "-t",
        "--separator",
        "--additional-suffix",
    ),
    "csplit": ("-f", "--prefix", "-b", "--suffix-format", "-n", "--digits"),
}


def _simple_write(env: Env, name: str, args: list[Word]) -> Part:
    if name in ("split", "install") and _abbrev(args, "--filter", "--strip-program"):
        return _unknown(f"{name}: флаг запускает программу")
    rest, found = _options(args, _WRITE_VALUED.get(name, ()))
    flags, pos = _split(rest)
    if not pos:
        return _unknown(f"{name} без аргументов")
    edits = name in ("mkdir", "touch", "ln", "install", "dos2unix", "unix2dos", "truncate")
    target_dir = [w for n, w in found if n in ("-t", "--target-directory")]
    if name == "install" and target_dir:  # install -t КАТАЛОГ файлы… — копирование, как cp -t
        bad = _check_read(env, pos, name)
        if bad is not None:
            return bad if bad.level is not None else _unknown(f"install: {bad.reason}")
        return _check_write(env, target_dir, name, edits=edits)
    return _check_write(env, pos, name, edits=edits)


def _ln(env: Env, args: list[Word]) -> Part:
    rest, found = _options(args, ("-t", "--target-directory", "-S", "--suffix"))
    flags, pos = _split(rest)
    if not pos:
        return _unknown("ln без аргументов")
    target_dir = [w for n, w in found if n in ("-t", "--target-directory")]
    # Ссылка на файл вне проекта превратит его в «файл проекта» для следующих команд.
    for target in pos if target_dir else pos[:1]:
        s = scope(target, root=env.root, cwd=env.cwd)
        if s not in (Scope.INSIDE,):
            return _danger(f"ln: ссылка на {target.raw} — вне проекта или секреты")
    if target_dir:  # ln -t КАТАЛОГ цели… — ссылки создаются в КАТАЛОГЕ
        return _check_write(env, target_dir, "ln", edits=True)
    return _check_write(env, pos[1:] or [Word(Path(pos[0].value).name)], "ln", edits=True)


def _chmod(env: Env, args: list[Word]) -> Part:
    flags, pos = _split(args)
    fl = _flagset(flags)
    if _has(fl, "-R", "--recursive"):
        return _danger("chmod -R")
    if not pos:
        return _unknown("chmod без аргументов")
    mode = pos[0].value if pos[0].static else ""
    if (
        not mode
        or re.search(r"[st]|777|[0-7]?[0-7][0-7][2367]$|^[1-7][0-7]{3}$", mode)
        or _chmod_world_writable(mode)
    ):
        return _danger(f"chmod {mode}: права для всех или setuid")
    return _check_write(env, pos[1:], "chmod")


def _chmod_world_writable(mode: str) -> bool:
    """Символьный режим даёт запись «прочим»: `o+w`, `a+rwx`, `go+w`, `o=u` (копия)."""
    for clause in mode.split(","):
        m = re.fullmatch(r"([ugoa]*)((?:[-+=][rwxXst]*|[-+=][ugo])+)", clause)
        if m is None:
            return bool(clause) and not clause.isdigit()  # не разобрать — осторожно
        who, ops = m.groups()
        if who and not set(who) & set("oa"):
            continue
        for op, perms in re.findall(r"([-+=])([rwxXstugo]*)", ops):
            if op in "+=" and ("w" in perms or set(perms) & set("ugo")) and who:
                return True
    return False


# Флаги tar со значением (короткие — как в getopt GNU tar).
_TAR_VALUED = (
    "-f",
    "--file",
    "-C",
    "--directory",
    "-b",
    "--blocking-factor",
    "-g",
    "--listed-incremental",
    "-H",
    "--format",
    "-I",
    "--use-compress-program",
    "-F",
    "--info-script",
    "--new-volume-script",
    "-K",
    "--starting-file",
    "-L",
    "--tape-length",
    "-N",
    "--newer",
    "--after-date",
    "-T",
    "--files-from",
    "-V",
    "--label",
    "-X",
    "--exclude-from",
    "--exclude",
    "--transform",
    "--xform",
    "--owner",
    "--group",
    "--mode",
    "--mtime",
    "--suffix",
)
# Значения — программа (-I, -F), списки файлов вне проверки путей (-T, -X) или файл
# снимка, который tar перепишет (-g).
_TAR_DANGER_VALUED = frozenset(
    {
        "-I",
        "--use-compress-program",
        "-F",
        "--info-script",
        "--new-volume-script",
        "-T",
        "--files-from",
        "-X",
        "--exclude-from",
        "-g",
        "--listed-incremental",
    }
)


def _tar_old_style(args: list[Word]) -> list[Word]:
    """``tar czf out.tgz src`` → ``-c -z -f out.tgz src``: буквы первого слова без «-»;
    значения букв со значением — следующие слова по порядку (``tar xfC a.tar dir``)."""
    if not args or not args[0].static or not args[0].value.isalpha():
        return args
    out: list[Word] = []
    rest = list(args[1:])
    for ch in args[0].value:
        out.append(Word("-" + ch, args[0].raw))
        if "-" + ch in _TAR_VALUED and rest:
            out.append(rest.pop(0))
    return out + rest


def _tar(env: Env, args: list[Word]) -> Part:
    hit = _abbrev(
        args,
        "--to-command",
        "--use-compress-program",
        "--checkpoint-action",
        "--info-script",
        "--new-volume-script",
        "--rsh-command",
        "--rmt-command",
        "--index-file",
        "--absolute-names",
        "--files-from",
        "--exclude-from",
    )
    if hit is not None:
        return _danger(f"tar {hit.value}: запуск программы или файлы вне проекта")
    if _abbrev_only(args, "--file", "--directory", "--extract", "--create", "--list"):
        return _unknown("tar: сокращённый флаг")
    for a in args:
        if a.static and re.match(
            r"^--(to-command|use-compress-program|checkpoint-action|"
            r"info-script|new-volume-script|rsh-command|rmt-command)",
            a.value,
        ):
            return _danger(f"tar {a.value}: запуск программы")
    rest, found = _options(_tar_old_style(args), _TAR_VALUED)
    flags, pos = _split(rest)
    fl = _flagset(flags)
    for n, _ in found:
        if n in _TAR_DANGER_VALUED:
            return _danger(f"tar {n}: запуск программы или файлы по списку")
    if _has(fl, "-P", "--absolute-names"):
        return _danger("tar -P: пути от корня")
    archive = [w for n, w in found if n in ("-f", "--file")]
    directory = [w for n, w in found if n in ("-C", "--directory")]
    if _has(fl, "-t", "--list"):
        bad = _check_read(env, archive, "tar -t")
        return bad if bad is not None else _read("tar: список файлов архива")
    if _has(fl, "-x", "--extract", "--get"):
        bad = _check_read(env, archive, "tar -x")
        if bad is not None:
            return bad
        return _check_write(env, directory or [Word(".")], "tar -x: распаковка", edits=False)
    if _has(fl, "-c", "-r", "-u", "-A", "--create", "--append", "--update", "--concatenate"):
        # Каталоги читаются целиком (как grep -r), файлы после -C — от его каталога.
        bad = _check_read(env, directory, "tar -C")
        if bad is not None:
            return bad
        for d in [Word("."), *directory]:
            rel = os.path.relpath((env.dir / d.value).resolve(), env.root)
            bad = _check_recursive(replace(env, cwd=rel), pos, "tar -c")
            if bad is not None:
                return bad
        target = [a for a in archive if a.value != "-"]
        return _check_write(env, target, "tar -c: архив") if target else _read("tar -c в stdout")
    return _unknown("tar: режим не распознан")


def _zip_family(env: Env, name: str, args: list[Word]) -> Part:
    flags, pos = _split(args)
    fl = _flagset(flags)
    if name in ("unzip",):
        if _has(fl, "-l", "-v", "-Z", "-t", "-z", "-p"):
            bad = _check_read(env, pos[:1], name)
            return bad if bad is not None else _read("unzip: просмотр архива")
        bad = _check_read(env, pos[:1], name)
        if bad is not None:
            return bad
        return _check_write(
            env,
            _values(args, "-d", valued=("-P",)) or [Word(".")],
            "unzip: распаковка",
            edits=False,
        )
    if name == "zip":
        if _has(fl, "-T", "-TT", "--test", "--unzip-command"):
            return _danger("zip -T/-TT: запуск программы")
        if not pos:
            return _unknown("zip без архива")
        bad = _check_read(env, pos[1:], name)
        if bad is not None:
            return bad
        return _check_write(env, pos[:1], "zip: архив")
    # gzip, bzip2, xz, zstd и обратные
    if _has(fl, "-c", "--stdout", "-l", "--list", "-t", "--test", "-k", "--keep") and _has(
        fl, "-c", "--stdout", "-l", "--list", "-t", "--test"
    ):
        bad = _check_read(env, pos, name)
        return bad if bad is not None else _read(f"{name}: в stdout")
    if not pos:
        return _read(f"{name}: stdin → stdout")
    return _check_write(env, pos, f"{name}: сжатие на месте", edits=False)


_PATCH_VALUED = (
    "-i",
    "--input",
    "-d",
    "--directory",
    "-o",
    "--output",
    "-r",
    "--reject-file",
    "-p",
    "--strip",
    "-F",
    "--fuzz",
    "-D",
    "--ifdef",
    "-B",
    "--prefix",
    "-Y",
    "--basename-prefix",
    "-z",
    "--suffix",
    "-V",
    "--version-control",
    "-g",
    "--get",
    "--quoting-style",
)


def _patch(env: Env, args: list[Word]) -> Part:
    """patch: файл правки читается; пишутся файлы в каталоге -d, -o и -r (отклонённые)."""
    rest, found = _options(args, _PATCH_VALUED)
    flags, pos = _split(rest)
    fl = _flagset(flags)
    inputs = [w for n, w in found if n in ("-i", "--input")]
    bad = _check_read(env, inputs + pos[1:], "patch")
    if bad is not None:
        return bad
    if any(n in ("-B", "--prefix", "-Y", "--basename-prefix") for n, _ in found):
        return _unknown("patch -B/-Y: каталог резервных копий")
    if _has(fl, "--dry-run", "-C", "--check"):
        return _read("patch --dry-run")
    outputs = [
        w for n, w in found if n in ("-o", "--output", "-r", "--reject-file") and w.value != "-"
    ]
    dirs = [w for n, w in found if n in ("-d", "--directory")]
    written = [*(dirs or [Word(".")]), *outputs, *pos[:1]]
    return _check_write(env, written, "patch: правка файлов", edits=True)


def _dd(env: Env, args: list[Word]) -> Part:
    src = [Word(a.value[3:], a.raw, a.static, a.glob) for a in args if a.value.startswith("if=")]
    dst = [Word(a.value[3:], a.raw, a.static, a.glob) for a in args if a.value.startswith("of=")]
    # Запись — первой: `dd if=/dev/zero of=/dev/sda` опасна, каким бы ни был источник.
    written = _check_write(env, dst, "dd of=", edits=False) if dst else None
    if written is not None and written.level is CommandLevel.DANGER:
        return written
    bad = _check_read(env, src, "dd")
    if bad is not None:
        return bad
    return written if written is not None else _read("dd: в stdout")


# --------------------------------------------------------------------------- #
# Сборка, тесты, пакеты
# --------------------------------------------------------------------------- #
# Инструменты, исполняющие код проекта (тесты, сборка, линтеры): EXEC.
_EXEC_TOOLS = frozenset(
    {
        "pytest",
        "py.test",
        "mypy",
        "pyright",
        "flake8",
        "pylint",
        "bandit",
        "tox",
        "nox",
        "jest",
        "vitest",
        "mocha",
        "ava",
        "tsc",
        "eslint",
        "stylelint",
        "playwright",
        "cypress",
        "webpack",
        "vite",
        "rollup",
        "esbuild",
        "parcel",
        "turbo",
        "nx",
        "lerna",
        "gcc",
        "g++",
        "cc",
        "c++",
        "clang",
        "clang++",
        "rustc",
        "javac",
        "kotlinc",
        "scalac",
        "swiftc",
        "ninja",
        "ctest",
        "meson",
        "bazel",
        "bazelisk",
        "buck",
        "buck2",
        "sbt",
        "lein",
        "stack",
        "cabal",
        "dune",
        "mix",
        "rspec",
        "phpunit",
        "pest",
        "golangci-lint",
        "shellcheck",
        "hadolint",
        "yamllint",
        "markdownlint",
        "actionlint",
        "cppcheck",
        "clang-tidy",
        "valgrind",
        "coverage",
        "hugo",
        "mkdocs",
        "sphinx-build",
        "jekyll",
        "xcodebuild",
        "swiftlint",
        "ktlint",
        "detekt",
        "checkstyle",
        "spotbugs",
        "pmd",
        "rubocop",
        "standardrb",
        "phpstan",
        "psalm",
        "elm",
        "tsx",
        "ts-node",
        "vue-tsc",
        "svelte-check",
        "astro",
        "next",
        "nuxt",
        "ng",
        "ember",
        "deno",
        "zig",
    }
)
# Форматтеры: с флагом проверки — EXEC, без — правят файлы (WRITE, как правки).
_FORMATTERS = {
    "black": ("--check", "--diff"),
    "isort": ("--check", "--check-only", "-c", "--diff"),
    "autopep8": ("--diff",),
    "yapf": ("--diff", "-d"),
    "prettier": ("--check", "-c", "--list-different", "-l"),
    "gofmt": ("-l", "-d"),
    "goimports": ("-l", "-d"),
    "rustfmt": ("--check",),
    "clang-format": ("--dry-run", "-n"),
    "shfmt": ("-l", "-d"),
    "stylua": ("--check",),
    "biome": ("check", "lint", "ci"),
    "dprint": ("check",),
}
# Активные флаги записи форматтеров (без них gofmt и clang-format пишут в stdout).
_FORMATTER_WRITE_FLAGS = {
    "gofmt": ("-w",),
    "goimports": ("-w",),
    "clang-format": ("-i",),
    "shfmt": ("-w",),
    "prettier": ("--write", "-w"),
}
_DEPLOY = re.compile(r"deploy|publish|release|upload|prod\b|production", re.I)


def _formatter(env: Env, name: str, args: list[Word]) -> Part:
    flags, pos = _split(args)
    names = {f.value.split("=", 1)[0] for f in flags} | {p.value for p in pos[:1]}
    write_flags = _FORMATTER_WRITE_FLAGS.get(name)
    if write_flags is not None:
        if not any(n in write_flags for n in names):
            return _exec_tool(env, args, f"{name}: проверка", name)
    elif any(n in _FORMATTERS[name] for n in names):
        return _exec_tool(env, args, f"{name}: проверка", name)
    bad = _check_exec_args(env, args, name)
    if bad is not None:
        return bad
    return _write(f"{name}: форматирование файлов проекта", edits=True)


def _ruff(env: Env, args: list[Word]) -> Part:
    sub, rest = _subcommand(args)
    fl = _flagset(_split(rest)[0])
    if sub in ("--version", "version", "-V", "help", "--help"):
        return _read("ruff --version")
    if sub == "format":
        if _has(fl, "--check", "--diff"):
            return _exec_tool(env, rest, "ruff format --check", "ruff format --check")
        return _write("ruff format: форматирование файлов проекта", edits=True)
    if sub in ("check", ""):
        if _has(fl, "--fix", "--unsafe-fixes", "--add-noqa"):
            return _write("ruff check --fix: правка файлов проекта", edits=True)
        return _exec_tool(env, rest, "ruff check", "ruff check")
    if sub in ("rule", "linter", "config"):
        return _read(f"ruff {sub}: справка")
    if sub == "clean":
        return _write("ruff clean: удаление кеша", edits=False)
    return _unknown(f"ruff {sub}: подкоманда не проверена")


def _npm_scripts(env: Env) -> set[str]:
    """Имена скриптов package.json в каталоге команды."""
    try:
        data = json.loads((env.dir / "package.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    scripts = data.get("scripts") if isinstance(data, dict) else None
    return set(scripts) if isinstance(scripts, dict) else set()


_NPM_READ = frozenset(
    {
        "ls",
        "list",
        "ll",
        "la",
        "explain",
        "why",
        "config",
        "get",
        "bin",
        "root",
        "prefix",
        "help",
        "-v",
        "--version",
        "version",
        "query",
        "pkg",
        "fund",
        "doctor",
    }
)
_NPM_NET = frozenset(
    {
        "install",
        "i",
        "ci",
        "add",
        "remove",
        "rm",
        "uninstall",
        "un",
        "update",
        "up",
        "upgrade",
        "link",
        "ln",
        "unlink",
        "dedupe",
        "prune",
        "rebuild",
        "dlx",
        "exec",
        "x",
        "create",
        "init",
        "patch",
        "import",
        "fetch",
        "install-test",
        "it",
        "global",
        "dist-tag",
        "pack",
        "cache",
        "set",
        "approve-builds",
    }
)
# Только запросы к реестру — ничего не меняют, но идут в сеть.
_NPM_NET_READ = frozenset({"view", "info", "show", "search", "outdated", "audit", "v"})
_NPM_DANGER = frozenset(
    {
        "publish",
        "unpublish",
        "deprecate",
        "owner",
        "access",
        "team",
        "token",
        "login",
        "logout",
        "adduser",
        "whoami",
        "star",
        "hook",
    }
)


def _before_sub(args: list[Word], rest: list[Word]) -> list[Word]:
    """Глобальные флаги до подкоманды (`npm --script-shell=x test`): _subcommand их
    пропускает, а они меняют, что запустится, — проверяются вместе с аргументами."""
    return args[: len(args) - len(rest) - 1]


def _npm(env: Env, name: str, args: list[Word]) -> Part:
    sub, rest = _subcommand(args)
    pre = _before_sub(args, rest)
    if not sub:
        if args and args[0].static and args[0].value in ("-v", "--version"):
            return _read(f"{name} --version")
        return _unknown(f"{name}: подкоманда не распознана")
    if sub in _NPM_DANGER:
        return _danger(f"{name} {sub}: публикация или учётная запись реестра")
    if sub in ("test", "t", "tst", "start", "stop", "restart", "build"):
        signature = f"{name} {sub}" if sub != "t" else f"{name} test"
        return _exec_tool(env, [*pre, *rest], f"{name} {sub}: скрипт package.json", signature)
    if sub in ("run", "run-script", "rum", "urn"):
        script, script_args = _subcommand(rest)
        if not script:
            return _read(f"{name} run: список скриптов")
        if _DEPLOY.search(script):
            return _danger(f"{name} run {script}: похоже на публикацию")
        if script not in _npm_scripts(env):
            return _unknown(f"{name} run {script}: скрипта нет в package.json")
        if re.search(r"format|fmt|fix|prettier|lint:fix", script):
            return _write(f"{name} run {script}: правка файлов", edits=True)
        return _exec_tool(
            env,
            [*pre, *script_args],
            f"{name} run {script}: скрипт package.json",
            f"{name} run {script}",
        )
    if sub == "config":
        action, _ = _subcommand(rest)
        return (
            _read(f"{name} config {action}")
            if action in ("get", "list", "ls", "")
            else (_write(f"{name} config {action}"))
        )
    if sub in _NPM_READ:
        return _read(f"{name} {sub}: сведения о пакетах")
    if sub in _NPM_NET_READ and not (sub == "audit" and _subcommand(rest)[0] == "fix"):
        return _unknown(f"{name} {sub}: обращение к реестру пакетов")
    if sub in _NPM_NET:
        return _net(f"{name} {sub}")
    if name in ("yarn", "pnpm", "bun") and sub in _npm_scripts(env):
        if _DEPLOY.search(sub):
            return _danger(f"{name} {sub}: похоже на публикацию")
        if re.search(r"format|fmt|fix", sub):
            return _write(f"{name} {sub}: правка файлов", edits=True)
        return _exec_tool(env, [*pre, *rest], f"{name} {sub}: скрипт package.json", f"{name} {sub}")
    if name == "bun" and sub in ("test", "build"):
        return _exec_tool(env, [*pre, *rest], f"bun {sub}", f"bun {sub}")
    return _unknown(f"{name} {sub}: подкоманда не проверена")


def _local_bin(env: Env, tool: str) -> bool:
    """Инструмент установлен в node_modules/.bin (npx не пойдёт в сеть)."""
    d = env.dir.resolve()
    while True:
        if (d / "node_modules" / ".bin" / tool).exists():
            return True
        if d == env.root or env.root not in d.parents:
            return False
        d = d.parent


def _npx(env: Env, args: list[Word], classify: Callable[[Command], Part]) -> Part:
    rest = list(args)
    while rest and rest[0].static and rest[0].value.startswith("-"):
        if rest[0].value in ("-p", "--package", "-c", "--call"):
            return _unknown("npx с пакетом или командой")
        rest = rest[1:]
    if not rest or not rest[0].static:
        return _unknown("npx без команды")
    tool = rest[0].value
    if not _local_bin(env, tool):
        return _net(f"npx {tool}: пакет не установлен в проекте — будет загружен")
    sub = classify(Command(argv=rest, cwd=env.cwd))
    if sub.level is None:
        return _exec_tool(env, rest[1:], f"npx {tool}: инструмент проекта", tool)
    return sub


_CARGO = {
    "build": EXEC,
    "b": EXEC,
    "check": EXEC,
    "c": EXEC,
    "test": EXEC,
    "t": EXEC,
    "clippy": EXEC,
    "run": EXEC,
    "r": EXEC,
    "bench": EXEC,
    "doc": EXEC,
    "nextest": EXEC,
    "tree": READ,
    "metadata": READ,
    "version": READ,
    "--version": READ,
    "-V": READ,
    "locate-project": READ,
    "pkgid": READ,
    "verify-project": READ,
    "help": READ,
}
_CARGO_NET = frozenset(
    {
        "add",
        "install",
        "update",
        "fetch",
        "generate-lockfile",
        "remove",
        "rm",
        "uninstall",
        "search",
        "vendor",
        "new",
        "init",
    }
)
_CARGO_DANGER = frozenset({"publish", "yank", "login", "logout", "owner"})


def _cargo(env: Env, args: list[Word]) -> Part:
    sub, rest = _subcommand(args)
    pre = _before_sub(args, rest)  # `cargo --config=build.rustc-wrapper=… test`
    if not sub and args and args[0].static and args[0].value in ("--version", "-V"):
        return _read("cargo --version")
    if sub in _CARGO_DANGER:
        return _danger(f"cargo {sub}: публикация")
    if sub in _CARGO_NET:
        return _net(f"cargo {sub}")
    if sub == "fmt":
        if "--check" in _flagset(_split(rest)[0]):
            return _exec_tool(env, [*pre, *rest], "cargo fmt --check", "cargo fmt --check")
        return _write("cargo fmt: форматирование", edits=True)
    if sub == "fix":
        return _write("cargo fix: правка файлов", edits=True)
    if sub == "clean":
        return _write("cargo clean: удаление сборки")
    level = _CARGO.get(sub)
    if level is CommandLevel.READ:
        return _read(f"cargo {sub}: сведения")
    if level is CommandLevel.EXEC:
        canon = {"b": "build", "c": "check", "t": "test", "r": "run"}.get(sub, sub)
        return _exec_tool(
            env, [*pre, *rest], f"cargo {canon}: сборка и запуск кода", f"cargo {canon}"
        )
    return _unknown(f"cargo {sub}: подкоманда не проверена")


def _go(env: Env, args: list[Word]) -> Part:
    sub, rest = _subcommand(args)
    if sub in ("version", "env", "list", "doc", "help"):
        if sub == "env" and any(a.static and a.value == "-w" for a in rest):
            return _write("go env -w: изменение настроек")
        return _read(f"go {sub}: сведения")
    if sub in ("build", "test", "run", "vet", "tool"):
        return _exec_tool(env, rest, f"go {sub}: сборка и запуск кода", f"go {sub}")
    if sub == "fmt":
        return _write("go fmt: форматирование", edits=True)
    if sub in ("get", "install", "mod", "work"):
        action, _ = _subcommand(rest)
        if sub == "mod" and action in ("graph", "why", "verify", "edit") and action != "edit":
            return _read(f"go mod {action}")
        return _net(f"go {sub} {action}".strip())
    if sub == "generate":
        return _unknown("go generate: запускает произвольные команды из комментариев")
    if sub == "clean":
        return _write("go clean")
    return _unknown(f"go {sub}: подкоманда не проверена")


def _make_jobs(args: list[Word]) -> list[Word]:
    """У -j/-l значение необязательно: `make -j test` — цель test, `make -j 4` — четыре
    задания. Число после них склеивается с флагом (`-j4`), иначе флаг — без значения."""
    out: list[Word] = []
    i = 0
    while i < len(args):
        a = args[i]
        nxt = args[i + 1] if i + 1 < len(args) else None
        if (
            a.static
            and nxt is not None
            and nxt.static
            and re.fullmatch(r"\d+(\.\d+)?", nxt.value)
            and (
                # в `-Cj 4` j — значение -C, а не флаг
                (re.fullmatch(r"-[a-zA-Z]*[jl]", a.value) and not set(a.value[1:-1]) & set("CfoWI"))
                or a.value in ("--jobs", "--load-average")
            )
        ):
            sep = "=" if a.value.startswith("--") else ""
            out.append(Word(a.value + sep + nxt.value, a.raw))
            i += 2
            continue
        out.append(a)
        i += 1
    return out


def _make(env: Env, args: list[Word]) -> Part:
    rest, found = _options(
        _make_jobs(args),
        ("-C", "--directory", "-f", "--file", "--makefile", "-o", "-W", "-I"),
    )
    dirs = [w for n, w in found if n in ("-C", "--directory")]
    makefiles = [w for n, w in found if n in ("-f", "--file", "--makefile")]
    bad = _check_read(env, dirs + makefiles, "make")
    if bad is not None:
        return bad
    flags, pos = _split(rest)
    targets = [p.value for p in pos if p.static and "=" not in p.value]
    if any(not p.static for p in pos):
        return _unknown("make: цель известна только при выполнении")
    assigned = [p for p in pos if "=" in p.value]
    for a in assigned:
        n = a.value.split("=", 1)[0]
        if _DANGER_ENV.match(n) or n in ("SHELL", "MAKE"):
            return _danger(f"make {n}=…: подмена программ")
    bad = _check_exec_args(env, assigned, "make")  # `make test PYTHON=/tmp/x/python`
    if bad is not None:
        return bad
    # -n/-q/-p — не «только чтение»: make всё равно разбирает Makefile ($(shell …),
    # пересборка включаемых файлов) и выполняет строки рецептов с «+» и $(MAKE), а -p
    # после вывода базы собирает цели как обычно. Оценка — как у запуска make.
    if any(_DEPLOY.search(t) or t in ("install", "uninstall", "push") for t in targets):
        return _danger(f"make {' '.join(targets)}: установка или публикация")
    if targets and all(
        t in ("clean", "distclean", "mostlyclean", "maintainer-clean") for t in targets
    ):
        return _write(f"make {' '.join(targets)}: удаление сборки")
    signature = "make" + ("".join(f" {t}" for t in targets))
    if (
        not _signature_args_ok(
            "make", [a for a in rest if not (a.static and a.value in targets)], env
        )
        or dirs
        or makefiles
        or assigned  # переменные меняют рецепты: только точная команда
    ):
        return _exec(f"{signature}: цели Makefile (код проекта)", None)
    return _exec(f"{signature}: цели Makefile (код проекта)", signature)


_JVM_EXEC = frozenset(
    {
        "test",
        "build",
        "check",
        "compile",
        "compileJava",
        "compileTestJava",
        "assemble",
        "package",
        "verify",
        "jar",
        "classes",
        "testClasses",
        "javadoc",
        "run",
        "bootRun",
        "lint",
        "detekt",
        "ktlintCheck",
        "clean",
        "dependencies",
        "tasks",
        "help",
        "validate",
        "spotlessCheck",
    }
)


def _jvm(env: Env, name: str, args: list[Word]) -> Part:
    flags, pos = _split(args)
    goals = [p.value for p in pos if p.static and not p.value.startswith("-D")]
    if not goals:
        if _has(_flagset(flags), "-v", "--version", "-version"):
            return _read(f"{name} --version")
        return _unknown(f"{name}: без целей")
    if any(_DEPLOY.search(g) or g in ("install", "publishToMavenLocal") for g in goals):
        return _danger(f"{name} {' '.join(goals)}: установка или публикация")
    if all(g.split(":")[-1] in _JVM_EXEC for g in goals):
        if all(g.split(":")[-1] == "clean" for g in goals):
            return _write(f"{name} clean")
        signature = f"{name.removeprefix('./')} {' '.join(goals)}"
        return _exec(f"{name}: сборка и тесты (код проекта)", signature)
    if any(g.startswith("spotlessApply") or g == "format" for g in goals):
        return _write(f"{name}: форматирование", edits=True)
    return _unknown(f"{name} {' '.join(goals)}: цели не проверены")


def _dotnet(env: Env, args: list[Word]) -> Part:
    sub, rest = _subcommand(args)
    if sub in ("--version", "--info", "--list-sdks", "--list-runtimes") or (
        not sub and args and args[0].value in ("--version", "--info")
    ):
        return _read("dotnet: сведения")
    if sub in ("build", "test", "run", "publish", "pack", "msbuild", "vstest"):
        return _exec_tool(env, rest, f"dotnet {sub}", f"dotnet {sub}")
    if sub == "format":
        return _write("dotnet format", edits=True)
    if sub in ("add", "restore", "remove", "new", "tool", "workload"):
        return _net(f"dotnet {sub}")
    if sub == "nuget":
        return _danger("dotnet nuget: публикация пакетов")
    if sub == "clean":
        return _write("dotnet clean")
    return _unknown(f"dotnet {sub}: подкоманда не проверена")


_PY_TOOLS_NET = frozenset({"install", "uninstall", "download", "wheel"})


def _pip(env: Env, name: str, args: list[Word]) -> Part:
    sub, rest = _subcommand(args)
    if not sub:
        if args and args[0].static and args[0].value in ("--version", "-V"):
            return _read(f"{name} --version")
        return _unknown(f"{name}: подкоманда не распознана")
    if sub in ("list", "show", "freeze", "check", "inspect", "help", "debug"):
        return _read(f"{name} {sub}: сведения о пакетах")
    if sub in ("index", "search"):
        return _unknown(f"{name} {sub}: обращение к сети")
    if sub in _PY_TOOLS_NET:
        if "--dry-run" in _flagset(_split(rest)[0]):
            return _unknown(f"{name} {sub} --dry-run: обращение к сети")
        return _net(f"{name} {sub}")
    if sub == "config":
        action, _ = _subcommand(rest)
        return (
            _read(f"{name} config {action}")
            if action in ("list", "get", "debug")
            else (_write(f"{name} config {action}"))
        )
    if sub == "cache":
        return _write(f"{name} cache")
    return _unknown(f"{name} {sub}: подкоманда не проверена")


_RUNNERS = frozenset(
    {"uv", "poetry", "pipenv", "pdm", "hatch", "rye", "conda", "mamba", "micromamba", "pixi"}
)


def _py_project_tool(
    env: Env, name: str, args: list[Word], classify: Callable[[Command], Part]
) -> Part:
    sub, rest = _subcommand(args)
    pre = _before_sub(args, rest)  # `uv --directory=/другой/проект run pytest`
    if not sub:
        if args and args[0].static and args[0].value in ("--version", "-V", "version"):
            return _read(f"{name} --version")
        return _unknown(f"{name}: подкоманда не распознана")
    if (
        sub == "run"
        or (name == "hatch" and sub == "run")
        or (name in ("conda", "mamba", "micromamba") and sub == "run")
    ):
        inner = _without_values(
            rest,
            "-n",
            "--name",
            "-p",
            "--prefix",
            "--env",
            "-e",
            "--python",
            "-p",
            "--with",
            "--directory",
            "--project",
            "--extra",
            "--group",
            "--package",
        )
        while inner and inner[0].static and inner[0].value.startswith("-"):
            inner = inner[1:]
        if not inner:
            return _unknown(f"{name} run без команды")
        bad = _check_exec_args(env, [*pre, *rest[: len(rest) - len(inner)]], f"{name} run")
        if bad is not None:  # --python /tmp/x/python, --directory вне проекта
            return bad
        # --with пакет, --python …, глобальные флаги: только точная команда
        flagged = bool(pre) or len(inner) != len(rest)
        if any(
            a.static
            and a.value.split("=", 1)[0]
            in ("--with", "--with-requirements", "--with-editable", "--from")
            for a in rest
        ):
            return _net(f"{name} run --with: установка пакетов")
        sub_part = classify(Command(argv=inner, cwd=env.cwd))
        if flagged:
            sub_part = replace(sub_part, signature=None)
        # Запуск через менеджер окружения может доустановить зависимости.
        if sub_part.level is CommandLevel.READ:
            return replace(
                sub_part,
                level=EXEC,
                reason=f"{name} run: {sub_part.reason}",
                signature=None if flagged else (sub_part.signature or _first(inner)),
            )
        return sub_part
    if name == "uv" and sub == "pip":
        return _pip(env, "uv pip", rest)
    if sub == "config":
        return _py_tool_config(name, rest)
    if sub == "env":
        # poetry env use/remove, conda env create, hatch env prune — меняют окружения
        action = _subcommand(rest)[0]
        if action in ("list", "show", "info", "find") and name != "poetry":
            return _read(f"{name} env {action}: сведения")
        if name == "poetry" and action in ("list", "info"):
            return _read(f"{name} env {action}: сведения")
        return _unknown(f"{name} env {action}".strip() + ": изменение окружений")
    if sub in ("--version", "version", "tree", "show", "list", "info", "help", "check"):
        return _read(f"{name} {sub}: сведения")
    if sub in ("build",):
        return _exec(f"{name} build: сборка пакета", f"{name} build")
    if sub in ("test",) and name in ("hatch", "pdm", "pixi"):
        return _exec_tool(env, rest, f"{name} test", f"{name} test")
    if sub in ("venv", "shell"):
        return _write(f"{name} {sub}: окружение") if sub == "venv" else _unknown(f"{name} shell")
    if sub in ("publish", "upload"):
        return _danger(f"{name} {sub}: публикация пакета")
    if sub in (
        "add",
        "remove",
        "sync",
        "lock",
        "install",
        "update",
        "upgrade",
        "tool",
        "python",
        "self",
        "export",
        "create",
        "init",
        "new",
        "dlx",
        "x",
    ):
        return _net(f"{name} {sub}")
    return _unknown(f"{name} {sub}: подкоманда не проверена")


def _py_tool_config(name: str, rest: list[Word]) -> Part:
    """poetry/pdm/conda/hatch config: просмотр — чтение; ключ со значением, --add/--set/
    --unset, set/unset — запись глобальных настроек (адрес индекса, учётные данные)."""
    flags, pos = _split(rest)
    fl = _flagset(flags)
    if any(not a.static for a in rest):
        return _unknown(f"{name} config: аргумент известен только при выполнении")
    writes = _has(
        fl,
        "--add",
        "--append",
        "--prepend",
        "--set",
        "--remove",
        "--remove-key",
        "--unset",
        "-d",
        "--delete",
        "--edit",
        "-e",
    ) or (pos[:1] and pos[0].value in ("set", "unset", "update", "restore", "edit", "init"))
    if writes or len(pos) > 1:  # `poetry config ключ значение`
        return _write(f"{name} config: изменение настроек")
    return _read(f"{name} config: просмотр настроек")


def _first(words: list[Word]) -> str | None:
    return words[0].value.rsplit("/", 1)[-1] if words and words[0].static else None


# --------------------------------------------------------------------------- #
# git
# --------------------------------------------------------------------------- #
_GIT_READ = frozenset(
    {
        "status",
        "log",
        "diff",
        "show",
        "blame",
        "annotate",
        "ls-files",
        "ls-tree",
        "rev-parse",
        "describe",
        "shortlog",
        "grep",
        "cat-file",
        "rev-list",
        "merge-base",
        "show-ref",
        "show-branch",
        "count-objects",
        "whatchanged",
        "name-rev",
        "for-each-ref",
        "diff-tree",
        "diff-index",
        "diff-files",
        "check-ignore",
        "check-attr",
        "var",
        "cherry",
        "range-diff",
        "version",
        "verify-commit",
        "verify-tag",
        "fsck",
        "check-ref-format",
        "symbolic-ref",
    }
)
_GIT_WRITE = frozenset(
    {
        "add",
        "commit",
        "merge",
        "rebase",
        "cherry-pick",
        "mv",
        "init",
        "revert",
        "am",
        "apply",
        "switch",
        "tag",
        "branch",
        "stash",
        "notes",
        "worktree",
        "restore",
        "checkout",
        "reset",
        "rm",
        "config",
        "remote",
        "bisect",
        "sparse-checkout",
        "maintenance",
        "gc",
        "repack",
        "prune",
        "pull",
        "fetch",
        "clone",
        "submodule",
        "lfs",
        "format-patch",
        "archive",
        "bundle",
        "push",
        "clean",
        "filter-branch",
        "filter-repo",
        "update-ref",
        "reflog",
        "replace",
        "update-index",
        "commit-tree",
        "write-tree",
        "read-tree",
        "hash-object",
        "pack-refs",
        "mergetool",
        "difftool",
        "send-email",
        "request-pull",
        "help",
        "credential",
        "daemon",
        "instaweb",
        "gui",
        "citool",
        "gitk",
        "svn",
        "p4",
        "fast-import",
        "fast-export",
        "ls-remote",
        "remote-ext",
        "upload-pack",
        "receive-pack",
    }
)
_GIT_DANGER_CFG = re.compile(
    r"^(core\.(pager|editor|fsmonitor|hookspath|sshcommand|askpass|gitproxy|attributesfile)|"
    r"alias\.|diff\..*\.(command|textconv)|diff\.external|filter\.|credential\.|"
    r"sequence\.editor|uploadpack\.|receivepack\.|protocol\.|include|gpg\.|url\.|http\.|"
    r"remote\..*\.(url|pushurl|uploadpack|receivepack)|merge\..*\.driver|"
    r"submodule\..*\.update)",
    re.I,
)


def _git(env: Env, args: list[Word]) -> Part:
    i = 0
    while i < len(args):
        a = args[i]
        if not a.static:
            return _unknown("git: аргумент известен только при выполнении")
        v = a.value
        if not v.startswith("-"):
            break
        if v == "-c" and i + 1 < len(args):
            key = args[i + 1].value.split("=", 1)[0]
            if not args[i + 1].static or _GIT_DANGER_CFG.match(key):
                return _danger(f"git -c {key}: настройка запускает программы")
            return _unknown(f"git -c {key}: настройка на один запуск не проверена")
        if v == "-C" and i + 1 < len(args):
            target = args[i + 1]
            if scope(target, root=env.root, cwd=env.cwd) is not Scope.INSIDE:
                return _unknown(f"git -C {target.raw}: каталог вне проекта")
            env = replace(env, cwd=os.path.relpath((env.dir / target.value).resolve(), env.root))
            i += 2
            continue
        if v in ("--no-pager", "-P", "--no-optional-locks", "--version"):
            if v in ("--version",):
                return _read("git --version")
            i += 1
            continue
        return _unknown(f"git {v}: глобальный флаг не проверен")
    if i >= len(args):
        return _read("git: справка")
    sub, rest = args[i].value, args[i + 1 :]
    if any(not w.static for w in rest):
        dynamic_ok = sub in _GIT_READ and all(w.static or not _looks_like_path(w.raw) for w in rest)
        if not dynamic_ok:
            return _unknown(f"git {sub}: аргумент известен только при выполнении")
    values = [w.value for w in rest if w.static]
    fl = _flagset([w for w in rest if w.static and w.value.startswith("-")])
    hit = _abbrev(
        rest,
        "--output",
        "--ext-diff",
        "--textconv",
        "--exec",
        "--upload-pack",
        "--receive-pack",
        "--open-files-in-pager",
        "--git-dir",
        "--work-tree",
        "--exec-path",
        "--output-directory",
        "--config-env",
        "--template",
    )
    if hit is not None and hit.value.split("=", 1)[0] not in ("--output", "--output-directory"):
        return _unknown(f"git {sub} {hit.value}: запуск программ или чужие каталоги")
    if hit is not None and hit.value.split("=", 1)[0] != "--output" and sub != "format-patch":
        return _unknown(f"git {sub} {hit.value}: сокращённый флаг")
    for v in values:
        name = v.split("=", 1)[0]
        if (
            name in ("--output", "-o")
            and sub not in ("worktree", "grep")  # у git grep -o — «только совпадения»
            or (name == "--output-directory" and sub == "format-patch")
        ):
            target = [Word(v.split("=", 1)[1])] if "=" in v else []
            if not target:
                idx = values.index(v)
                target = [Word(values[idx + 1])] if idx + 1 < len(values) else []
            return _check_write(env, target, f"git {sub} --output", edits=False)
        if name in (
            "--ext-diff",
            "--exec",
            "--upload-pack",
            "--receive-pack",
            "--open-files-in-pager",
            "-O",
            "--textconv",
        ) or (name.startswith("--exec-path") or name in ("--git-dir", "--work-tree")):
            return _unknown(f"git {sub} {name}: запуск программ")
    if sub == "grep":
        # -O[пейджер] в группе (`-nOid`): найденные файлы открываются программой
        for v in values:
            if v.startswith("-") and not v.startswith("--"):
                for ch in v[1:]:
                    if ch == "O":
                        return _unknown(f"git grep {v}: файлы открываются программой")
                    if ch in "efABCm":  # дальше — значение флага
                        break
    # Файлы, которые читаются по значению флага: шаблоны grep, содержимое blame.
    if sub in ("grep", "blame"):
        files = (
            _values(rest, "-f", valued=("-e", "-A", "-B", "-C", "-m", "--threads"))
            if sub == "grep"
            else _values(rest, "--contents", valued=("-L", "-S"))
        )
        bad = _check_read(env, files, f"git {sub}")
        if bad is not None:
            return bad
    if sub in _GIT_READ:
        # git diff с путём вне репозитория сравнивает файлы как --no-index (`git diff
        # ~/.ssh/id_rsa /dev/null`), поэтому пути проверяются и без флага.
        if sub == "diff" or (sub == "grep" and "--no-index" in fl):
            paths = [Word(v) for v in values if not v.startswith("-")]
            bad = _check_read(env, paths if sub == "diff" else paths[1:], f"git {sub}")
            if bad is not None:
                return bad
        if sub == "fsck" and any(v.startswith("--lost-found") for v in values):
            return _write("git fsck --lost-found")
        if sub == "symbolic-ref" and len([v for v in values if not v.startswith("-")]) > 1:
            return _write("git symbolic-ref: изменение ссылки")
        return _git_read_after_check(env, f"git {sub}: чтение репозитория")
    return _git_change(env, sub, rest, values, fl)


def _git_change(env: Env, sub: str, rest: list[Word], values: list[str], fl: set[str]) -> Part:
    positional = [v for v in values if not v.startswith("-")]
    if sub == "branch":
        if _has(fl, "-D") or _has(fl, "-d", "--delete") and _has(fl, "--force", "-f"):
            return _danger("git branch -D: удаление ветки без проверки слияния")
        if _has(fl, "-f", "--force", "-M", "-C"):  # перезапись существующей ветки
            return _danger("git branch -f: перезапись ветки")
        if _has(
            fl,
            "-d",
            "--delete",
            "-m",
            "-M",
            "--move",
            "-c",
            "-C",
            "--copy",
            "-f",
            "--force",
            "--set-upstream-to",
            "-u",
            "--unset-upstream",
            "--edit-description",
        ):
            return _write("git branch: изменение веток")
        if not positional or _has(
            fl,
            "-l",
            "--list",
            "--contains",
            "--merged",
            "--no-merged",
            "--points-at",
            "--show-current",
        ):
            return _git_read_after_check(env, "git branch: список веток")
        return _write("git branch: новая ветка")
    if sub == "tag":
        if _has(fl, "-d", "--delete", "-f", "--force"):
            return _danger("git tag -d/-f: удаление или перезапись тега")
        if not positional or _has(fl, "-l", "--list", "--contains", "--points-at", "-n"):
            return _git_read_after_check(env, "git tag: список тегов")
        return _write("git tag: новый тег")
    if sub == "stash":
        action = positional[0] if positional else "push"
        if action in ("list", "show"):
            return _git_read_after_check(env, f"git stash {action}")
        if action in ("drop", "clear"):
            return _danger(f"git stash {action}: удаление отложенных правок")
        return _write(f"git stash {action}")
    if sub == "remote":
        action = positional[0] if positional else ""
        if action in ("", "show", "get-url") or values == ["-v"] or values == ["--verbose"]:
            if action == "show":
                return _unknown("git remote show: обращение к сети")
            return _git_read_after_check(env, "git remote: список")
        return _danger(f"git remote {action}: изменение адресов репозитория")
    if sub == "config":
        if _has(fl, "--global", "--system", "--file", "-f", "--blob"):
            return _danger("git config --global/--system/--file")
        if _has(
            fl,
            "--get",
            "--get-all",
            "--get-regexp",
            "--list",
            "-l",
            "--show-origin",
            "--show-scope",
            "--get-urlmatch",
            "--name-only",
        ) or (
            len(positional) == 1
            and not _has(
                fl,
                "--unset",
                "--unset-all",
                "--add",
                "--replace-all",
                "--remove-section",
                "--rename-section",
                "-e",
                "--edit",
            )
        ):
            return _read("git config: чтение настроек")
        key = positional[0] if positional else ""
        if positional[:1] in (["get"], ["list"]):
            return _read("git config: чтение настроек")
        if _GIT_DANGER_CFG.match(key) or (
            positional[:1] == ["set"]
            and len(positional) > 1
            and _GIT_DANGER_CFG.match(positional[1])
        ):
            return _danger(f"git config {key}: настройка запускает программы")
        return _write(f"git config {key}")
    if sub == "reset":
        if _has(fl, "--hard", "--merge", "--keep"):
            return _danger("git reset --hard: потеря незакоммиченных правок")
        return _write("git reset")
    if sub in ("checkout", "restore"):
        if (
            _has(
                fl,
                "-f",
                "--force",
                "--source",
                "-s",
                "--ours",
                "--theirs",
                "--merge",
                "-m",
                "--patch",
                "-p",
                "--overlay",
                "--no-overlay",
            )
            and sub == "restore"
            or ("--" in values and sub == "checkout")
            or (sub == "restore" and not _has(fl, "--staged", "-S") and positional)
            or (sub == "checkout" and _has(fl, "-f", "--force", "-B"))
        ):
            return _danger(f"git {sub}: незакоммиченные правки будут потеряны")
        if sub == "checkout" and positional and any((env.dir / p).exists() for p in positional):
            return _danger("git checkout <файл>: незакоммиченные правки будут потеряны")
        if sub == "restore":
            return _write("git restore --staged")
        return _write("git checkout: переключение ветки")
    if sub == "clean":
        if _has(fl, "-n", "--dry-run") and not _has(fl, "-f", "--force"):
            return _read("git clean -n: показ файлов")
        return _danger("git clean: удаление неотслеживаемых файлов")
    if sub == "push":
        if _has(
            fl,
            "-f",
            "--force",
            "--force-with-lease",
            "--force-if-includes",
            "--mirror",
            "-d",
            "--delete",
            "--prune",
        ) or any(p.startswith("+") or p.startswith(":") for p in positional):
            return _danger("git push с перезаписью или удалением на сервере")
        return _write("git push: публикация коммитов в удалённый репозиторий")
    if sub in (
        "filter-branch",
        "filter-repo",
        "update-ref",
        "replace",
        "commit-tree",
        "write-tree",
        "read-tree",
        "update-index",
        "fast-import",
        "send-email",
        "daemon",
        "instaweb",
        "credential",
        "remote-ext",
        "upload-pack",
        "receive-pack",
    ):
        return _danger(f"git {sub}: низкоуровневая или внешняя операция")
    if sub == "reflog":
        if positional[:1] in (["expire"], ["delete"]):
            return _danger(f"git reflog {positional[0]}: удаление истории ссылок")
        return _git_read_after_check(env, "git reflog")
    if sub in ("gc", "prune", "repack", "maintenance"):
        if any(v.startswith("--prune") for v in values) or sub == "prune":
            return _danger(f"git {sub}: удаление недостижимых объектов")
        return _write(f"git {sub}")
    if sub == "worktree":
        action = positional[0] if positional else ""
        if action == "list":
            return _git_read_after_check(env, "git worktree list")
        if action in ("remove", "prune"):
            return _danger(f"git worktree {action}")
        targets = [Word(p) for p in positional[1:2]]
        return (
            _check_write(env, targets, f"git worktree {action}", edits=False)
            if targets
            else _write(f"git worktree {action}")
        )
    if sub == "bisect":
        if positional[:1] in (["log"], ["visualize"], ["view"]):
            return (
                _git_read_after_check(env, "git bisect log")
                if positional[0] == "log"
                else _unknown("git bisect visualize")
            )
        if positional[:1] == ["run"]:
            return _unknown("git bisect run: запуск команд")
        return _write("git bisect")
    if sub == "submodule" and "foreach" in positional:
        return _unknown("git submodule foreach: запуск команд")
    if sub == "rebase" and _has(fl, "-x", "--exec"):
        return _unknown("git rebase -x: запуск команд")
    if sub == "switch" and _has(fl, "-f", "--force", "--discard-changes", "-C", "--force-create"):
        return _danger("git switch -f/-C: незакоммиченные правки или ветка будут потеряны")
    if sub == "bundle" and positional[:1] == ["create"]:
        return _check_write(env, [Word(p) for p in positional[1:2]], "git bundle create")
    if sub in ("fetch", "pull", "clone", "submodule", "lfs", "ls-remote"):
        if sub == "clone":
            dest = [Word(p) for p in positional[1:2]]
            if dest:
                bad = _check_write(env, dest, "git clone")
                if bad.level is CommandLevel.DANGER:
                    return bad
        return _write(f"git {sub}: обращение к удалённому репозиторию")
    if sub in ("format-patch", "archive", "bundle"):
        return _write(f"git {sub}: создание файлов")
    if sub in ("rm",):
        if _has(fl, "--cached"):
            return _write("git rm --cached")
        if _has(fl, "-r", "-f", "--force"):
            return _danger("git rm -r/-f: удаление файлов")
        return _write("git rm: удаление файлов")
    if sub in (
        "add",
        "commit",
        "merge",
        "rebase",
        "cherry-pick",
        "mv",
        "init",
        "revert",
        "am",
        "apply",
        "switch",
        "notes",
        "sparse-checkout",
    ):
        if sub == "commit" and _has(fl, "--amend") and _has(fl, "--no-edit") is False:
            pass
        if sub == "apply" and _has(fl, "--check", "--stat", "--numstat", "--summary"):
            return _git_read_after_check(env, "git apply --check")
        if sub == "notes" and positional[:1] in ([], ["list"], ["show"]):
            return _git_read_after_check(env, "git notes: чтение")
        return _write(f"git {sub}")
    if sub in (
        "help",
        "mergetool",
        "difftool",
        "gui",
        "citool",
        "gitk",
        "svn",
        "p4",
        "fast-export",
        "hash-object",
        "pack-refs",
        "request-pull",
    ):
        return _unknown(f"git {sub}: не проверено")
    return _unknown(f"git {sub}: подкоманда не проверена")


def _git_read_after_check(env: Env, reason: str) -> Part:
    """Чтение git — чтение, только если git найдёт репозиторий проекта (а не вложенный,
    со своими настройками) и в его настройках нет программ."""
    from gigacrew.project.files import looks_like_git_dir
    from gigacrew.safety.gitconfig import repo_config_runs_programs

    d = env.dir.resolve()
    while d != env.root and env.root in d.parents:
        if (d / ".git").exists() or looks_like_git_dir(d):
            return _unknown(
                f"git: вложенный репозиторий ({os.path.relpath(d, env.root)}) со своими настройками"
            )
        d = d.parent
    hazard = repo_config_runs_programs(env.dir)
    if hazard is None:
        return _unknown("git: не удалось проверить настройки репозитория")
    if hazard:
        return _unknown(
            "git: в настройках репозитория есть программы (textconv, "
            "fsmonitor, фильтры) — чтение запустит их"
        )
    return _read(reason)


# --------------------------------------------------------------------------- #
# Интерпретаторы
# --------------------------------------------------------------------------- #
_PY_FLAGS_NOARG = set("IPEsSuBOqbdvxRih")
_PY_MODULE_LEVELS: dict[str, tuple[CommandLevel | None, str]] = {
    "pytest": (EXEC, "pytest"),
    "unittest": (EXEC, "unittest"),
    "mypy": (EXEC, "mypy"),
    "pylint": (EXEC, "pylint"),
    "flake8": (EXEC, "flake8"),
    "coverage": (EXEC, "coverage"),
    "tox": (EXEC, "tox"),
    "nox": (EXEC, "nox"),
    "build": (EXEC, "python -m build"),
    "doctest": (EXEC, "doctest"),
    "timeit": (EXEC, "timeit"),
    "pyright": (EXEC, "pyright"),
    "ruff": (EXEC, "ruff"),
    "black": (EXEC, "black"),
    "isort": (EXEC, "isort"),
    "cProfile": (EXEC, "cProfile"),
    "profile": (EXEC, "profile"),
    "trace": (EXEC, "trace"),
    "pdb": (None, "pdb"),
    "venv": (WRITE, "venv"),
    "virtualenv": (WRITE, "virtualenv"),
    "compileall": (WRITE, "compileall"),
    "py_compile": (WRITE, "py_compile"),
    "http.server": (DANGER, "http.server"),
    "smtpd": (DANGER, "smtpd"),
    "ensurepip": (WRITE, "ensurepip"),
    "pydoc": (None, "pydoc"),
    "site": (READ, "site"),
    "platform": (READ, "platform"),
    "sysconfig": (READ, "sysconfig"),
    "this": (READ, "this"),
    "tokenize": (READ, "tokenize"),
    "ast": (READ, "ast"),
    "dis": (READ, "dis"),
    "tabnanny": (READ, "tabnanny"),
    "pyclbr": (READ, "pyclbr"),
    "pickletools": (READ, "pickletools"),
    "calendar": (READ, "calendar"),
}
# Стандартные строки .pth (virtualenv, setuptools: distutils и editable-установка) —
# целиком: код, дописанный после разрешённого начала, исполнялся бы при каждом запуске.
_VENV_PTH_OK = re.compile(
    r"(import _virtualenv|import _distutils_hack(; _distutils_hack\.add_shim\(\))?|"
    r"import (__editable___\w+_finder); \3\.install\(\)|"
    r"import os; var = 'SETUPTOOLS_USE_DISTUTILS'; "
    r"enabled = os\.environ\.get\(var, '(local|stdlib)'\) == 'local'; "
    r"enabled and __import__\('_distutils_hack'\)\.add_shim\(\); ?)\s*"
)
# Модули, которые site импортирует при старте из любого каталога sys.path.
_STARTUP_MODULES = ("sitecustomize", "usercustomize")


def _startup_module(directory: Path) -> str | None:
    """sitecustomize/usercustomize (файл или пакет) в каталоге."""
    try:
        entries = os.listdir(directory)
    except OSError:
        return None
    for entry in entries:
        if entry.partition(".")[0] in _STARTUP_MODULES:
            return entry
    return None


def _interpreter_ok(env: Env, prog: str) -> Part | None:
    """Интерпретатор внутри проекта (venv): его .pth-файлы исполняются при каждом запуске.

    None — интерпретатор вне проекта или venv без подозрительных .pth.
    """
    path = env.program_path or _which(env, prog)
    if path is None:
        return _unknown(f"{prog}: интерпретатор не найден")
    lexical = Path(path)
    in_project = lexical == env.root or env.root in lexical.parents or _inside(env, lexical)
    if path == "builtin" or not in_project:
        return None
    venv = _venv_root(lexical)
    if venv is None:
        return Part(EXEC, f"{prog}: интерпретатор из файлов проекта")
    for site in venv.glob("lib*/python*/site-packages"):
        for pth in site.glob("*.pth"):
            try:
                lines = pth.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                return Part(EXEC, f"{prog}: .pth окружения не прочитан")
            for line in lines:
                if line.startswith(("import ", "import\t")) and not _VENV_PTH_OK.fullmatch(line):
                    return Part(EXEC, f"{prog}: {pth.name} исполняет код при запуске")
        startup = _startup_module(site)
        if startup is not None:
            return Part(EXEC, f"{prog}: {startup} в окружении исполняется при запуске")
    return None


def _venv_root(interpreter: Path) -> Path | None:
    for d in interpreter.parents:
        if (d / "pyvenv.cfg").is_file():
            return d
    return None


def _python(env: Env, cmd: Command, prog: str, args: list[Word]) -> Part:
    from gigacrew.safety.python import analyze, shadowed_stdlib

    code: str | None = None
    script: Path | None = None
    module: str | None = None
    module_args: list[Word] = []
    script_args: list[Word] = []
    # -I: без PYTHONPATH и без каталога запуска в sys.path; -P — только без каталога
    # запуска (PYTHONPATH действует); -E — только без PYTHONPATH; -S — без site.
    no_path0 = ignore_env = no_site = False
    i = 0
    while i < len(args):
        a = args[i]
        if not a.static:
            return _unknown(f"{prog}: аргумент известен только при выполнении")
        v = a.value
        if v in ("--version", "-V", "-VV", "--help", "-h"):
            return _read(f"{prog} --version")
        if v == "-":
            break
        if v.startswith("-") and not v.startswith("--"):
            letters = v[1:]
            j = 0
            while j < len(letters):
                ch = letters[j]
                if ch == "I":
                    no_path0 = ignore_env = True
                no_path0 = no_path0 or ch == "P"
                ignore_env = ignore_env or ch == "E"
                no_site = no_site or ch == "S"
                if ch == "c":
                    code = letters[j + 1 :] or (args[i + 1].value if i + 1 < len(args) else "")
                    if not letters[j + 1 :]:
                        if i + 1 >= len(args) or not args[i + 1].static:
                            return _unknown(f"{prog} -c: код известен только при выполнении")
                        i += 1
                    break
                if ch == "m":
                    module = letters[j + 1 :] or (args[i + 1].value if i + 1 < len(args) else "")
                    if not letters[j + 1 :]:
                        i += 1
                    module_args = args[i + 1 :]
                    break
                if ch in "WX":
                    if not letters[j + 1 :]:
                        i += 1
                    break
                if ch == "i":
                    return _unknown(f"{prog} -i: интерактивный режим")
                if ch not in _PY_FLAGS_NOARG:
                    return _unknown(f"{prog} -{ch}: флаг не проверен")
                j += 1
            if code is not None or module is not None:
                break
            i += 1
            continue
        if v.startswith("--"):
            if v in ("--check-hash-based-pycs",):
                i += 2
                continue
            return _unknown(f"{prog} {v}: флаг не проверен")
        script = Path(v)
        script_args = args[i + 1 :]
        break
    # Интерпретатор и затенение модулей
    trust = _interpreter_ok(env, prog)
    if trust is not None and trust.level is None:
        return trust
    path_dirs, bad = _pythonpath_dirs(env, cmd)
    if bad is not None:
        return bad
    if ignore_env:
        path_dirs = ()
    for d in () if no_site else path_dirs:
        startup = _startup_module(d)
        if startup is not None:  # site импортирует его при старте из PYTHONPATH
            found = Part(EXEC, f"{prog}: {startup} в PYTHONPATH исполняется при запуске")
            trust = found if trust is None else _combine(trust, found)
            break
    if module is not None:
        if module == "pip":
            return _combine(_pip(env, f"{prog} -m pip", module_args), trust)
        known = _PY_MODULE_LEVELS.get(module)
        if known is not None:
            level, signature = known
            if level is CommandLevel.EXEC:
                return _exec_tool(env, module_args, f"python -m {module}", signature)
            if level is CommandLevel.READ:
                if module_args:
                    bad = _check_read(
                        env,
                        [w for w in module_args if not w.value.startswith("-")],
                        f"python -m {module}",
                    )
                    if bad is not None:
                        return bad
                return _combine(_read(f"python -m {module}"), trust)
            if level is None:
                return _unknown(f"python -m {module}")
            return Part(level, f"python -m {module}")
        if module == "json.tool":
            files = [w for w in module_args if not w.value.startswith("-")]
            bad = _check_read(env, files[:1], "python -m json.tool")
            if bad is not None:
                return bad
            if len(files) > 1:
                return _check_write(env, files[1:2], "python -m json.tool: файл вывода")
            return _combine(_read("python -m json.tool"), trust)
        if (env.dir / Path(*module.split("."))).with_suffix(".py").exists() or (
            env.dir / Path(*module.split(".")) / "__main__.py"
        ).exists():
            return _exec_tool(
                env, module_args, f"python -m {module}: модуль проекта", f"python -m {module}"
            )
        return _unknown(f"python -m {module}: модуль не проверен")
    fingerprint = None
    if code is None:
        if script is not None:
            target = script if script.is_absolute() else env.dir / script
            if scope(Word(str(script)), root=env.root, cwd=env.cwd) is not Scope.INSIDE:
                return _unknown(f"{prog} {script}: скрипт вне проекта")
            try:
                data = target.read_bytes()
            except OSError:
                return _unknown(f"{prog} {script}: файл не прочитан")
            if len(data) > 256 * 1024:
                return _exec(f"{prog} {script}: скрипт проекта", None)
            code = data.decode("utf-8", errors="replace")
            fingerprint = hashlib.sha256(data).hexdigest()[:12]
            bad = _check_exec_args(env, script_args, f"{prog} {script}")
            if bad is not None:
                return bad
        else:
            code = _stdin_code(cmd)
            if code is None:
                return _unknown(f"{prog}: код приходит со stdin (известен только при выполнении)")
    result = analyze(
        code,
        root=env.root,
        cwd=env.cwd,
        script=(env.dir / script).resolve() if script is not None else None,
        path_dirs=path_dirs,
    )
    part = Part(result.level, result.reason)
    # Без каталога запуска в sys.path затенение в нём не действует — но только если
    # стандартную библиотеку не затеняет и каталог из PYTHONPATH.
    if (
        no_path0
        and result.level is CommandLevel.EXEC
        and "подменит модуль" in result.reason
        and not any(shadowed_stdlib(d) for d in path_dirs)
    ):
        part = Part(READ, "python: только чтение и вычисления")
    if script is not None and part.level in (CommandLevel.EXEC, CommandLevel.READ):
        rel = os.path.relpath((env.dir / script).resolve(), env.root)
        part = replace(part, signature=f"python {rel}#{fingerprint}")
    return _combine(part, trust) if trust is not None else part


def _pythonpath_dirs(env: Env, cmd: Command) -> tuple[tuple[Path, ...], Part | None]:
    """Каталоги PYTHONPATH (из окружения и префикса команды) внутри проекта."""
    values = [env.environ.get("PYTHONPATH", "")]
    for name, word in cmd.assigns:
        if name == "PYTHONPATH":
            if not word.static:
                return (), _unknown("PYTHONPATH известен только при выполнении")
            values.append(word.value)
    dirs: list[Path] = []
    for value in values:
        for entry in value.split(os.pathsep):
            if not entry:
                continue
            p = Path(entry) if Path(entry).is_absolute() else env.dir / entry
            if _inside(env, p):
                dirs.append(p.resolve())
            elif value is not values[0]:
                return (), _unknown(f"PYTHONPATH={entry}: каталог вне проекта")
    return tuple(dirs), None


def _stdin_code(cmd: Command) -> str | None:
    """Код, который интерпретатор прочитает со stdin: heredoc или статичный echo/cat."""
    doc = cmd.heredoc
    if doc is not None:
        return doc.body if doc.body_static else None
    src = cmd.stdin
    if src is None or src.name is None or not src.name.static:
        return None
    name = src.name.value
    if (
        name == "echo"
        and all(a.static for a in src.args)
        and not any(a.value.startswith("-") for a in src.args)
    ):
        return " ".join(a.value for a in src.args)
    if (
        name == "printf"
        and len(src.args) == 1
        and src.args[0].static
        and "%" not in (src.args[0].value)
        and "\\" not in src.args[0].value
    ):
        return src.args[0].value
    if name == "cat" and not src.args and src.heredoc is not None and src.heredoc.body_static:
        return src.heredoc.body
    return None


_SHELLS = frozenset({"sh", "bash", "zsh", "dash", "ksh", "ash", "fish", "mksh", "busybox"})


def _shell(env: Env, cmd: Command, prog: str, args: list[Word]) -> Part:
    flags = []
    i = 0
    while i < len(args) and args[i].static and args[i].value.startswith(("-", "+")):
        flags.append(args[i].value)
        if args[i].value in ("-o", "+o", "-O", "+O"):
            i += 1
        i += 1
    rest = args[i:]
    if any(f in ("-n",) for f in flags) and not any("c" in f.lstrip("-") for f in flags):
        return _read(f"{prog} -n: проверка синтаксиса")
    if any(f.lstrip("-+").count("c") and not f.startswith("--") for f in flags):
        if not rest:
            return _unknown(f"{prog} -c без команды")
        if not rest[0].static:
            return _danger(f"{prog} -c: команда известна только при выполнении")
        if env.nested is None or env.depth >= 3:
            return _unknown(f"{prog} -c: слишком глубокая вложенность")
        return env.nested(rest[0].value, replace(env, depth=env.depth + 1))
    if rest:
        script = rest[0]
        if scope(script, root=env.root, cwd=env.cwd) is not Scope.INSIDE:
            return _unknown(f"{prog} {script.raw}: скрипт вне проекта")
        if _DEPLOY.search(script.value):
            return _danger(f"{prog} {script.raw}: похоже на развёртывание или публикацию")
        return _exec_tool(
            env, rest[1:], f"{prog} {script.raw}: скрипт проекта", f"{prog} {script.value}"
        )
    # Команды со stdin: heredoc или конвейер.
    doc = cmd.heredoc
    if doc is not None and doc.body_static and env.nested is not None and env.depth < 3:
        return env.nested(doc.body, replace(env, depth=env.depth + 1))
    if cmd.stdin is not None or doc is not None:
        return _danger(f"передача команд в {prog} через stdin")
    return _unknown(f"{prog} без команды")


def _source(env: Env, args: list[Word]) -> Part:
    if not args or not args[0].static:
        return _unknown("source: файл известен только при выполнении")
    target = env.dir / args[0].value
    if scope(args[0], root=env.root, cwd=env.cwd) is not Scope.INSIDE:
        return _unknown(f"source {args[0].raw}: файл вне проекта")
    if target.name in ("activate", "activate.sh") and _venv_root(target) is not None:
        return Part(
            EXEC,
            "активация виртуального окружения проекта",
            signature="venv-activate",
            prelude=True,
        )
    return _exec(f"source {args[0].raw}: скрипт проекта", None)


# --------------------------------------------------------------------------- #
# Система, сеть, контейнеры, облака
# --------------------------------------------------------------------------- #
_DANGER_PROGS = frozenset(
    {
        "sudo",
        "su",
        "doas",
        "pkexec",
        "runuser",
        "chroot",
        "nsenter",
        "unshare",
        "setpriv",
        "mkfs",
        "fdisk",
        "sfdisk",
        "parted",
        "gparted",
        "mount",
        "umount",
        "swapon",
        "swapoff",
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
        "init",
        "telinit",
        "kexec",
        "insmod",
        "rmmod",
        "modprobe",
        "iptables",
        "ip6tables",
        "nft",
        "ufw",
        "firewall-cmd",
        "pfctl",
        "useradd",
        "userdel",
        "usermod",
        "groupadd",
        "groupdel",
        "passwd",
        "chpasswd",
        "visudo",
        "chattr",
        "setfacl",
        "shred",
        "wipe",
        "srm",
        "killall",
        "at",
        "batch",
        "curl",
        "wget",
        "nc",
        "ncat",
        "netcat",
        "socat",
        "telnet",
        "ftp",
        "sftp",
        "scp",
        "ssh",
        "tftp",
        "lftp",
        "aria2c",
        "httpie",
        "http",
        "https",
        "xh",
        "ngrok",
        "cloudflared",
        "apt",
        "apt-get",
        "aptitude",
        "dpkg",
        "yum",
        "dnf",
        "rpm",
        "pacman",
        "apk",
        "zypper",
        "snap",
        "flatpak",
        "port",
        "twine",
        "gem",
        "osascript",
        "defaults",
        "launchctl",
        "sysctl",
        "csrutil",
        "diskutil",
        "tmutil",
        "networksetup",
        "scutil",
        "dscl",
        "mkfs.ext4",
        "mkfs.xfs",
        "mkfs.vfat",
        "wipefs",
        "blkdiscard",
        "hdparm",
    }
)
_NETWORK_READ = frozenset(
    {
        "ping",
        "dig",
        "nslookup",
        "host",
        "traceroute",
        "tracepath",
        "whois",
        "mtr",
        "nmap",
        "ip",
        "ifconfig",
        "netstat",
        "ss",
        "arp",
        "route",
        "iwconfig",
        "nmcli",
        "resolvectl",
    }
)
_MUTATING_VERBS = re.compile(
    r"^(apply|create|delete|del|rm|remove|destroy|deploy|edit|patch|replace|scale|rollout|"
    r"exec|port-forward|cp|drain|cordon|uncordon|taint|label|annotate|install|uninstall|"
    r"upgrade|rollback|set|put|post|update|terminate|stop|start|restart|reboot|kill|run|"
    r"import|push|login|logout|attach|debug|expose|autoscale|certificate|undo|purge|wipe|"
    r"flush|truncate|drop|reset|init|plan|refresh|taint|untaint|state|workspace|sync|"
    r"invoke|publish|release|send|mv|move|copy|sign|revoke|enable|disable)$",
    re.I,
)


def _remote_cli(env: Env, name: str, args: list[Word]) -> Part:
    """kubectl, helm, terraform, облака, gh: чтение — обращение к сети («не знаю»),
    изменение — опасно."""
    words = [a.value for a in args if a.static and not a.value.startswith("-")]
    if name in ("gh", "glab", "hub"):
        return _gh(name, args, words)
    # У kubectl/helm/terraform глагол — первое слово; у облаков — после сервиса и ресурса.
    depth = 3 if name in ("aws", "gcloud", "az", "gsutil", "bq", "doctl") else 1
    for w in words[:depth]:
        if _MUTATING_VERBS.match(w):
            if name == "terraform" and w in ("plan", "init"):
                return (
                    _write(f"terraform {w}: провайдеры и сеть")
                    if w == "init"
                    else (_unknown("terraform plan: обращение к облаку"))
                )
            return _danger(f"{name} {w}: изменение удалённой инфраструктуры")
    if args and args[0].static and args[0].value in ("--version", "version", "-v"):
        return _read(f"{name} --version")
    return _unknown(f"{name}: обращение к удалённому сервису")


_GH_READ_ACTIONS = frozenset(
    {"list", "view", "status", "diff", "checks", "show", "watch", "ls", "search", "browse-print"}
)


def _gh(name: str, args: list[Word], words: list[str]) -> Part:
    """gh/glab: просмотр — обращение к сети («не знаю»), изменение — опасно."""
    group = words[0] if words else ""
    action = words[1] if len(words) > 1 else ""
    if group in ("--version", "version") or (args and args[0].value == "--version"):
        return _read(f"{name} --version")
    if group == "api":
        writes = any(
            a.static
            and a.value.split("=", 1)[0]
            in ("-X", "--method", "-f", "-F", "--field", "--raw-field", "--input")
            for a in args
        )
        return (
            _danger(f"{name} api: запрос с изменением")
            if writes
            else _unknown(f"{name} api: обращение к серверу")
        )
    if group in (
        "auth",
        "secret",
        "variable",
        "ssh-key",
        "gpg-key",
        "config",
        "extension",
        "alias",
    ) and action not in ("status", "list", "ls", "get"):
        return _danger(f"{name} {group} {action}: учётные данные и настройки")
    if action and action not in _GH_READ_ACTIONS:
        return _danger(f"{name} {group} {action}: изменение на сервере")
    return _unknown(f"{name} {group} {action}: обращение к серверу".strip())


# Короткие флаги docker run со значением: в группе `-itv /:/h` значение берёт последняя.
_DOCKER_VALUED = ("-e", "-p", "-w", "-u", "-l", "-h", "-m", "-c", "--name", "--env")


def _docker(env: Env, name: str, args: list[Word]) -> Part:
    sub, rest = _subcommand(args)
    if sub == "compose" or name in ("docker-compose", "podman-compose"):
        action, more = _subcommand(rest) if sub == "compose" else (sub, rest)
        if action in ("ps", "ls", "images", "top", "version", "port"):
            return _read(f"{name} compose {action}")
        if action in ("logs",):
            return _read(f"{name} compose logs")
        if action == "down" and any(a.value in ("-v", "--volumes", "--rmi") for a in more):
            return _danger("docker compose down -v: удаление томов")
        if action in ("config",):
            return _unknown("docker compose config: выводит переменные (секреты)")
        return _unknown(f"{name} compose {action}: управление контейнерами")
    if sub in ("ps", "images", "image", "container", "volume", "network") and _subcommand(rest)[
        0
    ] in ("ls", "list", "inspect", ""):
        return _read(f"{name} {sub}: список")
    if sub in (
        "ps",
        "images",
        "logs",
        "inspect",
        "version",
        "info",
        "top",
        "port",
        "diff",
        "history",
        "stats",
        "search",
    ):
        if sub == "stats" and not any(a.value == "--no-stream" for a in rest):
            return _read(f"{name} stats")
        return _read(f"{name} {sub}: сведения")
    if sub in (
        "push",
        "login",
        "logout",
        "system",
        "prune",
        "rmi",
        "rm",
        "kill",
        "volume",
        "network",
        "swarm",
        "service",
        "stack",
        "secret",
        "trust",
        "plugin",
        "context",
    ):
        return _danger(f"{name} {sub}: удаление, публикация или настройка")
    if sub in ("run", "exec", "create"):
        risky = [
            a.value
            for a in rest
            if a.static
            and (
                a.value.startswith(
                    (
                        "--privileged",
                        "--pid=host",
                        "--network=host",
                        "--net=host",
                        "--cap-add",
                        "--security-opt",
                        "--device",
                        "--userns=host",
                        "-v/",
                        "--volume=/",
                        "--mount",
                    )
                )
                or a.value.startswith("/:")
            )
        ]
        vols = _values(rest, "-v", "--volume", valued=_DOCKER_VALUED)
        if risky or any(
            v.value.startswith(("/", "~")) and not v.value.startswith(str(env.root)) for v in vols
        ):
            return _danger(f"{name} {sub}: доступ к хосту из контейнера")
        return _unknown(f"{name} {sub}: запуск контейнера")
    if sub in ("build", "pull", "buildx"):
        return _write(f"{name} {sub}: образ (сеть)")
    if sub in ("start", "stop", "restart", "pause", "unpause", "cp", "tag", "rename", "update"):
        return _write(f"{name} {sub}: управление контейнером")
    return _unknown(f"{name} {sub}: подкоманда не проверена")


_DB_DANGER = re.compile(
    r"\b(drop|delete|truncate|alter|update|insert|flushall|flushdb|"
    r"dropdatabase|remove|grant|revoke)\b",
    re.I,
)


def _db(env: Env, name: str, args: list[Word]) -> Part:
    text = " ".join(a.value for a in args)
    if _DB_DANGER.search(text):
        return _danger(f"{name}: изменение данных базы")
    return _unknown(f"{name}: обращение к базе данных")


# --------------------------------------------------------------------------- #
# Разрешение имени программы
# --------------------------------------------------------------------------- #
_SYSTEM_DIRS = (
    "/bin/",
    "/usr/bin/",
    "/usr/local/bin/",
    "/sbin/",
    "/usr/sbin/",
    "/opt/homebrew/bin/",
    "/usr/local/sbin/",
)


def _inside(env: Env, path: Path) -> bool:
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError):
        return True  # не знаем — считаем, что внутри (осторожнее)
    return resolved == env.root or env.root in resolved.parents


_which_cache: dict[tuple[str, str, str], str | None] = {}


def _which(env: Env, prog: str) -> str | None:
    """Где оболочка найдёт программу (PATH дочернего процесса, относительные элементы —
    от каталога команды). ``"builtin"`` — встроенная команда оболочки."""
    if prog in _DASH_BUILTINS:
        return "builtin"
    path = env.environ.get("PATH", os.defpath)
    key = (prog, path, str(env.dir))
    if key in _which_cache:
        return _which_cache[key]
    entries = []
    for entry in path.split(os.pathsep):
        p = Path(entry) if entry and Path(entry).is_absolute() else env.dir / (entry or ".")
        entries.append(str(p))
    found = shutil.which(prog, path=os.pathsep.join(entries))
    result = str(Path(found).resolve()) if found else None
    if len(_which_cache) > 512:
        _which_cache.clear()
    _which_cache[key] = result
    return result


def _program(env: Env, word: Word) -> tuple[str, Part | None]:
    """Каноническое имя программы. Вторая часть — оценка, если путь к программе особый
    (файл проекта, неизвестный каталог)."""
    value = word.value
    if "/" not in value:
        return value, None
    if value.startswith("~"):  # оболочка раскроет в домашний каталог — вне проекта
        return value, _unknown(f"{value}: программа вне проекта")
    lexical = Path(os.path.normpath(env.dir / value if not value.startswith("/") else value))
    # Системный каталог — после нормализации: `/usr/bin/../../проект/ls` — файл проекта.
    if str(lexical).startswith(_SYSTEM_DIRS) and not _inside(env, lexical):
        return lexical.name, None
    if lexical == env.root or env.root in lexical.parents:
        venv = _venv_root(lexical.parent) if lexical.parent.name in ("bin", "Scripts") else None
        if venv is not None:
            return lexical.name, None  # .venv/bin/pytest — инструмент окружения проекта
        if not _inside(env, lexical):
            return value, _unknown(f"{value}: ссылка на программу вне проекта")
        if _DEPLOY.search(lexical.name):
            return value, _danger(f"{value}: похоже на развёртывание или публикацию")
        # Сигнатура — с хешем содержимого: изменённый скрипт снова спросит.
        return value, _exec(f"{value}: программа из файлов проекта", _file_signature(lexical))
    return value, _unknown(f"{value}: программа вне проекта")


def _file_signature(path: Path) -> str | None:
    """«путь#хеш» исполняемого файла проекта; None — не прочитать или слишком большой."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) > 1024 * 1024:
        return None
    return f"{path.name}#{hashlib.sha256(data).hexdigest()[:12]}"


# Обёртки, которые только запускают вложенную команду (её и оцениваем).
_WRAPPERS = frozenset({"time", "nice", "nohup", "stdbuf", "timeout", "command", "exec"})


def _unwrap(name: str, args: list[Word]) -> list[Word] | None:
    """Аргументы-команда обёртки (``timeout 10 pytest`` → ``pytest``); None — не обёртка."""
    rest = list(args)
    if name == "timeout":
        while rest and rest[0].static and rest[0].value.startswith("-"):
            if rest[0].value in ("-s", "--signal", "-k", "--kill-after") and len(rest) > 1:
                rest = rest[2:]
            else:
                rest = rest[1:]
        return rest[1:]  # длительность
    if name == "nice":
        while rest and rest[0].static and rest[0].value.startswith("-"):
            rest = rest[2:] if rest[0].value in ("-n", "--adjustment") else rest[1:]
        return rest
    if name == "stdbuf":
        while rest and rest[0].static and rest[0].value.startswith("-"):
            rest = rest[1:] if len(rest[0].value) > 2 else rest[2:]
        return rest
    if name == "time":
        while rest and rest[0].static and rest[0].value.startswith("-"):
            rest = rest[2:] if rest[0].value in ("-o", "-f") else rest[1:]
        return rest
    return rest


def _env_options(args: list[Word]) -> list[Word] | None:
    """Аргументы env после её опций (присваивания и команда); None — опция не проверена.

    env разбирает опции через getopt и останавливается на первом не-флаге: в группе
    ``-iu NAME`` буква ``u`` берёт значение — остаток слова или следующее слово, и команда
    начинается позже. ``-C`` (каталог), ``-S`` (разбиение строки на команду), ``-a``
    (подмена argv[0]), BSD ``-P`` (свой PATH) и незнакомые опции — «не знаю».
    """
    rest = list(args)
    while rest and rest[0].static and rest[0].value.startswith("-"):
        v = rest[0].value
        rest = rest[1:]
        if v == "--":
            break
        if v == "-":  # то же, что -i
            continue
        if v.startswith("--"):
            if v in ("--ignore-environment", "--null", "--debug"):
                continue
            if v.startswith("--unset="):
                continue
            if v == "--unset" and rest:
                rest = rest[1:]
                continue
            return None  # --chdir, --split-string, сигналы, сокращения
        for k, ch in enumerate(v[1:], 1):
            if ch in "i0v":
                continue
            if ch == "u":
                if k + 1 == len(v):  # значение — следующее слово
                    if not rest:
                        return None
                    rest = rest[1:]
                break
            return None
    return rest


def classify_simple(cmd: Command, env: Env) -> Part:
    """Оценка простой команды (без перенаправлений — они у вызывающего)."""
    reading_env = True
    if not cmd.argv:
        return _read("присваивание переменных")
    name_word = cmd.argv[0]
    if not name_word.static:
        return _danger("имя программы получается при выполнении — разбор не видит, что запустится")
    prog, special = _program(env, name_word)
    if "/" in name_word.value and special is None:
        env = replace(env, program_path=str(Path(os.path.normpath(env.dir / name_word.value))))
    args = cmd.argv[1:]

    def classify(inner: Command) -> Part:
        return classify_simple(
            Command(
                argv=inner.argv,
                assigns=cmd.assigns,
                redirects=inner.redirects,
                cwd=inner.cwd,
                stdin=inner.stdin or cmd.stdin,
            ),
            env,
        )

    if special is not None:
        part = special
        if args and part.signature:
            part = replace(part, signature=None)  # скрипт проекта с аргументами — точно
    else:
        part = _dispatch(env, cmd, prog, args, classify)
    # Шаблон имён может раскрыться в имя файла, похожее на флаг (``-delete``, ``--pre=x``).
    if part.level is not CommandLevel.DANGER:
        injected = _glob_injection(env, args)
        if injected is not None:
            return _unknown(f"{prog}: шаблон {injected} раскрывается в имя, похожее на флаг")
    # Префиксные присваивания NAME=value cmd
    if part.signature and any(not _SAFE_ENV.match(n) for n, _ in cmd.assigns):
        part = replace(part, signature=None)
    for name, value in cmd.assigns:
        bad = check_env(name, value, reading=reading_env and part.level is CommandLevel.READ)
        if bad is not None:
            part = (
                _combine(part, bad)
                if bad.level is CommandLevel.DANGER
                else (bad if part.level is CommandLevel.READ else part)
            )
    # Подмена программы через PATH: найденная внутри проекта — код проекта.
    if part.level is CommandLevel.READ and special is None and prog not in _DASH_BUILTINS:
        found = _which(env, prog)
        # Не найдена — команда просто не запустится (код 127).
        if found is not None and found != "builtin" and _inside(env, Path(found)):
            return _exec(f"{prog}: найдена в файлах проекта ({found})", None)
    return part


def _glob_injection(env: Env, args: list[Word]) -> str | None:
    """Шаблон среди аргументов, который совпадает с файлом, чьё имя начинается с «-»."""
    import glob as globmod

    for a in args:
        if not (a.static and a.glob):
            continue
        try:
            matches = globmod.glob(a.value, root_dir=env.dir)
        except (OSError, ValueError):
            return a.raw
        if any(Path(m).name.startswith("-") or m.startswith("-") for m in matches):
            return a.raw
    return None


def _dispatch(
    env: Env, cmd: Command, prog: str, args: list[Word], classify: Callable[[Command], Part]
) -> Part:
    if prog in _WRAPPERS:
        if prog == "command" and args and args[0].static and args[0].value in ("-v", "-V"):
            return _read("command -v")
        if prog == "exec" and not args:
            return _unknown("exec без команды")
        inner = _unwrap(prog, args)
        if not inner:
            return _read(f"{prog}")
        return classify(Command(argv=inner, cwd=env.cwd, stdin=cmd.stdin, redirects=cmd.redirects))
    if prog == "env":
        rest = _env_options(args)
        if rest is None:
            return _unknown("env: опция меняет каталог, программу или разбор команды")
        assigns: list[tuple[str, Word]] = []
        while (
            rest and rest[0].static and "=" in rest[0].value and not rest[0].value.startswith("-")
        ):
            n, _, val = rest[0].value.partition("=")
            assigns.append((n, Word(val)))
            rest = rest[1:]
        if not rest:
            return _unknown("env без команды выводит всё окружение (там бывают секреты)")
        inner = Command(argv=rest, assigns=[*cmd.assigns, *assigns], cwd=env.cwd, stdin=cmd.stdin)
        return classify_simple(inner, env)
    if prog in ("busybox",) and args:
        return classify(Command(argv=args, cwd=env.cwd, stdin=cmd.stdin))
    if prog in ("cd", "pushd", "popd"):
        return _unknown(f"{prog} посреди команды: каталог меняется для следующих частей")
    if prog in ("set",):
        if not args:
            return _unknown("set без аргументов выводит переменные")
        if all(
            a.static
            and re.fullmatch(
                r"[-+][euxvfCnhbmBHPo]+|pipefail|errexit|nounset|"
                r"xtrace|noclobber|--",
                a.value,
            )
            for a in args
        ):
            return _read("set: режимы оболочки")
        return _unknown("set с аргументами")
    if prog in ("source", "."):
        return _source(env, args)
    if prog == "eval":
        return _danger("eval: команда собирается при выполнении — разбор её не видит")
    if prog in (
        "trap",
        "alias",
        "unalias",
        "builtin",
        "hash",
        "ulimit",
        "umask",
        "export",
        "unset",
        "readonly",
        "local",
        "declare",
        "typeset",
        "coproc",
        "fc",
        "history",
        "bind",
        "complete",
        "enable",
        "disown",
        "suspend",
        "fg",
        "bg",
        "watch",
        "script",
        "screen",
        "tmux",
        "setsid",
        "parallel",
        "expect",
        "ionice",
        "chrt",
        "taskset",
        "flock",
        "unbuffer",
        "chronic",
        "caffeinate",
        "noglob",
        "direnv",
        "mise",
        "asdf",
        "devbox",
        "nix-shell",
        "nix",
        "xdg-open",
        "open",
        "start",
        "man",
        "info",
        "help",
        "vim",
        "vi",
        "nvim",
        "nano",
        "emacs",
        "ed",
        "ex",
        "code",
        "subl",
        "gedit",
        "mc",
        "ranger",
        "nnn",
        "htop",
        "btop",
        "top",
        "iotop",
        "lazygit",
        "tig",
        "fzf",
    ):
        if prog == "top" and any(a.static and re.match(r"^-[a-zA-Z]*b", a.value) for a in args):
            return _read("top -b")
        if prog == "ulimit" and (
            not args
            or all(a.static and a.value in ("-a", "-n", "-s", "-u", "-v", "-H", "-S") for a in args)
        ):
            return _read("ulimit: просмотр лимитов")
        if prog == "umask" and not args:
            return _read("umask")
        if prog in ("vim", "vi", "nvim", "ex") and _editor_shell_command(args):
            return _danger(f"{prog}: команда редактора запускает оболочку (:!)")
        return _unknown(f"{prog}: интерактивная или меняющая оболочку команда")
    if prog in _READ_FILES:
        return _simple_read(env, prog, args)
    if prog in _NO_FILES:
        return _no_files(env, prog, args)
    if prog in ("grep", "egrep", "fgrep", "zgrep", "ack", "ag", "git-grep"):
        if prog in ("ack", "ag"):
            _, pos = _split(args)
            bad = _check_recursive(env, pos[1:], prog)
            return bad if bad is not None else _read(f"{prog}: поиск")
        return _grep(env, prog, args)
    if prog in ("rg", "ripgrep"):
        return _rg(env, prog, args)
    if prog == "find":
        return _find(env, prog, args, classify)
    if prog in ("fd", "fdfind"):
        return _fd(env, prog, args, classify)
    if prog == "xargs":
        return _xargs(env, cmd, args, classify)
    if prog in ("sed", "gsed"):
        return _sed(env, args)
    if prog in ("awk", "gawk", "mawk", "nawk"):
        return _awk(env, args)
    if prog in ("jq", "yq", "gojq", "jaq"):
        return _jq(env, prog, args)
    if prog == "sort":
        return _sort(env, args)
    if prog == "uniq":
        return _uniq(env, args)
    if prog == "xxd":
        return _xxd(env, args)
    if prog in ("base64",):
        return _base64(env, args)
    if prog == "ps":
        return _ps(args)
    if prog == "tr":
        bad = _check_vars(env, args, "tr")
        return bad if bad is not None else _read("tr")
    if prog == "tee":
        return _tee(env, args)
    if prog in ("less", "more", "most"):
        return _less(env, prog, args)
    if prog == "rm":
        return _rm(env, args)
    if prog in ("cp", "mv"):
        return _cp_mv(env, prog, args)
    if prog in (
        "mkdir",
        "touch",
        "rmdir",
        "truncate",
        "install",
        "dos2unix",
        "unix2dos",
        "mkfifo",
        "fallocate",
        "split",
        "csplit",
        "patch",
        "iconv",
        "trash",
        "trash-put",
        "unlink",
    ):
        if prog == "patch":
            return _patch(env, args)
        if prog == "iconv":
            rest, found = _options(args, ("-f", "-t", "--from-code", "--to-code", "-o", "--output"))
            out = [w for n, w in found if n in ("-o", "--output")]
            _, pos = _split(rest)
            bad = _check_read(env, pos, "iconv")
            if bad is not None:
                return bad
            return _check_write(env, out, "iconv -o") if out else _read("iconv")
        if prog in ("rmdir", "unlink", "trash", "trash-put"):
            _, pos = _split(args)
            if not pos:
                return _unknown(f"{prog} без аргументов")
            return _check_write(env, pos, f"{prog}: удаление", edits=False)
        return _simple_write(env, prog, args)
    if prog == "ln":
        return _ln(env, args)
    if prog in ("chmod",):
        return _chmod(env, args)
    if prog in ("chown", "chgrp"):
        if any(a.static and a.value in ("-R", "--recursive") for a in args):
            return _danger(f"{prog} -R")
        _, pos = _split(args)
        return _check_write(env, pos[1:], prog, edits=False) if len(pos) > 1 else _unknown(prog)
    if prog == "tar":
        return _tar(env, args)
    if prog in (
        "zip",
        "unzip",
        "gzip",
        "gunzip",
        "bzip2",
        "bunzip2",
        "xz",
        "unxz",
        "zstd",
        "unzstd",
        "lz4",
        "7z",
        "7za",
    ):
        if prog in ("7z", "7za"):
            sub, _ = _subcommand(args)
            if sub in ("l", "t"):
                return _read(f"{prog} {sub}")
            return _write(f"{prog} {sub}: архив")
        return _zip_family(env, prog, args)
    if prog == "dd":
        return _dd(env, args)
    if prog in ("kill", "pkill"):
        if prog == "pkill" or any(
            a.static and a.value in ("-9", "-KILL", "-SIGKILL", "-1", "1", "0", "-0") for a in args
        ):
            return _danger(f"{prog}: принудительное завершение процессов")
        return _write("kill: завершение процесса")
    if prog in _DANGER_PROGS or prog.startswith("mkfs."):
        return _danger(f"{prog}: системная, сетевая или привилегированная команда")
    if prog in _NETWORK_READ:
        return _unknown(f"{prog}: сетевая диагностика")
    if prog in ("systemctl", "service", "journalctl", "crontab", "launchctl"):
        words = [a.value for a in args if a.static]
        if (
            prog == "journalctl"
            or (
                prog == "systemctl"
                and words[:1]
                in (["status"], ["list-units"], ["is-active"], ["is-enabled"], ["show"], ["cat"])
            )
            or (prog == "service" and "status" in words)
            or (prog == "crontab" and words == ["-l"])
        ):
            return _read(f"{prog}: просмотр")
        return _danger(f"{prog}: изменение служб или расписаний")
    if prog in ("openssl",):
        sub, _ = _subcommand(args)
        if sub in ("version", "rand", "help", "list", "ciphers"):
            return _read(f"openssl {sub}")
        return _unknown(f"openssl {sub}: ключи, сертификаты или сеть")
    if prog == "rsync":
        _, pos = _split(args)
        if any(":" in p.value for p in pos if p.static) or any(
            a.value.startswith(("--rsh", "-e", "--rsync-path")) for a in args if a.static
        ):
            return _danger("rsync на удалённый хост")
        if any(a.value.startswith(("--delete", "--remove-source-files")) for a in args if a.static):
            return _danger("rsync --delete: удаление файлов")
        bad = _check_read(env, pos[:-1], "rsync")
        if bad is not None:
            return bad
        return _check_write(env, pos[-1:], "rsync", edits=False)
    if prog in ("git",):
        return _git(env, args)
    if prog in ("python", "python3", "python2", "pypy", "pypy3", "ipython", "py") or re.fullmatch(
        r"python3\.\d+t?|pypy3\.\d+", prog
    ):
        part = _python(env, cmd, prog, args)
        if prog == "ipython" and part.level is CommandLevel.READ:
            return replace(part, level=EXEC, reason="ipython: файлы автозапуска профиля")
        return part
    if prog in _SHELLS:
        if prog == "busybox":
            return _unknown("busybox")
        return _shell(env, cmd, prog, args)
    if prog == "deno":
        return _deno(env, args)
    if prog in (
        "node",
        "ruby",
        "perl",
        "php",
        "lua",
        "luajit",
        "Rscript",
        "R",
        "julia",
        "java",
        "kotlin",
        "scala",
        "groovy",
        "elixir",
        "erl",
        "swift",
        "ts-node",
        "tsx",
        "osascript",
        "pwsh",
        "powershell",
        "tclsh",
        "guile",
        "racket",
        "ghc",
        "runghc",
        "dart",
        "go-run",
    ):
        return _script_runner(env, prog, args, classify)
    if prog in _FORMATTERS:
        return _formatter(env, prog, args)
    if prog == "ruff":
        return _ruff(env, args)
    if prog in ("npm", "yarn", "pnpm", "bun"):
        return _npm(env, prog, args)
    if prog in ("npx", "pnpx", "bunx"):
        return _npx(env, args, classify)
    if prog == "cargo":
        return _cargo(env, args)
    if prog == "go":
        return _go(env, args)
    if prog in ("make", "gmake", "mingw32-make"):
        return _make(env, args)
    if prog in ("mvn", "./mvnw", "mvnw", "gradle", "./gradlew", "gradlew"):
        return _jvm(env, prog, args)
    if prog == "dotnet":
        return _dotnet(env, args)
    if prog in ("pip", "pip3", "pipx"):
        if prog == "pipx":
            sub, _ = _subcommand(args)
            return _read("pipx list") if sub in ("list", "--version") else _net(f"pipx {sub}")
        return _pip(env, prog, args)
    if prog in _RUNNERS:
        return _py_project_tool(env, prog, args, classify)
    if prog in ("bundle",):
        sub, rest = _subcommand(args)
        if sub == "exec":
            return classify(Command(argv=rest, cwd=env.cwd))
        if sub in ("list", "show", "info", "outdated", "check", "version"):
            return _read(f"bundle {sub}")
        return _net(f"bundle {sub}")
    if prog in (
        "rake",
        "rails",
        "composer",
        "artisan",
        "mix",
        "sbt",
        "lein",
        "stack",
        "cabal",
        "dune",
        "swift",
        "zig",
        "flutter",
        "dart-cli",
        "elm",
        "deno",
    ):
        return _build_cli(env, prog, args)
    if prog in ("cmake",):
        if any(a.static and a.value in ("--install", "-P") for a in args):
            return _unknown("cmake --install/-P")
        if any(a.static and a.value == "--build" for a in args):
            return _exec_tool(env, args, "cmake --build", "cmake --build")
        if any(a.static and a.value in ("--version", "--help", "-E") for a in args):
            if any(a.value == "-E" for a in args if a.static):
                return _unknown("cmake -E: команды файловой системы")
            return _read("cmake --version")
        return _exec_tool(env, args, "cmake: настройка сборки", "cmake")
    if prog in _EXEC_TOOLS:
        if (
            args
            and args[0].static
            and args[0].value in ("--version", "-V", "-v", "version")
            and len(args) == 1
        ):
            return _read(f"{prog} --version")
        if prog in ("eslint", "stylelint", "rubocop", "standardrb", "swiftlint", "ktlint") and any(
            a.static
            and a.value in ("--fix", "-a", "-A", "--autocorrect", "-F", "--format")
            and a.value != "--format"
            for a in args
        ):
            return _write(f"{prog} --fix: правка файлов", edits=True)
        if prog == "coverage":
            sub, _ = _subcommand(args)
            if sub in ("report", "debug"):
                return _read("coverage report")
            if sub in ("erase", "html", "xml", "json", "lcov", "combine", "annotate"):
                return _write(f"coverage {sub}: файлы отчёта", edits=True)
        if prog in ("tox", "nox") and any(
            a.static and a.value in ("-l", "--list", "--listenvs") for a in args
        ):
            return _read(f"{prog} --list")
        if prog in ("deno",):
            return _build_cli(env, prog, args)
        return _exec_tool(env, args, f"{prog}: проверка или сборка (код проекта)", prog)
    if prog in (
        "kubectl",
        "helm",
        "oc",
        "terraform",
        "tofu",
        "pulumi",
        "aws",
        "gcloud",
        "az",
        "gsutil",
        "bq",
        "doctl",
        "flyctl",
        "fly",
        "heroku",
        "vercel",
        "netlify",
        "firebase",
        "serverless",
        "sls",
        "sam",
        "cdk",
        "gh",
        "glab",
        "hub",
        "ansible",
        "ansible-playbook",
        "vagrant",
        "minikube",
        "kind",
        "k3d",
        "skaffold",
        "tilt",
        "eksctl",
        "nomad",
        "consul",
        "vault",
        "packer",
        "wrangler",
        "supabase",
        "railway",
        "render",
        "doppler",
    ):
        return _remote_cli(env, prog, args)
    if prog in ("docker", "podman", "docker-compose", "podman-compose", "nerdctl"):
        return _docker(env, prog, args)
    if prog in (
        "psql",
        "mysql",
        "mariadb",
        "sqlite3",
        "redis-cli",
        "mongo",
        "mongosh",
        "clickhouse-client",
        "cqlsh",
        "duckdb",
        "pg_dump",
        "mysqldump",
        "pg_restore",
    ):
        if prog in ("sqlite3", "duckdb"):
            _, pos = _split(args)
            if pos and pos[0].static and scope(pos[0], root=env.root, cwd=env.cwd) is Scope.INSIDE:
                sql = " ".join(p.value for p in pos[1:])
                if (
                    sql
                    and not _DB_DANGER.search(sql)
                    and not re.search(
                        r"\.(shell|system|output|once|import|load|backup|save|restore|open|"
                        r"read|cd|excel|log|dump|clone|recover)\b|\battach\b|\bvacuum\b",
                        sql,
                        re.I,
                    )
                    and re.match(
                        r"^\s*(select|\.tables|\.schema|pragma|with|"
                        r"explain)\b",
                        sql,
                        re.I,
                    )
                ):
                    return _read(f"{prog}: запрос на чтение")
        return _db(env, prog, args)
    return _unknown(f"{prog}: программа не известна классификатору")


def _editor_shell_command(args: list[Word]) -> bool:
    """vim -c ':!cmd', --cmd, +':!cmd' или system(): команда оболочки при запуске."""
    for i, a in enumerate(args):
        if not a.static:
            continue
        command = None
        if a.value.startswith("+"):
            command = a.value[1:]
        elif a.value in ("-c", "--cmd") and i + 1 < len(args):
            command = args[i + 1].value
        if command is not None and re.search(r"!|\bsystem\s*\(|\bterm", command):
            return True
    return False


def _deno(env: Env, args: list[Word]) -> Part:
    sub, rest = _subcommand(args)
    if not sub:
        if args and args[0].static and args[0].value in ("--version", "-V", "-v"):
            return _read("deno --version")
        return _unknown("deno без подкоманды")
    if sub in ("test", "lint", "check", "bench", "doc", "info", "coverage"):
        return _exec_tool(env, rest, f"deno {sub}", f"deno {sub}")
    if sub == "fmt":
        if any(a.static and a.value == "--check" for a in rest):
            return _exec_tool(env, rest, "deno fmt --check", "deno fmt --check")
        return _write("deno fmt: форматирование", edits=True)
    if sub in ("run", "task", "serve"):
        return _exec(f"deno {sub}: код проекта", None)
    if sub in ("eval", "repl"):
        return _unknown(f"deno {sub}: код в командной строке")
    if sub in (
        "install",
        "add",
        "remove",
        "cache",
        "upgrade",
        "compile",
        "outdated",
        "uninstall",
        "init",
        "publish",
        "jupyter",
    ):
        return _danger("deno publish") if sub == "publish" else _net(f"deno {sub}")
    return _unknown(f"deno {sub}: подкоманда не проверена")


def _script_runner(
    env: Env, prog: str, args: list[Word], classify: Callable[[Command], Part]
) -> Part:
    if (
        args
        and args[0].static
        and args[0].value in ("--version", "-v", "-V", "-version", "version")
    ):
        return _read(f"{prog} --version")
    flags, pos = _split(args)
    for f in flags:
        if f.value in (
            "-e",
            "-E",
            "--eval",
            "-p",
            "--print",
            "-r",
            "-c",
            "-x",
            "-pe",
            "-ne",
            "-pi",
            "-i",
            "-n",
            "--command",
            "-Command",
        ) or f.value.startswith(("-e", "--eval", "-pi", "-r")):
            return _unknown(f"{prog} {f.value}: код в командной строке не проверяется")
    if not pos:
        return _unknown(f"{prog}: код со stdin")
    script = pos[0]
    if prog == "java" and any(a.value in ("-jar",) for a in flags):
        script = _values(args, "-jar")[0] if _values(args, "-jar") else script
    if scope(script, root=env.root, cwd=env.cwd) is not Scope.INSIDE:
        return _unknown(f"{prog} {script.raw}: скрипт вне проекта или неизвестен")
    if _DEPLOY.search(script.value):
        return _danger(f"{prog} {script.raw}: похоже на развёртывание")
    return _exec_tool(env, pos[1:], f"{prog} {script.raw}: код проекта", f"{prog} {script.value}")


_BUILD_CLI_EXEC = frozenset(
    {
        "test",
        "build",
        "compile",
        "check",
        "lint",
        "analyze",
        "bench",
        "spec",
        "doc",
        "docs",
        "typecheck",
        "run",
        "exec",
        "task",
        "ci",
        "verify",
    }
)


def _build_cli(env: Env, prog: str, args: list[Word]) -> Part:
    sub, rest = _subcommand(args)
    if not sub:
        if args and args[0].static and args[0].value in ("--version", "-v", "-V", "version"):
            return _read(f"{prog} --version")
        return _unknown(f"{prog}: без подкоманды")
    if sub in (
        "--version",
        "version",
        "help",
        "info",
        "doctor",
        "list",
        "show",
        "outdated",
        "deps",
        "tree",
    ):
        return _read(f"{prog} {sub}")
    if prog in ("rake", "rails") and sub.startswith("db:"):
        return (
            _danger(f"{prog} {sub}: изменение базы данных")
            if re.search(r"drop|reset|purge|truncate|rollback", sub)
            else _write(f"{prog} {sub}: миграции")
        )
    if prog == "composer" and sub in (
        "install",
        "require",
        "update",
        "remove",
        "create-project",
        "global",
        "self-update",
    ):
        return _net(f"composer {sub}")
    if prog == "mix" and sub in ("deps.get", "deps.update", "local.hex", "archive.install"):
        return _net(f"mix {sub}")
    if prog in ("flutter", "dart-cli") and sub in ("pub", "upgrade", "create"):
        return _net(f"{prog} {sub}")
    if sub in ("fmt", "format"):
        return _write(f"{prog} {sub}: форматирование", edits=True)
    if _DEPLOY.search(sub) or sub in ("install", "push"):
        return _danger(f"{prog} {sub}: установка или публикация")
    if sub in _BUILD_CLI_EXEC or (
        prog in ("rake", "mix", "sbt", "lein") and sub in ("test", "spec", "compile", "default")
    ):
        return _exec_tool(env, rest, f"{prog} {sub}: код проекта", f"{prog} {sub}")
    return _unknown(f"{prog} {sub}: подкоманда не проверена")
