"""Статический анализ Python-кода из команды: ``python -c``, heredoc, ``python script.py``.

Модели часто проверяют что-то однострочником на Python. Код не исполняется — разбирается
модулем ``ast`` и сверяется с **белыми списками**: модули стандартной библиотеки без
побочных эффектов, безопасные встроенные функции, атрибуты модулей ``os``/``sys``/
``pathlib``. Чтение файлов — только по путям, которые заведомо внутри проекта и не
секреты: литерал, ``Path.cwd()``, ``__file__`` скрипта и обход каталогов проекта
(``rglob``, ``os.walk``) — при условии, что в обходимом дереве нет секретов и ссылок
наружу.

Итог — уровень (:class:`~gigacrew.safety.levels.CommandLevel`) или «не знаю» с причиной:

* READ — только чтение внутри проекта и вычисления;
* EXEC — импорт модулей проекта или сторонних пакетов (их код исполнится), модуль
  стандартной библиотеки затенён файлом проекта (``json.py`` рядом);
* WRITE — запись/удаление отдельных файлов внутри проекта;
* DANGER — процессы, сеть, рекурсивное удаление, запись вне проекта, чтение чужих секретов;
* None — динамический код (``eval``, ``getattr``, ``__dunder__``), пути, известные только
  при выполнении, незнакомые модули стандартной библиотеки.

Анализ потоконечувствительный: проверяется всё дерево, включая тела функций и классов,
которые могут и не выполниться, — ложная тревога лучше пропуска.
"""

from __future__ import annotations

import ast
import builtins
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from gigacrew.safety.levels import CommandLevel as Level
from gigacrew.safety.levels import worst

__all__ = ["PyResult", "analyze", "shadowed_stdlib"]

MAX_SOURCE = 256 * 1024  # больше — не разбираем

# Модули без побочных эффектов (подмодули — только перечисленные).
SAFE_MODULES = frozenset(
    {
        "abc",
        "argparse",
        "array",
        "ast",
        "base64",
        "binascii",
        "bisect",
        "calendar",
        "cmath",
        "collections",
        "collections.abc",
        "colorsys",
        "configparser",
        "copy",
        "csv",
        "dataclasses",
        "datetime",
        "decimal",
        "difflib",
        "enum",
        "errno",
        "fnmatch",
        "fractions",
        "functools",
        "glob",
        "graphlib",
        "hashlib",
        "heapq",
        "hmac",
        "html",
        "html.parser",
        "ipaddress",
        "itertools",
        "json",
        "json.decoder",
        "json.encoder",
        "keyword",
        "locale",
        "math",
        "numbers",
        "platform",
        "pprint",
        "random",
        "re",
        "reprlib",
        "secrets",
        "shlex",
        "statistics",
        "string",
        "struct",
        "sysconfig",
        "textwrap",
        "time",
        "token",
        "tokenize",
        "tomllib",
        "typing",
        "unicodedata",
        "urllib.parse",
        "uuid",
        "warnings",
        "zlib",
        "zoneinfo",
        # с белыми списками атрибутов (_MODULE_ATTRS)
        "os",
        "os.path",
        "posixpath",
        "sys",
        "pathlib",
        "shutil",
        "io",
        "codecs",
        "importlib.metadata",
    }
)
# Процессы, сеть, загрузка кода, низкоуровневый доступ.
DANGER_MODULES = frozenset(
    {
        "subprocess",
        "socket",
        "ssl",
        "http",
        "http.client",
        "http.server",
        "urllib",
        "urllib.request",
        "ftplib",
        "smtplib",
        "telnetlib",
        "poplib",
        "imaplib",
        "xmlrpc",
        "socketserver",
        "ctypes",
        "pty",
        "multiprocessing",
        "webbrowser",
        "requests",
        "httpx",
        "aiohttp",
        "urllib3",
        "paramiko",
        "fabric",
        "pexpect",
        "sh",
        "plumbum",
        "psutil",
        # загрузка и выполнение кода в обход разбора
        "pickle",
        "marshal",
        "shelve",
        "dill",
        "cloudpickle",
        "builtins",
        "gc",
        "runpy",
        "code",
        "codeop",
        "docker",
        "boto3",
        "pycurl",
    }
)
# Атрибуты модулей с белым списком; остальное — «не знаю» (или DANGER — _DANGER_ATTRS).
_MODULE_ATTRS: dict[str, frozenset[str]] = {
    "os": frozenset(
        {
            "getcwd",
            "getcwdb",
            "listdir",
            "scandir",
            "walk",
            "stat",
            "lstat",
            "path",
            "sep",
            "linesep",
            "pathsep",
            "curdir",
            "pardir",
            "extsep",
            "altsep",
            "name",
            "cpu_count",
            "getpid",
            "getppid",
            "fspath",
            "fsencode",
            "fsdecode",
            "get_terminal_size",
            "uname",
            "getloadavg",
            "devnull",
            "PathLike",
            "DirEntry",
            "strerror",
            "urandom",
            "access",
            "R_OK",
            "W_OK",
            "X_OK",
            "F_OK",
            "remove",
            "unlink",
            "rename",
            "replace",
            "mkdir",
            "makedirs",
            "rmdir",
            "removedirs",
        }
    ),
    "sys": frozenset(
        {
            "version",
            "version_info",
            "platform",
            "executable",
            "argv",
            "maxsize",
            "byteorder",
            "implementation",
            "prefix",
            "base_prefix",
            "exec_prefix",
            "stdout",
            "stderr",
            "exit",
            "getsizeof",
            "getrecursionlimit",
            "getdefaultencoding",
            "getfilesystemencoding",
            "float_info",
            "int_info",
            "hexversion",
            "api_version",
            "flags",
            "path",
            "maxunicode",
            "stdlib_module_names",
            "builtin_module_names",
        }
    ),
    "shutil": frozenset(
        {
            "which",
            "disk_usage",
            "get_terminal_size",
            "rmtree",
            "copy",
            "copy2",
            "copyfile",
            "copytree",
            "move",
        }
    ),
    "io": frozenset({"StringIO", "BytesIO", "open", "TextIOWrapper"}),
    "codecs": frozenset(
        {"open", "decode", "encode", "lookup", "getreader", "getwriter", "BOM_UTF8"}
    ),
    "importlib.metadata": frozenset(
        {"version", "metadata", "requires", "PackageNotFoundError", "packages_distributions"}
    ),
}
# Функции модулей, меняющие файлы (уровень — по пути) и опасные сами по себе.
_FILE_WRITERS = frozenset(
    {
        "remove",
        "unlink",
        "rename",
        "replace",
        "mkdir",
        "makedirs",
        "rmdir",
        "removedirs",
        "copy",
        "copy2",
        "copyfile",
        "copytree",
        "move",
    }
)
_RECURSIVE = frozenset({"rmtree", "removedirs", "copytree"})
# Методы с побочными эффектами — опасны у любого объекта (имена однозначны).
_DANGER_METHODS = frozenset(
    {
        "system",
        "popen",
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnv",
        "spawnve",
        "spawnvp",
        "execl",
        "execle",
        "execlp",
        "execlpe",
        "execv",
        "execve",
        "execvp",
        "execvpe",
        "fork",
        "forkpty",
        "kill",
        "killpg",
        "urlopen",
        "putenv",
        "unsetenv",
        "chroot",
        "setuid",
        "setgid",
        "symlink",
        "link",
        "rmtree",
        "extractall",
        "startfile",
        "run_module",
        "run_path",
    }
)
# Методы путей, меняющие файлы.
_PATH_WRITERS = frozenset(
    {
        "write_text",
        "write_bytes",
        "touch",
        "mkdir",
        "unlink",
        "rmdir",
        "rename",
        "replace",
        "symlink_to",
        "hardlink_to",
        "chmod",
        "lchmod",
        "copy",
        "copy_into",
        "move",
        "move_into",
    }
)
# Из них — только у путей: у объекта неизвестного типа такой вызов — возможно, запись.
_PATH_ONLY_WRITERS = _PATH_WRITERS - {"rename", "replace", "copy", "move"}
# Переносят или копируют: запись и туда, куда (первый аргумент).
_PATH_MOVERS = frozenset({"rename", "replace", "copy", "copy_into", "move", "move_into"})
_TARGET_KWARGS = ("target", "target_dir")
_MODE_TEXT = re.compile(r"[rwxabt+U]+")
# Методы путей, читающие содержимое.
_PATH_READERS = frozenset({"read_text", "read_bytes"})
# Методы путей, перечисляющие каталог.
_PATH_LISTERS = frozenset({"iterdir", "glob", "rglob", "walk"})
# Методы путей, сохраняющие «где путь» (результат — тоже путь там же).
_PATH_SAME = frozenset(
    {"resolve", "absolute", "with_suffix", "with_name", "with_stem", "joinpath", "expanduser"}
)
# Доступ к кадрам, коду и внутренностям интерпретатора.
_INTERNALS = frozenset(
    {
        "tb_frame",
        "f_globals",
        "f_locals",
        "f_builtins",
        "f_back",
        "f_code",
        "gi_frame",
        "gi_code",
        "cr_frame",
        "cr_code",
        "ag_frame",
        "ag_code",
        "co_code",
        "func_globals",
        "_getframe",
        "settrace",
        "setprofile",
        "addaudithook",
        "modules",
        "meta_path",
        "path_hooks",
        "path_importer_cache",
        "load_module",
        "exec_module",
        "format_map",
        "Formatter",
        "vformat",
        "get_field",
    }
)
_SAFE_DUNDERS = frozenset({"__name__", "__version__", "__file__", "__doc__", "__all__"})
_SAFE_BUILTINS = frozenset(
    {
        "print",
        "len",
        "sorted",
        "sum",
        "min",
        "max",
        "range",
        "enumerate",
        "zip",
        "map",
        "filter",
        "list",
        "dict",
        "set",
        "frozenset",
        "tuple",
        "str",
        "bytes",
        "bytearray",
        "int",
        "float",
        "complex",
        "bool",
        "repr",
        "ascii",
        "isinstance",
        "issubclass",
        "abs",
        "round",
        "divmod",
        "pow",
        "any",
        "all",
        "ord",
        "chr",
        "hex",
        "oct",
        "bin",
        "format",
        "reversed",
        "iter",
        "next",
        "hash",
        "id",
        "callable",
        "slice",
        "object",
        "property",
        "staticmethod",
        "classmethod",
        "super",
        "exit",
        "quit",
        "hasattr",
        "Exception",
        "ValueError",
        "TypeError",
        "KeyError",
        "IndexError",
        "OSError",
        "RuntimeError",
        "StopIteration",
        "FileNotFoundError",
        "NotImplementedError",
        "AttributeError",
        "ImportError",
        "ModuleNotFoundError",
        "SystemExit",
        "UnicodeDecodeError",
        "ZeroDivisionError",
        "AssertionError",
        "ArithmeticError",
        "LookupError",
        "PermissionError",
        "IsADirectoryError",
        "NotADirectoryError",
        "type",
        "open",
    }
)
_DYNAMIC_BUILTINS = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "getattr",
        "setattr",
        "delattr",
        "vars",
        "globals",
        "locals",
        "breakpoint",
        "input",
        "help",
        "memoryview",
        "__builtins__",
        "__loader__",
        "__spec__",
    }
)
_BUILTIN_NAMES = frozenset(dir(builtins))
_WRITE_MODES = set("wax+")
_PATH_CTORS = frozenset(
    {"Path", "PurePath", "PosixPath", "PurePosixPath", "WindowsPath", "PureWindowsPath"}
)


@dataclass
class PyResult:
    level: Level | None
    reason: str


@dataclass
class _Path:
    """Что известно о пути в выражении. ``static`` — точное значение (абсолютное);
    ``under`` — путь получен обходом этого каталога (значение известно при выполнении)."""

    static: Path | None = None
    under: Path | None = None
    pattern: str | None = None  # имена, которые даёт обход (rglob('*.py') → *.py)

    @property
    def known(self) -> bool:
        return self.static is not None or self.under is not None


_UNKNOWN_PATH = _Path()


@dataclass
class _Ctx:
    root: Path
    cwd: Path  # абсолютный каталог запуска
    script: Path | None  # исполняемый файл (для __file__)
    modules: dict[str, str] = field(default_factory=dict)  # имя → модуль (или модуль.атрибут)
    bindings: dict[str, list[ast.AST]] = field(default_factory=dict)
    iterations: dict[str, list[tuple[ast.AST, int | None]]] = field(default_factory=dict)
    defs: set[str] = field(default_factory=set)  # свои функции и классы
    # import a.b связывает a: у a доступен только подмодуль b (если a не в белом списке)
    partial: dict[str, set[str]] = field(default_factory=dict)
    star: bool = False  # был from X import *: незнакомое имя может прийти оттуда
    levels: list[tuple[Level | None, str]] = field(default_factory=list)
    hazard_cache: dict[tuple[Path, str | None], bool] = field(default_factory=dict)

    def flag(self, level: Level | None, reason: str) -> None:
        self.levels.append((level, reason))


# --------------------------------------------------------------------------- #
# Привязки имён (поток-нечувствительно: все присваивания имени)
# --------------------------------------------------------------------------- #
def _collect_bindings(tree: ast.AST, ctx: _Ctx) -> None:
    def bind_target(
        target: ast.AST,
        value: ast.AST | None,
        *,
        iter_of: ast.AST | None = None,
        index: int | None = None,
    ) -> None:
        if isinstance(target, ast.Name):
            if iter_of is not None:
                ctx.iterations.setdefault(target.id, []).append((iter_of, index))
            else:
                ctx.bindings.setdefault(target.id, []).append(
                    value if value is not None else ast.Constant(value=None, kind="?")
                )
        elif isinstance(target, (ast.Tuple, ast.List)):
            for i, elt in enumerate(target.elts):
                if iter_of is not None:
                    bind_target(elt, None, iter_of=iter_of, index=i)
                else:
                    bind_target(elt, _DYNAMIC_EXPR)
        elif isinstance(target, ast.Starred):
            bind_target(target.value, _DYNAMIC_EXPR)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                bind_target(t, node.value)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and node.value is not None:
            bind_target(
                node.target, node.value if isinstance(node, ast.AnnAssign) else _DYNAMIC_EXPR
            )
        elif isinstance(node, ast.NamedExpr):
            bind_target(node.target, node.value)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            bind_target(node.target, None, iter_of=node.iter)
        elif isinstance(node, ast.comprehension):
            bind_target(node.target, None, iter_of=node.iter)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    bind_target(item.optional_vars, item.context_expr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            if not isinstance(node, ast.Lambda):
                ctx.defs.add(node.name)
            args = node.args
            for a in (*args.posonlyargs, *args.args, *args.kwonlyargs):
                bind_target(ast.Name(id=a.arg), _DYNAMIC_EXPR)
            for a in (args.vararg, args.kwarg):
                if a is not None:
                    bind_target(ast.Name(id=a.arg), _DYNAMIC_EXPR)
        elif isinstance(node, ast.ClassDef):
            ctx.defs.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bind_target(ast.Name(id=node.name), _DYNAMIC_EXPR)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                bind_target(ast.Name(id=name), _DYNAMIC_EXPR)


_DYNAMIC_EXPR = ast.Constant(value=None, kind="dynamic")


# --------------------------------------------------------------------------- #
# Импорты
# --------------------------------------------------------------------------- #
def _module_level(name: str, ctx: _Ctx) -> tuple[Level | None, str] | None:
    """Уровень за импорт модуля (None — безопасный модуль)."""
    top = name.split(".")[0]
    if name in SAFE_MODULES:
        return None
    if name in DANGER_MODULES or top in DANGER_MODULES:
        return Level.DANGER, f"импорт {name}"
    if top in sys.stdlib_module_names:
        return None, f"модуль {name} не проверен классификатором"
    return Level.EXEC, f"импорт {name} (исполнится код модуля)"


def _check_imports(tree: ast.AST, ctx: _Ctx) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                got = _module_level(alias.name, ctx)
                if got is not None:
                    ctx.flag(*got)
                if alias.asname:
                    ctx.modules[alias.asname] = alias.name
                else:
                    top, _, sub = alias.name.partition(".")
                    ctx.modules[top] = top
                    if sub and top not in SAFE_MODULES:
                        ctx.partial.setdefault(top, set()).add(sub.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                ctx.flag(Level.EXEC, "относительный импорт (код проекта)")
                continue
            module = node.module or ""
            for alias in node.names:
                full = f"{module}.{alias.name}"
                if alias.name == "*":
                    if module in _MODULE_ATTRS or module not in SAFE_MODULES:
                        ctx.flag(None, f"from {module} import *")
                    got = _module_level(module, ctx)
                    if got is not None:
                        ctx.flag(*got)
                    ctx.star = True
                    continue
                resolved = _attr_of_module(module, alias.name)
                if full in SAFE_MODULES or full in DANGER_MODULES:
                    got = _module_level(full, ctx)
                elif resolved != full:  # from glob import os — это os
                    got = _module_level(resolved, ctx)
                    if (parent := _module_level(module, ctx)) is not None:
                        ctx.flag(*parent)
                elif module in _MODULE_ATTRS:
                    _check_module_attr(module, alias.name, ctx)
                    got = None
                else:
                    got = _module_level(module, ctx)
                if got is not None:
                    ctx.flag(*got)
                ctx.modules[alias.asname or alias.name] = resolved


# Безобидные встроенные из списка динамических: только «не знаю», без тревоги.
_HARMLESS_DYNAMIC = frozenset({"input", "help", "memoryview"})


def _dynamic(name: str, ctx: _Ctx) -> None:
    """Динамический код (eval, getattr…) — разбор его не видит: опасно, пусть человек
    посмотрит внимательно."""
    if name in _HARMLESS_DYNAMIC:
        ctx.flag(None, f"{name}()")
    else:
        ctx.flag(Level.DANGER, f"{name} — динамический код, разбор его не видит")


def _check_module_attr(module: str, attr: str, ctx: _Ctx) -> None:
    allowed = _MODULE_ATTRS.get(module)
    if allowed is None:
        return
    if attr in _DANGER_METHODS or (
        module == "os" and attr in ("system", "popen", "environ", "getenv", "environb")
    ):
        reason = "окружение процесса (там ключи и токены)" if "env" in attr else "процессы"
        ctx.flag(Level.DANGER, f"{module}.{attr}: {reason}")
    elif attr not in allowed:
        ctx.flag(None, f"{module}.{attr} не проверен классификатором")


# --------------------------------------------------------------------------- #
# Пути
# --------------------------------------------------------------------------- #
def _const_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr) and all(isinstance(v, ast.Constant) for v in node.values):
        return "".join(str(v.value) for v in node.values)  # type: ignore[attr-defined]
    return None


def _resolve_module(node: ast.AST, ctx: _Ctx) -> str | None:
    """Имя модуля или «модуль.атрибут» для ``os``, ``os.path.join``, ``Path``."""
    if isinstance(node, ast.Name):
        return ctx.modules.get(node.id)
    if isinstance(node, ast.Attribute):
        base = _resolve_module(node.value, ctx)
        if base is not None:
            return _attr_of_module(base, node.attr)
    return None


# Атрибуты с именем модуля стандартной библиотеки, которые сами не модули.
_NOT_SUBMODULES = frozenset(
    {"ast.keyword", "ast.operator", "datetime.time", "locale.gettext", "unicodedata.decimal"}
)


def _attr_of_module(base: str, attr: str) -> str:
    """«модуль.атрибут» — или сам модуль, если атрибут — модуль, импортированный модулем
    стандартной библиотеки: ``glob.os`` и ``random._os`` — это ``os``, и их функции
    проверяются как функции ``os``."""
    full = f"{base}.{attr}"
    name = attr.lstrip("_")
    if full in SAFE_MODULES or full in DANGER_MODULES or full in _NOT_SUBMODULES:
        return full
    if base.split(".")[0] not in sys.stdlib_module_names:
        return full  # код сторонних модулей и так не проверяется (EXEC)
    if name == base.rpartition(".")[2] or name in _MODULE_ATTRS.get(base, ()):
        return full  # glob.glob, time.time, os.stat — функции, а не модули
    if name in sys.stdlib_module_names or name in DANGER_MODULES:
        return name
    return full


def _is_path_ctor(func: ast.AST, ctx: _Ctx) -> bool:
    name = _resolve_module(func, ctx)
    return name is not None and name.split(".")[-1] in _PATH_CTORS and name.startswith("pathlib")


def _join(base: _Path, parts: list[_Path | str | None]) -> _Path:
    if base.static is not None:
        current = base.static
        for i, p in enumerate(parts):
            if isinstance(p, str):
                current = current / p
            elif isinstance(p, _Path) and p.static is not None:
                current = p.static if p.static.is_absolute() else current / p.static
            elif isinstance(p, _Path) and p.under is not None:
                # имя или путь из обхода: результат — где-то в общем дереве
                common = os.path.commonpath([str(base.static), str(p.under)])
                return _join(_Path(under=Path(common)), parts[i + 1 :])
            else:
                return _UNKNOWN_PATH
        return _Path(static=Path(os.path.normpath(current)))
    if base.under is not None:
        for p in parts:
            if isinstance(p, str):
                if ".." in PurePosixPath(p).parts or p.startswith("/"):
                    return _UNKNOWN_PATH
            elif not (isinstance(p, _Path) and p.under is not None):
                return _UNKNOWN_PATH
        return _Path(under=base.under, pattern=base.pattern)
    return _UNKNOWN_PATH


def _path_of(node: ast.AST, ctx: _Ctx, depth: int = 0) -> _Path:
    """Где путь, заданный выражением (литерал, Path(...), обход каталога…)."""
    if depth > 12:
        return _UNKNOWN_PATH
    text = _const_str(node)
    if text is not None:
        if text.startswith("~"):
            return _Path(static=Path(os.path.expanduser(text)))
        p = Path(text)
        return _Path(static=Path(os.path.normpath(p if p.is_absolute() else ctx.cwd / p)))
    if isinstance(node, ast.Name):
        if node.id == "__file__" and ctx.script is not None:
            return _Path(static=ctx.script)
        found: list[_Path] = []
        for value in ctx.bindings.get(node.id, []):
            found.append(_path_of(value, ctx, depth + 1))
        for source, index in ctx.iterations.get(node.id, []):
            found.append(_iterated(source, index, ctx, depth + 1))
        if not found:
            return _UNKNOWN_PATH
        if len(found) == 1:
            return found[0]
        if all(f.static is not None for f in found) and len({f.static for f in found}) == 1:
            return found[0]
        unders = []
        for f in found:
            if f.static is not None:
                unders.append(f.static if f.static.is_dir() else f.static.parent)
            elif f.under is not None:
                unders.append(f.under)
            else:
                return _UNKNOWN_PATH
        return _Path(under=Path(os.path.commonpath([str(u) for u in unders])))
    if isinstance(node, ast.Call):
        func = node.func
        if _is_path_ctor(func, ctx) or (isinstance(func, ast.Name) and func.id in ("str", "repr")):
            if not node.args:
                return _Path(static=ctx.cwd) if not isinstance(func, ast.Name) else _UNKNOWN_PATH
            first = _path_of(node.args[0], ctx, depth + 1)
            rest = [_const_str(a) or _path_of(a, ctx, depth + 1) for a in node.args[1:]]
            return _join(first, rest) if rest else first
        name = _resolve_module(func, ctx)
        # Домашний каталог — вне проекта: запись туда должна быть видна как запись наружу.
        if name in ("pathlib.Path.home",) or (
            isinstance(func, ast.Attribute)
            and func.attr == "home"
            and _is_path_ctor(func.value, ctx)
        ):
            return _Path(static=Path.home())
        if name in ("os.path.expanduser",) and node.args:
            text = _const_str(node.args[0])
            return _Path(static=Path(os.path.expanduser(text))) if text else _UNKNOWN_PATH
        if isinstance(func, ast.Attribute) and func.attr == "expanduser":
            base = func.value
            text = _const_str(base.args[0]) if isinstance(base, ast.Call) and base.args else None
            if text is not None and _is_path_ctor(base.func, ctx):
                return _Path(static=Path(os.path.expanduser(text)))
            return _UNKNOWN_PATH
        if name in ("os.getcwd", "pathlib.Path.cwd") or (
            isinstance(func, ast.Attribute)
            and func.attr == "cwd"
            and _is_path_ctor(func.value, ctx)
        ):
            return _Path(static=ctx.cwd)
        if name in ("os.path.join", "posixpath.join") and node.args:
            first = _path_of(node.args[0], ctx, depth + 1)
            rest = [_const_str(a) or _path_of(a, ctx, depth + 1) for a in node.args[1:]]
            return _join(first, rest)
        if name in ("os.path.abspath", "os.path.normpath", "os.path.realpath") and node.args:
            return _path_of(node.args[0], ctx, depth + 1)
        if isinstance(func, ast.Attribute):
            if func.attr in ("resolve", "absolute"):
                return _path_of(func.value, ctx, depth + 1)
            if func.attr == "joinpath":
                base = _path_of(func.value, ctx, depth + 1)
                return _join(
                    base, [_const_str(a) or _path_of(a, ctx, depth + 1) for a in node.args]
                )
            if func.attr in ("with_suffix", "with_name", "with_stem"):
                base = _path_of(func.value, ctx, depth + 1)
                arg = _const_str(node.args[0]) if node.args else None
                if arg is None or "/" in arg:
                    return _UNKNOWN_PATH
                if base.static is not None:
                    p = base.static
                    new = (
                        p.with_suffix(arg)
                        if func.attr == "with_suffix"
                        else p.with_name(arg)
                        if func.attr == "with_name"
                        else p.with_stem(arg)
                    )
                    return _Path(static=new)
                return base
        return _UNKNOWN_PATH
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _path_of(node.left, ctx, depth + 1)
        right = _const_str(node.right)
        return _join(left, [right if right is not None else _path_of(node.right, ctx, depth + 1)])
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        base = _path_of(node.value, ctx, depth + 1)
        if base.static is not None:
            return _Path(static=base.static.parent)
        return _UNKNOWN_PATH
    if isinstance(node, ast.Attribute) and node.attr == "path":  # DirEntry.path
        return _path_of(node.value, ctx, depth + 1)
    if isinstance(node, ast.Subscript):  # files[0] из обхода — там же, где обход
        inner = _path_of(node.value, ctx, depth + 1)
        return _Path(under=inner.under) if inner.under is not None else _UNKNOWN_PATH
    return _UNKNOWN_PATH


def _iterated(source: ast.AST, index: int | None, ctx: _Ctx, depth: int) -> _Path:
    """Элемент обхода: ``for p in Path('src').rglob('*.py')``, ``os.walk('.')``, ``glob.glob``."""
    node = source
    # sorted(...), list(...), enumerate(...) и т.п. не меняют, где лежат элементы
    while (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("sorted", "list", "tuple", "set", "reversed", "enumerate", "filter")
        and node.args
    ):
        if node.func.id == "enumerate":
            if index == 0:
                return _UNKNOWN_PATH
            index = None
        node = node.args[-1] if node.func.id == "filter" else node.args[0]
    if isinstance(node, ast.Name):  # for f in files — files из обхода
        inner = _path_of(node, ctx, depth + 1)
        return _Path(under=inner.under, pattern=inner.pattern) if inner.under else _UNKNOWN_PATH
    if not isinstance(node, ast.Call):
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            paths = [_path_of(e, ctx, depth + 1) for e in node.elts]
            if paths and all(p.static is not None for p in paths):
                common = os.path.commonpath([str(p.static) for p in paths])
                return _Path(under=Path(common))
        return _UNKNOWN_PATH
    func = node.func
    name = _resolve_module(func, ctx)
    if (
        isinstance(func, ast.Attribute)
        and func.attr in _PATH_LISTERS
        and name is None
        and _looks_like_path(func.value, ctx)
    ):
        base = _path_of(func.value, ctx, depth + 1)
        pattern = _const_str(node.args[0]) if node.args else "*"
        if func.attr in ("glob", "rglob") and (
            pattern is None or ".." in pattern or pattern.startswith("/")
        ):
            return _UNKNOWN_PATH
        target = base.static or base.under
        if target is None:
            return _UNKNOWN_PATH
        last = pattern.rsplit("/", 1)[-1] if func.attr in ("glob", "rglob") and pattern else None
        if last is not None and "**" in last:
            last = None
        return _Path(under=target, pattern=last)
    if name in ("os.walk", "os.scandir", "os.listdir") and node.args:
        base = _path_of(node.args[0], ctx, depth + 1)
        target = base.static or base.under
        if target is None:
            return _UNKNOWN_PATH
        if name == "os.listdir":
            # Имена, а не пути: открываются относительно каталога запуска.
            return _Path(under=target) if target == ctx.cwd else _Path(under=ctx.cwd)
        return _Path(under=target)
    if name in ("os.walk", "os.scandir", "os.listdir") and not node.args:
        return _Path(under=ctx.cwd)
    if name in ("glob.glob", "glob.iglob") and node.args:
        pattern = _const_str(node.args[0])
        if pattern is None or ".." in pattern or pattern.startswith(("/", "~")):
            return _UNKNOWN_PATH
        if any(kw.arg == "root_dir" for kw in node.keywords):
            return _UNKNOWN_PATH
        return _Path(under=ctx.cwd)
    return _UNKNOWN_PATH


# --------------------------------------------------------------------------- #
# Проверки
# --------------------------------------------------------------------------- #
def _inside(path: Path, ctx: _Ctx) -> bool:
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError):
        return False
    return resolved == ctx.root or ctx.root in resolved.parents


def _hazards(directory: Path, pattern: str | None, ctx: _Ctx) -> bool:
    """В дереве каталога есть секреты или ссылки наружу (обход их прочитает)."""
    key = (directory, pattern)
    if key in ctx.hazard_cache:
        return ctx.hazard_cache[key]
    from gigacrew.safety.paths import has_escaping_links, has_secrets_under

    found = (
        not _inside(directory, ctx)
        or has_secrets_under(directory, ctx.root, name_pattern=pattern)
        or has_escaping_links(directory, ctx.root, name_pattern=pattern)
    )
    ctx.hazard_cache[key] = found
    return found


def _check_read(target: _Path, ctx: _Ctx, what: str) -> None:
    from gigacrew.safety.paths import Scope, scope_of_path

    if target.static is not None:
        s = scope_of_path(target.static, ctx.root)
        if s is Scope.SECRET_OUTSIDE:
            ctx.flag(Level.DANGER, f"{what}: секреты вне проекта")
        elif s is Scope.SECRET:
            ctx.flag(None, f"{what}: файл с секретами")
        elif s is not Scope.INSIDE and s is not Scope.SINK:
            ctx.flag(None, f"{what}: путь вне проекта")
        return
    if target.under is not None:
        if _hazards(target.under, target.pattern, ctx):
            ctx.flag(None, f"{what}: в обходимом каталоге секреты или ссылки наружу")
        return
    ctx.flag(None, f"{what}: путь известен только при выполнении")


def _check_list(target: _Path, ctx: _Ctx, what: str) -> None:
    where = target.static or target.under
    if where is None:
        ctx.flag(None, f"{what}: каталог известен только при выполнении")
    elif not _inside(where, ctx):
        ctx.flag(None, f"{what}: каталог вне проекта")


def _check_write(target: _Path, ctx: _Ctx, what: str, *, recursive: bool = False) -> None:
    from gigacrew.safety.paths import is_data_path, is_git_internal

    if recursive:
        ctx.flag(Level.DANGER, f"{what}: рекурсивное изменение")
        return
    if target.static is None:
        ctx.flag(None, f"{what}: путь известен только при выполнении")
        return
    if not _inside(target.static, ctx):
        ctx.flag(Level.DANGER, f"{what}: запись вне проекта")
    elif is_git_internal(target.static.resolve(), ctx.root):
        ctx.flag(Level.DANGER, f"{what}: служебные файлы git")
    elif is_data_path(target.static.resolve(), ctx.root):
        ctx.flag(Level.DANGER, f"{what}: служебная папка gigacrew")  # как запись из shell
    else:
        ctx.flag(Level.WRITE, what)


def _open_mode(node: ast.Call, mode_index: int) -> str | None:
    mode: ast.AST | None = None
    if len(node.args) > mode_index:
        mode = node.args[mode_index]
    for kw in node.keywords:
        if kw.arg == "mode":
            mode = kw.value
    if mode is None:
        return "r"
    return _const_str(mode)


def _check_open(node: ast.Call, target: _Path, mode_index: int, ctx: _Ctx, what: str) -> None:
    mode = _open_mode(node, mode_index)
    if mode is None:
        ctx.flag(None, f"{what}: режим открытия неизвестен")
    elif _WRITE_MODES & set(mode):
        _check_write(target, ctx, f"{what} на запись")
    else:
        _check_read(target, ctx, what)


def _check_call(node: ast.Call, ctx: _Ctx) -> None:
    func = node.func
    if isinstance(func, ast.Name):
        name = func.id
        bound = ctx.modules.get(name)
        if bound is not None:
            _check_module_function(bound, node, ctx)
            return
        if name in _DYNAMIC_BUILTINS:
            _dynamic(name, ctx)
        elif name == "open":
            target = _path_of(node.args[0], ctx) if node.args else _UNKNOWN_PATH
            _check_open(node, target, 1, ctx, "open()")
        elif name == "type" and len(node.args) > 1:
            ctx.flag(Level.DANGER, "type() с тремя аргументами — класс собирается при выполнении")
        elif name in _SAFE_BUILTINS:
            pass
        elif name in ctx.bindings or name in ctx.iterations or _defined(name, ctx):
            pass  # своя функция/лямбда/класс — её тело проверяется целиком
        else:
            ctx.flag(None, f"вызов {name}() не проверен классификатором")
        return
    if isinstance(func, ast.Attribute):
        module = _resolve_module(func, ctx)
        if module is not None:  # функция модуля: os.remove, json.loads, Path.cwd
            _check_module_function(module, node, ctx)
            return
        attr = func.attr
        if attr in _DANGER_METHODS:
            ctx.flag(Level.DANGER, f".{attr}()")
        elif attr in _PATH_WRITERS and _looks_like_path(func.value, ctx):
            target = _path_of(func.value, ctx)
            _check_write(target, ctx, f".{attr}()", recursive=False)
            if attr in _PATH_MOVERS:
                _check_destination(node, ctx, f".{attr}()")
            if attr.startswith("copy"):
                _check_read(target, ctx, f".{attr}()")  # копия секрета — его чтение
        elif attr in _PATH_READERS:
            _check_read(_path_of(func.value, ctx), ctx, f".{attr}()")
        elif attr in _PATH_ONLY_WRITERS or (attr in _PATH_MOVERS and _one_target(node)):
            ctx.flag(None, f".{attr}(): объект неизвестен — возможно, запись в файл")
            if attr in _PATH_MOVERS:
                _check_destination(node, ctx, f".{attr}()")
        elif attr == "open" and _looks_like_path(func.value, ctx):
            _check_open(node, _path_of(func.value, ctx), 0, ctx, ".open()")
        elif attr == "open" and _opens_for_write(node):
            ctx.flag(None, ".open(): объект неизвестен, режим — запись")
        elif attr in _PATH_LISTERS and _looks_like_path(func.value, ctx):
            _check_list(_path_of(func.value, ctx), ctx, f".{attr}()")
            if attr in ("glob", "rglob"):
                pattern = _const_str(node.args[0]) if node.args else None
                if pattern is None or ".." in pattern or pattern.startswith("/"):
                    ctx.flag(None, f".{attr}(): шаблон неизвестен или ведёт наружу")
        elif attr in ("expanduser", "home"):
            ctx.flag(None, f".{attr}() — путь вне проекта")
        elif attr == "format":
            # Поля шаблона ({0.__class__}, {0[key]}) обходят атрибуты при выполнении — ast их
            # не видит. Литерал проверяем, шаблон из переменной — «не знаю».
            text = func.value.value if isinstance(func.value, ast.Constant) else None
            if isinstance(text, str) and _format_traverses(text):
                ctx.flag(Level.DANGER, "str.format обходит атрибуты объектов ({0.x}, {0[k]})")
            elif not isinstance(text, str):
                ctx.flag(None, "str.format: шаблон не литерал")
        return
    ctx.flag(None, "вызов результата выражения")


def _one_target(node: ast.Call) -> bool:
    """Вызов как ``p.rename(target)``: один позиционный аргумент или ``target=``
    (у ``str.replace`` их два, у ``dict.copy`` — ни одного)."""
    if len(node.args) == 1:
        return True
    return not node.args and any(kw.arg in _TARGET_KWARGS for kw in node.keywords)


def _check_destination(node: ast.Call, ctx: _Ctx, what: str) -> None:
    """Куда переносит или копирует ``p.rename/replace/copy/move(_into)``."""
    dest = _call_arg(node, 0, *_TARGET_KWARGS)
    _check_write(_path_of(dest, ctx) if dest is not None else _UNKNOWN_PATH, ctx, what)


def _call_arg(node: ast.Call, index: int, *names: str) -> ast.AST | None:
    """Аргумент вызова — позиционный или именованный. ``*args`` и ``**kwargs`` могут
    подставить любой: тогда значение известно только при выполнении."""
    if any(isinstance(a, ast.Starred) for a in node.args[: index + 1]) or any(
        kw.arg is None for kw in node.keywords
    ):
        return _RUNTIME_ARG
    if len(node.args) > index:
        return node.args[index]
    for kw in node.keywords:
        if kw.arg in names:
            return kw.value
    return None


def _opens_for_write(node: ast.Call) -> bool:
    """``x.open(...)`` у объекта неизвестного типа: режим записи или неизвестный режим."""
    for kw in node.keywords:
        if kw.arg == "mode" and not isinstance(_const_str(kw.value), str):
            return True
    modes = [_const_str(arg) for arg in [*node.args, *(kw.value for kw in node.keywords)]]
    return any(
        isinstance(m, str) and _MODE_TEXT.fullmatch(m) and _WRITE_MODES & set(m) for m in modes
    )


def _defined(name: str, ctx: _Ctx) -> bool:
    return name in ctx.bindings or name in ctx.defs


def _known_name(name: str, ctx: _Ctx) -> bool:
    """Имя связано в коде (импорт, присваивание, def) или встроенное."""
    return (
        name in ctx.modules
        or name in ctx.iterations
        or _defined(name, ctx)
        or name in _BUILTIN_NAMES
        or name in _SAFE_DUNDERS
    )


def _looks_like_path(node: ast.AST, ctx: _Ctx, depth: int = 0) -> bool:
    """Выражение — объект pathlib (а не строка: у str тоже есть ``replace``)."""
    if depth > 12:
        return False
    if isinstance(node, ast.Call):
        func = node.func
        if _is_path_ctor(func, ctx):
            return True
        if isinstance(func, ast.Attribute):
            if func.attr in _PATH_SAME or func.attr in ("cwd", "home"):
                return _looks_like_path(func.value, ctx, depth + 1) or _is_path_ctor(
                    func.value, ctx
                )
        return False
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _looks_like_path(node.left, ctx, depth + 1)
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        return _looks_like_path(node.value, ctx, depth + 1)
    if isinstance(node, ast.Name):
        values = ctx.bindings.get(node.id, [])
        sources = ctx.iterations.get(node.id, [])
        if any(_looks_like_path(v, ctx, depth + 1) for v in values):
            return True
        for source, _ in sources:
            src = source
            while (
                isinstance(src, ast.Call)
                and isinstance(src.func, ast.Name)
                and src.func.id in ("sorted", "list", "tuple", "set", "reversed")
                and src.args
            ):
                src = src.args[0]
            if (
                isinstance(src, ast.Call)
                and isinstance(src.func, ast.Attribute)
                and src.func.attr in _PATH_LISTERS
            ):
                return True
        return False
    return False


def _format_traverses(text: str) -> bool:
    import string

    try:
        for _, field_name, _, _ in string.Formatter().parse(text):
            if field_name and any(ch in field_name for ch in ".["):
                return True
    except ValueError:
        return True
    return False


def _check_module_function(full: str, node: ast.Call, ctx: _Ctx) -> None:
    module, _, attr = full.rpartition(".")
    if not module:
        return
    if module in DANGER_MODULES or module.split(".")[0] in DANGER_MODULES:
        ctx.flag(Level.DANGER, f"{full}()")
        return
    if module in ("os", "shutil") and attr in _FILE_WRITERS | _RECURSIVE:
        src = _call_arg(node, 0, "path", "name", "src")
        target = _path_of(src, ctx) if src is not None else _UNKNOWN_PATH
        _check_write(target, ctx, f"{full}()", recursive=attr in _RECURSIVE)
        if attr in ("rename", "replace", "copy", "copy2", "copyfile", "move"):
            dst = _call_arg(node, 1, "dst")
            if dst is not None:
                _check_write(_path_of(dst, ctx), ctx, f"{full}()")
            if attr.startswith("copy"):
                _check_read(target, ctx, f"{full}()")
        if any(
            kw.arg in ("dir_fd", "src_dir_fd", "dst_dir_fd")
            and not (isinstance(kw.value, ast.Constant) and kw.value.value is None)
            for kw in node.keywords
        ):
            ctx.flag(None, f"{full}(): путь относительно открытого каталога")
        return
    if module == "os" and attr in ("listdir", "scandir", "walk"):
        target = _path_of(node.args[0], ctx) if node.args else _Path(static=ctx.cwd)
        _check_list(target, ctx, f"{full}()")
        return
    if module in ("io", "codecs") and attr == "open":
        target = _path_of(node.args[0], ctx) if node.args else _UNKNOWN_PATH
        _check_open(node, target, 1, ctx, f"{full}()")
        return
    if module == "tokenize" and attr == "open":
        target = _path_of(node.args[0], ctx) if node.args else _UNKNOWN_PATH
        _check_read(target, ctx, f"{full}()")
        return
    if module == "glob" and attr in ("glob", "iglob"):
        pattern = _const_str(node.args[0]) if node.args else None
        if pattern is None or ".." in pattern or pattern.startswith(("/", "~")):
            ctx.flag(None, f"{full}(): шаблон неизвестен или ведёт наружу")
        if any(kw.arg == "root_dir" for kw in node.keywords):
            ctx.flag(None, f"{full}(root_dir=…)")
        return
    if module in ("os.path", "posixpath") and attr in ("expanduser", "expandvars"):
        ctx.flag(None, f"{full}() — путь вне проекта")
        return
    if module.startswith("pathlib") and attr in ("home",):
        ctx.flag(None, f"{full}() — путь вне проекта")
        return
    if (
        module.startswith("pathlib")
        and module.rpartition(".")[2] in _PATH_CTORS
        and attr in _PATH_WRITERS | _PATH_READERS | _PATH_LISTERS | {"open"}
    ):
        # метод, вызванный у класса: Path.write_text(p, 'x') — то же, что p.write_text('x')
        obj = node.args[0] if node.args else _RUNTIME_ARG
        method = ast.Attribute(value=obj, attr=attr, ctx=ast.Load())
        _check_call(ast.Call(func=method, args=node.args[1:], keywords=node.keywords), ctx)
        return
    if module in ("json", "tomllib", "csv", "pickle") and attr in ("dump",):
        return  # файл открыт open() — проверен там
    allowed = _MODULE_ATTRS.get(module)
    if allowed is not None and attr not in allowed:
        ctx.flag(None, f"{full}() не проверен классификатором")


def _collect_aliases(ctx: _Ctx) -> None:
    """``rm = os.remove``, ``P = Path``: имя — псевдоним модуля или его атрибута, вызовы
    через него проверяются как вызовы оригинала. Только если других значений у имени нет
    (иначе ``p.unlink()`` для ``p`` из обхода стал бы «функцией модуля»)."""
    changed = True
    while changed:  # псевдоним псевдонима
        changed = False
        for name, values in ctx.bindings.items():
            if name in ctx.modules or name in ctx.iterations or name in ctx.defs:
                continue
            resolved = {_resolve_module(v, ctx) for v in values}
            if len(resolved) == 1 and (target := resolved.pop()) is not None:
                ctx.modules[name] = target
                changed = True


def _is_alias(node: ast.AST, parent: ast.AST | None, ctx: _Ctx) -> bool:
    if (
        not isinstance(parent, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
        or parent.value is not node
    ):
        return False
    targets = parent.targets if isinstance(parent, ast.Assign) else [parent.target]
    resolved = _resolve_module(node, ctx)
    return all(isinstance(t, ast.Name) and ctx.modules.get(t.id) == resolved for t in targets)


# Модули с функциями записи и процессов: переданный куда-то модуль — не проследить.
_ESCAPING_MODULES = frozenset({*_MODULE_ATTRS, "pathlib"})
# Атрибуты, которые проверяются в месте вызова: как значения — проверяем вызов с
# аргументами, известными только при выполнении.
_REF_ATTRS = _PATH_WRITERS | _PATH_READERS | _PATH_LISTERS | _DANGER_METHODS | {"open", "format"}
_RUNTIME_ARG = ast.Name(id="<при выполнении>")


def _harmless_use(node: ast.AST, parent: ast.AST | None, ctx: _Ctx) -> bool:
    """Значение только читают: ``print(os)``, ``for p in sys.path``, ``'\\n'.join(...)``."""
    if isinstance(parent, ast.Call):
        func = parent.func
        if isinstance(func, ast.Name):
            return (
                func.id in _SAFE_BUILTINS
                and func.id not in ctx.modules
                and not _defined(func.id, ctx)
            )
        return isinstance(func, ast.Attribute) and isinstance(func.value, ast.Constant)
    if isinstance(parent, (ast.For, ast.AsyncFor, ast.comprehension)):
        return parent.iter is node
    return isinstance(parent, (ast.Compare, ast.FormattedValue, ast.Expr))


def _check_reference(node: ast.AST, parent: ast.AST | None, ctx: _Ctx) -> None:
    """Функция или модуль как значение (``map(os.remove, …)``, ``o = open``, ``f(os)``):
    проверки в месте вызова их не увидят."""
    if isinstance(parent, ast.Call) and parent.func is node:
        return  # вызов — проверяется в _check_call
    if isinstance(parent, ast.Attribute):
        return  # основание атрибута: os.path, Path('x').write_text
    resolved = _resolve_module(node, ctx)
    if resolved is not None and _is_alias(node, parent, ctx):
        return  # псевдоним — проверяются его вызовы
    if resolved in _ESCAPING_MODULES:
        if not _harmless_use(node, parent, ctx):
            ctx.flag(None, f"модуль {resolved} передан как значение — что с ним сделают, не видно")
        return
    if resolved is None:
        if isinstance(node, ast.Name):
            if node.id != "open" or _defined(node.id, ctx):
                return
        elif not isinstance(node, ast.Attribute) or node.attr not in _REF_ATTRS:
            return
    start = len(ctx.levels)
    _check_call(ast.Call(func=node, args=[_RUNTIME_ARG], keywords=[]), ctx)
    found = ctx.levels[start:]
    del ctx.levels[start:]
    if found:
        what = resolved or (node.id if isinstance(node, ast.Name) else f".{node.attr}")
        level = worst(*(lv for lv, _ in found))
        ctx.flag(level, f"{what} передана как значение: аргументы известны только при выполнении")


def _check_attributes(tree: ast.AST, ctx: _Ctx) -> None:
    parents = {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, ast.Load):
            _check_reference(node, parents.get(id(node)), ctx)
        if isinstance(node, ast.Attribute):
            attr = node.attr
            if attr.startswith("__") and attr.endswith("__") and attr not in _SAFE_DUNDERS:
                ctx.flag(Level.DANGER, f"доступ к {attr} — внутренности интерпретатора")
            elif attr in _INTERNALS:
                ctx.flag(Level.DANGER, f"доступ к {attr} — внутренности интерпретатора")
            module = _resolve_module(node.value, ctx)
            if module is not None:
                resolved = _attr_of_module(module, attr)
                if resolved != f"{module}.{attr}":  # glob.os — импорт модуля os
                    got = _module_level(resolved, ctx)
                    if got is not None:
                        ctx.flag(*got)
                _check_module_attr(module, attr, ctx)
                if module in ctx.partial and attr not in ctx.partial[module]:
                    if attr in ("import_module", "__import__", "reload", "util", "machinery"):
                        ctx.flag(Level.DANGER, f"{module}.{attr} — импорт при выполнении")
                    else:
                        ctx.flag(None, f"{module}.{attr} не проверен классификатором")
            if (
                isinstance(node.value, ast.Attribute)
                and node.value.attr == "path"
                and _resolve_module(node.value.value, ctx) == "sys"
            ):
                ctx.flag(Level.DANGER, "изменение sys.path — подмена модулей")
        elif isinstance(node, ast.Name):
            if node.id in _DYNAMIC_BUILTINS:
                _dynamic(node.id, ctx)
            elif (
                node.id.startswith("__") and node.id.endswith("__") and node.id not in _SAFE_DUNDERS
            ):
                ctx.flag(Level.DANGER, f"доступ к {node.id} — внутренности интерпретатора")
            elif ctx.star and not _known_name(node.id, ctx):
                # from platform import * приносит и os: что за объект — не узнать
                ctx.flag(None, f"{node.id}: имя из from … import *, не проверено")
        elif isinstance(node, ast.Call):
            _check_call(node, ctx)
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
            for t in targets:
                if isinstance(t, (ast.Attribute, ast.Subscript)):
                    base = t.value
                    while isinstance(base, (ast.Attribute, ast.Subscript)):
                        base = base.value
                    mod = _resolve_module(base, ctx) if isinstance(base, ast.Name) else None
                    if mod is not None:
                        ctx.flag(Level.DANGER, f"изменение модуля {mod}")
        elif isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Attribute) and node.value.attr == "environ":
                ctx.flag(Level.DANGER, "os.environ: окружение процесса (там ключи и токены)")


def shadowed_stdlib(directory: Path) -> str | None:
    """Модуль стандартной библиотеки, затенённый файлом в ``directory`` (``json.py``)."""
    try:
        entries = os.listdir(directory)
    except OSError:
        return None
    names = sys.stdlib_module_names
    for entry in entries:
        stem, dot, ext = entry.partition(".")
        if not stem or stem not in names:
            continue
        if not dot:
            path = directory / entry
            if (path / "__init__.py").exists() or (path / "__init__.pyc").exists():
                return entry
            continue
        if ext in ("py", "pyc", "pyw") or ext.endswith(("so", "pyd")):
            return entry
    return None


def analyze(
    code: str,
    *,
    root: Path,
    cwd: str = ".",
    script: Path | None = None,
    path_dirs: tuple[Path, ...] = (),
) -> PyResult:
    """Оценить Python-код. ``script`` — файл скрипта (для ``__file__`` и затенения),
    ``path_dirs`` — каталоги из PYTHONPATH (тоже проверяются на затенение);
    без ``script`` первым в ``sys.path`` стоит каталог запуска."""
    root = Path(root).resolve()
    run_dir = (root / cwd).resolve()
    if len(code) > MAX_SOURCE:
        return PyResult(None, "слишком большой код для разбора")
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return PyResult(None, "код не разобран (синтаксис)")
    ctx = _Ctx(root=root, cwd=run_dir, script=script)
    _collect_bindings(tree, ctx)
    _check_imports(tree, ctx)
    _collect_aliases(ctx)
    _check_attributes(tree, ctx)
    first = script.parent if script is not None else run_dir
    for directory in (first, *path_dirs):
        shadow = shadowed_stdlib(directory)
        if shadow is not None:
            ctx.flag(
                Level.EXEC, f"{shadow} в каталоге запуска подменит модуль стандартной библиотеки"
            )
    if not ctx.levels:
        return PyResult(Level.READ, "python: только чтение и вычисления")
    level = worst(*(lv for lv, _ in ctx.levels))
    # причина — первая из худшего уровня
    for lv, reason in ctx.levels:
        if lv == level:
            return PyResult(level, f"python: {reason}")
    return PyResult(level, f"python: {ctx.levels[0][1]}")
