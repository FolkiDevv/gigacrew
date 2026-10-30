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
* DANGER — процессы, сеть, рекурсивное удаление, запись вне проекта, чтение чужих секретов,
  динамический код (``eval``, ``getattr``) и внутренности интерпретатора (``__globals__``,
  вызов ``__init__``, ``__dict__`` модуля);
* None — пути, известные только при выполнении, незнакомые модули стандартной библиотеки,
  класс объекта как значение (``f = type(x)``: его вызов не проверить).

Описание объектов — чтение: ``X.__doc__``, ``X.__mro__``, ``print(X.__init__)``,
``dir(X)``, ``inspect.signature(X.__init__)``. Методы и пространства имён объектов
(``__init__``, ``__dict__``) — только для осмотра: напечатать, сравнить, перечислить имена.

Анализ потоконечувствительный: проверяется всё дерево, включая тела функций и классов,
которые могут и не выполниться, — ложная тревога лучше пропуска.
"""

from __future__ import annotations

import ast
import builtins
import os
import re
import string
import sys
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from gigacrew.safety.levels import CommandLevel as Level
from gigacrew.safety.levels import worst
from gigacrew.safety.paths import (
    Scope,
    has_escaping_links,
    has_secrets_under,
    is_data_path,
    is_vcs_internal,
    scope_of_path,
)

__all__ = ["PyResult", "analyze", "shadowed_stdlib"]

MAX_SOURCE = 262_144  # 256 КиБ: больше — не разбираем
# Глубина вложенности выражений, до которой прослеживаются пути.
_MAX_DEPTH = 12

# Модули без побочных эффектов (подмодули — только перечисленные).
SAFE_MODULES = frozenset((
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
    "inspect",
))
# Процессы, сеть, загрузка кода, низкоуровневый доступ.
DANGER_MODULES = frozenset((
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
))
# Атрибуты модулей с белым списком; остальное — «не знаю» (или DANGER — _DANGER_ATTRS).
_MODULE_ATTRS: Mapping[str, frozenset[str]] = MappingProxyType({
    "os": frozenset((
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
    )),
    "sys": frozenset((
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
    )),
    "shutil": frozenset((
        "which",
        "disk_usage",
        "get_terminal_size",
        "rmtree",
        "copy",
        "copy2",
        "copyfile",
        "copytree",
        "move",
    )),
    "io": frozenset(("StringIO", "BytesIO", "open", "TextIOWrapper")),
    "codecs": frozenset((
        "open",
        "decode",
        "encode",
        "lookup",
        "getreader",
        "getwriter",
        "BOM_UTF8",
    )),
    "importlib.metadata": frozenset((
        "version",
        "metadata",
        "requires",
        "PackageNotFoundError",
        "packages_distributions",
    )),
    # Без getmembers (значения, как vars), getmodule (модуль), unwrap, кадров и
    # get_annotations; eval_str проверяется отдельно.
    "inspect": frozenset((
        "signature",
        "getdoc",
        "cleandoc",
        "getcomments",
        "getfile",
        "getsourcefile",
        "getsource",
        "getsourcelines",
        "getmodulename",
        "getfullargspec",
        "getmro",
        "Parameter",
        "isabstract",
        "isasyncgenfunction",
        "isbuiltin",
        "isclass",
        "iscoroutinefunction",
        "isdatadescriptor",
        "isfunction",
        "isgeneratorfunction",
        "ismethod",
        "ismethoddescriptor",
        "ismodule",
        "isroutine",
    )),
})
# Функции inspect, которые только описывают переданный объект (позиция осмотра).
_INSPECT_READERS = frozenset(f"inspect.{name}" for name in _MODULE_ATTRS["inspect"] - {"Parameter"})
# Члены «безопасных» модулей, которые не только читают (причина — для пользователя).
_UNCHECKED_MEMBERS: Mapping[str, str] = MappingProxyType({
    "argparse.FileType": "открывает файлы (и на запись) по путям из аргументов",
    "typing.get_type_hints": "вычисляет строки-аннотации как код",
    "typing.ForwardRef": "вычисляет строку как код",
    "typing.evaluate_forward_ref": "вычисляет строку как код",
    "functools.singledispatch": "register вычисляет строки-аннотации как код",
    "functools.singledispatchmethod": "register вычисляет строки-аннотации как код",
})
# Функции модулей, меняющие файлы (уровень — по пути) и опасные сами по себе.
_FILE_WRITERS = frozenset((
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
))
_RECURSIVE = frozenset(("rmtree", "removedirs", "copytree"))
# Методы с побочными эффектами — опасны у любого объекта (имена однозначны).
_DANGER_METHODS = frozenset((
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
    "_evaluate",  # typing.ForwardRef: строка исполняется как код
))
# Методы путей, меняющие файлы.
_PATH_WRITERS = frozenset((
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
))
# Из них — только у путей: у объекта неизвестного типа такой вызов — возможно, запись.
_PATH_ONLY_WRITERS = _PATH_WRITERS - {"rename", "replace", "copy", "move"}
# Переносят или копируют: запись и туда, куда (первый аргумент).
_PATH_MOVERS = frozenset(("rename", "replace", "copy", "copy_into", "move", "move_into"))
_TARGET_KWARGS = ("target", "target_dir")
_MODE_TEXT = re.compile(r"[rwxabt+U]+")
# Методы путей, читающие содержимое.
_PATH_READERS = frozenset(("read_text", "read_bytes"))
# Методы путей, перечисляющие каталог.
_PATH_LISTERS = frozenset(("iterdir", "glob", "rglob", "walk"))
# Методы путей, сохраняющие «где путь» (результат — тоже путь там же).
_PATH_SAME = frozenset((
    "resolve",
    "absolute",
    "with_suffix",
    "with_name",
    "with_stem",
    "joinpath",
    "expanduser",
))
# Доступ к кадрам, коду и внутренностям интерпретатора.
_INTERNALS = frozenset((
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
))
# Dunder-атрибуты, описывающие объект: читать можно где угодно.
_SAFE_DUNDERS = frozenset((
    "__name__",
    "__qualname__",
    "__module__",
    "__doc__",
    "__version__",
    "__file__",
    "__cached__",
    "__all__",
    "__package__",
    "__path__",
    "__debug__",
    "__annotations__",
    "__type_params__",
    "__parameters__",
    "__args__",
    "__origin__",
    "__orig_bases__",
    "__mro__",
    "__bases__",
    "__base__",
    "__class__",
    "__slots__",
    "__match_args__",
    "__members__",
    "__dataclass_fields__",
    "__signature__",
    "__text_signature__",
    "__abstractmethods__",
    "__firstlineno__",
    "__static_attributes__",
))
# Из них дают классы: вызов класса не проверить (FileIO из open(…).raw пишет файл).
_CLASS_DUNDERS = frozenset(("__class__", "__base__", "__bases__", "__mro__", "__orig_bases__"))
# Ступени побега из разбора и утечки окружения: опасно даже просто показать.
_ESCAPE_DUNDERS = frozenset((
    "__globals__",
    "__builtins__",
    "__closure__",
    "__defaults__",
    "__kwdefaults__",
    "__subclasses__",
))
# Пространство имён объекта: только имена (print(os.__dict__) покажет os.environ).
_NAMESPACE_DUNDER = "__dict__"
# Поля объекта кода, которые только описывают функцию (f.__code__.co_varnames).
_CODE_FIELDS = frozenset((
    "co_varnames",
    "co_argcount",
    "co_posonlyargcount",
    "co_kwonlyargcount",
    "co_filename",
    "co_firstlineno",
    "co_name",
    "co_qualname",
))
# Методы базового класса, которые вызывает свой класс: super().__init__(…).
_SUPER_DUNDERS = frozenset(("__init__", "__new__", "__init_subclass__", "__post_init__"))
# Встроенные, которые только показывают или проверяют аргумент.
_INSPECTORS = frozenset((
    "print",
    "repr",
    "str",
    "ascii",
    "len",
    "type",
    "id",
    "hash",
    "callable",
    "isinstance",
    "issubclass",
    "dir",
    "bool",
    "help",
))
# Встроенные, которые берут у словаря только ключи.
_NAME_LISTERS = frozenset(("len", "sorted", "list", "tuple", "set", "frozenset", "dir"))
# Функции, которые с eval_str вычисляют строки-аннотации.
_ANNOTATION_READERS = frozenset(("signature", "from_callable", "get_annotations"))
_SAFE_BUILTINS = frozenset((
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
    "dir",
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
))
_DYNAMIC_BUILTINS = frozenset((
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
))
# Имена встроенных — ровно то, что перечисляет dir() у модуля builtins.
_BUILTIN_NAMES = frozenset(dir(builtins))  # noqa: WPS421
_WRITE_MODES = set("wax+")
_PATH_CTORS = frozenset((
    "Path",
    "PurePath",
    "PosixPath",
    "PurePosixPath",
    "WindowsPath",
    "PureWindowsPath",
))


# Источник элементов обхода и номер элемента в распаковке (``for a, b in …``).
type _Iteration = tuple[ast.AST, int | None]


@dataclass
class PyResult:
    """Итог разбора: уровень (None — «не знаю») и причина."""

    level: Level | None
    reason: str


@dataclass
class _Path:
    """Что известно о пути в выражении.

    ``static`` — точное значение (абсолютное); ``under`` — путь получен обходом этого
    каталога (значение известно при выполнении).
    """

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
    iterations: dict[str, list[_Iteration]] = field(default_factory=dict)
    defs: set[str] = field(default_factory=set)  # свои функции и классы
    # import a.b связывает a: у a доступен только подмодуль b (если a не в белом списке)
    partial: dict[str, set[str]] = field(default_factory=dict)
    star: bool = False  # был from X import *: незнакомое имя может прийти оттуда
    levels: list[tuple[Level | None, str]] = field(default_factory=list)
    hazard_cache: dict[tuple[Path, str | None], bool] = field(default_factory=dict)
    parents: dict[int, ast.AST] = field(default_factory=dict)  # id(узел) → родитель

    def flag(self, level: Level | None, reason: str) -> None:
        self.levels.append((level, reason))


# --------------------------------------------------------------------------- #
# Привязки имён (поток-нечувствительно: все присваивания имени)
# --------------------------------------------------------------------------- #
def _bind_target(
    target: ast.AST,
    assigned: ast.AST | None,
    ctx: _Ctx,
    *,
    iter_of: ast.AST | None = None,
    index: int | None = None,
) -> None:
    """Связать имена цели присваивания (или обхода ``iter_of``) со значением."""
    if isinstance(target, ast.Name):
        if iter_of is None:
            ctx.bindings.setdefault(target.id, []).append(
                ast.Constant(value=None, kind="?") if assigned is None else assigned,
            )
        else:
            ctx.iterations.setdefault(target.id, []).append((iter_of, index))
    elif isinstance(target, (ast.Tuple, ast.List)):
        for position, element in enumerate(target.elts):
            if iter_of is None:
                _bind_target(element, _DYNAMIC_EXPR, ctx)
            else:
                _bind_target(element, None, ctx, iter_of=iter_of, index=position)
    elif isinstance(target, ast.Starred):
        _bind_target(target.value, _DYNAMIC_EXPR, ctx)


def _bind_assignment(node: ast.AST, ctx: _Ctx) -> None:
    """Присваивания: ``=``, ``x: T = …``, ``+=``, ``:=``."""
    if isinstance(node, ast.Assign):
        for target in node.targets:
            _bind_target(target, node.value, ctx)
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and node.value is not None:
        _bind_target(
            node.target,
            node.value if isinstance(node, ast.AnnAssign) else _DYNAMIC_EXPR,
            ctx,
        )
    elif isinstance(node, ast.NamedExpr):
        _bind_target(node.target, node.value, ctx)


def _bind_block(node: ast.AST, ctx: _Ctx) -> None:
    """Имена из заголовков блоков: ``for``, включения, ``with … as``, ``except … as``."""
    if isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
        _bind_target(node.target, None, ctx, iter_of=node.iter)
    elif isinstance(node, (ast.With, ast.AsyncWith)):
        for with_item in node.items:
            if with_item.optional_vars is not None:
                _bind_target(with_item.optional_vars, with_item.context_expr, ctx)
    elif isinstance(node, ast.ExceptHandler) and node.name:
        _bind_target(ast.Name(id=node.name), _DYNAMIC_EXPR, ctx)


def _bind_arguments(arguments: ast.arguments, ctx: _Ctx) -> None:
    """Параметры функции или лямбды: значения известны только при выполнении."""
    for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs):
        _bind_target(ast.Name(id=argument.arg), _DYNAMIC_EXPR, ctx)
    for starred in (arguments.vararg, arguments.kwarg):
        if starred is not None:
            _bind_target(ast.Name(id=starred.arg), _DYNAMIC_EXPR, ctx)


def _bind_definition(node: ast.AST, ctx: _Ctx) -> None:
    """Свои функции, лямбды и классы, ``global`` и ``nonlocal``."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        if not isinstance(node, ast.Lambda):
            ctx.defs.add(node.name)
        _bind_arguments(node.args, ctx)
    elif isinstance(node, ast.ClassDef):
        ctx.defs.add(node.name)
    elif isinstance(node, (ast.Global, ast.Nonlocal)):
        for name in node.names:
            _bind_target(ast.Name(id=name), _DYNAMIC_EXPR, ctx)


def _collect_bindings(tree: ast.AST, ctx: _Ctx) -> None:
    """Все привязки имён в дереве (у каждого узла — не больше одной группы)."""
    for node in ast.walk(tree):
        _bind_assignment(node, ctx)
        _bind_block(node, ctx)
        _bind_definition(node, ctx)


_DYNAMIC_EXPR = ast.Constant(value=None, kind="dynamic")


# --------------------------------------------------------------------------- #
# Импорты
# --------------------------------------------------------------------------- #
def _module_level(name: str, ctx: _Ctx) -> tuple[Level | None, str] | None:
    """Уровень за импорт модуля (None — безопасный модуль)."""
    top = name.split(".", maxsplit=1)[0]
    if name in SAFE_MODULES:
        return None
    if name in DANGER_MODULES or top in DANGER_MODULES:
        return Level.DANGER, f"импорт {name}"
    if top in sys.stdlib_module_names:
        return None, f"модуль {name} не проверен классификатором"
    return Level.EXEC, f"импорт {name} (исполнится код модуля)"


def _check_import(node: ast.Import, ctx: _Ctx) -> None:
    """``import a.b [as c]``: уровень за модуль и привязка имени."""
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


def _check_star_import(module: str, ctx: _Ctx) -> None:
    """``from X import *``: незнакомое имя может прийти оттуда."""
    if module in _MODULE_ATTRS or module not in SAFE_MODULES:
        ctx.flag(None, f"from {module} import *")
    got = _module_level(module, ctx)
    if got is not None:
        ctx.flag(*got)
    ctx.star = True


def _from_import_level(
    module: str,
    name: str,
    resolved: str,
    ctx: _Ctx,
) -> tuple[Level | None, str] | None:
    """Уровень за ``from module import name`` (``resolved`` — что это за объект)."""
    full = f"{module}.{name}"
    if full in SAFE_MODULES or full in DANGER_MODULES:
        return _module_level(full, ctx)
    if resolved != full:  # from glob import os — это os
        got = _module_level(resolved, ctx)
        parent = _module_level(module, ctx)
        if parent is not None:
            ctx.flag(*parent)
        return got
    if module in _MODULE_ATTRS:
        _check_module_attr(module, name, ctx)
        return None
    return _module_level(module, ctx)


def _check_import_from(node: ast.ImportFrom, ctx: _Ctx) -> None:
    """``from module import name [as alias]``: уровень за импорт и привязка имени."""
    if node.level:
        ctx.flag(Level.EXEC, "относительный импорт (код проекта)")
        return
    module = node.module or ""
    for alias in node.names:
        if alias.name == "*":
            _check_star_import(module, ctx)
            continue
        resolved = _attr_of_module(module, alias.name)
        _check_unchecked(f"{module}.{alias.name}", ctx)
        got = _from_import_level(module, alias.name, resolved, ctx)
        if got is not None:
            ctx.flag(*got)
        ctx.modules[alias.asname or alias.name] = resolved


def _check_imports(tree: ast.AST, ctx: _Ctx) -> None:
    """Все импорты в дереве: уровни за модули и привязки имён к модулям."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            _check_import(node, ctx)
        elif isinstance(node, ast.ImportFrom):
            _check_import_from(node, ctx)


# Безобидные встроенные из списка динамических: только «не знаю», без тревоги.
_HARMLESS_DYNAMIC = frozenset(("input", "help", "memoryview"))


def _dynamic(name: str, ctx: _Ctx) -> None:
    """Динамический код (eval, getattr…): разбор его не видит.

    Опасно — пусть человек посмотрит внимательно.
    """
    if name in _HARMLESS_DYNAMIC:
        ctx.flag(None, f"{name}()")
    else:
        ctx.flag(Level.DANGER, f"{name} — динамический код, разбор его не видит")


def _check_module_attr(module: str, attr: str, ctx: _Ctx) -> None:
    allowed = _MODULE_ATTRS.get(module)
    if allowed is None:
        return
    if attr in _DANGER_METHODS or (
        module == "os" and attr in {"system", "popen", "environ", "getenv", "environb"}
    ):
        reason = "окружение процесса (там ключи и токены)" if "env" in attr else "процессы"
        ctx.flag(Level.DANGER, f"{module}.{attr}: {reason}")
    elif attr not in allowed and attr not in _SAFE_DUNDERS:
        ctx.flag(None, f"{module}.{attr} не проверен классификатором")


# --------------------------------------------------------------------------- #
# Пути
# --------------------------------------------------------------------------- #
def _const_str(node: ast.AST) -> str | None:
    """Строка-литерал (или f-строка без подстановок); иначе None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        constants = [part for part in node.values if isinstance(part, ast.Constant)]
        if len(constants) == len(node.values):
            return "".join(str(part.value) for part in constants)
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
_NOT_SUBMODULES = frozenset((
    "ast.keyword",
    "ast.operator",
    "datetime.time",
    "locale.gettext",
    "unicodedata.decimal",
))


def _attr_of_module(base: str, attr: str) -> str:
    """«модуль.атрибут» — или сам модуль, если атрибут — импортированный модуль.

    Модуль, импортированный модулем стандартной библиотеки: ``glob.os`` и ``random._os`` —
    это ``os``, и их функции проверяются как функции ``os``.
    """
    full = f"{base}.{attr}"
    name = attr.lstrip("_")
    if full in SAFE_MODULES or full in DANGER_MODULES or full in _NOT_SUBMODULES:
        return full
    if base.split(".", maxsplit=1)[0] not in sys.stdlib_module_names:
        return full  # код сторонних модулей и так не проверяется (EXEC)
    if name == base.rpartition(".")[2] or name in _MODULE_ATTRS.get(base, ()):
        return full  # glob.glob, time.time, os.stat — функции, а не модули
    if name in sys.stdlib_module_names or name in DANGER_MODULES:
        return name
    return full


def _is_path_ctor(func: ast.AST, ctx: _Ctx) -> bool:
    name = _resolve_module(func, ctx)
    return name is not None and name.split(".")[-1] in _PATH_CTORS and name.startswith("pathlib")


def _join_inexact(
    static: Path,
    part: _Path | str | None,
    rest: Sequence[_Path | str | None],
) -> _Path:
    """К точному пути ``static`` присоединяется часть, известная неточно."""
    if isinstance(part, _Path) and part.under is not None:
        # имя или путь из обхода: результат — где-то в общем дереве
        common = os.path.commonpath([str(static), str(part.under)])
        return _join(_Path(under=Path(common)), rest)
    return _UNKNOWN_PATH


def _join_static(static: Path, parts: Sequence[_Path | str | None]) -> _Path:
    """Присоединить части к точному пути ``static``."""
    current = static
    for position, part in enumerate(parts):
        if isinstance(part, str):
            current /= part
        elif isinstance(part, _Path) and part.static is not None:
            current = part.static if part.static.is_absolute() else current / part.static
        else:
            return _join_inexact(static, part, parts[position + 1 :])
    return _Path(static=Path(os.path.normpath(current)))


def _join_under(base: _Path, parts: Sequence[_Path | str | None]) -> _Path:
    """Присоединить части к пути из обхода: результат — в том же дереве или неизвестен."""
    for part in parts:
        if isinstance(part, str):
            if ".." in PurePosixPath(part).parts or part.startswith("/"):
                return _UNKNOWN_PATH
        elif not (isinstance(part, _Path) and part.under is not None):
            return _UNKNOWN_PATH
    return _Path(under=base.under, pattern=base.pattern)


def _join(base: _Path, parts: Sequence[_Path | str | None]) -> _Path:
    """Путь ``base``, к которому присоединены ``parts`` (как ``os.path.join``)."""
    if base.static is not None:
        return _join_static(base.static, parts)
    if base.under is not None:
        return _join_under(base, parts)
    return _UNKNOWN_PATH


def _expand_home(text: str) -> Path:
    """Путь с раскрытым ``~``, как его раскроет ``os.path.expanduser`` при выполнении."""
    # Path.expanduser бросает RuntimeError для неизвестного ~user, os.path — нет.
    return Path(os.path.expanduser(text))  # noqa: PTH111


def _path_parts(args: Sequence[ast.expr], ctx: _Ctx, depth: int) -> list[str | _Path]:
    """Части пути из аргументов: литерал — как есть, остальное — через ``_path_of``."""
    return [_const_str(arg) or _path_of(arg, ctx, depth) for arg in args]


def _first_arg_text(node: ast.Call) -> str | None:
    """Первый позиционный аргумент — строка-литерал; иначе (и без аргументов) None."""
    return _const_str(node.args[0]) if node.args else None


def _first_arg_path(node: ast.Call, ctx: _Ctx, depth: int = 0) -> _Path:
    """Путь из первого позиционного аргумента; без аргументов — неизвестен."""
    return _path_of(node.args[0], ctx, depth) if node.args else _UNKNOWN_PATH


def _pattern_escapes(pattern: str | None, roots: str | tuple[str, ...] = "/") -> bool:
    """Шаблон обхода неизвестен или ведёт наружу: ``..`` или начало из ``roots``."""
    return pattern is None or ".." in pattern or pattern.startswith(roots)


def _path_of_text(text: str, ctx: _Ctx) -> _Path:
    """Путь из строки-литерала: относительный — от каталога запуска."""
    if text.startswith("~"):
        return _Path(static=_expand_home(text))
    path = Path(text)
    return _Path(static=Path(os.path.normpath(path if path.is_absolute() else ctx.cwd / path)))


def _merge_paths(found: list[_Path]) -> _Path:
    """Путь имени с несколькими значениями: общее у всех или неизвестен."""
    if not found:
        return _UNKNOWN_PATH
    if len(found) == 1:
        return found[0]
    static_values = {candidate.static for candidate in found}
    if all(candidate.static is not None for candidate in found) and len(static_values) == 1:
        return found[0]
    unders = []
    for candidate in found:
        if candidate.static is not None:
            static = candidate.static
            unders.append(static if static.is_dir() else static.parent)
        elif candidate.under is not None:
            unders.append(candidate.under)
        else:
            return _UNKNOWN_PATH
    return _Path(under=Path(os.path.commonpath([str(under) for under in unders])))


def _path_of_name(node: ast.Name, ctx: _Ctx, depth: int) -> _Path:
    """Путь в переменной: все её значения и обходы, ``__file__`` — файл скрипта."""
    if node.id == "__file__" and ctx.script is not None:
        return _Path(static=ctx.script)
    found: list[_Path] = [_path_of(bound, ctx, depth) for bound in ctx.bindings.get(node.id, [])]
    for source, index in ctx.iterations.get(node.id, []):
        found.append(_iterated(source, index, ctx, depth))
    return _merge_paths(found)


def _is_path_method(func: ast.AST, attr: str, ctx: _Ctx) -> bool:
    """``func`` — метод ``attr`` у класса или объекта pathlib: ``Path.cwd``, ``Path().home``."""
    return isinstance(func, ast.Attribute) and func.attr == attr and _is_path_ctor(func.value, ctx)


def _path_of_ctor(node: ast.Call, ctx: _Ctx, depth: int) -> _Path:
    """``Path(…)``, ``str(…)``, ``repr(…)``: путь из аргументов."""
    if not node.args:
        return _UNKNOWN_PATH if isinstance(node.func, ast.Name) else _Path(static=ctx.cwd)
    first = _first_arg_path(node, ctx, depth)
    rest = _path_parts(node.args[1:], ctx, depth)
    return _join(first, rest) if rest else first


def _path_of_expanduser(base: ast.expr, ctx: _Ctx) -> _Path:
    """``Path('~/x').expanduser()``: известен, только если основание — литерал."""
    if isinstance(base, ast.Call) and base.args:
        text = _const_str(base.args[0])
        if text is not None and _is_path_ctor(base.func, ctx):
            return _Path(static=_expand_home(text))
    return _UNKNOWN_PATH


def _path_with(method: str, base: _Path, node: ast.Call) -> _Path:
    """``p.with_suffix(…)``, ``p.with_name(…)``, ``p.with_stem(…)``."""
    arg = _first_arg_text(node)
    if arg is None or "/" in arg:
        return _UNKNOWN_PATH
    if base.static is None:
        return base
    if method == "with_suffix":
        return _Path(static=base.static.with_suffix(arg))
    if method == "with_name":
        return _Path(static=base.static.with_name(arg))
    return _Path(static=base.static.with_stem(arg))


def _path_of_method(node: ast.Call, func: ast.Attribute, ctx: _Ctx, depth: int) -> _Path:
    """Метод пути, дающий путь (``resolve``, ``joinpath``, ``with_suffix``…)."""
    if func.attr in {"resolve", "absolute"}:
        return _path_of(func.value, ctx, depth)
    if func.attr == "joinpath":
        base = _path_of(func.value, ctx, depth)
        return _join(base, _path_parts(node.args, ctx, depth))
    if func.attr in {"with_suffix", "with_name", "with_stem"}:
        return _path_with(func.attr, _path_of(func.value, ctx, depth), node)
    return _UNKNOWN_PATH


def _path_of_function(name: str | None, node: ast.Call, ctx: _Ctx, depth: int) -> _Path:
    """Функции ``os.path`` и методы путей (``name`` — функция модуля или None)."""
    if name in {"os.path.join", "posixpath.join"} and node.args:
        first = _first_arg_path(node, ctx, depth)
        return _join(first, _path_parts(node.args[1:], ctx, depth))
    if name in {"os.path.abspath", "os.path.normpath", "os.path.realpath"} and node.args:
        return _first_arg_path(node, ctx, depth)
    if isinstance(node.func, ast.Attribute):
        return _path_of_method(node, node.func, ctx, depth)
    return _UNKNOWN_PATH


def _path_of_call(node: ast.Call, ctx: _Ctx, depth: int) -> _Path:
    """Путь — результат вызова (``Path(…)``, ``Path.cwd()``, ``os.path.join(…)``…)."""
    func = node.func
    if _is_path_ctor(func, ctx) or (isinstance(func, ast.Name) and func.id in {"str", "repr"}):
        return _path_of_ctor(node, ctx, depth)
    name = _resolve_module(func, ctx)
    # Домашний каталог — вне проекта: запись туда должна быть видна как запись наружу.
    if name == "pathlib.Path.home" or _is_path_method(func, "home", ctx):
        return _Path(static=Path.home())
    if name == "os.path.expanduser" and node.args:
        text = _first_arg_text(node)
        return _Path(static=_expand_home(text)) if text else _UNKNOWN_PATH
    if isinstance(func, ast.Attribute) and func.attr == "expanduser":
        return _path_of_expanduser(func.value, ctx)
    if name in {"os.getcwd", "pathlib.Path.cwd"} or _is_path_method(func, "cwd", ctx):
        return _Path(static=ctx.cwd)
    return _path_of_function(name, node, ctx, depth)


def _path_of_operation(node: ast.AST, ctx: _Ctx, depth: int) -> _Path:
    """Путь из ``a / b``, ``p.parent``, ``entry.path``, ``files[0]``."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _path_of(node.left, ctx, depth)
        right = _const_str(node.right)
        return _join(left, [_path_of(node.right, ctx, depth) if right is None else right])
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        base = _path_of(node.value, ctx, depth)
        if base.static is not None:
            return _Path(static=base.static.parent)
        return _UNKNOWN_PATH
    if isinstance(node, ast.Attribute) and node.attr == "path":  # DirEntry.path
        return _path_of(node.value, ctx, depth)
    if isinstance(node, ast.Subscript):  # files[0] из обхода — там же, где обход
        inner = _path_of(node.value, ctx, depth)
        return _UNKNOWN_PATH if inner.under is None else _Path(under=inner.under)
    return _UNKNOWN_PATH


def _path_of(node: ast.AST, ctx: _Ctx, depth: int = 0) -> _Path:
    """Где путь, заданный выражением (литерал, Path(...), обход каталога…)."""
    if depth > _MAX_DEPTH:
        return _UNKNOWN_PATH
    text = _const_str(node)
    if text is not None:
        return _path_of_text(text, ctx)
    inner = depth + 1  # глубина вложенных выражений
    if isinstance(node, ast.Name):
        return _path_of_name(node, ctx, inner)
    if isinstance(node, ast.Call):
        return _path_of_call(node, ctx, inner)
    return _path_of_operation(node, ctx, inner)


# Обёртки, не меняющие, где лежат элементы обхода.
_ITERATION_WRAPPERS = frozenset((
    "sorted",
    "list",
    "tuple",
    "set",
    "reversed",
    "enumerate",
    "filter",
))


def _unwrap_iterable(source: ast.AST, index: int | None) -> ast.AST | None:
    """Источник элементов под ``sorted(…)``, ``enumerate(…)``…; None — элемент не путь."""
    node = source
    position = index
    # sorted(...), list(...), enumerate(...) и т.п. не меняют, где лежат элементы
    while (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _ITERATION_WRAPPERS
        and node.args
    ):
        if node.func.id == "enumerate":
            if position == 0:
                return None
            position = None
        node = node.args[-1] if node.func.id == "filter" else node.args[0]
    return node


def _iterated_literal(node: ast.AST, ctx: _Ctx, depth: int) -> _Path:
    """Обход списка путей: ``for p in [Path('a'), Path('b')]``."""
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        paths = [_path_of(element, ctx, depth) for element in node.elts]
        if paths and all(path.static is not None for path in paths):
            common = os.path.commonpath([str(path.static) for path in paths])
            return _Path(under=Path(common))
    return _UNKNOWN_PATH


def _listed_by_method(node: ast.Call, func: ast.Attribute, ctx: _Ctx, depth: int) -> _Path:
    """Элемент ``p.iterdir()``, ``p.glob(…)``, ``p.rglob(…)``, ``p.walk()``."""
    base = _path_of(func.value, ctx, depth)
    pattern = _first_arg_text(node) if node.args else "*"
    globbing = func.attr in {"glob", "rglob"}
    if globbing and _pattern_escapes(pattern):
        return _UNKNOWN_PATH
    target = base.static or base.under
    if target is None:
        return _UNKNOWN_PATH
    last = pattern.rsplit("/", 1)[-1] if globbing and pattern else None
    if last is not None and "**" in last:
        last = None
    return _Path(under=target, pattern=last)


def _listed_by_os(name: str, node: ast.Call, ctx: _Ctx, depth: int) -> _Path:
    """Элемент ``os.walk(…)``, ``os.scandir(…)``, ``os.listdir(…)``."""
    if not node.args:
        return _Path(under=ctx.cwd)
    base = _first_arg_path(node, ctx, depth)
    target = base.static or base.under
    if target is None:
        return _UNKNOWN_PATH
    if name == "os.listdir":
        # Имена, а не пути: открываются относительно каталога запуска.
        return _Path(under=target) if target == ctx.cwd else _Path(under=ctx.cwd)
    return _Path(under=target)


def _globbed(node: ast.Call, ctx: _Ctx) -> _Path:
    """Элемент ``glob.glob(…)``: пути относительно каталога запуска."""
    if _pattern_escapes(_first_arg_text(node), ("/", "~")):
        return _UNKNOWN_PATH
    if any(keyword.arg == "root_dir" for keyword in node.keywords):
        return _UNKNOWN_PATH
    return _Path(under=ctx.cwd)


def _iterated_call(node: ast.Call, ctx: _Ctx, depth: int) -> _Path:
    """Элемент обхода, заданного вызовом: ``rglob``, ``os.walk``, ``glob.glob``."""
    func = node.func
    name = _resolve_module(func, ctx)
    if (
        isinstance(func, ast.Attribute)
        and func.attr in _PATH_LISTERS
        and name is None
        and _looks_like_path(func.value, ctx)
    ):
        return _listed_by_method(node, func, ctx, depth)
    if name in {"os.walk", "os.scandir", "os.listdir"}:
        return _listed_by_os(name, node, ctx, depth)
    if name in {"glob.glob", "glob.iglob"} and node.args:
        return _globbed(node, ctx)
    return _UNKNOWN_PATH


def _iterated(source: ast.AST, index: int | None, ctx: _Ctx, depth: int) -> _Path:
    """Элемент обхода: ``for p in Path('src').rglob('*.py')``, ``os.walk('.')``, ``glob.glob``."""
    node = _unwrap_iterable(source, index)
    if node is None:
        return _UNKNOWN_PATH
    inner = depth + 1  # глубина вложенных выражений
    if isinstance(node, ast.Name):  # for f in files — files из обхода
        found = _path_of(node, ctx, inner)
        return _Path(under=found.under, pattern=found.pattern) if found.under else _UNKNOWN_PATH
    if isinstance(node, ast.Call):
        return _iterated_call(node, ctx, inner)
    return _iterated_literal(node, ctx, inner)


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
    cached = ctx.hazard_cache.get(key)
    if cached is not None:
        return cached
    found = (
        not _inside(directory, ctx)
        or has_secrets_under(directory, ctx.root, name_pattern=pattern)
        or has_escaping_links(directory, ctx.root, name_pattern=pattern)
    )
    ctx.hazard_cache[key] = found
    return found


def _check_read(target: _Path, ctx: _Ctx, what: str) -> None:
    """Чтение пути: секреты и пути вне проекта — тревога или «не знаю»."""
    if target.static is not None:
        scope = scope_of_path(target.static, ctx.root)
        if scope is Scope.SECRET_OUTSIDE:
            ctx.flag(Level.DANGER, f"{what}: секреты вне проекта")
        elif scope is Scope.SECRET:
            ctx.flag(None, f"{what}: файл с секретами")
        elif scope is not Scope.INSIDE and scope is not Scope.SINK:
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
    """Запись по пути: внутри проекта — WRITE, наружу и в служебные папки — DANGER."""
    if recursive:
        ctx.flag(Level.DANGER, f"{what}: рекурсивное изменение")
        return
    if target.static is None:
        ctx.flag(None, f"{what}: путь известен только при выполнении")
        return
    if not _inside(target.static, ctx):
        ctx.flag(Level.DANGER, f"{what}: запись вне проекта")
        return
    if is_vcs_internal(target.static.resolve(), ctx.root):
        ctx.flag(Level.DANGER, f"{what}: служебные файлы git или jj")
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


def _callable_known(name: str, ctx: _Ctx) -> bool:
    """Вызов имени без проверки: безопасное встроенное или своя функция/лямбда/класс."""
    if name in _SAFE_BUILTINS:
        return True
    # своя функция/лямбда/класс — её тело проверяется целиком
    return name in ctx.bindings or name in ctx.iterations or _defined(name, ctx)


def _check_name_call(name: str, node: ast.Call, ctx: _Ctx) -> None:
    """Вызов по имени: ``open(…)``, ``eval(…)``, псевдоним функции модуля, своя функция."""
    bound = ctx.modules.get(name)
    if bound is not None:
        _check_module_function(bound, node, ctx)
        return
    if name in _DYNAMIC_BUILTINS:
        _dynamic(name, ctx)
    elif name == "open":
        _check_open(node, _first_arg_path(node, ctx), 1, ctx, "open()")
    elif name == "type" and len(node.args) > 1:
        ctx.flag(Level.DANGER, "type() с тремя аргументами — класс собирается при выполнении")
    elif not _callable_known(name, ctx):
        ctx.flag(None, f"вызов {name}() не проверен классификатором")


def _check_path_write(attr: str, func: ast.Attribute, node: ast.Call, ctx: _Ctx) -> None:
    """Метод пути, меняющий файл: ``p.write_text(…)``, ``p.rename(…)``, ``p.copy(…)``."""
    what = f".{attr}()"
    target = _path_of(func.value, ctx)
    _check_write(target, ctx, what, recursive=False)
    if attr in _PATH_MOVERS:
        _check_destination(node, ctx, what)
    if attr.startswith("copy"):
        _check_read(target, ctx, what)  # копия секрета — его чтение


def _check_path_io(attr: str, func: ast.Attribute, node: ast.Call, ctx: _Ctx) -> None:
    """Методы путей, читающие и меняющие файлы (``write_text``, ``rename``, ``read_text``…)."""
    what = f".{attr}()"
    if attr in _PATH_WRITERS and _looks_like_path(func.value, ctx):
        _check_path_write(attr, func, node, ctx)
    elif attr in _PATH_READERS:
        _check_read(_path_of(func.value, ctx), ctx, what)
    elif attr in _PATH_ONLY_WRITERS or (attr in _PATH_MOVERS and _one_target(node)):
        ctx.flag(None, f"{what}: объект неизвестен — возможно, запись в файл")
        if attr in _PATH_MOVERS:
            _check_destination(node, ctx, what)


def _check_listing(attr: str, func: ast.Attribute, node: ast.Call, ctx: _Ctx) -> None:
    """Обход каталога методом пути: ``p.iterdir()``, ``p.glob(…)``, ``p.rglob(…)``."""
    _check_list(_path_of(func.value, ctx), ctx, f".{attr}()")
    if attr in {"glob", "rglob"} and _pattern_escapes(_first_arg_text(node)):
        ctx.flag(None, f".{attr}(): шаблон неизвестен или ведёт наружу")


def _check_open_or_list(attr: str, func: ast.Attribute, node: ast.Call, ctx: _Ctx) -> None:
    """``x.open(…)`` и обход каталога методом пути."""
    if attr == "open" and _looks_like_path(func.value, ctx):
        _check_open(node, _path_of(func.value, ctx), 0, ctx, ".open()")
    elif attr == "open" and _opens_for_write(node):
        ctx.flag(None, ".open(): объект неизвестен, режим — запись")
    elif attr in _PATH_LISTERS and _looks_like_path(func.value, ctx):
        _check_listing(attr, func, node, ctx)


def _check_format(func: ast.Attribute, ctx: _Ctx) -> None:
    """``шаблон.format(…)``: поля шаблона могут обходить атрибуты объектов."""
    # Поля шаблона ({0.__class__}, {0[key]}) обходят атрибуты при выполнении — ast их
    # не видит. Литерал проверяем, шаблон из переменной — «не знаю».
    text = func.value.value if isinstance(func.value, ast.Constant) else None
    if isinstance(text, str) and _format_traverses(text):
        ctx.flag(Level.DANGER, "str.format обходит атрибуты объектов ({0.x}, {0[k]})")
    elif not isinstance(text, str):
        ctx.flag(None, "str.format: шаблон не литерал")


def _check_other_method(attr: str, func: ast.Attribute, ctx: _Ctx) -> None:
    """Прочие методы объектов: ``expanduser``, ``home``, ``format``, ``evaluate``."""
    if attr in {"expanduser", "home"}:
        ctx.flag(None, f".{attr}() — путь вне проекта")
    elif attr == "evaluate":  # typing.ForwardRef.evaluate() исполняет строку
        ctx.flag(None, ".evaluate() — возможно, строка исполняется как код")
    elif attr == "format":
        _check_format(func, ctx)


def _check_method_call(func: ast.Attribute, node: ast.Call, ctx: _Ctx) -> None:
    """Вызов метода объекта, а не модуля (``p.write_text(…)``, ``x.open(…)``…)."""
    attr = func.attr
    if attr in _DANGER_METHODS:
        ctx.flag(Level.DANGER, f".{attr}()")
    elif attr in _PATH_WRITERS or attr in _PATH_READERS:
        _check_path_io(attr, func, node, ctx)
    elif attr == "open" or attr in _PATH_LISTERS:
        _check_open_or_list(attr, func, node, ctx)
    else:
        _check_other_method(attr, func, ctx)


def _unpacked_keys(node: ast.AST) -> frozenset[str] | None:
    """Ключи в ``**{'a': …}``; None — не литерал или ключ известен только при выполнении."""
    if not isinstance(node, ast.Dict):
        return None
    keys = [_const_str(key) for key in node.keys if key is not None]
    if len(keys) != len(node.keys) or None in keys:
        return None
    return frozenset(key for key in keys if key is not None)


def _callee_name(node: ast.Call, ctx: _Ctx) -> str | None:
    """Последняя часть имени вызываемого: ``signature`` у ``inspect.signature`` и псевдонима."""
    resolved = _resolve_module(node.func, ctx)
    if resolved is not None:
        return resolved.rpartition(".")[2]
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return node.func.id if isinstance(node.func, ast.Name) else None


def _check_eval_str(node: ast.Call, ctx: _Ctx) -> None:
    """``eval_str=True`` (``inspect.signature``, ``get_annotations``): аннотации — код."""
    for keyword in node.keywords:
        names = frozenset((keyword.arg,)) if keyword.arg else _unpacked_keys(keyword.value)
        if names is not None and "eval_str" in names:
            ctx.flag(Level.DANGER, "eval_str: строки-аннотации исполняются как код")
        elif names is None and _callee_name(node, ctx) in _ANNOTATION_READERS:
            ctx.flag(None, "аргументы ** известны только при выполнении (eval_str?)")


def _check_call(node: ast.Call, ctx: _Ctx) -> None:
    """Вызов функции, функции модуля или метода."""
    _check_eval_str(node, ctx)
    func = node.func
    if isinstance(func, ast.Name):
        _check_name_call(func.id, node, ctx)
        return
    if isinstance(func, ast.Attribute):
        module = _resolve_module(func, ctx)
        if module is not None:  # функция модуля: os.remove, json.loads, Path.cwd
            _check_module_function(module, node, ctx)
            return
        _check_method_call(func, node, ctx)
        return
    ctx.flag(None, "вызов результата выражения")


def _one_target(node: ast.Call) -> bool:
    """Вызов как ``p.rename(target)``: один позиционный аргумент или ``target=``.

    У ``str.replace`` их два, у ``dict.copy`` — ни одного.
    """
    if len(node.args) == 1:
        return True
    return not node.args and any(kw.arg in _TARGET_KWARGS for kw in node.keywords)


def _check_destination(node: ast.Call, ctx: _Ctx, what: str) -> None:
    """Куда переносит или копирует ``p.rename/replace/copy/move(_into)``."""
    dest = _call_arg(node, 0, *_TARGET_KWARGS)
    _check_write(_UNKNOWN_PATH if dest is None else _path_of(dest, ctx), ctx, what)


def _call_arg(node: ast.Call, index: int, *names: str) -> ast.AST | None:
    """Аргумент вызова — позиционный или именованный.

    ``*args`` и ``**kwargs`` могут подставить любой: тогда значение известно только при
    выполнении.
    """
    starred = any(isinstance(arg, ast.Starred) for arg in node.args[: index + 1])
    if starred or any(kw.arg is None for kw in node.keywords):
        return _RUNTIME_ARG
    if len(node.args) > index:
        return node.args[index]
    for kw in node.keywords:
        if kw.arg in names:
            return kw.value
    return None


def _opens_for_write(node: ast.Call) -> bool:
    """``x.open(...)`` у объекта неизвестного типа: режим записи или неизвестный режим."""
    for keyword in node.keywords:
        if keyword.arg == "mode" and not isinstance(_const_str(keyword.value), str):
            return True
    candidates = (*node.args, *(named.value for named in node.keywords))
    modes = [_const_str(candidate) for candidate in candidates]
    return any(
        isinstance(mode, str) and _MODE_TEXT.fullmatch(mode) and _WRITE_MODES & set(mode)
        for mode in modes
    )


def _defined(name: str, ctx: _Ctx) -> bool:
    """Имя присвоено в коде или определено (def, class)."""
    return name in ctx.bindings or name in ctx.defs


def _known_name(name: str, ctx: _Ctx) -> bool:
    """Имя связано в коде (импорт, присваивание, def) или встроенное."""
    if name in ctx.modules or name in ctx.iterations or _defined(name, ctx):
        return True
    return name in _BUILTIN_NAMES or name in _SAFE_DUNDERS


def _call_gives_path(node: ast.Call, ctx: _Ctx, depth: int) -> bool:
    """Вызов возвращает объект pathlib: ``Path(…)``, ``p.resolve()``, ``Path.cwd()``."""
    func = node.func
    if _is_path_ctor(func, ctx):
        return True
    if isinstance(func, ast.Attribute) and (
        func.attr in _PATH_SAME or func.attr in {"cwd", "home"}
    ):
        return _looks_like_path(func.value, ctx, depth) or _is_path_ctor(func.value, ctx)
    return False


def _is_listing(source: ast.AST) -> bool:
    """Обход — перечисление каталога методом пути (под ``sorted(…)``, ``list(…)``…)."""
    node = source
    while (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"sorted", "list", "tuple", "set", "reversed"}
        and node.args
    ):
        node = node.args[0]
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _PATH_LISTERS
    )


def _name_holds_path(name: str, ctx: _Ctx, depth: int) -> bool:
    """В переменной — объект pathlib: одно из значений или элемент обхода каталога."""
    bound = ctx.bindings.get(name, [])
    sources = ctx.iterations.get(name, [])
    if any(_looks_like_path(candidate, ctx, depth) for candidate in bound):
        return True
    return any(_is_listing(source) for source, _ in sources)


def _looks_like_path(node: ast.AST, ctx: _Ctx, depth: int = 0) -> bool:
    """Выражение — объект pathlib (а не строка: у str тоже есть ``replace``)."""
    if depth > _MAX_DEPTH:
        return False
    inner = depth + 1  # глубина вложенных выражений
    if isinstance(node, ast.Call):
        return _call_gives_path(node, ctx, inner)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _looks_like_path(node.left, ctx, inner)
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        return _looks_like_path(node.value, ctx, inner)
    if isinstance(node, ast.Name):
        return _name_holds_path(node.id, ctx, inner)
    return False


def _format_traverses(text: str) -> bool:
    """Шаблон ``str.format`` обходит атрибуты или элементы (``{0.x}``, ``{0[k]}``)."""
    try:
        for _, field_name, _, _ in string.Formatter().parse(text):
            if field_name and any(ch in field_name for ch in ".["):
                return True
    except ValueError:
        return True
    return False


def _uses_dir_fd(node: ast.Call) -> bool:
    """Путь задан относительно открытого каталога (``dir_fd=…`` не None)."""
    return any(
        keyword.arg in {"dir_fd", "src_dir_fd", "dst_dir_fd"}
        and not (isinstance(keyword.value, ast.Constant) and keyword.value.value is None)
        for keyword in node.keywords
    )


def _check_file_change(attr: str, what: str, node: ast.Call, ctx: _Ctx) -> None:
    """``os.remove``, ``os.rename``, ``shutil.copy``…: запись (и чтение у копий)."""
    src = _call_arg(node, 0, "path", "name", "src")
    target = _UNKNOWN_PATH if src is None else _path_of(src, ctx)
    _check_write(target, ctx, what, recursive=attr in _RECURSIVE)
    if attr in {"rename", "replace", "copy", "copy2", "copyfile", "move"}:
        dst = _call_arg(node, 1, "dst")
        if dst is not None:
            _check_write(_path_of(dst, ctx), ctx, what)
        if attr.startswith("copy"):
            _check_read(target, ctx, what)
    if _uses_dir_fd(node):
        ctx.flag(None, f"{what}: путь относительно открытого каталога")


def _check_os_function(module: str, attr: str, node: ast.Call, ctx: _Ctx) -> bool:
    """Функции ``os`` и ``shutil``, меняющие файлы или перечисляющие каталог."""
    what = f"{module}.{attr}()"
    if module in {"os", "shutil"} and attr in _FILE_WRITERS | _RECURSIVE:
        _check_file_change(attr, what, node, ctx)
        return True
    if module == "os" and attr in {"listdir", "scandir", "walk"}:
        target = _first_arg_path(node, ctx) if node.args else _Path(static=ctx.cwd)
        _check_list(target, ctx, what)
        return True
    return False


def _check_glob(full: str, node: ast.Call, ctx: _Ctx) -> None:
    """``glob.glob(…)``, ``glob.iglob(…)``: шаблон и ``root_dir``."""
    if _pattern_escapes(_first_arg_text(node), ("/", "~")):
        ctx.flag(None, f"{full}(): шаблон неизвестен или ведёт наружу")
    if any(keyword.arg == "root_dir" for keyword in node.keywords):
        ctx.flag(None, f"{full}(root_dir=…)")


def _check_stream_function(module: str, attr: str, node: ast.Call, ctx: _Ctx) -> bool:
    """Открытие файлов (``io.open``, ``codecs.open``, ``tokenize.open``) и ``glob``."""
    what = f"{module}.{attr}()"
    if module in {"io", "codecs"} and attr == "open":
        _check_open(node, _first_arg_path(node, ctx), 1, ctx, what)
    elif module == "tokenize" and attr == "open":
        _check_read(_first_arg_path(node, ctx), ctx, what)
    elif module == "glob" and attr in {"glob", "iglob"}:
        _check_glob(f"{module}.{attr}", node, ctx)
    else:
        return False
    return True


def _check_unbound_method(attr: str, node: ast.Call, ctx: _Ctx) -> None:
    """Метод, вызванный у класса: ``Path.write_text(p, 'x')`` — то же, что ``p.write_text('x')``."""
    receiver = node.args[0] if node.args else _RUNTIME_ARG
    method = ast.Attribute(value=receiver, attr=attr, ctx=ast.Load())
    _check_call(ast.Call(func=method, args=node.args[1:], keywords=node.keywords), ctx)


def _check_path_function(module: str, attr: str, node: ast.Call, ctx: _Ctx) -> bool:
    """Функции путей: ``os.path.expanduser``, ``Path.home``, ``Path.write_text(p, …)``."""
    outside = module in {"os.path", "posixpath"} and attr in {"expanduser", "expandvars"}
    if outside or (module.startswith("pathlib") and attr == "home"):
        ctx.flag(None, f"{module}.{attr}() — путь вне проекта")
        return True
    if (
        module.startswith("pathlib")
        and module.rpartition(".")[2] in _PATH_CTORS
        and attr in _PATH_WRITERS | _PATH_READERS | _PATH_LISTERS | {"open"}
    ):
        _check_unbound_method(attr, node, ctx)
        return True
    return False


def _check_module_function(full: str, node: ast.Call, ctx: _Ctx) -> None:
    """Вызов функции модуля ``full`` («модуль.функция»)."""
    module, _, attr = full.rpartition(".")
    if not module:
        return
    if module in DANGER_MODULES or module.split(".")[0] in DANGER_MODULES:
        ctx.flag(Level.DANGER, f"{full}()")
        return
    for check in (_check_os_function, _check_stream_function, _check_path_function):
        if check(module, attr, node, ctx):
            return
    if module in {"json", "tomllib", "csv", "pickle"} and attr == "dump":
        return  # файл открыт open() — проверен там
    allowed = _MODULE_ATTRS.get(module)
    if allowed is not None and attr not in allowed:
        ctx.flag(None, f"{full}() не проверен классификатором")


def _alias_target(bound: list[ast.AST], ctx: _Ctx) -> str | None:
    """Модуль (или его атрибут), на который указывают все значения имени; иначе None."""
    resolved = {_resolve_module(candidate, ctx) for candidate in bound}
    return resolved.pop() if len(resolved) == 1 else None


def _collect_aliases(ctx: _Ctx) -> None:
    """``rm = os.remove``, ``P = Path``: имя — псевдоним модуля или его атрибута.

    Вызовы через псевдоним проверяются как вызовы оригинала. Только если других значений у
    имени нет (иначе ``p.unlink()`` для ``p`` из обхода стал бы «функцией модуля»).
    """
    changed = True
    while changed:  # псевдоним псевдонима
        changed = False
        for name, bound in ctx.bindings.items():
            if name in ctx.modules or name in ctx.iterations or name in ctx.defs:
                continue
            target = _alias_target(bound, ctx)
            if target is not None:
                ctx.modules[name] = target
                changed = True


def _is_alias(node: ast.AST, parent: ast.AST | None, ctx: _Ctx) -> bool:
    """``node`` — значение присваивания псевдониму, который уже указывает туда же."""
    if (
        not isinstance(parent, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
        or parent.value is not node
    ):
        return False
    targets = parent.targets if isinstance(parent, ast.Assign) else [parent.target]
    resolved = _resolve_module(node, ctx)
    return all(
        isinstance(target, ast.Name) and ctx.modules.get(target.id) == resolved
        for target in targets
    )


# Модули с функциями записи и процессов (и sys.path): переданное куда-то — не проследить.
_ESCAPING_MODULES = frozenset((*_MODULE_ATTRS, "pathlib", "sys.path"))
# Атрибуты, которые проверяются в месте вызова: как значения — проверяем вызов с
# аргументами, известными только при выполнении.
_REF_ATTRS = _PATH_WRITERS | _PATH_READERS | _PATH_LISTERS | _DANGER_METHODS | {"open", "format"}
_RUNTIME_ARG = ast.Name(id="<при выполнении>")


def _harmless_use(node: ast.AST, parent: ast.AST | None, ctx: _Ctx) -> bool:
    r"""Значение только читают: ``print(os)``, ``for p in sys.path``, ``'\\n'.join(...)``."""
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


def _reference_name(node: ast.Name | ast.Attribute, resolved: str | None) -> str:
    """Как назвать ссылку в причине: модуль.функция, имя или ``.метод``."""
    if resolved:
        return resolved
    if isinstance(node, ast.Name):
        return node.id
    return f".{node.attr}"


def _is_callable_reference(
    node: ast.Name | ast.Attribute,
    resolved: str | None,
    ctx: _Ctx,
) -> bool:
    """Ссылку проверяем как вызов: функция модуля, ``open`` или метод из ``_REF_ATTRS``."""
    if isinstance(node, ast.Name):
        return resolved is not None or (node.id == "open" and not _defined(node.id, ctx))
    if node.attr in _SAFE_DUNDERS:
        return False  # os.__file__ — описание модуля, не функция
    return resolved is not None or node.attr in _REF_ATTRS


def _check_as_call(node: ast.Name | ast.Attribute, resolved: str | None, ctx: _Ctx) -> None:
    """Проверить ссылку как вызов с аргументами, известными только при выполнении."""
    start = len(ctx.levels)
    _check_call(ast.Call(func=node, args=[_RUNTIME_ARG], keywords=[]), ctx)
    found = ctx.levels[start:]
    ctx.levels = ctx.levels[:start]
    if found:
        what = _reference_name(node, resolved)
        level = worst(*(found_level for found_level, _ in found))
        ctx.flag(level, f"{what} передана как значение: аргументы известны только при выполнении")


def _check_reference(
    node: ast.Name | ast.Attribute,
    parent: ast.AST | None,
    ctx: _Ctx,
) -> None:
    """Функция или модуль как значение: проверки в месте вызова их не увидят.

    Например, ``map(os.remove, …)``, ``o = open``, ``f(os)``.
    """
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
    if _is_callable_reference(node, resolved, ctx):
        _check_as_call(node, resolved, ctx)


def _is_dunder(name: str) -> bool:
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


def _builtin_call(call: ast.AST | None, names: Collection[str], ctx: _Ctx) -> bool:
    """Вызов встроенной из ``names`` (имя не переопределено в коде)."""
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        return False
    name = call.func.id
    return name in names and name not in ctx.modules and not _defined(name, ctx)


def _inspector_call(node: ast.Call, ctx: _Ctx) -> bool:
    """Вызов, который только показывает или проверяет аргумент: ``print``, ``inspect.signature``."""
    resolved = _resolve_module(node.func, ctx)
    if resolved is not None:
        return resolved in _INSPECT_READERS
    return _builtin_call(node, _INSPECTORS, ctx)


def _inspected(node: ast.AST, parent: ast.AST | None, ctx: _Ctx) -> bool:
    """Значение только осматривают: печатают, сравнивают, передают в ``inspect.signature``."""
    if isinstance(parent, ast.Call):
        return node in parent.args and _inspector_call(parent, ctx)
    return isinstance(parent, (ast.Compare, ast.FormattedValue, ast.Expr))


def _names_only(node: ast.AST, parent: ast.AST | None, ctx: _Ctx) -> bool:
    """От пространства имён берут только имена: ``sorted(X.__dict__)``, ``X.__dict__.keys()``."""
    if isinstance(parent, ast.Call):
        return node in parent.args and _builtin_call(parent, _NAME_LISTERS, ctx)
    if isinstance(parent, ast.Attribute):
        return parent.attr == "keys"
    return isinstance(parent, (ast.Compare, ast.Expr))


def _super_call(node: ast.Attribute, parent: ast.AST | None, ctx: _Ctx) -> bool:
    """``super().__init__(…)`` в своём классе — метод базового класса для своего объекта."""
    if node.attr not in _SUPER_DUNDERS or not isinstance(parent, ast.Call):
        return False
    base = node.value
    called = parent.func is node
    return (
        called
        and isinstance(base, ast.Call)
        and not base.args
        and _builtin_call(base, ("super",), ctx)
    )


def _dunder_read(node: ast.Attribute, parent: ast.AST | None, ctx: _Ctx) -> bool:
    """Чтение ``__dunder__`` безопасно: описание объекта или только осмотр."""
    attr = node.attr
    if attr in _SAFE_DUNDERS:
        return True
    if attr in _ESCAPE_DUNDERS:
        return False
    if attr == _NAMESPACE_DUNDER:
        return _names_only(node, parent, ctx)
    # основание атрибута-dunder проверяется вместе с ним: X.__init__.__doc__
    described = isinstance(parent, ast.Attribute) and (
        _is_dunder(parent.attr) or parent.attr in _CODE_FIELDS
    )
    return described or _inspected(node, parent, ctx) or _super_call(node, parent, ctx)


def _check_dunder(node: ast.Attribute, parent: ast.AST | None, ctx: _Ctx) -> None:
    """``__dunder__``: описание объекта — чтение; методы и пространства имён — только осмотр."""
    if not isinstance(node.ctx, ast.Load):
        ctx.flag(Level.DANGER, f"изменение {node.attr} — внутренности интерпретатора")
    elif not _dunder_read(node, parent, ctx):
        ctx.flag(Level.DANGER, f"доступ к {node.attr} — внутренности интерпретатора")


def _check_unchecked(full: str, ctx: _Ctx) -> None:
    """Член «безопасного» модуля, который не только читает (``argparse.FileType``)."""
    reason = _UNCHECKED_MEMBERS.get(full)
    if reason is not None:
        ctx.flag(None, f"{full}: {reason}")


def _check_module_member(module: str, attr: str, ctx: _Ctx) -> None:
    """Атрибут модуля: ``glob.os``, ``os.environ``, подмодуль не из ``import a.b``."""
    resolved = _attr_of_module(module, attr)
    if resolved != f"{module}.{attr}":  # glob.os — импорт модуля os
        got = _module_level(resolved, ctx)
        if got is not None:
            ctx.flag(*got)
    _check_module_attr(module, attr, ctx)
    _check_unchecked(f"{module}.{attr}", ctx)
    if module in ctx.partial and attr not in ctx.partial[module]:
        if attr in {"import_module", "__import__", "reload", "util", "machinery"}:
            ctx.flag(Level.DANGER, f"{module}.{attr} — импорт при выполнении")
        else:
            ctx.flag(None, f"{module}.{attr} не проверен классификатором")


def _check_attribute(node: ast.Attribute, parent: ast.AST | None, ctx: _Ctx) -> None:
    """Доступ к атрибуту: внутренности интерпретатора и атрибуты модулей."""
    attr = node.attr
    if _is_dunder(attr):
        _check_dunder(node, parent, ctx)
    elif attr in _INTERNALS:
        ctx.flag(Level.DANGER, f"доступ к {attr} — внутренности интерпретатора")
    module = _resolve_module(node.value, ctx)
    if module is not None:
        _check_module_member(module, attr, ctx)
    if module == "sys.path":  # и from sys import path, и p = sys.path
        ctx.flag(Level.DANGER, "изменение sys.path — подмена модулей")


def _check_name(node: ast.Name, ctx: _Ctx) -> None:
    """Имя: динамические встроенные, ``__dunder__``, имена из ``from … import *``."""
    if node.id in _DYNAMIC_BUILTINS:
        _dynamic(node.id, ctx)
    elif _is_dunder(node.id) and node.id not in _SAFE_DUNDERS:
        ctx.flag(Level.DANGER, f"доступ к {node.id} — внутренности интерпретатора")
    elif ctx.star and not _known_name(node.id, ctx):
        # from platform import * приносит и os: что за объект — не узнать
        ctx.flag(None, f"{node.id}: имя из from … import *, не проверено")


# --------------------------------------------------------------------------- #
# Класс объекта как значение
# --------------------------------------------------------------------------- #
def _type_of_object(node: ast.Call, ctx: _Ctx) -> bool:
    """``type(x)`` с одним аргументом — класс объекта."""
    return _builtin_call(node, ("type",), ctx) and len(node.args) == 1 and not node.keywords


def _gives_classes(node: ast.Call, ctx: _Ctx) -> bool:
    """Вызов возвращает класс или классы: ``type(x)``, ``X.mro()``, ``inspect.getmro(X)``."""
    if isinstance(node.func, ast.Attribute) and node.func.attr == "mro":
        return True
    return _type_of_object(node, ctx) or _resolve_module(node.func, ctx) == "inspect.getmro"


def _class_value(expr: ast.AST, ctx: _Ctx, depth: int = 0) -> bool:
    """Выражение — класс объекта или кортеж классов (``type(x)``, ``X.__mro__``).

    Вызов такого класса не проверить: ``type(open('a', 'rb').raw)`` — FileIO, и
    ``FileIO('/home/u/.bashrc', 'w')`` обнуляет файл. Имя — переменная обхода классов.
    """
    if depth > _MAX_DEPTH:
        return True
    if isinstance(expr, ast.Attribute):
        return expr.attr in _CLASS_DUNDERS
    if isinstance(expr, ast.Subscript):
        return _class_value(expr.value, ctx, depth + 1)
    if isinstance(expr, ast.Call):
        return _gives_classes(expr, ctx)
    if isinstance(expr, ast.Name):
        for source, index in ctx.iterations.get(expr.id, ()):
            element = _unwrap_iterable(source, index)
            if element is not None and _class_value(element, ctx, depth + 1):
                return True
    return False


def _loop_source(node: ast.AST, parent: ast.AST | None) -> bool:
    """``node`` — то, что обходит ``for`` или включение."""
    return isinstance(parent, (ast.For, ast.AsyncFor, ast.comprehension)) and parent.iter is node


def _iteration_source(node: ast.AST, parent: ast.AST | None, ctx: _Ctx) -> bool:
    """``for c in X.__mro__``, ``enumerate(X.__mro__)``: элементы проверяются как имя цикла."""
    if isinstance(parent, ast.Call) and node in parent.args:
        wrapped = _builtin_call(parent, ("enumerate", "reversed"), ctx)
        return wrapped and _loop_source(parent, ctx.parents.get(id(parent)))
    return _loop_source(node, parent)


def _check_class_use(expr: ast.AST, parent: ast.AST | None, ctx: _Ctx) -> None:
    """Класс объекта — только для осмотра: атрибут, ``print``, ``isinstance``, сравнение."""
    if isinstance(expr, (ast.Name, ast.Attribute, ast.Subscript)):
        if not isinstance(expr.ctx, ast.Load):
            return
    elif not isinstance(expr, ast.Call):
        return
    if not _class_value(expr, ctx):
        return
    based = isinstance(parent, (ast.Attribute, ast.Subscript)) and parent.value is expr
    if not (based or _inspected(expr, parent, ctx) or _iteration_source(expr, parent, ctx)):
        ctx.flag(None, "класс объекта как значение: что с ним сделают, не видно")


def _assigned_module(target: ast.AST, ctx: _Ctx) -> str | None:
    """Модуль, чей атрибут или элемент меняет цель присваивания (``os.x = …``); иначе None."""
    if not isinstance(target, (ast.Attribute, ast.Subscript)):
        return None
    base = target.value
    while isinstance(base, (ast.Attribute, ast.Subscript)):
        base = base.value
    return _resolve_module(base, ctx) if isinstance(base, ast.Name) else None


def _check_store(node: ast.AST, ctx: _Ctx) -> None:
    """Изменение модулей (``os.x = …``, ``del os.x``) и ``os.environ[…]``."""
    if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
        targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
        for target in targets:
            module = _assigned_module(target, ctx)
            if module is not None:
                ctx.flag(Level.DANGER, f"изменение модуля {module}")
    elif (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "environ"
    ):
        ctx.flag(Level.DANGER, "os.environ: окружение процесса (там ключи и токены)")


def _check_attributes(tree: ast.AST, ctx: _Ctx) -> None:
    """Обход дерева: ссылки, атрибуты, имена, вызовы и изменения модулей."""
    parents = {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    ctx.parents = parents
    for node in ast.walk(tree):
        parent = parents.get(id(node))
        if isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, ast.Load):
            _check_reference(node, parent, ctx)
        _check_class_use(node, parent, ctx)
        if isinstance(node, ast.Attribute):
            _check_attribute(node, parent, ctx)
        elif isinstance(node, ast.Name):
            _check_name(node, ctx)
        elif isinstance(node, ast.Call):
            _check_call(node, ctx)
        else:
            _check_store(node, ctx)


def _shadows(directory: Path, entry: str) -> bool:
    """Файл или пакет ``entry`` в ``directory`` затеняет модуль стандартной библиотеки."""
    stem, dot, ext = entry.partition(".")
    if not stem or stem not in sys.stdlib_module_names:
        return False
    if not dot:
        package = directory / entry
        return (package / "__init__.py").exists() or (package / "__init__.pyc").exists()
    return ext in {"py", "pyc", "pyw"} or ext.endswith(("so", "pyd"))


def shadowed_stdlib(directory: Path) -> str | None:
    """Модуль стандартной библиотеки, затенённый файлом в ``directory`` (``json.py``)."""
    try:
        entries = [child.name for child in directory.iterdir()]
    except OSError:
        return None
    for entry in entries:
        if _shadows(directory, entry):
            return entry
    return None


def _verdict(levels: list[tuple[Level | None, str]]) -> PyResult:
    """Итог по флагам: худший уровень и первая причина этого уровня."""
    if not levels:
        return PyResult(Level.READ, "python: только чтение и вычисления")
    level = worst(*(lv for lv, _ in levels))
    # причина — первая из худшего уровня
    for lv, reason in levels:
        if lv == level:
            return PyResult(level, f"python: {reason}")
    _, first_reason = levels[0]
    return PyResult(level, f"python: {first_reason}")


def analyze(
    code: str,
    *,
    root: Path,
    cwd: str = ".",
    script: Path | None = None,
    path_dirs: tuple[Path, ...] = (),
) -> PyResult:
    """Оценить Python-код.

    ``script`` — файл скрипта (для ``__file__`` и затенения), ``path_dirs`` — каталоги из
    PYTHONPATH (тоже проверяются на затенение); без ``script`` первым в ``sys.path`` стоит
    каталог запуска.
    """
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
    first = run_dir if script is None else script.parent
    for directory in (first, *path_dirs):
        shadow = shadowed_stdlib(directory)
        if shadow is not None:
            ctx.flag(
                Level.EXEC,
                f"{shadow} в каталоге запуска подменит модуль стандартной библиотеки",
            )
    return _verdict(ctx.levels)
