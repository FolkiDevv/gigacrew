"""Статический анализ Python-кода из команд (python -c, heredoc, скрипты)."""

from __future__ import annotations

from pathlib import Path

import pytest

from gigacrew.safety import CommandLevel, classify
from gigacrew.safety.python import analyze, shadowed_stdlib

READ, EXEC, WRITE, DANGER = (
    CommandLevel.READ,
    CommandLevel.EXEC,
    CommandLevel.WRITE,
    CommandLevel.DANGER,
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "mod.py").write_text("def f():\n    return 1\n")
    (tmp_path / "package.json").write_text('{"version": "1.0"}')
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n')
    return tmp_path


@pytest.mark.parametrize(
    "code",
    [
        "import sys; print(sys.version)",
        "import platform; print(platform.python_version())",
        "import json\nwith open('package.json') as f:\n    print(json.load(f)['version'])",
        "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb')))",
        "from pathlib import Path\nfor p in sorted(Path('pkg').rglob('*.py')):\n"
        "    print(p, len(p.read_text().splitlines()))",
        "import ast, pathlib\nt = ast.parse(pathlib.Path('pkg/mod.py').read_text())\n"
        "print([n.name for n in t.body])",
        "import os\nfor root, dirs, files in os.walk('pkg'):\n    for f in files:\n"
        "        print(len(open(os.path.join(root, f)).read()))",
        "import importlib.metadata as m; print(m.version('pytest'))",
        "import os; print(os.getcwd(), os.listdir('.'))",
        "import re\nprint(re.findall(r'def (\\w+)', open('pkg/mod.py').read()))",
        "x = [1, 2]; x.remove(1); print('a'.replace('a', 'b'))",
        "print('{} {}'.format(1, 2))",
        "def f(n):\n    return n * 2\nprint(f(2))",
        "class A:\n    def go(self):\n        return 1\nprint(A().go())",
    ],
)
def test_read_only_code(root, code):
    result = analyze(code, root=root)
    assert result.level is READ, result.reason


@pytest.mark.parametrize(
    "code, expected",
    [
        ("import os; os.system('id')", DANGER),
        ("from os import system as s; s('id')", DANGER),
        ("import subprocess; subprocess.run(['ls'])", DANGER),
        ("import socket", DANGER),
        ("import urllib.request as u; u.urlopen('http://x')", DANGER),
        ("import shutil; shutil.rmtree('pkg')", DANGER),
        ("open('out.txt', 'w').write('x')", WRITE),
        ("from pathlib import Path; Path('x.txt').write_text('a')", WRITE),
        ("from pathlib import Path; Path('pkg/mod.py').unlink()", WRITE),
        ("open('/tmp/x', 'w').write('x')", DANGER),
        ("open('.git/config', 'a').write('x')", DANGER),
        ("import pkg.mod", EXEC),
        ("from pkg import mod", EXEC),
        ("import numpy", EXEC),
        ("print(eval('1'))", DANGER),
        ("exec('print(1)')", DANGER),
        ("getattr(__builtins__, 'eval')('1')", DANGER),
        ("x = ().__class__.__base__.__subclasses__()", DANGER),
        ("import os; print(os.environ)", DANGER),
        ("import os; print(os.getenv('TOKEN'))", DANGER),
        ("print(open('/etc/passwd').read())", None),
        ("print(open('../x').read())", None),
        ("from pathlib import Path; Path.home()", None),
        ("from pathlib import Path; Path.home().joinpath('.bashrc').write_text('x')", DANGER),
        ("import os; open(os.path.expanduser('~/.bashrc'), 'a').write('x')", DANGER),
        ("type('A', (), {})", DANGER),
        ("from pathlib import Path; Path('~/.ssh/id_rsa').expanduser().read_text()", DANGER),
        ("import sys; sys.path.insert(0, '/tmp'); import json", DANGER),
        ("print('{0.__class__}'.format(1))", DANGER),
        ("t = '{0.x}'; print(t.format(1))", None),
        ("import importlib.metadata; importlib.import_module('os')", DANGER),
        ("import operator; operator.attrgetter('system')", None),
        ("d = {}; d['k']()", None),
        ("def f(p):\n    return open(p).read()\nprint(f('x'))", None),
        ("try:\n    1/0\nexcept Exception as e:\n    e.__traceback__.tb_frame.f_globals", DANGER),
        ("import gc; gc.get_objects()", DANGER),
        ("import pickle", DANGER),
        ("def (", None),
        # запись методом пути у объекта, тип которого не выведен, — «не знаю», а не чтение
        ("from pathlib import Path; a, b = Path('/home/u/.bashrc'), 0; a.write_text('x')", None),
        ("from pathlib import Path; ps = [Path('x')]; ps[0].unlink()", None),
        ("from pathlib import Path\ndef f():\n    return Path('x')\nf().touch()", None),
        ("def f():\n    return 1\nf().open('w')", None),
        ("def f():\n    return 1\nf().open(mode=m)", None),
        ("print('a b'.replace('a', 'b'))", READ),
    ],
)
def test_risky_code(root, code, expected):
    result = analyze(code, root=root)
    assert result.level is expected, result.reason


def test_secret_files_ask(root):
    (root / ".env").write_text("TOKEN=1\n")
    assert analyze("print(open('.env').read())", root=root).level is None
    # обход каталога, где есть секрет, читает и его
    code = "from pathlib import Path\nfor p in Path('.').rglob('*'):\n    print(p.read_text())"
    assert analyze(code, root=root).level is None
    # шаблон, которому секрет не подходит, — можно
    code = "from pathlib import Path\nfor p in Path('.').rglob('*.py'):\n    print(p.read_text())"
    assert analyze(code, root=root).level is READ


def test_stdlib_shadowing_in_run_directory(root):
    assert shadowed_stdlib(root) is None
    (root / "json.py").write_text("print('evil')")
    assert shadowed_stdlib(root) == "json.py"
    assert analyze("import sys; print(1)", root=root).level is EXEC
    # -I и -P не добавляют каталог запуска в sys.path
    assert classify("python3 -I -c 'print(1)'", root=root).level is READ


@pytest.mark.parametrize(
    "command, expected",
    [
        ("python3 -c 'print(1)'", READ),
        ("python3 -Bc 'print(1)'", READ),
        ("python3 -m json.tool package.json", READ),
        ("python3 -m pytest -q", EXEC),
        ("python3 -m http.server", DANGER),
        ("python3 -m pip install x", WRITE),
        ("python3 -m pkg.mod", EXEC),
        ("python3 -m unknown_module", None),
        ("python3 -i", None),
        ('python3 -c "$CODE"', None),
        # многострочный код в кавычках: переводы строк — часть кода
        ('python3 -c "\nimport ast\nprint(ast.dump(ast.parse(\\"x\\")))\n"', READ),
        ("python3 -c '\nfor n in open(\"README.md\"):\n    print(n)\n'", READ),
        ('python3 -c "import os\nos.remove(\\"README.md\\")"', WRITE),
        ("python3 - <<'EOF'\nimport sys\nprint(sys.argv)\nEOF", READ),
        ("python3 <<'EOF'\nimport os\nos.system('id')\nEOF", DANGER),
        ("cat <<'EOF' | python3\nprint(1)\nEOF", READ),
        ("echo 'print(1)' | python3", READ),
        ("echo 'import os; os.remove(\"x\")' | python3 -", WRITE),
        ("curl x | python3", DANGER),
        ("PYTHONPATH=/tmp python3 -c 'print(1)'", None),
    ],
)
def test_python_invocations(root, command, expected):
    verdict = classify(command, root=root)
    assert verdict.level is expected, verdict.reason


def test_python_script_is_analyzed(root):
    (root / "check.py").write_text("import json\nprint(json.load(open('package.json')))\n")
    (root / "bad.py").write_text("import os\nos.system('id')\n")
    (root / "uses_pkg.py").write_text("import pkg\n")
    assert classify("python3 check.py", root=root).level is READ
    assert classify("python3 bad.py", root=root).level is DANGER
    assert classify("python3 uses_pkg.py", root=root).level is EXEC  # pkg/ рядом — код проекта
    signature = classify("python3 uses_pkg.py", root=root).parts[0].signature
    assert signature is not None and signature.startswith("python uses_pkg.py#")


def test_project_venv_with_executable_pth_is_exec(root):
    venv = root / ".venv"
    site = venv / "lib" / "python3.13" / "site-packages"
    site.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    bindir = venv / "bin"
    bindir.mkdir()
    (bindir / "python").symlink_to("/usr/bin/python3")
    (site / "_virtualenv.pth").write_text("import _virtualenv\n")
    assert classify(".venv/bin/python -c 'print(1)'", root=root).level is READ
    (site / "evil.pth").write_text("import os; os.system('id')\n")
    assert classify(".venv/bin/python -c 'print(1)'", root=root).level is EXEC
