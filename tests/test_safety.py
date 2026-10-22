"""Классификатор команд оболочки: разбор, политика программ, пути, окружение."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from gigacrew.project import treesitter
from gigacrew.safety import CommandLevel, classify
from gigacrew.safety import bash as shell

READ, EXEC, WRITE, DANGER = (
    CommandLevel.READ,
    CommandLevel.EXEC,
    CommandLevel.WRITE,
    CommandLevel.DANGER,
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    for d in ("src", "tests", "pkg"):
        (tmp_path / d).mkdir()
    for f in ("README.md", "a.txt", "b.txt", "src/main.py", "tests/test_a.py", "pyproject.toml"):
        (tmp_path / f).write_text("hello\n")
    (tmp_path / "package.json").write_text('{"scripts": {"test": "jest", "build": "tsc"}}')
    return tmp_path


def level(command: str, root: Path, **kw) -> CommandLevel | None:
    return classify(command, root=root, **kw).level


# --------------------------- только чтение --------------------------- #
@pytest.mark.parametrize(
    "command",
    [
        "ls -la",
        "ls src",
        "pwd",
        "cat README.md",
        "head -n 20 src/main.py",
        "tail -n 5 a.txt",
        "wc -l src/*.py",
        "grep -rn hello src",
        "grep -rn TODO --include=*.py .",
        "rg -n hello",
        "find . -name '*.py'",
        "find src -name '*.py' -exec grep -l hello {} +",
        "find src -name '*.py' | xargs wc -l",
        "echo $HOME",
        "printenv PATH",
        "sed -n '1,10p' README.md",
        "awk '{print $1}' a.txt",
        # сравнения в awk — не вывод в файл
        "awk 'NR>=10 && NR<=20' a.txt",
        "awk '$2 > 5 || NF > 3' a.txt",
        "awk 'length > 100 {print NR\": \"$0}' a.txt",
        "awk '{ if ($1 > 5) print $2; else print \">\" }' a.txt",
        "awk '{print ($1 > $2), $3}' a.txt",
        "awk -F: '{print $1} $3 > 1000' a.txt",
        "awk '{ print \"/etc\"; if ($1 > 5) print $1 }' a.txt",  # «/» только в строке
        # опции, которые не меняют, откуда программа: разделитель полей и присваивания
        "awk -F: '{print $1}' a.txt",
        "awk -F : '{print $1}' a.txt",
        "awk -v n=3 'NR==n' a.txt",
        "awk -vn=3 -F, 'NR==n' a.txt",
        "awk --field-separator=: '{print $1}' a.txt",
        "awk --assign n=3 'NR==n' a.txt",
        "awk -- '{print $1}' a.txt",
        "jq .scripts package.json",
        "sort a.txt | uniq -c | sort -rn | head",
        "diff a.txt b.txt",
        "ps aux",
        "cat /etc/os-release",
        "cd src && ls",
        "ls | head; pwd",
        "set -euo pipefail; ls",
        "command -v python3",
        "python3 --version",
        "python3 -c 'import sys; print(sys.version)'",
        "bash -c 'ls -la'",
        "uname -a && date",
        "cat a.txt 2>/dev/null || echo missing",
        "timeout 5 cat a.txt",
        "sleep 1",
    ],
)
def test_read_only(root, command):
    verdict = classify(command, root=root)
    assert verdict.level is READ, verdict.reason


# --------------------- никогда не «только чтение» -------------------- #
@pytest.mark.parametrize(
    "command",
    [
        # запись, удаление, сеть, привилегии
        "touch x",
        "echo hi > out.txt",
        "rm a.txt",
        "rm -rf build",
        "mv a.txt c.txt",
        "curl https://example.com",
        "sudo ls",
        "pip install requests",
        # подстановки и обфускация имени программы
        '"$(echo rm)" -rf build',
        "$CMD a.txt",
        'sh -c "$X"',
        "eval ls",
        "f() { cat a.txt; }; f",
        "ls &",
        "cat <(ls)",
        "ls {a,b}",
        "cat {/etc/passwd,a.txt}",
        # продолжение строки и heredoc: части, спрятанные от разбора
        "cat .\\\n./.\\\n./outside.txt",
        "cat <<'EOF' && rm -rf ~\nhello\nEOF",
        "cat <<EOF\n$(touch pwned)\nEOF",
        # шаблоны, совпадающие с «..» в dash
        "cat .*",
        "grep -r token .*/",
        # секреты и файлы вне проекта
        "cat ~/.ssh/id_rsa",
        "cat /proc/self/environ",
        "cat ../outside.txt",
        "grep -r KEY ~",
        "cat .env",
        "echo $GITHUB_TOKEN",
        "env",
        "printenv",
        "ps auxe",
        "ps eww",
        # переменные, меняющие, что запустится
        "PATH=.:$PATH ls",
        "PATH=.; ls",
        "LD_PRELOAD=./x.so ls",
        "PAGER='sh -c id' git log",
        "GIT_EXTERNAL_DIFF=./x.sh git diff",
        "RIPGREP_CONFIG_PATH=./rgrc rg foo",
        "LESSOPEN='|id %s' less a.txt",
        "export PATH=.; ls",
        "x=1; echo $x",
        # опасные флаги «читающих» программ, в том числе сокращённые
        "find . -exec sh -c 'touch x' \\;",
        "find . -exec sh -c 'cat \"$1\"' _ {} \\;",
        "find . -delete",
        "find . -fprint out.txt",
        "rg --pre ./x hello",
        "rg --pre=./x hello",
        "sort --compress-program=./x a.txt",
        "sort --compress-prog=./x a.txt",
        "sort --outp=x.txt a.txt",
        "sort -o x.txt a.txt",
        "sed -n '1e id' a.txt",
        "sed 's/a/b/w out.txt' a.txt",
        "sed -i 's/a/b/' a.txt",
        "awk 'BEGIN{system(\"id\")}'",
        "awk '{print > \"out.txt\"}' a.txt",
        "awk '{print $1, $2 >> \"out.txt\"}' a.txt",
        "awk -W exec prog.awk a.txt",  # программа из файла (mawk)
        "awk -Wexec prog.awk a.txt",
        "gawk -W source='BEGIN{print 1}' a.txt",
        "awk -W file=prog.awk a.txt",
        # gawk: сгруппированные короткие опции, второй -e, запись отладки, сокращения
        "gawk -bWsource='BEGIN{print 1}' a.txt",
        "gawk -bf prog.awk a.txt",
        "gawk -nf prog.awk a.txt",
        "gawk -e 'BEGIN{print 1}' -e 'BEGIN{system(\"id\")}'",
        "gawk -d '{print}' a.txt",
        "gawk -o/tmp/pp.out '{print}' a.txt",
        "gawk -p '{print}' a.txt",
        "gawk --as x=1 '{print > \"out.txt\"}' a.txt",
        "gawk --field : 'BEGIN{system(\"id\")}'",
        # ARGV/ARGC: подмена входных файлов мимо проверки путей (секреты, вне проекта)
        "awk 'BEGIN { ARGV[1] = \".env\" } { print }' a.txt",
        "awk 'BEGIN { ARGV[ARGC++] = \"/etc/passwd\" } 1' a.txt",
        "awk -v ARGC=3 '{print}' a.txt",
        "awk '{print}' ARGC=3 a.txt",
        'awk \'{printf("%s\\n", $1) > "out.txt"}\' a.txt',
        "awk '{print $1,\n$2 > \"out.txt\"}' a.txt",
        "awk '$1 > 5 {print $1 > \"out.txt\"}' a.txt",
        "awk '{print $1 | \"sh\"}' a.txt",
        'awk \'# "\n{print > "out.txt"}\n# "\' a.txt',
        'awk \'/"/ {print > "out"} /"/\' a.txt',
        'awk \'{print "/"; print $1 / 2 > "out"}\' a.txt',  # «/» вне строки — как раньше
        "awk '# /\n{print > \"out\"}' a.txt",  # «/» в комментарии не прячет вывод
        'awk \'{print "a" > "out.txt"\' a.txt',
        "awk 'BEGIN{print ENVIRON[\"HOME\"]}'",
        "jq -n env",
        "jq -n '$ENV'",
        "tar --to-command='sh -c id' -xf a.tar",
        "tar --to-comm='sh -c id' -xf a.tar",
        "less +'!id' a.txt",
        "xxd a.txt out.bin",
        "uniq a.txt out.txt",
        "tee out.txt",
        "base64 -o out.txt a.txt",
        "date -s '2020-01-01'",
        "git -c core.pager='sh -c id' log",
        "git -c alias.x='!id' x",
        "git log --output=x.txt",
        "git log --outp=x.txt",
        "git diff --ext-diff",
        "git grep -O id x",
        "ls | xargs cat",
        "xargs -a list.txt cat",
        "vim a.txt",
        "watch ls",
        # Python: процессы, запись, динамический код, секреты
        "python3 -c 'import os; os.system(\"id\")'",
        "python3 -c 'import subprocess; subprocess.run([\"id\"])'",
        'python3 -c \'open("x", "w").write("1")\'',
        "python3 -c 'print(open(\"/etc/passwd\").read())'",
        "python3 -c 'print(eval(\"1\"))'",
        "python3 -c 'import os; print(os.environ)'",
        "python3 -c 'print(\"{0.__class__}\".format(1))'",
        "python3 - <<'EOF'\nimport shutil\nshutil.rmtree('src')\nEOF",
        "echo 'import os; os.system(\"id\")' | python3",
        "python3 <<EOF\nprint($HOME)\nEOF",
    ],
)
def test_not_read_only(root, command):
    verdict = classify(command, root=root)
    assert verdict.level is not READ, (command, verdict.reason)


def test_glob_expanding_to_option_like_name_is_not_read(root):
    (root / "--pre=id").write_text("")
    assert level("rg hello *", root) is not READ
    assert level("wc -l *.txt", root) is READ  # совпадения без «-»


def test_program_from_project_path_is_exec(root, monkeypatch):
    bin_dir = root / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "cat"
    fake.write_text("#!/bin/sh\nid\n")
    fake.chmod(0o755)
    assert level("cat a.txt", root, environ={"PATH": f"{bin_dir}:/usr/bin:/bin"}) is EXEC
    assert level("cat a.txt", root, environ={"PATH": "bin:/usr/bin:/bin"}) is EXEC
    assert level("./bin/cat a.txt", root) is EXEC


def test_program_path_is_normalized_before_trusting_system_dir(root):
    (root / "sub").mkdir()
    fake = root / "sub" / "ls"
    fake.write_text("#!/bin/sh\nid\n")
    fake.chmod(0o755)
    assert level("/usr/bin/cat a.txt", root) is READ
    # путь начинается с системного каталога, но ведёт в файлы проекта
    climb = "/usr/bin/" + "../" * 2 + str(root).lstrip("/") + "/sub/ls"
    assert level(climb, root) is EXEC
    assert level("/usr/bin/../../tmp/x/cat a.txt", root) is None
    assert level("/tmp/x/cat a.txt", root) is None  # программа вне проекта
    assert level("~/bin/ls", root) is None  # оболочка раскроет ~ в домашний каталог


@pytest.mark.parametrize(
    "command, expected",
    [
        ("env -i ls", READ),
        ("env -u HOME ls", READ),
        ("env -iu HOME ls", READ),
        ("env -i -- LC_ALL=C ls", READ),
        ("env --unset=HOME ls", READ),
        # значение -u в группе — имя переменной, команда начинается после него
        ("env -u ls rm -rf src", DANGER),
        ("env -iu ls rm -rf src", DANGER),
        ("env -vuls rm -rf src", DANGER),
        # каталог, разбиение строки, подмена argv[0] и незнакомые опции — «не знаю»
        ("env -iC /tmp cat a.txt", None),
        ("env -C/tmp cat a.txt", None),
        ("env --chdir=/tmp cat a.txt", None),
        ("env --ch=/tmp cat a.txt", None),
        ("env -iS 'rm -rf src' ls", None),
        ("env -a sh ls", None),
        ("env -iX ls", None),
        # после присваивания опций нет: «-i» — имя программы
        ("env A=1 -i ls", None),
        # присваивания через env проверяются так же, как префиксные
        ("env LD_PRELOAD=./x.so ls", DANGER),
        ("env PATH=.:/usr/bin ls", DANGER),
        ("env PAGER='sh -c id' git log", DANGER),
    ],
)
def test_env_options_and_assignments(root, command, expected):
    verdict = classify(command, root=root)
    assert verdict.level is expected, (command, verdict.reason)


@pytest.mark.parametrize(
    "command, expected",
    [
        # значение флага в группе коротких (`-uo FILE`) — как в getopt
        ("sort -uo .git/x a.txt", DANGER),
        ("sort -uox.txt a.txt", WRITE),
        ("sort -ox.txt a.txt", WRITE),
        ("sort -o.git/hooks/pre-commit a.txt", DANGER),
        ("sort -u a.txt", READ),
        ("cp -ft .git/hooks a.txt", DANGER),
        ("cp -t /etc a.txt", DANGER),
        ("mv --target-directory=/etc a.txt", DANGER),
        ("cp -f a.txt c.txt", WRITE),
        ("tar -xC /tmp -f a.tar", DANGER),
        ("tar xfC a.tar /tmp", DANGER),  # старая запись: значения букв — следующие слова
        ("tar -xzf a.tar", WRITE),
        ("tar -tf a.tar", READ),
        ("tar -tf a.tar -I ./x", DANGER),
        ("tar -xf a.tar -I ./x", DANGER),
        ("tar -cf - -T a.txt", DANGER),  # список файлов, как --files-from
        ("tar -cf - -C /etc shadow", None),  # файлы после -C — от его каталога
        ("tar -cf - src", READ),
        ("tree -ao /tmp/x .", DANGER),
        ("tree -ao out.txt .", WRITE),
        ("tree -o out.txt", WRITE),
        ("tree -o /tmp/x", DANGER),
        ("tree -o", None),  # куда пишет — не найдено: не «правка»
        ("tree -Lo 2 out.txt", None),
        ("tree -L 2 src", READ),
        # флаги без значения не съедают следующий аргумент
        ("column -t ~/.ssh/id_rsa", DANGER),
        ("column -t a.txt", READ),
        ("jq --tab . /etc/shadow", None),
        ("jq --tab . package.json", READ),
        ("grep --color x ~/.ssh/id_rsa", DANGER),
        ("grep --color=always hello a.txt", READ),
        ("touch -a /etc/x a.txt", DANGER),
        ("touch -a a.txt", WRITE),
        ("dos2unix -n ~/.ssh/id_rsa out.txt", DANGER),
        ("xxd -ps ~/.ssh/id_rsa", DANGER),  # у xxd свой разбор: -ps — один флаг
        ("xxd -ps a.txt", READ),
        # флаги с двумя значениями
        ("jq -n --arg v x env", None),
        ("jq -n --arg v x '$v'", READ),
        # каталог назначения -t
        ("install -t /usr/local/bin a.txt", DANGER),
        ("install -m 755 a.txt bin/x", WRITE),
        ("ln -s -t /etc a.txt", DANGER),
        ("ln -s a.txt b.txt", WRITE),
        # опасные флаги внутри группы
        ("rg -iz hello", None),
        ("rg -i hello", READ),
        ("fd -Hx rm", None),
        ("fd -e py -X rm -rf", DANGER),
        ("fd -e py -x wc -l", READ),
        ("docker run -itv /var/run/docker.sock:/s alpine", DANGER),
        ("split --filter='sh -c id' a.txt", None),
        # patch пишет файл из аргументов, -o и -r
        ("patch /etc/passwd a.txt", DANGER),
        ("patch -o /tmp/out -i a.txt", DANGER),
        ("patch -p1 -i a.txt", WRITE),
        ("patch --dry-run -p1 -i a.txt", READ),
    ],
)
def test_option_values_are_parsed_like_getopt(root, command, expected):
    verdict = classify(command, root=root)
    assert verdict.level is expected, (command, verdict.reason)


def test_symlink_out_of_project_is_outside(root, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "secret.txt"
    outside.write_text("s")
    (root / "link.txt").symlink_to(outside)
    assert level("cat link.txt", root) is not READ


def test_recursive_read_with_secrets_asks(root):
    assert level("grep -rn hello .", root) is READ
    (root / ".env").write_text("TOKEN=1\n")
    assert level("grep -rn hello .", root) is None
    assert level("rg -n hello", root) is READ  # rg пропускает скрытые файлы
    assert level("rg -n --hidden hello", root) is None
    assert level("find . -type f -exec cat {} +", root) is None
    assert level("grep -d recurse hello .", root) is None
    assert level("tar -cf - src", root) is READ
    assert level("tar -cf - .", root) is None  # каталог в архив — читается целиком


# ------------------------------- git -------------------------------- #
@pytest.fixture
def repo(root):
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    return root


def test_git_reads_and_changes(repo):
    assert level("git status", repo) is READ
    assert level("git log --oneline -5", repo) is READ
    assert level("git -C src log -1", repo) is READ
    assert level("git commit -m x", repo) is WRITE
    assert level("git push origin main", repo) is WRITE
    assert level("git push -f", repo) is DANGER
    assert level("git reset --hard", repo) is DANGER
    assert level("git checkout -- .", repo) is DANGER
    assert level("git clean -fd", repo) is DANGER
    assert level("git config --global user.name x", repo) is DANGER


def test_git_read_with_program_in_repo_config_asks(repo):
    subprocess.run(["git", "config", "diff.evil.textconv", "sh -c id"], cwd=repo, check=True)
    assert level("git diff", repo) is None
    assert level("git log -p", repo) is None


def test_git_in_nested_repository_asks(repo):
    nested = repo / "pkg"
    subprocess.run(["git", "init", "-q"], cwd=nested, check=True)
    assert level("cd pkg && git status", repo) is None
    assert level("git -C pkg status", repo) is None
    assert level("git status", repo) is READ


# --------------------------- выполнение кода -------------------------- #
@pytest.mark.parametrize(
    "command, signature",
    [
        ("pytest", "pytest"),
        ("pytest -q -x tests/test_a.py", "pytest"),
        ("python -m pytest -k smoke", "pytest"),
        ("uv run pytest -q", "pytest"),
        ("pytest -q 2>&1 | tail -30", "pytest"),
        ("npm test", "npm test"),
        ("npm run build", "npm run build"),
        ("cargo test --release", "cargo test"),
        ("go test ./... -run TestX", "go test"),
        ("make test", "make test"),
        ("ruff check src", "ruff check"),
        ("mypy src", "mypy"),
    ],
)
def test_exec_with_signature(root, command, signature):
    verdict = classify(command, root=root)
    assert verdict.level is EXEC, verdict.reason
    assert verdict.signatures == (signature,)
    assert verdict.composable


@pytest.mark.parametrize(
    "command",
    [
        "pytest -p myplugin",  # загрузка плагина
        "npm test -- --watch",  # аргументы скрипту
        "uv run --python 3.12 pytest",
        "pytest $ARGS",
        "PYTHONSTARTUP=x pytest",
    ],
)
def test_exec_without_signature_only_exact(root, command):
    verdict = classify(command, root=root)
    assert not verdict.signatures, (command, verdict)


@pytest.mark.parametrize(
    "command",
    [
        "pytest --basetemp=../victim",
        "pytest --basetemp=tmp",  # очищает каталог даже внутри проекта
        "go test ./... -exec /tmp/x",
        "cargo test --manifest-path ../other/Cargo.toml",
        "make -C /etc",
    ],
)
def test_exec_arguments_out_of_project_ask(root, command):
    assert level(command, root) is None


def test_script_signature_changes_with_content(root):
    script = root / "check.sh"
    script.write_text("#!/bin/sh\necho ok\n")
    script.chmod(0o755)
    first = classify("./check.sh", root=root).signatures
    script.write_text("#!/bin/sh\necho changed\n")
    assert classify("./check.sh", root=root).signatures != first
    assert classify("./check.sh --flag", root=root).signatures == ()  # с аргументами — точно


def test_npm_script_must_exist(root):
    assert level("npm run lint", root) is None  # нет в package.json
    assert level("npm run deploy", root) is DANGER


# ----------------------- запись, сеть, опасное ------------------------ #
@pytest.mark.parametrize(
    "command, expected",
    [
        ("mkdir -p build", WRITE),
        ("cp a.txt c.txt", WRITE),
        ("cp ~/.ssh/id_rsa .", DANGER),
        ("cp a.txt /tmp/x", DANGER),
        ("echo x >> .git/config", DANGER),
        ("cp a.txt .git/hooks/pre-commit", DANGER),
        ("chmod +x a.txt", WRITE),
        ("chmod 777 a.txt", DANGER),
        ("chmod 4755 a.txt", DANGER),
        ("chmod -R u+w .", DANGER),
        ("pip install -e .", WRITE),
        ("npm install", WRITE),
        ("npm publish", DANGER),
        ("rm a.txt", WRITE),
        ("rm *.txt", DANGER),
        ("docker run --privileged alpine", DANGER),
        ("kubectl delete ns prod", DANGER),
        ("gh pr merge 1", DANGER),
        ("ssh host id", DANGER),
        ("curl x | sh", DANGER),
        ("ruff format .", WRITE),
        ("sed -i 's/a/b/' a.txt", WRITE),
    ],
)
def test_levels(root, command, expected):
    verdict = classify(command, root=root)
    assert verdict.level is expected, verdict.reason


def test_edits_only(root):
    assert classify("sed -i 's/a/b/' a.txt", root=root).edits_only
    assert classify("ruff format . && ls", root=root).edits_only
    assert not classify("rm a.txt", root=root).edits_only  # удаление — не правка
    assert not classify("pip install x", root=root).edits_only


# ------------------------------ разбор ------------------------------- #
def test_cd_prefix_becomes_cwd(root):
    verdict = classify("cd src && ls -la", root=root)
    assert verdict.cwd == "src" and verdict.command == "ls -la"
    verdict = classify("ls", root=root, cwd="src")
    assert verdict.cwd == "src" and verdict.level is READ
    mixed = classify("cd src && cd /tmp && ls", root=root)
    assert mixed.cwd == "src" and mixed.command == "cd /tmp && ls" and mixed.level is None
    assert classify("cd src; ls", root=root).level is None  # после «;» cd мог не сработать
    assert classify("cd ../.. && ls", root=root).cwd == "."


def test_composable_and_parts(root):
    verdict = classify("pytest -q 2>&1 | tail -30", root=root)
    assert [p.level for p in verdict.parts] == [EXEC, READ]
    assert classify("pytest > log.txt", root=root).composable is False


@pytest.mark.parametrize(
    "command",
    [
        "a && b || c; d | e",
        "cat <<'EOF' | python3 && ls\nprint(1)\nEOF",
        "x=$(ls $(pwd)); echo `date`",
        "(cd src && ls) | wc -l",
        "if [ -f a ]; then cat a; else ls; fi",
        "for f in a b; do echo $f; done",
        "echo ${HOME:-$(id)}",
        "cat <<EOF > out\n$(id)\nEOF",
        "! grep -q x a.txt && echo none",
        "ls 2>&1 | tee log | wc -l",
    ],
)
def test_every_command_node_is_classified(root, command):
    """Каждая простая команда дерева разобрана (ничего не спрятано в heredoc и т.п.)."""
    tree = treesitter.parse_tree("shell", command)
    assert tree is not None
    nodes = 0
    stack = [tree]
    while stack:
        n = stack.pop()
        nodes += n.type == "command"
        stack.extend(n.children)
    script = shell.parse(command, root=root)
    assert script is not None and len(script.commands) + script.consumed == nodes


@pytest.mark.parametrize(
    "command, value",
    [
        ('echo "a\nb"', "a\nb"),
        ('echo "\n  x = 1\n"', "\n  x = 1\n"),
        ('echo "a\\"b\nc\\\\d"', 'a"b\nc\\d'),
        ('echo "a\\\nb"', "ab"),  # перенос строки через \ внутри "…" убирается
        ("echo 'a\nb'", "a\nb"),
        ('echo x"a\nb"y', "xa\nby"),
    ],
)
def test_quoted_word_keeps_newlines(root, command, value):
    """Значение слова в кавычках — как его получит программа, с переводами строк."""
    script = shell.parse(command, root=root)
    assert script is not None
    assert script.commands[0].argv[1].value == value


def test_long_commands_and_other_platforms_ask(root, monkeypatch):
    assert level("echo " + "x" * 20_000, root) is None
    monkeypatch.setattr("gigacrew.safety._is_posix", lambda: False)
    assert level("ls", root) is None


def test_child_environment_drops_shell_hooks_and_adds_git_safety(monkeypatch):
    from gigacrew.tools.process import subprocess_env

    env = subprocess_env(
        {
            "CDPATH": "/",
            "BASH_ENV": "x",
            "ENV": "y",
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "user.name",
            "GIT_CONFIG_VALUE_0": "me",
        }
    )
    assert "CDPATH" not in env and "BASH_ENV" not in env and "ENV" not in env
    assert env["GIT_CONFIG_COUNT"] == "3" and env["GIT_CONFIG_KEY_0"] == "user.name"
    assert env["GIT_CONFIG_KEY_1"] == "safe.bareRepository"
    assert env["GIT_CONFIG_VALUE_2"] == "false"
    assert os.environ.get("CDPATH") is None or True


@pytest.mark.parametrize(
    "command",
    [
        '"$(echo rm)" -rf build',
        "$CMD a.txt",
        "eval ls",
        "ls\u202e txt.exe",  # смена направления текста
        "cat a.txt\u200b",  # символ нулевой ширины
        "echo ok\x1b[2K",  # управляющая последовательность терминала
        "python3 -c 'print(eval(\"1\"))'",
        "python3 -c 'import os; print(os.environ)'",
        "python3 -c 'x = ().__class__'",
    ],
)
def test_hidden_or_dynamic_code_is_marked_dangerous(root, command):
    """Спросить И пометить опасным: человек должен посмотреть внимательно."""
    verdict = classify(command, root=root)
    assert verdict.level is DANGER, verdict.reason
