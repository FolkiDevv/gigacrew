"""Классификатор команд оболочки: разбор, политика программ, пути, окружение."""

from __future__ import annotations

import hashlib
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

from gigacrew.project import treesitter
from gigacrew.safety import CommandLevel, classify
from gigacrew.safety import bash as shell
from gigacrew.security import subprocess_env

READ, EXEC, WRITE, DANGER = (
    CommandLevel.READ,
    CommandLevel.EXEC,
    CommandLevel.WRITE,
    CommandLevel.DANGER,
)
# Обратная косая черта в тестовых командах: без неявных raw-строк
_BACKSLASH = "\\"
# Пустой набор подписей: сравнение именно с пустым кортежем
_NO_SIGNATURES: tuple[str, ...] = ()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    for dirname in ("src", "tests", "pkg"):
        (tmp_path / dirname).mkdir()
    for filename in (
        "README.md",
        "a.txt",
        "b.txt",
        "src/main.py",
        "tests/test_a.py",
        "pyproject.toml",
    ):
        (tmp_path / filename).write_text("hello\n")
    (tmp_path / "package.json").write_text('{"scripts": {"test": "jest", "build": "tsc"}}')
    return tmp_path


def level(
    command: str,
    root: Path,
    *,
    cwd: str = ".",
    environ: Mapping[str, str] | None = None,
) -> CommandLevel | None:
    return classify(command, root=root, cwd=cwd, environ=environ).level


def _not_posix() -> bool:
    return False


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
def test_read_only(root: Path, command: str) -> None:
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
        f"cat .{_BACKSLASH}\n./.{_BACKSLASH}\n./outside.txt",
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
        r"find . -exec sh -c 'touch x' \;",
        f"find . -exec sh -c 'cat \"$1\"' _ {{}} {_BACKSLASH};",
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
        f'awk \'{{printf("%s{_BACKSLASH}n", $1) > "out.txt"}}\' a.txt',
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
def test_not_read_only(root: Path, command: str) -> None:
    verdict = classify(command, root=root)
    assert verdict.level is not READ, (command, verdict.reason)


def test_glob_expanding_to_option_like_name_is_not_read(root: Path) -> None:
    (root / "--pre=id").write_text("")
    assert level("rg hello *", root) is not READ
    assert level("wc -l *.txt", root) is READ  # совпадения без «-»


def test_program_from_project_path_is_exec(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bin_dir = root / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "cat"
    fake.write_text("#!/bin/sh\nid\n")
    fake.chmod(0o755)
    assert level("cat a.txt", root, environ={"PATH": f"{bin_dir}:/usr/bin:/bin"}) is EXEC
    assert level("cat a.txt", root, environ={"PATH": "bin:/usr/bin:/bin"}) is EXEC
    assert level("./bin/cat a.txt", root) is EXEC


def test_program_path_is_normalized_before_trusting_system_dir(root: Path) -> None:
    (root / "sub").mkdir()
    fake = root / "sub" / "ls"
    fake.write_text("#!/bin/sh\nid\n")
    fake.chmod(0o755)
    assert level("/usr/bin/cat a.txt", root) is READ
    # путь начинается с системного каталога, но ведёт в файлы проекта
    up_twice = "../" * 2
    root_without_slash = str(root).lstrip("/")
    climb = f"/usr/bin/{up_twice}{root_without_slash}/sub/ls"
    assert level(climb, root) is EXEC
    assert level("/usr/bin/../../tmp/x/cat a.txt", root) is None
    assert level("/tmp/x/cat a.txt", root) is None  # программа вне проекта
    assert level("~/bin/ls", root) is None  # оболочка раскроет ~ в домашний каталог


@pytest.mark.parametrize(
    ("command", "expected"),
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
def test_env_options_and_assignments(
    root: Path,
    command: str,
    expected: CommandLevel | None,
) -> None:
    verdict = classify(command, root=root)
    assert verdict.level is expected, (command, verdict.reason)


@pytest.mark.parametrize(
    ("command", "expected"),
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
        ("jq --tab . /etc/hosts", None),
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
        # каталоги поиска перед -x/-X проверяются, как без --exec
        ("fd x /etc -x cat", None),
        ("fd x /etc", None),
        ("fd id_rsa ~/.ssh -x cat", DANGER),
        ("fd x --search-path /etc -x cat", None),
        ("fd x --search-path=/etc -X cat", None),
        ("fd x --base-directory /etc -x cat", None),
        ("fd -C /etc -x cat", None),  # -C — короткая форма --base-directory
        ("fd -C/etc x -x cat", None),
        ("fd -HC /etc x -x cat", None),
        ("fd -C /etc x", None),
        ("fd -C src x -x cat", READ),
        ("fd x src -x cat", READ),
        ("docker run -itv /var/run/docker.sock:/s alpine", DANGER),
        ("split --filter='sh -c id' a.txt", None),
        # patch пишет файл из аргументов, -o и -r
        ("patch /etc/passwd a.txt", DANGER),
        ("patch -o /tmp/out -i a.txt", DANGER),
        ("patch -p1 -i a.txt", WRITE),
        ("patch --dry-run -p1 -i a.txt", READ),
    ],
)
def test_option_values_are_parsed_like_getopt(
    root: Path,
    command: str,
    expected: CommandLevel | None,
) -> None:
    verdict = classify(command, root=root)
    assert verdict.level is expected, (command, verdict.reason)


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        # значения флагов, которые читают файлы, проверяются как пути
        ("date -f ~/.ssh/id_rsa", DANGER),
        ("date -f /etc/hosts", None),
        ("date -f a.txt", READ),
        ("date -d 2020-01-01", READ),  # только показ даты
        ("date -s 'next monday'", None),
        ("date --set=tomorrow", None),
        ("date +%Y", READ),
        ("diff --from-file=/etc/hosts a.txt", None),
        ("diff --from-file=b.txt a.txt", READ),
        ("diff -u a.txt b.txt", READ),
        ("du -X /etc/hosts .", None),
        ("du -sh src", READ),
        ("file -m /etc/hosts a.txt", None),
        ("file a.txt", READ),
        ("dmesg -F ~/.ssh/id_rsa", DANGER),
        ("dircolors ~/.ssh/id_rsa", DANGER),
        # файлы по списку из файла — вне проверки путей
        ("wc --files0-from=list a.txt", None),
        ("du --files0-from=list", None),
        ("file -f list.txt", None),
        ("bat --pager='sh -c id' --paging=always a.txt", None),
        ("bat a.txt", READ),
    ],
)
def test_file_reading_flag_values_are_checked(
    root: Path,
    command: str,
    expected: CommandLevel | None,
) -> None:
    verdict = classify(command, root=root)
    assert verdict.level is expected, (command, verdict.reason)


def test_symlink_out_of_project_is_outside(
    root: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    outside = tmp_path_factory.mktemp("outside") / "secret.txt"
    outside.write_text("s")
    (root / "link.txt").symlink_to(outside)
    assert level("cat link.txt", root) is not READ


def test_recursive_read_with_secrets_asks(root: Path) -> None:
    assert level("grep -rn hello .", root) is READ
    assert level("diff -rN src .", root) is READ
    (root / ".env").write_text("TOKEN=1\n")
    assert level("grep -rn hello .", root) is None
    assert level("rg -n hello", root) is READ  # rg пропускает скрытые файлы
    assert level("rg -n --hidden hello", root) is None
    assert level("find . -type f -exec cat {} +", root) is None
    assert level("grep -d recurse hello .", root) is None
    assert level("diff -rN src .", root) is None  # diff -r печатает содержимое .env
    assert level("tar -cf - src", root) is READ
    assert level("tar -cf - .", root) is None  # каталог в архив — читается целиком


# ------------------------------- git -------------------------------- #
@pytest.fixture
def repo(root: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    return root


def test_git_reads_and_changes(repo: Path) -> None:
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


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        # настройки, запускающие программы, — на один запуск и постоянно
        ("git -c core.pager='sh -c id' log", DANGER),
        ("git -c alias.x='!id' x", DANGER),
        ('git -c "$X" log', DANGER),
        ("git -c user.name=x log", None),
        ("git config core.hooksPath /tmp/hooks", DANGER),
        ("git config alias.x '!id'", DANGER),
        ("git config diff.x.textconv cat", DANGER),
        ("git config credential.helper store", DANGER),
        ("git config user.name x", WRITE),
        ("git config user.name", READ),
        # каталоги и глобальные флаги
        ("git -C /tmp status", None),
        ("git -C .. log", None),
        ("git --git-dir=/tmp/x/.git log -p", None),
        ("git --exec-path=/tmp/x status", None),
        ("git --work-tree=/ status", None),
        ("git --no-pager log", READ),
        # запись файлов: внутри проекта — WRITE, снаружи и в .git — DANGER
        ("git log --output=/tmp/x", DANGER),
        ("git diff --output=.git/hooks/pre-commit", DANGER),
        ("git log --output=x.txt", WRITE),
        ("git archive -o /tmp/src.zip HEAD", DANGER),
        ("git archive -o out.zip HEAD", WRITE),
        ("git format-patch --output-directory=/tmp/x HEAD~1", DANGER),
        ("git format-patch -o /tmp/x HEAD~1", DANGER),
        ("git format-patch -o patches HEAD~1", WRITE),
        ("git bundle create /tmp/x.bundle HEAD", DANGER),
        ("git bundle create x.bundle HEAD", WRITE),
        ("git grep -o hello", READ),
        # чтение файлов вне репозитория (diff — и без --no-index)
        ("git diff --no-index a.txt ~/.ssh/id_rsa", DANGER),
        ("git diff --no-index a.txt /etc/passwd", None),
        ("git diff --no-index a.txt b.txt", READ),
        ("git diff ../outside.txt /dev/null", None),
        ("git diff ~/.ssh/id_rsa /dev/null", DANGER),
        ("git diff HEAD~1 -- src", READ),
        ("git grep --no-index KEY ~/.ssh", DANGER),
        ("git grep KEY", READ),
        # программа-пейджер для найденных файлов, в том числе в группе флагов
        ("git grep -Oid x", None),
        ("git grep -nOid x", None),
        ("git grep -O id x", None),
        ("git grep --open-files-in-pager=id x", None),
        ("git grep --open x", None),
        ("git grep -e Oid x", READ),  # «O» — часть образца
        ("git grep -n hello", READ),
        ("git -p log", None),
        ("git log --ext-diff -p", None),
        ("git show --textconv HEAD", None),
        ("git cat-file --textconv HEAD:a.txt", None),
        ("git log --no-ext-diff -p", READ),
        # файлы, которые читаются по значению флага
        ("git blame --contents ~/.ssh/id_rsa a.txt", DANGER),
        ("git blame --contents=/etc/hosts a.txt", None),
        ("git blame --contents b.txt a.txt", READ),
        ("git grep -f ~/.ssh/id_rsa", DANGER),
        ("git grep -f a.txt", READ),
        # запуск команд
        ("git rebase -x 'rm -rf ~' main", None),
        ("git rebase -ix 'rm -rf ~' main", None),
        ("git rebase main", WRITE),
        ("git submodule foreach 'rm -rf ~'", None),
        ("git submodule update --init", WRITE),
        ("git bisect run ./x.sh", None),
        # необратимые изменения
        ("git remote set-url origin https://x/y.git", DANGER),
        ("git remote add up https://x", DANGER),
        ("git remote -v", READ),
        ("git stash drop", DANGER),
        ("git stash clear", DANGER),
        ("git stash list", READ),
        ("git stash", WRITE),
        ("git restore .", DANGER),
        ("git restore src/main.py", DANGER),
        ("git restore --staged a.txt", WRITE),
        ("git checkout .", DANGER),
        ("git checkout src/main.py", DANGER),
        ("git checkout feature", WRITE),
        ("git checkout -B feature", DANGER),
        ("git checkout -b feature", WRITE),
        ("git switch -f main", DANGER),
        ("git switch --discard-changes main", DANGER),
        ("git switch -C main", DANGER),
        ("git switch main", WRITE),
        ("git branch -D x", DANGER),
        ("git branch -d -f x", DANGER),
        ("git branch -df x", DANGER),
        ("git branch --delete --force x", DANGER),
        ("git branch --force x HEAD~3", DANGER),
        ("git branch -d x", WRITE),
        ("git branch x", WRITE),
        ("git branch", READ),
        ("git tag -d v1", DANGER),
        ("git tag v1", WRITE),
        ("git push --force-with-lease", DANGER),
        ("git push --mirror", DANGER),
        ("git push --prune", DANGER),
        ("git push origin :old", DANGER),
        ("git push origin +main", DANGER),
        ("git push --delete origin x", DANGER),
        ("git gc --prune=now", DANGER),
        ("git gc", WRITE),
        ("git reflog expire --all", DANGER),
        ("git reflog", READ),
        ("git worktree remove x", DANGER),
        ("git filter-branch --all", DANGER),
    ],
)
def test_git_levels(repo: Path, command: str, expected: CommandLevel | None) -> None:
    verdict = classify(command, root=repo)
    assert verdict.level is expected, (command, verdict.reason)


def test_git_in_nested_bare_repository_asks(repo: Path) -> None:
    subprocess.run(["git", "init", "-q", "--bare", "pkg.git"], cwd=repo, check=True)
    assert level("git -C pkg.git log", repo) is None
    assert level("git log", repo) is READ


def test_git_read_when_repo_config_cannot_be_checked(
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert level("git status", repo) is READ
    monkeypatch.setattr(
        "gigacrew.safety.gitconfig.repo_config_runs_programs",
        lambda *args, **kwargs: None,
    )
    assert level("git status", repo) is None


def test_git_read_with_program_in_repo_config_asks(repo: Path) -> None:
    subprocess.run(["git", "config", "diff.evil.textconv", "sh -c id"], cwd=repo, check=True)
    assert level("git diff", repo) is None
    assert level("git log -p", repo) is None


def test_git_in_nested_repository_asks(repo: Path) -> None:
    nested = repo / "pkg"
    subprocess.run(["git", "init", "-q"], cwd=nested, check=True)
    assert level("cd pkg && git status", repo) is None
    assert level("git -C pkg status", repo) is None
    assert level("git status", repo) is READ


# --------------------------- выполнение кода -------------------------- #
@pytest.mark.parametrize(
    ("command", "signature"),
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
def test_exec_with_signature(root: Path, command: str, signature: str) -> None:
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
def test_exec_without_signature_only_exact(root: Path, command: str) -> None:
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
def test_exec_arguments_out_of_project_ask(root: Path, command: str) -> None:
    assert level(command, root) is None


def test_script_signature_changes_with_content(root: Path) -> None:
    script = root / "check.sh"
    script.write_text("#!/bin/sh\necho ok\n")
    script.chmod(0o755)
    first = classify("./check.sh", root=root).signatures
    script.write_text("#!/bin/sh\necho changed\n")
    assert classify("./check.sh", root=root).signatures != first
    with_args = classify("./check.sh --flag", root=root)
    assert with_args.signatures == _NO_SIGNATURES  # с аргументами — точно


def test_make_dry_run_still_runs_makefile_code(root: Path) -> None:
    # GNU make разбирает Makefile ($(shell …)) и выполняет «+»-рецепты и при -n/-q, а -p
    # после вывода базы собирает цели как обычно
    (root / "Makefile").write_text("X := $(shell touch pwned)\nall:\n\t+touch built\n")
    for command in ("make -n", "make -p", "make -q", "make --dry-run", "make -n test"):
        verdict = classify(command, root=root)
        assert verdict.level is EXEC, (command, verdict.reason)
        assert verdict.signatures == _NO_SIGNATURES, command  # только точная команда
    assert classify("make test", root=root).signatures == ("make test",)


@pytest.mark.parametrize(
    ("command", "expected", "signatures"),
    [
        ("make -j 4 test", EXEC, ("make test",)),
        ("make -j8 test", EXEC, ("make test",)),
        ("make --jobs 4 test", EXEC, ("make test",)),
        # у -j значение необязательно: следующее слово — цель
        ("make -j test", EXEC, ("make test",)),
        ("make -j install", DANGER, ()),
        ("make -j deploy", DANGER, ()),
        # переменные в аргументах меняют рецепты
        ("make test PYTHON=python3", EXEC, ()),
        ("make test PYTHON=/tmp/evil", None, ()),
        ("make SHELL=./x test", DANGER, ()),
        ("make MAKE=./x test", DANGER, ()),
    ],
)
def test_make_arguments(
    root: Path,
    command: str,
    expected: CommandLevel | None,
    signatures: tuple[str, ...],
) -> None:
    verdict = classify(command, root=root)
    assert verdict.level is expected, (command, verdict.reason)
    assert verdict.signatures == signatures, command


def test_npm_script_must_exist(root: Path) -> None:
    assert level("npm run lint", root) is None  # нет в package.json
    assert level("npm run deploy", root) is DANGER


# ----------------------- запись, сеть, опасное ------------------------ #
@pytest.mark.parametrize(
    ("command", "expected"),
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
def test_levels(root: Path, command: str, expected: CommandLevel | None) -> None:
    verdict = classify(command, root=root)
    assert verdict.level is expected, verdict.reason


def test_edits_only(root: Path) -> None:
    assert classify("sed -i 's/a/b/' a.txt", root=root).edits_only
    assert classify("ruff format . && ls", root=root).edits_only
    assert not classify("rm a.txt", root=root).edits_only  # удаление — не правка
    assert not classify("pip install x", root=root).edits_only


# ------------------------------ разбор ------------------------------- #
def test_cd_prefix_becomes_cwd(root: Path) -> None:
    verdict = classify("cd src && ls -la", root=root)
    assert verdict.cwd == "src"
    assert verdict.command == "ls -la"
    verdict = classify("ls", root=root, cwd="src")
    assert verdict.cwd == "src"
    assert verdict.level is READ
    mixed = classify("cd src && cd /tmp && ls", root=root)
    assert mixed.cwd == "src"
    assert mixed.command == "cd /tmp && ls"
    assert mixed.level is None
    assert classify("cd src; ls", root=root).level is None  # после «;» cd мог не сработать
    assert classify("cd ../.. && ls", root=root).cwd == "."


def test_composable_and_parts(root: Path) -> None:
    verdict = classify("pytest -q 2>&1 | tail -30", root=root)
    assert [part.level for part in verdict.parts] == [EXEC, READ]
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
def test_every_command_node_is_classified(root: Path, command: str) -> None:
    """Каждая простая команда дерева разобрана (ничего не спрятано в heredoc и т.п.)."""
    tree = treesitter.parse_tree("shell", command)
    assert tree is not None
    nodes = 0
    stack = [tree]
    while stack:
        node = stack.pop()
        nodes += node.type == "command"
        stack.extend(node.children)
    script = shell.parse(command, root=root)
    assert script is not None
    assert len(script.commands) + script.consumed == nodes


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ('echo "a\nb"', "a\nb"),
        ('echo "\n  x = 1\n"', "\n  x = 1\n"),
        (f'echo "a{_BACKSLASH}"b\nc{_BACKSLASH}{_BACKSLASH}d"', f'a"b\nc{_BACKSLASH}d'),
        (f'echo "a{_BACKSLASH}\nb"', "ab"),  # перенос строки через \ внутри "…" убирается
        ("echo 'a\nb'", "a\nb"),
        ('echo x"a\nb"y', "xa\nby"),
    ],
)
def test_quoted_word_keeps_newlines(root: Path, command: str, expected: str) -> None:
    """Значение слова в кавычках — как его получит программа, с переводами строк."""
    script = shell.parse(command, root=root)
    assert script is not None
    first_command = script.commands[0]
    assert first_command.argv[1].text == expected


def test_long_commands_and_other_platforms_ask(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    long_word = "x" * 20_000
    assert level(f"echo {long_word}", root) is None
    monkeypatch.setattr("gigacrew.safety.classifier._is_posix", _not_posix)
    assert level("ls", root) is None


def test_child_environment_drops_shell_hooks_and_adds_git_safety(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = subprocess_env(
        {
            "CDPATH": "/",
            "BASH_ENV": "x",
            "ENV": "y",
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "user.name",
            "GIT_CONFIG_VALUE_0": "me",
        },
    )
    assert "CDPATH" not in env
    assert "BASH_ENV" not in env
    assert "ENV" not in env
    assert env["GIT_CONFIG_COUNT"] == "3"
    assert env["GIT_CONFIG_KEY_0"] == "user.name"
    assert env["GIT_CONFIG_KEY_1"] == "safe.bareRepository"
    assert env["GIT_CONFIG_VALUE_2"] == "false"
    # окружение самого процесса не меняется
    monkeypatch.setenv("CDPATH", "/x")
    assert "CDPATH" not in subprocess_env()
    assert os.environ["CDPATH"] == "/x"


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
        "python3 -c 'print(().__class__.__base__.__subclasses__())'",
    ],
)
def test_hidden_or_dynamic_code_is_marked_dangerous(root: Path, command: str) -> None:
    """Спросить И пометить опасным: человек должен посмотреть внимательно."""
    verdict = classify(command, root=root)
    assert verdict.level is DANGER, verdict.reason


# ------------------------------ Python ------------------------------- #
@pytest.fixture
def venv_site(root: Path) -> Path:
    venv = root / ".venv"
    site = venv / "lib" / "python3.13" / "site-packages"
    site.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    (venv / "bin").mkdir()
    (venv / "bin" / "python").symlink_to("/usr/bin/python3")
    return site


@pytest.mark.parametrize(
    "line",
    [
        "import _virtualenv",
        "import _distutils_hack",
        "import __editable___pkg_1_0_finder; __editable___pkg_1_0_finder.install()",
        (
            "import os; var = 'SETUPTOOLS_USE_DISTUTILS'; enabled = os.environ.get(var, 'local') "
            "== 'local'; enabled and __import__('_distutils_hack').add_shim(); "
        ),
    ],
)
def test_standard_pth_lines_are_trusted(root: Path, venv_site: Path, line: str) -> None:
    (venv_site / "std.pth").write_text(f"{line}\n")
    assert level(".venv/bin/python -c 'print(1)'", root) is READ


@pytest.mark.parametrize(
    "line",
    [
        # код, дописанный после разрешённого начала строки
        "import _virtualenv; __import__('os').system('id')",
        "import _distutils_hack; __import__('os').system('id')",
        "import __editable___pkg_finder; __editable___pkg_finder.install(); import os",
        "import __editable___pkg_finder; __editable___pkg_finder.install(os.system('id'))",
        "import __editable___a_finder; __editable___b_finder.install()",
        "import os; var = 'SETUPTOOLS_USE_DISTUTILS'; os.system('id')",
        "import os; os.system('id')",
    ],
)
def test_pth_with_appended_code_is_exec(root: Path, venv_site: Path, line: str) -> None:
    (venv_site / "evil.pth").write_text(f"{line}\n")
    for command in (
        ".venv/bin/python -c 'print(1)'",
        ".venv/bin/python -m platform",
        ".venv/bin/python -m json.tool package.json",
    ):
        verdict = classify(command, root=root)
        assert verdict.level is EXEC, (command, verdict.reason)


def test_sitecustomize_in_venv_is_exec(root: Path, venv_site: Path) -> None:
    assert level(".venv/bin/python -c 'print(1)'", root) is READ
    (venv_site / "sitecustomize.py").write_text("import os; os.system('id')\n")
    assert level(".venv/bin/python -c 'print(1)'", root) is EXEC


def test_python_startup_code_on_pythonpath(root: Path) -> None:
    lib = root / "lib"
    lib.mkdir()
    (lib / "json.py").write_text("print('evil')\n")
    # -P убирает только каталог запуска: PYTHONPATH действует
    assert level("PYTHONPATH=lib python3 -P -c 'import json'", root) is EXEC
    assert level("PYTHONPATH=lib python3 -c 'import json'", root) is EXEC
    assert level("PYTHONPATH=lib python3 -I -c 'import json'", root) is READ
    assert level("PYTHONPATH=lib python3 -E -c 'print(1)'", root) is READ
    # sitecustomize/usercustomize из PYTHONPATH site импортирует при старте
    site = root / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text("import os; os.system('id')\n")
    assert level("PYTHONPATH=site python3 -c 'print(1)'", root) is EXEC
    assert level("PYTHONPATH=site python3 -P -c 'print(1)'", root) is EXEC
    assert level("PYTHONPATH=site python3 -m json.tool package.json", root) is EXEC
    assert level("PYTHONPATH=site python3 -I -c 'print(1)'", root) is READ
    assert level("PYTHONPATH=site python3 -S -c 'print(1)'", root) is READ
    assert level("python3 -c 'print(1)'", root) is READ  # каталог запуска — не PYTHONPATH


def test_stdlib_shadowing_in_run_directory_is_exec(root: Path) -> None:
    (root / "json.py").write_text("print('evil')\n")
    assert level("python3 -c 'print(1)'", root) is EXEC
    assert level("python3 -I -c 'print(1)'", root) is READ
    assert level("python3 -P -c 'print(1)'", root) is READ
    # -I не делает «чтением» сторонний код
    (root / "numpy").mkdir()
    (root / "numpy" / "__init__.py").write_text("")
    assert level("python3 -c 'import numpy'", root) is EXEC
    assert level("python3 -I -c 'import numpy'", root) is EXEC


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("python3 -m json.tool package.json /tmp/o.json", DANGER),
        ("python3 -m json.tool package.json .git/config", DANGER),
        ("python3 -m json.tool package.json out.json", WRITE),
        ("python3 -m json.tool package.json", READ),
        ("python3 -W ignore -c 'print(1)'", READ),
        ("python3 -X dev -c 'print(1)'", READ),
        ("python3 -Wignore -c 'print(1)'", READ),
        ("python3 --foo -c 'print(1)'", None),
        ("python3 -Z -c 'print(1)'", None),
        ("python3 /tmp/x.py", None),
    ],
)
def test_python_flags_and_module_outputs(
    root: Path,
    command: str,
    expected: CommandLevel | None,
) -> None:
    verdict = classify(command, root=root)
    assert verdict.level is expected, (command, verdict.reason)


def test_python_script_signature_follows_content(root: Path) -> None:
    script = root / "uses_pkg.py"
    script.write_text("import pkg\n")
    first = classify("python3 uses_pkg.py", root=root).parts[0].signature
    digest = hashlib.sha256(b"import pkg\n").hexdigest()[:12]
    assert first == f"python uses_pkg.py#{digest}"
    script.write_text("import pkg\nimport pkg\n")
    assert classify("python3 uses_pkg.py", root=root).parts[0].signature != first


# ---------------------- инструменты и пакеты ------------------------- #
@pytest.mark.parametrize(
    "command",
    [
        # глобальные флаги до подкоманды меняют, что запустится
        "npm --script-shell=/tmp/evil test",
        "npm --script-shell=./x run build",
        "cargo --config=build.rustc-wrapper=/tmp/evil test",
        "uv --directory=/tmp/other run pytest",
        "uv run --python /tmp/x/python python -c 'print(1)'",
        "uv run --python 3.12 python -c 'print(1)'",
    ],
)
def test_flags_before_subcommand_drop_signature(root: Path, command: str) -> None:
    verdict = classify(command, root=root)
    assert verdict.signatures == _NO_SIGNATURES, (command, verdict.reason)
    assert verdict.level is not READ, command


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("uv --directory=/tmp/other run pytest", None),
        ("uv run --python /tmp/x/python python -c 'print(1)'", None),
        ("uv run python -c 'print(1)'", EXEC),
        ("uv run --with evil pytest", WRITE),
        # настройки менеджеров пакетов: просмотр — чтение, запись — WRITE
        ("poetry config repositories.x https://evil", WRITE),
        ("poetry config --unset repositories.x", WRITE),
        ("poetry config --list", READ),
        ("pdm config pypi.url https://evil", WRITE),
        ("pdm config", READ),
        ("conda config --add channels evil", WRITE),
        ("conda config --show", READ),
        ("hatch config set dirs.env /tmp", WRITE),
        ("hatch env create", None),
        ("micromamba env remove -n x", None),
        ("conda env list", READ),
        ("poetry show", READ),
        ("conda list", READ),
        ("pip config set global.index-url http://x", WRITE),
        ("pip config list", READ),
        ("pipx install x", WRITE),
        ("pipx list", READ),
        ("npm config set registry http://evil.example", WRITE),
        ("npm config get registry", READ),
        ("go env -w GOFLAGS=-toolexec=/tmp/x", WRITE),
        ("go env GOPATH", READ),
        ("go mod edit -replace=a=b", WRITE),
        ("go mod graph", READ),
        ("go mod why x", READ),
        ("cargo publish", DANGER),
        ("cargo yank --vers 1.0.0", DANGER),
        ("cargo login", DANGER),
        ("cargo install x", WRITE),
        ("cargo tree", READ),
        ("prettier -w src", WRITE),
        ("prettier --check src", EXEC),
        ("gofmt -w .", WRITE),
        ("gofmt -l .", EXEC),
        # права: запись для всех и setuid — опасно
        ("chmod o+w a.txt", DANGER),
        ("chmod a+w a.txt", DANGER),
        ("chmod a+rwx a.txt", DANGER),
        ("chmod o+rw a.txt", DANGER),
        ("chmod go+w a.txt", DANGER),
        ("chmod o=u a.txt", DANGER),
        ("chmod 666 a.txt", DANGER),
        ("chmod 1777 a.txt", DANGER),
        ("chmod u+w a.txt", WRITE),
        ("chmod go-w a.txt", WRITE),
        ("chmod u=rwx,go=rx a.txt", WRITE),
        ("chmod 644 a.txt", WRITE),
        # списки файлов из файла
        ("find -files0-from a.txt -exec cat {} +", None),
    ],
)
def test_tool_levels(root: Path, command: str, expected: CommandLevel | None) -> None:
    verdict = classify(command, root=root)
    assert verdict.level is expected, (command, verdict.reason)


def test_npm_format_script_edits_files(root: Path) -> None:
    (root / "package.json").write_text('{"scripts": {"format": "prettier -w .", "lint": "x"}}')
    verdict = classify("npm run format", root=root)
    assert verdict.level is WRITE
    assert verdict.edits_only
    assert verdict.signatures == _NO_SIGNATURES
    assert classify("npm run lint", root=root).signatures == ("npm run lint",)


# ------------- точные уровни правил (обе стороны каждого) ------------- #
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        # sed: правка на месте и в группе флагов, запуск команд
        ("sed -Ei 's/a/b/' a.txt", WRITE),
        ("sed -ni 's/a/b/p' a.txt", WRITE),
        ("sed -i 's/a/b/' a.txt", WRITE),
        ("sed -n 's/a/b/p' a.txt", READ),
        ("sed 's/a/b/2g' a.txt", READ),
        ("sed 's/.*/id/e' a.txt", DANGER),
        ("sed -n '1e id' a.txt", DANGER),
        ("sed 's/a/b/w out.txt' a.txt", None),
        # awk: запуск команд — опасно, чтение файлов из программы — «не знаю»
        ("awk 'BEGIN{system(\"id\")}'", DANGER),
        ("awk '{print $1 | \"sh\"}' a.txt", DANGER),
        ("awk 'BEGIN{\"id\" | getline x; print x}'", DANGER),
        ("awk 'BEGIN{while((getline l < \".env\")>0) print l}'", None),
        ("awk 'BEGIN{getline l < \"/etc/shadow\"; print l}'", None),
        ('awk \'$1 == "x" || $2 == "y"\' a.txt', READ),
        # less: «+!» и «+|» запускают оболочку
        ("less +'!id' a.txt", DANGER),
        ("less '+|id' a.txt", DANGER),
        ("less +G a.txt", None),
        ("less a.txt", READ),
        ("vim -c ':!id' -c ':q' a.txt", DANGER),
        ("vim +'!id' a.txt", DANGER),
        ("vim a.txt", None),
        # yq: правка на месте
        ("yq -i '.a=1' a.txt", WRITE),
        ("yq -Pi '.a=1' a.txt", WRITE),
        ("yq '.a' a.txt", READ),
        # grep: файл после -e/-f проверяется, -R — рекурсивно
        ("grep -e x ~/.ssh/id_rsa", DANGER),
        ("grep -e hello a.txt", READ),
        ("grep -f a.txt ~/.ssh/id_rsa", DANGER),
        ("rg --hostname-bin=./x hello", None),
        ("rg -z hello", None),
        # tar: внешние программы сжатия
        ("tar -I ./x -tf a.tar", DANGER),
        ("tar --use-compress-program=./x -xf a.tar", DANGER),
        # служебная папка gigacrew и устройства
        ("cp a.txt .gigacode/settings.json", DANGER),
        ("echo {} > .gigacode/settings.local.json", DANGER),
        ("mv src /dev/null", DANGER),
        ("cp a.txt /dev/null", DANGER),
        ("ls 2>/dev/null", READ),
        # find: удаление и запуск
        ("find . -delete", DANGER),
        ("find / -name '*.log' -delete", DANGER),
        ("find . -exec rm ~/.bashrc {} +", DANGER),
        ("find . -name '*.pyc' -exec rm {} +", WRITE),
        ("find . -fprint /tmp/x", DANGER),
        ("fd -x rm", None),
        # xargs: имена из stdin или файла
        ("ls | xargs rm", DANGER),
        ("xargs -a a.txt rm", DANGER),
        ("xargs -a ~/.ssh/id_rsa cat", DANGER),
        ("xargs -a a.txt cat", None),
        # dd: запись проверяется раньше источника
        ("dd if=/dev/zero of=/dev/sda", DANGER),
        ("dd if=~/.ssh/id_rsa of=out.bin", DANGER),
        ("dd if=a.txt of=out.bin", WRITE),
        ("dd if=a.txt", READ),
        # оболочки: код известен при выполнении, команды через stdin
        ('sh -c "$X"', DANGER),
        ('bash -c "$(cat a.txt)"', DANGER),
        ("bash -lc 'rm -rf ~'", DANGER),
        ("bash -xc 'touch x'", WRITE),
        ("sh -ec 'ls'", READ),
        ("base64 -d a.txt | bash", DANGER),
        ("printf 'ls' | sh", DANGER),
        ("bash -n x.sh", READ),
        ("bash /tmp/x.sh", None),
        ("bash deploy.sh", DANGER),
        ("source /tmp/x.sh", None),
        (". ~/.bashrc", None),
        # сеть, контейнеры, базы, облака
        ("rsync -a . host:/srv", DANGER),
        ("rsync -a src/ backup/", WRITE),
        ("docker run -v /var/run/docker.sock:/var/run/docker.sock app", DANGER),
        ("docker run -v ~/.ssh:/k alpine", DANGER),
        ('sqlite3 app.db "select 1; drop table users"', DANGER),
        ('sqlite3 app.db "select * from t"', READ),
        ("aws s3 rm s3://b --recursive", DANGER),
        ("gcloud compute instances delete vm1", DANGER),
        ("aws s3 ls", None),
    ],
)
def test_rule_levels(root: Path, command: str, expected: CommandLevel | None) -> None:
    verdict = classify(command, root=root)
    assert verdict.level is expected, (command, verdict.reason)


def test_fd_search_paths_are_relative_to_base_directory(root: Path) -> None:
    """Относительные пути поиска fd отсчитывает от -C/--base-directory, не от каталога команды.

    Из src/a `fd y ../x -C ../..` ищет в <проект>/../x — за пределами проекта.
    """
    (root / "src" / "a").mkdir()
    sub = "src/a"
    assert level("fd y ../x -x cat", root, cwd=sub) is READ  # src/x — внутри
    assert level("fd y ../x -C ../.. -x cat", root, cwd=sub) is None
    assert level("fd y ../x --base-directory=../.. -x cat", root, cwd=sub) is None
    assert level("fd y --search-path ../x -C ../.. -X cat", root, cwd=sub) is None
    assert level("fd y ../x -C ../..", root, cwd=sub) is None
    assert level("fd y src -C ../.. -x cat", root, cwd=sub) is READ  # <проект>/src
    assert level(f"fd y {root}/src -C .. -x cat", root, cwd=sub) is READ  # абсолютный — как есть
    assert level("fd y '*' -C ../.. -x cat", root, cwd=sub) is READ  # шаблон fd в кавычках
    assert level("fd y * -C ../.. -x cat", root, cwd=sub) is None  # имена раскроет оболочка


def test_recursive_reads_follow_secrets_and_links(
    root: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    outside = tmp_path_factory.mktemp("outside") / "secret.txt"
    outside.write_text("s")
    assert level("grep -R hello .", root) is READ
    assert level("find . -type f | xargs cat", root) is READ
    assert level("fd -x cat", root) is READ
    (root / "src" / "lnk").symlink_to(outside)
    assert level("find src -type f -exec cat {} +", root) is None
    assert level("find src -type f | xargs cat", root) is None
    assert level("fd -x cat", root) is None
    assert level("fd x src -x cat", root) is None  # ссылка в каталоге поиска до -x
    assert level("fd x . src -x cat", root) is None
    (root / "src" / "lnk").unlink()
    (root / ".env").write_text("TOKEN=1\n")
    assert level("grep -R hello .", root) is None
    assert level("grep --directories=recurse x .", root) is None
    assert level("rg -uu x", root) is None
    assert level("find . -type f | xargs cat", root) is None
    assert level("fd -x cat", root) is None
    assert level("grep -e TOKEN .env", root) is None


def test_source_project_venv_activate_is_prelude(root: Path) -> None:
    venv = root / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    (venv / "bin" / "activate").write_text("# activate\n")
    (root / "scripts").mkdir()
    (root / "scripts" / "activate").write_text("echo hi\n")
    real = classify("source .venv/bin/activate && pytest", root=root)
    assert real.parts[0].prelude
    assert real.parts[0].signature == "venv-activate"
    fake = classify("source scripts/activate && pytest", root=root)
    assert not fake.parts[0].prelude
    assert fake.parts[0].signature is None
