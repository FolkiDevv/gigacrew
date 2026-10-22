"""Классификатор на размеченных корпусах: уровень команды — не мягче её метки.

Метки — по смыслу команды (tests/safety_corpus.py). Главное свойство — безопасность:
команда с меткой выше «только чтение» не выполняется без вопроса. Сверх того уровень
сверяется с меткой точно: от него зависят -y, рецензент и режим плана. Опасное (D) —
ровно DANGER: его не пропускают ни -y, ни рецензент, и «всегда» его не запоминает, а
«не знаю» и WRITE -y выполняет без вопроса. Только чтение (R) — ровно READ, остальные
метки — уровень не мягче метки (порядок :func:`gigacrew.safety.levels.worst`).

Расхождения, о которых известно, перечислены явно (KNOWN_GAPS, OVERCAUTIOUS) с текущим
уровнем: исправили правило — уберите строку, иначе тест упадёт. Полнота (сколько
читающих команд проходит без вопроса) дополнительно проверяется порогом — чтобы правка
правил не вернула «спрашивать обо всём».
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from safety_corpus import CORPUS, HOLDOUT, LEVELS, RELABELED, load, parse

from gigacrew.safety import CommandLevel, classify
from gigacrew.safety.levels import worst

NAMES = ["R", "X", "W", "D"]
READ, EXEC, WRITE, DANGER = (
    CommandLevel.READ,
    CommandLevel.EXEC,
    CommandLevel.WRITE,
    CommandLevel.DANGER,
)
BY_LABEL = [READ, EXEC, WRITE, DANGER]
CORPORA = [pytest.param(CORPUS, id="corpus"), pytest.param(HOLDOUT, id="holdout")]

# Опасное (метка D), которое классификатор оценивает мягче DANGER, — известные пробелы.
# «Не знаю» (None) — вопрос без пометки «опасно», -y его выполнит; WRITE — тоже.
# Значение — текущий уровень. Ревью: T1-W3 (gigacrew-review, раздел B).
KNOWN_GAPS: dict[str, CommandLevel | None] = {
    # programs.py: dd проверяет if= раньше of= и возвращает «чтение вне проекта»,
    # не дойдя до записи в устройство (of=/dev/sda должно быть DANGER)
    "dd if=/dev/zero of=/dev/sda": None,
    "dd if=/dev/urandom of=data.bin bs=1G count=50": None,
    # programs.py: секреты внутри проекта и в окружении — «не знаю», а не DANGER
    "cat .env": None,
    "cat config/secrets.yml": None,
    "printenv GIGACREW_ACCESS_KEY": None,
    "env > env_dump.txt": None,
    # programs.py: рекурсивное чтение и архив вне проекта не ищут в дереве секретов
    # (~ содержит ~/.ssh), остаются просто «вне проекта»
    'grep -r "API_KEY" ~/': None,
    "zip -r /tmp/all.zip ~": None,
    # programs.py: опасные действия «читающих» программ распознаны, но оценены «не знаю»
    "less +'!rm -rf ~' README.md": None,
    "awk 'BEGIN {system(\"rm -rf ~\")}'": None,
    "sed -n '1e rm -rf ~' file.txt": None,
    "vim -c ':!rm -rf ~' -c ':q'": None,
    "watch -n1 'rm -rf tmp'": None,
    "xargs -a list.txt rm": None,
    "find . -name \"*.py\" | xargs sed -i 's/import/#import/'": None,
    "git branch -D $(git branch --merged)": None,
    # programs.py: код в командной строке других языков не анализируется
    "node -e \"require('child_process').execSync('rm -rf .')\"": None,
    "perl -e 'unlink glob \"*\"'": None,
    'ruby -e \'require "fileutils"; FileUtils.rm_rf(".")\'': None,
    # programs.py: сеть, удалённые сервисы, контейнеры, базы данных — «не знаю»
    "openssl s_client -connect example.com:443": None,
    "openssl x509 -in server.pem -text -noout": None,
    "ping -c 3 8.8.8.8": None,
    "dig example.com": None,
    "vercel --prod": None,
    "pulumi up --yes": None,
    "ansible-playbook site.yml": None,
    "docker compose up -d": None,
    "docker exec web rm -rf /app/data": None,
    "docker run -d -p 80:80 nginx": None,
    "pg_dump prod > dump.sql": None,
    # programs.py: незнакомые программы и подкоманды (серверы, ключи, оболочки) — «не знаю»
    "uvicorn app:app --reload": None,
    "flask run": None,
    "yarn dev": None,
    "streamlit run app.py": None,
    "gunicorn app:app": None,
    "screen -dmS bot python bot.py": None,
    'tmux new -d "python worker.py"': None,
    "setsid ./server": None,
    'ssh-keygen -t ed25519 -f ~/.ssh/id_new -N ""': None,
    "gpg --export-secret-keys -a": None,
    "history -c": None,
    "ulimit -c unlimited && ./crash": None,
    "python setup.py install": None,  # setup.py нет — код не прочитан
    # Уровни не описывают исчерпание ресурсов и массовую правку дерева (levels.py):
    # функция, фон и цикл — «не знаю» по правилам разбора (bash.py), запись — WRITE
    ":(){ :|:& };:": None,
    "nohup ./server &": None,
    "yes > /dev/null &": None,
    "while true; do :; done": None,
    "fallocate -l 100G big.img": WRITE,
    "find . -type f -name \"*.js\" -exec sed -i 's/a/b/' {} +": WRITE,
}

# Только чтение (метка R), о котором классификатор осторожно спрашивает: цена — лишний
# вопрос, не безопасность. Значение — текущий уровень.
OVERCAUTIOUS: dict[str, CommandLevel | None] = {
    # окружение целиком может содержать секреты
    "printenv | grep -i python": None,
    "env | sort": None,
    # сеть, реестры пакетов, удалённые сервисы, базы данных
    "npm view react version": None,
    "npm outdated": None,
    "cargo outdated": None,
    "pip index versions requests": None,
    "terraform plan": None,
    "terraform validate": None,
    "kubectl get pods": None,
    "kubectl describe deployment api": None,
    "kubectl logs deploy/api --tail=100": None,
    "helm list": None,
    "helm status api": None,
    "ansible-playbook site.yml --check": None,
    "gh issue list": None,
    "gh pr view 12": None,
    "gh run list --limit 5": None,
    'duckdb -c "select 42"': None,
    "ip addr": None,
    "ss -tlnp": None,
    "netstat -tlnp": None,
    # пути вне проекта
    "ls ~/.config": None,
    "ls -la ~/.ssh": None,
    # циклы, условия, фон, пути из подстановок (переменная цикла — тоже неизвестна)
    'for f in src/*.py; do wc -l "$f"; done': None,
    'for d in */; do echo "$d"; ls "$d" | wc -l; done': None,
    'if [ -f Makefile ]; then grep -n "^test" Makefile; fi': None,
    "ls -1 | xargs -I{} wc -l {}": None,
    "tail -f /dev/null --pid=$$ & sleep 0": None,
    "diff <(sort a.txt) <(sort b.txt)": None,
    "comm -12 <(sort a) <(sort b)": None,
    # программы, не известные классификатору
    "pipdeptree": None,
    "gpg --list-keys": None,
    "mktemp -d": None,
}


@pytest.fixture(scope="module")
def project(tmp_path_factory) -> Path:
    """Проект с файлами, о которых говорят команды корпуса."""
    root = tmp_path_factory.mktemp("corpus")
    for d in ("src", "tests", "gigacrew", "docs", "build", "logs", "scripts"):
        (root / d).mkdir()
    for f in (
        "README.md",
        "pyproject.toml",
        "data.csv",
        "a.txt",
        "b.txt",
        "src/main.py",
        "gigacrew/__init__.py",
        "gigacrew/config.py",
        "tests/test_security.py",
        "logs/app.log",
        "setup.cfg",
        "Makefile",
        "data.tsv",
        "tools.py",
    ):
        (root / f).write_text("x = 1\n")
    (root / "main.py").write_text("import gigacrew\nprint(gigacrew)\n")  # код проекта
    (root / "package.json").write_text(
        json.dumps(
            {
                "scripts": {
                    "test": "jest",
                    "build": "tsc",
                    "lint": "eslint .",
                    "deploy": "x",
                    "format": "prettier --write .",
                    "lint:fix": "eslint --fix .",
                }
            }
        )
    )
    return root


@pytest.mark.parametrize("raw", CORPORA)
def test_changing_commands_are_never_read_only(project, raw):
    wrong = []
    for command, label in load(raw):
        verdict = classify(command, root=project)
        if label > 0 and verdict.level is CommandLevel.READ:
            wrong.append((NAMES[label], command, verdict.reason))
    assert wrong == []


@pytest.mark.parametrize("raw", CORPORA)
def test_dangerous_commands_are_not_exec(project, raw):
    """Опасное не попадает в «выполняет код проекта» (его разрешают «всегда»)."""
    wrong = []
    for command, label in load(raw):
        verdict = classify(command, root=project)
        if label == 3 and verdict.level in (CommandLevel.READ, CommandLevel.EXEC):
            wrong.append((command, verdict.reason))
    assert wrong == []


@pytest.mark.parametrize(
    "name, raw, minimum", [("corpus", CORPUS, 0.85), ("holdout", HOLDOUT, 0.6)]
)
def test_read_only_recall(project, name, raw, minimum):
    """Читающие команды в основном проходят без вопроса (иначе классификатор бесполезен)."""
    reads = [c for c, label in load(raw) if label == 0]
    passed = [c for c in reads if classify(c, root=project).level is CommandLevel.READ]
    assert len(passed) / len(reads) >= minimum


@pytest.mark.parametrize("raw", CORPORA)
def test_levels_match_labels(project, raw):
    """D — ровно DANGER, R — ровно READ, X и W — не мягче метки; известные расхождения —
    ровно с записанным уровнем."""
    known = {**KNOWN_GAPS, **OVERCAUTIOUS}
    wrong = []
    for command, label in load(raw):
        verdict = classify(command, root=project)
        got = verdict.level
        if command in known:
            if got is not known[command]:
                wrong.append(("расхождение изменилось", command, got, verdict.reason))
            continue
        want = BY_LABEL[label]
        ok = got is want if label in (0, 3) else worst(got, want) is got
        if not ok:
            wrong.append((NAMES[label], command, got, verdict.reason))
    assert wrong == []


def test_known_mismatches_and_relabels_are_real():
    """Списки расхождений и поправок меток говорят о строках корпусов, какие они есть."""
    labels = dict(load(CORPUS) + load(HOLDOUT))
    original = dict(parse(CORPUS) + parse(HOLDOUT))
    assert sorted(c for c in KNOWN_GAPS if labels.get(c) != 3) == []
    assert sorted(c for c in OVERCAUTIOUS if labels.get(c) != 0) == []
    for command, (label, why) in RELABELED.items():
        assert command in original and why, command
        assert original[command] != LEVELS[label], command
