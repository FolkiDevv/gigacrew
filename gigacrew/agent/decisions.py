"""Память решений пользователя о командах: разрешения «всегда» и журнал выборов.

Когда классификатор не уверен, решает пользователь — и его выбор сохраняется вместе с
командой и контекстом. Хранится в каталоге пользователя, а не проекта: иначе репозиторий
мог бы принести готовые «разрешения» (``.gigacode/`` бывает закоммичен вместе со своим
``.gitignore``). Каталог проекта — ``~/.gigacode/projects/<ключ>/``:

* ``approvals.json`` — разрешения «всегда» (переживают сессии). Правило — программа с
  подкомандой (``pytest``, ``npm test``) или точная простая команда, в рабочем каталоге.
  Команда разрешена правилами, если каждая её часть — чтение или совпадает с правилом;
  перенаправления в файлы, опасные части и непонятные конструкции правилами не
  разрешаются. Перед записью файл перечитывается (другое окно gigacrew).
* ``decisions.jsonl`` — журнал: что спросили, почему, что ответил пользователь (и что
  написал вместо «да/нет»), в каком каталоге и режиме, чем кончилось выполнение. Также
  команды, выполненные без вопроса (только чтение, сохранённое правило). Это данные для
  будущей модели-классификатора и история для подсказок «раньше вы разрешали».
  Секреты (``TOKEN=…``, пароли в URL и флагах, заголовки авторизации — см. :func:`redact`)
  вырезаются из каждой записи; файл доступен только владельцу. Журнал отключается вместе
  с сохранением чатов (``--no-save``).

Команда с секретом в ``approvals.json`` не сохраняется: «всегда» для неё — до конца сессии.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import secrets
import tempfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from gigacrew.project.workspace import user_data_dir
from gigacrew.safety.levels import RULES_VERSION, CommandLevel, Verdict

__all__ = ["DecisionMemory", "Rule", "history_key", "project_key", "redact"]

FORMAT_VERSION = 1
PROJECTS_DIR = "projects"
APPROVALS_FILE = "approvals.json"
JOURNAL_FILE = "decisions.jsonl"
MAX_JOURNAL_BYTES = 5 * 1024 * 1024
MAX_RULES_PER_ANSWER = 5

RuleKind = Literal["program", "exact", "command"]
_KIND_LABELS = {"program": "программа", "exact": "команда", "command": "команда целиком"}


@dataclass(frozen=True)
class Rule:
    """Сохранённое разрешение.

    ``kind``: ``program`` — сигнатура (``pytest``) с проверенными аргументами;
    ``exact`` — простая команда как есть; ``command`` — вся команда как есть (когда
    по частям сверять нельзя: перенаправления в файлы, условия).
    """

    id: str
    cwd: str
    kind: RuleKind
    value: str
    level: str = ""
    created: str = ""
    hits: int = 0
    last_used: str = ""

    def describe(self) -> str:
        where = "корень проекта" if self.cwd == "." else self.cwd
        if self.kind == "program":
            return f"«{self.value}» с проверенными аргументами — {where}"
        return f"«{self.value}» — {where}"


def _now() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def project_key(root: Path) -> str:
    """Ключ проекта: путь корня; рабочее дерево git — путь основного репозитория."""
    return hashlib.sha256(str(_key_root(root)).encode("utf-8")).hexdigest()[:16]


def _key_root(root: Path) -> Path:
    """Путь, от которого считается ключ: у рабочего дерева git — основной репозиторий."""
    root = Path(os.path.normcase(str(Path(root).resolve())))
    return _main_worktree(root) or root


def _main_worktree(root: Path) -> Path | None:
    """Основной репозиторий связанного рабочего дерева (``root/.git`` — файл) или None.

    Ссылке верим, только если её подтверждает git: каталог ``<gitdir>`` есть и его файл
    ``gitdir`` указывает обратно на ``root/.git``. Иначе скачанный проект с поддельным
    ``.git`` делил бы с чужим проектом его память решений.
    """
    dot_git = root / ".git"
    try:
        if not dot_git.is_file():
            return None
        for line in dot_git.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith("gitdir:"):
                continue
            target = Path(line[len("gitdir:") :].strip())
            gitdir = (target if target.is_absolute() else root / target).resolve()
            # <основной>/.git/worktrees/<имя>
            if gitdir.parent.name != "worktrees" or gitdir.parent.parent.name != ".git":
                return None
            back = Path((gitdir / "gitdir").read_text(encoding="utf-8").strip())
            back = (back if back.is_absolute() else gitdir / back).resolve()
            if os.path.normcase(str(back)) != os.path.normcase(str(dot_git.resolve())):
                return None
            return gitdir.parent.parent.parent
    except (OSError, ValueError):
        return None
    return None


# --------------------------------------------------------------------------- #
# Вырезание секретов из журнала
# --------------------------------------------------------------------------- #
_SECRET_ASSIGN = re.compile(
    r"\b([A-Za-z0-9_]*(?:KEY(?:_ID)?|TOKEN|SECRET|PASS(?:WORD)?|PWD|CRED\w*|AUTH\w*))="
    r"(\"[^\"]*\"|'[^']*'|\S+)",
    re.IGNORECASE,
)
_URL_CREDS = re.compile(r"(\b[a-z][a-z0-9+.-]*://)([^/\s:@]+):([^/\s@]+)@", re.IGNORECASE)
_AUTH_HEADER = re.compile(r"(authorization:\s*(?:bearer|basic|token)?\s*)[^\s'\"]+", re.I)
# --password, --access-token, --client-secret, --api-key…
_FLAG_SECRET = re.compile(r"(--[\w-]*(?:password|passwd|token|secret|api-?key)[=\s]+)(\S+)", re.I)
_SECRET_NAME = r"[\w-]*(?:key|token|secret|password|passwd)"
# Значение в кавычках целиком или слово.
_VALUE = r"('[^']*'|\"[^\"]*\"|[^\s'\"]+)"
# mysql -phunter2 (пароль слитно с -p); sshpass -p hunter2
_MYSQL_PASS = re.compile(r"(\b(?:mysql\w*|mariadb\w*)\b[^|;&\n]*?\s-p)" + _VALUE)
_SSHPASS = re.compile(r"(\bsshpass\b[^|;&\n]*?\s-p\s*)" + _VALUE)
# curl -u user:pass, --user=user:pass
_USER_PASS = re.compile(r"(\s(?:-u\s*|--user(?:=|\s+))['\"]?)([^\s:'\"]+):([^\s'\"]+)")
# httpie: token:=…
_HTTPIE_JSON = re.compile(rf"(\b{_SECRET_NAME}[\w-]*:=)(\S+)", re.I)
# заголовки и YAML: X-Api-Key: …, api_token: …
_HEADER = re.compile(rf"(\b{_SECRET_NAME}:[ \t]*)([^\s'\"=][^\s'\"]*)", re.I)
# JSON: "api_key": "…"
_JSON_FIELD = re.compile(rf"(\"{_SECRET_NAME}\"\s*:\s*\")[^\"]*(\")", re.I)
# Узнаваемые токены сервисов (GitHub, GitLab, Slack, OpenAI, AWS)
_KNOWN_TOKEN = re.compile(
    r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_\w{20,}|glpat-[\w-]{20,}"
    r"|xox[abprs]-[\w-]{10,}|sk-[\w-]{20,}|AKIA[0-9A-Z]{16})\b"
)


def redact(text: str) -> str:
    """Убрать из команды похожее на секреты: ``TOKEN=…``, пароли в URL и флагах
    (``--password``, ``mysql -p…``, ``curl -u user:pass``), заголовки авторизации и
    ключей, поля JSON с ключами, узнаваемые токены сервисов."""
    text = _SECRET_ASSIGN.sub(lambda m: f"{m.group(1)}=***", text)
    text = _URL_CREDS.sub(lambda m: f"{m.group(1)}{m.group(2)}:***@", text)
    text = _AUTH_HEADER.sub(lambda m: f"{m.group(1)}***", text)
    text = _FLAG_SECRET.sub(lambda m: f"{m.group(1)}***", text)
    text = _MYSQL_PASS.sub(lambda m: f"{m.group(1)}***", text)
    text = _SSHPASS.sub(lambda m: f"{m.group(1)}***", text)
    text = _USER_PASS.sub(lambda m: f"{m.group(1)}{m.group(2)}:***", text)
    text = _HTTPIE_JSON.sub(lambda m: f"{m.group(1)}***", text)
    text = _HEADER.sub(lambda m: f"{m.group(1)}***", text)
    text = _JSON_FIELD.sub(lambda m: f"{m.group(1)}***{m.group(2)}", text)
    return _KNOWN_TOKEN.sub("***", text)


# --------------------------------------------------------------------------- #
# Память
# --------------------------------------------------------------------------- #
@dataclass
class _History:
    allowed: int = 0
    rejected: int = 0


@dataclass
class DecisionMemory:
    """Разрешения «всегда» и журнал решений проекта.

    ``directory=None`` — только память (тесты, агент без CLI): разрешения действуют до
    конца сессии, журнал не пишется. ``remember=False`` — разрешения не сохраняются на
    диск (``GIGACREW_REMEMBER_APPROVALS=0``), ``log=False`` — журнал не пишется.
    """

    directory: Path | None = None
    root: str = ""
    remember: bool = True
    log: bool = True
    rules: list[Rule] = field(default_factory=list)
    _history: dict[tuple[str, str], _History] = field(default_factory=dict, repr=False)
    # Что пользователь отклонил в этой сессии (ключ истории): такое не отдаётся рецензенту.
    _rejected: set[tuple[str, str]] = field(default_factory=set, repr=False)

    # ------------------------------------------------------------------ #
    @classmethod
    def load(
        cls,
        root: Path,
        *,
        remember: bool = True,
        log: bool = True,
        base: Path | None = None,
    ) -> tuple[DecisionMemory, str | None]:
        """Память проекта ``root`` из ``~/.gigacode/projects/<ключ>/`` (или ``base``).

        Ошибка чтения — пустая память и текст предупреждения (файл перезапишется).
        """
        directory = (
            (base if base is not None else user_data_dir()) / PROJECTS_DIR / project_key(root)
        )
        # Корень в файле — путь ключа: рабочие деревья одного репозитория делят разрешения.
        memory = cls(directory, str(_key_root(root)), remember=remember, log=log)
        warning = None
        try:
            memory.rules = memory._read_rules()
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as e:
            warning = f"не удалось прочитать {directory / APPROVALS_FILE}: {e}"
        if log:
            memory._load_history()
        return memory, warning

    @property
    def approvals_path(self) -> Path | None:
        return None if self.directory is None else self.directory / APPROVALS_FILE

    @property
    def journal_path(self) -> Path | None:
        return None if self.directory is None else self.directory / JOURNAL_FILE

    # ----------------------------- правила ----------------------------- #
    def match(self, verdict: Verdict, *, plan: bool = False) -> Rule | None:
        """Правило, разрешающее команду, или None.

        Опасное (DANGER) не разрешается никогда. В режиме плана — только то, что не
        меняет файлы (части уровня READ и EXEC).
        """
        if verdict.level is CommandLevel.DANGER:
            return None
        if not verdict.composable:
            if plan or verdict.level is None:
                return None
            return self._find("command", verdict.cwd, verdict.command)
        matched: Rule | None = None
        program_hit = False
        pending_preludes = False
        for part in verdict.parts:
            if part.level is CommandLevel.READ:
                continue
            if part.level is CommandLevel.DANGER:
                return None
            if plan and part.level is not CommandLevel.EXEC:
                return None
            if part.prelude:
                pending_preludes = True
                continue
            rule = None
            if part.signature and part.level is CommandLevel.EXEC:
                rule = self._find("program", verdict.cwd, part.signature)
                program_hit = program_hit or rule is not None
            if rule is None and part.text:
                rule = self._find("exact", verdict.cwd, part.text)
            if rule is None:
                return None
            matched = matched or rule
        if pending_preludes and not program_hit:
            return None  # активация venv — только вместе с разрешённой программой
        if matched is not None:
            self._touch(matched)
        return matched

    def offer(self, verdict: Verdict | None) -> str:
        """Что запомнит ответ «всегда» (для вопроса пользователю); "" — нечего."""
        rules = self._candidates(verdict)
        if not rules:
            return ""
        what = "; ".join(r.describe() for r in rules)
        where = (
            "для проекта" if self.directory is not None and self.remember else ("до конца сессии")
        )
        return f"всегда разрешать {what} (запомнится {where}; /permissions — список)"

    def approve(self, verdict: Verdict) -> list[Rule]:
        """Ответ «всегда»: сохранить правила для частей команды, которые спрашивались."""
        rules = self._candidates(verdict)
        added: list[Rule] = []
        for rule in rules:
            if self._find(rule.kind, rule.cwd, rule.value) is None:
                self.rules.append(rule)
                added.append(rule)
        if added:
            self._save(added=added)
        return added

    def _candidates(self, verdict: Verdict | None) -> list[Rule]:
        rules = self._proposed(verdict)
        # Правила, которые match() к этой же команде не применит (непроверенная команда
        # с условием или перенаправлением, активация venv без программы, частей больше
        # предела), не предлагаются: иначе «всегда» обещает то, чего не будет.
        if verdict is None or not rules or DecisionMemory(rules=rules).match(verdict) is None:
            return []
        # Правило — команда как есть: секрет из него не вырезать, и на диск оно не идёт.
        if any(r.kind != "program" and redact(r.value) != r.value for r in rules):
            return []
        return rules

    def _proposed(self, verdict: Verdict | None) -> list[Rule]:
        if verdict is None or verdict.level is CommandLevel.DANGER:
            return []
        now = _now()

        def make(kind: RuleKind, value: str, level: CommandLevel | None) -> Rule:
            return Rule(
                secrets.token_hex(4),
                verdict.cwd,
                kind,
                value,
                level.name if level is not None else "UNKNOWN",
                now,
            )

        if not verdict.composable or not verdict.parts:
            return [make("command", verdict.command, verdict.level)] if verdict.command else []
        out: list[Rule] = []
        has_program = any(
            p.signature and p.level is CommandLevel.EXEC and not p.prelude for p in verdict.parts
        )
        for part in verdict.parts:
            if part.level is CommandLevel.READ or part.level is CommandLevel.DANGER:
                continue
            if part.prelude and has_program:
                continue
            if part.signature and part.level is CommandLevel.EXEC and not part.prelude:
                rule = make("program", part.signature, part.level)
            elif part.text:
                rule = make("exact", part.text, part.level)
            else:
                return [make("command", verdict.command, verdict.level)]
            if all((r.kind, r.value) != (rule.kind, rule.value) for r in out):
                out.append(rule)
        if len(out) > MAX_RULES_PER_ANSWER:
            return [make("command", verdict.command, verdict.level)]
        return out

    def _find(self, kind: str, cwd: str, value: str) -> Rule | None:
        for rule in self.rules:
            if rule.kind == kind and rule.cwd == cwd and rule.value == value:
                return rule
        return None

    def _touch(self, rule: Rule) -> None:
        updated = replace(rule, hits=rule.hits + 1, last_used=_now())
        self.rules = [updated if r.id == rule.id else r for r in self.rules]
        with contextlib.suppress(OSError):
            self._save(touched=[updated])

    def forget(self, index: int) -> Rule:
        """Удалить правило по номеру в списке (с 1). IndexError — нет такого."""
        if index < 1 or index > len(self.rules):
            raise IndexError(index)
        rule = self.rules.pop(index - 1)
        self._save(removed={rule.id})
        return rule

    def clear(self) -> int:
        n = len(self.rules)
        removed = {r.id for r in self.rules}
        self.rules = []
        self._save(removed=removed)
        return n

    # ----------------------------- файлы ------------------------------ #
    def _read_rules(self) -> list[Rule]:
        path = self.approvals_path
        if path is None:
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("version") != FORMAT_VERSION:
            raise ValueError("неподдерживаемая версия формата")
        stored = data.get("root")
        # Корень рабочего дерева, а не основного репозитория (так писали прежние версии) —
        # тот же проект.
        if stored not in (None, "", self.root) and not (
            isinstance(stored, str) and project_key(Path(stored)) == project_key(Path(self.root))
        ):
            raise ValueError(f"файл другого проекта ({stored})")
        rules = []
        for item in data.get("rules", []):
            if not isinstance(item, dict):
                continue
            try:
                rule = Rule(**{k: item[k] for k in Rule.__dataclass_fields__ if k in item})
            except TypeError:
                continue
            if rule.kind in ("program", "exact", "command"):
                rules.append(rule)
        return rules

    def _save(
        self, *, added: list[Rule] = (), touched: list[Rule] = (), removed: set[str] = frozenset()
    ) -> None:
        """Перечитать файл, применить изменения и записать атомарно."""
        path = self.approvals_path
        if path is None or not self.remember:
            return
        try:
            current = {r.id: r for r in self._read_rules()}
        except FileNotFoundError:
            current = {}
        except (OSError, ValueError):
            current = {}
        for r in added:
            current[r.id] = r
        for r in touched:
            if r.id in current:
                current[r.id] = r
        for rid in removed:
            current.pop(rid, None)
        self.rules = list(current.values())
        _write_json(
            path,
            {
                "version": FORMAT_VERSION,
                "root": self.root,
                "rules": [asdict(r) for r in self.rules],
            },
        )

    # ----------------------------- журнал ----------------------------- #
    def record(self, entry: dict[str, Any]) -> str:
        """Записать решение в журнал. Возвращает id записи (для :meth:`outcome`)."""
        entry_id = secrets.token_hex(6)
        decision = entry.get("decision")
        if entry.get("source") == "user" and "key" in entry:
            key = (str(entry.get("cwd", ".")), str(entry["key"]))
            h = self._history.setdefault(key, _History())
            if decision in ("yes", "always"):
                h.allowed += 1
            elif decision == "no":
                h.rejected += 1
                self._rejected.add(key)
        if not self.log or self.journal_path is None:
            return entry_id
        line = {
            "id": entry_id,
            "ts": _now(),
            "event": "decision",
            "rules_version": RULES_VERSION,
            **entry,
        }
        for k in ("command", "feedback", "description", "key"):
            if isinstance(line.get(k), str):
                line[k] = redact(line[k])
        for k in ("verdict", "review"):  # причины могут цитировать команду
            nested = line.get(k)
            if isinstance(nested, dict) and isinstance(nested.get("reason"), str):
                line[k] = {**nested, "reason": redact(nested["reason"])}
        self._append(line)
        return entry_id

    def outcome(self, entry_id: str, **fields: Any) -> None:
        if not self.log or self.journal_path is None:
            return
        # Текст ошибки может цитировать команду.
        fields = {k: redact(v) if isinstance(v, str) else v for k, v in fields.items()}
        self._append({"id": entry_id, "ts": _now(), "event": "outcome", **fields})

    def history(self, key: tuple[str, str]) -> tuple[int, int]:
        """(сколько раз разрешали, сколько отклоняли) команду или программу в каталоге."""
        h = self._history.get(key)
        return (h.allowed, h.rejected) if h is not None else (0, 0)

    def rejected_in_session(self, key: tuple[str, str]) -> bool:
        """Пользователь отклонял эту команду (программу) в каталоге в этой сессии."""
        return key in self._rejected

    def _append(self, line: dict[str, Any]) -> None:
        path = self.journal_path
        assert path is not None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with contextlib.suppress(OSError):
                if path.stat().st_size > MAX_JOURNAL_BYTES:
                    os.replace(path, path.with_suffix(".1.jsonl"))
            fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        except OSError:
            pass  # журнал — не повод прерывать работу

    def _load_history(self) -> None:
        for line in self._journal_lines():
            if line.get("event") != "decision" or line.get("source") != "user":
                continue
            key = (str(line.get("cwd", ".")), str(line.get("key", "")))
            h = self._history.setdefault(key, _History())
            if line.get("decision") in ("yes", "always"):
                h.allowed += 1
            elif line.get("decision") == "no":
                h.rejected += 1

    def _journal_lines(self) -> Iterator[dict[str, Any]]:
        path = self.journal_path
        if path is None:
            return
        try:
            with open(path, encoding="utf-8") as f:
                for raw in f:
                    with contextlib.suppress(ValueError):
                        item = json.loads(raw)
                        if isinstance(item, dict):
                            yield item
        except OSError:
            return

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """Последние решения пользователя и модели-рецензента из журнала (новые в конце)."""
        items = [
            x
            for x in self._journal_lines()
            if x.get("event") == "decision" and x.get("source") in ("user", "reviewer")
        ]
        return items[-limit:]


def history_key(verdict: Verdict) -> tuple[str, str]:
    """Ключ истории: программа (если одна) или команда целиком, в рабочем каталоге."""
    sigs = verdict.signatures
    return (verdict.cwd, sigs[0] if len(sigs) == 1 else verdict.command)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
