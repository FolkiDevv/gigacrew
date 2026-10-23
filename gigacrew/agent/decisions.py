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
import json
import os
import re
import secrets
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal

from gigacrew.project.workspace import (
    PROJECTS_DIR,
    project_key,
    project_key_root,
    user_data_dir,
)
from gigacrew.safety.levels import RULES_VERSION, CommandLevel, Part, Verdict

__all__ = ["DecisionMemory", "Rule", "history_key", "project_key", "redact"]

FORMAT_VERSION = 1
APPROVALS_FILE = "approvals.json"
JOURNAL_FILE = "decisions.jsonl"
MAX_JOURNAL_BYTES = 5 * 1024 * 1024
MAX_RULES_PER_ANSWER = 5

RuleKind = Literal["program", "exact", "command"]
_PROGRAM: Final = "program"
_EXACT: Final = "exact"
_COMMAND: Final = "command"
_RULE_KINDS = frozenset((_PROGRAM, _EXACT, _COMMAND))
# Предложение правила до выдачи id: вид, текст, уровень.
_RuleSpec = tuple[RuleKind, str, CommandLevel | None]
_KIND_LABELS = MappingProxyType({
    _PROGRAM: "программа",
    _EXACT: "команда",
    _COMMAND: "команда целиком",
})

# Поле записи журнала с ответом и тип события «решение» (одно и то же слово).
_DECISION_KEY = "decision"
_DECISION_EVENT = "decision"
# Файлы памяти доступны только владельцу.
_PRIVATE_FILE_MODE = 0o600
_PRIVATE_DIR_MODE = 0o700


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
    # Имя поля — ключ в approvals.json и часть публичного API правила.
    value: str  # noqa: WPS110
    level: str = ""
    created: str = ""
    hits: int = 0
    last_used: str = ""

    def describe(self) -> str:
        """Описание правила для пользователя: что разрешено и в каком каталоге."""
        where = "корень проекта" if self.cwd == "." else self.cwd
        if self.kind == _PROGRAM:
            return f"«{self.value}» с проверенными аргументами — {where}"
        return f"«{self.value}» — {where}"


def _now() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


# --------------------------------------------------------------------------- #
# Вырезание секретов из журнала
# --------------------------------------------------------------------------- #
_SECRET_ASSIGN = re.compile(
    r"\b([A-Za-z0-9_]*(?:KEY(?:_ID)?|TOKEN|SECRET|PASS(?:WORD)?|PWD|CRED\w*|AUTH\w*))="
    r"(\"[^\"]*\"|'[^']*'|\S+)",
    re.IGNORECASE,
)
_URL_CREDS = re.compile(r"(\b[a-z][a-z0-9+.-]*://)([^/\s:@]+):([^/\s@]+)@", re.IGNORECASE)
_AUTH_HEADER = re.compile(r"(authorization:\s*(?:bearer|basic|token)?\s*)[^\s'\"]+", re.IGNORECASE)
# --password, --access-token, --client-secret, --api-key…
_FLAG_SECRET = re.compile(
    r"(--[\w-]*(?:password|passwd|token|secret|api-?key)[=\s]+)(\S+)",
    re.IGNORECASE,
)
_CREDENTIAL_NAME = r"[\w-]*(?:key|token|secret|password|passwd)"
# Значение в кавычках целиком или слово.
_QUOTED_OR_WORD = r"('[^']*'|\"[^\"]*\"|[^\s'\"]+)"
# mysql -phunter2 (пароль слитно с -p); sshpass -p hunter2
_MYSQL_PASS = re.compile(rf"(\b(?:mysql\w*|mariadb\w*)\b[^|;&\n]*?\s-p){_QUOTED_OR_WORD}")
_SSHPASS = re.compile(rf"(\bsshpass\b[^|;&\n]*?\s-p\s*){_QUOTED_OR_WORD}")
# curl -u user:pass, --user=user:pass
_USER_PASS = re.compile(r"(\s(?:-u\s*|--user(?:=|\s+))['\"]?)([^\s:'\"]+):([^\s'\"]+)")
# httpie: token:=…
_HTTPIE_JSON = re.compile(rf"(\b{_CREDENTIAL_NAME}[\w-]*:=)(\S+)", re.IGNORECASE)
# заголовки и YAML: X-Api-Key: …, api_token: …
_HEADER = re.compile(rf"(\b{_CREDENTIAL_NAME}:[ \t]*)([^\s'\"=][^\s'\"]*)", re.IGNORECASE)
# JSON: "api_key": "…"
_JSON_FIELD = re.compile(rf"(\"{_CREDENTIAL_NAME}\"\s*:\s*\")[^\"]*(\")", re.IGNORECASE)
# Узнаваемые токены сервисов (GitHub, GitLab, Slack, OpenAI, AWS)
_KNOWN_TOKEN = re.compile(
    r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_\w{20,}|glpat-[\w-]{20,}"
    r"|xox[abprs]-[\w-]{10,}|sk-[\w-]{20,}|AKIA[0-9A-Z]{16})\b",
)
# Замена: оставить первую группу (имя, флаг, заголовок), значение скрыть.
_KEEP_FIRST_GROUP = r"\1***"


def redact(text: str) -> str:
    """Убрать из команды похожее на секреты.

    Вырезаются ``TOKEN=…``, пароли в URL и флагах (``--password``, ``mysql -p…``,
    ``curl -u user:pass``), заголовки авторизации и ключей, поля JSON с ключами,
    узнаваемые токены сервисов.
    """
    text = _SECRET_ASSIGN.sub(r"\1=***", text)
    text = _URL_CREDS.sub(r"\1\2:***@", text)
    text = _AUTH_HEADER.sub(_KEEP_FIRST_GROUP, text)
    text = _FLAG_SECRET.sub(_KEEP_FIRST_GROUP, text)
    text = _MYSQL_PASS.sub(_KEEP_FIRST_GROUP, text)
    text = _SSHPASS.sub(_KEEP_FIRST_GROUP, text)
    text = _USER_PASS.sub(r"\1\2:***", text)
    text = _HTTPIE_JSON.sub(_KEEP_FIRST_GROUP, text)
    text = _HEADER.sub(_KEEP_FIRST_GROUP, text)
    text = _JSON_FIELD.sub(r"\1***\2", text)
    return _KNOWN_TOKEN.sub("***", text)


# --------------------------------------------------------------------------- #
# Память
# --------------------------------------------------------------------------- #
@dataclass
class _History:
    allowed: int = 0
    rejected: int = 0

    def count(self, decision: object) -> None:
        """Учесть ответ пользователя: разрешение или отказ."""
        if decision in {"yes", "always"}:
            self.allowed += 1
        elif decision == "no":
            self.rejected += 1


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
        home = user_data_dir() if base is None else base
        directory = home / PROJECTS_DIR / project_key(root)
        # Корень в файле — путь ключа: рабочие деревья одного репозитория делят разрешения.
        memory = cls(directory, str(project_key_root(root)), remember=remember, log=log)
        approvals = directory / APPROVALS_FILE
        warning = None
        try:
            memory.rules = _read_rules_file(approvals, memory.root)
        except FileNotFoundError:
            warning = None  # файла ещё нет — правил нет, предупреждать не о чем
        except (OSError, ValueError) as exc:
            warning = f"не удалось прочитать {approvals}: {exc}"
        if log:
            for line in _read_journal(memory.journal_path):
                if line.get("event") != _DECISION_EVENT or line.get("source") != "user":
                    continue
                key = (str(line.get("cwd", ".")), str(line.get("key", "")))
                memory._history.setdefault(key, _History()).count(line.get(_DECISION_KEY))
        return memory, warning

    @property
    def approvals_path(self) -> Path | None:
        """Файл разрешений «всегда»; None — память без каталога."""
        return None if self.directory is None else self.directory / APPROVALS_FILE

    @property
    def journal_path(self) -> Path | None:
        """Файл журнала решений; None — память без каталога."""
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
            return _find_rule(self.rules, _COMMAND, verdict.cwd, verdict.command)
        matched = _match_parts(self.rules, verdict, plan=plan)
        if matched is not None:
            updated = replace(matched, hits=matched.hits + 1, last_used=_now())
            self.rules = [updated if rule.id == matched.id else rule for rule in self.rules]
            with contextlib.suppress(OSError):
                self._save(touched=[updated])
        return matched

    def offer(self, verdict: Verdict | None) -> str:
        """Что запомнит ответ «всегда» (для вопроса пользователю); "" — нечего."""
        rules = self._candidates(verdict)
        if not rules:
            return ""
        what = "; ".join(rule.describe() for rule in rules)
        where = (
            "для проекта" if self.directory is not None and self.remember else ("до конца сессии")
        )
        return f"всегда разрешать {what} (запомнится {where}; /permissions — список)"

    def approve(self, verdict: Verdict) -> list[Rule]:
        """Ответ «всегда»: сохранить правила для частей команды, которые спрашивались."""
        rules = self._candidates(verdict)
        added: list[Rule] = []
        for rule in rules:
            if _find_rule(self.rules, rule.kind, rule.cwd, rule.value) is None:
                self.rules.append(rule)
                added.append(rule)
        if added:
            self._save(added=added)
        return added

    def forget(self, index: int) -> Rule:
        """Удалить правило по номеру в списке (с 1). IndexError — нет такого."""
        if index < 1 or index > len(self.rules):
            raise IndexError(index)
        rule = self.rules.pop(index - 1)
        self._save(removed={rule.id})
        return rule

    def clear(self) -> int:
        """Удалить все правила; возвращает, сколько их было."""
        count = len(self.rules)
        removed = {rule.id for rule in self.rules}
        self.rules = []
        self._save(removed=removed)
        return count

    # ----------------------------- журнал ----------------------------- #
    def record(self, entry: dict[str, Any]) -> str:
        """Записать решение в журнал. Возвращает id записи (для :meth:`outcome`)."""
        entry_id = secrets.token_hex(6)
        decision = entry.get(_DECISION_KEY)
        if entry.get("source") == "user" and "key" in entry:
            key = (str(entry.get("cwd", ".")), str(entry["key"]))
            self._history.setdefault(key, _History()).count(decision)
            if decision == "no":
                self._rejected.add(key)
        path = self.journal_path
        if not self.log or path is None:
            return entry_id
        line = {
            "id": entry_id,
            "ts": _now(),
            "event": _DECISION_EVENT,
            "rules_version": RULES_VERSION,
            **entry,
        }
        for name in ("command", "feedback", "description", "key"):
            if isinstance(line.get(name), str):
                line[name] = redact(line[name])
        for name in ("verdict", "review"):  # причины могут цитировать команду
            nested = line.get(name)
            if isinstance(nested, dict) and isinstance(nested.get("reason"), str):
                line[name] = {**nested, "reason": redact(nested["reason"])}
        _append_journal(path, line)
        return entry_id

    def outcome(self, entry_id: str, **fields: Any) -> None:
        """Дописать в журнал, чем кончилось выполнение команды записи ``entry_id``."""
        path = self.journal_path
        if not self.log or path is None:
            return
        # Текст ошибки может цитировать команду.
        fields = {
            name: redact(field_value) if isinstance(field_value, str) else field_value
            for name, field_value in fields.items()
        }
        _append_journal(path, {"id": entry_id, "ts": _now(), "event": "outcome", **fields})

    def history(self, key: tuple[str, str]) -> tuple[int, int]:
        """(сколько раз разрешали, сколько отклоняли) команду или программу в каталоге."""
        stats = self._history.get(key)
        return (0, 0) if stats is None else (stats.allowed, stats.rejected)

    def rejected_in_session(self, key: tuple[str, str]) -> bool:
        """Пользователь отклонял эту команду (программу) в каталоге в этой сессии."""
        return key in self._rejected

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """Последние решения пользователя и модели-рецензента из журнала (новые в конце)."""
        decisions = [
            entry
            for entry in _read_journal(self.journal_path)
            if entry.get("event") == _DECISION_EVENT and entry.get("source") in {"user", "reviewer"}
        ]
        return decisions[-limit:]

    # ------------------------- внутреннее ------------------------- #
    def _candidates(self, verdict: Verdict | None) -> list[Rule]:
        rules = _proposed_rules(verdict)
        # Правила, которые match() к этой же команде не применит (непроверенная команда
        # с условием или перенаправлением, активация venv без программы, частей больше
        # предела), не предлагаются: иначе «всегда» обещает то, чего не будет.
        if verdict is None or not rules or DecisionMemory(rules=rules).match(verdict) is None:
            return []
        # Правило — команда как есть: секрет из него не вырезать, и на диск оно не идёт.
        if any(rule.kind != _PROGRAM and redact(rule.value) != rule.value for rule in rules):
            return []
        return rules

    def _save(
        self,
        *,
        added: Iterable[Rule] = (),
        touched: Iterable[Rule] = (),
        removed: Iterable[str] = (),
    ) -> None:
        """Перечитать файл, применить изменения и записать атомарно."""
        path = self.approvals_path
        if path is None or not self.remember:
            return
        try:
            current = {rule.id: rule for rule in _read_rules_file(path, self.root)}
        except FileNotFoundError:
            current = {}
        except (OSError, ValueError):
            current = {}
        for new_rule in added:
            current[new_rule.id] = new_rule
        current.update({rule.id: rule for rule in touched if rule.id in current})
        for rule_id in removed:
            current.pop(rule_id, None)
        self.rules = list(current.values())
        _write_json(
            path,
            {
                "version": FORMAT_VERSION,
                "root": self.root,
                "rules": [asdict(saved) for saved in self.rules],
            },
        )


def history_key(verdict: Verdict) -> tuple[str, str]:
    """Ключ истории: программа (если одна) или команда целиком, в рабочем каталоге."""
    sigs = verdict.signatures
    return (verdict.cwd, sigs[0] if len(sigs) == 1 else verdict.command)


# --------------------------------------------------------------------------- #
# Правила: поиск и предложение
# --------------------------------------------------------------------------- #
def _find_rule(rules: Iterable[Rule], kind: str, cwd: str, text: str) -> Rule | None:
    for rule in rules:
        if rule.kind == kind and rule.cwd == cwd and rule.value == text:
            return rule
    return None


def _part_rule(rules: list[Rule], cwd: str, part: Part) -> Rule | None:
    """Правило для одной части: программа с проверенными аргументами или точная команда."""
    rule = None
    if part.signature and part.level is CommandLevel.EXEC:
        rule = _find_rule(rules, _PROGRAM, cwd, part.signature)
    if rule is None and part.text:
        rule = _find_rule(rules, _EXACT, cwd, part.text)
    return rule


def _match_parts(rules: list[Rule], verdict: Verdict, *, plan: bool) -> Rule | None:
    """Правило для составной команды: каждая часть — чтение или совпадает с правилом."""
    found: list[Rule] = []
    pending_preludes = False
    for part in verdict.parts:
        if part.level is CommandLevel.READ:
            continue
        if part.level is CommandLevel.DANGER or (plan and part.level is not CommandLevel.EXEC):
            return None
        if part.prelude:
            pending_preludes = True
            continue
        rule = _part_rule(rules, verdict.cwd, part)
        if rule is None:
            return None
        found.append(rule)
    # Правило вида program находит только поиск по сигнатуре программы.
    if pending_preludes and not any(found_rule.kind == _PROGRAM for found_rule in found):
        return None  # активация venv — только вместе с разрешённой программой
    return found[0] if found else None


def _part_specs(verdict: Verdict) -> list[_RuleSpec]:
    """Что предложить запомнить по частям команды (или её целиком, если по частям нельзя)."""
    whole: list[_RuleSpec] = [(_COMMAND, verdict.command, verdict.level)]
    has_program = any(
        step.signature and step.level is CommandLevel.EXEC and not step.prelude
        for step in verdict.parts
    )
    # Чтение не спрашивается, опасное не разрешается; активация venv идёт с программой.
    asked = [
        part
        for part in verdict.parts
        if part.level is not CommandLevel.READ
        and part.level is not CommandLevel.DANGER
        and not (part.prelude and has_program)
    ]
    # Одинаковые (вид, текст) — одно предложение, первое по порядку частей.
    specs: dict[tuple[str, str], _RuleSpec] = {}
    for part in asked:
        spec: _RuleSpec
        if part.signature and part.level is CommandLevel.EXEC and not part.prelude:
            spec = (_PROGRAM, part.signature, part.level)
        elif part.text:
            spec = (_EXACT, part.text, part.level)
        else:
            return whole
        specs.setdefault(spec[:2], spec)
    if len(specs) > MAX_RULES_PER_ANSWER:
        return whole
    return list(specs.values())


def _proposed_rules(verdict: Verdict | None) -> list[Rule]:
    if verdict is None or verdict.level is CommandLevel.DANGER:
        return []
    now = _now()
    specs: list[_RuleSpec] = []
    if verdict.composable and verdict.parts:
        specs = _part_specs(verdict)
    elif verdict.command:
        specs = [(_COMMAND, verdict.command, verdict.level)]
    return [
        Rule(
            secrets.token_hex(4),
            verdict.cwd,
            kind,
            text,
            "UNKNOWN" if level is None else level.name,
            now,
        )
        for kind, text, level in specs
    ]


# --------------------------------------------------------------------------- #
# Файлы
# --------------------------------------------------------------------------- #
def _read_rules_file(path: Path, root: str) -> list[Rule]:
    """Правила из ``approvals.json``; ValueError — чужой или непонятный файл."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != FORMAT_VERSION:
        raise ValueError("неподдерживаемая версия формата")
    stored = payload.get("root")
    # Корень рабочего дерева, а не основного репозитория (так писали прежние версии) —
    # тот же проект.
    if stored not in {None, "", root} and not (
        isinstance(stored, str) and project_key(Path(stored)) == project_key(Path(root))
    ):
        raise ValueError(f"файл другого проекта ({stored})")
    rules = []
    for raw_rule in payload.get("rules", []):
        if not isinstance(raw_rule, dict):
            continue
        try:
            rule = Rule(**{
                name: raw_rule[name] for name in Rule.__dataclass_fields__ if name in raw_rule
            })
        except TypeError:
            continue
        if rule.kind in _RULE_KINDS:
            rules.append(rule)
    return rules


def _append_journal(path: Path, line: dict[str, Any]) -> None:
    # Журнал — не повод прерывать работу: ошибки файловой системы пропускаются.
    with contextlib.suppress(OSError):
        path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            if path.stat().st_size > MAX_JOURNAL_BYTES:
                Path(path).replace(path.with_suffix(".1.jsonl"))
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, _PRIVATE_FILE_MODE)
        with os.fdopen(fd, "a", encoding="utf-8") as journal:
            journal.write(f"{json.dumps(line, ensure_ascii=False)}\n")


def _read_journal(path: Path | None) -> Iterator[dict[str, Any]]:
    if path is None:
        return
    try:
        with Path(path).open(encoding="utf-8") as journal:
            for raw in journal:
                entry = None  # битая строка — пропустить
                with contextlib.suppress(ValueError):
                    entry = json.loads(raw)
                if isinstance(entry, dict):
                    yield entry
    except OSError:
        return


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=_PRIVATE_DIR_MODE)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as target:
            json.dump(payload, target, ensure_ascii=False, indent=2)
        Path(temp_name).chmod(_PRIVATE_FILE_MODE)
        Path(temp_name).replace(path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
