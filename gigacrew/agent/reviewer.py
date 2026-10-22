"""Модель-рецензент изменяющих команд (режим «авто»).

Команду, которую локальный классификатор (:mod:`gigacrew.safety`) не признал «только
чтение» и не счёл опасной, в режиме «авто» сначала смотрит модель: она разрешает
выполнить команду без вопроса или передаёт решение пользователю. Отказать агенту сама
она не может — ошибка в осторожную сторону стоит одного вопроса.

Рецензент «слеп к рассуждениям»: видит запросы пользователя, саму команду и факты
статического разбора, но не ответы агента, не его пояснение к команде и не вывод
инструментов — содержимое файлов и страниц не может его уговорить.

* :func:`review_command` — один запрос к модели;
* :class:`Reviewer` — состояние на сессию: какая модель, включён ли, счётчик сбоев;
* :func:`check_reviewer` — проверка модели из ``GIGACREW_REVIEWER_MODEL`` при старте.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence

from gigacrew.agent.compaction import clip_middle
from gigacrew.agent.events import Review
from gigacrew.agent.prompts import REVIEWER_PROMPT
from gigacrew.llm.base import LLMError, LLMProvider
from gigacrew.llm.types import Message, Usage
from gigacrew.safety import CommandLevel, Verdict

REVIEWER_ENV = "GIGACREW_REVIEWER_MODEL"
# Команду длиннее рецензент не смотрит: её читает пользователь.
MAX_REVIEW_COMMAND = 4_000
# Сколько последних запросов пользователя видит рецензент и сколько символов каждого.
MAX_REQUESTS = 3
REQUEST_CLIP = 1_500
PART_CLIP = 200
# Столько сбоев подряд — и рецензент отключается до конца сессии.
MAX_FAILURES = 3
DISABLED_NOTE = (
    "модель-рецензент отключена: в режиме «авто» больше команд будут требовать ручного "
    "подтверждения"
)


class ReviewError(LLMError):
    """Рецензент не дал пригодного ответа."""


# ------------------------------- запрос ------------------------------- #
def recent_requests(messages: Sequence[Message], limit: int = MAX_REQUESTS) -> list[str]:
    """Последние ``limit`` запросов пользователя (без служебных сообщений harness'а)."""
    requests = [m.content for m in messages if m.is_request and m.content.strip()]
    return requests[-limit:]


def _level(level: CommandLevel | None) -> str:
    return level.name if level is not None else "UNVERIFIED"


def _fence(text: str) -> str:
    """Ограждение длиннее любой цепочки обратных кавычек в тексте: команда не закроет его."""
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def _one_line(text: str) -> str:
    """Причина разбора одной строкой: она может цитировать пути и имена из команды."""
    return " ".join(text.split())


def review_request(command: str, verdict: Verdict, requests: Sequence[str]) -> str:
    """Текст запроса рецензенту: запросы пользователя, команда, факты разбора."""
    lines = ["# The user's requests (oldest first)"]
    if requests:
        lines += [f"{i}. {clip_middle(r.strip(), REQUEST_CLIP)}" for i, r in enumerate(requests, 1)]
    else:
        lines.append("(none)")
    fence = _fence(command)
    lines += [
        "",
        "# Command",
        f"Working directory (relative to the project root): {verdict.cwd}",
        fence,
        command,
        fence,
        "",
        "# Static analysis",
        f"Overall: {_level(verdict.level)} — {_one_line(verdict.reason) or 'no details'}",
    ]
    parts = [p for p in verdict.parts if p.level is not CommandLevel.READ]
    if parts:
        # Текст части — из команды (с переводами строк в кавычках) и стоит вне ограждения:
        # строкой JSON, чтобы не начать в запросе свой раздел («# The user's requests»).
        lines.append("Parts that are not read-only:")
        lines += [
            f"- {json.dumps(clip_middle(p.text or '?', PART_CLIP), ensure_ascii=False)}: "
            f"{_level(p.level)} — {_one_line(p.reason)}"
            for p in parts
        ]
    lines.append(
        "Levels: EXEC runs project code or tools, WRITE changes files or the local "
        "environment, UNVERIFIED — the analyzer could not check the command.",
    )
    return "\n".join(lines)


def parse_review(text: str) -> tuple[bool, str]:
    """Решение из ответа модели: JSON-объект с полем ``allow``.

    Модели нередко оборачивают JSON в блок кода или пишут текст вокруг — это не ошибка.
    Объекта нет — :class:`ReviewError`. Объектов с разными ``allow`` несколько (рецензент
    процитировал объект из команды и ответил другим) — тоже: какой из них решение, не
    понять, а ошибка — в осторожную сторону. Согласных объектов несколько — причина
    последнего.
    """
    decoder = json.JSONDecoder()
    found: list[tuple[bool, str]] = []
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text, match.start())
        except ValueError:
            continue
        if not isinstance(obj, dict) or "allow" not in obj:
            continue
        allow = obj["allow"]
        if isinstance(allow, str) and allow.strip().lower() in {"true", "false"}:
            allow = allow.strip().lower() == "true"
        if not isinstance(allow, bool):
            continue
        reason = " ".join(str(obj.get("reason") or "").split())
        found.append((allow, reason[:300]))
    snippet = " ".join(text.split())[:120]
    if not found:
        raise ReviewError(f"ответ рецензента не разобран: {snippet or '(пусто)'}")
    if len({allow for allow, _ in found}) > 1:
        raise ReviewError(f"противоречивый ответ рецензента: {snippet}")
    return found[-1]


def review_command(
    provider: LLMProvider,
    *,
    command: str,
    verdict: Verdict,
    requests: Sequence[str],
    model: str,
    on_usage: Callable[[Usage], None] | None = None,
) -> Review:
    """Спросить модель ``model``, можно ли выполнить команду без вопроса.

    Ошибки обращения — :class:`LLMError`, непригодный ответ — :class:`ReviewError`.
    """
    request = [
        Message(role="system", content=REVIEWER_PROMPT),
        Message(role="user", content=review_request(command, verdict, requests)),
    ]
    turn = provider.complete(request, tools=None, model=model, temperature=0.0)
    if on_usage is not None:
        on_usage(turn.usage)
    if turn.finish_reason != "stop":
        raise ReviewError(f"ответ рецензента не закончен (finish_reason={turn.finish_reason})")
    allowed, reason = parse_review(turn.message.content)
    return Review(model, allowed, reason)


# ------------------------------ состояние ------------------------------ #
class Reviewer:
    """Модель-рецензент сессии; общая у агента и его суб-агентов.

    ``model`` — явно заданная модель (``GIGACREW_REVIEWER_MODEL``); None — модель чата,
    какая выбрана в момент проверки. ``disabled`` — почему отключён ("" — работает).
    """

    def __init__(self, model: str | None = None, *, disabled: str = ""):
        self.model = model
        self.disabled = disabled
        self._failures = 0

    @property
    def enabled(self) -> bool:
        return not self.disabled

    @property
    def warning(self) -> str:
        """Предупреждение об отключении ("" — рецензент работает)."""
        return f"{self.disabled} — {DISABLED_NOTE}" if self.disabled else ""

    def model_for(self, chat_model: str) -> str:
        return self.model or chat_model

    def describe(self, chat_model: str) -> str:
        """Кто проверяет команды (для ``/mode``)."""
        if self.disabled:
            return f"рецензент отключён: {self.disabled}"
        if self.model:
            return f"команды проверяет {self.model} ({REVIEWER_ENV})"
        return f"команды проверяет модель чата ({chat_model})"

    def succeeded(self) -> None:
        self._failures = 0

    def failed(self, error: str) -> bool:
        """Учесть сбой; True — рецензент только что отключён (сбои подряд)."""
        self._failures += 1
        if self._failures < MAX_FAILURES or self.disabled:
            return False
        self.disabled = f"{MAX_FAILURES} сбоя рецензента подряд (последний: {error})"
        return True


def check_reviewer(provider: LLMProvider, model: str | None) -> tuple[Reviewer, str | None]:
    """Рецензент для сессии и предупреждение (None — всё в порядке).

    Явно заданную модель ищем среди чат-моделей API (без учёта регистра, как ``/model``);
    модели нет — рецензент отключён. Список не получен — проверить нельзя: рецензент
    работает, а несуществующую модель отключат сбои подряд. Модель не задана — модель
    чата, проверять нечего (и сеть не нужна).
    """
    if model is None:
        return Reviewer(), None
    unchecked = f"не удалось проверить модель-рецензент «{model}» ({REVIEWER_ENV})"
    try:
        known = provider.list_models()
    except Exception as e:  # сеть/авторизация: проверим делом
        return Reviewer(model), f"{unchecked}: {' '.join(str(e).split())[:200]}"
    if not known:
        return Reviewer(model), f"{unchecked}: список моделей пуст"
    match = (
        model
        if model in known
        else next((name for name in known if name.lower() == model.lower()), None)
    )
    if match is None:
        reviewer = Reviewer(model, disabled=f"модели «{model}» ({REVIEWER_ENV}) нет")
        return reviewer, reviewer.warning
    return Reviewer(match), None
