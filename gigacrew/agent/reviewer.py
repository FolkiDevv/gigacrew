"""Модель-рецензент изменяющих команд (режим «авто»).

Команду, которую локальный классификатор (:mod:`gigacrew.safety`) не признал «только
чтение» и не счёл опасной, в режиме «авто» сначала смотрит модель: она разрешает
выполнить команду без вопроса или не разрешает — тогда команда не выполняется, а агент
получает причину и ищет другой путь. Запасной путь — вопрос пользователю: если агент
повторил ту же команду, если отказы идут подряд (:data:`MAX_DENIALS_IN_ROW`) или их много
за ход (:data:`MAX_DENIALS_PER_TURN`). Запретить команду насовсем рецензент не может.

Рецензент «слеп к рассуждениям»: видит запросы пользователя, одобренный пользователем
план, саму команду и факты статического разбора, но не ответы агента, не его пояснение к
команде и не вывод инструментов — содержимое файлов и страниц не может его уговорить.

Сбой рецензента — не повод звать пользователя сразу: быстрый сбой повторяется один раз
(:meth:`Reviewer.ask`), сбои подряд ставят рецензента на паузу
(:data:`REVIEWER_COOLDOWN`), а не отключают до конца сессии. Разрешение запоминается на
сессию по тексту запроса (:meth:`Reviewer.cached`): тот же запрос второй раз не отправляется.

* :func:`review_command` — один запрос к модели;
* :class:`Reviewer` — состояние на сессию: какая модель, включён ли, счётчики сбоев и
  отказов, разрешённые запросы;
* :func:`check_reviewer` — проверка модели из ``GIGACREW_REVIEWER_MODEL`` при старте.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

from gigacrew.agent.compaction import clip_middle
from gigacrew.agent.events import Review
from gigacrew.agent.prompts import REVIEWER_PROMPT
from gigacrew.llm.base import LLMError, LLMProvider
from gigacrew.llm.types import Message, Usage
from gigacrew.project.workspace import DATA_DIR_NAME, TEMP_DIR_NAME, Workspace
from gigacrew.safety import CommandLevel, Part, Verdict

REVIEWER_ENV = "GIGACREW_REVIEWER_MODEL"
# Команду длиннее рецензент не смотрит: её читает пользователь.
MAX_REVIEW_COMMAND = 4_000
# Сколько последних запросов пользователя видит рецензент и сколько символов каждого.
MAX_REQUESTS = 3
REQUEST_CLIP = 1_500
PART_CLIP = 200
# Сколько символов одобренного пользователем плана видит рецензент.
PLAN_CLIP = 3_000
# Сколько символов чернового скрипта из .gigacode/tmp/ видит рецензент и сколько скриптов.
SCRIPT_CLIP = 3_000
MAX_SCRIPTS = 2
# Столько сбоев подряд — и рецензент отключается на REVIEWER_COOLDOWN секунд.
MAX_FAILURES = 3
# Пауза после сбоев подряд, с: потом одна проба, её сбой — снова пауза.
REVIEWER_COOLDOWN = 300
# Сбой повторяется один раз, если ответ не разобран или ошибка пришла быстрее, с:
# ждать второй тайм-аут пользователю дороже, чем ответить на вопрос.
RETRY_WITHIN = 10
# Сколько разрешённых запросов помнит рецензент за сессию.
MAX_CACHED_REVIEWS = 256
# После стольких отказов подряд (без разрешений между ними) решает пользователь.
MAX_DENIALS_IN_ROW = 3
# После стольких отказов за ход пользователя решает пользователь — до конца хода.
MAX_DENIALS_PER_TURN = 10
DISABLED_NOTE = (
    "модель-рецензент отключена: в режиме «авто» больше команд будут требовать ручного "
    "подтверждения"
)
_SECONDS_IN_MINUTE = 60
# Сколько символов причины из ответа рецензента сохраняется.
_REASON_CLIP = 300
# Сколько символов ответа рецензента цитирует ошибка разбора.
_SNIPPET_CLIP = 120
# Сколько символов ошибки проверки модели попадает в предупреждение.
_ERROR_CLIP = 200


class ReviewError(LLMError):
    """Рецензент не дал пригодного ответа.

    ``retry`` — стоит ли спросить ещё раз: противоречивый ответ повтор не исправит.
    """

    def __init__(self, message: str, *, retry: bool = True):
        """Ошибка с текстом ``message``."""
        super().__init__(message)
        self.retry = retry


# ------------------------------- запрос ------------------------------- #
def recent_requests(messages: Sequence[Message], limit: int = MAX_REQUESTS) -> list[str]:
    """Последние ``limit`` запросов пользователя (без служебных сообщений harness'а)."""
    requests = [
        message.content for message in messages if message.is_request and message.content.strip()
    ]
    return requests[-limit:]


def _level(level: CommandLevel | None) -> str:
    return "UNVERIFIED" if level is None else level.name


def _fence(text: str) -> str:
    """Ограждение длиннее любой цепочки обратных кавычек в тексте: команда не закроет его."""
    longest = max((len(ticks) for ticks in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def _one_line(text: str) -> str:
    """Причина разбора одной строкой: она может цитировать пути и имена из команды."""
    return " ".join(text.split())


def _part_line(part: Part) -> str:
    """Строка о части команды, которая не только читает.

    Текст части — из команды (с переводами строк в кавычках) и стоит вне ограждения:
    строкой JSON, чтобы не начать в запросе свой раздел («# The user's requests»).
    """
    text = json.dumps(clip_middle(part.text or "?", PART_CLIP), ensure_ascii=False)
    return f"- {text}: {_level(part.level)} — {_one_line(part.reason)}"


def review_request(
    command: str,
    verdict: Verdict,
    requests: Sequence[str],
    plan: str = "",
) -> str:
    """Текст запроса рецензенту: запросы пользователя, команда, факты разбора.

    ``plan`` — план, который пользователь одобрил ("" — не одобрял): на что он согласился.
    Текст плана писала модель, поэтому он стоит в ограждении, как команда.

    Код черновых скриптов из ``.gigacode/tmp/``, которые команда запускает, — тоже: их
    пишет модель, как и heredoc. Подключённые пользователем каталоги
    (``verdict.extra_dirs``) для рецензента — часть проекта.
    """
    lines = ["# The user's requests (oldest first)"]
    if requests:
        lines += [
            f"{number}. {clip_middle(request.strip(), REQUEST_CLIP)}"
            for number, request in enumerate(requests, 1)
        ]
    else:
        lines.append("(none)")
    if plan.strip():
        shown = clip_middle(plan.strip(), PLAN_CLIP)
        plan_fence = _fence(shown)
        lines += ["", "# Plan the user approved", plan_fence, shown, plan_fence]
    fence = _fence(command)
    overall_level = _level(verdict.level)
    overall_reason = _one_line(verdict.reason) or "no details"
    lines += [
        "",
        "# Command",
        f"Working directory (relative to the project root): {verdict.cwd}",
        *_extra_lines(verdict.extra_dirs),
        fence,
        command,
        fence,
        "",
        "# Static analysis",
        f"Overall: {overall_level} — {overall_reason}",
    ]
    parts = [part for part in verdict.parts if part.level is not CommandLevel.READ]
    if parts:
        lines.append("Parts that are not read-only:")
        lines += [_part_line(part) for part in parts]
    lines += _scratch_scripts(verdict)
    if verdict.writes:
        lines.append("Redirects into files:")
        lines += [
            f"- {json.dumps(clip_middle(_one_line(write), PART_CLIP), ensure_ascii=False)}"
            for write in verdict.writes
        ]
    lines.append(
        "Levels: EXEC runs project code, tools or installed libraries (what that code does "
        "inside was not analysed), WRITE changes files or the local environment, "
        "UNVERIFIED — the analyzer could not check the command. «временная папка модели» "
        "marks the scratch folder .gigacode/tmp/.",
    )
    return "\n".join(lines)


def _scratch_scripts(verdict: Verdict) -> list[str]:
    """Код черновых скриптов (``.gigacode/tmp/*.py``), которые запускает команда."""
    found: dict[Path, Path] = {}  # скрипт → корень проекта
    for part in verdict.parts:
        if part.level is CommandLevel.READ:
            continue
        for path in map(Path, part.inputs):
            root = _project_of(path)
            if path.suffix == ".py" and root is not None:
                found.setdefault(path, root)
    lines: list[str] = []
    for script, root in list(found.items())[:MAX_SCRIPTS]:
        try:
            code = script.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        shown = clip_middle(code, SCRIPT_CLIP)
        fence = _fence(shown)
        name = json.dumps(script.relative_to(root).as_posix(), ensure_ascii=False)
        lines += [f"Script {name} (scratch file written by the assistant):", fence, shown, fence]
    return lines


def _extra_lines(extra_dirs: Sequence[str]) -> list[str]:
    """Подключённые каталоги: рецензент считает их частью проекта, а не «снаружи»."""
    if not extra_dirs:
        return []
    listed = ", ".join(json.dumps(folder, ensure_ascii=False) for folder in extra_dirs)
    note = (
        f"Additional directories the user attached to the project: {listed}. Treat them "
        "as part of the project, not as outside it."
    )
    return [note]


def _project_of(path: Path) -> Path | None:
    """Корень проекта, во временной папке которого лежит ``path`` (входы оценки — абсолютные)."""
    segments = path.parts
    for index in range(len(segments) - 2):
        root = Path(*segments[:index])
        if segments[index : index + 2] == (DATA_DIR_NAME, TEMP_DIR_NAME):
            return root if Workspace(root).is_temp_path(path) else None
    return None


def _decision(candidate: object) -> tuple[bool, str] | None:
    """Решение из одного JSON-значения ответа: ``(allow, причина)`` или None — не решение."""
    if not isinstance(candidate, dict) or "allow" not in candidate:
        return None
    allow = candidate["allow"]
    if isinstance(allow, str) and allow.strip().lower() in {"true", "false"}:
        allow = allow.strip().lower() == "true"
    if not isinstance(allow, bool):
        return None
    reason = " ".join(str(candidate.get("reason") or "").split())
    return allow, reason[:_REASON_CLIP]


def parse_review(text: str) -> tuple[bool, str]:
    """Решение из ответа модели: JSON-объект с полем ``allow``.

    Модели нередко оборачивают JSON в блок кода или пишут текст вокруг — это не ошибка.
    Объекта нет — :class:`ReviewError`. Объектов с разными ``allow`` несколько (рецензент
    процитировал объект из команды и ответил другим) — тоже, без повтора: какой из них
    решение, не понять, а ошибка — в осторожную сторону. Согласных объектов несколько —
    причина последнего.
    """
    decoder = json.JSONDecoder()
    found: list[tuple[bool, str]] = []
    for match in re.finditer(r"\{", text):
        try:
            candidate, _ = decoder.raw_decode(text, match.start())
        except ValueError:
            continue
        decision = _decision(candidate)
        if decision is not None:
            found.append(decision)
    snippet = " ".join(text.split())[:_SNIPPET_CLIP]
    if not found:
        shown = snippet or "(пусто)"
        raise ReviewError(f"ответ рецензента не разобран: {shown}")
    if len({allow for allow, _ in found}) > 1:
        raise ReviewError(f"противоречивый ответ рецензента: {snippet}", retry=False)
    return found[-1]


def review_command(
    provider: LLMProvider,
    request_text: str,
    *,
    model: str,
    timeout: float | None = None,
    on_usage: Callable[[Usage], None] | None = None,
) -> Review:
    """Спросить модель ``model``, можно ли выполнить команду без вопроса.

    ``request_text`` — запрос о команде (:func:`review_request`). ``timeout`` — сколько
    ждать ответа, с (None — ожидание провайдера). Ошибки обращения — :class:`LLMError`,
    непригодный ответ — :class:`ReviewError`.
    """
    request = [
        Message(role="system", content=REVIEWER_PROMPT),
        Message(role="user", content=request_text),
    ]
    turn = provider.complete(
        request,
        tools=None,
        model=model,
        temperature=float(0),
        timeout=timeout,
    )
    if on_usage is not None:
        on_usage(turn.usage)
    if turn.finish_reason != "stop":
        raise ReviewError(f"ответ рецензента не закончен (finish_reason={turn.finish_reason})")
    allowed, reason = parse_review(turn.message.content)
    return Review(model, allowed, reason)


# ------------------------------ состояние ------------------------------ #
# Состояние сессии в одном месте: сбои, пауза, отказы и разрешённые запросы.
class Reviewer:  # noqa: WPS214
    """Модель-рецензент сессии; общая у агента и его суб-агентов.

    ``model`` — явно заданная модель (``GIGACREW_REVIEWER_MODEL``); None — модель чата,
    какая выбрана в момент проверки. ``disabled`` — почему отключён ("" — работает):
    заданная при создании причина — до конца сессии, сбои подряд — на
    :data:`REVIEWER_COOLDOWN` секунд.

    Отказы считаются за ход пользователя (:meth:`new_turn`): вызов, который уже получил
    отказ, отказы подряд и отказы за ход решают, вернуть ли следующий отказ агенту или
    передать команду пользователю.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        disabled: str = "",
        clock: Callable[[], float] = time.monotonic,
    ):
        """Рецензент с моделью ``model``; ``disabled`` — причина отключения ("" — работает)."""
        self.model = model
        self._off = disabled  # отключён до конца сессии
        self._clock = clock
        self._paused = ""  # почему на паузе
        self._paused_until = float(0)
        self._failures = 0
        self._denials_in_row = 0
        self._denials = 0
        self._denied: dict[str, str] = {}  # ключ вызова → причина отказа
        self._allowed: dict[tuple[str, str], Review] = {}  # (модель, запрос) → разрешение

    @property
    def disabled(self) -> str:
        """Почему рецензент отключён ("" — работает); пауза после сбоев кончается сама."""
        if self._off:
            return self._off
        if self._paused and self._clock() < self._paused_until:
            return self._paused
        return ""

    @property
    def enabled(self) -> bool:
        """Рецензент работает (не отключён)."""
        return not self.disabled

    @property
    def warning(self) -> str:
        """Предупреждение об отключении ("" — рецензент работает)."""
        return f"{self.disabled} — {DISABLED_NOTE}" if self.disabled else ""

    def model_for(self, chat_model: str) -> str:
        """Модель для проверки: заданная явно или модель чата ``chat_model``."""
        return self.model or chat_model

    def describe(self, chat_model: str) -> str:
        """Кто проверяет команды (для ``/mode``)."""
        if self.disabled:
            return f"рецензент отключён: {self.disabled}"
        if self.model:
            return f"команды проверяет {self.model} ({REVIEWER_ENV})"
        return f"команды проверяет модель чата ({chat_model})"

    def succeeded(self) -> None:
        """Учесть успешную проверку: серия сбоев прервалась."""
        self._failures = 0

    def failed(self, error: str) -> bool:
        """Учесть сбой; True — рецензент только что отключён (сбои подряд).

        Отключение — пауза на :data:`REVIEWER_COOLDOWN` секунд. После неё хватит одного
        сбоя, чтобы уйти на паузу снова; успех (:meth:`succeeded`) возвращает полный счёт.
        """
        self._failures += 1
        if self._failures < MAX_FAILURES or self.disabled:
            return False
        minutes = REVIEWER_COOLDOWN // _SECONDS_IN_MINUTE
        self._paused = (
            f"{MAX_FAILURES} сбоя рецензента подряд (последний: {error}), пауза {minutes} мин"
        )
        self._paused_until = self._clock() + REVIEWER_COOLDOWN
        self._failures = MAX_FAILURES - 1
        return True

    def ask(self, question: Callable[[], Review]) -> Review:
        """Ответ рецензента; сбой повторяется один раз, если повтор обойдётся недорого.

        Повторяется неразобранный или оборванный ответ (модель ответила, но не по форме) и
        ошибка, пришедшая быстрее :data:`RETRY_WITHIN` секунд (обрыв связи, перегрузка).
        Противоречивый ответ и долгий сбой (тайм-аут) не повторяются.
        """
        started = self._clock()
        try:
            return question()
        except ReviewError as error:
            if not error.retry:
                raise
        except Exception:
            if self._clock() - started > RETRY_WITHIN:
                raise
        return question()

    def cached(self, model: str, request_text: str) -> Review | None:
        """Разрешение, уже выданное в этой сессии моделью ``model`` на тот же запрос.

        Ключ — модель и весь текст запроса: всё, что рецензент видел.
        """
        return self._allowed.get((model, request_text))

    def remember(self, request_text: str, review: Review) -> None:
        """Запомнить разрешение на запрос до конца сессии (отказы не хранятся)."""
        if not review.allowed or review.failed:
            return
        if len(self._allowed) >= MAX_CACHED_REVIEWS:
            self._allowed.pop(next(iter(self._allowed)))
        self._allowed[review.model, request_text] = replace(review, cached=True)

    def allowed(self) -> None:
        """Учесть разрешение: серия отказов подряд прервалась."""
        self._denials_in_row = 0

    def denied(self, key: str, reason: str) -> bool:
        """Учесть отказ вызову ``key``; True — пора спросить пользователя, а не агента.

        Пользователь решает после :data:`MAX_DENIALS_IN_ROW` отказов подряд — серия
        начинается заново — и после :data:`MAX_DENIALS_PER_TURN` отказов за ход, до его
        конца.
        """
        self._denied[key] = reason
        self._denials += 1
        self._denials_in_row += 1
        if self._denials_in_row >= MAX_DENIALS_IN_ROW or self._denials >= MAX_DENIALS_PER_TURN:
            self._denials_in_row = 0
            return True
        return False

    def denied_reason(self, key: str) -> str | None:
        """Причина отказа вызову ``key`` в этом ходе; None — отказа не было."""
        return self._denied.get(key)

    def new_turn(self) -> None:
        """Новый ход пользователя: отказы прошлых ходов не учитываются."""
        self._denials_in_row = 0
        self._denials = 0
        self._denied.clear()


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
    except Exception as error:  # сеть/авторизация: проверим делом
        detail = " ".join(str(error).split())[:_ERROR_CLIP]
        return Reviewer(model), f"{unchecked}: {detail}"
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
