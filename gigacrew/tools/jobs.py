"""Фоновые задачи: работа, которая идёт, пока основной агент делает другое.

Фоновая задача (:class:`Job`) — команда ``run_shell`` с ``run_in_background`` или
суб-агент ``task`` с ``run_in_background``. Её запускает инструмент через
:class:`BackgroundJobs` из ``ToolContext.background``; вызов сразу возвращает id задачи, а
итог (:class:`JobReport`) агентный цикл доставит модели, когда задача завершится.

Гарантии:

* задачи одного вида (``Job.pool``) работают не больше заданного числа одновременно
  (:class:`PoolLimits`): лишние ждут в очереди ограниченной длины, а сверх неё запуск
  отклоняется. Суб-агенты по умолчанию — 2 одновременно и до 4 в очереди, команды — 4 и
  без очереди (модель ждёт, что команда запустится сразу);
* задача не висит бесконечно: у каждой — жёсткий предел времени, он отсчитывается с
  запуска (не с постановки в очередь). Команду по нему убивает её надзиратель
  (:class:`~gigacrew.tools.process.BackgroundProcess`), суб-агент подводит итог по своему
  пределу времени; а если задача всё же не завершилась к пределу с запасом
  (:data:`HARD_STOP_GRACE`), :class:`BackgroundJobs` останавливает её сам: итог «превышен
  лимит времени» записывается без её участия. Очередь движется только завершением
  работающих задач — значит, и ожидание в ней ограничено;
* итог не теряется: завершившаяся задача ждёт доставки (:meth:`BackgroundJobs.take_finished`),
  а подписчики (:meth:`BackgroundJobs.subscribe`) узнают о ней сразу, чтобы разбудить
  агента;
* при выходе из gigacrew задачи останавливаются (команды — вместе с потомками).
"""

from __future__ import annotations

import atexit
import contextlib
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import ClassVar

from gigacrew.errors import ToolError

__all__ = [
    "DEFAULT_LIMITS",
    "HARD_STOP_GRACE",
    "SHELL_POOL",
    "STATE_LABELS",
    "STOP_GRACE",
    "SUBAGENT_POOL",
    "BackgroundJobs",
    "Job",
    "JobCancelledError",
    "JobInfo",
    "JobReport",
    "JobState",
    "PoolLimits",
    "job_heading",
    "job_info",
]

# Виды задач, у каждого — свои пределы.
SHELL_POOL = "shell"
SUBAGENT_POOL = "subagent"
# Запас сверх предела задачи, после которого её останавливает BackgroundJobs: суб-агенту —
# на итог после его собственного предела времени, команде — на остановку надзирателем.
HARD_STOP_GRACE = 180
# Сколько ждать, пока остановленная задача завершится сама (команда — после SIGKILL,
# суб-агент — до ближайшей проверки отмены); дальше итог записывается без неё.
STOP_GRACE = 2.0


class JobState(StrEnum):
    """Состояние фоновой задачи."""

    QUEUED = "queued"  # ждёт свободного места своего вида
    RUNNING = "running"
    DONE = "done"  # завершилась сама: команда — с любым кодом, суб-агент — с отчётом
    FAILED = "failed"  # сбой: ошибка модели, внутренняя ошибка, суб-агент без отчёта
    STOPPED = "stopped"  # остановлена агентом, пользователем, сменой чата или выходом
    TIMED_OUT = "timed_out"  # превысила предел времени и остановлена


# Состояния для пользователя.
STATE_LABELS = MappingProxyType({
    JobState.QUEUED: "в очереди",
    JobState.RUNNING: "работает",
    JobState.DONE: "завершена",
    JobState.FAILED: "сбой",
    JobState.STOPPED: "остановлена",
    JobState.TIMED_OUT: "превышен лимит времени",
})
# Состояния для модели (заметки модели — на английском, как и промпты).
_MODEL_STATES = MappingProxyType({
    JobState.QUEUED: "queued, starts when a running job of its kind finishes",
    JobState.RUNNING: "running",
    JobState.DONE: "finished",
    JobState.FAILED: "failed",
    JobState.STOPPED: "stopped",
    JobState.TIMED_OUT: "stopped by its time limit",
})


@dataclass(frozen=True)
class PoolLimits:
    """Пределы задач одного вида: сколько работают одновременно и сколько ждут в очереди."""

    running: int
    queued: int = 0


# Пределы по умолчанию. Команды — без очереди: модель ждёт, что команда запустится сразу.
DEFAULT_LIMITS: Mapping[str, PoolLimits] = MappingProxyType({
    SHELL_POOL: PoolLimits(running=4),
    SUBAGENT_POOL: PoolLimits(running=2, queued=4),
})
_FALLBACK_LIMITS = PoolLimits(running=1)


class JobCancelledError(Exception):
    """Фоновая задача отменена: суб-агент прекращает работу на ближайшей проверке."""


@dataclass(frozen=True)
class JobReport:
    """Итог фоновой задачи: ``content`` — для модели, ``summary`` — строка пользователю."""

    state: JobState
    content: str
    summary: str = ""


@dataclass(frozen=True)
class JobInfo:
    """Снимок фоновой задачи для интерфейса.

    ``kind`` — ``shell`` или тип суб-агента; ``activity`` — чем занята сейчас (последняя
    строка вывода, инструмент суб-агента); ``summary`` — итог завершившейся.
    """

    id: str
    kind: str
    title: str
    state: JobState
    elapsed: float
    activity: str = ""
    summary: str = ""


class Job(ABC):
    """Фоновая задача: запуск, отмена и итог — у подклассов, жизненный цикл — здесь.

    Итог записывается один раз (:meth:`finish`): первым — завершением самой работы или
    остановкой (:meth:`request_stop`, затем итог «остановлена»), что раньше.
    ``pool`` — вид задачи для пределов :class:`BackgroundJobs`.
    """

    pool: ClassVar[str]

    def __init__(self, kind: str, title: str, *, timeout: float) -> None:
        """Задача вида ``kind`` с описанием ``title`` и пределом ``timeout`` секунд."""
        self.id = ""  # назначает BackgroundJobs.start
        self.kind = kind
        self.title = title
        self.timeout = timeout
        # Итог не будит агента: задачу остановил пользователь или выход — он доставляется
        # со следующим запросом к модели.
        self.quiet = False
        # Запрошенная остановка: состояние итога и причина (None — не запрашивалась).
        self.stop_request: tuple[JobState, str] | None = None
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._started: float | None = None
        self._ended: float | None = None
        self._report: JobReport | None = None
        self._on_finish: Callable[[Job], None] | None = None
        self._announced = False  # об итоге извещён тот, кто запускал (on_finish)

    # ------------------------------------------------------------------ #
    @abstractmethod
    def start(self) -> None:
        """Запустить работу (не ждёт её завершения)."""

    @abstractmethod
    def cancel(self) -> None:
        """Прекратить работу: убить команду, отменить суб-агента (не ждёт)."""

    @abstractmethod
    def stopped_report(self, state: JobState, reason: str) -> JobReport:
        """Итог задачи, остановленной до завершения (``state`` — STOPPED или TIMED_OUT)."""

    @abstractmethod
    def progress(self) -> str:
        """Что нового у работающей задачи — для модели (``background_output``)."""

    def activity(self) -> str:
        """Чем задача занята сейчас — одна строка для пользователя."""
        return ""

    # ------------------------------------------------------------------ #
    @property
    def finished(self) -> bool:
        """Итог записан (задача больше не работает и не ждёт)."""
        return self._done.is_set()

    @property
    def queued(self) -> bool:
        """Задача ждёт в очереди: ещё не запускалась и не остановлена."""
        return self._started is None and not self.finished

    @property
    def report(self) -> JobReport | None:
        """Итог (None — задача ещё работает или ждёт)."""
        return self._report

    @property
    def state(self) -> JobState:
        """Текущее состояние."""
        report = self._report
        if report is not None:
            return report.state
        return JobState.QUEUED if self._started is None else JobState.RUNNING

    @property
    def elapsed(self) -> float:
        """Сколько секунд задача работает (работала); в очереди — 0."""
        started = self._started
        if started is None:
            return float(0)
        ended = time.monotonic() if self._ended is None else self._ended
        return ended - started

    def launch(self, on_finish: Callable[[Job], None]) -> None:
        """Запустить работу; ``on_finish`` — кого известить об итоге (из потока задачи).

        Задачу остановили раньше (снята с очереди, но ещё не запущена) — работа не
        начинается, итог — «остановлена». Запуск и остановка решают это под одной
        блокировкой: остановка либо застаёт задачу запущенной и отменяет её, либо
        запрещает запуск. Итог, записанный ещё до запуска (остановка не дождалась его),
        ``on_finish`` получает здесь — ровно один раз: иначе очередь за задачей не сдвинется.
        """
        with self._lock:
            self._on_finish = on_finish
            allowed = self.stop_request is None and self._report is None
            if allowed:
                self._started = time.monotonic()
            missed = self._report is not None and not self._announced
            self._announced = self._announced or missed
        if allowed:
            self.start()
        elif missed:
            on_finish(self)
        else:
            state, reason = self.stop_request or (JobState.STOPPED, "")
            self.finish(self.stopped_report(state, reason))

    def wait(self, timeout: float | None = None) -> bool:
        """Ждать итога не дольше ``timeout`` секунд; True — итог есть."""
        return self._done.wait(timeout)

    def request_stop(self, reason: str, *, state: JobState = JobState.STOPPED) -> bool:
        """Остановить задачу (не ждёт); False — у неё уже есть итог или остановка идёт."""
        with self._lock:
            if self._report is not None or self.stop_request is not None:
                return False
            self.stop_request = (state, reason)
            started = self._started is not None
        # Не запущенную задачу отменять нечем: launch увидит запрос и не запустит её.
        if started:
            self.cancel()
        return True

    def finish(self, report: JobReport) -> bool:
        """Записать итог; False — итог уже был (этот отброшен)."""
        with self._lock:
            if self._report is not None:
                return False
            self._report = report
            self._ended = time.monotonic()
            callback = self._on_finish
            self._announced = callback is not None
        self._done.set()
        if callback is not None:
            callback(self)
        return True


class BackgroundJobs:
    """Фоновые задачи чата основного агента: запуск, очередь, пределы, итоги к доставке.

    Потокобезопасен: задачи завершаются в своих потоках, а читает их основной.
    Подписчики (:meth:`subscribe`) вызываются из потока завершившейся задачи.
    """

    def __init__(self, limits: Mapping[str, PoolLimits] = DEFAULT_LIMITS) -> None:
        """Пределы ``limits`` — по видам задач (``Job.pool``)."""
        self._limits = limits
        self._cond = threading.Condition()
        self._jobs: dict[str, Job] = {}
        self._queue: deque[Job] = deque()
        self._delivered: set[str] = set()
        self._timers: dict[str, threading.Timer] = {}
        self._listeners: list[Callable[[], None]] = []
        self._counter = 0
        self._exit_hook = False

    # ------------------------------------------------------------------ #
    def subscribe(self, listener: Callable[[], None]) -> None:
        """Вызывать ``listener`` при каждом завершении задачи (из её потока)."""
        with self._cond:
            self._listeners.append(listener)

    def start(self, job: Job) -> str:
        """Запустить задачу или поставить её в очередь; вернуть id.

        ToolError — работают и ждут столько задач этого вида, сколько позволяют пределы.
        Поставлена ли задача в очередь — ``job.queued``.
        """
        with self._cond:
            now = _admission(job, self._jobs.values(), self._queue, self._limit(job.pool))
            self._counter += 1
            job.id = f"bg{self._counter}"
            self._jobs[job.id] = job
            if not now:
                self._queue.append(job)
            if not self._exit_hook:
                # Выход из gigacrew мимо обычного пути (исключение, sys.exit) — команды
                # не должны пережить его.
                atexit.register(self.stop_all, "выход из gigacrew")
                self._exit_hook = True
        if now:
            try:
                self._launch(job)
            except BaseException:
                with self._cond:
                    self._jobs.pop(job.id, None)
                raise
        return job.id

    def get(self, job_id: str) -> Job | None:
        """Задача по id (``bg3``; можно и просто номер); None — нет такой."""
        key = job_id.strip().lower()
        if key.isdecimal():
            key = f"bg{key}"
        with self._cond:
            return self._jobs.get(key)

    def jobs(self) -> list[Job]:
        """Все задачи чата в порядке запуска (и завершившиеся)."""
        with self._cond:
            return list(self._jobs.values())

    def active(self) -> list[Job]:
        """Незавершённые задачи: работающие и ждущие в очереди."""
        return [job for job in self.jobs() if not job.finished]

    def pending(self, *, waking: bool = False) -> bool:
        """Есть итоги, ещё не доставленные модели; ``waking`` — только те, что будят."""
        with self._cond:
            return _pending(self._jobs.values(), self._delivered, waking=waking)

    def take_finished(self) -> list[Job]:
        """Завершившиеся задачи, итог которых модель ещё не получила (отмечаются как полученные)."""
        with self._cond:
            finished = [
                job for job in self._jobs.values() if job.finished and job.id not in self._delivered
            ]
            self._delivered.update(job.id for job in finished)
        return finished

    def mark_delivered(self, job: Job) -> None:
        """Итог задачи модель получила иначе (``background_output``, ``background_stop``)."""
        with self._cond:
            self._delivered.add(job.id)

    def wait(self, timeout: float) -> bool:
        """Ждать итога, который будит агента, не дольше ``timeout`` секунд.

        True — такой итог есть или ждать больше нечего (незавершённых задач нет).
        """
        with self._cond:
            return self._cond.wait_for(
                lambda: (
                    _pending(self._jobs.values(), self._delivered, waking=True)
                    or all(job.finished for job in self._jobs.values())
                ),
                timeout,
            )

    def stop(self, job: Job, reason: str, *, quiet: bool = False) -> bool:
        """Остановить задачу (из очереди — снять) и дождаться итога; False — уже завершилась.

        ``quiet`` — итог не будит агента (остановил пользователь).
        """
        if quiet:
            job.quiet = True
        with self._cond:
            dequeued = job in self._queue
            if dequeued:
                self._queue.remove(job)
        if dequeued:
            job.finish(job.stopped_report(JobState.STOPPED, f"{reason} (before it started)"))
            self._finished(job)  # на запуск она не подписывалась — известить самим
            return True
        if not job.request_stop(reason):
            return False
        _settle(job, STOP_GRACE)
        return True

    def stop_all(self, reason: str, *, forget: bool = False) -> list[Job]:
        """Остановить все незавершённые задачи (их итоги не будят агента); вернуть их.

        Очередь снимается первой: остановка работающих не должна запускать ждущие.
        ``forget`` — забыть и все задачи с их итогами (новый чат).
        """
        with self._cond:
            queued = list(self._queue)
            self._queue.clear()
        for waiting in queued:
            waiting.quiet = True
            waiting.finish(waiting.stopped_report(JobState.STOPPED, reason))
        running = self.active()
        for job in running:
            job.quiet = True
            job.request_stop(reason)
        deadline = time.monotonic() + STOP_GRACE
        for job in running:
            _settle(job, max(deadline - time.monotonic(), 0))
        with self._cond:
            if forget:
                self._jobs.clear()
                self._delivered.clear()
            self._cond.notify_all()
        return [*queued, *running]

    # ------------------------------------------------------------------ #
    def _limit(self, pool: str) -> PoolLimits:
        return self._limits.get(pool, _FALLBACK_LIMITS)

    def _launch(self, job: Job) -> None:
        """Запустить задачу и завести таймер её жёсткого предела."""
        job.launch(self._finished)
        timer = threading.Timer(job.timeout + HARD_STOP_GRACE, _expire, args=(job,))
        timer.daemon = True
        with self._cond:
            if job.finished:
                return
            self._timers[job.id] = timer
        timer.start()

    def _finished(self, job: Job) -> None:
        """Итог записан: снять таймер, запустить ждущую задачу того же вида, известить."""
        with self._cond:
            timer = self._timers.pop(job.id, None)
            listeners = list(self._listeners)
            following = _next_in_queue(self._queue, self._jobs.values(), job.pool, self._limit)
            self._cond.notify_all()
        if timer is not None:
            timer.cancel()
        if following is not None:
            try:
                self._launch(following)
            except Exception as exc:  # задача обещана модели — сбой запуска становится итогом
                following.finish(
                    JobReport(JobState.FAILED, f"The job could not start: {exc}", "не запустилась"),
                )
        for listener in listeners:
            # Сбой подписчика (интерфейс) не должен терять итог: он уже записан.
            with contextlib.suppress(Exception):
                listener()


def job_info(job: Job) -> JobInfo:
    """Снимок задачи для интерфейса."""
    report = job.report
    if report is None:
        activity = job.activity()
        return JobInfo(job.id, job.kind, job.title, job.state, job.elapsed, activity)
    return JobInfo(
        job.id,
        job.kind,
        job.title,
        report.state,
        job.elapsed,
        summary=report.summary,
    )


def job_heading(job: Job) -> str:
    """Строка о задаче для модели: id, вид, описание, состояние и время работы."""
    state = _MODEL_STATES[job.state]
    title = f"{job.id} ({job.kind}: {job.title})"
    elapsed = int(job.elapsed)
    if job.queued:
        return f"{title} — {state}"
    if job.finished:
        return f"{title} — {state} after {elapsed} s"
    limit = int(job.timeout)
    return f"{title} — {state}, {elapsed} s of {limit} s"


def _admission(
    job: Job,
    jobs: Iterable[Job],
    queue: Iterable[Job],
    limits: PoolLimits,
) -> bool:
    """Запускать задачу сейчас (True) или в очередь (False); ToolError — мест нет нигде."""
    running = _occupied(jobs, queue, job.pool)
    if len(running) < limits.running:
        return True
    waiting = [known for known in queue if known.pool == job.pool]
    if len(waiting) < limits.queued:
        return False
    ids = ", ".join(known.id for known in (*running, *waiting))
    total = len(running) + len(waiting)
    raise ToolError(
        f"фоновых задач этого вида уже {total} ({ids}): работают "
        f"{len(running)} из {limits.running}, ждут в очереди {len(waiting)} из "
        f"{limits.queued}. Дождись их итогов или останови ненужную (background_stop).",
    )


def _next_in_queue(
    queue: deque[Job],
    jobs: Iterable[Job],
    pool: str,
    limit: Callable[[str], PoolLimits],
) -> Job | None:
    """Снять с очереди первую задачу вида ``pool``, если для неё освободилось место."""
    if len(_occupied(jobs, queue, pool)) >= limit(pool).running:
        return None
    for waiting in queue:
        if waiting.pool == pool:
            queue.remove(waiting)
            return waiting
    return None


def _occupied(jobs: Iterable[Job], queue: Iterable[Job], pool: str) -> list[Job]:
    """Задачи вида ``pool``, занимающие место: не завершены и не ждут в очереди.

    Задача, снятая с очереди, занимает место сразу, под блокировкой реестра, — ещё до того,
    как запустится её работа: иначе параллельный запуск увидел бы то же свободное место.
    """
    waiting = list(queue)
    return [
        known
        for known in jobs
        if known.pool == pool and not known.finished and known not in waiting
    ]


def _settle(job: Job, grace: float) -> None:
    """После остановки: дать работе завершиться, иначе записать итог «остановлена» самим."""
    if job.wait(grace):
        return
    state, reason = job.stop_request or (JobState.STOPPED, "")
    job.finish(job.stopped_report(state, reason))


def _expire(job: Job) -> None:
    """Жёсткий предел задачи истёк: остановить её (из потока таймера)."""
    limit = int(job.timeout)
    if job.request_stop(f"превышен лимит времени ({limit} с)", state=JobState.TIMED_OUT):
        _settle(job, STOP_GRACE)


def _pending(jobs: Iterable[Job], delivered: set[str], *, waking: bool) -> bool:
    """Есть завершившиеся задачи, чей итог не доставлен; ``waking`` — без «тихих»."""
    return any(
        job.finished and job.id not in delivered and not (waking and job.quiet) for job in jobs
    )
