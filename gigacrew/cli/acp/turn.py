"""Ходы сессии ACP: агент работает в своём потоке, ``session/cancel`` его прерывает.

Обработчик ``session/prompt`` запускает ход в отдельном потоке и ждёт его конца или
отмены; так же, в очереди ходов, выполняется и сжатие истории (``/compact``), а скилл
(``/<скилл> аргументы``) раскрывается в запрос, как в CLI. Режим, который
агент сменил за ход (одобренный план), сообщается клиенту до ответа на ход.

Отмена посылает в поток хода асинхронный ``KeyboardInterrupt`` (не больше одного на ход),
затем поднимает флаг хода (процессы ``run_process`` убиваются, см.
:func:`gigacrew.tools.cancel.cancellation`) и закрывает ход для клиента: поздние события не
отправляются — кроме, возможно, нескольких, которые поток успел отправить между
прерыванием и закрытием. Поток, который исключения не видит (ждёт сеть без отрезков),
ответ ``cancelled`` не задерживает дольше :data:`_CANCEL_GRACE`; следующий ход сессии
дожидается его конца.
"""

from __future__ import annotations

import ctypes
import threading
from collections.abc import Callable, Generator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from functools import partial
from typing import Final

from gigacrew.agent.compaction import CompactionError
from gigacrew.agent.context_window import DEFAULT_CONTEXT_WINDOW
from gigacrew.agent.events import TurnStats
from gigacrew.agent.loop import Agent
from gigacrew.cli.acp.events import AcpEvents
from gigacrew.cli.acp.protocol import CANCELLED, stop_reason
from gigacrew.cli.acp.rpc import INTERNAL_ERROR, INVALID_REQUEST, RpcError
from gigacrew.cli.acp.slash import SlashCommands, compact_instructions
from gigacrew.llm.base import LLMError
from gigacrew.permissions import PermissionMode
from gigacrew.tools.cancel import cancellation

__all__ = ["TurnRunner"]

# Сколько ответ cancelled ждёт, пока прерванный поток хода доработает, с.
_CANCEL_GRACE: Final = 0.5
# Как часто ожидающий своей очереди ход проверяет, не отменён ли он, с.
_POLL_INTERVAL: Final = 0.05

# Работа хода в его потоке; итоги хода модели (None — модель не отвечала).
type _Work = Callable[[], TurnStats | None]


@dataclass
class _Turn:
    """Ход: работа, флаг отмены и итог; поля потока меняются под блокировкой бегуна."""

    work: _Work
    cancel: threading.Event = field(default_factory=threading.Event)
    wake: threading.Event = field(default_factory=threading.Event)  # конец хода или отмена
    done: threading.Event = field(default_factory=threading.Event)  # поток хода завершился
    thread_id: int | None = None  # поток, выполняющий агента (None — ещё не начал)
    interrupted: bool = False  # в поток послано асинхронное исключение
    finished: bool = False  # агент больше не выполняется: прерывать нечего
    failure: BaseException | None = None
    stats: TurnStats | None = None  # итоги хода модели


class TurnRunner:
    """Ходы одного агента и его слэш-команды: не больше одного хода одновременно."""

    def __init__(
        self,
        agent: Agent,
        events: AcpEvents,
        after_turn: Callable[[], None],
    ) -> None:
        """Ходы агента ``agent``, события которого получает ``events``.

        ``after_turn`` выполняется в потоке хода, когда агент закончил (и ход отменённый),
        до ответа на ход: сохранение чата.
        """
        self._agent = agent
        self._events = events
        self._after_turn = after_turn
        self._lock = threading.Lock()
        self._active: _Turn | None = None  # принятый и ещё не отвеченный ход
        self._agent_free = threading.Semaphore(1)  # свободен: поток прошлого хода завершился
        self._commands = SlashCommands(agent.workspace)

    def prompt(self, text: str) -> str:
        """Выполняет ход с запросом ``text``; stopReason ответа.

        ``/compact [пожелания]`` сжимает историю без хода модели, ``/<скилл> аргументы``
        уходит модели раскрытым скиллом. Идущий неотменённый ход — ошибка «Invalid Request»;
        ход упал — ошибка JSON-RPC с текстом сбоя (сессия остаётся рабочей).
        """
        instructions = compact_instructions(text)
        work: _Work
        if instructions is None:
            work = partial(_model_turn, self._agent, self._commands.expand(text))
        else:
            work = partial(_compact, self._agent, self._events, instructions)
        turn = self._accept(work)
        mode = self._agent.mode
        with ExitStack() as stack:
            stack.callback(self._forget, turn)
            stack.callback(self._report_mode, mode)
            return self._answer(turn)

    def advertise_commands(self) -> None:
        """Сообщает клиенту слэш-команды сессии (``available_commands_update``)."""
        self._events.report_commands(self._commands.advertised())

    def cancel(self) -> None:
        """Отменяет текущий ход (нет хода или уже отменён — ничего не делает)."""
        with self._lock:
            turn = self._active
            if turn is None or turn.cancel.is_set():
                return
            # Прерывание — раньше флага: поток хода, ждущий ответа клиента, иначе увидел бы
            # флаг и прервался сам, а отложенное прерывание сработало бы второй раз — в
            # обработчиках агента, посреди починки истории.
            if turn.thread_id is not None and not turn.finished:
                turn.interrupted = True
                _raise_in_thread(turn.thread_id, KeyboardInterrupt)
            turn.cancel.set()
            self._events.close_turn()
        turn.wake.set()

    @contextmanager
    def idle(self, timeout: float) -> Generator[bool]:
        """Агент свободен внутри блока: поток прошлого хода доработал (ждём до ``timeout`` с).

        False — не дождались (поток не видит прерывания): агента трогать нельзя.
        """
        free = self._agent_free.acquire(timeout=timeout)
        try:
            yield free
        finally:
            if free:
                self._agent_free.release()

    def _accept(self, work: _Work) -> _Turn:
        with self._lock:
            if self._active is not None and not self._active.cancel.is_set():
                raise RpcError(INVALID_REQUEST, "в сессии уже идёт ход")
            turn = _Turn(work)
            self._active = turn
        return turn

    def _forget(self, turn: _Turn) -> None:
        """На ход ответили: он больше не текущий (если его не сменил следующий)."""
        with self._lock:
            if self._active is turn:
                self._active = None

    def _report_mode(self, before: PermissionMode) -> None:
        """Режим сменился за ход (одобрен план): клиент узнаёт об этом до ответа на ход."""
        mode = self._agent.mode
        if mode is not before:
            self._events.report_mode(mode)

    def _answer(self, turn: _Turn) -> str:
        if not self._start(turn):
            return CANCELLED
        turn.wake.wait()
        if turn.cancel.is_set():
            turn.done.wait(_CANCEL_GRACE)
            return CANCELLED
        return self._outcome(turn)

    def _start(self, turn: _Turn) -> bool:
        """Дожидается конца прошлого хода и запускает поток; False — отменён раньше."""
        while not self._agent_free.acquire(timeout=_POLL_INTERVAL):
            if turn.cancel.is_set():
                return False
        with self._lock:
            if turn.cancel.is_set():
                self._agent_free.release()
                return False
            self._events.open_turn(turn.cancel)
        threading.Thread(target=self._work, args=(turn,), name="acp-turn", daemon=True).start()
        return True

    def _outcome(self, turn: _Turn) -> str:
        """Ответ на завершившийся без отмены ход."""
        failure = turn.failure
        if failure is None:
            stats = turn.stats
            if stats is not None:
                size = self._agent.context_window or DEFAULT_CONTEXT_WINDOW
                self._events.report_usage(stats.context_tokens, size)
            self._events.close_turn()
            return stop_reason(stats)
        self._events.close_turn()
        if isinstance(failure, KeyboardInterrupt):
            return CANCELLED
        if isinstance(failure, LLMError):
            raise RpcError(INTERNAL_ERROR, f"ошибка модели: {failure}")
        raise RpcError(INTERNAL_ERROR, f"сбой хода: {type(failure).__name__}: {failure}")

    def _work(self, turn: _Turn) -> None:
        """Тело потока хода; асинхронное исключение отмены из него не выходит.

        Исключение может прийти на любом шаге до :meth:`_settle`, поэтому все шаги до
        него — внутри ``try`` этой функции, а не в вызываемых.
        """
        try:
            self._run(turn)
        # Любой итог хода, и KeyboardInterrupt отмены тоже, — обработчику запроса.
        except BaseException as exc:  # noqa: WPS424
            turn.failure = exc
        settled = False
        while not settled:
            try:
                settled = self._settle(turn)
            except KeyboardInterrupt:
                # Исключение отмены дошло только сейчас: оно было одно — повторяем.
                settled = False
        with ExitStack() as stack:
            stack.callback(turn.wake.set)
            stack.callback(turn.done.set)
            stack.callback(self._agent_free.release)
            self._after_turn()

    def _run(self, turn: _Turn) -> None:
        with self._lock:
            if turn.cancel.is_set():
                raise KeyboardInterrupt
            turn.thread_id = threading.get_ident()
        with cancellation(turn.cancel):
            turn.stats = turn.work()

    def _settle(self, turn: _Turn) -> bool:
        """Агент закончил: прерывать нечего, недоставленное исключение отмены снимается."""
        with self._lock:
            turn.finished = True
            if turn.interrupted and turn.thread_id is not None:
                _raise_in_thread(turn.thread_id, None)
        return True


def _model_turn(agent: Agent, text: str) -> TurnStats | None:
    """Ход модели с запросом ``text``; его итоги."""
    agent.run_turn(text)
    return agent.last_turn


def _compact(agent: Agent, events: AcpEvents, instructions: str) -> None:
    """Сжатие истории с пожеланиями ``instructions`` (``/compact``): хода модели нет.

    Итог — текстом ответа (сжато или сжимать нечего); пригодного краткого содержания нет (например,
    оно не короче истории) — сообщение об этом, а не сбой. Прочие ошибки модели — сбой
    хода. История в этих случаях не меняется.
    """
    try:
        compacted = agent.compact(instructions)
    except CompactionError as error:
        events.on_assistant_text(f"не удалось сжать контекст: {error} — история не изменилась")
        return
    # Ответ без текста T3 показывает как пустой ход: итог сжатия — текстом ответа.
    if compacted is None:
        events.on_assistant_text("сжимать нечего: история пуста или уже короткая")
        return
    events.on_assistant_text(
        f"контекст сжат: ~{compacted.before_tokens} → ~{compacted.after_tokens} ток., "
        f"в краткое содержание свёрнуто сообщений: {compacted.messages}",
    )


def _raise_in_thread(thread_id: int, exception: type[BaseException] | None) -> None:
    """Асинхронное исключение в поток ``thread_id``; None — снять недоставленное."""
    target = None if exception is None else ctypes.py_object(exception)
    ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread_id), target)
