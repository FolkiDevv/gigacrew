"""Модели в CLI: каталог чат-моделей для ``/model`` и замер окна контекста.

* :class:`ModelCatalog` — список чат-моделей API, загружается фоновым потоком при
  старте REPL (строка ввода не ждёт сети); источник автодополнения ``/model``;
* :func:`ensure_context_window` — если окно выбранной модели ещё не замерено,
  предлагает замерить его (при старте, ``/model``, ``/resume``; с ``-y`` — без
  вопроса) и сохраняет в ``~/.gigacrew``;
* :func:`measure_context_window` — замер с индикатором (Esc/Ctrl+C — отмена);
  его же вызывает ``gigacrew --test-context``.
"""

from __future__ import annotations

import contextlib
import threading
from contextlib import AbstractContextManager

from gigacrew.agent.context_window import DEFAULT_CONTEXT_WINDOW
from gigacrew.agent.loop import Agent
from gigacrew.llm.base import LLMError, LLMProvider
from gigacrew.llm.context_probe import (
    MAX_CONTEXT_WINDOW,
    ProbeResult,
    ProbeStep,
    probe_context_window,
)
from gigacrew.llm.model_windows import ModelWindows
from gigacrew.ui.console import Console
from gigacrew.ui.format import format_tokens, plural


class ModelCatalog:
    """Чат-модели провайдера. Ошибка загрузки — пустой список (без вывода: это фон)."""

    def __init__(self, provider: LLMProvider):
        self._provider = provider
        self._models: list[str] | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Загрузить список в фоне (повторный вызов ничего не делает)."""
        if self._thread is None and self._models is None:
            self._thread = threading.Thread(target=self.load, name="model-catalog", daemon=True)
            self._thread.start()

    def load(self) -> list[str]:
        try:
            models = self._provider.list_models()
        except Exception:  # сеть/авторизация: автодополнения просто не будет
            models = []
        self._models = models
        return models

    @property
    def models(self) -> list[str] | None:
        """Загруженный список; None — ещё загружается (или не запрашивался)."""
        return self._models

    def choices(self, windows: ModelWindows, current: str) -> list[tuple[str, str]]:
        """Пары (модель, пояснение) для меню автодополнения."""
        return [(name, describe_model(name, windows, current)) for name in self._models or []]


def describe_model(name: str, windows: ModelWindows, current: str = "") -> str:
    window = windows.get(name)
    text = f"окно {format_tokens(window)}" if window else "окно не замерено"
    return f"текущая · {text}" if name == current else text


# ------------------------------- замер окна ------------------------------- #
def _step_text(step: ProbeStep) -> str:
    if step.ok:
        return f"~{format_tokens(step.target)} — прошло ({step.prompt_tokens} ток.)"
    code = step.status if step.status is not None else "нет ответа"
    return f"~{format_tokens(step.target)} — отказ ({code})"


def describe_result(result: ProbeResult) -> str:
    if result.capped:
        bound = f"не меньше {format_tokens(result.window)} (максимум замера)"
    else:
        bound = f"{result.window} токенов"
    probes = f"{result.probes} {plural(result.probes, 'проба', 'пробы', 'проб')}"
    return (
        f"окно {result.model}: {bound} · {probes}, "
        f"потрачено {format_tokens(result.billed_tokens)} токенов"
    )


def measure_context_window(
    provider: LLMProvider,
    model: str,
    ui: Console,
    *,
    interrupt: AbstractContextManager[object] | None = None,
    verbose: bool = False,
) -> ProbeResult | None:
    """Замер окна ``model`` с индикатором (сохраняет :func:`save_window`).

    None — замер не удался (предупреждение выведено). ``KeyboardInterrupt``
    пробрасывается. ``verbose`` — печатать каждую пробу.
    """
    last = "калибровка"

    def on_step(step: ProbeStep) -> None:
        nonlocal last
        last = _step_text(step)
        if verbose:
            ui.system(f"  {last}")

    guard = interrupt if interrupt is not None else contextlib.nullcontext()
    # Подробный режим печатает пробы строками — индикатор поверх них не нужен.
    indicator = (
        contextlib.nullcontext() if verbose else ui.progress(f"замеряю окно {model}", lambda: last)
    )
    try:
        with guard, indicator:
            result = probe_context_window(provider, model, on_step=on_step)
    except LLMError as e:
        ui.warn(f"не удалось замерить окно {model}: {e}")
        return None
    return result


def save_window(windows: ModelWindows, result: ProbeResult, ui: Console, *, base_url: str) -> bool:
    """Запоминает замер; False — файл записать не удалось (в памяти замер остаётся)."""
    try:
        windows.record(result, base_url=base_url)
    except OSError as e:
        ui.warn(f"замер окна не сохранён в {windows.path}: {e}")
        return False
    return True


def ensure_context_window(
    agent: Agent,
    ui: Console,
    *,
    interrupt: AbstractContextManager[object] | None = None,
    interactive: bool = True,
    declined: set[str] | None = None,
) -> None:
    """Предлагает замерить окно текущей модели агента, если оно неизвестно.

    Замер оплачивается (проба размером до окна модели), поэтому без ``-y`` он
    только с согласия пользователя; не интерактивно (``-p``, ввод не с терминала)
    — не замеряется, выводится подсказка про ``--test-context``. Отказ запоминается
    в ``declined`` до конца сессии. Не нужен при явном бюджете
    (``GIGACREW_CONTEXT_TOKENS``), при ``GIGACREW_AUTO_MEASURE=0`` и у провайдера
    без замера. Отмена или ошибка — работаем с окном по умолчанию.
    """
    if (
        agent.config.context_budget_tokens is not None
        or agent.context_window is not None
        or not agent.provider.supports_measure
        or not agent.config.auto_measure
    ):
        return
    model = agent.model
    if declined is not None and model in declined:
        return
    fallback = format_tokens(DEFAULT_CONTEXT_WINDOW)
    later = f"замерить: gigacrew --test-context {model}"
    if not agent.config.auto_approve:
        if not interactive or not ui.interactive():
            ui.system(f"окно контекста {model} не замерено — считаем {fallback}; {later}")
            return
        question = (
            f"Окно контекста модели {model} не замерено. Замерить? Пробные запросы "
            f"оплачиваются — до размера окна модели (≤{format_tokens(MAX_CONTEXT_WINDOW)} ток.)"
        )
        if not ui.ask(question):
            if declined is not None:
                declined.add(model)
            ui.system(f"пока считаем окно {model} равным {fallback}; {later}")
            return
    ui.info(f"замеряю окно контекста модели {model} (несколько запросов)")
    try:
        result = measure_context_window(agent.provider, model, ui, interrupt=interrupt)
    except KeyboardInterrupt:
        ui.warn(f"замер окна прерван — пока считаем окно {fallback}")
        return
    if result is None:
        ui.system(f"пока считаем окно {model} равным {fallback}")
        return
    ui.system(describe_result(result))
    save_window(agent.windows, result, ui, base_url=agent.config.base_url)
