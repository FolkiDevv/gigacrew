"""Замер окна контекста модели пробными запросами.

API не сообщает размер окна, а развёрнутые модели могут иметь
произвольные окна, поэтому лимит прощупывается: модели отправляется текст
заданного размера с ответом в один токен (:meth:`LLMProvider.measure_prompt`).
Результат консервативный — наибольший **подтверждённый** размер запроса
(``prompt_tokens`` по данным API), а не оценка.

Порядок проб:
  1. калибровка — небольшой запрос: соотношение символов и токенов филлера для
     токенизатора именно этой модели (уточняется после каждой успешной пробы);
     отказ здесь — ошибка не про размер, замер прерывается;
  2. потолок :data:`MAX_CONTEXT_WINDOW` — прошёл, значит окно не меньше его;
  3. кандидаты сверху вниз: числа из текста отказа (например, «maximum context
     length is 32768») и типовые размеры окон. Проба чуть ниже кандидата, после
     успеха — чуть выше: окно обычно найдено за одну оплачиваемую пробу, а отказы
     отклоняются до генерации;
  4. если кандидаты кончились, а вилка шире :func:`_tolerance`, — бисекция.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from gigacrew.llm.base import LLMError, LLMProvider, PromptTooLongError

MAX_CONTEXT_WINDOW = 262_144  # 256K — больше окна у доступных моделей не бывает
CALIBRATION_TOKENS = 1_000
MAX_PROBES = 40
_INITIAL_CHARS_PER_TOKEN = 3.0
# Проба после калибровки, по данным API меньше этой доли целевого размера, — сервер обрезал
# запрос, а не промахнулась оценка.
_TRUNCATED_SHARE = 0.5

# Типовые размеры окон (в токенах), проверяются сверху вниз.
STANDARD_WINDOWS = tuple(
    sorted(
        {2**k for k in range(12, 19)}
        | {8_000, 16_000, 32_000, 64_000, 100_000, 128_000, 160_000, 200_000},
        reverse=True,
    ),
)

_NUMBER_RE = re.compile(r"\d[\d_,]*\d|\d")


@dataclass(frozen=True)
class ProbeStep:
    """Одна проба: целевой размер (оценка) и исход."""

    target: int
    ok: bool
    prompt_tokens: int = 0  # при успехе — точный размер запроса по данным API
    status: int | None = None  # при отказе — HTTP-статус (None — таймаут/обрыв)


@dataclass(frozen=True)
class ProbeResult:
    model: str
    window: int  # наибольший подтверждённый размер запроса, токены
    upper_bound: int | None  # наименьший отклонённый размер (оценка); None — упёрлись в потолок
    capped: bool  # прошёл запрос размером MAX_CONTEXT_WINDOW
    probes: int
    billed_tokens: int  # токены успешных проб (отказы не тарифицируются)


class _Filler:
    """Детерминированный разнообразный текст: русский, английский, код, числа.

    Числа фиксированной ширины: строки одинаково токенизируются в начале и в конце
    текста, и соотношение символов и токенов не плывёт с размером пробы.
    """

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._size = 0
        self._text = ""

    def text(self, chars: int) -> str:
        if len(self._text) < chars:
            while self._size < chars:
                n = len(self._parts) % 1000
                line = (
                    f"{n:03d}. Замер окна контекста: def step_{n:03d}(x): return x * "
                    f"{n % 97:02d} + {n % 13:02d}  # the quick brown fox jumps over the lazy dog\n"
                )
                self._parts.append(line)
                self._size += len(line)
            self._text = "".join(self._parts)
        return self._text[:chars]


def _tolerance(lo: int) -> int:
    """Точность замера: вилка уже 2% (но не меньше 512 токенов) — достаточно:
    бюджет всё равно берётся от окна с запасом (``budget_for_window``).
    """
    return max(512, lo // 50)


def _margin(candidate: int) -> int:
    """Отступ от кандидата: проба «чуть ниже» и «чуть выше» его."""
    return max(32, candidate // 400)


def _hints(detail: str, lo: int, sent: int) -> list[int]:
    """Числа из текста отказа, которые могут быть лимитом: больше подтверждённого и
    заметно меньше отправленного (``sent``) — «you requested N tokens» сюда не попадёт.
    """
    found = set()
    for raw in _NUMBER_RE.findall(detail):
        try:
            value = int(raw.replace("_", "").replace(",", ""))
        except ValueError:
            continue
        if max(lo, 1_024) < value < sent * 0.97:
            found.add(value)
    return sorted(found, reverse=True)


def probe_context_window(
    provider: LLMProvider,
    model: str,
    *,
    on_step: Callable[[ProbeStep], None] | None = None,
    cap: int = MAX_CONTEXT_WINDOW,
) -> ProbeResult:
    """Замеряет окно ``model``. Ошибки обращения — :class:`LLMError`;
    провайдер без поддержки замера — ``NotImplementedError``.
    """
    filler = _Filler()
    chars_per_token = _INITIAL_CHARS_PER_TOKEN
    lo = 0  # наибольший подтверждённый размер, токены
    # Наименьший отклонённый текст хранится в символах: соотношение символов и
    # токенов уточняется по ходу замера, а длина отклонённого текста известна точно.
    rejected_chars: int | None = None
    probes = billed = 0
    tried: set[int] = set()
    passed: set[int] = set()
    hints: list[int] = []

    def upper() -> int:
        if rejected_chars is None:
            return cap + 1
        return max(int(rejected_chars / chars_per_token), lo + 1)

    def attempt(target: int) -> bool:
        nonlocal chars_per_token, lo, rejected_chars, probes, billed, hints
        tried.add(target)
        text = filler.text(max(int(target * chars_per_token), 1))
        probes += 1
        try:
            prompt_tokens = provider.measure_prompt(text, model=model)
        except PromptTooLongError as e:
            if rejected_chars is None or len(text) < rejected_chars:
                rejected_chars = len(text)
            hints = sorted(set(hints) | set(_hints(e.detail, lo, target)), reverse=True)
            if on_step:
                on_step(ProbeStep(target, ok=False, status=e.status))
            return False
        billed += prompt_tokens + 1
        if lo and prompt_tokens < target * _TRUNCATED_SHARE:
            # После калибровки соотношение промахивается на десятки процентов, не вдвое:
            # сервер молча обрезал запрос. Доверься замер ему — соотношение росло бы с каждой
            # пробой, а пробы раздувались бы в геометрической прогрессии.
            raise LLMError(
                f"сервер учёл {prompt_tokens} токенов из ~{target} отправленных — похоже, "
                "он молча обрезает запрос; окно так не замерить",
            )
        chars_per_token = len(text) / prompt_tokens
        lo = max(lo, prompt_tokens)
        passed.add(target)
        if on_step:
            on_step(ProbeStep(target, ok=True, prompt_tokens=prompt_tokens))
        return True

    try:
        attempt(CALIBRATION_TOKENS)
    except LLMError as e:
        raise LLMError(f"пробный запрос ~{CALIBRATION_TOKENS} токенов не прошёл: {e}") from e
    if lo == 0:
        raise LLMError(f"модель отклонила даже запрос ~{CALIBRATION_TOKENS} токенов")

    # Цель — оценка: калибровка на маленьком запросе может заметно промахнуться.
    # Прошедший запрос меньше потолка перепроверяется с уточнённым соотношением.
    while attempt(cap):
        if lo >= cap - _tolerance(cap) or probes >= MAX_PROBES:
            return ProbeResult(model, min(lo, cap), None, True, probes, billed)

    while upper() - lo > _tolerance(lo) and probes < MAX_PROBES:
        attempt(_next_target(lo, upper(), hints, tried, passed))

    return ProbeResult(model, min(lo, cap), upper(), False, probes, billed)


def _next_target(lo: int, hi: int, hints: list[int], tried: set[int], passed: set[int]) -> int:
    """Следующий размер пробы внутри вилки (lo, hi)."""
    for candidate in (*hints, *STANDARD_WINDOWS):
        m = _margin(candidate)
        below, above = candidate - m, candidate + m
        # Чуть ниже кандидата прошло — проверяем чуть выше, чтобы закрыть вилку.
        if below in passed and lo < above < hi and above not in tried:
            return above
        if lo < below < hi and below not in tried:
            return below
    return (lo + hi) // 2
