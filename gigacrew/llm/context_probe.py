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
# Точность замера (:func:`_tolerance`): вилка уже 1/50 (2%), но не меньше 512 токенов.
_MIN_TOLERANCE = 512
_TOLERANCE_DIVISOR = 50
# Отступ от кандидата (:func:`_margin`): 1/400 его размера, но не меньше 32 токенов.
_MIN_MARGIN = 32
_MARGIN_DIVISOR = 400
# Число из текста отказа — возможный лимит, если оно больше 1024 и заметно (на 3%) меньше
# отправленного.
_MIN_HINT = 1_024
_HINT_SHARE = 0.97
# Множитель и слагаемое в строке филлера: остатки по разным модулям — строки разнообразнее.
_FILLER_FACTOR_MODULO = 97
_FILLER_OFFSET_MODULO = 13
# Типовые окна — степени двойки от 4K (2**12) до 256K (2**18) включительно.
_SMALLEST_POWER = 12
_LARGEST_POWER = 18

# Типовые размеры окон (в токенах), проверяются сверху вниз.
STANDARD_WINDOWS = tuple(
    sorted(
        {2**power for power in range(_SMALLEST_POWER, _LARGEST_POWER + 1)}
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
    """Итог замера окна модели."""

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
                index = len(self._parts) % 1000
                number = format(index, "03d")
                factor = format(index % _FILLER_FACTOR_MODULO, "02d")
                offset = format(index % _FILLER_OFFSET_MODULO, "02d")
                line = (
                    f"{number}. Замер окна контекста: def step_{number}(x): return x * "
                    f"{factor} + {offset}  # the quick brown fox jumps over the lazy dog\n"
                )
                self._parts.append(line)
                self._size += len(line)
            self._text = "".join(self._parts)
        return self._text[:chars]


def _tolerance(lo: int) -> int:
    """Точность замера: вилка уже 2% (но не меньше 512 токенов) — достаточно.

    Бюджет всё равно берётся от окна с запасом (``budget_for_window``).
    """
    return max(_MIN_TOLERANCE, lo // _TOLERANCE_DIVISOR)


def _margin(candidate: int) -> int:
    """Отступ от кандидата: проба «чуть ниже» и «чуть выше» его."""
    return max(_MIN_MARGIN, candidate // _MARGIN_DIVISOR)


def _hints(detail: str, lo: int, sent: int) -> list[int]:
    """Числа из текста отказа, которые могут быть лимитом.

    Это числа больше подтверждённого и заметно меньше отправленного (``sent``) —
    «you requested N tokens» сюда не попадёт.
    """
    found = set()
    for raw in _NUMBER_RE.findall(detail):
        try:
            number = int(raw.replace("_", "").replace(",", ""))
        except ValueError:
            continue
        if max(lo, _MIN_HINT) < number < sent * _HINT_SHARE:
            found.add(number)
    return sorted(found, reverse=True)


class _Probe:
    """Ход замера окна одной модели: вилка, счётчики проб и подсказки из отказов."""

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        on_step: Callable[[ProbeStep], None] | None,
        cap: int,
    ) -> None:
        self._provider = provider
        self._model = model
        self._on_step = on_step
        self._cap = cap
        self._filler = _Filler()
        self._chars_per_token = _INITIAL_CHARS_PER_TOKEN
        self.lo = 0  # наибольший подтверждённый размер, токены
        # Наименьший отклонённый текст хранится в символах: соотношение символов и
        # токенов уточняется по ходу замера, а длина отклонённого текста известна точно.
        self._rejected_chars: int | None = None
        self.probes = 0
        self._billed = 0
        self._tried: set[int] = set()
        self._passed: set[int] = set()
        self._hints: list[int] = []

    def upper(self) -> int:
        """Верх вилки, токены: наименьший отклонённый размер по текущему соотношению."""
        if self._rejected_chars is None:
            return self._cap + 1
        return max(int(self._rejected_chars / self._chars_per_token), self.lo + 1)

    def next_target(self) -> int:
        """Следующий размер пробы внутри вилки."""
        return _next_target(self.lo, self.upper(), self._hints, self._tried, self._passed)

    def attempt(self, target: int) -> bool:
        """Проба размером ~``target`` токенов; True — запрос прошёл."""
        self._tried.add(target)
        text = self._filler.text(max(int(target * self._chars_per_token), 1))
        self.probes += 1
        try:
            prompt_tokens = self._provider.measure_prompt(text, model=self._model)
        except PromptTooLongError as exc:
            self._note_rejected(target, len(text), exc)
            return False
        self._note_passed(target, len(text), prompt_tokens)
        return True

    def measurement(self, upper_bound: int | None, *, capped: bool) -> ProbeResult:
        """Итог замера с верхней границей ``upper_bound``."""
        return ProbeResult(
            self._model,
            min(self.lo, self._cap),
            upper_bound,
            capped=capped,
            probes=self.probes,
            billed_tokens=self._billed,
        )

    def _note_rejected(self, target: int, sent_chars: int, error: PromptTooLongError) -> None:
        """Учитывает отказ: сужает вилку сверху, добавляет подсказки из текста отказа."""
        if self._rejected_chars is None or sent_chars < self._rejected_chars:
            self._rejected_chars = sent_chars
        found = _hints(error.detail, self.lo, target)
        self._hints = sorted(set(self._hints) | set(found), reverse=True)
        if self._on_step:
            self._on_step(ProbeStep(target, ok=False, status=error.status))

    def _note_passed(self, target: int, sent_chars: int, prompt_tokens: int) -> None:
        """Учитывает прошедшую пробу: уточняет соотношение, сужает вилку снизу."""
        self._billed += prompt_tokens + 1
        if self.lo and prompt_tokens < target * _TRUNCATED_SHARE:
            # После калибровки соотношение промахивается на десятки процентов, не вдвое:
            # сервер молча обрезал запрос. Доверься замер ему — соотношение росло бы с каждой
            # пробой, а пробы раздувались бы в геометрической прогрессии.
            raise LLMError(
                f"сервер учёл {prompt_tokens} токенов из ~{target} отправленных — похоже, "
                "он молча обрезает запрос; окно так не замерить",
            )
        self._chars_per_token = sent_chars / prompt_tokens
        self.lo = max(self.lo, prompt_tokens)
        self._passed.add(target)
        if self._on_step:
            self._on_step(ProbeStep(target, ok=True, prompt_tokens=prompt_tokens))


def probe_context_window(
    provider: LLMProvider,
    model: str,
    *,
    on_step: Callable[[ProbeStep], None] | None = None,
    cap: int = MAX_CONTEXT_WINDOW,
) -> ProbeResult:
    """Замеряет окно ``model``.

    Ошибки обращения — :class:`LLMError`; провайдер без поддержки замера —
    ``NotImplementedError``.
    """
    probe = _Probe(provider, model, on_step, cap)
    try:
        probe.attempt(CALIBRATION_TOKENS)
    except LLMError as exc:
        raise LLMError(f"пробный запрос ~{CALIBRATION_TOKENS} токенов не прошёл: {exc}") from exc
    if probe.lo == 0:
        raise LLMError(f"модель отклонила даже запрос ~{CALIBRATION_TOKENS} токенов")

    # Цель — оценка: калибровка на маленьком запросе может заметно промахнуться.
    # Прошедший запрос меньше потолка перепроверяется с уточнённым соотношением.
    while probe.attempt(cap):
        if probe.lo >= cap - _tolerance(cap) or probe.probes >= MAX_PROBES:
            return probe.measurement(None, capped=True)

    while probe.upper() - probe.lo > _tolerance(probe.lo) and probe.probes < MAX_PROBES:
        probe.attempt(probe.next_target())

    return probe.measurement(probe.upper(), capped=False)


def _next_target(lo: int, hi: int, hints: list[int], tried: set[int], passed: set[int]) -> int:
    """Следующий размер пробы внутри вилки (lo, hi)."""
    for candidate in (*hints, *STANDARD_WINDOWS):
        margin = _margin(candidate)
        below, above = candidate - margin, candidate + margin
        # Чуть ниже кандидата прошло — проверяем чуть выше, чтобы закрыть вилку.
        if below in passed and lo < above < hi and above not in tried:
            return above
        if lo < below < hi and below not in tried:
            return below
    return (lo + hi) // 2
