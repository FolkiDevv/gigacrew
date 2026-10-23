"""Форматирование чисел и текста для терминала (без зависимостей от rich)."""

from __future__ import annotations

from datetime import datetime

from gigacrew.permissions import PermissionMode
from gigacrew.ui.theme import MODE_ICONS

SKIPPED_MARK = "…"

# Второй десяток — всегда «много»: «11 шагов», «12–14 шагов», хотя оканчиваются на 1–4.
_TEEN_ONE = 11
_TEEN_FEW_FIRST = 12
_TEEN_FEW_LAST = 14
_MILLION = 1_000_000
# С этого значения десятые не пишутся: 99.95k округлилось бы до «100.0k».
_FRACTION_LIMIT = 99.95
_HOUR_SECONDS = 3600
_ELLIPSIS = "…"


def plural(n: int, one: str, few: str, many: str) -> str:
    """Форма слова для числа: 1 шаг, 2 шага, 5 шагов."""
    number = abs(n)
    if number % 10 == 1 and number % 100 != _TEEN_ONE:
        return one
    if 2 <= number % 10 <= 4 and not _TEEN_FEW_FIRST <= number % 100 <= _TEEN_FEW_LAST:
        return few
    return many


def chat_size(requests: int, messages: int) -> str:
    """Размер чата: «2 запроса · 108 сообщений» (``messages`` 0 — неизвестно, не пишется).

    Запросы — только набранные в приглашении; ответы на вопросы модели и шаги с
    инструментами видны лишь в числе сообщений.
    """
    requests_word = plural(requests, "запрос", "запроса", "запросов")
    text = f"{requests} {requests_word}"
    if messages:
        messages_word = plural(messages, "сообщение", "сообщения", "сообщений")
        text = f"{text} · {messages} {messages_word}"
    return text


def mode_badge(mode: PermissionMode) -> str:
    """Режим разрешений для статус-строки и индикатора: значок + название."""
    icon = MODE_ICONS.get(mode.value, "")
    return f"{icon} {mode.label}" if icon else mode.label


def format_tokens(n: int) -> str:
    """Компактная запись числа токенов: 950, 12.3k, 120k, 1.2M."""
    if n < 1000:
        return str(n)
    scaled, suffix = n / 1000, "k"
    if round(scaled) >= 1000:
        scaled, suffix = n / _MILLION, "M"
    text = f"{scaled:.1f}" if scaled < _FRACTION_LIMIT else f"{scaled:.0f}"
    return text.removesuffix(".0") + suffix


def skipped_line(count: int) -> str:
    """Строка на месте пропущенных: «… пропущено 5 строк …»."""
    word = plural(count, "строка", "строки", "строк")
    return f"{SKIPPED_MARK} пропущено {count} {word} …"


def _clip_line(line: str, max_line: int) -> str:
    """Обрезает строку длиннее ``max_line`` символов, последний из них — «…»."""
    if len(line) <= max_line:
        return line
    kept = line[: max_line - 1]
    return f"{kept}{_ELLIPSIS}"


def clip_lines(
    text: str,
    *,
    head: int = 6,
    tail: int = 10,
    max_line: int = 300,
    max_chars: int = 4000,
) -> str:
    """Сокращает длинный вывод: начало + конец (итог pytest/компилятора обычно в конце).

    Длинные строки обрезаются по ``max_line`` (одна минифицированная строка иначе
    обошла бы лимит по строкам), общий размер — не больше ``max_chars``.
    """
    lines = text.rstrip("\n").split("\n")
    if len(lines) > head + tail + 1:
        skipped = skipped_line(len(lines) - head - tail)
        lines = [*lines[:head], skipped, *lines[-tail:]]
    lines = [_clip_line(line, max_line) for line in lines]
    clipped = "\n".join(lines)
    if len(clipped) > max_chars:
        # Конец важнее начала (итог команды) — делим бюджет, пропуск посередине.
        head_chars = (max_chars - 1) // 2
        tail_chars = max_chars - 1 - head_chars
        start, end = clipped[:head_chars], clipped[len(clipped) - tail_chars :]
        clipped = f"{start}{_ELLIPSIS}{end}"
    return clipped


def format_when(when: datetime, now: datetime | None = None) -> str:
    """Когда это было: «только что», «5 мин назад», «сегодня 14:30», «вчера 09:05», «03.10»."""
    # Наивное время считается местным: aware и naive нельзя вычитать друг из друга.
    now = datetime.now().astimezone() if now is None else now
    if now.tzinfo is None:
        now = now.astimezone()
    when = when.astimezone(now.tzinfo)
    seconds = (now - when).total_seconds()
    if 0 <= seconds < 60:
        return "только что"
    if 0 <= seconds < _HOUR_SECONDS:
        return f"{int(seconds // 60)} мин назад"
    days = (now.date() - when.date()).days
    clock = when.strftime("%H:%M")
    if days == 0:
        return f"сегодня {clock}"
    if days == 1:
        return f"вчера {clock}"
    if when.year == now.year:
        return when.strftime("%d.%m %H:%M")
    return when.strftime("%d.%m.%Y")
