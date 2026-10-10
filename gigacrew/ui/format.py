"""Форматирование чисел и текста для терминала (без зависимостей от rich)."""

from __future__ import annotations

from datetime import datetime

from gigacrew.permissions import PermissionMode
from gigacrew.ui.theme import MODE_ICONS

SKIPPED_MARK = "…"


def plural(n: int, one: str, few: str, many: str) -> str:
    """Форма слова для числа: 1 шаг, 2 шага, 5 шагов."""
    n = abs(n)
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def mode_badge(mode: PermissionMode) -> str:
    """Режим разрешений для статус-строки и индикатора: значок + название."""
    icon = MODE_ICONS.get(mode.value, "")
    return f"{icon} {mode.label}" if icon else mode.label


def format_tokens(n: int) -> str:
    """Компактная запись числа токенов: 950, 12.3k, 120k, 1.2M."""
    if n < 1000:
        return str(n)
    value, suffix = n / 1000, "k"
    if round(value) >= 1000:
        value, suffix = n / 1_000_000, "M"
    text = f"{value:.1f}" if value < 99.95 else f"{value:.0f}"
    return text.removesuffix(".0") + suffix


def skipped_line(count: int) -> str:
    return f"{SKIPPED_MARK} пропущено {count} {plural(count, 'строка', 'строки', 'строк')} …"


def clip_lines(
    text: str, *, head: int = 6, tail: int = 10, max_line: int = 300, max_chars: int = 4000
) -> str:
    """Сокращает длинный вывод: начало + конец (итог pytest/компилятора обычно в конце).

    Длинные строки обрезаются по ``max_line`` (одна минифицированная строка иначе
    обошла бы лимит по строкам), общий размер — не больше ``max_chars``.
    """
    lines = text.rstrip("\n").split("\n")
    if len(lines) > head + tail + 1:
        lines = [*lines[:head], skipped_line(len(lines) - head - tail), *lines[-tail:]]
    lines = [line if len(line) <= max_line else line[: max_line - 1] + "…" for line in lines]
    result = "\n".join(lines)
    if len(result) > max_chars:
        # Конец важнее начала (итог команды) — делим бюджет, пропуск посередине.
        head_chars = (max_chars - 1) // 2
        tail_chars = max_chars - 1 - head_chars
        result = result[:head_chars] + "…" + result[len(result) - tail_chars :]
    return result


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
    if 0 <= seconds < 3600:
        return f"{int(seconds // 60)} мин назад"
    days = (now.date() - when.date()).days
    if days == 0:
        return f"сегодня {when:%H:%M}"
    if days == 1:
        return f"вчера {when:%H:%M}"
    if when.year == now.year:
        return f"{when:%d.%m %H:%M}"
    return f"{when:%d.%m.%Y}"
