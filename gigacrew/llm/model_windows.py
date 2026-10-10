"""Замеренные окна контекста моделей: ``~/.gigacrew/models.json``.

Файл общий для всех проектов пользователя (окно — свойство модели, а не проекта),
ключ — имя модели. Пишется атомарно (временный файл + ``os.replace``) и перед
записью перечитывается, чтобы не затереть замер из другого окна gigacrew.

Формат::

    {"version": 1, "models": {"model-large": {"context_window": 126998,
        "upper_bound": 128319, "capped": false, "probes": 8,
        "measured_at": "2026-10-09T12:00:00+03:00", "base_url": "https://..."}}}
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from gigacrew.llm.context_probe import ProbeResult

FORMAT_VERSION = 1
USER_DIR_NAME = ".gigacrew"
FILE_NAME = "models.json"


def default_path() -> Path:
    return Path.home() / USER_DIR_NAME / FILE_NAME


class ModelWindows:
    """Окна моделей в памяти + (необязательно) файл, куда сохраняются замеры.

    ``path=None`` — только память (тесты, агент без CLI): конструктор и чтение
    не трогают файловую систему.
    """

    def __init__(
        self,
        path: Path | None = None,
        entries: dict[str, dict[str, Any]] | None = None,
        *,
        base_url: str = "",
    ):
        self.path = path
        self._entries: dict[str, dict[str, Any]] = dict(entries or {})
        # Текущий эндпоинт: замер с другого эндпоинта (иное окно у той же модели) не берётся.
        self.base_url = base_url

    @classmethod
    def load(
        cls, path: Path | None = None, *, base_url: str = ""
    ) -> tuple[ModelWindows, str | None]:
        """Читает файл (по умолчанию :func:`default_path`).

        Нет файла — пустое хранилище; битый файл — пустое хранилище и текст
        предупреждения (файл перезапишется при следующем замере).
        """
        path = default_path() if path is None else path
        try:
            entries = _read(path)
        except FileNotFoundError:
            return cls(path, base_url=base_url), None
        except (OSError, ValueError) as e:
            warning = f"не удалось прочитать {path}: {e} — окна моделей замеряются заново"
            return cls(path, base_url=base_url), warning
        return cls(path, entries, base_url=base_url), None

    def get(self, model: str) -> int | None:
        """Замеренное окно модели (токены) или None.

        Замер другого эндпоинта (поле ``base_url``) не используется: разные
        эндпоинты могут разворачивать модель с разными окнами.
        """
        entry = self._entries.get(model)
        if not isinstance(entry, dict):
            return None
        measured_at = entry.get("base_url")
        if self.base_url and measured_at and measured_at != self.base_url:
            return None
        value = entry.get("context_window")
        return value if isinstance(value, int) and value > 0 else None

    def record(self, result: ProbeResult, *, base_url: str = "") -> None:
        """Запоминает замер и сохраняет файл. Ошибки записи — ``OSError``
        (в памяти замер остаётся)."""
        entry = {
            "context_window": result.window,
            "upper_bound": result.upper_bound,
            "capped": result.capped,
            "probes": result.probes,
            "measured_at": datetime.now().astimezone().replace(microsecond=0).isoformat(),
            "base_url": base_url,
        }
        self._entries[result.model] = entry
        if self.path is None:
            return
        try:
            entries = _read(self.path)
        except FileNotFoundError:
            entries = {}
        except (OSError, ValueError):
            entries = {}  # битый файл перезаписывается
        entries[result.model] = entry
        _write(self.path, entries)


def _read(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != FORMAT_VERSION:
        raise ValueError("неподдерживаемая версия формата")
    models = data.get("models")
    if not isinstance(models, dict):
        raise ValueError("поле models — не объект")
    return {str(k): v for k, v in models.items() if isinstance(v, dict)}


def _write(path: Path, entries: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"version": FORMAT_VERSION, "models": dict(sorted(entries.items()))}
    fd, tmp = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
