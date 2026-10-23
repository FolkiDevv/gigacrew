"""Замеренные окна контекста моделей: ``~/.gigacode/models.json``.

Файл общий для всех проектов пользователя (окно — свойство модели, а не проекта),
ключ — имя модели. Пишется атомарно (временный файл + ``os.replace``) и перед
записью перечитывается, чтобы не затереть замер из другого окна gigacrew.

Формат::

    {
        "version": 1,
        "models": {
            "model-large": {
                "context_window": 126998,
                "upper_bound": 128319,
                "capped": false,
                "probes": 8,
                "measured_at": "2026-10-09T12:00:00+03:00",
                "base_url": "https://...",
            }
        },
    }
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from gigacrew.llm.context_probe import ProbeResult

FORMAT_VERSION = 1
USER_DIR_NAME = ".gigacode"
FILE_NAME = "models.json"


def default_path() -> Path:
    """Файл замеров по умолчанию: ``~/.gigacode/models.json``."""
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
        """Хранилище с начальными записями ``entries`` для эндпоинта ``base_url``."""
        self.path = path
        self._entries: dict[str, dict[str, Any]] = dict(entries or {})
        # Текущий эндпоинт: замер с другого эндпоинта (иное окно у той же модели) не берётся.
        self.base_url = base_url
        # Окна, которые сообщил API (каталог моделей бэкенда): только в памяти.
        self._reported: dict[str, int] = {}

    @classmethod
    def load(
        cls,
        path: Path | None = None,
        *,
        base_url: str = "",
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
        except (OSError, ValueError) as exc:
            warning = f"не удалось прочитать {path}: {exc} — окна моделей замеряются заново"
            return cls(path, base_url=base_url), warning
        return cls(path, entries, base_url=base_url), None

    def get(self, model: str) -> int | None:
        """Окно модели (токены) или None.

        Замер этого эндпоинта важнее окна, которое сообщил API (:meth:`report`), а оно —
        старого замера без эндпоинта. Замер другого эндпоинта (поле ``base_url``) не
        используется: разные эндпоинты могут разворачивать модель с разными окнами.
        """
        legacy: int | None = None  # замер без поля base_url (до его появления)
        entry = self._entries.get(model)
        if isinstance(entry, dict):
            window = entry.get("context_window")
            measured_at = entry.get("base_url")
            if isinstance(window, int) and window > 0:
                if not measured_at:
                    legacy = window
                elif not self.base_url or measured_at == self.base_url:
                    return window
        return self._reported.get(model, legacy)

    def report(self, windows: Mapping[str, int | None]) -> None:
        """Окна моделей, которые сообщил API: только в памяти, в файл не пишутся.

        Заменяют прежние; не положительные числа и None пропускаются.
        """
        self._reported = {
            str(name): window
            for name, window in windows.items()
            if isinstance(window, int) and not isinstance(window, bool) and window > 0
        }

    def record(self, measurement: ProbeResult, *, base_url: str = "") -> None:
        """Запоминает замер и сохраняет файл.

        Ошибки записи — ``OSError`` (в памяти замер остаётся).
        """
        entry = {
            "context_window": measurement.window,
            "upper_bound": measurement.upper_bound,
            "capped": measurement.capped,
            "probes": measurement.probes,
            "measured_at": datetime.now().astimezone().replace(microsecond=0).isoformat(),
            "base_url": base_url,
        }
        self._entries[measurement.model] = entry
        if self.path is None:
            return
        try:
            entries = _read(self.path)
        except FileNotFoundError:
            entries = {}
        except (OSError, ValueError):
            entries = {}  # битый файл перезаписывается
        entries[measurement.model] = entry
        _write(self.path, entries)


def _read(path: Path) -> dict[str, dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("version") != FORMAT_VERSION:
        raise ValueError("неподдерживаемая версия формата")
    models = document.get("models")
    if not isinstance(models, dict):
        # ValueError, а не TypeError: вызывающие считают файл битым по ValueError.
        raise ValueError("поле models — не объект")  # noqa: TRY004
    return {str(name): entry for name, entry in models.items() if isinstance(entry, dict)}


def _write(path: Path, entries: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"version": FORMAT_VERSION, "models": dict(sorted(entries.items()))}
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2)
        Path(tmp_name).replace(path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
