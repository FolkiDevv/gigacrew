"""Замер окна контекста пробными запросами и хранилище ~/.gigacode/models.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fakes import WindowProvider

from gigacrew.llm.base import LLMError
from gigacrew.llm.context_probe import (
    MAX_CONTEXT_WINDOW,
    MAX_PROBES,
    ProbeStep,
    probe_context_window,
)
from gigacrew.llm.model_windows import ModelWindows, default_path


@pytest.mark.parametrize(
    "window",
    [4_096, 8_192, 32_000, 32_768, 50_000, 128_000, 131_072, 200_000],
)
@pytest.mark.parametrize("hint", [False, True])
def test_result_is_confirmed_and_close(window: int, *, hint: bool) -> None:
    provider = WindowProvider(window, hint=hint)
    steps: list[ProbeStep] = []
    measurement = probe_context_window(provider, "M", on_step=steps.append)
    assert measurement.window <= window  # консервативно: только подтверждённый размер
    assert measurement.window >= window * 0.98 - 512
    assert not measurement.capped
    assert measurement.upper_bound is not None
    assert measurement.window < measurement.upper_bound
    assert measurement.probes == len(steps)
    assert len(steps) <= MAX_PROBES
    assert {model for model, _ in provider.measured} == {"M"}
    # оплачиваются только успешные пробы
    assert measurement.billed_tokens == sum(step.prompt_tokens + 1 for step in steps if step.ok)


@pytest.mark.parametrize("window", [32_768, 128_000, 131_072])
def test_standard_window_costs_one_large_probe(window: int) -> None:
    """Отказы бесплатны; типовое окно находится одной оплачиваемой пробой (+ калибровка)."""
    steps: list[ProbeStep] = []
    probe_context_window(WindowProvider(window), "M", on_step=steps.append)
    assert sum(1 for step in steps if step.ok) == 2


def test_limit_from_error_text_shortcuts_search() -> None:
    steps: list[ProbeStep] = []
    provider = WindowProvider(50_000, hint=True)
    measurement = probe_context_window(provider, "M", on_step=steps.append)
    assert measurement.probes <= 4
    assert sum(1 for step in steps if step.ok) == 2
    assert 49_000 <= measurement.window <= 50_000


def test_window_above_cap_is_capped() -> None:
    measurement = probe_context_window(WindowProvider(1_000_000), "M")
    assert measurement.capped
    assert measurement.upper_bound is None
    assert measurement.probes == 2
    assert MAX_CONTEXT_WINDOW * 0.98 <= measurement.window <= MAX_CONTEXT_WINDOW


def test_capped_rechecks_undershoot_after_calibration() -> None:
    """Калибровка на маленьком запросе промахивается (большие служебные токены).

    Прошедший «потолок» с заметным недобором перепроверяется с новым соотношением.
    """
    steps: list[ProbeStep] = []
    measurement = probe_context_window(
        WindowProvider(1_000_000, overhead=300),
        "M",
        on_step=steps.append,
    )
    assert steps[1].ok
    assert steps[1].prompt_tokens < MAX_CONTEXT_WINDOW * 0.9
    assert measurement.capped
    assert measurement.window >= MAX_CONTEXT_WINDOW * 0.98


def test_capped_recheck_rejected_continues_search() -> None:
    measurement = probe_context_window(WindowProvider(250_000, overhead=300), "M")
    assert not measurement.capped
    assert 245_000 <= measurement.window <= 250_000


def test_rejections_without_status_count_as_too_large() -> None:
    measurement = probe_context_window(WindowProvider(32_768, status=None), "M")
    assert 32_000 <= measurement.window <= 32_768


def test_calibration_failure_aborts() -> None:
    with pytest.raises(LLMError, match="даже запрос"):
        probe_context_window(WindowProvider(500), "M")
    with pytest.raises(LLMError, match="не прошёл: нет модели"):
        probe_context_window(WindowProvider(32_768, error=LLMError("нет модели")), "M")


@pytest.mark.parametrize("window", [8_192, 32_768])
def test_server_that_truncates_silently_stops_the_probe(window: int) -> None:
    """Сервер молча обрезает запрос до окна и считает обрезанное.

    prompt_tokens не растёт вслед за текстом. Доверься замер ему — соотношение символов
    и токенов росло бы с каждой пробой, и пробы раздувались бы в геометрической прогрессии.
    """
    provider = WindowProvider(window, truncate=True, max_chars=MAX_CONTEXT_WINDOW * 4)
    with pytest.raises(LLMError, match="обрезает") as exc_info:
        probe_context_window(provider, "M")
    assert str(window - 1) in str(exc_info.value)
    assert len(provider.sent_chars) == 2  # калибровка и первая же большая проба


# ------------------------------ хранилище ------------------------------ #
def test_store_round_trip_and_merge(tmp_path: Path) -> None:
    path = tmp_path / ".gigacode" / "models.json"
    windows, warning = ModelWindows.load(path)
    assert warning is None
    assert windows.get("A") is None
    assert not path.exists()

    windows.record(probe_context_window(WindowProvider(32_768), "A"), base_url="https://x")
    # другое окно gigacrew успело записать свою модель — она не теряется
    other, _ = ModelWindows.load(path)
    other.record(probe_context_window(WindowProvider(8_192), "B"))
    windows.record(probe_context_window(WindowProvider(16_384), "A"))

    reloaded, warning = ModelWindows.load(path)
    assert warning is None
    window_a = reloaded.get("A")
    assert window_a is not None
    assert 16_000 <= window_a <= 16_384
    window_b = reloaded.get("B")
    assert window_b is not None
    assert 8_000 <= window_b <= 8_192
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["version"] == 1
    entry_a = stored["models"]["A"]
    assert entry_a["capped"] is False
    base_url = entry_a["base_url"]
    assert isinstance(base_url, str)
    assert not base_url
    assert [child.name for child in path.parent.iterdir()] == ["models.json"]  # без .tmp


def test_store_broken_file_warns_and_is_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text("{битый", encoding="utf-8")
    windows, warning = ModelWindows.load(path)
    assert warning
    assert "models.json" in warning
    assert windows.get("A") is None
    windows.record(probe_context_window(WindowProvider(8_192), "A"))
    assert ModelWindows.load(path)[0].get("A") is not None


def test_store_of_an_unknown_format_version_is_not_trusted(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"version": 2, "models": {"A": {"context_window": 1000}}}))
    windows, warning = ModelWindows.load(path)
    assert warning
    assert windows.get("A") is None


@pytest.fixture
def user_home(request: pytest.FixtureRequest) -> Path:
    """Временный домашний каталог из автоматической фикстуры _isolated_home (conftest.py)."""
    home: Path = request.getfixturevalue("_isolated_home")
    return home


def test_store_in_memory_and_default_path(user_home: Path) -> None:
    windows = ModelWindows()
    windows.record(probe_context_window(WindowProvider(8_192), "A"))
    assert windows.get("A") is not None
    assert not (user_home / ".gigacode").exists()
    assert default_path() == user_home / ".gigacode" / "models.json"


def test_window_from_another_endpoint_is_ignored(tmp_path: Path) -> None:
    entries: dict[str, dict[str, Any]] = {
        "M": {"context_window": 32_000, "base_url": "https://external/v1"},
        "Old": {"context_window": 8_000},  # запись без эндпоинта (старый формат)
    }
    assert ModelWindows(entries=entries, base_url="https://external/v1").get("M") == 32_000
    assert ModelWindows(entries=entries, base_url="https://internal/v1").get("M") is None
    assert ModelWindows(entries=entries).get("M") == 32_000
    assert ModelWindows(entries=entries, base_url="https://internal/v1").get("Old") == 8_000


def test_windows_reported_by_api(tmp_path: Path) -> None:
    """Окно из каталога API — без платного замера; замер этого эндпоинта точнее и важнее."""
    path = tmp_path / "models.json"
    entries: dict[str, dict[str, Any]] = {
        "Here": {"context_window": 100_000, "base_url": "https://b"},
        "There": {"context_window": 50_000, "base_url": "https://other"},
        "Old": {"context_window": 8_000},
    }
    windows = ModelWindows(path, entries, base_url="https://b")
    windows.report({"Here": 262_144, "There": 131_072, "Old": 32_768, "New": 65_536, "Bad": 0})
    assert windows.get("Here") == 100_000  # замер этого эндпоинта
    assert windows.get("There") == 131_072  # замер другого — не берётся, есть окно API
    assert windows.get("Old") == 32_768  # окно API важнее замера без эндпоинта
    assert windows.get("New") == 65_536
    assert windows.get("Bad") is None
    assert not path.exists()  # только в памяти
    windows.report({})
    assert windows.get("New") is None
    assert windows.get("Old") == 8_000
