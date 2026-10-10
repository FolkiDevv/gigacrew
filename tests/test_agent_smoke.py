"""Сквозной smoke-тест агента на реальной мини-задаче (живое API).

Сценарий: создать файл → отредактировать его → выполнить команду.
Проверяем итоговое состояние файловой системы, т.к. формулировки модели
недетерминированы.

Помечен маркером ``live``; пропускается без ключа.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from gigacrew.agent.loop import Agent
from gigacrew.llm.api import ApiProvider
from gigacrew.tools.base import build_default_registry
from gigacrew.ui.console import Console

pytestmark = pytest.mark.live


def test_end_to_end_mini_task(live_config, tmp_path):
    # конфиг для temp-проекта с авто-подтверждением (без интерактива)
    cfg = replace(
        live_config,
        project_root=tmp_path,
        auto_approve=True,
        max_steps=14,
    )
    provider = ApiProvider(cfg)
    ui = Console(no_color=True)
    agent = Agent(provider, build_default_registry(), cfg, ui)

    try:
        agent.run_turn(
            "Создай файл greet.py с функцией greet(name), которая возвращает строку "
            "'Привет, <name>!'. В конце файла добавь print(greet('Мир')). "
            "Затем запусти его командой: python3 greet.py и покажи вывод. "
            "Используй инструменты write_file и run_shell."
        )
    finally:
        provider.close()

    target = tmp_path / "greet.py"
    assert target.exists(), "агент должен был создать greet.py"
    text = target.read_text(encoding="utf-8")
    assert "def greet" in text, f"в файле нет функции greet:\n{text}"
    assert "print" in text


def test_agent_reads_and_edits_existing_file(live_config, tmp_path):
    """Агент читает существующий файл и вносит точечную правку."""
    (tmp_path / "version.txt").write_text("version = 1.0.0\n", encoding="utf-8")
    cfg = replace(live_config, project_root=tmp_path, auto_approve=True, max_steps=12)
    provider = ApiProvider(cfg)
    ui = Console(no_color=True)
    agent = Agent(provider, build_default_registry(), cfg, ui)
    try:
        agent.run_turn(
            "В файле version.txt замени версию 1.0.0 на 2.0.0. "
            "Сначала прочитай файл, потом используй edit_file."
        )
    finally:
        provider.close()

    text = (tmp_path / "version.txt").read_text(encoding="utf-8")
    assert "2.0.0" in text, f"версия не обновлена:\n{text}"
