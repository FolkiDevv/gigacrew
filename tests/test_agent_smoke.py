"""Сквозной smoke-тест агента на реальной мини-задаче (живое API).

Сценарий: создать файл → отредактировать его → выполнить команду.
Проверяем итоговое состояние файловой системы, т.к. формулировки модели
недетерминированы.

Помечен маркером ``live``; пропускается без ключа.
"""

from __future__ import annotations

import subprocess
import sys
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
    # файл делает то, что просили: запуск печатает приветствие
    run = subprocess.run(
        [sys.executable, str(target)], capture_output=True, text=True, timeout=30, cwd=tmp_path
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "Привет, Мир!", f"вывод: {run.stdout!r}\n{text}"
    # и агент сам выполнил команду: в истории — run_shell и её вывод
    shell = [
        m for m in agent.conversation.messages if m.role == "function" and m.name == "run_shell"
    ]
    assert shell, "агент не запускал команду"
    assert any("Привет, Мир!" in m.content for m in shell), [m.content for m in shell]


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
    assert text == "version = 2.0.0\n", f"ожидалась точечная правка версии:\n{text}"
    calls = [c.name for m in agent.conversation.messages for c in m.function_calls]
    assert "read_file" in calls and "edit_file" in calls, calls


def test_agent_continues_after_tool_call(live_config):
    """Регрессия: запрос после вызова инструмента на настоящем системном промпте и наборе
    инструментов. API v1 отвечал на него 422 «Function calls count does not match function
    results count»: модель делала несколько вызовов, а v1 показывал только первый
    (на упрощённом запросе не воспроизводилось).

    Только чтение проекта: правки и команды без подтверждения отклоняются.
    """
    cfg = replace(live_config, max_steps=3, ty=False)
    provider = ApiProvider(cfg)
    agent = Agent(provider, build_default_registry(), cfg)
    try:
        agent.run_turn("расскажи об этом проекте")
    finally:
        provider.close()

    roles = [m.role for m in agent.conversation.messages]
    assert "function" in roles, "модель должна была вызвать инструмент"
    # после результата инструмента модель ответила — запрос с ним принят
    assert "assistant" in roles[roles.index("function") + 1 :]
