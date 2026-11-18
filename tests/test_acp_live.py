"""Сквозной live-тест ACP-сервера на модели OpenRouter: ход целиком, как его видит T3.

Это тест сервера, а не подкоманды ``gigacrew acp`` (её проверяет ``test_acp_cli.py``
без сети): сервер (:class:`~gigacrew.cli.acp.server.AcpServer`) работает в процессе теста,
провайдер
сессий — :class:`openrouter.OpenRouterProvider` с настройками сессии
(:func:`openrouter.openrouter_config`). Нужны ``GIGACREW_TEST_PROVIDER=openrouter``,
``GIGACREW_OPENROUTER_API_KEY`` и ``GIGACREW_TEST_MODEL`` (окружение или ``.env``
репозитория); без них тест пропускается. Запуск: ``pytest -m live``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any, Final, override

import pytest
from acp_harness import AcpHarness, ProviderFactory, acp_environment
from openrouter import (
    OPENROUTER,
    REPO_ROOT,
    LiveUnavailableError,
    OpenRouterProvider,
    live_provider_kind,
    openrouter_config,
    openrouter_key,
)

from gigacrew.config import Config, load_environment
from gigacrew.llm.base import LLMProvider

# Исключение в любом потоке сервера роняет тест, а не теряется.
pytestmark = [
    pytest.mark.live,
    pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning"),
]

# Сколько ждать OpenRouter, с: каталог моделей, очередь, несколько шагов агента за ход.
LIVE_TIMEOUT: Final = 300.0
CODEWORD: Final = "малахит"
CODEWORD_FILE: Final = "codeword.txt"

type Update = dict[str, Any]


class _OpenRouterFactory(ProviderFactory):
    """Провайдер сессии — OpenRouter с настройками агента из конфигурации сессии."""

    def __init__(self, key: str) -> None:
        super().__init__()
        self._key = key

    @override
    def __call__(self, config: Config) -> LLMProvider:
        self.configs.append(config)
        provider = OpenRouterProvider(openrouter_config(config), self._key)
        self.issued.append(provider)
        return provider


@pytest.fixture
def acp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[AcpHarness]:
    """Сервер с сессиями на OpenRouter; модель сессий — GIGACREW_TEST_MODEL."""
    env = load_environment(REPO_ROOT, trusted_dotenv=True)
    model = env.get("GIGACREW_TEST_MODEL")
    if not model:
        pytest.skip("Не задана GIGACREW_TEST_MODEL — live-тесты пропущены")
    if live_provider_kind(env) != OPENROUTER:
        pytest.skip("live-тест ACP — на OpenRouter (GIGACREW_TEST_PROVIDER=openrouter)")
    try:
        key = openrouter_key(env)
    except LiveUnavailableError as error:
        pytest.skip(f"{error} — live-тесты пропущены")
    acp_environment(monkeypatch, tmp_path_factory.mktemp("process-cwd"))
    monkeypatch.setenv("GIGACREW_MODEL", model)
    with closing(AcpHarness(_OpenRouterFactory(key))) as harness:
        harness.initialize()
        yield harness


def _new_session(acp: AcpHarness, cwd: Path) -> str:
    """``session/new`` в ``cwd``: провайдер загружает каталог OpenRouter по сети — ждём дольше."""
    msg_id = acp.begin("session/new", {"cwd": str(cwd), "mcpServers": []})
    reply = acp.client.response(msg_id, timeout=LIVE_TIMEOUT)
    assert "result" in reply, reply
    session_id = reply["result"]["sessionId"]
    assert isinstance(session_id, str)
    return session_id


def _prompt(acp: AcpHarness, session_id: str, text: str) -> tuple[str, list[Update]]:
    """Ход с запросом ``text``: stopReason и уведомления ``session/update`` этого хода."""
    msg_id = acp.begin(
        "session/prompt",
        {"sessionId": session_id, "prompt": [{"type": "text", "text": text}]},
    )
    reply = acp.client.response(msg_id, timeout=LIVE_TIMEOUT)
    assert "result" in reply, reply
    notifications = acp.client.take_notifications("session/update")
    assert all(params["sessionId"] == session_id for params in notifications)
    return reply["result"]["stopReason"], [params["update"] for params in notifications]


def _answer(updates: list[Update]) -> str:
    return "".join(
        update["content"]["text"]
        for update in updates
        if update["sessionUpdate"] == "agent_message_chunk"
    )


def test_live_one_word_answer(acp: AcpHarness, project: Path) -> None:
    """Простой ход: текст ответа приходит кусками agent_message_chunk, ход — end_turn."""
    session_id = _new_session(acp, project.resolve())

    stop_reason, updates = _prompt(
        acp,
        session_id,
        "Ответь одним словом, без пояснений: какого цвета ясное небо днём?",
    )

    assert stop_reason == "end_turn"
    assert _answer(updates).strip()


def test_live_tool_call(acp: AcpHarness, project: Path) -> None:
    """Ход с инструментом: tool_call и tool_call_update completed, ответ — из файла."""
    root = project.resolve()
    (root / CODEWORD_FILE).write_text(f"Кодовое слово: {CODEWORD}\n", encoding="utf-8")
    session_id = _new_session(acp, root)

    stop_reason, updates = _prompt(
        acp,
        session_id,
        f"Прочитай файл {CODEWORD_FILE} в проекте и ответь одним словом — кодовым словом из него.",
    )

    assert stop_reason == "end_turn"
    started = {update["toolCallId"] for update in updates if update["sessionUpdate"] == "tool_call"}
    completed = {
        update["toolCallId"]
        for update in updates
        if update["sessionUpdate"] == "tool_call_update" and update.get("status") == "completed"
    }
    assert started & completed, updates
    assert CODEWORD in _answer(updates).lower()
