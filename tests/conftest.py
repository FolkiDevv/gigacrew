"""Общие фикстуры для тестов gigacrew."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gigacrew.cli import app as cli_app  # noqa: E402
from gigacrew.config import Config, load_environment  # noqa: E402
from gigacrew.llm.base import LLMProvider  # noqa: E402
from gigacrew.project.workspace import Workspace  # noqa: E402
from gigacrew.tools.base import ToolContext  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_home(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """~/.gigacode (замеры окон моделей) — во временной папке, не у пользователя."""
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return home


@pytest.fixture
def user_home(request: pytest.FixtureRequest) -> Path:
    """Временный домашний каталог из автоматической фикстуры ``_isolated_home``."""
    home: Path = request.getfixturevalue("_isolated_home")
    return home


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Пустой временный «проект»."""
    return tmp_path


@pytest.fixture
def config(project: Path) -> Config:
    return Config(project_root=project, access_key="dummy")


@pytest.fixture
def ctx(project: Path) -> ToolContext:
    return ToolContext(workspace=Workspace(project))


@pytest.fixture(scope="session")
def live_config() -> Config:
    """Конфиг для live-тестов (окружение + .env репозитория, OAuth или mTLS).

    Пропускает тест, если не заданы реквизиты, адрес API или GIGACREW_TEST_MODEL.
    """
    env = load_environment(REPO_ROOT)
    model = env.get("GIGACREW_TEST_MODEL")
    if not model:
        pytest.skip("Не задана GIGACREW_TEST_MODEL — live-тесты пропущены")
    cfg = Config.load(project_root=REPO_ROOT, trusted_dotenv=True, model=model)
    if cfg.provider != "api":
        pytest.skip("live-тесты — для API v2 (GIGACREW_PROVIDER=api)")
    try:
        cfg.require_credentials()
    except RuntimeError as error:
        pytest.skip(f"{error} — live-тесты пропущены")
    return cfg


@pytest.fixture
def oneshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[LLMProvider], None]:
    """Окружение для ``app.main(["-p", ...])``; вызов ``oneshot(provider)`` подменяет провайдера."""
    monkeypatch.chdir(tmp_path)
    # Настройки пользователя из окружения (любые GIGACREW_*/GIGACODE_*, и новые тоже) не
    # должны менять поведение теста — убираем все, а не перечисленные.
    user_settings = [
        env_name for env_name in os.environ if env_name.startswith(("GIGACREW_", "GIGACODE_"))
    ]
    for name in user_settings:
        monkeypatch.delenv(name)
    monkeypatch.setenv("GIGACREW_ACCESS_KEY", "dummy")
    monkeypatch.setenv("GIGACREW_AUTH_URL", "https://auth.test/oauth")
    monkeypatch.setenv("GIGACREW_API_URL", "https://api.test/v1")
    monkeypatch.setenv("GIGACREW_MODEL", "M-default")

    def use(provider: LLMProvider) -> None:
        monkeypatch.setattr(cli_app, "_make_provider", lambda _cfg: provider)

    return use
