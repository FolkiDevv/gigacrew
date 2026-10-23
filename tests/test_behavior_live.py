"""Поведенческие сценарии на живых моделях (eval).

Каждый сценарий (``behavior.SCENARIOS``) прогоняется на каждой модели несколько раз в
свежей копии учебного проекта: выходит ли модель из плана сама, исправляет ли план по
замечаниям, разведывает ли код суб-агентами — и сколько это стоит.

Запуск — ``make eval`` (только эти тесты; дорого). Настройки (окружение или ``.env``):

* ``GIGACREW_EVAL_MODELS`` — модели через запятую (по умолчанию ``GIGACREW_TEST_MODEL``);
* ``GIGACREW_EVAL_RUNS`` — повторов каждого сценария (по умолчанию 3);
* ``GIGACREW_EVAL_LABEL`` — метка отчёта (``baseline-t02``…);
* ``GIGACREW_EVAL_DIR`` — папка отчётов (по умолчанию ``.gigacode/evals``);
* ``GIGACREW_EVAL_TIMEOUT`` — предел одного прогона, с (по умолчанию 1800);
* ``GIGACREW_TEMPERATURE`` и прочие настройки агента — как у ``gigacrew``.

Отдельные сценарии — ``-k``: ``make eval ARGS="-k plan_exit"``. Окна моделей берутся из
замеров ``~/.gigacode/models.json`` (незамеренная модель работает с окном 32K — замерьте
заранее: ``gigacrew --test-context <модель>``).

Тест падает, только если нарушен инвариант harness'а (история вызовов, запрет правок в
режиме плана) или прогон сорвался (ошибка API, таймаут). Поведение моделей — в отчёте:
таблица в конце вывода, JSON и истории прогонов — в папке отчётов; сравнение —
``uv run python tests/behavior.py compare <было>.json <стало>.json``.
"""

from __future__ import annotations

import contextlib
import itertools
import sys
from collections.abc import Iterator
from pathlib import Path

import behavior
import pytest

from gigacrew.config import Config, load_environment
from gigacrew.llm.api import ApiProvider
from gigacrew.llm.model_windows import ModelWindows

pytestmark = [pytest.mark.live, pytest.mark.eval]

ENV = load_environment(behavior.REPO_ROOT)
MODELS = behavior.eval_models(ENV)
RUNS = behavior.env_int(ENV, behavior.EVAL_RUNS_ENV, behavior.DEFAULT_RUNS)
TIMEOUT = behavior.env_int(ENV, behavior.EVAL_TIMEOUT_ENV, behavior.DEFAULT_TIMEOUT)

CASES = tuple(
    pytest.param(model, scenario, run, id=f"{model}-{scenario.id}-r{run}")
    for model, scenario, run in itertools.product(MODELS, behavior.SCENARIOS, range(1, RUNS + 1))
) or (
    pytest.param(
        None,
        None,
        0,
        id="no-models",
        marks=pytest.mark.skip(reason="не заданы GIGACREW_EVAL_MODELS и GIGACREW_TEST_MODEL"),
    ),
)

# Сессия eval (одна на прогон pytest) — для таблицы итогов в конце модуля.
_SESSION_KEY = pytest.StashKey[behavior.EvalSession]()


@pytest.fixture(scope="session")
def eval_session(request: pytest.FixtureRequest) -> Iterator[behavior.EvalSession]:
    """Конфиг, провайдер и окна моделей — общие для всех прогонов.

    Окна читаются до того, как фикстура ``_isolated_home`` подменит домашнюю папку
    (фикстуры сессии создаются раньше фикстур теста).
    """
    config = Config.load(project_root=behavior.REPO_ROOT, trusted_dotenv=True, model=MODELS[0])
    if config.provider != "api":
        pytest.skip("eval — для API v2 (GIGACREW_PROVIDER=api)")
    try:
        config.require_credentials()
    except RuntimeError as error:
        pytest.skip(f"{error} — eval пропущен")
    windows, _ = ModelWindows.load(base_url=config.base_url)
    provider = ApiProvider(config)
    session = behavior.EvalSession(
        config=config,
        provider=provider,
        windows=windows,
        directory=behavior.report_dir(ENV),
        models=MODELS,
        runs=RUNS,
        label=(ENV.get(behavior.EVAL_LABEL_ENV) or "").strip(),
    )
    request.config.stash.setdefault(_SESSION_KEY, session)
    yield session
    provider.close()


@pytest.fixture(scope="module", autouse=True)
def _summary(request: pytest.FixtureRequest) -> Iterator[None]:
    """После всех прогонов — таблица итогов (вне перехвата вывода pytest)."""
    yield
    session = request.config.stash.get(_SESSION_KEY, None)
    if session is None or not session.records:
        return
    lines = ["", f"Отчёт eval: {session.path}", *behavior.summary_table(session.records)]
    report = "\n".join(lines)
    capture = request.config.pluginmanager.getplugin("capturemanager")
    guard = capture.global_and_fixture_disabled() if capture else contextlib.nullcontext()
    with guard:
        sys.stdout.write(f"{report}\n")


@pytest.mark.timeout(TIMEOUT)
@pytest.mark.parametrize(("model", "scenario", "run"), CASES)
def test_scenario(
    model: str,
    scenario: behavior.Scenario,
    run: int,
    eval_session: behavior.EvalSession,
    tmp_path: Path,
) -> None:
    root = behavior.make_project(tmp_path / "shop")
    outcome = behavior.run_scenario(
        scenario,
        provider=eval_session.provider,
        config=eval_session.config_for(model, scenario, root),
        windows=eval_session.windows,
    )
    record = eval_session.add(outcome, run=run)
    if outcome.error is not None:
        raise outcome.error
    problems = behavior.invariant_errors(scenario, record)
    assert not problems, "; ".join(problems)
