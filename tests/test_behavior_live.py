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

import pytest
from behavior import (
    DEFAULT_RUNS,
    DEFAULT_TIMEOUT,
    EVAL_LABEL_ENV,
    EVAL_RUNS_ENV,
    EVAL_TIMEOUT_ENV,
    REPO_ROOT,
    SCENARIOS,
    EvalSession,
    env_int,
    eval_models,
    invariant_errors,
    make_project,
    report_dir,
    run_scenario,
    summary_table,
)

from gigacrew.config import Config, load_environment
from gigacrew.llm.api import ApiProvider
from gigacrew.llm.model_windows import ModelWindows

pytestmark = [pytest.mark.live, pytest.mark.eval]

ENV = load_environment(REPO_ROOT)
MODELS = eval_models(ENV)
RUNS = env_int(ENV, EVAL_RUNS_ENV, DEFAULT_RUNS)
TIMEOUT = env_int(ENV, EVAL_TIMEOUT_ENV, DEFAULT_TIMEOUT)

CASES = [
    pytest.param(model, scenario, run, id=f"{model}-{scenario.id}-r{run}")
    for model in MODELS
    for scenario in SCENARIOS
    for run in range(1, RUNS + 1)
] or [
    pytest.param(
        None,
        None,
        0,
        id="no-models",
        marks=pytest.mark.skip(reason="не заданы GIGACREW_EVAL_MODELS и GIGACREW_TEST_MODEL"),
    ),
]

_SESSIONS: list[EvalSession] = []


@pytest.fixture(scope="session")
def eval_session():
    """Конфиг, провайдер и окна моделей — общие для всех прогонов.

    Окна читаются до того, как фикстура ``_isolated_home`` подменит домашнюю папку
    (фикстуры сессии создаются раньше фикстур теста).
    """
    config = Config.load(project_root=REPO_ROOT, trusted_dotenv=True, model=MODELS[0])
    if config.provider != "api":
        pytest.skip("eval — для API v2 (GIGACREW_PROVIDER=api)")
    try:
        config.require_credentials()
    except RuntimeError as e:
        pytest.skip(f"{e} — eval пропущен")
    windows, _ = ModelWindows.load(base_url=config.base_url)
    provider = ApiProvider(config)
    session = EvalSession(
        config=config,
        provider=provider,
        windows=windows,
        directory=report_dir(ENV),
        models=MODELS,
        runs=RUNS,
        label=(ENV.get(EVAL_LABEL_ENV) or "").strip(),
    )
    _SESSIONS.append(session)
    yield session
    provider.close()


@pytest.fixture(scope="module", autouse=True)
def _summary(request):
    """После всех прогонов — таблица итогов (вне перехвата вывода pytest)."""
    yield
    if not _SESSIONS or not _SESSIONS[0].records:
        return
    session = _SESSIONS[0]
    lines = ["", f"Отчёт eval: {session.path}", *summary_table(session.records)]
    capture = request.config.pluginmanager.getplugin("capturemanager")
    guard = capture.global_and_fixture_disabled() if capture else contextlib.nullcontext()
    with guard:
        print("\n".join(lines))


@pytest.mark.timeout(TIMEOUT)
@pytest.mark.parametrize(("model", "scenario", "run"), CASES)
def test_scenario(model, scenario, run, eval_session, tmp_path):
    root = make_project(tmp_path / "shop")
    result = run_scenario(
        scenario,
        provider=eval_session.provider,
        config=eval_session.config_for(model, scenario, root),
        windows=eval_session.windows,
    )
    record = eval_session.add(result, run=run)
    if result.error is not None:
        raise result.error
    problems = invariant_errors(scenario, record)
    assert not problems, "; ".join(problems)
