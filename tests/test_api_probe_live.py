"""Пробы шлюза модели (live): что он делает с рассуждениями и параметрами генерации.

Ничего не утверждают о поведении шлюза — собирают факты для этапа 0 плана улучшений
промпта и печатают их в конце вывода:

1. отдаёт ли шлюз рассуждения модели (``role: "reasoning"``) в ответе с вызовом функции;
2. хранит ли он их сам между шагами: ``input_tokens`` одного и того же второго шага с
   ``tools_state_id`` и без него — если с ним заметно больше, шлюз подставляет скрытое
   состояние (рассуждения) сам, и возвращать их gigacrew не нужно;
3. принимает ли рассуждения в запросе (сообщение ``role: "reasoning"`` перед ответом
   ассистента) и учитывает ли их (рост ``input_tokens``);
4. принимает ли ``top_p`` в ``model_options`` (принято без ошибки ≠ учтено);
5. сообщает ли о кешировании префикса (``precached_prompt_tokens`` на повторе запроса);
6. доходит ли текстовый поток до окончания ответа: иначе gigacrew считал бы каждый ответ
   оборванным и повторял запрос.

Модели — как у eval: ``GIGACREW_EVAL_MODELS``, иначе ``GIGACREW_TEST_MODEL``. Запуск —
``make probe``; находки — в конце вывода и в ``.gigacrew/evals/probe-<время>.json``.
"""

from __future__ import annotations

import contextlib
import json
from datetime import datetime
from typing import Any

import pytest
from behavior import (
    REPO_ROOT,
    describe_probe,
    eval_models,
    git_revision,
    reasoning_texts,
    report_dir,
    with_reasoning,
)

from gigacrew.agent.prompts import SYSTEM_PROMPT, build_project_context, compose_system_prompt
from gigacrew.config import Config, load_environment
from gigacrew.llm.api import ApiProvider
from gigacrew.llm.types import Message, ToolSpec, Usage
from gigacrew.project.workspace import Workspace

pytestmark = pytest.mark.live

ENV = load_environment(REPO_ROOT)
MODELS = eval_models(ENV)
MODEL_PARAMS = MODELS or [
    pytest.param(
        None, marks=pytest.mark.skip(reason="не заданы GIGACREW_EVAL_MODELS и GIGACREW_TEST_MODEL")
    )
]
FINDINGS: dict[str, dict[str, Any]] = {}

WEATHER_TOOL = ToolSpec(
    name="get_weather",
    description="Возвращает текущую погоду в указанном городе",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string", "description": "город"}},
        "required": ["city"],
    },
)
WEATHER_PROMPT = (
    "Какая сейчас погода в Москве и стоит ли брать зонт? Вызови get_weather, потом ответь."
)
WEATHER_RESULT = '{"city": "Москва", "temperature_c": 17, "condition": "дождь"}'
SAMPLING = {
    "temperature=1.0": {"temperature": 1.0},
    "temperature=1.0, top_p=0.95": {"temperature": 1.0, "top_p": 0.95},
    "top_p=0.95": {"top_p": 0.95},
}


@pytest.fixture(scope="module")
def api():
    if not MODELS:
        pytest.skip("не заданы модели")
    config = Config.load(project_root=REPO_ROOT, model=MODELS[0])
    if config.provider != "api":
        pytest.skip("пробы шлюза — для API v2 (GIGACREW_PROVIDER=api)")
    try:
        config.require_credentials()
    except RuntimeError as e:
        pytest.skip(f"{e} — пробы пропущены")
    provider = ApiProvider(config)
    yield provider
    provider.close()


@pytest.fixture(scope="module", autouse=True)
def _report(request):
    """После проб — JSON с находками и сводка (вне перехвата вывода pytest)."""
    yield
    if not FINDINGS:
        return
    directory = report_dir(ENV)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"probe-{datetime.now():%Y%m%d-%H%M%S}.json"
    report = {"git": git_revision(), "findings": FINDINGS}
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["", f"Пробы шлюза: {path}"]
    for model, finding in FINDINGS.items():
        lines += describe_probe(model, finding)
    capture = request.config.pluginmanager.getplugin("capturemanager")
    guard = capture.global_and_fixture_disabled() if capture else contextlib.nullcontext()
    with guard:
        print("\n".join(lines))


def _post(api: ApiProvider, payload: dict[str, Any]) -> tuple[int, dict[str, Any] | None, str]:
    """Запрос как есть: (HTTP-статус, тело ответа или None, текст ошибки)."""
    resp = api._post_chat(payload)
    if resp.status_code >= 400:
        return resp.status_code, None, resp.text.strip()[:300]
    return resp.status_code, api._json(resp), ""


@pytest.mark.parametrize("model", MODEL_PARAMS)
def test_reasoning_and_tool_state(api, model):
    finding = FINDINGS.setdefault(model, {})
    messages = [Message(role="user", content=WEATHER_PROMPT)]
    status, data, error = _post(
        api, api._payload(messages, [WEATHER_TOOL], model=model, temperature=1.0)
    )
    assert data is not None, f"первый шаг не прошёл ({status}): {error}"
    turn = ApiProvider._parse_response(data)
    reasoning = reasoning_texts(data)
    finding["reasoning_chars"] = sum(len(text) for text in reasoning)
    finding["tool_call"] = turn.wants_tool
    finding["state_id"] = bool(turn.message.state_id)
    if not turn.wants_tool:
        pytest.skip("модель не вызвала функцию — шаги не сравнить")

    results = [
        Message(role="function", name=call.name, call_id=call.id, content=WEATHER_RESULT)
        for call in turn.message.function_calls
    ]
    stateless = turn.message.model_copy(update={"state_id": None})
    variants = {
        "with_state": api._payload(
            [*messages, turn.message, *results], [WEATHER_TOOL], model=model, temperature=1.0
        ),
        "without_state": api._payload(
            [*messages, stateless, *results], [WEATHER_TOOL], model=model, temperature=1.0
        ),
    }
    if reasoning:
        text = "\n".join(reasoning)
        variants["with_state+reasoning"] = with_reasoning(variants["with_state"], text)
        variants["without_state+reasoning"] = with_reasoning(variants["without_state"], text)
    steps = {}
    for name, payload in variants.items():
        status, answer, error = _post(api, payload)
        usage = Usage.from_raw(answer.get("usage")) if answer is not None else None
        steps[name] = {
            "status": status,
            "input_tokens": usage.prompt_tokens if usage is not None else None,
            "error": error,
        }
    finding["step2"] = steps
    assert steps["with_state"]["status"] < 400, (
        f"обычный второй шаг не прошёл: {steps['with_state']['error']}"
    )


@pytest.mark.parametrize("model", MODEL_PARAMS)
def test_sampling_options(api, model):
    finding = FINDINGS.setdefault(model, {})
    prompt = [Message(role="user", content="Ответь одним словом: столица Франции?")]
    outcome = {}
    for name, options in SAMPLING.items():
        payload = api._payload(prompt, None, model=model, temperature=1.0)
        payload["model_options"] = dict(options)
        status, _, error = _post(api, payload)
        outcome[name] = {"status": status, "error": error}
    finding["sampling"] = outcome
    assert any(step["status"] < 400 for step in outcome.values()), outcome


@pytest.mark.parametrize("model", MODEL_PARAMS)
def test_text_stream_reaches_the_end(api, model):
    """Поток без окончания (``done``, ``finish_reason``, ``usage``) gigacrew считает
    оборванным. Здесь — что присылает шлюз на обычный короткий ответ."""
    finding = FINDINGS.setdefault(model, {})
    prompt = [Message(role="user", content="Ответь одним словом: столица Франции?")]
    turn = api.stream(prompt, model=model, temperature=1.0)
    finding["text_stream_finish"] = turn.finish_reason
    assert turn.finish_reason != "incomplete", "шлюз не присылает окончание текстового потока"


@pytest.mark.parametrize("model", MODEL_PARAMS)
def test_prefix_cache(api, model):
    """Два одинаковых запроса с настоящим системным промптом gigacrew: учтён ли кеш."""
    finding = FINDINGS.setdefault(model, {})
    system = compose_system_prompt(SYSTEM_PROMPT, build_project_context(Workspace(REPO_ROOT)))
    messages = [
        Message(role="system", content=system),
        Message(role="user", content="Ответь одним словом: столица Франции?"),
    ]
    usages = []
    for _ in range(2):
        status, data, error = _post(api, api._payload(messages, None, model=model, temperature=1.0))
        assert data is not None, f"запрос не прошёл ({status}): {error}"
        usages.append(Usage.from_raw(data.get("usage")))
    finding["cache"] = [
        {"input_tokens": usage.prompt_tokens, "cached_tokens": usage.precached_prompt_tokens}
        for usage in usages
    ]
