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
``make probe``; находки — в конце вывода и в ``.gigacode/evals/probe-<время>.json``.
"""

from __future__ import annotations

import contextlib
import json
import sys
from collections.abc import Iterator, Sequence
from datetime import datetime
from types import MappingProxyType
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
from gigacrew.llm.types import AssistantTurn, Message, ToolSpec, Usage
from gigacrew.project.workspace import Workspace

pytestmark = pytest.mark.live

# Находки проб: модель → что о ней узнали (уходит в JSON-отчёт и сводку).
type _Findings = dict[str, dict[str, Any]]

ENV = load_environment(REPO_ROOT)
MODELS = eval_models(ENV)
MODEL_PARAMS: Sequence[object] = MODELS or [
    pytest.param(
        None,
        marks=pytest.mark.skip(reason="не заданы GIGACREW_EVAL_MODELS и GIGACREW_TEST_MODEL"),
    ),
]

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
SAMPLING = MappingProxyType({
    "temperature=1.0": {"temperature": 1.0},
    "temperature=1.0, top_p=0.95": {"temperature": 1.0, "top_p": 0.95},
    "top_p=0.95": {"top_p": 0.95},
})


@pytest.fixture(scope="module")
def api() -> Iterator[ApiProvider]:
    if not MODELS:
        pytest.skip("не заданы модели")
    config = Config.load(project_root=REPO_ROOT, trusted_dotenv=True, model=MODELS[0])
    if config.provider != "api":
        pytest.skip("пробы шлюза — для API v2 (GIGACREW_PROVIDER=api)")
    try:
        config.require_credentials()
    except RuntimeError as exc:
        pytest.skip(f"{exc} — пробы пропущены")
    provider = ApiProvider(config)
    yield provider
    provider.close()


@pytest.fixture(scope="module")
def findings() -> _Findings:
    """Находки проб модуля: их дополняют тесты, а в отчёт собирает ``_report``."""
    return {}


@pytest.fixture(scope="module", autouse=True)
def _report(request: pytest.FixtureRequest, findings: _Findings) -> Iterator[None]:
    """После проб — JSON с находками и сводка (вне перехвата вывода pytest)."""
    yield
    if not findings:
        return
    directory = report_dir(ENV)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    path = directory / f"probe-{stamp}.json"
    report = {"git": git_revision(), "findings": findings}
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    details = [
        line for model, finding in findings.items() for line in describe_probe(model, finding)
    ]
    summary = "\n".join(["", f"Пробы шлюза: {path}", *details])
    capture = request.config.pluginmanager.getplugin("capturemanager")
    guard = capture.global_and_fixture_disabled() if capture else contextlib.nullcontext()
    with guard:
        sys.stdout.write(f"{summary}\n")


def _post(api: ApiProvider, payload: dict[str, Any]) -> tuple[int, dict[str, Any] | None, str]:
    """Запрос как есть: (HTTP-статус, тело ответа или None, текст ошибки)."""
    resp = api._post_chat(payload)
    if resp.status_code >= 400:
        return resp.status_code, None, resp.text.strip()[:300]
    return resp.status_code, api._json(resp), ""


def _first_step(api: ApiProvider, model: str, messages: list[Message]) -> dict[str, Any]:
    """Первый шаг (запрос с функцией get_weather) — тело ответа; запрос обязан пройти."""
    status, response, error = _post(
        api,
        api._payload(messages, [WEATHER_TOOL], model=model, temperature=1.0),
    )
    assert response is not None, f"первый шаг не прошёл ({status}): {error}"
    return response


def _second_step_payloads(
    api: ApiProvider,
    model: str,
    messages: list[Message],
    turn: AssistantTurn,
    reasoning: list[str],
) -> dict[str, dict[str, Any]]:
    """Запросы второго шага: с ``tools_state_id`` и без, при рассуждениях — и с ними."""
    function_results = [
        Message(role="function", name=call.name, call_id=call.id, content=WEATHER_RESULT)
        for call in turn.message.function_calls
    ]
    stateless = turn.message.model_copy(update={"state_id": None})
    variants = {
        "with_state": api._payload(
            [*messages, turn.message, *function_results],
            [WEATHER_TOOL],
            model=model,
            temperature=1.0,
        ),
        "without_state": api._payload(
            [*messages, stateless, *function_results],
            [WEATHER_TOOL],
            model=model,
            temperature=1.0,
        ),
    }
    if reasoning:
        text = "\n".join(reasoning)
        variants["with_state+reasoning"] = with_reasoning(variants["with_state"], text)
        variants["without_state+reasoning"] = with_reasoning(variants["without_state"], text)
    return variants


def _second_step(api: ApiProvider, payload: dict[str, Any]) -> dict[str, Any]:
    """Второй шаг: HTTP-статус, размер контекста (``input_tokens``) и текст ошибки."""
    status, answer, error = _post(api, payload)
    usage = None if answer is None else Usage.from_raw(answer.get("usage"))
    return {
        "status": status,
        "input_tokens": None if usage is None else usage.prompt_tokens,
        "error": error,
    }


@pytest.mark.parametrize("model", MODEL_PARAMS)
def test_reasoning_and_tool_state(api: ApiProvider, model: str, findings: _Findings) -> None:
    finding = findings.setdefault(model, {})
    messages = [Message(role="user", content=WEATHER_PROMPT)]
    first_step = _first_step(api, model, messages)
    turn = ApiProvider._parse_response(first_step)
    reasoning = reasoning_texts(first_step)
    finding["reasoning_chars"] = sum(len(text) for text in reasoning)
    finding["tool_call"] = turn.wants_tool
    finding["state_id"] = bool(turn.message.state_id)
    if not turn.wants_tool:
        pytest.skip("модель не вызвала функцию — шаги не сравнить")

    variants = _second_step_payloads(api, model, messages, turn, reasoning)
    steps = {name: _second_step(api, payload) for name, payload in variants.items()}
    finding["step2"] = steps
    with_state = steps["with_state"]
    assert with_state["status"] < 400, f"обычный второй шаг не прошёл: {with_state['error']}"


@pytest.mark.parametrize("model", MODEL_PARAMS)
def test_sampling_options(api: ApiProvider, model: str, findings: _Findings) -> None:
    finding = findings.setdefault(model, {})
    prompt = [Message(role="user", content="Ответь одним словом: столица Франции?")]
    outcome: dict[str, dict[str, Any]] = {}
    for name, options in SAMPLING.items():
        payload = api._payload(prompt, None, model=model, temperature=1.0)
        payload["model_options"] = dict(options)
        status, _, error = _post(api, payload)
        outcome[name] = {"status": status, "error": error}
    finding["sampling"] = outcome
    assert any(step["status"] < 400 for step in outcome.values()), outcome


@pytest.mark.parametrize("model", MODEL_PARAMS)
def test_text_stream_reaches_the_end(api: ApiProvider, model: str, findings: _Findings) -> None:
    """Поток без окончания (``done``, ``finish_reason``, ``usage``) gigacrew считает оборванным.

    Здесь — что присылает шлюз на обычный короткий ответ.
    """
    finding = findings.setdefault(model, {})
    prompt = [Message(role="user", content="Ответь одним словом: столица Франции?")]
    turn = api.stream(prompt, model=model, temperature=1.0)
    finding["text_stream_finish"] = turn.finish_reason
    assert turn.finish_reason != "incomplete", "шлюз не присылает окончание текстового потока"


@pytest.mark.parametrize("model", MODEL_PARAMS)
def test_prefix_cache(api: ApiProvider, model: str, findings: _Findings) -> None:
    """Два одинаковых запроса с настоящим системным промптом gigacrew: учтён ли кеш."""
    finding = findings.setdefault(model, {})
    system = compose_system_prompt(SYSTEM_PROMPT, build_project_context(Workspace(REPO_ROOT)))
    messages = [
        Message(role="system", content=system),
        Message(role="user", content="Ответь одним словом: столица Франции?"),
    ]
    usages = []
    for _ in range(2):
        status, response, error = _post(
            api,
            api._payload(messages, None, model=model, temperature=1.0),
        )
        assert response is not None, f"запрос не прошёл ({status}): {error}"
        usages.append(Usage.from_raw(response.get("usage")))
    finding["cache"] = [
        {"input_tokens": usage.prompt_tokens, "cached_tokens": usage.precached_prompt_tokens}
        for usage in usages
    ]
