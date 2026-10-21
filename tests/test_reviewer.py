"""Модель-рецензент изменяющих команд (режим «авто»): запрос, разбор ответа, проверка
модели при старте и решения в агентном цикле."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fakes import RecordingEvents, ScriptedProvider, cut_turn, text_turn, tool_turn

from gigacrew.agent.decisions import DecisionMemory
from gigacrew.agent.events import Approval, ConfirmResult, Review
from gigacrew.agent.loop import Agent
from gigacrew.agent.prompts import REVIEWER_PROMPT
from gigacrew.agent.reviewer import (
    DISABLED_NOTE,
    MAX_FAILURES,
    MAX_REVIEW_COMMAND,
    Reviewer,
    ReviewError,
    check_reviewer,
    parse_review,
    recent_requests,
    review_command,
    review_request,
)
from gigacrew.config import Config
from gigacrew.llm.base import LLMError
from gigacrew.llm.types import Message, Usage
from gigacrew.permissions import PermissionMode
from gigacrew.safety import classify
from gigacrew.tools.base import build_default_registry

ALLOW = '{"allow": true, "reason": "сборка проекта по просьбе пользователя"}'
DENY = '{"allow": false, "reason": "об этом не просили"}'


def _script(tmp_path: Path, name: str = "build.sh") -> str:
    path = tmp_path / name
    path.write_text("#!/bin/sh\necho built\n")
    path.chmod(0o755)
    return f"./{name}"


def _agent(provider, tmp_path: Path, *, events=None, **kw) -> Agent:
    cfg = Config(
        access_key="x", project_root=tmp_path, stream=False, model="M-chat", **kw.pop("cfg", {})
    )
    return Agent(provider, build_default_registry(), cfg, events or RecordingEvents(), **kw)


def _run(command: str, **args) -> object:
    return tool_turn("run_shell", {"command": command, **args})


# ------------------------------- разбор ответа ------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        ALLOW,
        f"```json\n{ALLOW}\n```",
        f"Проверил команду. {ALLOW} Удачи.",
        '{"allow": "true", "reason": "сборка проекта по просьбе пользователя"}',
        '{"note": {"x": 1}} ' + ALLOW,  # объект без allow пропускается
    ],
)
def test_parse_review_accepts_json_in_text(text):
    assert parse_review(text) == (True, "сборка проекта по просьбе пользователя")


def test_parse_review_deny_and_missing_reason():
    assert parse_review('{"allow": false}') == (False, "")
    assert parse_review(DENY) == (False, "об этом не просили")


@pytest.mark.parametrize("text", ["", "сводка", '{"allow": "может быть"}', "{broken", "[true]"])
def test_parse_review_rejects_garbage(text):
    with pytest.raises(ReviewError, match="не разобран"):
        parse_review(text)


# ---------------------------------- запрос ---------------------------------- #
def test_request_has_user_words_command_and_facts_only(tmp_path):
    command = "pip install requests  # ```"
    verdict = classify(command, root=tmp_path)
    text = review_request(command, verdict, ["поставь requests", "и проверь"])
    assert "1. поставь requests" in text and "2. и проверь" in text
    assert command in text
    assert "````" in text  # ограждение длиннее кавычек внутри команды
    assert "WRITE" in text and verdict.reason in text
    assert "Working directory (relative to the project root): ." in text


def test_request_marks_unverified_commands(tmp_path):
    verdict = classify("frobnicate --all", root=tmp_path)
    assert verdict.level is None
    text = review_request("frobnicate --all", verdict, [])
    assert "UNVERIFIED" in text and "(none)" in text


def test_recent_requests_skip_harness_messages():
    messages = [Message(role="user", content=f"запрос {i}") for i in range(5)] + [
        Message(role="user", content="служебное", meta=True),
        Message(role="assistant"),
    ]
    assert recent_requests(messages) == ["запрос 2", "запрос 3", "запрос 4"]


def test_review_command_calls_model_without_tools(tmp_path):
    provider = ScriptedProvider(summaries=[text_turn(ALLOW, Usage(prompt_tokens=7))])
    used: list[Usage] = []
    verdict = classify("pip install requests", root=tmp_path)
    review = review_command(
        provider,
        command="pip install requests",
        verdict=verdict,
        requests=["поставь requests"],
        model="R",
        on_usage=used.append,
    )
    assert review == Review("R", True, "сборка проекта по просьбе пользователя")
    (request,) = provider.summary_requests
    assert request["model"] == "R" and request["temperature"] == 0.0
    system, user = request["messages"]
    assert system.content == REVIEWER_PROMPT and "поставь requests" in user.content
    assert used[0].prompt_tokens == 7


def test_review_command_cut_answer_is_an_error(tmp_path):
    provider = ScriptedProvider(summaries=[cut_turn(ALLOW)])
    with pytest.raises(ReviewError, match="не закончен"):
        review_command(
            provider,
            command="ls",
            verdict=classify("ls", root=tmp_path),
            requests=[],
            model="R",
        )


# ------------------------------ модель рецензента ------------------------------ #
class _Listing(ScriptedProvider):
    def __init__(self, models):
        super().__init__()
        self._models = models
        self.listed = 0

    def list_models(self):
        self.listed += 1
        if isinstance(self._models, BaseException):
            raise self._models
        return self._models


def test_check_reviewer_without_env_needs_no_network():
    provider = _Listing(["A"])
    reviewer, warning = check_reviewer(provider, None)
    assert warning is None and reviewer.enabled and reviewer.model is None
    assert provider.listed == 0
    assert reviewer.model_for("M-chat") == "M-chat"


@pytest.mark.parametrize("name", ["Reviewer-Max", "reviewer-max"])
def test_check_reviewer_finds_model(name):
    reviewer, warning = check_reviewer(_Listing(["M-chat", "Reviewer-Max"]), name)
    assert warning is None and reviewer.enabled
    assert reviewer.model_for("M-chat") == "Reviewer-Max"  # имя — как в API


def test_check_reviewer_disables_missing_model():
    reviewer, warning = check_reviewer(_Listing(["M-chat"]), "nope")
    assert not reviewer.enabled
    assert warning == (
        "модели «nope» (GIGACREW_REVIEWER_MODEL) нет — модель-рецензент отключена: "
        "в режиме «авто» больше команд будут требовать ручного подтверждения"
    )
    assert reviewer.warning == warning


@pytest.mark.parametrize("models", [LLMError("сеть недоступна"), []])
def test_check_reviewer_keeps_unverified_model(models):
    reviewer, warning = check_reviewer(_Listing(models), "R")
    assert reviewer.enabled and reviewer.model == "R"
    assert warning is not None and "не удалось проверить модель-рецензент «R»" in warning


def test_reviewer_disables_after_failures_in_a_row():
    reviewer = Reviewer()
    assert not reviewer.failed("a")
    reviewer.succeeded()  # успех обнуляет счёт
    assert [reviewer.failed("e") for _ in range(MAX_FAILURES)] == [False] * (MAX_FAILURES - 1) + [
        True
    ]
    assert not reviewer.enabled and DISABLED_NOTE in reviewer.warning
    assert not reviewer.failed("ещё")  # отключение сообщается один раз


# ------------------------------ в агентном цикле ------------------------------ #
def test_allowed_command_runs_without_asking(tmp_path):
    script = _script(tmp_path)
    memory, _ = DecisionMemory.load(tmp_path, base=tmp_path / "home")
    provider = ScriptedProvider(
        [_run(script, description="пояснение модели"), text_turn("ок")],
        summaries=[text_turn(ALLOW, Usage(prompt_tokens=100, completion_tokens=10))],
    )
    events = RecordingEvents(confirm_answer=False)
    agent = _agent(provider, tmp_path, events=events, decisions=memory)
    agent.run_turn("собери проект")
    assert events.confirms == []
    assert "built" in [m.content for m in agent.conversation.messages if m.role == "function"][0]
    assert events.reviews == [Review("M-chat", True, "сборка проекта по просьбе пользователя")]
    # рецензент видит слова пользователя, но не пояснение модели и не её ответы
    (request,) = provider.summary_requests
    assert request["model"] == "M-chat"
    user = request["messages"][1].content
    assert "собери проект" in user and script in user and "пояснение модели" not in user
    # расход рецензента учтён в сессии и в ходе
    assert agent.billed_tokens == 110 and events.stats[-1].prompt_tokens == 100
    entry = next(
        json.loads(line)
        for line in memory.journal_path.read_text().splitlines()
        if '"decision"' in line and '"event": "decision"' in line
    )
    assert entry["source"] == "reviewer" and entry["review"]["allowed"] is True
    assert memory.recent()[-1]["source"] == "reviewer"


def test_doubtful_command_goes_to_user_with_reviewer_reason(tmp_path):
    script = _script(tmp_path)
    provider = ScriptedProvider([_run(script), text_turn("ок")], summaries=[text_turn(DENY)])
    events = RecordingEvents(confirm_answer=False)
    agent = _agent(provider, tmp_path, events=events)
    agent.run_turn("x")
    (call, _, dangerous) = events.confirms[0]
    assert call.review == "рецензент (M-chat): об этом не просили" and not dangerous
    assert "REJECTED" in agent.conversation.messages[-2].content


def test_reviewer_failures_ask_user_and_disable_it(tmp_path):
    script = _script(tmp_path)
    runs = [_run(f"{script} {i}") for i in range(MAX_FAILURES + 1)]
    provider = ScriptedProvider(
        [*runs, text_turn("ок")], summaries=[LLMError("шлюз недоступен")] * MAX_FAILURES
    )
    events = RecordingEvents(confirm_answer=True)
    agent = _agent(provider, tmp_path, events=events)
    agent.run_turn("x")
    reviews = [call.review for call, _, _ in events.confirms]
    assert reviews[0] == "рецензент не ответил: шлюз недоступен"
    assert reviews[-1].startswith("рецензент отключён: 3 сбоя рецензента подряд")
    assert len(provider.summary_requests) == MAX_FAILURES  # после отключения не спрашиваем
    warnings = [text for level, text in events.notices if level == "warn"]
    assert len(warnings) == 1 and DISABLED_NOTE in warnings[0]


def test_dangerous_command_never_reaches_reviewer(tmp_path):
    provider = ScriptedProvider(
        [_run("rm -rf build"), text_turn("ок")], summaries=[text_turn(ALLOW)]
    )
    events = RecordingEvents(confirm_answer=False)
    _agent(provider, tmp_path, events=events).run_turn("x")
    assert provider.summary_requests == [] and events.reviews == []
    assert events.confirms[0][2] is True  # опасно


def test_rejected_command_is_not_reviewed_again(tmp_path):
    script = _script(tmp_path)
    provider = ScriptedProvider(
        [_run(script), text_turn("ок"), _run(script), text_turn("ок")],
        summaries=[text_turn(DENY), text_turn(ALLOW)],
    )
    events = RecordingEvents(confirm_answer=False)
    agent = _agent(provider, tmp_path, events=events)
    agent.run_turn("x")
    agent.run_turn("ещё раз")
    assert len(provider.summary_requests) == 1  # второй раз — сразу пользователю
    assert events.confirms[1][0].review == "рецензент не спрашивался: такую команду уже отклоняли"


def test_saved_rule_wins_before_reviewer(tmp_path):
    script = _script(tmp_path)
    provider = ScriptedProvider(
        [_run(script), text_turn("ок"), _run(script), text_turn("ок")],
        summaries=[text_turn(DENY)],
    )
    events = RecordingEvents(confirm_answer=Approval.ALWAYS)
    agent = _agent(provider, tmp_path, events=events)
    agent.run_turn("x")
    agent.run_turn("ещё раз")
    assert len(events.confirms) == 1 and len(provider.summary_requests) == 1


def test_reviewer_model_follows_chat_or_env(tmp_path):
    script = _script(tmp_path)
    provider = ScriptedProvider(
        [_run(script), text_turn("ок"), _run(f"{script} 2"), text_turn("ок")],
        summaries=[text_turn(ALLOW), text_turn(ALLOW)],
    )
    agent = _agent(provider, tmp_path)
    agent.run_turn("x")
    agent.set_model("M-other")
    agent.run_turn("y")
    assert [r["model"] for r in provider.summary_requests] == ["M-chat", "M-other"]

    provider = ScriptedProvider([_run(script), text_turn("ок")], summaries=[text_turn(ALLOW)])
    _agent(provider, tmp_path, reviewer=Reviewer("R-env")).run_turn("x")
    assert provider.summary_requests[0]["model"] == "R-env"


def test_disabled_reviewer_leaves_commands_to_user(tmp_path):
    script = _script(tmp_path)
    provider = ScriptedProvider([_run(script), text_turn("ок")])
    events = RecordingEvents(confirm_answer=True)
    reviewer = Reviewer("nope", disabled="модели «nope» (GIGACREW_REVIEWER_MODEL) нет")
    _agent(provider, tmp_path, events=events, reviewer=reviewer).run_turn("x")
    assert provider.summary_requests == []
    assert events.confirms[0][0].review.startswith("рецензент отключён: модели «nope»")


def test_long_command_is_left_to_user(tmp_path):
    script = _script(tmp_path)
    command = f"{script} {'a' * MAX_REVIEW_COMMAND}"
    provider = ScriptedProvider([_run(command), text_turn("ок")])
    events = RecordingEvents(confirm_answer=False)
    _agent(provider, tmp_path, events=events).run_turn("x")
    assert provider.summary_requests == []
    assert "длиннее" in events.confirms[0][0].review


def test_plan_mode_does_not_use_reviewer(tmp_path):
    script = _script(tmp_path)
    provider = ScriptedProvider([_run(script), text_turn("ок")], summaries=[text_turn(ALLOW)])
    events = RecordingEvents(confirm_answer=False)
    agent = _agent(provider, tmp_path, events=events, cfg={"mode": PermissionMode.PLAN})
    agent.chat_id = "c1"
    agent.run_turn("x")
    assert events.reviews == [] and len(events.confirms) == 1


def test_interrupt_during_review_stops_the_turn(tmp_path):
    script = _script(tmp_path)
    provider = ScriptedProvider([_run(script)], summaries=[KeyboardInterrupt()])
    events = RecordingEvents(confirm_answer=True)
    agent = _agent(provider, tmp_path, events=events)
    with pytest.raises(KeyboardInterrupt):
        agent.run_turn("x")
    assert events.reviews == [None] and events.confirms == []


def test_subagent_review_sees_user_request_not_the_brief(tmp_path):
    script = _script(tmp_path)
    provider = ScriptedProvider(
        [
            tool_turn(
                "task", {"agent": "general", "prompt": "поручение модели", "description": "сборка"}
            ),
            _run(script),
            text_turn("собрал"),
            text_turn("ок"),
        ],
        summaries=[text_turn(ALLOW)],
    )
    events = RecordingEvents(confirm_answer=False)
    agent = _agent(provider, tmp_path, events=events)
    agent.run_turn("собери проект")
    assert events.confirms == []
    user = provider.summary_requests[0]["messages"][1].content
    assert "собери проект" in user and "поручение модели" not in user
    assert (
        "info",
        "general: рецензент (M-chat) разрешил: сборка проекта по просьбе пользователя",
    ) in (events.notices)
    assert ("subagent_activity", ("рецензент проверяет команду", False)) in events.events


def test_rejection_with_feedback_is_remembered_for_session(tmp_path):
    script = _script(tmp_path)
    provider = ScriptedProvider(
        [_run(script), text_turn("ок"), _run(script), text_turn("ок")],
        summaries=[text_turn(DENY)],
    )

    class Answers(RecordingEvents):
        def confirm(self, call, preview, *, dangerous):
            self.confirms.append((call, preview, dangerous))
            return ConfirmResult(Approval.NO, feedback="не надо")

    events = Answers()
    agent = _agent(provider, tmp_path, events=events)
    agent.run_turn("x")
    agent.run_turn("y")
    assert len(provider.summary_requests) == 1
