"""Офлайн-тесты инфраструктуры eval (``tests/behavior.py``): учебный проект, сценарный
пользователь, метрики прогона, отчёт и разбор проб — на ScriptedProvider, без сети."""

from __future__ import annotations

import shutil
from dataclasses import replace

import pytest
from behavior import (
    FIXTURE_FILES,
    SCENARIO_BY_ID,
    SCENARIOS,
    EvalSession,
    ScriptedUser,
    compare_reports,
    describe_probe,
    digest_report,
    eval_models,
    invariant_errors,
    is_safe_command,
    load_report,
    main,
    make_project,
    reasoning_texts,
    run_fixture_tests,
    run_scenario,
    summary_table,
    well_formed,
    with_reasoning,
)
from fakes import ScriptedProvider, text_turn, tool_turn

from gigacrew.agent.events import ToolCallInfo
from gigacrew.config import Config
from gigacrew.llm.base import LLMError
from gigacrew.llm.model_windows import ModelWindows
from gigacrew.llm.types import FunctionCall, Message
from gigacrew.permissions import ToolKind
from gigacrew.tools.base import Display
from gigacrew.tools.plan import APPROVE_EDITS, PLAN_QUESTION
from gigacrew.tools.questions import Answer, Question, QuestionOption

PLAN = ".gigacrew/plans/chat1.md"
CLEAR_METHOD = "\n    def clear(self) -> None:\n        self._items.clear()\n"


@pytest.fixture
def shop(tmp_path):
    return make_project(tmp_path / "shop")


def _run(scenario_id: str, turns: list, root):
    scenario = SCENARIO_BY_ID[scenario_id]
    config = Config(
        access_key="x",
        project_root=root,
        model="scripted",
        mode=scenario.mode,
        save_chats=False,
        auto_measure=False,
    )
    provider = ScriptedProvider(turns)
    return run_scenario(scenario, provider=provider, config=config, chat_id="chat1")


def _cart_with_clear(root) -> str:
    return (root / "shop/cart.py").read_text(encoding="utf-8") + CLEAR_METHOD


def test_fixture_project_is_consistent(shop):
    for rel, text in FIXTURE_FILES.items():
        assert (shop / rel).read_text(encoding="utf-8") == text
        if rel.endswith(".py"):
            compile(text, rel, "exec")
    assert run_fixture_tests(shop)
    if shutil.which("git"):
        assert (shop / ".git").is_dir()


def test_scenarios_are_well_defined():
    assert len({scenario.id for scenario in SCENARIOS}) == len(SCENARIOS)
    for scenario in SCENARIOS:
        assert scenario.turns and scenario.goal
        assert all(
            rel in FIXTURE_FILES or rel.startswith("tests/") for rel in scenario.files_expect
        )


@pytest.mark.parametrize(
    "command",
    [
        "pytest -q",
        "python -m pytest tests/test_cart.py -q",
        "ls -la shop",
        "git status",
        "git diff HEAD",
        "grep -rn ValidationError shop | head -5",
        "cd shop && ls",
        "PYTHONPATH=. pytest",
    ],
)
def test_safe_commands_are_allowed(command):
    assert is_safe_command(command)


@pytest.mark.parametrize(
    "command",
    [
        "rm -f shop/cart.py",
        "python script.py",
        "pip install requests",
        "git commit -am x",
        "echo x > shop/cart.py",
        "find . -name '*.py' -delete",
        "cat $(ls)",
        "uv run pytest",
        "pytest -q; rm -f x",
        "git",
        "sort -o shop/cart.py shop/cart.py",
        "git diff --output=patch.diff",
        "find . -fprint list.txt",
    ],
)
def test_other_commands_are_rejected(command):
    assert not is_safe_command(command)


def test_scripted_user_answers_plan_and_model_questions():
    plan = replace(PLAN_QUESTION, body="# План")
    user = ScriptedUser([Answer((APPROVE_EDITS,)), None])
    assert user.ask_user([plan]) == [Answer((APPROVE_EDITS,))]
    assert user.ask_user([plan]) is None
    assert user.ask_user([plan]) is None  # ответы кончились — как Esc
    question = Question("Какая БД?", (QuestionOption("SQLite"), QuestionOption("Postgres")))
    assert user.ask_user([question]) == [Answer(("SQLite",))]
    assert user.plans == ["# План"] * 3 and user.model_questions == 1


def test_scripted_user_confirms_edits_and_safe_commands_only():
    user = ScriptedUser()
    edit = ToolCallInfo("write_file", "a.py", ToolKind.EDIT)
    assert user.confirm(edit, None, dangerous=False) is True
    assert user.confirm(edit, None, dangerous=True) is False
    command = ToolCallInfo("run_shell", "pytest -q", ToolKind.COMMAND)
    assert user.confirm(command, Display("$ pytest -q"), dangerous=False) is True
    # в описании вызова команда обрезана — решает полная команда из превью
    assert (
        user.confirm(command, Display("$ pytest -q; rm -f shop/cart.py"), dangerous=False) is False
    )
    assert user.confirm(ToolCallInfo("git", "commit -m x"), None, dangerous=False) is False
    assert user.commands == [("pytest -q", True), ("pytest -q; rm -f shop/cart.py", False)]


def test_plan_submitted_by_model(shop):
    result = _run(
        "plan_exit",
        [
            tool_turn("write_file", {"path": PLAN, "content": "# План\n1. Промокоды в checkout"}),
            tool_turn("exit_plan_mode", {}),
            text_turn("Жду решения."),
        ],
        shop,
    )
    record = result.record
    assert result.error is None and record["ok"]
    assert record["exit_calls"] == record["plan_shown"] == record["plan_writes"] == 1
    assert record["plan_turns_ended_in_text"] == 0 and record["plan_reminders"] == 0
    assert record["plan_written"] and record["changed_files"] == []
    assert record["final_mode"] == "plan" and record["requests"] == 3
    assert invariant_errors(SCENARIO_BY_ID["plan_exit"], record) == []
    assert result.transcript["plans"][0].startswith("# План")


def test_plan_left_in_text_is_shown_by_harness(shop):
    result = _run(
        "plan_exit",
        [
            tool_turn("write_file", {"path": PLAN, "content": "# План"}),
            text_turn("Вот план: …"),  # без exit_plan_mode — harness показывает план сам
            text_turn("лишний запрос"),
        ],
        shop,
    )
    record = result.record
    assert record["ok"] and record["exit_calls"] == 0 and record["plan_shown"] == 1
    assert record["plan_reminders"] == 1 and record["self_checks"] == 0
    assert record["plan_turns_ended_in_text"] == 0 and record["requests"] == 2


def test_promise_without_plan_gets_self_check(shop):
    result = _run(
        "plan_exit",
        [
            text_turn("Сейчас запишу план."),
            tool_turn("write_file", {"path": PLAN, "content": "# План"}),
            tool_turn("exit_plan_mode", {}),
            text_turn("Жду решения."),
        ],
        shop,
    )
    record = result.record
    assert record["ok"] and record["exit_calls"] == record["plan_shown"] == 1
    assert record["self_checks"] == record["self_checks_resumed"] == 1
    assert record["requests"] == 4 and record["plan_reminders"] == 0


def test_plan_revised_after_remarks(shop):
    result = _run(
        "plan_remarks",
        [
            tool_turn("write_file", {"path": PLAN, "content": "# План\n1. Промокоды"}),
            tool_turn("exit_plan_mode", {}),
            tool_turn(
                "edit_file",
                {
                    "path": PLAN,
                    "old_string": "1. Промокоды",
                    "new_string": "1. Промокоды\n2. Тесты",
                },
            ),
            tool_turn("exit_plan_mode", {}),
            text_turn("Жду."),
        ],
        shop,
    )
    record = result.record
    assert record["ok"] and record["plan_revised"]
    assert record["plan_shown"] == record["exit_calls"] == 2 and record["plan_edits"] == 1


def test_approved_plan_is_implemented(shop):
    result = _run(
        "plan_implement",
        [
            tool_turn("write_file", {"path": PLAN, "content": "1. Cart.clear"}),
            tool_turn("exit_plan_mode", {}),
            tool_turn("write_file", {"path": "shop/cart.py", "content": _cart_with_clear(shop)}),
            text_turn("Готово."),
        ],
        shop,
    )
    record = result.record
    assert record["ok"] and record["final_mode"] == "edits"
    assert record["changed_files"] == ["shop/cart.py"] and record["changed_before_plan"] == []
    assert record["files_ok"] and record["tests_pass"]
    assert invariant_errors(SCENARIO_BY_ID["plan_implement"], record) == []


def test_manual_exit_from_plan_mode(shop):
    result = _run(
        "plan_manual_exit",
        [
            tool_turn("write_file", {"path": PLAN, "content": "1. Cart.clear"}),
            tool_turn("exit_plan_mode", {}),
            text_turn("Жду."),
            tool_turn("write_file", {"path": "shop/cart.py", "content": _cart_with_clear(shop)}),
            text_turn("Готово."),
        ],
        shop,
    )
    record = result.record
    assert record["ok"] and record["answer"] == "Готово."
    assert [turn["mode_before"] for turn in record["turns"]] == ["plan", "edits"]
    assert record["turns"][0]["plan_shown"] == 1 and record["turns"][1]["exit_calls"] == 0


def test_question_in_plan_mode(shop):
    result = _run(
        "plan_question",
        [
            tool_turn("read_file", {"path": "shop/cart.py"}),
            text_turn("Не больше 50 товаров: проверка в Cart.add, shop/cart.py:17."),
        ],
        shop,
    )
    record = result.record
    assert record["ok"] and record["answer_hits"] == 1.0
    assert record["plan_shown"] == 0 and not record["plan_written"]
    assert record["research_calls"] == 1 and record["plan_turns_ended_in_text"] == 1
    # ответ на вопрос пережил самопроверку: +1 запрос, служебного обмена в истории нет
    assert record["self_checks"] == 1 and record["self_checks_resumed"] == 0
    assert record["requests"] == 3


def test_broad_question_delegated_to_explore(shop):
    result = _run(
        "delegate_broad",
        [
            tool_turn(
                "task",
                {
                    "agent": "explore",
                    "prompt": "Где возникают исключения?",
                    "description": "ошибки",
                },
            ),
            tool_turn("read_file", {"path": "shop/api.py"}),  # суб-агент
            text_turn("shop/errors.py, shop/api.py:25, shop/storage.py, shop/payments.py"),
            text_turn("Ошибки: shop/errors.py, shop/api.py, shop/storage.py, shop/payments.py."),
        ],
        shop,
    )
    record = result.record
    assert record["ok"] and record["answer_hits"] == 1.0
    assert record["task_calls"] == record["explore_calls"] == record["subagent_runs"] == 1
    assert record["requests"] == 4 and record["requests_subagent"] == 2
    assert record["research_calls"] == 0 and record["subagent_tool_calls"] == 1


def test_broad_question_explored_with_explore_tool(shop):
    result = _run(
        "delegate_broad",
        [
            tool_turn("explore", {"question": "Где возникают исключения и как становятся HTTP?"}),
            text_turn("shop/errors.py, shop/api.py:25, shop/storage.py, shop/payments.py"),
            text_turn("Ошибки: shop/errors.py, shop/api.py, shop/storage.py, shop/payments.py."),
        ],
        shop,
    )
    record = result.record
    assert record["ok"] and record["task_calls"] == record["explore_calls"] == 1
    assert record["general_calls"] == 0 and record["subagent_runs"] == 1


PAYMENTS_TESTS = """\
import pytest

from shop.errors import PaymentError
from shop.payments import PaymentGateway


def test_charge_returns_payment_id():
    assert PaymentGateway().charge("4111111111111111", 100) == "pay-1"


def test_declined_card():
    with pytest.raises(PaymentError):
        PaymentGateway().charge("4000000000000002", 100)
"""
STORAGE_TESTS = """\
import pytest

from shop.errors import NotFoundError
from shop.storage import JsonStorage


def test_missing_order(tmp_path):
    with pytest.raises(NotFoundError):
        JsonStorage(tmp_path / "orders.json").order_status("nope")
"""


@pytest.mark.parametrize(
    ("storage", "ok"),
    [(STORAGE_TESTS, True), (STORAGE_TESTS.replace("NotFoundError):", "KeyError):"), False)],
    ids=["tests_pass", "tests_fail"],
)
def test_subtasks_done_by_main_agent_count(shop, storage, ok):
    """Справочный сценарий: успех — тесты написаны и проходят, суб-агенты — на выбор модели."""
    result = _run(
        "delegate_subtasks",
        [
            tool_turn("write_file", {"path": "tests/test_payments.py", "content": PAYMENTS_TESTS}),
            tool_turn("write_file", {"path": "tests/test_storage.py", "content": storage}),
            text_turn("Готово."),
        ],
        shop,
    )
    record = result.record
    assert record["task_calls"] == 0 and record["files_ok"]
    assert record["tests_pass"] is ok and record["ok"] is ok


def test_pinpoint_question_answered_directly(shop):
    result = _run(
        "delegate_pinpoint",
        [tool_turn("read_file", {"path": "shop/money.py"}), text_turn("ValidationError")],
        shop,
    )
    record = result.record
    assert record["ok"] and record["task_calls"] == 0 and record["research_calls"] == 1


def test_api_error_is_recorded_not_lost(shop):
    result = _run("delegate_pinpoint", [LLMError("шлюз недоступен")], shop)
    assert isinstance(result.error, LLMError)
    assert result.record["error"] == "LLMError: шлюз недоступен" and not result.record["ok"]
    assert result.record["well_formed"]


def test_invariants_are_about_the_harness():
    record = {
        "well_formed": True,
        "changed_files": ["shop/cart.py"],
        "changed_before_plan": ["shop/cart.py"],
    }
    assert len(invariant_errors(SCENARIO_BY_ID["plan_exit"], record)) == 2
    calm = {**record, "changed_before_plan": []}
    assert invariant_errors(SCENARIO_BY_ID["delegate_broad"], calm) == []
    assert invariant_errors(SCENARIO_BY_ID["delegate_broad"], {**calm, "well_formed": False})


def test_well_formed_history():
    call = FunctionCall(name="read_file", arguments={"path": "a"})
    result = Message(role="function", name="read_file", content="ok")
    user = Message(role="user", content="?")
    assert well_formed([user, Message(role="assistant", function_calls=[call]), result])
    assert not well_formed([user, result])
    assert not well_formed([Message(role="assistant", function_calls=[call]), user])


def test_report_is_written_after_each_run_and_compared(shop, tmp_path):
    result = _run(
        "delegate_pinpoint",
        [tool_turn("read_file", {"path": "shop/money.py"}), text_turn("ValidationError")],
        shop,
    )
    session = EvalSession(
        config=Config(access_key="x", project_root=shop, model="scripted"),
        provider=ScriptedProvider(),
        windows=ModelWindows(),
        directory=tmp_path / "evals",
        models=["scripted"],
        runs=1,
        label="baseline t0.2",
    )
    record = session.add(result, run=1)
    assert session.path.name.endswith("-baseline_t0.2.json")
    assert (session.directory / record["transcript"]).is_file()
    report = load_report(session.path)
    assert report["meta"]["models"] == ["scripted"]
    assert report["runs"][0]["scenario"] == "delegate_pinpoint"
    lines = summary_table(report["runs"])
    assert any("delegate_pinpoint" in line and "1/1" in line for line in lines)
    newer = {
        "meta": {**report["meta"], "label": "после"},
        "runs": [{**report["runs"][0], "ok": False}],
    }
    compared = compare_reports(report, newer)
    assert any("1/1 → 0/1" in line for line in compared)
    assert all(name in compared[3] for name in ("err", "rem", "chk", "txt", "res"))
    assert main(["show", str(session.path)]) == 0
    assert main(["digest", str(session.path)]) == 0
    assert report["meta"]["reasoning"] == "turn" and "reasoning=turn" in compared[0]
    off_meta = {**newer["meta"], "reasoning": "off"}
    assert "reasoning=off" in compare_reports(report, {**newer, "meta": off_meta})[1]
    old_meta = {k: v for k, v in report["meta"].items() if k != "reasoning"}  # до этапа 5
    assert "reasoning=" not in compare_reports({**report, "meta": old_meta}, newer)[0]


def test_digest_fits_a_chat_message():
    def run(i: int, scenario: str, **extra) -> dict:
        return {
            "model": f"model-{i % 4}",
            "scenario": scenario,
            "run": i,
            "ok": True,
            "error": None,
            "plan_shown": 1,
            "exit_calls": 1,
            "plan_reminders": 0,
            "self_checks": 1,
            "plan_turns_ended_in_text": 0,
            "research_calls": 9,
            "requests": 12,
            "cut_retries": 0,
            "max_completion_tokens": 6_500,
            "max_request_s": 205.0,
            "stop_reasons": [],
            "task_calls": 0,
            "answer": "Жду решения.",
            **extra,
        }

    runs = [run(i, scenario.id) for i, scenario in enumerate(SCENARIOS * 4)]
    runs += [
        run(i, "plan_exit", ok=False, error="LLMError: The read operation timed out")
        for i in range(40)
    ]
    runs.append(run(1, "plan_question", ok=False, stop_reasons=["tool_failures"], answer=""))
    old = {**runs[0]}
    runs[0]["rate_limit_waits"] = 3
    for key in ("self_checks", "max_completion_tokens", "max_request_s"):
        del old[key]  # отчёт до этапа 3
    report = {"meta": {"created": "20261013-120000", "temperature": 1.0}, "runs": [*runs, old]}
    lines = digest_report(report)
    assert len(lines) <= 100
    text = "\n".join(lines)
    assert "Неуспешные прогоны: 41 из 82" in text and "… и ещё 11" in text
    assert "40× LLMError: The read operation timed out" in text
    assert "стоп tool_failures" in text
    assert "Пауз из-за лимита запросов (429): 3 в 1 прогонах" in text
    row = next(line for line in lines if line.startswith("model-1") and "plan_exit" in line)
    assert "6.5" in row and "205.0" in row


def test_eval_models_from_environment():
    env = {"GIGACREW_EVAL_MODELS": "a, b,,a", "GIGACREW_TEST_MODEL": "c"}
    assert eval_models(env) == ["a", "b"]
    assert eval_models({"GIGACREW_TEST_MODEL": "c"}) == ["c"]
    assert eval_models({}) == []


def test_probe_helpers():
    data = {
        "messages": [
            {"role": "reasoning", "content": [{"text": "думаю"}]},
            {"role": "reasoning", "content": "ещё"},
            {"role": "assistant", "content": [{"text": "ок"}]},
        ]
    }
    assert reasoning_texts(data) == ["думаю", "ещё"]
    payload = {
        "messages": [
            {"role": "user", "content": [{"text": "?"}]},
            {"role": "assistant", "content": [{"text": ""}]},
            {"role": "tool", "content": []},
        ]
    }
    patched = with_reasoning(payload, "думаю")
    assert [m["role"] for m in patched["messages"]] == ["user", "reasoning", "assistant", "tool"]
    assert [m["role"] for m in payload["messages"]] == ["user", "assistant", "tool"]
    finding = {
        "reasoning_chars": 300,
        "tool_call": True,
        "step2": {
            "with_state": {"status": 200, "input_tokens": 500},
            "without_state": {"status": 200, "input_tokens": 400},
            "with_state+reasoning": {"status": 422, "input_tokens": None},
            "without_state+reasoning": {"status": 200, "input_tokens": 480},
        },
        "sampling": {"top_p=0.95": {"status": 200}},
        "text_stream_finish": "stop",
        "cache": [
            {"input_tokens": 1000, "cached_tokens": 0},
            {"input_tokens": 1000, "cached_tokens": 900},
        ],
    }
    text = "\n".join(describe_probe("m", finding))
    assert "да, 300 симв." in text and "ошибка 422" in text and "480 ток. (+80)" in text
    assert "разница +100 ток.: похоже, шлюз сам подставляет" in text
    assert "из кеша 900 (90%)" in text and "top_p=0.95 → принято" in text
    assert "текстовый поток: finish_reason=stop — окончание приходит" in text
    same = {**finding["step2"], "without_state": {"status": 200, "input_tokens": 498}}
    text = "\n".join(describe_probe("m", {**finding, "step2": same}))
    assert "разница +2 ток.: признаков, что шлюз хранит рассуждения сам, нет" in text
