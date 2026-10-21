"""Хуки пользователя в агентном цикле: раннер с заготовленными ответами вместо процессов."""

from __future__ import annotations

import json
import os
import shlex
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from fakes import FakeTool, RecordingEvents, ScriptedProvider, text_turn, tool_turn

from gigacrew.agent.chat_store import chat_title
from gigacrew.agent.events import Approval
from gigacrew.agent.hooks import HookResult, HookRunner, PromptBlocked
from gigacrew.agent.loop import (
    HOOK_BLOCKED_NOTE,
    MAX_STOP_HOOK_BLOCKS,
    PLAN_BLOCKED_NOTE,
    STOP_HOOK_NOTE,
    Agent,
)
from gigacrew.config import Config
from gigacrew.permissions import Decision, PermissionMode, ToolKind, decide
from gigacrew.project.settings import HookCommand, HookEvent, HookGroup, HookSettings
from gigacrew.security import RiskLevel
from gigacrew.tools.base import ToolRegistry, ToolResult, build_default_registry
from gigacrew.tools.questions import Question, QuestionOption
from gigacrew.tools.task import EXPLORE

Answer = HookResult | Callable[[dict], HookResult] | list[HookResult]


class FakeHooks(HookRunner):
    """Раннер без процессов: на событие — заготовленный ответ (или функция от payload)."""

    def __init__(self, answers: dict[HookEvent, Answer]):
        super().__init__()
        self.answers = answers
        self.calls: list[tuple[HookEvent, str, dict]] = []

    def has(self, event, target=""):
        return event in self.answers

    def run(self, event, payload, *, cwd, events, target=""):
        self.calls.append((event, target, payload))
        answer = self.answers[event]
        if isinstance(answer, list):
            return answer.pop(0) if answer else HookResult()
        return answer(payload) if callable(answer) else answer

    def of(self, event: HookEvent) -> list[dict]:
        return [payload for e, _, payload in self.calls if e is event]


def _agent(provider, tmp_path: Path, hooks: HookRunner, *, events=None, tool=None, **cfg_kw):
    cfg_kw.setdefault("auto_approve", True)
    cfg = Config(access_key="x", project_root=tmp_path, stream=False, **cfg_kw)
    if tool is not None:
        registry = ToolRegistry()
        registry.register(tool)
    else:
        registry = build_default_registry()
    return Agent(provider, registry, cfg, events or RecordingEvents(), hooks=hooks)


def _function_results(agent: Agent) -> list[str]:
    return [m.content for m in agent.conversation.messages if m.role == "function"]


# --------------------------------------------------------------------------- #
# decide(hook=...)


@pytest.mark.parametrize(
    ("mode", "risk", "kind", "auto", "hook", "expected"),
    [
        (PermissionMode.AUTO, RiskLevel.WRITE, ToolKind.OTHER, False, "allow", Decision.ALLOW),
        # allow снимает и рецензента изменяющих команд
        (PermissionMode.AUTO, RiskLevel.WRITE, ToolKind.COMMAND, False, "allow", Decision.ALLOW),
        # опасное хук не разрешает молча — как -y без --yes-all
        (
            PermissionMode.AUTO,
            RiskLevel.DANGEROUS,
            ToolKind.COMMAND,
            False,
            "allow",
            Decision.ASK,
        ),
        (PermissionMode.AUTO, RiskLevel.SAFE, ToolKind.OTHER, False, "ask", Decision.ASK),
        (PermissionMode.AUTO, RiskLevel.WRITE, ToolKind.EDIT, True, "ask", Decision.ASK),
        # ask — решает пользователь, а не рецензент
        (PermissionMode.AUTO, RiskLevel.WRITE, ToolKind.COMMAND, False, "ask", Decision.ASK),
        # запрет режима плана хук не снимает
        (PermissionMode.PLAN, RiskLevel.WRITE, ToolKind.EDIT, False, "allow", Decision.BLOCK),
        (PermissionMode.PLAN, RiskLevel.WRITE, ToolKind.EDIT, False, "ask", Decision.BLOCK),
        (PermissionMode.AUTO, RiskLevel.WRITE, ToolKind.OTHER, False, None, Decision.ASK),
        (PermissionMode.AUTO, RiskLevel.WRITE, ToolKind.COMMAND, False, None, Decision.REVIEW),
    ],
)
def test_decide_with_hook(mode, risk, kind, auto, hook, expected):
    assert decide(mode, risk, kind, auto_approve=auto, hook=hook) is expected


def test_decide_hook_keeps_read_only_block():
    assert (
        decide(
            PermissionMode.AUTO,
            RiskLevel.WRITE,
            ToolKind.COMMAND,
            auto_approve=True,
            read_only=True,
            hook="allow",
        )
        is Decision.BLOCK
    )


# --------------------------------------------------------------------------- #
# PreToolUse


def test_pre_tool_use_deny_blocks_call(tmp_path):
    tool = FakeTool()
    hooks = FakeHooks({HookEvent.PRE_TOOL_USE: HookResult(blocked=True, reason="только чтение")})
    provider = ScriptedProvider([tool_turn("fake_write", {"path": "a"}), text_turn("понял")])
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, hooks, events=events, tool=tool)
    assert agent.run_turn("пиши") == "понял"
    assert tool.runs == 0 and events.confirms == []
    assert _function_results(agent) == [HOOK_BLOCKED_NOTE.format(reason="только чтение")]
    (payload,) = hooks.of(HookEvent.PRE_TOOL_USE)
    assert payload["tool_name"] == "fake_write" and payload["tool_input"] == {"path": "a"}
    assert payload["session_id"] == agent.chat_id
    assert payload["transcript_path"].endswith(f"{agent.chat_id}.json")
    assert payload["permission_mode"] == "auto" and payload["cwd"] == str(tmp_path.resolve())


def test_pre_tool_use_denials_are_not_a_failure_series(tmp_path):
    tool = FakeTool()
    hooks = FakeHooks({HookEvent.PRE_TOOL_USE: HookResult(blocked=True, reason="нет")})
    turns = [tool_turn("fake_write", {"path": f"f{i}"}) for i in range(4)] + [text_turn("всё")]
    agent = _agent(ScriptedProvider(turns), tmp_path, hooks, tool=tool, max_tool_failures=2)
    assert agent.run_turn("пиши") == "всё"
    assert agent.last_turn is not None and not agent.last_turn.stop_reason


def test_pre_tool_use_allow_skips_confirmation(tmp_path):
    tool = FakeTool()  # не правка и не команда: в режиме «авто» спрашивается
    hooks = FakeHooks({HookEvent.PRE_TOOL_USE: HookResult(decision="allow")})
    provider = ScriptedProvider([tool_turn("fake_write", {}), text_turn("ок")])
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, hooks, events=events, tool=tool, auto_approve=False)
    agent.run_turn("пиши")
    assert tool.runs == 1 and events.confirms == []


def test_pre_tool_use_ask_confirms_every_time(tmp_path):
    tool = FakeTool()
    hooks = FakeHooks({HookEvent.PRE_TOOL_USE: HookResult(decision="ask")})
    calls = [tool_turn("fake_write", {"path": "a"}), tool_turn("fake_write", {"path": "a"})]
    events = RecordingEvents(confirm_answer=Approval.ALWAYS)
    agent = _agent(
        ScriptedProvider([*calls, text_turn("ок")]), tmp_path, hooks, events=events, tool=tool
    )
    agent.run_turn("пиши")
    # «всегда» не запоминается: хук требует спрашивать каждый раз
    assert len(events.confirms) == 2 and tool.runs == 2


# PreToolUse и команды: рецензент режима «авто» и запрет режима плана
shell_only = pytest.mark.skipif(os.name != "posix", reason="классификатор команд — для /bin/sh")
REVIEWER_DENY = '{"allow": false, "reason": "об этом не просили"}'


def _shell_agent(provider, tmp_path: Path, hooks: HookRunner, *, events, mode=PermissionMode.AUTO):
    script = tmp_path / "build.sh"
    script.write_text("#!/bin/sh\necho built\n")
    script.chmod(0o755)
    cfg = Config(access_key="x", project_root=tmp_path, stream=False, model="M-chat", mode=mode)
    return Agent(provider, build_default_registry(), cfg, events, hooks=hooks)


@shell_only
def test_pre_tool_use_allow_skips_reviewer(tmp_path):
    hooks = FakeHooks({HookEvent.PRE_TOOL_USE: HookResult(decision="allow")})
    provider = ScriptedProvider(
        [tool_turn("run_shell", {"command": "./build.sh"}), text_turn("ок")],
        summaries=[text_turn(REVIEWER_DENY)],
    )
    events = RecordingEvents(confirm_answer=False)
    agent = _shell_agent(provider, tmp_path, hooks, events=events)
    agent.run_turn("собери")
    assert provider.summary_requests == [] and events.confirms == []
    assert "built" in _function_results(agent)[0]


@shell_only
def test_pre_tool_use_ask_goes_to_user_not_reviewer(tmp_path):
    hooks = FakeHooks({HookEvent.PRE_TOOL_USE: HookResult(decision="ask")})
    provider = ScriptedProvider(
        [tool_turn("run_shell", {"command": "./build.sh"}), text_turn("ок")],
        summaries=[text_turn('{"allow": true, "reason": "можно"}')],
    )
    events = RecordingEvents(confirm_answer=True)
    agent = _shell_agent(provider, tmp_path, hooks, events=events)
    agent.run_turn("собери")
    assert provider.summary_requests == []
    ((call, _, _),) = events.confirms
    assert "хук PreToolUse требует подтверждения" in call.reason


@shell_only
def test_pre_tool_use_allow_keeps_plan_mode_block(tmp_path):
    hooks = FakeHooks({HookEvent.PRE_TOOL_USE: HookResult(decision="allow")})
    provider = ScriptedProvider(
        [tool_turn("run_shell", {"command": "mkdir newdir"}), text_turn("ок")]
    )
    events = RecordingEvents(confirm_answer=True)
    agent = _shell_agent(provider, tmp_path, hooks, events=events, mode=PermissionMode.PLAN)
    agent.run_turn("создай каталог")
    assert not (tmp_path / "newdir").exists() and events.confirms == []
    assert _function_results(agent)[0] == PLAN_BLOCKED_NOTE.format(path=agent._plan_display())


def test_notification_before_confirmation(tmp_path):
    tool = FakeTool()
    events = RecordingEvents()

    def notify(payload: dict) -> HookResult:
        assert events.confirms == []  # хук — до вопроса
        return HookResult()

    hooks = FakeHooks({HookEvent.NOTIFICATION: notify})
    provider = ScriptedProvider([tool_turn("fake_write", {}), text_turn("ок")])
    agent = _agent(provider, tmp_path, hooks, events=events, tool=tool, auto_approve=False)
    agent.run_turn("пиши")
    (payload,) = hooks.of(HookEvent.NOTIFICATION)
    assert payload["notification_type"] == "permission_prompt"
    assert "fake_write" in payload["message"]
    assert hooks.calls[0][1] == "permission_prompt"  # matcher — по типу
    assert len(events.confirms) == 1


def test_notification_on_question(tmp_path):
    hooks = FakeHooks({HookEvent.NOTIFICATION: HookResult()})
    events = RecordingEvents(answers=[])
    agent = _agent(ScriptedProvider(), tmp_path, hooks, events=events)
    question = Question(
        "Какую базу?", (QuestionOption("pg"), QuestionOption("sqlite")), header="База"
    )
    agent._ctx.ask_user([question])
    (payload,) = hooks.of(HookEvent.NOTIFICATION)
    assert payload["notification_type"] == "question" and "База" in payload["message"]
    assert events.questions


# --------------------------------------------------------------------------- #
# PostToolUse


def test_post_tool_use_feedback_goes_to_model(tmp_path):
    tool = FakeTool(run=lambda: ToolResult(content="записано", summary="ok"))
    hooks = FakeHooks(
        {HookEvent.POST_TOOL_USE: HookResult(blocked=True, reason="ruff: E501", context=["лог"])}
    )
    provider = ScriptedProvider([tool_turn("fake_write", {"path": "a"}), text_turn("исправлю")])
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, hooks, events=events, tool=tool)
    agent.run_turn("пиши")
    (result,) = _function_results(agent)
    assert result.startswith("записано") and "ruff: E501" in result and "лог" in result
    assert "<system-reminder>" in result and "PostToolUse" in result
    (payload,) = hooks.of(HookEvent.POST_TOOL_USE)
    assert payload["tool_response"] == {"ok": True, "content": "записано"}
    assert any("ruff: E501" in text for _, text in events.notices)


def test_post_tool_use_interrupt_keeps_tool_result(tmp_path):
    tool = FakeTool(run=lambda: ToolResult(content="записано", summary="ok"))

    def interrupt(payload):
        raise KeyboardInterrupt

    hooks = FakeHooks({HookEvent.POST_TOOL_USE: interrupt})
    agent = _agent(ScriptedProvider([tool_turn("fake_write", {})]), tmp_path, hooks, tool=tool)
    with pytest.raises(KeyboardInterrupt):
        agent.run_turn("пиши")
    assert _function_results(agent) == ["записано"]  # а не «No result» от repair()


def test_post_tool_use_without_output_changes_nothing(tmp_path):
    tool = FakeTool(run=lambda: ToolResult(content="записано", summary="ok"))
    hooks = FakeHooks({HookEvent.POST_TOOL_USE: HookResult()})
    agent = _agent(ScriptedProvider([tool_turn("fake_write", {})]), tmp_path, hooks, tool=tool)
    agent.run_turn("пиши")
    assert _function_results(agent) == ["записано"]


# --------------------------------------------------------------------------- #
# UserPromptSubmit


def test_prompt_blocked_leaves_history_untouched(tmp_path):
    hooks = FakeHooks({HookEvent.USER_PROMPT_SUBMIT: HookResult(blocked=True, reason="секрет")})
    provider = ScriptedProvider()
    agent = _agent(provider, tmp_path, hooks)
    with pytest.raises(PromptBlocked) as exc:
        agent.run_turn("мой пароль 123")
    assert exc.value.reason == "секрет"
    assert len(agent.conversation) == 0 and provider.calls == 0


def test_prompt_context_is_appended_but_title_is_clean(tmp_path):
    hooks = FakeHooks({HookEvent.USER_PROMPT_SUBMIT: HookResult(context=["ветка feature/x"])})
    provider = ScriptedProvider([text_turn("ок")])
    agent = _agent(provider, tmp_path, hooks)
    agent.run_turn("почини тесты")
    assert hooks.of(HookEvent.USER_PROMPT_SUBMIT)[0]["prompt"] == "почини тесты"
    sent = provider.requests[0]["messages"][-1].content
    assert sent.startswith("почини тесты") and "ветка feature/x" in sent
    assert chat_title(agent.conversation.messages) == "почини тесты"


# --------------------------------------------------------------------------- #
# Stop / SubagentStop


def test_stop_hook_returns_model_to_work(tmp_path):
    hooks = FakeHooks(
        {HookEvent.STOP: [HookResult(blocked=True, reason="тесты падают"), HookResult()]}
    )
    provider = ScriptedProvider([text_turn("готово"), text_turn("теперь точно готово")])
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, hooks, events=events)
    assert agent.run_turn("сделай") == "теперь точно готово"
    assert provider.calls == 2
    last = provider.requests[1]["messages"][-1]
    assert STOP_HOOK_NOTE.format(event="Stop", reason="тесты падают") in last.content
    first, second = hooks.of(HookEvent.STOP)
    assert first["stop_hook_active"] is False and first["last_assistant_message"] == "готово"
    assert second["stop_hook_active"] is True
    assert len(events.stats) == 1  # один ход


def test_stop_hook_limit(tmp_path):
    hooks = FakeHooks({HookEvent.STOP: lambda payload: HookResult(blocked=True, reason="ещё")})
    provider = ScriptedProvider()  # отвечает «конец» сколько угодно раз
    events = RecordingEvents()
    agent = _agent(provider, tmp_path, hooks, events=events)
    agent.run_turn("сделай")
    assert provider.calls == MAX_STOP_HOOK_BLOCKS + 1
    assert any(level == "warn" and "не дал закончить" in text for level, text in events.notices)


def test_stop_hook_interrupt_finishes_with_answer(tmp_path):
    def interrupt(payload):
        raise KeyboardInterrupt

    hooks = FakeHooks({HookEvent.STOP: interrupt})
    agent = _agent(ScriptedProvider([text_turn("готово")]), tmp_path, hooks)
    assert agent.run_turn("сделай") == "готово"


def test_subagent_stop(tmp_path):
    hooks = FakeHooks(
        {
            HookEvent.SUBAGENT_STOP: [HookResult(blocked=True, reason="мало"), HookResult()],
            HookEvent.STOP: HookResult(),
            HookEvent.USER_PROMPT_SUBMIT: HookResult(),
        }
    )
    provider = ScriptedProvider([text_turn("отчёт"), text_turn("полный отчёт")])
    parent = _agent(provider, tmp_path, hooks)
    child = Agent(
        provider,
        build_default_registry(),
        parent.config,
        RecordingEvents(),
        parent=parent,
        subagent=EXPLORE,
    )
    assert child.run_turn("исследуй") == "полный отчёт"
    payloads = hooks.of(HookEvent.SUBAGENT_STOP)
    assert len(payloads) == 2 and payloads[0]["agent_type"] == EXPLORE.name
    assert payloads[0]["session_id"] == parent.chat_id
    assert hooks.calls[0][1] == EXPLORE.name
    # у суб-агента нет Stop и UserPromptSubmit: его задачу пишет модель
    assert hooks.of(HookEvent.STOP) == [] and hooks.of(HookEvent.USER_PROMPT_SUBMIT) == []


# --------------------------------------------------------------------------- #
# SessionStart / SessionEnd / PreCompact


def test_session_context_in_system_message_until_reset(tmp_path):
    hooks = FakeHooks(
        {
            HookEvent.SESSION_START: HookResult(context=["задача JIRA-1"]),
            HookEvent.SESSION_END: HookResult(),
        }
    )
    provider = ScriptedProvider([text_turn("ок"), text_turn("ок")])
    agent = _agent(provider, tmp_path, hooks)
    agent.start_session("startup")
    agent.run_turn("привет")
    assert "задача JIRA-1" in provider.requests[0]["messages"][0].content
    assert hooks.of(HookEvent.SESSION_START)[0]["source"] == "startup"
    agent.end_session("clear")
    agent.reset()
    agent.run_turn("снова")
    assert "задача JIRA-1" not in provider.requests[1]["messages"][0].content
    assert hooks.of(HookEvent.SESSION_END)[0]["reason"] == "clear"


def test_pre_compact(tmp_path):
    hooks = FakeHooks({HookEvent.PRE_COMPACT: HookResult()})
    agent = _agent(ScriptedProvider([text_turn("ответ " * 50)]), tmp_path, hooks)
    agent.run_turn("вопрос " * 50)
    assert agent.compact("покороче") is not None
    (payload,) = hooks.of(HookEvent.PRE_COMPACT)
    assert payload["trigger"] == "manual" and payload["custom_instructions"] == "покороче"


def test_no_hooks_no_overhead(tmp_path):
    tool = FakeTool()
    hooks = FakeHooks({})
    agent = _agent(
        ScriptedProvider([tool_turn("fake_write", {}), text_turn("ок")]), tmp_path, hooks, tool=tool
    )
    agent.run_turn("пиши")
    assert hooks.calls == []


# --------------------------------------------------------------------------- #
# Сквозная проверка с настоящими командами


@pytest.mark.skipif(os.name != "posix", reason="команды хуков — через /bin/sh")
def test_end_to_end_with_real_commands(tmp_path):
    log = tmp_path / "events.log"
    script = tmp_path / "hook.py"
    script.write_text(
        "import json, sys\n"
        "data = json.load(sys.stdin)\n"
        f"open({str(log)!r}, 'a').write(data['hook_event_name'] + '\\n')\n"
        "if data['hook_event_name'] == 'PreToolUse' and data['tool_input'].get('path') == 'bad':\n"
        "    print('нельзя', file=sys.stderr); sys.exit(2)\n",
        encoding="utf-8",
    )
    cmd = f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"
    groups = tuple(
        HookGroup(event, "", (HookCommand(cmd),))
        for event in (
            HookEvent.SESSION_START,
            HookEvent.USER_PROMPT_SUBMIT,
            HookEvent.PRE_TOOL_USE,
            HookEvent.POST_TOOL_USE,
            HookEvent.STOP,
            HookEvent.SESSION_END,
        )
    )
    tool = FakeTool()
    provider = ScriptedProvider(
        [
            tool_turn("fake_write", {"path": "bad"}),
            tool_turn("fake_write", {"path": "ok"}),
            text_turn("всё"),
        ]
    )
    agent = _agent(provider, tmp_path, HookRunner(HookSettings(groups)), tool=tool)
    agent.start_session("startup")
    agent.run_turn("пиши")
    agent.end_session("other")
    assert tool.runs == 1
    assert log.read_text().split() == [
        "SessionStart",
        "UserPromptSubmit",
        "PreToolUse",
        "PreToolUse",
        "PostToolUse",
        "Stop",
        "SessionEnd",
    ]
    assert "нельзя" in _function_results(agent)[0]
    assert json.loads(json.dumps(agent.conversation.to_dict()))  # история сериализуется
