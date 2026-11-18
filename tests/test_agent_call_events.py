"""Агентный цикл: id, аргументы и пути вызова в ToolCallInfo; событие рассуждений."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import override

from fakes import RecordingEvents, ScriptedProvider, text_turn, tool_turn, tools_turn

from gigacrew.agent.events import ToolCallInfo
from gigacrew.agent.loop import Agent
from gigacrew.config import Config
from gigacrew.llm.types import AssistantTurn, Message, Usage
from gigacrew.tools.base import build_default_registry


class _Events(RecordingEvents):
    """Записывает и рассуждения — в общий журнал, по порядку с текстом ответа."""

    @override
    def on_reasoning(self, text: str) -> None:
        self.events.append(("reasoning", text))


def _run(provider: ScriptedProvider, root: Path, *, stream: bool = False) -> _Events:
    events = _Events()
    config = Config(access_key="x", project_root=root, stream=stream, auto_approve=True)
    Agent(provider, build_default_registry(), config, events).run_turn("сделай")
    return events


def _calls(events: _Events) -> list[ToolCallInfo]:
    return [entry for kind, entry in events.events if kind == "tool_call"]


def test_call_info_carries_model_id_arguments_and_paths(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("A", encoding="utf-8")
    batch = tools_turn([("read_file", {"path": "a.txt"}), ("no_such_tool", {"x": 1})])
    events = _run(ScriptedProvider([batch, text_turn("готово")]), tmp_path)

    read, unknown = _calls(events)
    assert (read.call_id, dict(read.arguments)) == ("c1", {"path": "a.txt"})
    assert read.paths == (str((tmp_path / "a.txt").resolve()),)
    assert (unknown.call_id, dict(unknown.arguments)) == ("c2", {"x": 1})
    assert replace(read, reason="спросить").call_id == "c1"


def test_call_without_model_id_gets_unique_one(tmp_path: Path) -> None:
    read = tool_turn("read_file", {"path": "a.txt"})
    events = _run(ScriptedProvider([read, read, text_turn("готово")]), tmp_path)

    ids = [call.call_id for call in _calls(events)]
    assert len(ids) == 2
    assert all(ids)
    assert len(set(ids)) == 2


def test_path_outside_project_is_not_reported(tmp_path: Path) -> None:
    events = _run(ScriptedProvider([tool_turn("read_file", {"path": "/etc/hostname"})]), tmp_path)

    assert not _calls(events)[0].paths


def _reasoning_turn() -> AssistantTurn:
    message = Message(role="assistant", content="Ответ", reasoning="Подумаю")
    return AssistantTurn(message=message, usage=Usage())


def test_reasoning_comes_after_answer_and_before_its_text(tmp_path: Path) -> None:
    events = _run(ScriptedProvider([_reasoning_turn()]), tmp_path)

    kinds = [kind for kind, _ in events.events]
    assert kinds.index("stream_end") < kinds.index("reasoning") < kinds.index("text")
    assert ("reasoning", "Подумаю") in events.events


def test_no_reasoning_event_without_reasoning(tmp_path: Path) -> None:
    events = _run(ScriptedProvider([text_turn("Ответ")]), tmp_path, stream=True)

    assert all(kind != "reasoning" for kind, _ in events.events)
