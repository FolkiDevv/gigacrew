"""Меню /settings: курсор, флаги, варианты, ввод чисел, сброс и отрисовка."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from gigacrew.project.preferences import SETTINGS, Setting
from gigacrew.ui.settings_picker import Change, SettingRow, SettingsState, edit_settings, render

DOWN, UP, RIGHT, ESC = "\x1b[B", "\x1b[A", "\x1b[C", "\x1b"
ENTER, DELETE, BACKSPACE = "\r", "\x1b[3~", "\x7f"

_BY_KEY = MappingProxyType({spec.key: spec for spec in SETTINGS})


class _Recorder:
    """Обработчик изменений: записывает их и применяет к строке."""

    def __init__(self) -> None:
        self.changes: list[tuple[str, Setting]] = []

    def __call__(self, row: SettingRow, setting: Setting) -> Change:
        self.changes.append((row.spec.key, setting))
        return Change(replace(row, setting=setting, stored=True), "сохранено")


def _rows(*keys: str, models: tuple[str, ...] = ()) -> list[SettingRow]:
    return [
        SettingRow(_BY_KEY[key], _BY_KEY[key].default, choices=models)
        for key in keys or [spec.key for spec in SETTINGS]
    ]


def _state(*keys: str, models: tuple[str, ...] = ()) -> tuple[SettingsState, _Recorder]:
    recorder = _Recorder()
    return SettingsState(_rows(*keys, models=models), recorder), recorder


def _text(state: SettingsState, width: int = 200) -> str:
    return "".join(text for _, text in render(state, width))


def test_flag_toggles_and_choice_cycles() -> None:
    state, recorder = _state("permissions.safeCommands", "model.reasoning")
    state.activate()
    assert state.selected.setting is False
    state.move(1)
    state.cycle(1)
    state.cycle(-1)
    assert recorder.changes == [
        ("permissions.safeCommands", False),
        ("model.reasoning", "off"),
        ("model.reasoning", "turn"),
    ]


def _is_editing(state: SettingsState) -> bool:
    """Идёт ли ввод (функция — чтобы mypy не сужал тип после прежних проверок)."""
    return state.editing is not None


def test_number_input_validates_before_saving() -> None:
    state, recorder = _state("context.compactThreshold")
    state.activate()
    assert state.editing == "80"
    state.clear()
    state.type_text("5")
    state.commit()
    assert state.failed
    assert "не меньше 10" in state.status
    assert state.editing == "5"  # ввод остаётся
    state.type_text("0")
    state.commit()
    assert recorder.changes == [("context.compactThreshold", 50)]
    assert not _is_editing(state)
    assert not state.failed


def test_cancel_and_reset() -> None:
    state, recorder = _state("network.timeout")
    state.activate()
    state.erase()
    state.cancel()
    assert state.editing is None
    assert not recorder.changes
    state.reset()
    assert recorder.changes == [("network.timeout", 120)]


def test_model_cycles_catalogue_and_unset() -> None:
    state, recorder = _state("permissions.reviewerModel", models=("A", "B"))
    state.cycle(1)
    state.cycle(1)
    state.cycle(1)
    assert [setting for _, setting in recorder.changes] == ["A", "B", None]


def test_cursor_wraps_and_scrolls_with_section_header() -> None:
    state, _ = _state()
    state.move(-1)
    assert state.selected.spec is SETTINGS[-1]
    assert state.offset > 0
    state.jump(-len(state.rows))
    assert state.cursor == 0
    assert state.offset == 0
    assert isinstance(state.lines[0], str)  # заголовок раздела виден


def test_render_shows_connection_override_and_hints() -> None:
    rows = _rows("model.temperature", "tools.ty")
    rows[1] = replace(rows[1], setting=False, source="~/.gigacode/.env")
    state = SettingsState(rows, _Recorder(), connection=[("адрес", "https://api.test")])
    text = _text(state)
    assert "Подключение" in text
    assert "https://api.test" in text
    assert "Температура" in text
    assert "перекрыто: ~/.gigacode/.env" in text
    assert "GIGACREW_TEMPERATURE" in text
    state.move(1)
    text = _text(state)
    assert "Действует со следующего запуска" in text
    assert "главнее файла" in text


def test_keyboard_session() -> None:
    state, recorder = _state("permissions.safeCommands", "network.timeout", "model.reasoning")
    keys = [" ", DOWN, ENTER, BACKSPACE * 3, "45", ENTER, DOWN, RIGHT, UP, DELETE, ESC]
    with create_pipe_input() as pipe:
        for key in keys:
            pipe.send_text(key)
        edit_settings(state, input=pipe, output=DummyOutput())
    assert recorder.changes == [
        ("permissions.safeCommands", False),
        ("network.timeout", 45),
        ("model.reasoning", "off"),
        ("network.timeout", 120),
    ]
