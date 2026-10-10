"""Тесты индикатора ожидания (ракета с пламенем)."""

from __future__ import annotations

from gigacrew.ui.spinner import FRAMES, INTERVAL, WIDTH, rocket_frame


def test_frames_have_constant_width():
    assert FRAMES
    assert all(frame.cell_len == WIDTH for frame in FRAMES)


def test_rocket_crosses_the_whole_line():
    positions = {frame.plain.index("➤") for frame in FRAMES if "➤" in frame.plain}
    assert positions == set(range(WIDTH))


def test_frames_are_deterministic_and_loop():
    assert rocket_frame(0) is FRAMES[0]
    assert rocket_frame(INTERVAL * 1.5) is FRAMES[1]
    assert rocket_frame(INTERVAL * len(FRAMES)) is FRAMES[0]
