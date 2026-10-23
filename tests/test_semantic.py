"""Тесты вспомогательной части интеграции с ty: позиции, уведомления, состояние."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NoReturn

import pytest

from gigacrew.project import semantic
from gigacrew.project.workspace import Workspace


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16", "utf-32"])
def test_column_conversion_roundtrip(encoding: str) -> None:
    text = "ёж = '😀'; obj.метод()"
    for col in range(len(text) + 1):
        units = semantic._to_units(text, col, encoding)
        assert semantic._from_units(text, units, encoding) == col
    assert semantic._to_units(text, 2, "utf-8") == 4  # «ёж» — 4 байта
    assert semantic._to_units(text, 8, "utf-16") == 9  # эмодзи — суррогатная пара


def _file_name(uri: str) -> str:
    """Имя файла — последняя часть URI."""
    return uri.rsplit("/", 1)[-1]


def test_notify_changes_sends_only_python_and_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = semantic.TyServer(tmp_path, binary="ty")
    sent: list[dict[str, Any]] = []
    monkeypatch.setattr(server._client, "notify", lambda method, params: sent.append(params))
    server.notify_changes(("a.py", "b.md"), ("pyproject.toml", "c.pyi"), ("d.py", "e.txt"))
    changes = [(_file_name(change["uri"]), change["type"]) for change in sent[0]["changes"]]
    assert changes == [("a.py", 1), ("pyproject.toml", 2), ("c.pyi", 2), ("d.py", 3)]
    sent.clear()
    server.notify_changes(("README.md",), (), ())
    assert sent == []


def test_status_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = Workspace(tmp_path)
    assert semantic.status(ws).startswith("не запущен")

    def missing() -> NoReturn:
        raise semantic.SemanticUnavailableError("пакет ty не установлен")

    monkeypatch.setattr(semantic, "find_binary", missing)
    with pytest.raises(semantic.SemanticUnavailableError):
        semantic.server_for(ws)
    semantic.shutdown_all()


def test_location_decodes_uri_once(tmp_path: Path) -> None:
    server = semantic.TyServer(tmp_path, binary="ty")
    target = tmp_path / "a%41.py"  # «%41» в имени файла — не «A»
    target.write_text("x = 1\n", encoding="utf-8")
    location = server._location(target.as_uri(), {"line": 0, "character": 0}, semantic._Lines())
    assert (location.path, location.in_project) == ("a%41.py", True)


def test_interrupted_start_does_not_leave_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[semantic.TyServer] = []

    def interrupted(self: semantic.TyServer) -> NoReturn:
        raise KeyboardInterrupt

    # Функция, а не closed.append: атрибут класса получает self при вызове метода.
    def record_close(server: semantic.TyServer) -> None:
        closed.append(server)

    monkeypatch.setattr(semantic, "find_binary", lambda: "ty")
    monkeypatch.setattr(semantic.TyServer, "start", interrupted)
    monkeypatch.setattr(semantic.TyServer, "close", record_close)
    with pytest.raises(KeyboardInterrupt):
        semantic.server_for(Workspace(tmp_path))
    assert len(closed) == 1
    semantic.shutdown_all()


def test_lines_split_like_lsp(tmp_path: Path) -> None:
    path = tmp_path / "f.py"
    path.write_text("#\f\né = 1\n", encoding="utf-8")
    assert semantic._Lines().get(path, 2) == "é = 1"  # \f — не разрыв строки


def test_location_keeps_unc_host(tmp_path: Path) -> None:
    server = semantic.TyServer(tmp_path, binary="ty")
    location = server._location(
        "file://server/share/a.py",
        {"line": 0, "character": 0},
        semantic._Lines(),
    )
    assert location.path.replace("\\", "/").endswith("server/share/a.py")
    assert not location.in_project
