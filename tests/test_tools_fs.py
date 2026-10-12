"""Тесты файловых инструментов и песочницы."""

from __future__ import annotations

import pytest

from gigacrew.security import SandboxError
from gigacrew.tools.base import ToolError
from gigacrew.tools.fs import (
    EditFileTool,
    FindFilesTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
    repair_escaped_content,
)


def test_write_then_read(ctx):
    w = WriteFileTool()
    res = w.run(w.parse({"path": "a/b.txt", "content": "привет\nмир"}), ctx)
    assert res.ok
    assert (ctx.root / "a" / "b.txt").read_text(encoding="utf-8") == "привет\nмир"

    r = ReadFileTool()
    out = r.run(r.parse({"path": "a/b.txt"}), ctx)
    assert "привет" in out.content
    assert "1\t" in out.content  # номера строк


def test_write_overwrite_and_diff(ctx):
    w = WriteFileTool()
    w.run(w.parse({"path": "f.txt", "content": "old\n"}), ctx)
    params = w.parse({"path": "f.txt", "content": "new\n"})
    preview = w.preview(params, ctx)
    assert preview.kind == "diff"
    assert "-old" in preview.text and "+new" in preview.text
    res = w.run(params, ctx)
    assert "перезаписан" in res.summary


def test_read_missing(ctx):
    r = ReadFileTool()
    with pytest.raises(ToolError):
        r.run(r.parse({"path": "nope.txt"}), ctx)


def test_read_line_range(ctx):
    w = WriteFileTool()
    w.run(w.parse({"path": "n.txt", "content": "l1\nl2\nl3\nl4"}), ctx)
    r = ReadFileTool()
    out = r.run(r.parse({"path": "n.txt", "start_line": 2, "end_line": 3}), ctx)
    assert "l2" in out.content and "l3" in out.content
    assert "l1" not in out.content and "l4" not in out.content


def test_edit_single_occurrence(ctx):
    w = WriteFileTool()
    w.run(w.parse({"path": "c.py", "content": "x = 1\ny = 2\n"}), ctx)
    e = EditFileTool()
    res = e.run(e.parse({"path": "c.py", "old_string": "y = 2", "new_string": "y = 3"}), ctx)
    assert res.ok
    assert (ctx.root / "c.py").read_text() == "x = 1\ny = 3\n"


def test_edit_ambiguous_without_replace_all(ctx):
    w = WriteFileTool()
    w.run(w.parse({"path": "d.txt", "content": "a\na\n"}), ctx)
    e = EditFileTool()
    with pytest.raises(ToolError, match="встречается"):
        e.run(e.parse({"path": "d.txt", "old_string": "a", "new_string": "b"}), ctx)


def test_edit_replace_all(ctx):
    w = WriteFileTool()
    w.run(w.parse({"path": "d.txt", "content": "a\na\n"}), ctx)
    e = EditFileTool()
    res = e.run(
        e.parse({"path": "d.txt", "old_string": "a", "new_string": "b", "replace_all": True}),
        ctx,
    )
    assert res.ok
    assert (ctx.root / "d.txt").read_text() == "b\nb\n"


def test_edit_tolerant_trailing_whitespace(ctx):
    # в файле есть хвостовые пробелы, модель прислала фрагмент без них
    w = WriteFileTool()
    w.run(w.parse({"path": "t.py", "content": "def f():   \n    return 1   \n"}), ctx)
    e = EditFileTool()
    res = e.run(
        e.parse(
            {
                "path": "t.py",
                "old_string": "def f():\n    return 1",
                "new_string": "def f():\n    return 2",
            }
        ),
        ctx,
    )
    assert res.ok
    assert "return 2" in (ctx.root / "t.py").read_text()


def test_edit_tolerant_indentation(ctx):
    # модель прислала фрагмент с другим отступом
    w = WriteFileTool()
    w.run(w.parse({"path": "i.py", "content": "class A:\n        x = 1\n        y = 2\n"}), ctx)
    e = EditFileTool()
    res = e.run(
        e.parse({"path": "i.py", "old_string": "x = 1\ny = 2", "new_string": "x = 10\ny = 20"}),
        ctx,
    )
    assert res.ok
    # отступ файла сохранён — иначе тело класса «выпало» бы (SyntaxError)
    assert (ctx.root / "i.py").read_text() == "class A:\n        x = 10\n        y = 20\n"


def test_edit_with_double_escaped_old_string(ctx):
    # модель прислала old_string с литеральными \n (двойная экранизация)
    w = WriteFileTool()
    w.run(
        w.parse({"path": "m.py", "content": "def a():\n    return 1\n\ndef b():\n    return 2\n"}),
        ctx,
    )
    e = EditFileTool()
    res = e.run(
        e.parse(
            {
                "path": "m.py",
                "old_string": "def b():\\n    return 2",
                "new_string": "def b():\\n    return 22",
            }
        ),
        ctx,
    )
    assert res.ok
    assert "return 22" in (ctx.root / "m.py").read_text()


def test_edit_old_string_with_line_number_prefixes(ctx):
    # модель скопировала old_string ВМЕСТЕ с префиксами номеров строк read_file
    w = WriteFileTool()
    w.run(
        w.parse(
            {
                "path": "k.py",
                "content": "def add(a, b):\n    return a + b\n\ndef divide(a, b):\n    return a / b\n",
            }
        ),
        ctx,
    )
    e = EditFileTool()
    res = e.run(
        e.parse(
            {
                "path": "k.py",
                "old_string": "4\tdef divide(a, b):\n5\t    return a / b",
                "new_string": "4\tdef divide(a, b):\n5\t    return a / b\n6\t\n7\tdef power(b, e):\n8\t    return b ** e",
            }
        ),
        ctx,
    )
    assert res.ok
    text = (ctx.root / "k.py").read_text()
    assert "def power(b, e):" in text
    assert "\t" not in text  # префиксы номеров не попали в файл


def test_edit_not_found_shows_context(ctx):
    w = WriteFileTool()
    w.run(w.parse({"path": "c.py", "content": "alpha\nbeta\ngamma\n"}), ctx)
    e = EditFileTool()
    with pytest.raises(ToolError) as ei:
        e.run(e.parse({"path": "c.py", "old_string": "zzz", "new_string": "q"}), ctx)
    # сообщение об ошибке содержит пронумерованный контекст файла
    msg = str(ei.value)
    assert "alpha" in msg and "1\t" in msg


def test_edit_not_found(ctx):
    w = WriteFileTool()
    w.run(w.parse({"path": "e.txt", "content": "hello"}), ctx)
    e = EditFileTool()
    with pytest.raises(ToolError, match="не найден"):
        e.run(e.parse({"path": "e.txt", "old_string": "zzz", "new_string": "q"}), ctx)


def test_list_dir(ctx):
    w = WriteFileTool()
    w.run(w.parse({"path": "src/main.py", "content": "pass"}), ctx)
    w.run(w.parse({"path": "readme.md", "content": "# hi"}), ctx)
    lister = ListDirTool()
    out = lister.run(lister.parse({"path": "."}), ctx)
    assert "src/" in out.content
    assert "readme.md" in out.content


def test_find_files(ctx):
    w = WriteFileTool()
    for p in ["a.py", "b.py", "c.txt", "sub/d.py"]:
        w.run(w.parse({"path": p, "content": "x"}), ctx)
    f = FindFilesTool()
    out = f.run(f.parse({"pattern": "*.py"}), ctx)
    assert "a.py" in out.content and "sub/d.py" in out.content
    assert "c.txt" not in out.content


def test_repair_double_escaped_content():
    # одна физическая строка с литеральными \n и \" — типичный брак слабой модели
    broken = 'def f():\\n    return "x"\\n'
    fixed, repaired = repair_escaped_content(broken)
    assert repaired is True
    assert fixed == 'def f():\n    return "x"\n'


def test_repair_strips_line_numbers():
    broken = "1\\timport os\\n2\\tx = 1\\n3\\tprint(x)"
    fixed, repaired = repair_escaped_content(broken)
    assert repaired is True
    assert fixed == "import os\nx = 1\nprint(x)"


def test_repair_leaves_normal_content_untouched():
    normal = "def f():\n    return 1\n"
    fixed, repaired = repair_escaped_content(normal)
    assert repaired is False
    assert fixed == normal
    # строка с настоящими переносами и одним литералом \n внутри тоже не трогается
    code = 'print("a\\nb")\nx = 1\n'
    fixed2, repaired2 = repair_escaped_content(code)
    assert repaired2 is False
    assert fixed2 == code


def test_write_file_autorepairs_escaped(ctx):
    w = WriteFileTool()
    broken = 'print("hi")\\nx = 1\\n'
    res = w.run(w.parse({"path": "g.py", "content": broken}), ctx)
    assert "автокоррекция" in res.summary
    assert (ctx.root / "g.py").read_text() == 'print("hi")\nx = 1\n'


def test_sandbox_escape_blocked(ctx):
    w = WriteFileTool()
    with pytest.raises(SandboxError):
        w.run(w.parse({"path": "../escape.txt", "content": "x"}), ctx)
    r = ReadFileTool()
    with pytest.raises(SandboxError):
        r.run(r.parse({"path": "/etc/passwd"}), ctx)


def test_data_dir_check_ignores_case(ctx):
    """На macOS/Windows .GIGACREW — та же папка: запрет не обходится регистром."""
    chats = ctx.root / ".gigacrew" / "chats"
    chats.mkdir(parents=True)
    (chats / "x.json").write_text("{}", encoding="utf-8")
    w, e = WriteFileTool(), EditFileTool()
    for path in (".GIGACREW/chats/x.json", ".GigaCrew/a"):
        with pytest.raises(ToolError, match="Служебная папка"):
            w.run(w.parse({"path": path, "content": "x"}), ctx)
    with pytest.raises(ToolError, match="Служебная папка"):
        e.run(
            e.parse({"path": ".gigacrew/chats/x.json", "old_string": "{}", "new_string": "[]"}),
            ctx,
        )
    assert (chats / "x.json").read_text(encoding="utf-8") == "{}"
    # чтение по-прежнему разрешено, вложенная папка с тем же именем — обычная
    r = ReadFileTool()
    assert r.run(r.parse({"path": ".gigacrew/chats/x.json"}), ctx).ok
    assert w.run(w.parse({"path": "docs/.gigacrew/a.md", "content": "x"}), ctx).ok


def _plan_ctx(ctx):
    from dataclasses import replace

    from gigacrew.tools.base import PlanFile

    return replace(ctx, plan=PlanFile(ctx.workspace.plans_dir / "chat1.md"))


def test_plan_file_is_writable_without_risk(ctx):
    """Файл плана чата — единственный файл .gigacrew/, который можно писать."""
    from gigacrew.security import RiskLevel

    pctx = _plan_ctx(ctx)
    w, e = WriteFileTool(), EditFileTool()
    write = w.parse({"path": ".gigacrew/plans/chat1.md", "content": "# План\n1. a\n"})
    assert w.risk(write, pctx) is RiskLevel.SAFE
    res = w.run(write, pctx)
    assert res.ok and res.summary == "создан план" and pctx.plan.dirty
    # папка данных получила .gitignore — план не попадёт в репозиторий
    assert (ctx.root / ".gigacrew" / ".gitignore").exists()

    pctx.plan.dirty = False
    edit = e.parse({"path": ".gigacrew/plans/chat1.md", "old_string": "1. a", "new_string": "1. b"})
    assert e.risk(edit, pctx) is RiskLevel.SAFE
    assert e.run(edit, pctx).summary == "изменён план" and pctx.plan.dirty
    assert pctx.plan.path.read_text(encoding="utf-8") == "# План\n1. b\n"

    # чужой план, чаты и обычный контекст без плана — по-прежнему закрыты
    for c, path in (
        (pctx, ".gigacrew/plans/other.md"),
        (pctx, ".gigacrew/chats/x.json"),
        (ctx, ".gigacrew/plans/chat1.md"),
    ):
        params = w.parse({"path": path, "content": "x"})
        assert w.risk(params, c) is RiskLevel.WRITE
        with pytest.raises(ToolError, match="Служебная папка"):
            w.run(params, c)


@pytest.mark.parametrize("target", [".git/config", ".gigacrew/chats/x.json"])
def test_symlinked_plan_file_is_not_a_plan(ctx, target):
    """Ссылка на месте файла плана не уводит запись без вопроса в .git/ или к чатам."""
    import os

    from gigacrew.security import RiskLevel

    victim = ctx.root / target
    victim.parent.mkdir(parents=True, exist_ok=True)
    victim.write_text("orig", encoding="utf-8")
    pctx = _plan_ctx(ctx)
    pctx.plan.path.parent.mkdir(parents=True)
    os.symlink(victim, pctx.plan.path)
    w = WriteFileTool()
    params = w.parse({"path": ".gigacrew/plans/chat1.md", "content": "x"})
    assert w.risk(params, pctx) is RiskLevel.WRITE
    with pytest.raises(ToolError):
        w.preview(params, pctx)
    with pytest.raises(ToolError):
        w.run(w.parse({"path": target, "content": "x"}), pctx)
    assert victim.read_text(encoding="utf-8") == "orig" and not pctx.plan.dirty


def test_plan_dir_symlinked_into_git_is_rejected(ctx):
    import os

    git = ctx.root / ".git"
    git.mkdir()
    (ctx.root / ".gigacrew").mkdir()
    os.symlink(git, ctx.workspace.plans_dir)
    pctx = _plan_ctx(ctx)
    w = WriteFileTool()
    with pytest.raises(ToolError, match="Внутренности git"):
        w.run(w.parse({"path": ".gigacrew/plans/chat1.md", "content": "x"}), pctx)
    assert list(git.iterdir()) == []


def test_plan_file_edit_does_not_mark_other_writes(ctx):
    pctx = _plan_ctx(ctx)
    w = WriteFileTool()
    res = w.run(w.parse({"path": "a.txt", "content": "x"}), pctx)
    assert res.summary == "создан a.txt" and not pctx.plan.dirty


def test_data_dir_write_rejected_before_confirmation(tmp_path):
    from fakes import RecordingEvents, ScriptedProvider, text_turn, tool_turn

    from gigacrew.agent.loop import Agent
    from gigacrew.config import Config
    from gigacrew.tools.base import build_default_registry

    provider = ScriptedProvider(
        [
            tool_turn("write_file", {"path": ".gigacrew/chats/a.json", "content": "{}"}),
            text_turn("ок"),
        ]
    )
    events = RecordingEvents()
    cfg = Config(access_key="x", project_root=tmp_path, stream=False)
    Agent(provider, build_default_registry(), cfg, events).run_turn("испорти чат")
    assert events.confirms == []
    assert not (tmp_path / ".gigacrew").exists()
    assert "Служебная папка" in provider.requests[-1]["messages"][-1].content


def test_find_files_rejects_escape_patterns(ctx):
    f = FindFilesTool()
    for pattern in ("../*.toml", "src/../../x", "/etc/*"):
        with pytest.raises(ToolError):
            f.run(f.parse({"pattern": pattern}), ctx)


def test_find_files_glob_semantics_and_ignored_dirs(ctx):
    w = WriteFileTool()
    for path in ("src/a.py", "src/x/b.py", "node_modules/c.py", "z.py"):
        w.run(w.parse({"path": path, "content": "x"}), ctx)
    f = FindFilesTool()
    out = f.run(f.parse({"pattern": "src/**/*.py"}), ctx).content
    assert out.splitlines() == ["src/a.py", "src/x/b.py"]
    out = f.run(f.parse({"pattern": "*.py"}), ctx).content
    assert "node_modules" not in out
    assert out.splitlines() == ["src/a.py", "src/x/b.py", "z.py"]  # отсортировано


def test_find_files_truncation_is_reported(ctx):
    w = WriteFileTool()
    for i in range(5):
        w.run(w.parse({"path": f"f{i}.txt", "content": "x"}), ctx)
    f = FindFilesTool()
    out = f.run(f.parse({"pattern": "*.txt", "max_results": 2}), ctx).content
    assert out.splitlines()[:2] == ["f0.txt", "f1.txt"]
    assert "показано 2 из 5" in out


def test_list_dir_shows_hidden(ctx):
    (ctx.root / ".github").mkdir()
    (ctx.root / ".gitignore").write_text("x", encoding="utf-8")
    lister = ListDirTool()
    out = lister.run(lister.parse({}), ctx).content
    assert ".github/" in out and ".gitignore" in out


@pytest.mark.parametrize("path", [".gigacrew/index/index.sqlite3", ".gigacrew", "./.gigacrew/x"])
def test_data_dir_is_not_writable(ctx, path):
    """Индекс, история и чаты агента — не для правки моделью (отказ ещё в превью)."""
    (ctx.root / ".gigacrew" / "index").mkdir(parents=True)
    (ctx.root / ".gigacrew" / "x").write_text("old", encoding="utf-8")
    w, e = WriteFileTool(), EditFileTool()
    write = w.parse({"path": path, "content": "new"})
    edit = e.parse({"path": path, "old_string": "old", "new_string": "new"})
    for tool, params in ((w, write), (e, edit)):
        with pytest.raises(ToolError, match="Служебная папка"):
            tool.preview(params, ctx)
        with pytest.raises(ToolError, match="Служебная папка"):
            tool.run(params, ctx)
    assert (ctx.root / ".gigacrew" / "x").read_text(encoding="utf-8") == "old"


def test_similar_names_outside_data_dir_are_writable(ctx):
    w = WriteFileTool()
    assert w.run(w.parse({"path": ".gigacrew.md", "content": "ok"}), ctx).ok


def test_data_dir_symlink_target_is_not_writable(ctx):
    """.gigacrew — симлинк на каталог проекта: запись по любому из путей запрещена."""
    import os

    if os.name == "nt":
        pytest.skip("симлинки")
    (ctx.root / "agent-data").mkdir()
    (ctx.root / ".gigacrew").symlink_to(ctx.root / "agent-data", target_is_directory=True)
    w = WriteFileTool()
    for path in (".gigacrew/index.sqlite3", "agent-data/index.sqlite3"):
        with pytest.raises(ToolError, match="Служебная папка"):
            w.preview(w.parse({"path": path, "content": "x"}), ctx)


# ------------------------------ внутренности git ------------------------------ #
def _git_layout(root):
    (root / ".git" / "hooks").mkdir(parents=True)
    (root / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (root / "sub" / ".git").mkdir(parents=True)
    (root / "sub" / ".git" / "config").write_text("[core]\n", encoding="utf-8")


@pytest.mark.parametrize(
    "path",
    [".git/config", ".GIT/config", ".git/hooks/pre-commit", "sub/.git/config", ".git", "./.git/x"],
)
def test_git_internals_are_not_writable(ctx, path):
    """Правка .git/config превратила бы `git status` (без подтверждения) в запуск
    произвольной команды (core.fsmonitor, diff.external) — отказ ещё в превью."""
    _git_layout(ctx.root)
    w, e = WriteFileTool(), EditFileTool()
    write = w.parse({"path": path, "content": "[core]\n\tfsmonitor = touch pwned\n"})
    edit = e.parse({"path": path, "old_string": "[core]", "new_string": "[core]\n\tx = 1"})
    for tool, params in ((w, write), (e, edit)):
        with pytest.raises(ToolError, match="Внутренности git"):
            tool.preview(params, ctx)
        with pytest.raises(ToolError, match="Внутренности git"):
            tool.run(params, ctx)
    assert (ctx.root / ".git" / "config").read_text(encoding="utf-8") == "[core]\n"
    assert (ctx.root / "sub" / ".git" / "config").read_text(encoding="utf-8") == "[core]\n"


def test_git_dir_via_symlink_and_gitdir_file_is_not_writable(ctx):
    import os

    if os.name == "nt":
        pytest.skip("симлинки")
    # раскладка «.git-файл → .bare/»: каталог git лежит в проекте под другим именем
    (ctx.root / ".bare").mkdir()
    (ctx.root / ".bare" / "config").write_text("[core]\n", encoding="utf-8")
    (ctx.root / ".git").write_text("gitdir: ./.bare\n", encoding="utf-8")
    (ctx.root / "alias").symlink_to(ctx.root / ".bare", target_is_directory=True)
    w = WriteFileTool()
    for path in (".bare/config", "alias/config", ".git"):
        with pytest.raises(ToolError, match="Внутренности git"):
            w.preview(w.parse({"path": path, "content": "x"}), ctx)
    assert (ctx.root / ".bare" / "config").read_text(encoding="utf-8") == "[core]\n"


def test_git_like_names_stay_writable(ctx):
    _git_layout(ctx.root)
    w = WriteFileTool()
    for path in (".github/workflows/ci.yml", ".gitignore", ".gitattributes", "docs/.gitkeep"):
        assert w.run(w.parse({"path": path, "content": "x\n"}), ctx).ok


def test_git_config_write_rejected_in_accept_edits_mode(tmp_path):
    """В режиме авто-правок правки не подтверждаются — запись в .git не должна пройти."""
    from fakes import RecordingEvents, ScriptedProvider, text_turn, tool_turn

    from gigacrew.agent.loop import Agent
    from gigacrew.config import Config
    from gigacrew.permissions import PermissionMode
    from gigacrew.tools.base import build_default_registry

    _git_layout(tmp_path)
    provider = ScriptedProvider(
        [
            tool_turn(
                "write_file",
                {"path": ".git/config", "content": "[core]\n\tfsmonitor = touch pwned\n"},
            ),
            text_turn("ок"),
        ]
    )
    events = RecordingEvents()
    cfg = Config(
        access_key="x", project_root=tmp_path, stream=False, mode=PermissionMode.ACCEPT_EDITS
    )
    Agent(provider, build_default_registry(), cfg, events).run_turn("настрой git")
    assert events.confirms == []
    assert (tmp_path / ".git" / "config").read_text(encoding="utf-8") == "[core]\n"
    assert "Внутренности git" in provider.requests[-1]["messages"][-1].content


# ------------------------- переводы строк, BOM, запись ------------------------- #
def _edit(ctx, path, old, new, **kw):
    e = EditFileTool()
    return e.run(e.parse({"path": path, "old_string": old, "new_string": new, **kw}), ctx)


def test_edit_keeps_crlf_line_endings(ctx):
    (ctx.root / "w.txt").write_bytes(b"one\r\ntwo\r\nthree\r\n")
    _edit(ctx, "w.txt", "two", "TWO")
    assert (ctx.root / "w.txt").read_bytes() == b"one\r\nTWO\r\nthree\r\n"
    # многострочный фрагмент от модели — с \n; несторогое совпадение (хвостовые пробелы)
    _edit(ctx, "w.txt", "one\nTWO", "1\n2")
    assert (ctx.root / "w.txt").read_bytes() == b"1\r\n2\r\nthree\r\n"
    (ctx.root / "t.txt").write_bytes(b"a  \r\nb\r\nc\r\n")
    _edit(ctx, "t.txt", "a\nb", "x\ny")
    assert (ctx.root / "t.txt").read_bytes() == b"x\r\ny\r\nc\r\n"


def test_write_over_crlf_file_keeps_its_line_endings(ctx):
    (ctx.root / "w.bat").write_bytes(b"@echo off\r\necho 1\r\n")
    w = WriteFileTool()
    params = w.parse({"path": "w.bat", "content": "@echo off\necho 2\n"})
    preview = w.preview(params, ctx)
    assert "\r" not in preview.text and "-echo 1\n+echo 2\n" in preview.text
    w.run(params, ctx)
    assert (ctx.root / "w.bat").read_bytes() == b"@echo off\r\necho 2\r\n"


def test_new_files_and_lf_files_stay_lf(ctx):
    w = WriteFileTool()
    w.run(w.parse({"path": "n.py", "content": "a = 1\nb = 2\n"}), ctx)
    assert (ctx.root / "n.py").read_bytes() == b"a = 1\nb = 2\n"
    _edit(ctx, "n.py", "b = 2", "b = 3")
    assert (ctx.root / "n.py").read_bytes() == b"a = 1\nb = 3\n"


def test_edit_keeps_bom(ctx):
    (ctx.root / "b.cs").write_bytes("﻿using System;\r\nclass A {}\r\n".encode())
    _edit(ctx, "b.cs", "using System;", "using System.IO;")
    assert (ctx.root / "b.cs").read_bytes() == "﻿using System.IO;\r\nclass A {}\r\n".encode()


def test_mixed_line_endings_are_unified_with_a_note(ctx):
    (ctx.root / "m.txt").write_bytes(b"a\r\nb\r\nc\nd\r\n")
    res = _edit(ctx, "m.txt", "a", "A")
    assert (ctx.root / "m.txt").read_bytes() == b"A\r\nb\r\nc\r\nd\r\n"
    assert "приведены к CRLF" in res.content


def test_make_diff_marks_missing_final_newline():
    from gigacrew.tools.fs import make_diff

    assert make_diff("a", "b", "x") == (
        "--- a/x\n+++ b/x\n@@ -1 +1 @@\n"
        "-a\n\\ No newline at end of file\n+b\n\\ No newline at end of file\n"
    )
    assert make_diff("a\n", "a\nb", "x").endswith("+b\n\\ No newline at end of file\n")


def test_make_diff_splits_only_on_newlines():
    """\\x0c, \\x85, \\u2028 — не переводы строк: строки диффа не склеиваются."""
    from gigacrew.tools.fs import make_diff

    diff = make_diff("a\x0cb\nc d\nend\n", "a\x0cb\nC d\nend\n", "x")
    lines = [line for line in diff.split("\n")[2:] if line]
    assert lines and all(line[:1] in " +-@\\" for line in lines)
    assert "-c d" in diff and "+C d" in diff


def test_interrupted_write_leaves_file_intact(ctx, monkeypatch):
    import os

    (ctx.root / "k.py").write_text("x = 1\n", encoding="utf-8")

    def interrupted(src, dst):
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _edit(ctx, "k.py", "x = 1", "x = 2")
    assert (ctx.root / "k.py").read_text(encoding="utf-8") == "x = 1\n"
    assert sorted(p.name for p in ctx.root.iterdir()) == ["k.py"]  # временный файл убран


def test_edit_keeps_file_mode(ctx):
    import os

    if os.name == "nt":
        pytest.skip("права POSIX")
    script = ctx.root / "run.sh"
    script.write_text("echo 1\n", encoding="utf-8")
    script.chmod(0o755)
    _edit(ctx, "run.sh", "echo 1", "echo 2")
    assert script.stat().st_mode & 0o777 == 0o755


# -------------------------- edit_file: пустой фрагмент и отступы -------------------------- #
@pytest.mark.parametrize("replace_all", [False, True])
def test_edit_rejects_empty_old_string(ctx, replace_all):
    (ctx.root / "e.txt").write_text("abc\n", encoding="utf-8")
    e = EditFileTool()
    params = e.parse(
        {"path": "e.txt", "old_string": "", "new_string": "X", "replace_all": replace_all}
    )
    for call in (e.preview, e.run):
        with pytest.raises(ToolError, match="old_string пуст"):
            call(params, ctx)
    assert (ctx.root / "e.txt").read_text(encoding="utf-8") == "abc\n"


def test_edit_tolerant_removes_extra_indentation(ctx):
    (ctx.root / "f.py").write_text("def f():\n    a = 1\n    return a\n", encoding="utf-8")
    _edit(ctx, "f.py", "        a = 1\n        return a", "        a = 2\n        return a")
    assert (ctx.root / "f.py").read_text() == "def f():\n    a = 2\n    return a\n"


def test_edit_tolerant_keeps_nested_structure(ctx):
    (ctx.root / "n.py").write_text("class A:\n    def f(self):\n        pass\n", encoding="utf-8")
    _edit(
        ctx,
        "n.py",
        "def f(self):\n    pass",
        "def f(self):\n    if self:\n        return 1\n    return 0",
    )
    assert (ctx.root / "n.py").read_text() == (
        "class A:\n    def f(self):\n        if self:\n            return 1\n        return 0\n"
    )


def test_edit_tolerant_maps_spaces_to_tabs(ctx):
    (ctx.root / "t.go").write_text("func f() {\n\tif x {\n\t\ty()\n\t}\n}\n", encoding="utf-8")
    _edit(
        ctx,
        "t.go",
        "    if x {\n        y()\n    }",
        "    if x {\n        y()\n        z()\n    }",
    )
    assert (ctx.root / "t.go").read_text() == ("func f() {\n\tif x {\n\t\ty()\n\t\tz()\n\t}\n}\n")


def test_edit_tolerant_refuses_inconsistent_indentation(ctx):
    original = "if a:\n    b = 1\n    if b:\n        c = 2\n"
    (ctx.root / "i.py").write_text(original, encoding="utf-8")
    e = EditFileTool()
    # один и тот же отступ шаблона соответствует разным отступам файла — не угадываем
    params = e.parse(
        {"path": "i.py", "old_string": "b = 1\nif b:\nc = 2", "new_string": "b = 1\nif b:\nc = 3"}
    )
    with pytest.raises(ToolError, match="не найден"):
        e.run(params, ctx)
    assert (ctx.root / "i.py").read_text() == original


def test_edit_tolerant_does_not_add_blank_lines(ctx):
    (ctx.root / "b.py").write_text("def f():\n    return 1   \nz = 0\n", encoding="utf-8")
    _edit(ctx, "b.py", "def f():\n    return 1\n", "def f():\n    return 2\n")
    assert (ctx.root / "b.py").read_text() == "def f():\n    return 2\nz = 0\n"
    (ctx.root / "c.py").write_text("a = 1  \nb = 2\n", encoding="utf-8")
    _edit(ctx, "c.py", "\na = 1\nb = 2", "\na = 10\nb = 2")
    assert (ctx.root / "c.py").read_text() == "a = 10\nb = 2\n"


# ------------------------- не-UTF-8, бинарные, не файлы ------------------------- #
@pytest.mark.parametrize(
    "data", [b"\x89PNG\r\n\x1a\n\x00\x00", "привет, мир\n".encode("cp1251")], ids=["bin", "cp1251"]
)
def test_write_and_edit_refuse_binary_and_non_utf8(ctx, data):
    (ctx.root / "f.dat").write_bytes(data)
    w, e = WriteFileTool(), EditFileTool()
    write = w.parse({"path": "f.dat", "content": "новое\n"})
    edit = e.parse({"path": "f.dat", "old_string": "x", "new_string": "y"})
    for tool, params in ((w, write), (e, edit)):
        for call in (tool.preview, tool.run):
            with pytest.raises(ToolError, match="Бинарный|не в UTF-8"):
                call(params, ctx)
    assert (ctx.root / "f.dat").read_bytes() == data


def test_write_refuses_directory_in_preview(ctx):
    (ctx.root / "d").mkdir()
    w = WriteFileTool()
    with pytest.raises(ToolError, match="директория"):
        w.preview(w.parse({"path": "d", "content": "x"}), ctx)


@pytest.mark.skipif(not hasattr(__import__("os"), "mkfifo"), reason="FIFO")
def test_write_refuses_fifo(ctx):
    import os

    os.mkfifo(ctx.root / "pipe")
    w = WriteFileTool()
    with pytest.raises(ToolError, match="Не обычный файл"):
        w.run(w.parse({"path": "pipe", "content": "x"}), ctx)


def test_write_preview_reports_unchanged_content(ctx):
    (ctx.root / "same.txt").write_text("a\n", encoding="utf-8")
    w = WriteFileTool()
    assert w.preview(w.parse({"path": "same.txt", "content": "a\n"}), ctx).text == "(без изменений)"
    assert w.preview(w.parse({"path": "new.txt", "content": ""}), ctx).text == "(новый пустой файл)"


def test_read_range_reports_total_lines(ctx):
    (ctx.root / "r.txt").write_text("".join(f"l{i}\n" for i in range(1, 11)), encoding="utf-8")
    r = ReadFileTool()
    out = r.run(r.parse({"path": "r.txt", "start_line": 1, "end_line": 3}), ctx).content
    assert out.endswith("… показаны строки 1–3 из 10. Продолжение: start_line=4.")
    tail = r.run(r.parse({"path": "r.txt", "start_line": 9}), ctx).content
    assert tail.endswith("… показаны строки 9–10 из 10.")
    whole = r.run(r.parse({"path": "r.txt"}), ctx).content
    assert "показаны строки" not in whole


def test_read_cp1251_file(ctx):
    (ctx.root / "old.txt").write_bytes("Привет\r\nмир\r\n".encode("cp1251"))
    r = ReadFileTool()
    res = r.run(r.parse({"path": "old.txt"}), ctx)
    assert "1\tПривет" in res.content and "2\tмир" in res.content
    assert "показан как cp1251" in res.content and "cp1251" in res.summary


def test_hard_linked_file_is_edited_in_place(ctx):
    import os

    if not hasattr(os, "link"):
        pytest.skip("жёсткие ссылки")
    (ctx.root / "a.txt").write_text("x = 1\n", encoding="utf-8")
    os.link(ctx.root / "a.txt", ctx.root / "b.txt")
    _edit(ctx, "a.txt", "x = 1", "x = 2")
    assert (ctx.root / "b.txt").read_text(encoding="utf-8") == "x = 2\n"  # ссылка цела


def test_file_with_xattrs_is_edited_in_place(ctx):
    import os

    if not hasattr(os, "setxattr"):
        pytest.skip("xattr")
    path = ctx.root / "acl.txt"
    path.write_text("x = 1\n", encoding="utf-8")
    try:
        os.setxattr(path, "user.gigacrew", b"1")
    except OSError:
        pytest.skip("ФС без user xattr")
    inode = path.stat().st_ino
    _edit(ctx, "acl.txt", "x = 1", "x = 2")
    assert path.stat().st_ino == inode and os.getxattr(path, "user.gigacrew") == b"1"


def _fake_git_dir(directory):
    (directory / "objects").mkdir(parents=True)
    (directory / "refs").mkdir()
    (directory / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (directory / "config").write_text("[core]\n", encoding="utf-8")


def test_nested_repository_git_dir_is_not_writable(ctx):
    """Вложенный репозиторий, чей .git — симлинк или gitdir:-файл на каталог в проекте:
    ни путь через .git, ни прямой путь к его каталогу git не пишутся."""
    import os

    if os.name == "nt":
        pytest.skip("симлинки")
    _fake_git_dir(ctx.root / "nested" / "store")
    (ctx.root / "nested" / ".git").symlink_to(ctx.root / "nested" / "store")
    _fake_git_dir(ctx.root / "store2")
    (ctx.root / "other").mkdir()
    (ctx.root / "other" / ".git").write_text("gitdir: ../store2\n", encoding="utf-8")
    _fake_git_dir(ctx.root / "bare.git")
    w = WriteFileTool()
    for path in (
        "nested/.git/config",
        "nested/store/config",
        "nested/store/hooks/pre-commit",
        "store2/config",
        "bare.git/config",
    ):
        with pytest.raises(ToolError, match="Внутренности git"):
            w.preview(w.parse({"path": path, "content": "x"}), ctx)
    assert (ctx.root / "nested" / "store" / "config").read_text(encoding="utf-8") == "[core]\n"
    # каталог с одним файлом HEAD — не каталог git
    (ctx.root / "docs").mkdir()
    (ctx.root / "docs" / "HEAD").write_text("x", encoding="utf-8")
    assert w.run(w.parse({"path": "docs/HEAD", "content": "y"}), ctx).ok
