"""Тесты файловых инструментов и песочницы."""

from __future__ import annotations

import json
import pathlib

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
    assert "-old" in preview.text
    assert "+new" in preview.text
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
    assert "l2" in out.content
    assert "l3" in out.content
    assert "l1" not in out.content
    assert "l4" not in out.content


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


@pytest.mark.parametrize("trailing", ["", "  "], ids=["indent", "rstrip"])
def test_edit_tolerant_match_must_be_unique(ctx, trailing):
    # одинаковые блоки в двух функциях: несторогое совпадение тоже должно быть единственным
    text = f"def a():\n    x = 1{trailing}\n    y = 2\ndef b():\n    x = 1{trailing}\n    y = 2\n"
    (ctx.root / "u.py").write_text(text, encoding="utf-8")
    old = "x = 1\ny = 2" if not trailing else "    x = 1\n    y = 2"
    with pytest.raises(ToolError, match="не найден"):
        _edit(ctx, "u.py", old, old.replace("1", "9"))
    assert (ctx.root / "u.py").read_text() == text


def test_edit_same_old_and_new_is_refused(ctx):
    (ctx.root / "s.txt").write_text("a\n", encoding="utf-8")
    mtime = (ctx.root / "s.txt").stat().st_mtime_ns
    with pytest.raises(ToolError, match="совпадают"):
        _edit(ctx, "s.txt", "a", "a")
    assert (ctx.root / "s.txt").stat().st_mtime_ns == mtime


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
            },
        ),
        ctx,
    )
    assert res.ok
    assert (ctx.root / "t.py").read_text() == "def f():\n    return 2\n"


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
            },
        ),
        ctx,
    )
    assert res.ok
    assert (ctx.root / "m.py").read_text() == "def a():\n    return 1\n\ndef b():\n    return 22\n"


def test_edit_double_escaped_keeps_escapes_inside_strings(ctx):
    # двойная экранизация фрагмента с \n внутри строки: \\n — экранированная \ и n,
    # а не перевод строки (разэкранирование — за один проход слева направо)
    (ctx.root / "p.py").write_text('def f():\n    print("done")\n', encoding="utf-8")
    _edit(
        ctx,
        "p.py",
        'def f():\\n    print(\\"done\\")',
        'def f():\\n    print(\\"done\\\\n\\")',
    )
    assert (ctx.root / "p.py").read_text() == 'def f():\n    print("done\\n")\n'


def test_edit_old_string_with_line_number_prefixes(ctx):
    # модель скопировала old_string ВМЕСТЕ с префиксами номеров строк read_file
    w = WriteFileTool()
    w.run(
        w.parse(
            {
                "path": "k.py",
                "content": "def add(a, b):\n    return a + b\n\ndef divide(a, b):\n    return a / b\n",
            },
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
            },
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
    assert "alpha" in msg
    assert "1\t" in msg


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
    assert "a.py" in out.content
    assert "sub/d.py" in out.content
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


@pytest.mark.parametrize(
    "content",
    [
        'print("a\\nb\\nc")',  # одна строка Python — верна как есть
        '{"greeting":"Hello\\nWorld","bye":"See\\nyou","q":"say \\"hi\\""}',  # JSON
        "printf 'a\\nb'",  # один литерал \n
        "echo ok\\n",  # один литерал, пусть и в конце, — не повод переписывать файл
        'x = "a\\n"\ny = "b\\n"\nz = "c\\n"\n',  # настоящие переносы и три литерала
    ],
    ids=["one-line-code", "minified-json", "single-escape", "single-trailing", "multiline"],
)
def test_repair_leaves_valid_content_untouched(ctx, content):
    assert repair_escaped_content(content) == (content, False)
    w = WriteFileTool()
    res = w.run(w.parse({"path": "v.txt", "content": content}), ctx)
    assert "автокоррекция" not in res.summary
    assert (ctx.root / "v.txt").read_bytes() == content.encode("utf-8")


def test_repair_decodes_escaped_backslash_in_one_pass():
    # задуманный файл: print("a\n") с переводом строки внутри строкового литерала
    intended = 'print("a\\n")\nx = 1\n'
    broken = json.dumps(intended)[1:-1]  # модель экранировала всё ещё раз
    assert repair_escaped_content(broken) == (intended, True)


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
    """На macOS/Windows .GIGACODE — та же папка: запрет не обходится регистром."""
    chats = ctx.root / ".gigacode" / "chats"
    chats.mkdir(parents=True)
    (chats / "x.json").write_text("{}", encoding="utf-8")
    w, e = WriteFileTool(), EditFileTool()
    for path in (".GIGACODE/chats/x.json", ".GigaCode/a"):
        with pytest.raises(ToolError, match="Служебная папка"):
            w.run(w.parse({"path": path, "content": "x"}), ctx)
    with pytest.raises(ToolError, match="Служебная папка"):
        e.run(
            e.parse({"path": ".gigacode/chats/x.json", "old_string": "{}", "new_string": "[]"}),
            ctx,
        )
    assert (chats / "x.json").read_text(encoding="utf-8") == "{}"
    # чтение по-прежнему разрешено, вложенная папка с тем же именем — обычная
    r = ReadFileTool()
    assert r.run(r.parse({"path": ".gigacode/chats/x.json"}), ctx).ok
    assert w.run(w.parse({"path": "docs/.gigacode/a.md", "content": "x"}), ctx).ok


def _plan_ctx(ctx):
    from dataclasses import replace

    from gigacrew.tools.base import PlanFile

    return replace(ctx, plan=PlanFile(ctx.workspace.plans_dir / "chat1.md"))


def test_plan_file_is_writable_without_risk(ctx):
    """Файл плана чата — единственный файл .gigacode/, который можно писать."""
    from gigacrew.security import RiskLevel

    pctx = _plan_ctx(ctx)
    w, e = WriteFileTool(), EditFileTool()
    write = w.parse({"path": ".gigacode/plans/chat1.md", "content": "# План\n1. a\n"})
    assert w.risk(write, pctx) is RiskLevel.SAFE
    res = w.run(write, pctx)
    assert res.ok
    assert res.summary == "создан план"
    assert pctx.plan.dirty
    # папка данных получила .gitignore — план не попадёт в репозиторий
    assert (ctx.root / ".gigacode" / ".gitignore").exists()

    pctx.plan.dirty = False
    edit = e.parse({"path": ".gigacode/plans/chat1.md", "old_string": "1. a", "new_string": "1. b"})
    assert e.risk(edit, pctx) is RiskLevel.SAFE
    assert e.run(edit, pctx).summary == "изменён план"
    assert pctx.plan.dirty
    assert pctx.plan.path.read_text(encoding="utf-8") == "# План\n1. b\n"

    # чужой план, чаты и обычный контекст без плана — по-прежнему закрыты
    for c, path in (
        (pctx, ".gigacode/plans/other.md"),
        (pctx, ".gigacode/chats/x.json"),
        (pctx, ".gigacode/skills/demo/SKILL.md"),  # скиллы агент не пишет
        (ctx, ".gigacode/plans/chat1.md"),
    ):
        params = w.parse({"path": path, "content": "x"})
        assert w.risk(params, c) is RiskLevel.WRITE
        with pytest.raises(ToolError, match="Служебная папка"):
            w.run(params, c)


@pytest.mark.parametrize("target", [".git/config", ".gigacode/chats/x.json"])
def test_symlinked_plan_file_is_not_a_plan(ctx, target):
    """Ссылка на месте файла плана не уводит запись без вопроса в .git/ или к чатам."""
    from gigacrew.security import RiskLevel

    victim = ctx.root / target
    victim.parent.mkdir(parents=True, exist_ok=True)
    victim.write_text("orig", encoding="utf-8")
    pctx = _plan_ctx(ctx)
    pctx.plan.path.parent.mkdir(parents=True)
    pathlib.Path(pctx.plan.path).symlink_to(victim)
    w = WriteFileTool()
    params = w.parse({"path": ".gigacode/plans/chat1.md", "content": "x"})
    assert w.risk(params, pctx) is RiskLevel.WRITE
    with pytest.raises(ToolError):
        w.preview(params, pctx)
    with pytest.raises(ToolError):
        w.run(w.parse({"path": target, "content": "x"}), pctx)
    assert victim.read_text(encoding="utf-8") == "orig"
    assert not pctx.plan.dirty


def test_plan_dir_symlinked_into_git_is_rejected(ctx):

    git = ctx.root / ".git"
    git.mkdir()
    (ctx.root / ".gigacode").mkdir()
    pathlib.Path(ctx.workspace.plans_dir).symlink_to(git)
    pctx = _plan_ctx(ctx)
    w = WriteFileTool()
    with pytest.raises(ToolError, match="Внутренности git"):
        w.run(w.parse({"path": ".gigacode/plans/chat1.md", "content": "x"}), pctx)
    assert list(git.iterdir()) == []


def test_plan_under_symlinked_data_dir_is_not_a_plan(ctx):
    """.gigacode — ссылка на каталог проекта: файл плана под ней — не план, запись туда —
    правка проекта (без «безопасно» и без вопроса — нельзя).
    """
    import os

    from gigacrew.security import RiskLevel

    if os.name == "nt":
        pytest.skip("симлинки")
    (ctx.root / "src").mkdir()
    (ctx.root / ".gigacode").symlink_to(ctx.root / "src", target_is_directory=True)
    pctx = _plan_ctx(ctx)
    w = WriteFileTool()
    params = w.parse({"path": ".gigacode/plans/chat1.md", "content": "x"})
    assert w.risk(params, pctx) is RiskLevel.WRITE
    with pytest.raises(ToolError, match="Служебная папка"):
        w.run(params, pctx)
    assert not (ctx.root / "src" / "plans").exists()
    assert not pctx.plan.dirty


def test_plan_file_edit_does_not_mark_other_writes(ctx):
    pctx = _plan_ctx(ctx)
    w = WriteFileTool()
    res = w.run(w.parse({"path": "a.txt", "content": "x"}), pctx)
    assert res.summary == "создан a.txt"
    assert not pctx.plan.dirty


def test_data_dir_write_rejected_before_confirmation(tmp_path):
    from fakes import RecordingEvents, ScriptedProvider, text_turn, tool_turn

    from gigacrew.agent.loop import Agent
    from gigacrew.config import Config
    from gigacrew.tools.base import build_default_registry

    provider = ScriptedProvider(
        [
            tool_turn("write_file", {"path": ".gigacode/chats/a.json", "content": "{}"}),
            text_turn("ок"),
        ],
    )
    events = RecordingEvents()
    cfg = Config(access_key="x", project_root=tmp_path, stream=False)
    Agent(provider, build_default_registry(), cfg, events).run_turn("испорти чат")
    assert events.confirms == []
    assert not (tmp_path / ".gigacode").exists()
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
    assert ".github/" in out
    assert ".gitignore" in out


@pytest.mark.parametrize("path", [".gigacode/index/index.sqlite3", ".gigacode", "./.gigacode/x"])
def test_data_dir_is_not_writable(ctx, path):
    """Индекс, история и чаты агента — не для правки моделью (отказ ещё в превью)."""
    (ctx.root / ".gigacode" / "index").mkdir(parents=True)
    (ctx.root / ".gigacode" / "x").write_text("old", encoding="utf-8")
    w, e = WriteFileTool(), EditFileTool()
    write = w.parse({"path": path, "content": "new"})
    edit = e.parse({"path": path, "old_string": "old", "new_string": "new"})
    for tool, params in ((w, write), (e, edit)):
        with pytest.raises(ToolError, match="Служебная папка"):
            tool.preview(params, ctx)
        with pytest.raises(ToolError, match="Служебная папка"):
            tool.run(params, ctx)
    assert (ctx.root / ".gigacode" / "x").read_text(encoding="utf-8") == "old"


def test_similar_names_outside_data_dir_are_writable(ctx):
    w = WriteFileTool()
    assert w.run(w.parse({"path": ".gigacode.md", "content": "ok"}), ctx).ok


def test_data_dir_symlink_target_is_not_writable(ctx):
    """.gigacode — симлинк на каталог проекта: запись по любому из путей запрещена."""
    import os

    if os.name == "nt":
        pytest.skip("симлинки")
    (ctx.root / "agent-data").mkdir()
    (ctx.root / ".gigacode").symlink_to(ctx.root / "agent-data", target_is_directory=True)
    w = WriteFileTool()
    for path in (".gigacode/index.sqlite3", "agent-data/index.sqlite3"):
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
    произвольной команды (core.fsmonitor, diff.external) — отказ ещё в превью.
    """
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
        ],
    )
    events = RecordingEvents()
    cfg = Config(access_key="x", project_root=tmp_path, stream=False, mode=PermissionMode.AUTO)
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


def test_edit_cr_only_file(ctx):
    # старые файлы Mac: многострочный фрагмент от модели — с \n
    (ctx.root / "mac.txt").write_bytes(b"a\rb\rc\r")
    _edit(ctx, "mac.txt", "a\nb", "A\nB")
    assert (ctx.root / "mac.txt").read_bytes() == b"A\rB\rc\r"
    (ctx.root / "mix.txt").write_bytes(b"a\rb\r\nc\r")
    assert "приведены к CR" in _edit(ctx, "mix.txt", "c", "C").content
    assert (ctx.root / "mix.txt").read_bytes() == b"a\rb\rC\r"


def test_write_over_crlf_file_keeps_its_line_endings(ctx):
    (ctx.root / "w.bat").write_bytes(b"@echo off\r\necho 1\r\n")
    w = WriteFileTool()
    params = w.parse({"path": "w.bat", "content": "@echo off\necho 2\n"})
    preview = w.preview(params, ctx)
    assert "\r" not in preview.text
    assert "-echo 1\n+echo 2\n" in preview.text
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
    r"""\\x0c, \\x85, \\u2028 — не переводы строк: строки диффа не склеиваются."""
    from gigacrew.tools.fs import make_diff

    diff = make_diff("a\x0cb\nc d\nend\n", "a\x0cb\nC d\nend\n", "x")
    lines = [line for line in diff.split("\n")[2:] if line]
    assert lines
    assert all(line[:1] in " +-@\\" for line in lines)
    assert "-c d" in diff
    assert "+C d" in diff


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
        {"path": "e.txt", "old_string": "", "new_string": "X", "replace_all": replace_all},
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
        {"path": "i.py", "old_string": "b = 1\nif b:\nc = 2", "new_string": "b = 1\nif b:\nc = 3"},
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
    ("data", "error"),
    [
        (b"abc\0def\n", "Бинарный"),  # NUL при верном UTF-8 (например, UTF-16LE с ASCII)
        (b"\x89PNG\r\n\x1a\n\x00\x00", "Бинарный"),
        ("привет, мир\n".encode("cp1251"), "не в UTF-8"),
    ],
    ids=["nul", "png", "cp1251"],
)
def test_write_and_edit_refuse_binary_and_non_utf8(ctx, data, error):
    (ctx.root / "f.dat").write_bytes(data)
    w, e = WriteFileTool(), EditFileTool()
    write = w.parse({"path": "f.dat", "content": "новое\n"})
    edit = e.parse({"path": "f.dat", "old_string": "abc", "new_string": "y"})
    for tool, params in ((w, write), (e, edit)):
        for call in (tool.preview, tool.run):
            with pytest.raises(ToolError, match=error):
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
    assert "1\tПривет" in res.content
    assert "2\tмир" in res.content
    assert "показан как cp1251" in res.content
    assert "cp1251" in res.summary


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
        os.setxattr(path, "user.gigacode", b"1")
    except OSError:
        pytest.skip("ФС без user xattr")
    inode = path.stat().st_ino
    _edit(ctx, "acl.txt", "x = 1", "x = 2")
    assert path.stat().st_ino == inode
    assert os.getxattr(path, "user.gigacode") == b"1"


def _fake_git_dir(directory):
    (directory / "objects").mkdir(parents=True)
    (directory / "refs").mkdir()
    (directory / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (directory / "config").write_text("[core]\n", encoding="utf-8")


def test_nested_repository_git_dir_is_not_writable(ctx):
    """Вложенный репозиторий, чей .git — симлинк или gitdir:-файл на каталог в проекте:
    ни путь через .git, ни прямой путь к его каталогу git не пишутся.
    """
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


@pytest.mark.parametrize(
    "order",
    [
        ("config", "objects/.keep", "refs/.keep", "HEAD"),
        ("HEAD", "config", "refs/heads/main", "objects/info/x"),
        ("objects/x", "HEAD", "config", "refs/x"),
    ],
)
def test_git_dir_cannot_be_built_file_by_file(ctx, order):
    """Каталог git нельзя собрать по файлу: запись, которая завершила бы раскладку
    (HEAD, objects/, refs/), отклоняется в любом порядке — иначе config, записанный
    первым, стал бы настройкой репозитория, который git найдёт в этом каталоге.
    """
    w = WriteFileTool()
    *first, last = order
    for name in first:
        assert w.run(w.parse({"path": f"sub/{name}", "content": "x\n"}), ctx).ok
    with pytest.raises(ToolError, match="Внутренности git"):
        w.preview(w.parse({"path": f"sub/{last}", "content": "x\n"}), ctx)
    assert not (ctx.root / "sub" / last).exists()


# ------------------------------ файлы с секретами ------------------------------ #
def _decide(tool, params, ctx, mode="auto", **kw):
    from gigacrew.permissions import PermissionMode, decide

    assessment = tool.assess(params, ctx)
    kw.setdefault("auto_approve", False)
    return decide(PermissionMode(mode), assessment.risk, assessment.kind or tool.kind, **kw)


@pytest.mark.parametrize("path", [".env", "sub/.env", ".ENV", "link.txt"])
def test_env_write_always_asks(ctx, path):
    """.env — настройки самого gigacrew (ключ, адреса API, TLS): правка — только с
    подтверждения, в любом режиме и даже с -y или разрешением хука.
    """
    import os

    from gigacrew.permissions import Decision
    from gigacrew.security import RiskLevel

    if path == "link.txt":
        if os.name == "nt":
            pytest.skip("симлинки")
        (ctx.root / ".env").write_text("GIGACREW_API_URL=https://api.test\n", encoding="utf-8")
        (ctx.root / "link.txt").symlink_to(ctx.root / ".env")
    w, e = WriteFileTool(), EditFileTool()
    calls = [
        (w, w.parse({"path": path, "content": "GIGACREW_API_URL=https://evil.test\n"})),
        (e, e.parse({"path": path, "old_string": "api.test", "new_string": "evil.test"})),
    ]
    for tool, params in calls:
        assessment = tool.assess(params, ctx)
        assert assessment.risk is RiskLevel.DANGEROUS
        assert ".env" in assessment.reason.casefold()
        assert _decide(tool, params, ctx) is Decision.ASK
        assert _decide(tool, params, ctx, auto_approve=True) is Decision.ASK
        assert _decide(tool, params, ctx, auto_approve=True, hook="allow") is Decision.ASK
        assert _decide(tool, params, ctx, mode="plan") is Decision.BLOCK


@pytest.mark.parametrize("path", [".env.local", ".npmrc", "deploy.pem", ".ssh/config"])
def test_secret_file_write_asks_in_auto_mode(ctx, path):
    from gigacrew.permissions import Decision

    w = WriteFileTool()
    params = w.parse({"path": path, "content": "x\n"})
    assert "секрет" in w.assess(params, ctx).reason
    assert _decide(w, params, ctx) is Decision.ASK  # правка обычного файла — без вопроса
    assert _decide(w, params, ctx, auto_approve=True) is Decision.ALLOW  # -y — как у run_shell


def test_ordinary_and_template_files_are_plain_edits(ctx):
    from gigacrew.permissions import Decision

    w = WriteFileTool()
    for path in (".env.example", "src/env.py", "environment.md"):
        params = w.parse({"path": path, "content": "x\n"})
        assert w.assess(params, ctx).reason == ""
        assert _decide(w, params, ctx) is Decision.ALLOW


def test_env_edit_in_auto_mode_is_confirmed(tmp_path):
    """Сквозной путь: модель правит .env в режиме «авто» с -y — пользователя спрашивают."""
    from fakes import RecordingEvents, ScriptedProvider, text_turn, tool_turn

    from gigacrew.agent.loop import Agent
    from gigacrew.config import Config
    from gigacrew.tools.base import build_default_registry

    env = tmp_path / ".env"
    env.write_text("GIGACREW_AUTH_URL=https://auth.test/oauth\n", encoding="utf-8")
    edit = {"path": ".env", "old_string": "auth.test", "new_string": "evil.test"}
    provider = ScriptedProvider([tool_turn("edit_file", edit), text_turn("ок")])
    events = RecordingEvents(confirm_answer=False)
    cfg = Config(access_key="x", project_root=tmp_path, stream=False, auto_approve=True)
    Agent(provider, build_default_registry(), cfg, events).run_turn("поправь адрес")
    assert len(events.confirms) == 1
    assert events.confirms[0][2]
    assert env.read_text(encoding="utf-8") == "GIGACREW_AUTH_URL=https://auth.test/oauth\n"


@pytest.mark.parametrize(
    "path",
    [".env", ".Env.Local", "deploy.pem", ".npmrc", "credentials.json", ".ssh/id_rsa"],
)
def test_read_file_refuses_secret_files(ctx, path):
    target = ctx.root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("TOKEN=s3cr3t\n", encoding="utf-8")
    r = ReadFileTool()
    with pytest.raises(ToolError, match="секрет") as err:
        r.run(r.parse({"path": path}), ctx)
    assert "s3cr3t" not in str(err.value)
    assert "run_shell" in str(err.value)


def test_read_file_refuses_symlink_to_secret_and_reads_template(ctx):
    import os

    (ctx.root / ".env").write_text("TOKEN=s3cr3t\n", encoding="utf-8")
    (ctx.root / ".env.example").write_text("TOKEN=\n", encoding="utf-8")
    r = ReadFileTool()
    assert "TOKEN=" in r.run(r.parse({"path": ".env.example"}), ctx).content
    if os.name == "nt":
        pytest.skip("симлинки")
    (ctx.root / "settings.txt").symlink_to(ctx.root / ".env")
    with pytest.raises(ToolError, match="секрет"):
        r.run(r.parse({"path": "settings.txt"}), ctx)


def test_search_skips_and_refuses_secret_files(ctx):
    from gigacrew.tools.search import SearchContentTool

    for path in ("src/a.py", ".env", ".ENV.local", ".npmrc", "deploy.pem", "credentials.json"):
        target = ctx.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("needle=s3cr3t\n", encoding="utf-8")
    (ctx.root / ".env.example").write_text("needle=\n", encoding="utf-8")
    s = SearchContentTool()
    out = s.run(s.parse({"pattern": "needle"}), ctx).content
    assert sorted(ln.split(":", 1)[0] for ln in out.splitlines()) == [".env.example", "src/a.py"]
    for path in (".env", ".npmrc"):  # явно названный файл с секретами — тоже нет
        with pytest.raises(ToolError, match="секрет") as err:
            s.run(s.parse({"pattern": "needle", "path": path}), ctx)
        assert "s3cr3t" not in str(err.value)


def test_edit_not_found_in_secret_file_shows_no_content(ctx):
    """Неудачная правка показывает модели файл — но не файл с секретами (ошибка превью
    уходит модели до вопроса о подтверждении).
    """
    (ctx.root / ".env").write_text("TOKEN=s3cr3t\n", encoding="utf-8")
    with pytest.raises(ToolError, match="не найден") as err:
        _edit(ctx, ".env", "MISSING=1", "MISSING=2")
    assert "s3cr3t" not in str(err.value)


def test_search_ignore_case(ctx):
    from gigacrew.tools.search import SearchContentTool

    (ctx.root / "a.txt").write_text("Needle\n", encoding="utf-8")
    s = SearchContentTool()
    assert (
        s.run(s.parse({"pattern": "needle", "ignore_case": True}), ctx).content == "a.txt:1:Needle"
    )
    assert s.run(s.parse({"pattern": "needle"}), ctx).content == "(совпадений не найдено)"
