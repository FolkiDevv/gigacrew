"""Тесты файловых инструментов и песочницы."""

from __future__ import annotations

import json
import os
import pathlib
from dataclasses import replace
from typing import Any, Literal, NoReturn

import pytest
from fakes import RecordingEvents, ScriptedProvider, text_turn, tool_turn
from pydantic import BaseModel

from gigacrew.agent.loop import Agent
from gigacrew.config import Config
from gigacrew.permissions import Decision, PermissionMode, decide
from gigacrew.security import RiskLevel, SandboxError
from gigacrew.tools.base import (
    PlanFile,
    Tool,
    ToolContext,
    ToolError,
    ToolResult,
    build_default_registry,
)
from gigacrew.tools.fs import (
    EditFileTool,
    FindFilesTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
    make_diff,
    repair_escaped_content,
)
from gigacrew.tools.search import SearchContentTool

# Маркер unified diff: у последней строки нет перевода строки.
_NO_NEWLINE_MARK = r"\ No newline at end of file"
# Литерал \n в тексте файла: два символа — обратная косая черта и «n».
_ESCAPED_NL = r"\n"

# Инструмент и разобранные параметры его вызова.
type _ToolCall = tuple[Tool[Any], BaseModel]


def test_write_then_read(ctx: ToolContext) -> None:
    writer = WriteFileTool()
    res = writer.run(writer.parse({"path": "a/b.txt", "content": "привет\nмир"}), ctx)
    assert res.ok
    assert (ctx.root / "a" / "b.txt").read_text(encoding="utf-8") == "привет\nмир"

    reader = ReadFileTool()
    out = reader.run(reader.parse({"path": "a/b.txt"}), ctx)
    assert "привет" in out.content
    assert "1\t" in out.content  # номера строк


def test_write_overwrite_and_diff(ctx: ToolContext) -> None:
    writer = WriteFileTool()
    writer.run(writer.parse({"path": "f.txt", "content": "old\n"}), ctx)
    params = writer.parse({"path": "f.txt", "content": "new\n"})
    preview = writer.preview(params, ctx)
    assert preview is not None
    assert preview.kind == "diff"
    assert "-old" in preview.text
    assert "+new" in preview.text
    res = writer.run(params, ctx)
    assert "перезаписан" in res.summary


def test_read_missing(ctx: ToolContext) -> None:
    reader = ReadFileTool()
    with pytest.raises(ToolError):
        reader.run(reader.parse({"path": "nope.txt"}), ctx)


def test_read_line_range(ctx: ToolContext) -> None:
    writer = WriteFileTool()
    writer.run(writer.parse({"path": "n.txt", "content": "l1\nl2\nl3\nl4"}), ctx)
    reader = ReadFileTool()
    out = reader.run(reader.parse({"path": "n.txt", "start_line": 2, "end_line": 3}), ctx)
    assert "l2" in out.content
    assert "l3" in out.content
    assert "l1" not in out.content
    assert "l4" not in out.content


def test_edit_single_occurrence(ctx: ToolContext) -> None:
    writer = WriteFileTool()
    writer.run(writer.parse({"path": "c.py", "content": "x = 1\ny = 2\n"}), ctx)
    editor = EditFileTool()
    res = editor.run(
        editor.parse({"path": "c.py", "old_string": "y = 2", "new_string": "y = 3"}),
        ctx,
    )
    assert res.ok
    assert (ctx.root / "c.py").read_text() == "x = 1\ny = 3\n"


def test_edit_ambiguous_without_replace_all(ctx: ToolContext) -> None:
    writer = WriteFileTool()
    writer.run(writer.parse({"path": "d.txt", "content": "a\na\n"}), ctx)
    editor = EditFileTool()
    with pytest.raises(ToolError, match="встречается"):
        editor.run(editor.parse({"path": "d.txt", "old_string": "a", "new_string": "b"}), ctx)


@pytest.mark.parametrize("trailing", ["", "  "], ids=["indent", "rstrip"])
def test_edit_tolerant_match_must_be_unique(ctx: ToolContext, trailing: str) -> None:
    # одинаковые блоки в двух функциях: несторогое совпадение тоже должно быть единственным
    text = f"def a():\n    x = 1{trailing}\n    y = 2\ndef b():\n    x = 1{trailing}\n    y = 2\n"
    (ctx.root / "u.py").write_text(text, encoding="utf-8")
    old = "    x = 1\n    y = 2" if trailing else "x = 1\ny = 2"
    with pytest.raises(ToolError, match="не найден"):
        _edit(ctx, "u.py", old, old.replace("1", "9"))
    assert (ctx.root / "u.py").read_text() == text


def test_edit_same_old_and_new_is_refused(ctx: ToolContext) -> None:
    (ctx.root / "s.txt").write_text("a\n", encoding="utf-8")
    mtime = (ctx.root / "s.txt").stat().st_mtime_ns
    with pytest.raises(ToolError, match="совпадают"):
        _edit(ctx, "s.txt", "a", "a")
    assert (ctx.root / "s.txt").stat().st_mtime_ns == mtime


def test_edit_replace_all(ctx: ToolContext) -> None:
    writer = WriteFileTool()
    writer.run(writer.parse({"path": "d.txt", "content": "a\na\n"}), ctx)
    editor = EditFileTool()
    res = editor.run(
        editor.parse({"path": "d.txt", "old_string": "a", "new_string": "b", "replace_all": True}),
        ctx,
    )
    assert res.ok
    assert (ctx.root / "d.txt").read_text() == "b\nb\n"


def test_edit_tolerant_trailing_whitespace(ctx: ToolContext) -> None:
    # в файле есть хвостовые пробелы, модель прислала фрагмент без них
    writer = WriteFileTool()
    writer.run(writer.parse({"path": "t.py", "content": "def f():   \n    return 1   \n"}), ctx)
    editor = EditFileTool()
    res = editor.run(
        editor.parse(
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


def test_edit_tolerant_indentation(ctx: ToolContext) -> None:
    # модель прислала фрагмент с другим отступом
    writer = WriteFileTool()
    writer.run(
        writer.parse({"path": "i.py", "content": "class A:\n        x = 1\n        y = 2\n"}),
        ctx,
    )
    editor = EditFileTool()
    res = editor.run(
        editor.parse(
            {"path": "i.py", "old_string": "x = 1\ny = 2", "new_string": "x = 10\ny = 20"},
        ),
        ctx,
    )
    assert res.ok
    # отступ файла сохранён — иначе тело класса «выпало» бы (SyntaxError)
    assert (ctx.root / "i.py").read_text() == "class A:\n        x = 10\n        y = 20\n"


def test_edit_with_double_escaped_old_string(ctx: ToolContext) -> None:
    # модель прислала old_string с литеральными \n (двойная экранизация)
    writer = WriteFileTool()
    writer.run(
        writer.parse(
            {"path": "m.py", "content": "def a():\n    return 1\n\ndef b():\n    return 2\n"},
        ),
        ctx,
    )
    editor = EditFileTool()
    res = editor.run(
        editor.parse(
            {
                "path": "m.py",
                "old_string": r"def b():\n    return 2",
                "new_string": r"def b():\n    return 22",
            },
        ),
        ctx,
    )
    assert res.ok
    assert (ctx.root / "m.py").read_text() == "def a():\n    return 1\n\ndef b():\n    return 22\n"


def test_edit_double_escaped_keeps_escapes_inside_strings(ctx: ToolContext) -> None:
    # двойная экранизация фрагмента с \n внутри строки: \\n — экранированная \ и n,
    # а не перевод строки (разэкранирование — за один проход слева направо)
    (ctx.root / "p.py").write_text('def f():\n    print("done")\n', encoding="utf-8")
    _edit(
        ctx,
        "p.py",
        r"def f():\n    print(\"done\")",
        r"def f():\n    print(\"done\\n\")",
    )
    assert (ctx.root / "p.py").read_text() == f'def f():\n    print("done{_ESCAPED_NL}")\n'


def test_edit_old_string_with_line_number_prefixes(ctx: ToolContext) -> None:
    # модель скопировала old_string ВМЕСТЕ с префиксами номеров строк read_file
    writer = WriteFileTool()
    writer.run(
        writer.parse(
            {
                "path": "k.py",
                "content": (
                    "def add(a, b):\n    return a + b\n\ndef divide(a, b):\n    return a / b\n"
                ),
            },
        ),
        ctx,
    )
    editor = EditFileTool()
    res = editor.run(
        editor.parse(
            {
                "path": "k.py",
                "old_string": "4\tdef divide(a, b):\n5\t    return a / b",
                "new_string": (
                    "4\tdef divide(a, b):\n5\t    return a / b\n6\t\n"
                    "7\tdef power(b, e):\n8\t    return b ** e"
                ),
            },
        ),
        ctx,
    )
    assert res.ok
    text = (ctx.root / "k.py").read_text()
    assert "def power(b, e):" in text
    assert "\t" not in text  # префиксы номеров не попали в файл


def test_edit_not_found_shows_context(ctx: ToolContext) -> None:
    writer = WriteFileTool()
    writer.run(writer.parse({"path": "c.py", "content": "alpha\nbeta\ngamma\n"}), ctx)
    editor = EditFileTool()
    with pytest.raises(ToolError) as exc_info:
        editor.run(editor.parse({"path": "c.py", "old_string": "zzz", "new_string": "q"}), ctx)
    # сообщение об ошибке содержит пронумерованный контекст файла
    assert "alpha" in str(exc_info.value)
    assert "1\t" in str(exc_info.value)


def test_edit_not_found(ctx: ToolContext) -> None:
    writer = WriteFileTool()
    writer.run(writer.parse({"path": "e.txt", "content": "hello"}), ctx)
    editor = EditFileTool()
    with pytest.raises(ToolError, match="не найден"):
        editor.run(editor.parse({"path": "e.txt", "old_string": "zzz", "new_string": "q"}), ctx)


def test_list_dir(ctx: ToolContext) -> None:
    writer = WriteFileTool()
    writer.run(writer.parse({"path": "src/main.py", "content": "pass"}), ctx)
    writer.run(writer.parse({"path": "readme.md", "content": "# hi"}), ctx)
    lister = ListDirTool()
    out = lister.run(lister.parse({"path": "."}), ctx)
    assert "src/" in out.content
    assert "readme.md" in out.content


def test_find_files(ctx: ToolContext) -> None:
    writer = WriteFileTool()
    for path in ("a.py", "b.py", "c.txt", "sub/d.py"):
        writer.run(writer.parse({"path": path, "content": "x"}), ctx)
    finder = FindFilesTool()
    out = finder.run(finder.parse({"pattern": "*.py"}), ctx)
    assert "a.py" in out.content
    assert "sub/d.py" in out.content
    assert "c.txt" not in out.content


def test_repair_double_escaped_content() -> None:
    # одна физическая строка с литеральными \n и \" — типичный брак слабой модели
    broken = r'def f():\n    return "x"\n'
    fixed, repaired = repair_escaped_content(broken)
    assert repaired is True
    assert fixed == 'def f():\n    return "x"\n'


def test_repair_strips_line_numbers() -> None:
    broken = r"1\timport os\n2\tx = 1\n3\tprint(x)"
    fixed, repaired = repair_escaped_content(broken)
    assert repaired is True
    assert fixed == "import os\nx = 1\nprint(x)"


def test_repair_leaves_normal_content_untouched() -> None:
    normal = "def f():\n    return 1\n"
    fixed, repaired = repair_escaped_content(normal)
    assert repaired is False
    assert fixed == normal
    # строка с настоящими переносами и одним литералом \n внутри тоже не трогается
    code = f'print("a{_ESCAPED_NL}b")\nx = 1\n'
    fixed2, repaired2 = repair_escaped_content(code)
    assert repaired2 is False
    assert fixed2 == code


@pytest.mark.parametrize(
    "content",
    [
        r'print("a\nb\nc")',  # одна строка Python — верна как есть
        r'{"greeting":"Hello\nWorld","bye":"See\nyou","q":"say \"hi\""}',  # JSON
        r"printf 'a\nb'",  # один литерал \n
        r"echo ok\n",  # один литерал, пусть и в конце, — не повод переписывать файл
        # настоящие переносы и три литерала
        f'x = "a{_ESCAPED_NL}"\ny = "b{_ESCAPED_NL}"\nz = "c{_ESCAPED_NL}"\n',
    ],
    ids=["one-line-code", "minified-json", "single-escape", "single-trailing", "multiline"],
)
def test_repair_leaves_valid_content_untouched(ctx: ToolContext, content: str) -> None:
    assert repair_escaped_content(content) == (content, False)
    writer = WriteFileTool()
    res = writer.run(writer.parse({"path": "v.txt", "content": content}), ctx)
    assert "автокоррекция" not in res.summary
    assert (ctx.root / "v.txt").read_bytes() == content.encode("utf-8")


def test_repair_decodes_escaped_backslash_in_one_pass() -> None:
    # задуманный файл: print("a\n") с переводом строки внутри строкового литерала
    intended = f'print("a{_ESCAPED_NL}")\nx = 1\n'
    broken = json.dumps(intended)[1:-1]  # модель экранировала всё ещё раз
    assert repair_escaped_content(broken) == (intended, True)


def test_write_file_autorepairs_escaped(ctx: ToolContext) -> None:
    writer = WriteFileTool()
    broken = r'print("hi")\nx = 1\n'
    res = writer.run(writer.parse({"path": "g.py", "content": broken}), ctx)
    assert "автокоррекция" in res.summary
    assert (ctx.root / "g.py").read_text() == 'print("hi")\nx = 1\n'


def test_sandbox_escape_blocked(ctx: ToolContext) -> None:
    writer = WriteFileTool()
    with pytest.raises(SandboxError):
        writer.run(writer.parse({"path": "../escape.txt", "content": "x"}), ctx)
    reader = ReadFileTool()
    with pytest.raises(SandboxError):
        reader.run(reader.parse({"path": "/etc/passwd"}), ctx)


def test_data_dir_check_ignores_case(ctx: ToolContext) -> None:
    """На macOS/Windows .GIGACODE — та же папка: запрет не обходится регистром."""
    chats = ctx.root / ".gigacode" / "chats"
    chats.mkdir(parents=True)
    (chats / "x.json").write_text("{}", encoding="utf-8")
    writer, editor = WriteFileTool(), EditFileTool()
    for path in (".GIGACODE/chats/x.json", ".GigaCode/a"):
        with pytest.raises(ToolError, match="Служебная папка"):
            writer.run(writer.parse({"path": path, "content": "x"}), ctx)
    with pytest.raises(ToolError, match="Служебная папка"):
        editor.run(
            editor.parse(
                {"path": ".gigacode/chats/x.json", "old_string": "{}", "new_string": "[]"},
            ),
            ctx,
        )
    assert (chats / "x.json").read_text(encoding="utf-8") == "{}"
    # чтение по-прежнему разрешено, вложенная папка с тем же именем — обычная
    reader = ReadFileTool()
    assert reader.run(reader.parse({"path": ".gigacode/chats/x.json"}), ctx).ok
    assert writer.run(writer.parse({"path": "docs/.gigacode/a.md", "content": "x"}), ctx).ok


def _plan_ctx(ctx: ToolContext) -> tuple[ToolContext, PlanFile]:
    plan = PlanFile(ctx.workspace.plans_dir / "chat1.md")
    return replace(ctx, plan=plan), plan


def test_plan_file_is_writable_without_risk(ctx: ToolContext) -> None:
    """Файл плана чата — единственный файл .gigacode/, который можно писать."""
    pctx, plan = _plan_ctx(ctx)
    writer, editor = WriteFileTool(), EditFileTool()
    write = writer.parse({"path": ".gigacode/plans/chat1.md", "content": "# План\n1. a\n"})
    assert writer.risk(write, pctx) is RiskLevel.SAFE
    res = writer.run(write, pctx)
    assert res.ok
    assert res.summary == "создан план"
    assert plan.dirty
    # папка данных получила .gitignore — план не попадёт в репозиторий
    assert (ctx.root / ".gigacode" / ".gitignore").exists()

    plan.dirty = False
    edit = editor.parse(
        {"path": ".gigacode/plans/chat1.md", "old_string": "1. a", "new_string": "1. b"},
    )
    assert editor.risk(edit, pctx) is RiskLevel.SAFE
    assert editor.run(edit, pctx).summary == "изменён план"
    assert plan.dirty
    assert plan.path.read_text(encoding="utf-8") == "# План\n1. b\n"

    # чужой план, чаты и обычный контекст без плана — по-прежнему закрыты
    for check_ctx, path in (
        (pctx, ".gigacode/plans/other.md"),
        (pctx, ".gigacode/chats/x.json"),
        (pctx, ".gigacode/skills/demo/SKILL.md"),  # скиллы агент не пишет
        (ctx, ".gigacode/plans/chat1.md"),
    ):
        params = writer.parse({"path": path, "content": "x"})
        assert writer.risk(params, check_ctx) is RiskLevel.WRITE
        with pytest.raises(ToolError, match="Служебная папка"):
            writer.run(params, check_ctx)


@pytest.mark.parametrize("target", [".git/config", ".gigacode/chats/x.json"])
def test_symlinked_plan_file_is_not_a_plan(ctx: ToolContext, target: str) -> None:
    """Ссылка на месте файла плана не уводит запись без вопроса в .git/ или к чатам."""
    victim = ctx.root / target
    victim.parent.mkdir(parents=True, exist_ok=True)
    victim.write_text("orig", encoding="utf-8")
    pctx, plan = _plan_ctx(ctx)
    plan.path.parent.mkdir(parents=True)
    pathlib.Path(plan.path).symlink_to(victim)
    writer = WriteFileTool()
    params = writer.parse({"path": ".gigacode/plans/chat1.md", "content": "x"})
    assert writer.risk(params, pctx) is RiskLevel.WRITE
    with pytest.raises(ToolError):
        writer.preview(params, pctx)
    with pytest.raises(ToolError):
        writer.run(writer.parse({"path": target, "content": "x"}), pctx)
    assert victim.read_text(encoding="utf-8") == "orig"
    assert not plan.dirty


def test_plan_dir_symlinked_into_git_is_rejected(ctx: ToolContext) -> None:
    git = ctx.root / ".git"
    git.mkdir()
    (ctx.root / ".gigacode").mkdir()
    pathlib.Path(ctx.workspace.plans_dir).symlink_to(git)
    pctx, _ = _plan_ctx(ctx)
    writer = WriteFileTool()
    with pytest.raises(ToolError, match="Внутренности git"):
        writer.run(writer.parse({"path": ".gigacode/plans/chat1.md", "content": "x"}), pctx)
    assert not list(git.iterdir())


def test_plan_under_symlinked_data_dir_is_not_a_plan(ctx: ToolContext) -> None:
    """.gigacode — ссылка на каталог проекта: файл плана под ней — не план.

    Запись туда — правка проекта (без «безопасно» и без вопроса — нельзя).
    """
    if os.name == "nt":
        pytest.skip("симлинки")
    (ctx.root / "src").mkdir()
    (ctx.root / ".gigacode").symlink_to(ctx.root / "src", target_is_directory=True)
    pctx, plan = _plan_ctx(ctx)
    writer = WriteFileTool()
    params = writer.parse({"path": ".gigacode/plans/chat1.md", "content": "x"})
    assert writer.risk(params, pctx) is RiskLevel.WRITE
    with pytest.raises(ToolError, match="Служебная папка"):
        writer.run(params, pctx)
    assert not (ctx.root / "src" / "plans").exists()
    assert not plan.dirty


def test_plan_file_edit_does_not_mark_other_writes(ctx: ToolContext) -> None:
    pctx, plan = _plan_ctx(ctx)
    writer = WriteFileTool()
    res = writer.run(writer.parse({"path": "a.txt", "content": "x"}), pctx)
    assert res.summary == "создан a.txt"
    assert not plan.dirty


def test_temp_dir_is_writable(ctx: ToolContext) -> None:
    """В .gigacode/tmp/ модель пишет и правит, как в проекте; папка — со своим .gitignore."""
    writer, editor = WriteFileTool(), EditFileTool()
    writer.run(writer.parse({"path": ".gigacode/tmp/notes.txt", "content": "old"}), ctx)
    edit = {"path": ".gigacode/tmp/notes.txt", "old_string": "old", "new_string": "new"}
    editor.run(editor.parse(edit), ctx)
    temp_dir = ctx.workspace.data_dir / "tmp"
    assert (temp_dir / "notes.txt").read_text(encoding="utf-8") == "new"
    assert (temp_dir / ".gitignore").is_file()
    with pytest.raises(ToolError, match="Служебная папка"):
        writer.preview(writer.parse({"path": ".gigacode/tmp/../x.txt", "content": "x"}), ctx)


@pytest.mark.skipif(os.name == "nt", reason="симлинки")
def test_symlinked_temp_dir_is_not_writable(ctx: ToolContext) -> None:
    """Симлинк tmp → skills из репозитория не открывает запись в служебную папку."""
    skills = ctx.workspace.data_dir / "skills"
    skills.mkdir(parents=True)
    (ctx.workspace.data_dir / "tmp").symlink_to(skills, target_is_directory=True)
    writer = WriteFileTool()
    with pytest.raises(ToolError, match="Служебная папка"):
        writer.run(writer.parse({"path": ".gigacode/tmp/SKILL.md", "content": "x"}), ctx)
    assert not any(skills.iterdir())


def test_data_dir_write_rejected_before_confirmation(tmp_path: pathlib.Path) -> None:
    provider = ScriptedProvider(
        [
            tool_turn("write_file", {"path": ".gigacode/chats/a.json", "content": "{}"}),
            text_turn("ок"),
        ],
    )
    events = RecordingEvents()
    cfg = Config(access_key="x", project_root=tmp_path, stream=False)
    Agent(provider, build_default_registry(), cfg, events).run_turn("испорти чат")
    assert not events.confirms
    assert not (tmp_path / ".gigacode").exists()
    messages = provider.requests[-1]["messages"]
    assert "Служебная папка" in messages[-1].content


def test_find_files_rejects_escape_patterns(ctx: ToolContext) -> None:
    finder = FindFilesTool()
    for pattern in ("../*.toml", "src/../../x", "/etc/*"):
        with pytest.raises(ToolError):
            finder.run(finder.parse({"pattern": pattern}), ctx)


def test_find_files_glob_semantics_and_ignored_dirs(ctx: ToolContext) -> None:
    writer = WriteFileTool()
    for path in ("src/a.py", "src/x/b.py", "node_modules/c.py", "z.py"):
        writer.run(writer.parse({"path": path, "content": "x"}), ctx)
    finder = FindFilesTool()
    out = finder.run(finder.parse({"pattern": "src/**/*.py"}), ctx).content
    assert out.splitlines() == ["src/a.py", "src/x/b.py"]
    out = finder.run(finder.parse({"pattern": "*.py"}), ctx).content
    assert "node_modules" not in out
    assert out.splitlines() == ["src/a.py", "src/x/b.py", "z.py"]  # отсортировано


def test_find_files_truncation_is_reported(ctx: ToolContext) -> None:
    writer = WriteFileTool()
    for num in range(5):
        writer.run(writer.parse({"path": f"f{num}.txt", "content": "x"}), ctx)
    finder = FindFilesTool()
    out = finder.run(finder.parse({"pattern": "*.txt", "max_results": 2}), ctx).content
    assert out.splitlines()[:2] == ["f0.txt", "f1.txt"]
    assert "показано 2 из 5" in out


def test_list_dir_shows_hidden(ctx: ToolContext) -> None:
    (ctx.root / ".github").mkdir()
    (ctx.root / ".gitignore").write_text("x", encoding="utf-8")
    lister = ListDirTool()
    out = lister.run(lister.parse({}), ctx).content
    assert ".github/" in out
    assert ".gitignore" in out


@pytest.mark.parametrize("path", [".gigacode/index/index.sqlite3", ".gigacode", "./.gigacode/x"])
def test_data_dir_is_not_writable(ctx: ToolContext, path: str) -> None:
    """Индекс, история и чаты агента — не для правки моделью (отказ ещё в превью)."""
    (ctx.root / ".gigacode" / "index").mkdir(parents=True)
    (ctx.root / ".gigacode" / "x").write_text("old", encoding="utf-8")
    writer, editor = WriteFileTool(), EditFileTool()
    write = writer.parse({"path": path, "content": "new"})
    edit = editor.parse({"path": path, "old_string": "old", "new_string": "new"})
    cases: tuple[_ToolCall, ...] = ((writer, write), (editor, edit))
    for tool, params in cases:
        with pytest.raises(ToolError, match="Служебная папка"):
            tool.preview(params, ctx)
        with pytest.raises(ToolError, match="Служебная папка"):
            tool.run(params, ctx)
    assert (ctx.root / ".gigacode" / "x").read_text(encoding="utf-8") == "old"


def test_similar_names_outside_data_dir_are_writable(ctx: ToolContext) -> None:
    writer = WriteFileTool()
    assert writer.run(writer.parse({"path": ".gigacode.md", "content": "ok"}), ctx).ok


def test_data_dir_symlink_target_is_not_writable(ctx: ToolContext) -> None:
    """.gigacode — симлинк на каталог проекта: запись по любому из путей запрещена."""
    if os.name == "nt":
        pytest.skip("симлинки")
    (ctx.root / "agent-data").mkdir()
    (ctx.root / ".gigacode").symlink_to(ctx.root / "agent-data", target_is_directory=True)
    writer = WriteFileTool()
    for path in (".gigacode/index.sqlite3", "agent-data/index.sqlite3"):
        with pytest.raises(ToolError, match="Служебная папка"):
            writer.preview(writer.parse({"path": path, "content": "x"}), ctx)


# ------------------------------ внутренности git ------------------------------ #
def _git_layout(root: pathlib.Path) -> None:
    (root / ".git" / "hooks").mkdir(parents=True)
    (root / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (root / "sub" / ".git").mkdir(parents=True)
    (root / "sub" / ".git" / "config").write_text("[core]\n", encoding="utf-8")


@pytest.mark.parametrize(
    "path",
    [
        ".git/config",
        ".GIT/config",
        ".git/hooks/pre-commit",
        "sub/.git/config",
        ".git",
        "./.git/x",
        ".jj/repo/config.toml",
        ".JJ/repo/config-id",
        "sub/.jj/workspace-config.toml",
    ],
)
def test_git_internals_are_not_writable(ctx: ToolContext, path: str) -> None:
    """Внутренности git и jj не пишутся: отказ ещё в превью.

    Правка .git/config превратила бы `git status` (без подтверждения) в запуск
    произвольной команды (core.fsmonitor, diff.external), настроек jj в .jj/ — `jj diff`
    (ui.diff-formatter).
    """
    _git_layout(ctx.root)
    writer, editor = WriteFileTool(), EditFileTool()
    write = writer.parse({"path": path, "content": "[core]\n\tfsmonitor = touch pwned\n"})
    edit = editor.parse({"path": path, "old_string": "[core]", "new_string": "[core]\n\tx = 1"})
    cases: tuple[_ToolCall, ...] = ((writer, write), (editor, edit))
    for tool, params in cases:
        with pytest.raises(ToolError, match="Внутренности git"):
            tool.preview(params, ctx)
        with pytest.raises(ToolError, match="Внутренности git"):
            tool.run(params, ctx)
    assert (ctx.root / ".git" / "config").read_text(encoding="utf-8") == "[core]\n"
    assert (ctx.root / "sub" / ".git" / "config").read_text(encoding="utf-8") == "[core]\n"


def test_git_dir_via_symlink_and_gitdir_file_is_not_writable(ctx: ToolContext) -> None:
    if os.name == "nt":
        pytest.skip("симлинки")
    # раскладка «.git-файл → .bare/»: каталог git лежит в проекте под другим именем
    (ctx.root / ".bare").mkdir()
    (ctx.root / ".bare" / "config").write_text("[core]\n", encoding="utf-8")
    (ctx.root / ".git").write_text("gitdir: ./.bare\n", encoding="utf-8")
    (ctx.root / "alias").symlink_to(ctx.root / ".bare", target_is_directory=True)
    writer = WriteFileTool()
    for path in (".bare/config", "alias/config", ".git"):
        with pytest.raises(ToolError, match="Внутренности git"):
            writer.preview(writer.parse({"path": path, "content": "x"}), ctx)
    assert (ctx.root / ".bare" / "config").read_text(encoding="utf-8") == "[core]\n"


def test_git_like_names_stay_writable(ctx: ToolContext) -> None:
    _git_layout(ctx.root)
    writer = WriteFileTool()
    for path in (".github/workflows/ci.yml", ".gitignore", ".gitattributes", "docs/.gitkeep"):
        assert writer.run(writer.parse({"path": path, "content": "x\n"}), ctx).ok


def test_git_config_write_rejected_in_accept_edits_mode(tmp_path: pathlib.Path) -> None:
    """В режиме авто-правок правки не подтверждаются — запись в .git не должна пройти."""
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
    assert not events.confirms
    assert (tmp_path / ".git" / "config").read_text(encoding="utf-8") == "[core]\n"
    messages = provider.requests[-1]["messages"]
    assert "Внутренности git" in messages[-1].content


# ------------------------- переводы строк, BOM, запись ------------------------- #
def _edit(ctx: ToolContext, path: str, old: str, new: str) -> ToolResult:
    editor = EditFileTool()
    return editor.run(editor.parse({"path": path, "old_string": old, "new_string": new}), ctx)


def test_edit_keeps_crlf_line_endings(ctx: ToolContext) -> None:
    (ctx.root / "w.txt").write_bytes(b"one\r\ntwo\r\nthree\r\n")
    _edit(ctx, "w.txt", "two", "TWO")
    assert (ctx.root / "w.txt").read_bytes() == b"one\r\nTWO\r\nthree\r\n"
    # многострочный фрагмент от модели — с \n; несторогое совпадение (хвостовые пробелы)
    _edit(ctx, "w.txt", "one\nTWO", "1\n2")
    assert (ctx.root / "w.txt").read_bytes() == b"1\r\n2\r\nthree\r\n"
    (ctx.root / "t.txt").write_bytes(b"a  \r\nb\r\nc\r\n")
    _edit(ctx, "t.txt", "a\nb", "x\ny")
    assert (ctx.root / "t.txt").read_bytes() == b"x\r\ny\r\nc\r\n"


def test_edit_cr_only_file(ctx: ToolContext) -> None:
    # старые файлы Mac: многострочный фрагмент от модели — с \n
    (ctx.root / "mac.txt").write_bytes(b"a\rb\rc\r")
    _edit(ctx, "mac.txt", "a\nb", "A\nB")
    assert (ctx.root / "mac.txt").read_bytes() == b"A\rB\rc\r"
    (ctx.root / "mix.txt").write_bytes(b"a\rb\r\nc\r")
    assert "приведены к CR" in _edit(ctx, "mix.txt", "c", "C").content
    assert (ctx.root / "mix.txt").read_bytes() == b"a\rb\rC\r"


def test_write_over_crlf_file_keeps_its_line_endings(ctx: ToolContext) -> None:
    (ctx.root / "w.bat").write_bytes(b"@echo off\r\necho 1\r\n")
    writer = WriteFileTool()
    params = writer.parse({"path": "w.bat", "content": "@echo off\necho 2\n"})
    preview = writer.preview(params, ctx)
    assert preview is not None
    assert "\r" not in preview.text
    assert "-echo 1\n+echo 2\n" in preview.text
    writer.run(params, ctx)
    assert (ctx.root / "w.bat").read_bytes() == b"@echo off\r\necho 2\r\n"


def test_new_files_and_lf_files_stay_lf(ctx: ToolContext) -> None:
    writer = WriteFileTool()
    writer.run(writer.parse({"path": "n.py", "content": "a = 1\nb = 2\n"}), ctx)
    assert (ctx.root / "n.py").read_bytes() == b"a = 1\nb = 2\n"
    _edit(ctx, "n.py", "b = 2", "b = 3")
    assert (ctx.root / "n.py").read_bytes() == b"a = 1\nb = 3\n"


def test_edit_keeps_bom(ctx: ToolContext) -> None:
    (ctx.root / "b.cs").write_bytes("\ufeffusing System;\r\nclass A {}\r\n".encode())
    _edit(ctx, "b.cs", "using System;", "using System.IO;")
    assert (ctx.root / "b.cs").read_bytes() == "\ufeffusing System.IO;\r\nclass A {}\r\n".encode()


def test_mixed_line_endings_are_unified_with_a_note(ctx: ToolContext) -> None:
    (ctx.root / "m.txt").write_bytes(b"a\r\nb\r\nc\nd\r\n")
    res = _edit(ctx, "m.txt", "a", "A")
    assert (ctx.root / "m.txt").read_bytes() == b"A\r\nb\r\nc\r\nd\r\n"
    assert "приведены к CRLF" in res.content


def test_make_diff_marks_missing_final_newline() -> None:
    assert make_diff("a", "b", "x") == (
        f"--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n{_NO_NEWLINE_MARK}\n+b\n{_NO_NEWLINE_MARK}\n"
    )
    assert make_diff("a\n", "a\nb", "x").endswith(f"+b\n{_NO_NEWLINE_MARK}\n")


def test_make_diff_splits_only_on_newlines() -> None:
    r"""\\x0c, \\x85, \\u2028 — не переводы строк: строки диффа не склеиваются."""
    diff = make_diff("a\x0cb\nc\u2028d\nend\n", "a\x0cb\nC\u2028d\nend\n", "x")
    lines = [line for line in diff.split("\n")[2:] if line]
    assert lines
    assert all(line[:1] in " +-@\\" for line in lines)
    assert "-c\u2028d" in diff
    assert "+C\u2028d" in diff


def test_interrupted_write_leaves_file_intact(
    ctx: ToolContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (ctx.root / "k.py").write_text("x = 1\n", encoding="utf-8")

    def interrupted(src: object, dst: object) -> NoReturn:
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _edit(ctx, "k.py", "x = 1", "x = 2")
    assert (ctx.root / "k.py").read_text(encoding="utf-8") == "x = 1\n"
    assert sorted(entry.name for entry in ctx.root.iterdir()) == ["k.py"]  # временный файл убран


def test_edit_keeps_file_mode(ctx: ToolContext) -> None:
    if os.name == "nt":
        pytest.skip("права POSIX")
    script = ctx.root / "run.sh"
    script.write_text("echo 1\n", encoding="utf-8")
    script.chmod(0o755)
    _edit(ctx, "run.sh", "echo 1", "echo 2")
    assert script.stat().st_mode & 0o777 == 0o755


# -------------------------- edit_file: пустой фрагмент и отступы -------------------------- #
@pytest.mark.parametrize("replace_all", [False, True])
def test_edit_rejects_empty_old_string(ctx: ToolContext, *, replace_all: bool) -> None:
    (ctx.root / "e.txt").write_text("abc\n", encoding="utf-8")
    editor = EditFileTool()
    params = editor.parse(
        {"path": "e.txt", "old_string": "", "new_string": "X", "replace_all": replace_all},
    )
    for call in (editor.preview, editor.run):
        with pytest.raises(ToolError, match="old_string пуст"):
            call(params, ctx)
    assert (ctx.root / "e.txt").read_text(encoding="utf-8") == "abc\n"


def test_edit_tolerant_removes_extra_indentation(ctx: ToolContext) -> None:
    (ctx.root / "f.py").write_text("def f():\n    a = 1\n    return a\n", encoding="utf-8")
    _edit(ctx, "f.py", "        a = 1\n        return a", "        a = 2\n        return a")
    assert (ctx.root / "f.py").read_text() == "def f():\n    a = 2\n    return a\n"


def test_edit_tolerant_keeps_nested_structure(ctx: ToolContext) -> None:
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


def test_edit_tolerant_maps_spaces_to_tabs(ctx: ToolContext) -> None:
    (ctx.root / "t.go").write_text("func f() {\n\tif x {\n\t\ty()\n\t}\n}\n", encoding="utf-8")
    _edit(
        ctx,
        "t.go",
        "    if x {\n        y()\n    }",
        "    if x {\n        y()\n        z()\n    }",
    )
    assert (ctx.root / "t.go").read_text() == ("func f() {\n\tif x {\n\t\ty()\n\t\tz()\n\t}\n}\n")


def test_edit_tolerant_refuses_inconsistent_indentation(ctx: ToolContext) -> None:
    original = "if a:\n    b = 1\n    if b:\n        c = 2\n"
    (ctx.root / "i.py").write_text(original, encoding="utf-8")
    editor = EditFileTool()
    # один и тот же отступ шаблона соответствует разным отступам файла — не угадываем
    params = editor.parse(
        {"path": "i.py", "old_string": "b = 1\nif b:\nc = 2", "new_string": "b = 1\nif b:\nc = 3"},
    )
    with pytest.raises(ToolError, match="не найден"):
        editor.run(params, ctx)
    assert (ctx.root / "i.py").read_text() == original


def test_edit_tolerant_does_not_add_blank_lines(ctx: ToolContext) -> None:
    (ctx.root / "b.py").write_text("def f():\n    return 1   \nz = 0\n", encoding="utf-8")
    _edit(ctx, "b.py", "def f():\n    return 1\n", "def f():\n    return 2\n")
    assert (ctx.root / "b.py").read_text() == "def f():\n    return 2\nz = 0\n"
    (ctx.root / "c.py").write_text("a = 1  \nb = 2\n", encoding="utf-8")
    _edit(ctx, "c.py", "\na = 1\nb = 2", "\na = 10\nb = 2")
    assert (ctx.root / "c.py").read_text() == "a = 10\nb = 2\n"


# ------------------------- не-UTF-8, бинарные, не файлы ------------------------- #
@pytest.mark.parametrize(
    ("raw_bytes", "error"),
    [
        (b"abc\0def\n", "Бинарный"),  # NUL при верном UTF-8 (например, UTF-16LE с ASCII)
        (b"\x89PNG\r\n\x1a\n\x00\x00", "Бинарный"),
        ("привет, мир\n".encode("cp1251"), "не в UTF-8"),
    ],
    ids=["nul", "png", "cp1251"],
)
def test_write_and_edit_refuse_binary_and_non_utf8(
    ctx: ToolContext,
    raw_bytes: bytes,
    error: str,
) -> None:
    (ctx.root / "f.dat").write_bytes(raw_bytes)
    writer, editor = WriteFileTool(), EditFileTool()
    write = writer.parse({"path": "f.dat", "content": "новое\n"})
    edit = editor.parse({"path": "f.dat", "old_string": "abc", "new_string": "y"})
    cases: tuple[_ToolCall, ...] = ((writer, write), (editor, edit))
    for tool, params in cases:
        for call in (tool.preview, tool.run):
            with pytest.raises(ToolError, match=error):
                call(params, ctx)
    assert (ctx.root / "f.dat").read_bytes() == raw_bytes


def test_write_refuses_directory_in_preview(ctx: ToolContext) -> None:
    (ctx.root / "d").mkdir()
    writer = WriteFileTool()
    with pytest.raises(ToolError, match="директория"):
        writer.preview(writer.parse({"path": "d", "content": "x"}), ctx)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO")
def test_write_refuses_fifo(ctx: ToolContext) -> None:
    os.mkfifo(ctx.root / "pipe")
    writer = WriteFileTool()
    with pytest.raises(ToolError, match="Не обычный файл"):
        writer.run(writer.parse({"path": "pipe", "content": "x"}), ctx)


def test_write_preview_reports_unchanged_content(ctx: ToolContext) -> None:
    (ctx.root / "same.txt").write_text("a\n", encoding="utf-8")
    writer = WriteFileTool()
    unchanged = writer.preview(writer.parse({"path": "same.txt", "content": "a\n"}), ctx)
    assert unchanged is not None
    assert unchanged.text == "(без изменений)"
    empty = writer.preview(writer.parse({"path": "new.txt", "content": ""}), ctx)
    assert empty is not None
    assert empty.text == "(новый пустой файл)"


def test_read_range_reports_total_lines(ctx: ToolContext) -> None:
    numbered = "".join(f"l{num}\n" for num in range(1, 11))
    (ctx.root / "r.txt").write_text(numbered, encoding="utf-8")
    reader = ReadFileTool()
    out = reader.run(reader.parse({"path": "r.txt", "start_line": 1, "end_line": 3}), ctx).content
    assert out.endswith("… показаны строки 1–3 из 10. Продолжение: start_line=4.")
    tail = reader.run(reader.parse({"path": "r.txt", "start_line": 9}), ctx).content
    assert tail.endswith("… показаны строки 9–10 из 10.")
    whole = reader.run(reader.parse({"path": "r.txt"}), ctx).content
    assert "показаны строки" not in whole


def test_read_cp1251_file(ctx: ToolContext) -> None:
    (ctx.root / "old.txt").write_bytes("Привет\r\nмир\r\n".encode("cp1251"))
    reader = ReadFileTool()
    res = reader.run(reader.parse({"path": "old.txt"}), ctx)
    assert "1\tПривет" in res.content
    assert "2\tмир" in res.content
    assert "показан как cp1251" in res.content
    assert "cp1251" in res.summary


def test_hard_linked_file_is_edited_in_place(ctx: ToolContext) -> None:
    if not hasattr(os, "link"):
        pytest.skip("жёсткие ссылки")
    (ctx.root / "a.txt").write_text("x = 1\n", encoding="utf-8")
    os.link(ctx.root / "a.txt", ctx.root / "b.txt")
    _edit(ctx, "a.txt", "x = 1", "x = 2")
    assert (ctx.root / "b.txt").read_text(encoding="utf-8") == "x = 2\n"  # ссылка цела


def test_file_with_xattrs_is_edited_in_place(ctx: ToolContext) -> None:
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


def _fake_git_dir(directory: pathlib.Path) -> None:
    (directory / "objects").mkdir(parents=True)
    (directory / "refs").mkdir()
    (directory / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (directory / "config").write_text("[core]\n", encoding="utf-8")


def test_nested_repository_git_dir_is_not_writable(ctx: ToolContext) -> None:
    """Каталог git вложенного репозитория не пишется ни через .git, ни напрямую.

    Вложенный репозиторий, чей .git — симлинк или gitdir:-файл на каталог в проекте:
    ни путь через .git, ни прямой путь к его каталогу git не пишутся.
    """
    if os.name == "nt":
        pytest.skip("симлинки")
    _fake_git_dir(ctx.root / "nested" / "store")
    (ctx.root / "nested" / ".git").symlink_to(ctx.root / "nested" / "store")
    _fake_git_dir(ctx.root / "store2")
    (ctx.root / "other").mkdir()
    (ctx.root / "other" / ".git").write_text("gitdir: ../store2\n", encoding="utf-8")
    _fake_git_dir(ctx.root / "bare.git")
    writer = WriteFileTool()
    for path in (
        "nested/.git/config",
        "nested/store/config",
        "nested/store/hooks/pre-commit",
        "store2/config",
        "bare.git/config",
    ):
        with pytest.raises(ToolError, match="Внутренности git"):
            writer.preview(writer.parse({"path": path, "content": "x"}), ctx)
    assert (ctx.root / "nested" / "store" / "config").read_text(encoding="utf-8") == "[core]\n"
    # каталог с одним файлом HEAD — не каталог git
    (ctx.root / "docs").mkdir()
    (ctx.root / "docs" / "HEAD").write_text("x", encoding="utf-8")
    assert writer.run(writer.parse({"path": "docs/HEAD", "content": "y"}), ctx).ok


@pytest.mark.parametrize(
    "order",
    [
        ("config", "objects/.keep", "refs/.keep", "HEAD"),
        ("HEAD", "config", "refs/heads/main", "objects/info/x"),
        ("objects/x", "HEAD", "config", "refs/x"),
    ],
)
def test_git_dir_cannot_be_built_file_by_file(
    ctx: ToolContext,
    order: tuple[str, str, str, str],
) -> None:
    """Каталог git нельзя собрать по файлу.

    Запись, которая завершила бы раскладку (HEAD, objects/, refs/), отклоняется в любом
    порядке — иначе config, записанный первым, стал бы настройкой репозитория, который
    git найдёт в этом каталоге.
    """
    writer = WriteFileTool()
    *first, last = order
    for name in first:
        assert writer.run(writer.parse({"path": f"sub/{name}", "content": "x\n"}), ctx).ok
    with pytest.raises(ToolError, match="Внутренности git"):
        writer.preview(writer.parse({"path": f"sub/{last}", "content": "x\n"}), ctx)
    assert not (ctx.root / "sub" / last).exists()


# ------------------------------ файлы с секретами ------------------------------ #
def _decide(
    tool: Tool[Any],
    params: BaseModel,
    ctx: ToolContext,
    mode: str = "auto",
    *,
    auto_approve: bool = False,
    hook: Literal["allow", "ask"] | None = None,
) -> Decision:
    assessment = tool.assess(params, ctx)
    return decide(
        PermissionMode(mode),
        assessment.risk,
        assessment.kind or tool.kind,
        auto_approve=auto_approve,
        hook=hook,
    )


@pytest.mark.parametrize("path", [".env", "sub/.env", ".ENV", "link.txt"])
def test_env_write_always_asks(ctx: ToolContext, path: str) -> None:
    """.env — настройки самого gigacrew (ключ, адреса API, TLS).

    Правка — только с подтверждения, в любом режиме и даже с -y или разрешением хука.
    """
    if path == "link.txt":
        if os.name == "nt":
            pytest.skip("симлинки")
        (ctx.root / ".env").write_text("GIGACREW_API_URL=https://api.test\n", encoding="utf-8")
        (ctx.root / "link.txt").symlink_to(ctx.root / ".env")
    writer, editor = WriteFileTool(), EditFileTool()
    calls: list[_ToolCall] = [
        (writer, writer.parse({"path": path, "content": "GIGACREW_API_URL=https://evil.test\n"})),
        (editor, editor.parse({"path": path, "old_string": "api.test", "new_string": "evil.test"})),
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
def test_secret_file_write_asks_in_auto_mode(ctx: ToolContext, path: str) -> None:
    writer = WriteFileTool()
    params = writer.parse({"path": path, "content": "x\n"})
    assert "секрет" in writer.assess(params, ctx).reason
    assert _decide(writer, params, ctx) is Decision.ASK  # правка обычного файла — без вопроса
    assert _decide(writer, params, ctx, auto_approve=True) is Decision.ALLOW  # -y — как у run_shell


def test_ordinary_and_template_files_are_plain_edits(ctx: ToolContext) -> None:
    writer = WriteFileTool()
    for path in (".env.example", "src/env.py", "environment.md"):
        params = writer.parse({"path": path, "content": "x\n"})
        assert not writer.assess(params, ctx).reason
        assert _decide(writer, params, ctx) is Decision.ALLOW


def test_env_edit_in_auto_mode_is_confirmed(tmp_path: pathlib.Path) -> None:
    """Сквозной путь: модель правит .env в режиме «авто» с -y — пользователя спрашивают."""
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
def test_read_file_refuses_secret_files(ctx: ToolContext, path: str) -> None:
    target = ctx.root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("TOKEN=s3cr3t\n", encoding="utf-8")
    reader = ReadFileTool()
    with pytest.raises(ToolError, match="секрет") as err:
        reader.run(reader.parse({"path": path}), ctx)
    assert "s3cr3t" not in str(err.value)
    assert "run_shell" in str(err.value)


def test_read_file_refuses_symlink_to_secret_and_reads_template(ctx: ToolContext) -> None:
    (ctx.root / ".env").write_text("TOKEN=s3cr3t\n", encoding="utf-8")
    (ctx.root / ".env.example").write_text("TOKEN=\n", encoding="utf-8")
    reader = ReadFileTool()
    assert "TOKEN=" in reader.run(reader.parse({"path": ".env.example"}), ctx).content
    if os.name == "nt":
        pytest.skip("симлинки")
    (ctx.root / "settings.txt").symlink_to(ctx.root / ".env")
    with pytest.raises(ToolError, match="секрет"):
        reader.run(reader.parse({"path": "settings.txt"}), ctx)


def test_search_skips_and_refuses_secret_files(ctx: ToolContext) -> None:
    for path in ("src/a.py", ".env", ".ENV.local", ".npmrc", "deploy.pem", "credentials.json"):
        target = ctx.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("needle=s3cr3t\n", encoding="utf-8")
    (ctx.root / ".env.example").write_text("needle=\n", encoding="utf-8")
    searcher = SearchContentTool()
    out = searcher.run(searcher.parse({"pattern": "needle"}), ctx).content
    assert sorted(ln.split(":", 1)[0] for ln in out.splitlines()) == [".env.example", "src/a.py"]
    for path in (".env", ".npmrc"):  # явно названный файл с секретами — тоже нет
        with pytest.raises(ToolError, match="секрет") as err:
            searcher.run(searcher.parse({"pattern": "needle", "path": path}), ctx)
        assert "s3cr3t" not in str(err.value)


def test_edit_not_found_in_secret_file_shows_no_content(ctx: ToolContext) -> None:
    """Неудачная правка показывает модели файл — но не файл с секретами.

    Ошибка превью уходит модели до вопроса о подтверждении.
    """
    (ctx.root / ".env").write_text("TOKEN=s3cr3t\n", encoding="utf-8")
    with pytest.raises(ToolError, match="не найден") as err:
        _edit(ctx, ".env", "MISSING=1", "MISSING=2")
    assert "s3cr3t" not in str(err.value)


def test_search_ignore_case(ctx: ToolContext) -> None:
    (ctx.root / "a.txt").write_text("Needle\n", encoding="utf-8")
    searcher = SearchContentTool()
    found = searcher.run(searcher.parse({"pattern": "needle", "ignore_case": True}), ctx).content
    assert found == "a.txt:1:Needle"
    assert (
        searcher.run(searcher.parse({"pattern": "needle"}), ctx).content
        == "(совпадений не найдено)"
    )
