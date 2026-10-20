"""Скиллы: разбор SKILL.md, поиск в .gigacode/skills и ~/.gigacode/skills, инструмент skill,
вызов /скилл в REPL и одноразовом режиме."""

from __future__ import annotations

from pathlib import Path

import pytest
from fakes import ScriptedProvider, text_turn

import gigacrew.cli.app as app
from gigacrew.agent.chat_store import ChatStore
from gigacrew.agent.loop import Agent
from gigacrew.agent.prompts import build_project_context, skill_invocation_prompt
from gigacrew.cli.commands import CommandContext, default_commands, expand_skill
from gigacrew.cli.repl import run_repl
from gigacrew.config import Config
from gigacrew.project.skills import (
    SkillCatalog,
    discover_skills,
    parse_skill,
    split_frontmatter,
    user_skills_dir,
)
from gigacrew.project.workspace import Workspace
from gigacrew.tools.base import ToolError, build_default_registry
from gigacrew.tools.skill import SkillTool
from gigacrew.ui.console import Console


def _skill(base: Path, dirname: str, text: str) -> Path:
    path = base / dirname / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _project_skill(root: Path, name: str, description: str, body: str = "Сделай X.") -> Path:
    text = f"---\nname: {name}\ndescription: {description}\n---\n{body}\n"
    return _skill(root / ".gigacode" / "skills", name, text)


# ------------------------------- разбор ------------------------------- #
def test_frontmatter_values():
    fields, body = split_frontmatter(
        "---\n"
        "name: 'release'\n"
        'description: "Выпуск: версия и тег"\n'
        "notes: |\n"
        "  строка 1\n"
        "  строка 2\n"
        "summary: >\n"
        "  склеенные\n"
        "  строки\n"
        "metadata:\n"
        "  owner: me\n"
        "---\n"
        "Тело\n"
    )
    assert fields["name"] == "release"
    assert fields["description"] == "Выпуск: версия и тег"
    assert fields["notes"] == "строка 1\nстрока 2"
    assert fields["summary"] == "склеенные строки"
    assert "owner" not in fields  # вложенные ключи — часть значения metadata
    assert body == "Тело"


@pytest.mark.parametrize("text", ["Просто текст", "---\nname: x\nбез закрытия\n"])
def test_no_frontmatter_keeps_whole_text(text):
    assert split_frontmatter(text) == ({}, text)


def test_parse_skill_name_defaults_to_directory(tmp_path):
    skill = parse_skill(_skill(tmp_path, "lint-fix", "Без frontmatter.\n"))
    assert skill is not None
    assert (skill.name, skill.description, skill.command) == ("lint-fix", "", "/lint-fix")
    assert skill.body() == "Без frontmatter."


@pytest.mark.parametrize("name", ["with space", "-dash", "a/b", "_x", "2fa", "αβ"])
def test_parse_skill_rejects_names_unfit_for_command(tmp_path, name):
    assert parse_skill(_skill(tmp_path, "d", f"---\nname: {name}\n---\nx\n")) is None


def test_body_strips_frontmatter(tmp_path):
    skill = parse_skill(_skill(tmp_path, "d", "---\nname: d\ndescription: x\n---\n\nШаги.\n"))
    assert skill is not None and skill.body() == "Шаги."


# ------------------------------- поиск -------------------------------- #
def test_discover_project_and_user_skills(tmp_path, _isolated_home):
    project = tmp_path / "proj"
    _project_skill(project, "deploy", "проектный")
    _skill(user_skills_dir(), "deploy", "---\ndescription: пользовательский\n---\nx\n")
    _skill(user_skills_dir(), "notes", "---\ndescription: заметки\n---\nx\n")
    (project / ".gigacode" / "skills" / "empty").mkdir()  # без SKILL.md — не скилл

    skills = discover_skills(Workspace(project))
    assert [(s.name, s.description, s.source) for s in skills] == [
        ("deploy", "проектный", "project"),  # проектный перекрывает пользовательский
        ("notes", "заметки", "user"),
    ]
    assert user_skills_dir() == _isolated_home / ".gigacode" / "skills"


def test_old_gigacrew_folder_is_not_used(tmp_path):
    _skill(tmp_path / ".gigacrew" / "skills", "old", "---\ndescription: x\n---\nx\n")
    assert discover_skills(Workspace(tmp_path)) == []


def test_catalog_refresh_and_lookup(tmp_path):
    catalog = SkillCatalog(Workspace(tmp_path))
    _project_skill(tmp_path, "Review", "ревью")
    assert len(catalog) == 0  # до refresh ФС не читается
    catalog.refresh()
    assert catalog.get("/review") is catalog.get("REVIEW") is not None
    assert catalog.get("nope") is None


# ------------------------------ промпты ------------------------------- #
def test_project_context_lists_skills(tmp_path):
    assert "Skills" not in build_project_context(Workspace(tmp_path))
    _project_skill(tmp_path, "deploy", "выкатка на стенд")
    context = build_project_context(Workspace(tmp_path))
    assert "skill tool" in context and "- deploy: выкатка на стенд" in context


def test_invocation_prompt_substitutes_arguments(tmp_path):
    skill = parse_skill(_project_skill(tmp_path, "fix", "x"))
    assert skill is not None
    text = skill_invocation_prompt(skill, "Почини $ARGUMENTS.", "/fix тесты", "тесты")
    assert text.startswith("/fix тесты\n")
    assert "Почини тесты." in text and "Arguments:" not in text
    text = skill_invocation_prompt(skill, "Почини всё.", "/fix  срочно", "срочно")
    assert text.rstrip().endswith("Arguments: срочно")


# ----------------------------- инструмент ----------------------------- #
def test_skill_tool_loads_instructions_and_files(ctx):
    _project_skill(ctx.root, "deploy", "выкатка", body="Прочитай reference.md.")
    (ctx.root / ".gigacode" / "skills" / "deploy" / "reference.md").write_text("справка")
    tool = SkillTool()
    result = tool.run(tool.parse({"name": "deploy"}), ctx)
    assert result.ok and "Прочитай reference.md." in result.content
    assert "description:" not in result.content  # frontmatter не нужен модели
    assert "- reference.md" in result.content
    ref = tool.run(tool.parse({"name": "deploy", "file": "reference.md"}), ctx)
    assert ref.content == "справка"


def test_skill_tool_reads_user_skill_outside_project(ctx):
    _skill(user_skills_dir(), "notes", "---\ndescription: x\n---\nТело.\n")
    (user_skills_dir() / "notes" / "tpl.md").write_text("шаблон")
    tool = SkillTool()
    assert "Тело." in tool.run(tool.parse({"name": "notes"}), ctx).content
    assert tool.run(tool.parse({"name": "notes", "file": "tpl.md"}), ctx).content == "шаблон"


@pytest.mark.parametrize("file", ["../other/SKILL.md", "/etc/passwd", "missing.md"])
def test_skill_tool_file_stays_in_skill_dir(ctx, file):
    _project_skill(ctx.root, "deploy", "x")
    _project_skill(ctx.root, "other", "x")
    tool = SkillTool()
    with pytest.raises(ToolError):
        tool.run(tool.parse({"name": "deploy", "file": file}), ctx)


def test_skill_tool_unknown_lists_available(ctx):
    _project_skill(ctx.root, "deploy", "x")
    tool = SkillTool()
    with pytest.raises(ToolError, match="Доступные: deploy"):
        tool.run(tool.parse({"name": "nope"}), ctx)


def test_skill_tool_truncates_large_file(ctx, monkeypatch):
    import gigacrew.tools.skill as skill_module

    monkeypatch.setattr(skill_module, "MAX_SKILL_CHARS", 10)
    _project_skill(ctx.root, "deploy", "x")
    (ctx.root / ".gigacode" / "skills" / "deploy" / "big.md").write_text("я" * 50)
    tool = SkillTool()
    text = tool.run(tool.parse({"name": "deploy", "file": "big.md"}), ctx).content
    assert text.startswith("я" * 10 + "\n…") and "обрезано" in text


def test_skill_tool_registered():
    assert "skill" in build_default_registry()


# ------------------------------ REPL и -p ----------------------------- #
@pytest.fixture
def repl_env(tmp_path):
    cfg = Config(access_key="x", project_root=tmp_path, stream=False, auto_approve=True)
    provider = ScriptedProvider([text_turn("готово")])
    ui = Console(no_color=True)
    agent = Agent(provider, build_default_registry(), cfg, ui)
    return agent, ui, provider


def _reader(*items):
    queue = list(items)
    return lambda: queue.pop(0)


def test_expand_skill(tmp_path):
    _project_skill(tmp_path, "deploy", "x", body="Выкати $ARGUMENTS.")
    _project_skill(tmp_path, "help", "скилл с именем встроенной команды")
    catalog = SkillCatalog(Workspace(tmp_path))
    catalog.refresh()
    commands = default_commands()
    text = expand_skill("/deploy stage", commands, catalog)
    assert text is not None and "Выкати stage." in text
    assert expand_skill("/help", commands, catalog) is None  # встроенная команда важнее
    assert expand_skill("/nope", commands, catalog) is None
    assert expand_skill("deploy", commands, catalog) is None


def test_repl_runs_skill_as_agent_turn(repl_env, tmp_path, capsys):
    agent, ui, _ = repl_env
    _project_skill(tmp_path, "deploy", "выкатка", body="Выкати на стенд.")
    read = _reader("/deploy stage", "/exit")
    assert run_repl(agent, ui, default_commands(), read_input=read, index_on_start=False) == 0
    first = agent.conversation.messages[0]
    assert first.role == "user" and first.content.startswith("/deploy stage\n")
    assert "Выкати на стенд." in first.content and "Arguments: stage" in first.content
    assert "готово" in capsys.readouterr().out


def test_repl_sees_skill_installed_during_session(repl_env, tmp_path, capsys):
    agent, ui, _ = repl_env
    lines = iter(["/later", "/later", "/exit"])
    calls = 0

    def read():
        nonlocal calls
        calls += 1
        if calls == 2:  # скилл установлен между первым и вторым вводом
            _project_skill(tmp_path, "later", "появился позже")
        return next(lines)

    run_repl(agent, ui, default_commands(), read_input=read, index_on_start=False)
    out = capsys.readouterr().out
    assert out.count("неизвестная команда: /later") == 1
    assert len(agent.conversation.messages) == 2


def test_help_and_unknown_hint_include_skills(repl_env, tmp_path, capsys):
    agent, ui, _ = repl_env
    _project_skill(tmp_path, "deploy", "выкатка [на стенд]")
    catalog = SkillCatalog(agent.workspace)
    catalog.refresh()
    commands = default_commands()
    ctx = CommandContext(agent=agent, ui=ui, commands=commands, skills=catalog)
    commands.dispatch("/help", ctx)
    out = capsys.readouterr().out
    assert "/deploy [аргументы]" in out and "скилл · выкатка [на стенд]" in out
    assert "/resume [id]" in out  # квадратные скобки — не разметка Rich
    commands.dispatch("/deplyo", ctx)
    assert "возможно, /deploy?" in capsys.readouterr().out


def test_oneshot_expands_skill(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name, value in {
        "GIGACREW_ACCESS_KEY": "dummy",
        "GIGACREW_AUTH_URL": "https://auth.test/oauth",
        "GIGACREW_API_URL": "https://api.test/v1",
        "GIGACREW_MODEL": "M-default",
        "GIGACREW_AUTO_MEASURE": "0",
    }.items():
        monkeypatch.setenv(name, value)
    provider = ScriptedProvider([text_turn("сделано")])
    monkeypatch.setattr(app, "_make_provider", lambda _cfg, _ui: provider)
    _project_skill(tmp_path, "deploy", "x", body="Выкати $ARGUMENTS.")
    code = app.main(["-C", str(tmp_path), "-p", "/deploy prod", "--no-color", "--no-stream"])
    assert code == 0
    assert "Выкати prod." in provider.requests[-1]["messages"][-1].content
    (saved,) = ChatStore(Workspace(tmp_path)).recent()
    assert saved.title.startswith("/deploy prod ")  # название чата — с команды
