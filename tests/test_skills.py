"""Скиллы: разбор SKILL.md, поиск, инструмент skill и вызов /скилл.

Поиск — в .gigacode/skills и ~/.gigacode/skills; вызов /скилл — в REPL и одноразовом режиме.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from itertools import count
from pathlib import Path

import pytest
from fakes import ScriptedProvider, text_turn

from gigacrew.agent.chat_store import ChatStore
from gigacrew.agent.loop import Agent
from gigacrew.agent.prompts import build_project_context, skill_invocation_prompt
from gigacrew.cli import app
from gigacrew.cli.commands import CommandContext, default_commands, expand_skill
from gigacrew.cli.repl import run_repl
from gigacrew.config import Config
from gigacrew.project.skills import (
    MAX_SKILL_BYTES,
    SkillCatalog,
    discover_skills,
    parse_skill,
    split_frontmatter,
    user_skills_dir,
)
from gigacrew.project.workspace import Workspace
from gigacrew.tools import skill as skill_module
from gigacrew.tools.base import ToolContext, ToolError, build_default_registry
from gigacrew.tools.skill import SkillTool
from gigacrew.ui.console import Console

_ReplEnv = tuple[Agent, Console, ScriptedProvider]


def _skill(base: Path, dirname: str, text: str) -> Path:
    path = base / dirname / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _project_skill(root: Path, name: str, description: str, body: str = "Сделай X.") -> Path:
    text = f"---\nname: {name}\ndescription: {description}\n---\n{body}\n"
    return _skill(root / ".gigacode" / "skills", name, text)


# ------------------------------- разбор ------------------------------- #
def test_frontmatter_values() -> None:
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
        "Тело\n",
    )
    assert fields["name"] == "release"
    assert fields["description"] == "Выпуск: версия и тег"
    assert fields["notes"] == "строка 1\nстрока 2"
    assert fields["summary"] == "склеенные строки"
    assert "owner" not in fields  # вложенные ключи — часть значения metadata
    assert body == "Тело"


@pytest.mark.parametrize("text", ["Просто текст", "---\nname: x\nбез закрытия\n"])
def test_no_frontmatter_keeps_whole_text(text: str) -> None:
    assert split_frontmatter(text) == ({}, text)


def test_parse_skill_name_defaults_to_directory(tmp_path: Path) -> None:
    skill = parse_skill(_skill(tmp_path, "lint-fix", "Без frontmatter.\n"))
    assert skill is not None
    assert (skill.name, skill.description, skill.command) == ("lint-fix", "", "/lint-fix")
    assert skill.body() == "Без frontmatter."


@pytest.mark.parametrize("name", ["with space", "-dash", "a/b", "_x", "2fa", "αβ"])
def test_parse_skill_rejects_names_unfit_for_command(tmp_path: Path, name: str) -> None:
    assert parse_skill(_skill(tmp_path, "d", f"---\nname: {name}\n---\nx\n")) is None


def test_body_strips_frontmatter(tmp_path: Path) -> None:
    skill = parse_skill(_skill(tmp_path, "d", "---\nname: d\ndescription: x\n---\n\nШаги.\n"))
    assert skill is not None
    assert skill.body() == "Шаги."


# ------------------------------- поиск -------------------------------- #
@pytest.fixture
def home(_isolated_home: Path) -> Path:
    """Временный домашний каталог теста (его подставляет ``_isolated_home`` из conftest)."""
    return _isolated_home


def test_discover_project_and_user_skills(tmp_path: Path, home: Path) -> None:
    project = tmp_path / "proj"
    _project_skill(project, "deploy", "проектный")
    _skill(user_skills_dir(), "deploy", "---\ndescription: пользовательский\n---\nx\n")
    _skill(user_skills_dir(), "notes", "---\ndescription: заметки\n---\nx\n")
    (project / ".gigacode" / "skills" / "empty").mkdir()  # без SKILL.md — не скилл

    skills = discover_skills(Workspace(project))
    assert [(skill.name, skill.description, skill.source) for skill in skills] == [
        ("deploy", "проектный", "project"),  # проектный перекрывает пользовательский
        ("notes", "заметки", "user"),
    ]
    assert user_skills_dir() == home / ".gigacode" / "skills"


def test_project_skill_overrides_user_skill_ignoring_case(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    _skill(user_skills_dir(), "Deploy", "---\ndescription: пользовательский\n---\nx\n")
    _project_skill(project, "deploy", "проектный")
    skills = discover_skills(Workspace(project))
    assert [(skill.name, skill.description, skill.source) for skill in skills] == [
        ("deploy", "проектный", "project"),
    ]


def test_oversized_skill_is_ignored(tmp_path: Path) -> None:
    filler = "x" * MAX_SKILL_BYTES
    path = _skill(tmp_path, "big", f"---\nname: big\n---\n{filler}")
    assert parse_skill(path) is None


def test_old_gigacrew_folder_is_not_used(tmp_path: Path) -> None:
    _skill(tmp_path / ".gigacrew" / "skills", "old", "---\ndescription: x\n---\nx\n")
    assert not discover_skills(Workspace(tmp_path))


def test_catalog_refresh_and_lookup(tmp_path: Path) -> None:
    catalog = SkillCatalog(Workspace(tmp_path))
    _project_skill(tmp_path, "Review", "ревью")
    assert len(catalog) == 0  # до refresh ФС не читается
    catalog.refresh()
    review = catalog.get("/review")
    assert review is catalog.get("REVIEW")
    assert review is not None
    assert catalog.get("nope") is None


# ------------------------------ промпты ------------------------------- #
def test_project_context_lists_skills(tmp_path: Path) -> None:
    assert "Skills" not in build_project_context(Workspace(tmp_path))
    _project_skill(tmp_path, "deploy", "выкатка на стенд")
    context = build_project_context(Workspace(tmp_path))
    assert "skill tool" in context
    assert "- deploy: выкатка на стенд" in context


def test_untrusted_project_skills_are_not_loaded(tmp_path: Path) -> None:
    project = tmp_path / "project"
    user_dir = tmp_path / "user"
    _project_skill(project, "review", "скилл проекта")
    _project_skill(project, "deploy", "только в проекте")
    _skill(user_dir, "review", "---\nname: review\ndescription: скилл пользователя\n---\nX\n")
    workspace = Workspace(project, untrusted=True)
    skills = discover_skills(workspace, user_dir)
    assert [(skill.name, skill.source) for skill in skills] == [("review", "user")]
    assert "deploy" not in build_project_context(workspace)


def test_invocation_prompt_substitutes_arguments(tmp_path: Path) -> None:
    skill = parse_skill(_project_skill(tmp_path, "fix", "x"))
    assert skill is not None
    text = skill_invocation_prompt(skill, "Почини $ARGUMENTS.", "/fix тесты", "тесты")
    assert text.startswith("/fix тесты\n")
    assert "Почини тесты." in text
    assert "Arguments:" not in text
    text = skill_invocation_prompt(skill, "Почини всё.", "/fix  срочно", "срочно")
    assert text.rstrip().endswith("Arguments: срочно")


# ----------------------------- инструмент ----------------------------- #
def test_skill_tool_loads_instructions_and_files(ctx: ToolContext) -> None:
    _project_skill(ctx.root, "deploy", "выкатка", body="Прочитай reference.md.")
    (ctx.root / ".gigacode" / "skills" / "deploy" / "reference.md").write_text("справка")
    tool = SkillTool()
    loaded = tool.run(tool.parse({"name": "deploy"}), ctx)
    assert loaded.ok
    assert "Прочитай reference.md." in loaded.content
    assert "description:" not in loaded.content  # frontmatter не нужен модели
    assert "- reference.md" in loaded.content
    ref = tool.run(tool.parse({"name": "deploy", "file": "reference.md"}), ctx)
    assert ref.content == "справка"


def test_skill_tool_reads_user_skill_outside_project(ctx: ToolContext) -> None:
    _skill(user_skills_dir(), "notes", "---\ndescription: x\n---\nТело.\n")
    (user_skills_dir() / "notes" / "tpl.md").write_text("шаблон")
    tool = SkillTool()
    assert "Тело." in tool.run(tool.parse({"name": "notes"}), ctx).content
    assert tool.run(tool.parse({"name": "notes", "file": "tpl.md"}), ctx).content == "шаблон"


@pytest.mark.parametrize("rel_path", ["../other/SKILL.md", "/etc/passwd", "missing.md"])
def test_skill_tool_file_stays_in_skill_dir(ctx: ToolContext, rel_path: str) -> None:
    _project_skill(ctx.root, "deploy", "x")
    _project_skill(ctx.root, "other", "x")
    tool = SkillTool()
    with pytest.raises(ToolError):
        tool.run(tool.parse({"name": "deploy", "file": rel_path}), ctx)


def _outside_skill(base: Path) -> Path:
    """Файл вне проекта и вне каталогов скиллов (как ~/.aws/credentials)."""
    return _skill(base / "outside", "keys", "---\nname: keys\n---\nTOKEN=abc\n")


@pytest.mark.skipif(os.name == "nt", reason="симлинки")
@pytest.mark.parametrize("link", ["file", "dir"])
def test_symlinked_project_skill_outside_is_not_a_skill(tmp_path: Path, link: str) -> None:
    # Репозиторий коммитит .gigacode/skills/<имя>/SKILL.md (или весь каталог) симлинком
    # наружу: содержимое чужого файла не должно попасть к модели без вопроса.
    project = tmp_path / "proj"
    target = _outside_skill(tmp_path)
    skills = project / ".gigacode" / "skills"
    if link == "file":
        (skills / "keys").mkdir(parents=True)
        (skills / "keys" / "SKILL.md").symlink_to(target)
    else:
        skills.mkdir(parents=True)
        (skills / "keys").symlink_to(target.parent)
    ws = Workspace(project)
    assert not discover_skills(ws)
    tool = SkillTool()
    with pytest.raises(ToolError, match="Скилл не найден"):
        tool.run(tool.parse({"name": "keys"}), ToolContext(workspace=ws))


@pytest.mark.skipif(os.name == "nt", reason="симлинки")
def test_symlinked_skills_dir_outside_project_is_ignored(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    target = _outside_skill(tmp_path)
    (project / ".gigacode").mkdir(parents=True)
    (project / ".gigacode" / "skills").symlink_to(target.parent.parent)
    assert not discover_skills(Workspace(project))


@pytest.mark.skipif(os.name == "nt", reason="симлинки")
def test_symlinked_user_skill_outside_is_not_a_skill(tmp_path: Path) -> None:
    target = _outside_skill(tmp_path)
    user_skills_dir().mkdir(parents=True)
    (user_skills_dir() / "keys").symlink_to(target.parent)
    assert not discover_skills(Workspace(tmp_path / "proj"))


@pytest.mark.skipif(os.name == "nt", reason="симлинки")
def test_symlinks_inside_skills_dir_still_work(tmp_path: Path, home: Path) -> None:
    # Каталог скиллов целиком — симлинк (dotfiles), скилл — симлинк на соседний: можно.
    real = home / "dotfiles" / "skills"
    _skill(real, "notes", "---\ndescription: заметки\n---\nТело.\n")
    (real / "alias").symlink_to(real / "notes")
    user_skills_dir().parent.mkdir(parents=True)
    user_skills_dir().symlink_to(real)
    assert [skill.name for skill in discover_skills(Workspace(tmp_path))] == ["alias", "notes"]


@pytest.mark.skipif(os.name == "nt", reason="симлинки")
def test_skill_tool_does_not_list_or_read_files_outside_skill_dir(
    ctx: ToolContext,
    tmp_path: Path,
) -> None:
    secret = tmp_path / "outside" / "id_rsa"
    secret.parent.mkdir()
    secret.write_text("PRIVATE KEY")
    _project_skill(ctx.root, "deploy", "x")
    skill_dir = ctx.root / ".gigacode" / "skills" / "deploy"
    (skill_dir / "key").symlink_to(secret)
    (skill_dir / "ref").symlink_to(secret.parent)
    tool = SkillTool()
    listing = tool.run(tool.parse({"name": "deploy"}), ctx)
    assert "- key" not in listing.content
    assert "- ref/" not in listing.content
    for rel_path in ("key", "ref/id_rsa"):
        with pytest.raises(ToolError, match="за пределы каталога скилла"):
            tool.run(tool.parse({"name": "deploy", "file": rel_path}), ctx)


def test_skill_tool_unknown_lists_available(ctx: ToolContext) -> None:
    _project_skill(ctx.root, "deploy", "x")
    tool = SkillTool()
    with pytest.raises(ToolError, match="Доступные: deploy"):
        tool.run(tool.parse({"name": "nope"}), ctx)


def test_skill_tool_truncates_large_file(ctx: ToolContext, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(skill_module, "MAX_SKILL_CHARS", 10)
    _project_skill(ctx.root, "deploy", "x")
    (ctx.root / ".gigacode" / "skills" / "deploy" / "big.md").write_text("я" * 50)
    tool = SkillTool()
    text = tool.run(tool.parse({"name": "deploy", "file": "big.md"}), ctx).content
    head = "я" * 10
    assert text.startswith(f"{head}\n…")
    assert "обрезано" in text


def test_skill_tool_registered() -> None:
    assert "skill" in build_default_registry()


# ------------------------------ REPL и -p ----------------------------- #
@pytest.fixture
def repl_env(tmp_path: Path) -> _ReplEnv:
    cfg = Config(access_key="x", project_root=tmp_path, stream=False, auto_approve=True)
    provider = ScriptedProvider([text_turn("готово")])
    ui = Console(no_color=True)
    agent = Agent(provider, build_default_registry(), cfg, ui)
    return agent, ui, provider


def _reader(*lines: str) -> Callable[[], str]:
    queue = list(lines)
    return lambda: queue.pop(0)


def test_expand_skill(tmp_path: Path) -> None:
    _project_skill(tmp_path, "deploy", "x", body="Выкати $ARGUMENTS.")
    _project_skill(tmp_path, "help", "скилл с именем встроенной команды")
    catalog = SkillCatalog(Workspace(tmp_path))
    catalog.refresh()
    commands = default_commands()
    text = expand_skill("/deploy stage", commands, catalog)
    assert text is not None
    assert "Выкати stage." in text
    assert expand_skill("/help", commands, catalog) is None  # встроенная команда важнее
    assert expand_skill("/nope", commands, catalog) is None
    assert expand_skill("deploy", commands, catalog) is None


def test_repl_runs_skill_as_agent_turn(
    repl_env: _ReplEnv,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent, ui, _ = repl_env
    _project_skill(tmp_path, "deploy", "выкатка", body="Выкати на стенд.")
    read = _reader("/deploy stage", "/exit")
    assert run_repl(agent, ui, default_commands(), read_input=read, index_on_start=False) == 0
    first = agent.conversation.messages[0]
    assert first.role == "user"
    assert first.content.startswith("/deploy stage\n")
    assert "Выкати на стенд." in first.content
    assert "Arguments: stage" in first.content
    assert "готово" in capsys.readouterr().out


def test_repl_sees_skill_installed_during_session(
    repl_env: _ReplEnv,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent, ui, _ = repl_env
    lines = iter(["/later", "/later", "/exit"])
    calls = count(1)

    def read() -> str:
        if next(calls) == 2:  # скилл установлен между первым и вторым вводом
            _project_skill(tmp_path, "later", "появился позже")
        return next(lines)

    run_repl(agent, ui, default_commands(), read_input=read, index_on_start=False)
    out = capsys.readouterr().out
    assert out.count("неизвестная команда: /later") == 1
    assert len(agent.conversation.messages) == 2


def test_help_and_unknown_hint_include_skills(
    repl_env: _ReplEnv,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent, ui, _ = repl_env
    _project_skill(tmp_path, "deploy", "выкатка [на стенд]")
    catalog = SkillCatalog(agent.workspace)
    catalog.refresh()
    commands = default_commands()
    ctx = CommandContext(agent=agent, ui=ui, commands=commands, skills=catalog)
    commands.dispatch("/help", ctx)
    out = capsys.readouterr().out
    assert "/deploy [аргументы]" in out
    assert "скилл · выкатка [на стенд]" in out
    assert "/resume [id]" in out  # квадратные скобки — не разметка Rich
    commands.dispatch("/deplyo", ctx)
    assert "возможно, /deploy?" in capsys.readouterr().out


def test_oneshot_expands_skill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    for name, env_value in {
        "GIGACREW_ACCESS_KEY": "dummy",
        "GIGACREW_AUTH_URL": "https://auth.test/oauth",
        "GIGACREW_API_URL": "https://api.test/v1",
        "GIGACREW_MODEL": "M-default",
        "GIGACREW_AUTO_MEASURE": "0",
    }.items():
        monkeypatch.setenv(name, env_value)
    provider = ScriptedProvider([text_turn("сделано")])
    monkeypatch.setattr(app, "_make_provider", lambda _cfg: provider)
    _project_skill(tmp_path, "deploy", "x", body="Выкати $ARGUMENTS.")
    code = app.main(["-C", str(tmp_path), "-p", "/deploy prod", "--no-color", "--no-stream"])
    assert code == 0
    last_request = provider.requests[-1]
    assert "Выкати prod." in last_request["messages"][-1].content
    saved = ChatStore(Workspace(tmp_path)).recent()
    assert len(saved) == 1
    assert saved[0].title.startswith("/deploy prod ")  # название чата — с команды
