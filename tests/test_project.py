"""Тесты слоя project: Workspace, обход файлов, glob, дерево, инструкции, промпт."""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from gigacrew.agent.prompts import SYSTEM_PROMPT, build_system_prompt, nested_instructions_prompt
from gigacrew.project.files import TREE_TRUNCATED, build_file_tree, glob_match, walk_files
from gigacrew.project.instructions import (
    INSTRUCTION_FILES,
    InstructionFile,
    NestedInstructions,
    load_instructions,
)
from gigacrew.project.workspace import Workspace


# ------------------------------ Workspace ------------------------------ #
def test_workspace_constructor_has_no_side_effects(tmp_path):
    ws = Workspace(tmp_path)
    assert ws.data_dir == tmp_path.resolve() / ".gigacode"
    assert ws.chats_dir == ws.data_dir / "chats"
    assert ws.index_dir == ws.data_dir / "index"
    assert ws.skills_dir == ws.data_dir / "skills"
    assert not ws.data_dir.exists()


def _gitignored(ws: Workspace) -> set[str]:
    lines = (ws.data_dir / ".gitignore").read_text(encoding="utf-8").splitlines()
    return {line.strip("/") for line in lines if line and not line.startswith("#")}


def test_gitignore_covers_agent_data_but_not_skills(tmp_path):
    """Рабочие данные агента не коммитятся, а скиллы проекта — коммитятся."""
    ws = Workspace(tmp_path)
    ws.ensure_data_dir()
    ignored = _gitignored(ws)
    for path in (ws.chats_dir, ws.plans_dir, ws.index_dir, ws.history_file):
        assert path.relative_to(ws.data_dir).as_posix() in ignored
    assert not any(p == "*" or p.startswith("skills") for p in ignored)


@pytest.mark.skipif(shutil.which("git") is None, reason="нужен git")
def test_gitignore_keeps_skills_in_git(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    ws = Workspace(tmp_path)
    ws.ensure_data_dir()

    def ignored(rel: str) -> bool:
        run = subprocess.run(["git", "-C", str(tmp_path), "check-ignore", "-q", rel])
        return run.returncode == 0

    for rel in (
        ".gigacode/chats/a.json",
        ".gigacode/plans/a.md",
        ".gigacode/history",
        ".gigacode/index/index.sqlite3",
    ):
        assert ignored(rel), rel
    for rel in (".gigacode/skills/demo/SKILL.md", ".gigacode/.gitignore"):
        assert not ignored(rel), rel


def test_ensure_data_dir_creates_gitignore_idempotently(tmp_path):
    ws = Workspace(tmp_path)
    assert ws.ensure_data_dir() == ws.data_dir
    gi = ws.data_dir / ".gitignore"
    assert "/chats/" in gi.read_text(encoding="utf-8").splitlines()
    gi.write_text("# мой\n*\n!keep\n", encoding="utf-8")
    ws.ensure_data_dir()  # повторно — без ошибок и без перезаписи
    assert gi.read_text(encoding="utf-8") == "# мой\n*\n!keep\n"


# ------------------------------ glob_match ------------------------------ #
@pytest.mark.parametrize(
    "path, pattern, expected",
    [
        ("a.py", "*.py", True),
        ("src/x/a.py", "*.py", True),  # без '/' — по имени файла
        ("src/a.py", "src/*.py", True),
        ("src/x/a.py", "src/*.py", False),  # '*' не пересекает '/'
        ("src/a.py", "src/**/*.py", True),  # '**/' — ноль и более каталогов
        ("src/x/y/a.py", "src/**/*.py", True),
        ("lib/a.py", "src/**/*.py", False),
        ("src/a.py", "./src/*.py", True),
        ("a1.txt", "a?.txt", True),
        ("ab.txt", "a[!b].txt", False),
        ("ac.txt", "a[!b].txt", True),
        ("x/README.md", "**/README.md", True),
        ("README.md", "**/README.md", True),
        ("a+b.txt", "a+b.txt", True),  # спецсимволы regex экранируются
    ],
)
def test_glob_match(path, pattern, expected):
    assert glob_match(path, pattern) is expected


# ------------------------------ walk / tree ------------------------------ #
def _make(root, rel, text="x"):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_walk_files_prunes_ignored_and_sorts(tmp_path):
    for rel in ("b.py", "a.py", "src/c.py", "node_modules/m.js", ".git/HEAD", "pkg.egg-info/x"):
        _make(tmp_path, rel)
    rels = [p.relative_to(tmp_path).as_posix() for p in walk_files(tmp_path)]
    assert rels == ["a.py", "b.py", "src/c.py"]


@pytest.mark.skipif(os.name == "nt", reason="симлинки")
def test_walk_files_skips_symlinks_outside_root(tmp_path):
    root, outside = tmp_path / "root", tmp_path / "outside"
    _make(outside, "secret.txt")
    _make(root, "in.txt")
    (root / "link.txt").symlink_to(outside / "secret.txt")
    (root / "inner_link.txt").symlink_to(root / "in.txt")
    rels = sorted(p.name for p in walk_files(root))
    assert rels == ["in.txt", "inner_link.txt"]


def test_file_tree_shows_dotfiles_hides_service_dirs(tmp_path):
    _make(tmp_path, "src/main.py")
    _make(tmp_path, ".github/workflows/ci.yml")
    _make(tmp_path, "node_modules/junk.js")
    _make(tmp_path, ".git/HEAD")
    _make(tmp_path, ".gigacode/chats/1.json")
    tree = build_file_tree(tmp_path)
    assert "src/" in tree and "main.py" in tree
    assert ".github/" in tree
    for hidden in ("node_modules", ".git/", ".gigacode"):
        assert hidden not in tree


def test_file_tree_keeps_root_files_when_a_directory_is_big(tmp_path):
    for i in range(300):
        _make(tmp_path, f"big/f{i:03d}.py")
    _make(tmp_path, "README.md")
    _make(tmp_path, "pyproject.toml")
    _make(tmp_path, "zzz/main.py")
    tree = build_file_tree(tmp_path, max_entries=50)
    lines = tree.splitlines()
    for name in ("big/", "README.md", "pyproject.toml", "zzz/", "  main.py"):
        assert name in lines
    assert "  … 270 more" in lines  # big/ урезан до TREE_DIR_CAP
    assert tree.endswith(TREE_TRUNCATED)


def test_file_tree_marks_unexpanded_directories(tmp_path):
    for d in range(5):
        _make(tmp_path, f"d{d}/sub/x.py")
    tree = build_file_tree(tmp_path, max_entries=7)
    assert "d0/" in tree and "  sub/ …" in tree and tree.endswith(TREE_TRUNCATED)


def test_empty_directory_does_not_hide_siblings_from_the_budget(tmp_path):
    (tmp_path / "a").mkdir()
    _make(tmp_path, "b/main.py")
    tree = build_file_tree(tmp_path, max_entries=3)
    assert "  main.py" in tree.splitlines()


def test_small_file_tree_is_complete(tmp_path):
    _make(tmp_path, "src/pkg/a.py")
    _make(tmp_path, "src/b.py")
    _make(tmp_path, "README.md")
    assert build_file_tree(tmp_path) == "src/\n  pkg/\n    a.py\n  b.py\nREADME.md"


def test_in_git_repo(tmp_path):
    from gigacrew.project.workspace import Workspace

    assert not Workspace(tmp_path).in_git_repo()
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    (tmp_path / "repo" / "sub").mkdir()
    assert Workspace(tmp_path / "repo").in_git_repo()
    assert Workspace(tmp_path / "repo" / "sub").in_git_repo()  # предок — репозиторий
    (tmp_path / "wt").mkdir()
    (tmp_path / "wt" / ".git").write_text("gitdir: ../repo/.git/worktrees/wt\n")
    assert Workspace(tmp_path / "wt").in_git_repo()  # рабочее дерево: .git — файл


# ------------------------------ instructions ------------------------------ #
def test_load_instructions_order_and_truncation(tmp_path):
    (tmp_path / "GIGACREW.local.md").write_text("личное", encoding="utf-8")
    (tmp_path / "GIGACREW.md").write_text("x" * 50, encoding="utf-8")
    docs = load_instructions(tmp_path, max_chars=10)
    assert [d.name for d in docs] == ["GIGACREW.md", "GIGACREW.local.md"]
    assert docs[0].truncated and docs[0].content == "x" * 10
    assert docs[1].content == "личное" and not docs[1].truncated


def test_load_instructions_missing_and_empty(tmp_path):
    (tmp_path / "GIGACREW.md").write_text("   \n", encoding="utf-8")
    assert load_instructions(tmp_path) == []


def test_instruction_files_in_priority_order(tmp_path):
    for name, text in [
        ("GIGACREW.local.md", "личное"),
        ("GIGACREW.md", "свои правила"),
        ("AGENTS.md", "для всех агентов"),
    ]:
        (tmp_path / name).write_text(text, encoding="utf-8")
    docs = load_instructions(tmp_path)
    assert [d.name for d in docs] == list(INSTRUCTION_FILES)
    assert INSTRUCTION_FILES[:2] == ("AGENTS.md", "GIGACREW.md")
    prompt = build_system_prompt(Workspace(tmp_path))
    positions = [prompt.index(text) for text in ("для всех", "свои", "личное")]
    assert positions == sorted(positions)
    for name in INSTRUCTION_FILES:  # правило приоритета называет все файлы
        assert name in SYSTEM_PROMPT


def test_instruction_symlink_counted_once(tmp_path):
    (tmp_path / "AGENTS.md").write_text("общие правила", encoding="utf-8")
    try:
        (tmp_path / "GIGACREW.md").symlink_to("AGENTS.md")
    except OSError:
        pytest.skip("симлинки недоступны")
    assert [d.name for d in load_instructions(tmp_path)] == ["AGENTS.md"]


def test_instruction_total_limit_prefers_higher_priority(tmp_path):
    (tmp_path / "AGENTS.md").write_text("a" * 50, encoding="utf-8")
    (tmp_path / "GIGACREW.md").write_text("d" * 50, encoding="utf-8")
    agents, own = load_instructions(tmp_path, max_total=70)
    assert own.content == "d" * 50 and not own.truncated
    assert agents.content == "a" * 20 and agents.truncated


def test_oversized_instructions_point_to_file(tmp_path):
    for name in ("AGENTS.md", "GIGACREW.md", "GIGACREW.local.md"):
        (tmp_path / name).write_text(name[0] * 25_000, encoding="utf-8")
    agents, own, local = load_instructions(tmp_path)  # 20K на файл, 30K на все
    assert (len(local.content), len(own.content), agents.content) == (20_000, 10_000, "")
    prompt = build_system_prompt(Workspace(tmp_path))
    assert "(AGENTS.md): too large for the context" in prompt
    assert "(GIGACREW.md) (truncated — read the whole file with read_file)" in prompt


# ------------------------- инструкции подкаталогов ------------------------- #
def test_nested_instructions_found_outside_in_once(tmp_path):
    _make(tmp_path, "AGENTS.md", "корень")
    _make(tmp_path, "pkg/AGENTS.md", "пакет")
    _make(tmp_path, "pkg/api/GIGACREW.md", "api")
    _make(tmp_path, "pkg/api/handlers.py")
    nested = NestedInstructions(tmp_path)
    found = nested.add_paths(["pkg/api/handlers.py"])
    assert [(d.name, d.content) for d in found] == [
        ("pkg/AGENTS.md", "пакет"),
        ("pkg/api/GIGACREW.md", "api"),
    ]
    assert nested.add_paths(["pkg/api", "pkg/api/handlers.py", "pkg"]) == []  # уже проверены
    assert len(nested.files) == 2
    nested.reset()
    assert nested.files == () and len(nested.add_paths(["pkg/api"])) == 2  # каталог как путь


def test_nested_instructions_skip_ignored_dirs_and_outside_paths(tmp_path):
    _make(tmp_path, ".gitignore", "build/\nGIGACREW.local.md\n")
    _make(tmp_path, "node_modules/lib/AGENTS.md", "чужое")
    _make(tmp_path, "build/AGENTS.md", "сгенерировано")
    _make(tmp_path, ".gigacode/AGENTS.md", "служебное")
    _make(tmp_path, "src/GIGACREW.local.md", "моё")  # сам файл в .gitignore — не важно
    nested = NestedInstructions(tmp_path)
    found = nested.add_paths(
        ["node_modules/lib/x.js", "build/out.js", ".gigacode/x", "../x", "/etc/passwd", "", "src"]
    )
    assert [d.name for d in found] == ["src/GIGACREW.local.md"]
    assert nested.add_paths(["README.md", "."]) == []  # корень — уже в системном промпте


def test_nested_instructions_prompt_limits(tmp_path):
    assert nested_instructions_prompt([]) == ""
    docs = [
        InstructionFile("a/AGENTS.md", "правило a"),
        InstructionFile("b/AGENTS.md", "b" * 30_000),
    ]
    prompt = nested_instructions_prompt(docs)
    assert "Instructions of directory a (a/AGENTS.md):\nправило a" in prompt
    assert (
        "bbbb" not in prompt and "Too large for the context" in prompt and "b/AGENTS.md" in prompt
    )


def test_system_prompt_contains_project_context(tmp_path):
    _make(tmp_path, "src/main.py")
    (tmp_path / "GIGACREW.md").write_text("важная заметка", encoding="utf-8")
    prompt = build_system_prompt(Workspace(tmp_path))
    assert str(tmp_path.resolve()) in prompt
    assert "важная заметка" in prompt
    assert "Project structure" in prompt and "main.py" in prompt


def test_system_prompt_points_large_projects_to_index_tools(tmp_path):
    _make(tmp_path, "small.py")
    assert "find_symbol" in build_system_prompt(Workspace(tmp_path))  # в правилах работы
    assert "the tree is incomplete" not in build_system_prompt(Workspace(tmp_path))
    for i in range(250):
        _make(tmp_path, f"pkg/m{i:03}.py")
    assert "the tree is incomplete" in build_system_prompt(Workspace(tmp_path))
