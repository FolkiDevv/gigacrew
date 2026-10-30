"""Тесты слоя project: Workspace, обход файлов, glob, дерево, инструкции, промпт."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from gigacrew.agent.decisions import project_key
from gigacrew.agent.prompts import SYSTEM_PROMPT, build_system_prompt, nested_instructions_prompt
from gigacrew.project.files import (
    TREE_TRUNCATED,
    build_file_tree,
    glob_match,
    is_git_internal,
    is_secret_file,
    is_secret_path,
    is_within,
    walk_files,
)
from gigacrew.project.instructions import (
    INSTRUCTION_FILES,
    InstructionFile,
    NestedInstructions,
    load_instructions,
)
from gigacrew.project.workspace import TEMP_DIR_NAME, Workspace


# ------------------------------ Workspace ------------------------------ #
def test_workspace_constructor_has_no_side_effects(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    home: Path = request.getfixturevalue("_isolated_home")
    ws = Workspace(tmp_path)
    assert ws.data_dir == tmp_path.resolve() / ".gigacode"
    # чаты — у пользователя, рядом с разрешениями проекта; в проекте — только прежнее место
    assert ws.chats_dir == home / ".gigacode" / "projects" / project_key(tmp_path) / "chats"
    assert ws.legacy_chats_dir == ws.data_dir / "chats"
    assert ws.index_dir == ws.data_dir / "index"
    assert ws.skills_dir == ws.data_dir / "skills"
    assert not ws.data_dir.exists()


def _gitignored(ws: Workspace) -> set[str]:
    lines = (ws.data_dir / ".gitignore").read_text(encoding="utf-8").splitlines()
    return {line.strip("/") for line in lines if line and not line.startswith("#")}


def test_gitignore_covers_agent_data_but_not_skills(tmp_path: Path) -> None:
    """Рабочие данные агента не коммитятся, а скиллы проекта — коммитятся."""
    ws = Workspace(tmp_path)
    ws.ensure_data_dir()
    ignored = _gitignored(ws)
    for path in (
        ws.legacy_chats_dir,  # что не удалось перенести, тоже не коммитится
        ws.plans_dir,
        ws.index_dir,
        ws.data_dir / TEMP_DIR_NAME,
        ws.history_file,
        ws.local_settings_file,
    ):
        assert path.relative_to(ws.data_dir).as_posix() in ignored
    assert not any(entry == "*" or entry.startswith("skills") for entry in ignored)
    assert "settings.json" not in ignored


@pytest.mark.skipif(shutil.which("git") is None, reason="нужен git")
def test_gitignore_keeps_skills_in_git(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    ws = Workspace(tmp_path)
    ws.ensure_data_dir()

    def ignored(rel: str) -> bool:
        run = subprocess.run(["git", "-C", str(tmp_path), "check-ignore", "-q", rel], check=False)
        return run.returncode == 0

    for rel in (
        ".gigacode/chats/a.json",
        ".gigacode/plans/a.md",
        ".gigacode/history",
        ".gigacode/index/index.sqlite3",
        ".gigacode/tmp/out.txt",
        ".gigacode/settings.local.json",
    ):
        assert ignored(rel), rel
    for rel in (
        ".gigacode/skills/demo/SKILL.md",
        ".gigacode/.gitignore",
        ".gigacode/settings.json",
    ):
        assert not ignored(rel), rel


def test_ensure_data_dir_creates_gitignore_idempotently(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    assert ws.ensure_data_dir() == ws.data_dir
    gi = ws.data_dir / ".gitignore"
    assert "/chats/" in gi.read_text(encoding="utf-8").splitlines()
    gi.write_text("# мой\n*\n!keep\n", encoding="utf-8")
    ws.ensure_data_dir()  # повторно — без ошибок и без перезаписи
    assert gi.read_text(encoding="utf-8") == "# мой\n*\n!keep\n"


@pytest.mark.skipif(os.name == "nt", reason="симлинки")
@pytest.mark.parametrize(
    "entry",
    ["", ".gitignore", "chats", "index", "plans", "tmp", "history"],
)
def test_ensure_data_dir_refuses_symlinks_out_of_it(tmp_path: Path, entry: str) -> None:
    """Симлинк на месте .gigacode или её служебной записи не уводит запись из проекта.

    Репозиторий может принести симлинк на месте .gigacode или её служебной записи (git
    восстанавливает симлинки при клоне): gigacrew не пишет по нему за пределы проекта.
    """
    root, outside = tmp_path / "proj", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    ws = Workspace(root)
    if entry:
        ws.data_dir.mkdir()
        target = outside if entry in {"chats", "index", "plans", "tmp"} else outside / "victim"
        (ws.data_dir / entry).symlink_to(target)  # .gitignore — висячий симлинк
    else:
        ws.data_dir.symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError, match="симлинк"):
        ws.ensure_data_dir()
    assert list(outside.iterdir()) == []


def test_ensure_data_dir_keeps_regular_entries(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    for path in (ws.legacy_chats_dir, ws.index_dir):
        path.mkdir(parents=True)
    ws.history_file.write_text("+a\n", encoding="utf-8")
    assert ws.ensure_data_dir() == ws.data_dir
    assert (ws.data_dir / ".gitignore").is_file()


@pytest.mark.skipif(shutil.which("git") is None, reason="нужен git")
def test_temp_dir_is_ignored_with_old_gitignore(tmp_path: Path) -> None:
    """В прежнем проекте .gigacode/.gitignore без /tmp/: временные файлы не коммитятся всё равно."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    ws = Workspace(tmp_path)
    ws.data_dir.mkdir()
    (ws.data_dir / ".gitignore").write_text("/chats/\n", encoding="utf-8")
    temp_dir = ws.ensure_temp_dir()
    assert temp_dir == ws.data_dir / TEMP_DIR_NAME
    assert ws.ensure_temp_dir() == temp_dir  # повторно — без ошибок
    (temp_dir / "out.txt").write_text("x", encoding="utf-8")
    run = subprocess.run(
        ["git", "-C", str(tmp_path), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "tmp/" not in run.stdout
    assert (ws.data_dir / ".gitignore").read_text(encoding="utf-8") == "/chats/\n"


@pytest.mark.parametrize(
    ("rel", "expected"),
    [
        (".gigacode/tmp", True),
        (".gigacode/tmp/a/b.txt", True),
        (".GIGACODE/TMP/x", True),  # без учёта регистра, как is_data_path
        (".gigacode/tmpx", False),
        (".gigacode/skills/tmp/x", False),
        (".gigacode", False),
        ("tmp/x", False),
    ],
)
def test_is_temp_path(tmp_path: Path, rel: str, *, expected: bool) -> None:
    ws = Workspace(tmp_path)
    assert ws.is_temp_path(ws.root / rel) is expected


# ------------------------------ glob_match ------------------------------ #
@pytest.mark.parametrize(
    ("path", "pattern", "expected"),
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
        ("src/ab/x.py", "src/a**/x.py", True),  # '**' внутри сегмента — как '*'
        ("src/ab/c/x.py", "src/a**/x.py", False),
    ],
)
def test_glob_match(path: str, pattern: str, *, expected: bool) -> None:
    assert glob_match(path, pattern) is expected


# ------------------------------ walk / tree ------------------------------ #
def _make(root: Path, rel: str, text: str = "x") -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_walk_files_prunes_ignored_and_sorts(tmp_path: Path) -> None:
    for rel in ("b.py", "a.py", "src/c.py", "node_modules/m.js", ".git/HEAD", "pkg.egg-info/x"):
        _make(tmp_path, rel)
    rels = [path.relative_to(tmp_path).as_posix() for path in walk_files(tmp_path)]
    assert rels == ["a.py", "b.py", "src/c.py"]


@pytest.mark.skipif(os.name == "nt", reason="симлинки")
def test_walk_files_skips_symlinks_outside_root(tmp_path: Path) -> None:
    root, outside = tmp_path / "root", tmp_path / "outside"
    _make(outside, "secret.txt")
    _make(root, "in.txt")
    (root / "link.txt").symlink_to(outside / "secret.txt")
    (root / "inner_link.txt").symlink_to(root / "in.txt")
    rels = sorted(path.name for path in walk_files(root))
    assert rels == ["in.txt", "inner_link.txt"]


@pytest.mark.skipif(os.name == "nt", reason="симлинки")
def test_sibling_with_root_prefix_is_outside(tmp_path: Path) -> None:
    # /x/root-evil — не внутри /x/root, хотя строка пути начинается так же
    root, evil = tmp_path / "root", tmp_path / "root-evil"
    _make(evil, "secret.txt")
    _make(root, "in.txt")
    (root / "link.txt").symlink_to(evil / "secret.txt")
    assert not is_within(root.resolve(), evil / "secret.txt")
    assert [path.name for path in walk_files(root)] == ["in.txt"]


@pytest.mark.skipif(os.name == "nt", reason="симлинки")
def test_git_internal_by_path_as_given(tmp_path: Path) -> None:
    # nested/.git — симлинк на обычный каталог: путь через .git — внутренности git, даже
    # если после раскрытия симлинка компонента .git в нём нет
    root = tmp_path.resolve()
    (root / "nested" / "plain").mkdir(parents=True)
    (root / "nested" / ".git").symlink_to(root / "nested" / "plain")
    resolved = (root / "nested" / ".git" / "config").resolve()
    assert is_git_internal(root, resolved, "nested/.git/config")
    assert not is_git_internal(root, resolved)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (".env", True),
        (".ENV", True),  # на macOS/Windows — тот же файл
        (".Env.Local", True),
        (".env.production", True),
        (".env.example", False),
        (".ENV.Example", False),
        ("env.py", False),
    ],
)
def test_secret_file_names(name: str, *, expected: bool) -> None:
    assert is_secret_file(name) is expected


@pytest.mark.skipif(os.name == "nt", reason="симлинки")
def test_secret_path_through_symlink(tmp_path: Path) -> None:
    _make(tmp_path, ".env", "TOKEN=1")
    (tmp_path / "settings.py").symlink_to(tmp_path / ".env")
    assert is_secret_path(tmp_path / "settings.py")
    assert not is_secret_path(tmp_path / ".env.example")


def test_file_tree_shows_dotfiles_hides_service_dirs(tmp_path: Path) -> None:
    _make(tmp_path, "src/main.py")
    _make(tmp_path, ".github/workflows/ci.yml")
    _make(tmp_path, "node_modules/junk.js")
    _make(tmp_path, ".git/HEAD")
    _make(tmp_path, ".gigacode/chats/1.json")
    tree = build_file_tree(tmp_path)
    assert "src/" in tree
    assert "main.py" in tree
    assert ".github/" in tree
    for hidden in ("node_modules", ".git/", ".gigacode"):
        assert hidden not in tree


def test_file_tree_keeps_root_files_when_a_directory_is_big(tmp_path: Path) -> None:
    for index in range(300):
        number = format(index, "03d")
        _make(tmp_path, f"big/f{number}.py")
    _make(tmp_path, "README.md")
    _make(tmp_path, "pyproject.toml")
    _make(tmp_path, "zzz/main.py")
    tree = build_file_tree(tmp_path, max_entries=50)
    lines = tree.splitlines()
    for name in ("big/", "README.md", "pyproject.toml", "zzz/", "  main.py"):
        assert name in lines
    assert "  … 270 more" in lines  # big/ урезан до TREE_DIR_CAP
    assert tree.endswith(TREE_TRUNCATED)


def test_file_tree_marks_unexpanded_directories(tmp_path: Path) -> None:
    for dir_index in range(5):
        _make(tmp_path, f"d{dir_index}/sub/x.py")
    tree = build_file_tree(tmp_path, max_entries=7)
    assert "d0/" in tree
    assert "  sub/ …" in tree
    assert tree.endswith(TREE_TRUNCATED)


def test_empty_directory_does_not_hide_siblings_from_the_budget(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    _make(tmp_path, "b/main.py")
    tree = build_file_tree(tmp_path, max_entries=3)
    assert "  main.py" in tree.splitlines()


def test_small_file_tree_is_complete(tmp_path: Path) -> None:
    _make(tmp_path, "src/pkg/a.py")
    _make(tmp_path, "src/b.py")
    _make(tmp_path, "README.md")
    assert build_file_tree(tmp_path) == "src/\n  pkg/\n    a.py\n  b.py\nREADME.md"


def test_in_git_repo(tmp_path: Path) -> None:
    assert not Workspace(tmp_path).in_git_repo()
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    (tmp_path / "repo" / "sub").mkdir()
    assert Workspace(tmp_path / "repo").in_git_repo()
    assert Workspace(tmp_path / "repo" / "sub").in_git_repo()  # предок — репозиторий
    (tmp_path / "wt").mkdir()
    (tmp_path / "wt" / ".git").write_text("gitdir: ../repo/.git/worktrees/wt\n")
    assert Workspace(tmp_path / "wt").in_git_repo()  # рабочее дерево: .git — файл


# ------------------------------ instructions ------------------------------ #
def test_load_instructions_order_and_truncation(tmp_path: Path) -> None:
    (tmp_path / "GIGACREW.local.md").write_text("личное", encoding="utf-8")
    (tmp_path / "GIGACREW.md").write_text("x" * 50, encoding="utf-8")
    docs = load_instructions(tmp_path, max_chars=10)
    assert [doc.name for doc in docs] == ["GIGACREW.md", "GIGACREW.local.md"]
    assert docs[0].truncated
    assert docs[0].content == "x" * 10
    assert docs[1].content == "личное"
    assert not docs[1].truncated


def test_load_instructions_missing_and_empty(tmp_path: Path) -> None:
    (tmp_path / "GIGACREW.md").write_text("   \n", encoding="utf-8")
    assert load_instructions(tmp_path) == []


def test_instruction_files_in_priority_order(tmp_path: Path) -> None:
    for name, text in (
        ("GIGACREW.local.md", "личное"),
        ("GIGACREW.md", "свои правила"),
        ("AGENTS.md", "для всех агентов"),
    ):
        (tmp_path / name).write_text(text, encoding="utf-8")
    docs = load_instructions(tmp_path)
    assert [doc.name for doc in docs] == list(INSTRUCTION_FILES)
    assert INSTRUCTION_FILES[:2] == ("AGENTS.md", "GIGACREW.md")
    prompt = build_system_prompt(Workspace(tmp_path))
    positions = [prompt.index(fragment) for fragment in ("для всех", "свои", "личное")]
    assert positions == sorted(positions)
    for name in INSTRUCTION_FILES:  # правило приоритета называет все файлы
        assert name in SYSTEM_PROMPT


def test_instruction_symlink_counted_once(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("общие правила", encoding="utf-8")
    try:
        (tmp_path / "GIGACREW.md").symlink_to("AGENTS.md")
    except OSError:
        pytest.skip("симлинки недоступны")
    assert [doc.name for doc in load_instructions(tmp_path)] == ["AGENTS.md"]


@pytest.mark.skipif(os.name == "nt", reason="симлинки")
def test_instruction_symlink_outside_project_is_not_loaded(tmp_path: Path) -> None:
    # Репозиторий не может подсунуть в системный промпт файл вне проекта.
    root = tmp_path / "proj"
    root.mkdir()
    (tmp_path / "secret.txt").write_text("TOKEN=abc", encoding="utf-8")
    (root / "AGENTS.md").symlink_to(tmp_path / "secret.txt")
    assert load_instructions(root) == []
    assert "TOKEN=abc" not in build_system_prompt(Workspace(root))


def test_instruction_total_limit_prefers_higher_priority(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("a" * 50, encoding="utf-8")
    (tmp_path / "GIGACREW.md").write_text("d" * 50, encoding="utf-8")
    agents, own = load_instructions(tmp_path, max_total=70)
    assert own.content == "d" * 50
    assert not own.truncated
    assert agents.content == "a" * 20
    assert agents.truncated


def test_oversized_instructions_point_to_file(tmp_path: Path) -> None:
    for name in ("AGENTS.md", "GIGACREW.md", "GIGACREW.local.md"):
        (tmp_path / name).write_text(name[0] * 25_000, encoding="utf-8")
    agents, own, local = load_instructions(tmp_path)  # 20K на файл, 30K на все
    assert (len(local.content), len(own.content), agents.content) == (20_000, 10_000, "")
    prompt = build_system_prompt(Workspace(tmp_path))
    assert "(AGENTS.md): too large for the context" in prompt
    assert "(GIGACREW.md) (truncated — read the whole file with read_file)" in prompt


# ------------------------- инструкции подкаталогов ------------------------- #
def test_nested_instructions_found_outside_in_once(tmp_path: Path) -> None:
    _make(tmp_path, "AGENTS.md", "корень")
    _make(tmp_path, "pkg/AGENTS.md", "пакет")
    _make(tmp_path, "pkg/api/GIGACREW.md", "api")
    _make(tmp_path, "pkg/api/handlers.py")
    nested = NestedInstructions(tmp_path)
    found = nested.add_paths(["pkg/api/handlers.py"])
    assert [(doc.name, doc.content) for doc in found] == [
        ("pkg/AGENTS.md", "пакет"),
        ("pkg/api/GIGACREW.md", "api"),
    ]
    assert nested.add_paths(["pkg/api", "pkg/api/handlers.py", "pkg"]) == []  # уже проверены
    files_count = len(nested.files)
    assert files_count == 2
    nested.reset()
    assert nested.files == ()
    assert len(nested.add_paths(["pkg/api"])) == 2


def test_nested_instructions_skip_ignored_dirs_and_outside_paths(tmp_path: Path) -> None:
    _make(tmp_path, ".gitignore", "build/\nGIGACREW.local.md\n")
    _make(tmp_path, "node_modules/lib/AGENTS.md", "чужое")
    _make(tmp_path, "build/AGENTS.md", "сгенерировано")
    _make(tmp_path, ".gigacode/AGENTS.md", "служебное")
    _make(tmp_path, "src/GIGACREW.local.md", "моё")  # сам файл в .gitignore — не важно
    nested = NestedInstructions(tmp_path)
    found = nested.add_paths(
        ["node_modules/lib/x.js", "build/out.js", ".gigacode/x", "../x", "/etc/passwd", "", "src"],
    )
    assert [doc.name for doc in found] == ["src/GIGACREW.local.md"]
    assert nested.add_paths(["README.md", "."]) == []  # корень — уже в системном промпте


def test_nested_instructions_prompt_limits(tmp_path: Path) -> None:
    assert not nested_instructions_prompt([])
    docs = [
        InstructionFile("a/AGENTS.md", "правило a"),
        InstructionFile("b/AGENTS.md", "b" * 30_000),
    ]
    prompt = nested_instructions_prompt(docs)
    assert "Instructions of directory a (a/AGENTS.md):\nправило a" in prompt
    assert "bbbb" not in prompt
    assert "Too large for the context" in prompt
    assert "b/AGENTS.md" in prompt


def test_nested_instructions_budget_is_shared(tmp_path: Path) -> None:
    # Лимит — на все инструкции подкаталогов вместе: каждая меньше лимита, но вместе
    # они больше — последние перечисляются путями.
    docs = [InstructionFile(f"d{index}/AGENTS.md", str(index) * 8_000) for index in range(4)]
    prompt = nested_instructions_prompt(docs)
    assert "0" * 8_000 in prompt
    assert "1" * 8_000 in prompt
    assert "2" * 8_000 not in prompt
    assert "3" * 8_000 not in prompt
    skipped = prompt.split("Too large for the context")[1]
    assert "d2/AGENTS.md" in skipped
    assert "d3/AGENTS.md" in skipped
    assert "d0/AGENTS.md" not in skipped
    assert "d1/AGENTS.md" not in skipped


def test_system_prompt_contains_project_context(tmp_path: Path) -> None:
    _make(tmp_path, "src/main.py")
    (tmp_path / "GIGACREW.md").write_text("важная заметка", encoding="utf-8")
    prompt = build_system_prompt(Workspace(tmp_path))
    assert str(tmp_path.resolve()) in prompt
    assert "важная заметка" in prompt
    assert "Project structure" in prompt
    assert "main.py" in prompt


def test_system_prompt_points_large_projects_to_index_tools(tmp_path: Path) -> None:
    _make(tmp_path, "small.py")
    assert "find_symbol" in build_system_prompt(Workspace(tmp_path))  # в правилах работы
    assert "the tree is incomplete" not in build_system_prompt(Workspace(tmp_path))
    for index in range(250):
        number = format(index, "03")
        _make(tmp_path, f"pkg/m{number}.py")
    assert "the tree is incomplete" in build_system_prompt(Workspace(tmp_path))
