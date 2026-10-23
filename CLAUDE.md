# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

gigacrew is a personal console coding assistant (a CLI agent) for the GigaChat API v2 and the
GigaCode backend. Python 3.13+, managed with uv; flat layout (the `gigacrew` package sits at the
repository root). Comments, docstrings, user-facing text, README and commit messages are in
Russian; commit messages follow conventional commits (`fix(safety): …`, `test(agent): …`). Keep
both conventions.

## Commands

```sh
make dev       # uv sync: package + dev group (pytest, linters, mypy)
make format    # ruff check --fix + ruff format
make lint      # every static check, same as the CI lint job
make test      # offline tests: pytest -m "not live"
make check     # lint + offline tests — run before pushing
make lock      # after editing dependencies in pyproject.toml
make audit     # pip-audit (not blocking in CI)
```

- One test: `uv run pytest tests/test_safety.py::test_name` (or `-k pattern`).
- Test order is random (pytest-randomly). Add `-p no:randomly` for a fixed order; reproduce a
  failing order with the `--randomly-seed=N` printed in the pytest header.
- `live`/`eval` tests call real APIs and need credentials from `.env.example`: `make test-all`,
  `make eval` (expensive behavioural scenarios), `make probe`.

## Verification is mandatory

A change is done only when `make lint` and `make test` both pass. `make lint` runs ruff (check and
format), flake8 with wemake-python-styleguide, import-linter (`lint-imports`), mypy `--strict` over
both `gigacrew/` and `tests/`, codespell, typos, yamllint, dotenv-linter, deptry, `uv lock --check`
and `uv pip check`. CI runs the same checks plus the offline tests on Python 3.13 and 3.14.
flake8, codespell and yamllint do not honour `.gitignore`: stray worktrees or venvs inside the
repository (for example `.claude/worktrees/`) make `make lint` slow and noisy — remove them first.

## Linter configuration is frozen

Do not change linter or type-checker configuration without the user's explicit consent. This
covers `[tool.ruff*]`, `[tool.mypy]`, `[tool.pytest.ini_options]`, `[tool.deptry]`,
`[tool.codespell]` and `[tool.typos]` in `pyproject.toml`; `setup.cfg` (wemake thresholds,
allowed names, `per-file-ignores`); `.importlinter`; `.yamllint.yml`; and the lint steps in the
`Makefile` and `.github/workflows/ci.yml`. Adding a per-file ignore, raising a threshold or
disabling a rule counts as a change: fix the code instead, or ask.

An inline `# noqa: CODE` / `# type: ignore[code]` is acceptable only for a genuine false positive,
always with a specific code and a one-line reason comment on the line above (never inside the
noqa comment). wemake caps a file at 10 noqa comments (WPS402).

Conventions the configuration enforces that are easy to trip over:
- Package `__init__.py` files contain only imports; re-export as `from x import Name as Name`
  (WPS412 plus mypy's no-implicit-reexport). Other modules list their exports in `__all__`.
- Overriding methods carry `@override` (mypy `explicit-override`).
- No single-letter names and no wemake-blacklisted names (`data`, `result`, `value`, `item`,
  `info`, …); `params`, `content` and `handler` are allowed as domain terms.
- Line length 100, double quotes, ruff preview formatting.

## Architecture

### Layers

`.importlinter` enforces the layering; a layer may import only the layers below it:

```
__main__ → cli → ui → agent → tools → llm | safety → config → project → permissions → security → errors
```

Everything except `cli` and `ui` is the core and must not import `rich` or `prompt_toolkit`. Only
`project/treesitter.py` imports `tree_sitter`; grammar packages are loaded by name through
importlib. `tests/test_layering.py` also checks at runtime that every module imports on its own
(no import cycles) and that importing the core never loads UI modules.

### Request flow

- `cli/app.py:main` parses arguments, loads `Config` (environment, then `~/.gigacode/.env`, then
  the project `.env`; security settings such as API URLs and TLS options are never taken from a
  project `.env`), builds the provider (`llm/api.py` for API v2, `llm/gigacode.py` for the GigaCode
  backend), the tool registry and the `Agent`, then runs the REPL (`cli/repl.py`) or a one-shot
  `-p` turn.
- `agent/loop.py` — `Agent.run_turn` streams from the model and executes tool calls. Each call
  passes through `Tool.assess` → `permissions.decide` (modes `auto`/`plan`, tool kinds) → for
  shell commands the local classifier → saved decisions (`agent/decisions.py`, stored per project
  in `~/.gigacode/projects/<key>/`) → in `auto` mode a reviewer model for mutating commands
  (`agent/reviewer.py`) → user confirmation. `agent/guard.py` stops loops and error series,
  `agent/compaction.py` shrinks the history, `agent/subagents.py` runs the `task`/explore
  subagents, `agent/hooks.py` runs user hooks.
- The agent never talks to the terminal directly: it reports through the `AgentEvents` protocol
  (`agent/events.py`), which `ui/console.py` implements with rich. Tests drive the agent with
  `tests/fakes.py` (`ScriptedProvider`, `RecordingEvents`, `FakeTool`).

### Tools

`tools/base.py` defines `Tool[P]`, generic over a pydantic `Params` model; the function-calling
JSON schema is derived from `Params` (model docstrings are dropped from it). A new tool is a
subclass registered in `build_default_registry`. File tools stay inside the project via
`security.resolve_in_root` and refuse secret files through `project/secret_paths.py`, the same
rule the command classifier uses. Child processes run through `tools/process.py` with the
environment from `security.subprocess_env`, which strips `GIGACREW_*` secrets.

### Command-safety classifier (security boundary)

`safety.classify` decides whether a shell command only reads the project: tree-sitter-bash parse
(`safety/bash.py`) → per-program policies (`safety/programs.py`) → AST analysis of inline Python
(`safety/python.py`). It is default-deny: anything unfamiliar is "unknown" and goes to the user.
Change it together with `tests/test_safety*.py` and the labelled corpus in
`tests/safety_corpus.py`.

### Project knowledge

`project/` holds the workspace (`.gigacode/` in the project, `~/.gigacode/` for user data), the
SQLite symbol index (`project/index.py`, tree-sitter), precise Python navigation through the ty
LSP server (`project/semantic.py`), the repo map, skills, settings and project instructions. The
agent reads its project instructions from `AGENTS.md`, `GIGACREW.md` and `GIGACREW.local.md`,
not from this file.
