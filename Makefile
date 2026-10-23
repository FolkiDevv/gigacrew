.DEFAULT_GOAL := help
UV ?= uv

.PHONY: help install dev lock run test test-all eval probe lint audit format check clean

help: ## Показать список команд
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Установить runtime-зависимости в .venv (без dev-группы)
	$(UV) sync --no-dev

dev: ## Установить пакет + dev-группу (pytest, линтеры, mypy)
	$(UV) sync

lock: ## Обновить uv.lock после правки зависимостей в pyproject.toml
	$(UV) lock

run: ## Запустить gigacrew (REPL) из .venv
	$(UV) run gigacrew

test: ## Быстрые офлайн-тесты (без сети)
	$(UV) run pytest -m "not live"

test-all: ## Все тесты, включая live, кроме eval (нужны реквизиты API, см. .env.example)
	$(UV) run pytest -m "not eval"

eval: ## Поведенческие сценарии на живых моделях — дорого (ARGS="-k plan_exit" — выборка)
	$(UV) run pytest -m eval -v $(ARGS)

probe: ## Пробы шлюза: рассуждения, top_p, кеш префикса (live)
	$(UV) run pytest tests/test_api_probe_live.py -v $(ARGS)

lint: ## Все статические проверки — как в CI (по образцу wemake-django-template)
	$(UV) run ruff check .
	$(UV) run ruff format --check --diff .
	$(UV) run flake8 .
	$(UV) run lint-imports
	$(UV) run mypy
	$(UV) run codespell .
	$(UV) run typos .
	$(UV) run yamllint -s .
	$(UV) run dotenv-linter .env.example
	$(UV) run deptry .
	$(UV) lock --check
	$(UV) pip check

audit: ## Известные уязвимости зависимостей (pip-audit; в CI не блокирует)
	$(UV) run pip-audit --skip-editable --desc

format: ## Автоформатирование и автофиксы (ruff)
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

check: lint test ## Линт + офлайн-тесты (запускайте перед пушем)

clean: ## Удалить кеши и артефакты сборки
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
