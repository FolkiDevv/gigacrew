.DEFAULT_GOAL := help
UV ?= uv

.PHONY: help install dev lock run test test-all lint format check clean

help: ## Показать список команд
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Установить runtime-зависимости в .venv (без dev-группы)
	$(UV) sync --no-dev

dev: ## Установить пакет + dev-группу (pytest, ruff)
	$(UV) sync

lock: ## Обновить uv.lock после правки зависимостей в pyproject.toml
	$(UV) lock

run: ## Запустить gigacrew (REPL) из .venv
	$(UV) run gigacrew

test: ## Быстрые офлайн-тесты (без сети)
	$(UV) run pytest -m "not live"

test-all: ## Все тесты, включая live (нужны реквизиты API, см. .env.example)
	$(UV) run pytest

lint: ## Проверить стиль и ошибки (ruff)
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format: ## Автоформатирование и автофиксы (ruff)
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

check: lint test ## Линт + офлайн-тесты (запускайте перед пушем)
	$(UV) lock --check

clean: ## Удалить кеши и артефакты сборки
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
