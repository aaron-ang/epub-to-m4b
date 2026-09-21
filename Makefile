.PHONY: check lint format typecheck test
check: lint typecheck test
lint:
	uv run ruff check
	uv run ruff format --check
format:
	uv run ruff format
	uv run ruff check --fix
typecheck:
	uv run mypy
test:
	uv run pytest
