.PHONY: check ci lint format typecheck test coverage
check: lint typecheck test
ci: check
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
coverage:
	uv run pytest --cov --cov-report=term-missing
