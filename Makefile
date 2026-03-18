.PHONY: install install-all test lint format typecheck check run validate info build clean

# Development
install:
	uv sync --group dev

install-all:
	uv sync --group dev --all-extras

# Quality
test:
	uv run pytest

test-cov:
	uv run pytest --cov=agentloom --cov-report=term-missing

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

typecheck:
	uv run mypy src/

check: lint typecheck test

# CLI shortcuts
run:
	uv run agentloom run $(WORKFLOW)

validate:
	uv run agentloom validate $(WORKFLOW)

info:
	uv run agentloom info

# Build
build:
	uv build --wheel

# Cleanup
clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
