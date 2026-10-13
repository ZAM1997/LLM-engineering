.PHONY: setup lint format test clean cdsw

setup:
	python -m venv .venv
	.venv/bin/pip install -e ".[dev]"
	@echo "Done. Activate with: source .venv/bin/activate"

lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

test:
	pytest

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache

sync:
	python -m uv sync --dev

