.PHONY: setup venv install-uv sync lint format test clean cdsw

# Override if needed, e.g. `make venv PYTHON=python3.12`
PYTHON ?= python
VENV   := .venv

# Windows (Git Bash) puts the venv interpreter in Scripts/python.exe, Linux/macOS in bin/python.
# $(OS) is set to Windows_NT by Windows itself.
ifeq ($(OS),Windows_NT)
VENV_PY  := $(VENV)/Scripts/python.exe
ACTIVATE := source .venv/Scripts/activate
else
VENV_PY  := $(VENV)/bin/python
ACTIVATE := source .venv/bin/activate
endif

# Create the virtual environment only if it does not exist yet.
# Because the target is the interpreter file itself, make skips this when .venv is present.
$(VENV_PY):
	$(PYTHON) -m venv $(VENV)
	@echo "Created $(VENV)"

venv: $(VENV_PY)

# Install uv only if it is not already available (on PATH or inside .venv).
install-uv: venv
	@if command -v uv >/dev/null 2>&1 || $(VENV_PY) -m uv --version >/dev/null 2>&1; then \
		echo "uv already installed"; \
	else \
		echo "uv not found, installing into $(VENV)..."; \
		$(VENV_PY) -m pip install --quiet uv; \
	fi

setup: venv
	$(VENV_PY) -m pip install -e ".[dev]"
	@echo "Done. Activate with: $(ACTIVATE)"

# Recommended: installs the exact versions pinned in uv.lock (including dev tools).
sync: install-uv
	@if command -v uv >/dev/null 2>&1; then \
		uv sync --dev; \
	else \
		$(VENV_PY) -m uv sync --dev; \
	fi
	@echo "Done. Activate with: $(ACTIVATE)"

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