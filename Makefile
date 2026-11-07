PY ?= python

.PHONY: setup test lint experiments

setup:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src scripts tests

experiments:   # ~20 minutes: tables, artifacts/results.json, robustness.png, learned.png
	$(PY) scripts/run_experiments.py
