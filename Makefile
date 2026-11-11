PY ?= python

.PHONY: setup test lint experiments azure

setup:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src scripts tests

experiments:   # ~20 minutes: tables, artifacts/results.json, robustness.png, learned.png
	$(PY) scripts/run_experiments.py

azure:         # real Azure VM trace (one-time 53 MB download into data/), ~1 minute
	$(PY) scripts/run_azure.py
