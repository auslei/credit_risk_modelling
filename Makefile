.PHONY: help venv install train evaluate test clean

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

DATA := data/loan_data_2007_2014.csv
MODEL := artifacts/model.joblib
METRICS := artifacts/metrics.json

help:
	@echo "Available targets: venv, install, train, evaluate, test, clean"

venv:
	@test -d $(VENV) || python -m venv $(VENV)

install: venv
	@$(PIP) install -r requirements.txt

train: install
	@$(PY) -m scripts.train --input $(DATA) --out $(MODEL)

evaluate: install
	@$(PY) -m scripts.evaluate --input $(DATA) --model $(MODEL) --out $(METRICS)

test: install
	@$(PY) -m pytest -q

clean:
	@rm -rf artifacts .pytest_cache **/__pycache__ **/*.pyc .mypy_cache

