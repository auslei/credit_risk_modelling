# Repository Guidelines

## Project Structure & Module Organization
- src/credit_risk: Core package. Modules: `config.py` (paths, seeds), `data_io.py` (load/save), `preprocessing.py` (cleaning, labeling), `features.py` (ColumnTransformer), `model.py` (pipeline builder), `evaluate.py` (metrics).
- scripts: Entry points for training/evaluation (`train.py`, `evaluate.py`).
- data: Raw CSV (`loan_data_2007_2014.csv`), intermediate pickles if needed.
- notebooks: Exploratory work; keep production logic in `src/`.

## Build, Test, and Development Commands
- Create venv: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt`
- Train locally: `python -m scripts.train --input data/loan_data_2007_2014.csv --out artifacts/model.joblib`
- Evaluate: `python -m scripts.evaluate --input data/loan_data_2007_2014.csv --model artifacts/model.joblib`
- Lint (optional): `ruff check .` or `flake8 .` if installed.

## Coding Style & Naming Conventions
- Python 3.10+. Use black-compatible formatting (4‑space indent, 88 cols). Snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants.
- Prefer pure, deterministic functions. Parameterize dates and thresholds via `config.py` or CLI flags.
- Avoid in-notebook logic duplication; import from `src/credit_risk` instead.

## Testing Guidelines
- Framework: `pytest` (add later). Place tests under `tests/` mirroring module names (e.g., `tests/test_preprocessing.py`).
- Minimum: unit tests for `process_emp_length`, `months_from_today`, high-null/single-value filters, and correlation filter.
- Run: `pytest -q`.

## Commit & Pull Request Guidelines
- Commits: short imperative subject, scope prefix when helpful (e.g., `feat(preprocessing): add emp_length parser`).
- PRs: include purpose, approach, and screenshots/metrics (AUROC/Gini) where relevant. Link issues and note data/path assumptions.
- Keep diffs focused; refactors and behavior changes go in separate PRs.

## Security & Configuration Tips
- Do not commit raw data or credentials. Paths and seeds live in `config.py`; override via CLI.
- Beware target leakage: drop identity, location, and future-event columns before modeling.
- Set `random_state=42` and use stratified splits for reproducibility.

