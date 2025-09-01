# Credit Risk Modelling

A lightweight, reproducible pipeline for credit default prediction using LendingClub-style data. The project separates exploratory notebooks from production code and provides simple CLIs for training and evaluation.

## Directory Structure
- src/credit_risk: Core package
  - config.py: Paths, seeds, thresholds
  - data_io.py: CSV/Pickle load & save
  - preprocessing.py: Cleaning, labeling, date/field transforms
  - features.py: ColumnTransformer + imputers/encoders
  - model.py: Pipeline assembly and CV training
  - evaluate.py: AUROC/Gini/AUPR utilities
- scripts: Command‑line entry points (`train.py`, `evaluate.py`)
- data: Place raw `loan_data_2007_2014.csv` here
- notebooks: For EDA; import from `src/credit_risk` in notebooks
- tests: Minimal unit tests for preprocessing
- artifacts: Models/metrics output (created at runtime)
- AGENTS.md: Contributor guidelines
- requirements.txt: Python dependencies

## Setup
- Python 3.10+
- Create env and install:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`

## Usage
- Train model (Logistic Regression, balanced):
  - `python -m scripts.train --input data/loan_data_2007_2014.csv --out artifacts/model.joblib`
  - Prints cross‑validated AUROC and saves the fitted pipeline.
- Evaluate model (hold‑in evaluation with consistent preprocessing):
  - `python -m scripts.evaluate --input data/loan_data_2007_2014.csv --model artifacts/model.joblib --out artifacts/metrics.json`
  - Prints and saves `{\"auroc\", \"gini\", \"aupr\"}`.
- Run tests:
  - `pytest -q`

## Data & Features
- Target: `label` (derived from `loan_status`). Leakage‑prone fields (IDs, location, future events, free text) are dropped.
- Dates (`earliest_cr_line`, `issue_d`, etc.) converted to months from `2020‑01‑01` (configurable).
- Preprocessing uses `SimpleImputer`, `StandardScaler` for numerics; `OneHotEncoder(handle_unknown='ignore')` for categoricals.

## Configuration & Reproducibility
- Edit `src/credit_risk/config.py` for thresholds (null/correlation), today’s date, and random seed (default 42).
- Splits and cross‑validation use stratification and fixed seeds.

For contribution standards and PR expectations, see AGENTS.md.

