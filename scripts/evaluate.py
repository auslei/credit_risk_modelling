from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from credit_risk.config import CFG
from credit_risk.data_io import load_raw_csv
from credit_risk.evaluate import compute_metrics
from credit_risk.preprocessing import initial_clean, make_label_columns


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate saved credit risk model")
    p.add_argument("--input", type=str, default=CFG.data_raw_csv, help="Path to raw CSV")
    p.add_argument("--model", type=str, required=True, help="Path to saved joblib model")
    p.add_argument("--out", type=str, default="artifacts/metrics.json", help="Where to save metrics JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = load_raw_csv(args.input)
    df = make_label_columns(df)
    df = initial_clean(df)

    if CFG.target not in df.columns:
        raise RuntimeError(f"Target column '{CFG.target}' not present after preprocessing.")

    X = df.drop(columns=[CFG.target])
    y = df[CFG.target].astype(int).to_numpy()

    payload = joblib.load(args.model)
    model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload

    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Loaded model does not support predict_proba.")

    y_prob = model.predict_proba(X)[:, 1]
    metrics = compute_metrics(y, y_prob)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()

