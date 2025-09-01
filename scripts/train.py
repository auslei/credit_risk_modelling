from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from credit_risk.config import CFG
from credit_risk.data_io import load_raw_csv
from credit_risk.model import train_eval


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train credit risk model")
    p.add_argument("--input", type=str, default=CFG.data_raw_csv, help="Path to raw CSV")
    p.add_argument("--out", type=str, default="artifacts/model.joblib", help="Output model path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load_raw_csv(args.input)
    model, auroc = train_eval(df)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "auroc_cv": auroc}, args.out)
    print(f"Saved model to {args.out} | CV AUROC={auroc:.4f}")


if __name__ == "__main__":
    main()

