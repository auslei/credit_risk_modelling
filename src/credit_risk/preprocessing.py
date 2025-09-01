from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import CFG


def identify_high_null_columns(df: pd.DataFrame, threshold: float) -> List[str]:
    return df.isnull().mean().sort_values(ascending=False)[lambda s: s > threshold].index.tolist()


def identify_single_value_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if df[c].nunique(dropna=False) == 1]


def process_emp_length(x: object) -> int:
    if pd.isnull(x):
        return 0
    m = re.search(r"\d+", str(x))
    return int(m.group()) if m else 0


def map_pymnt_plan(x: object) -> int:
    if pd.isnull(x):
        return 0
    s = str(x).strip().lower()
    return 0 if s == "n" else 1


def months_from_today(date_str: object, today: pd.Timestamp) -> float:
    try:
        # Input format like 'Jan-12'; normalize to YYYY-MM-01
        ds = str(date_str)
        date_obj = pd.to_datetime(ds + "-01", format="%b-%y-%d", errors="raise")
    except Exception:
        return np.nan
    return (today - date_obj).days // 30


def transform_dates(df: pd.DataFrame, date_columns: Sequence[str], today_str: str) -> pd.DataFrame:
    out = df.copy()
    today = pd.to_datetime(today_str)
    for col in date_columns:
        if col in out.columns:
            out[col] = out[col].map(lambda v: months_from_today(v, today))
    return out


def make_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    default_loan_status = {
        "Charged Off",
        "Default",
        "Does not meet the credit policy. Status:Charged Off",
        "Late (31-120 days)",
    }
    early_dlq = {"In Grace Period", "Late (16-30 days)"}
    if "loan_status" in out.columns:
        out[CFG.target] = out["loan_status"].map(lambda x: 1 if x in default_loan_status else 0)
        out["early_dlq"] = out["loan_status"].map(lambda x: 1 if x in early_dlq else 0)
        out.drop(columns=["loan_status", "early_dlq"], inplace=True, errors="ignore")
    return out


def drop_known_noise(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    to_drop: List[str] = [
        *CFG.free_text_columns,
        *CFG.future_events,
        *CFG.location_info,
        *CFG.identity_columns,
    ]
    out.drop(columns=[c for c in to_drop if c in out.columns], inplace=True, errors="ignore")
    return out


def split_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    if CFG.target in num_cols:
        num_cols.remove(CFG.target)
    return cat_cols, num_cols


def initial_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal deterministic cleaning aligned with the notebook decisions."""
    out = df.copy()
    # High-null and single-value columns
    high_null = identify_high_null_columns(out, CFG.high_null_threshold)
    single_val = identify_single_value_columns(out)
    out.drop(columns=list(set(high_null + single_val)), inplace=True, errors="ignore")

    # Domain-specific mappings
    if "emp_length" in out.columns:
        out["emp_length"] = out["emp_length"].map(process_emp_length)
    if "pymnt_plan" in out.columns:
        out["pymnt_plan"] = out["pymnt_plan"].map(map_pymnt_plan)

    # Dates to months-from-today
    date_columns = [c for c in ["earliest_cr_line", "issue_d", "last_pymnt_d", "last_credit_pull_d"] if c in out.columns]
    out = transform_dates(out, date_columns, CFG.today_str)

    # Known noise/leakage
    out = drop_known_noise(out)
    return out

