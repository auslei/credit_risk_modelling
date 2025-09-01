from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Config:
    # Data
    data_raw_csv: str = "data/loan_data_2007_2014.csv"
    target: str = "label"

    # Reproducibility
    random_state: int = 42

    # Cleaning thresholds
    high_null_threshold: float = 0.80
    corr_threshold: float = 0.70
    k_numeric: int = 20

    # Date parsing baseline (align with notebook default)
    today_str: str = "2020-01-01"

    # Columns to drop to avoid leakage or noise
    free_text_columns: tuple[str, ...] = ("url", "desc", "title", "emp_title")
    future_events: tuple[str, ...] = (
        "recoveries",
        "next_pymnt_d",
        "collection_recovery_fee",
        "collections_12_mths_ex_med",
    )
    location_info: tuple[str, ...] = ("zip_code", "addr_state")
    identity_columns: tuple[str, ...] = ("id", "member_id")

    @property
    def today(self) -> datetime:
        return datetime.fromisoformat(self.today_str)


CFG = Config()

