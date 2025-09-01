from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def load_raw_csv(path: str, index_col: Optional[int] = 0) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, index_col=index_col)


def save_pickle(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)


def load_pickle(path: str) -> pd.DataFrame:
    return pd.read_pickle(path)

