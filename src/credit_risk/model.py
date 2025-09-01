from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from .config import CFG
from .features import build_preprocessor
from .preprocessing import initial_clean, make_label_columns, split_types


def build_pipeline(cat_cols, num_cols) -> Pipeline:
    pre = build_preprocessor(cat_cols, num_cols)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=CFG.random_state)
    return Pipeline(steps=[("preprocess", pre), ("clf", clf)])


def train_eval(df, target_col: str = CFG.target, n_splits: int = 5, n_repeats: int = 3) -> Tuple[Pipeline, float]:
    df = make_label_columns(df)
    df = initial_clean(df)
    cat_cols, num_cols = split_types(df.drop(columns=[target_col], errors="ignore"))

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    pipe = build_pipeline(cat_cols, num_cols)

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=CFG.random_state)
    scores = cross_val_score(pipe, X, y, scoring="roc_auc", cv=cv, n_jobs=None)
    auroc = float(np.mean(scores))

    pipe.fit(X, y)
    return pipe, auroc

