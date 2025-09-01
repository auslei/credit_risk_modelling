from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve


def compute_metrics(y_true, y_prob) -> Dict[str, float]:
    auroc = roc_auc_score(y_true, y_prob)
    gini = 2 * auroc - 1

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    aupr = auc(recall, precision)
    return {"auroc": float(auroc), "gini": float(gini), "aupr": float(aupr)}

