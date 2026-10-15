from __future__ import annotations

import numpy as np


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rmse(x, y) -> float:
    return float(np.sqrt(np.mean((np.asarray(x, float) - np.asarray(y, float)) ** 2)))


def weighted_mse(pred, target, weights) -> float:
    """pred/target (..., N, n) or (N, n); weights (N,). omega-weighted sum over options, per item mean."""
    pred, target = np.asarray(pred, float), np.asarray(target, float)
    w = np.asarray(weights, float)
    sq = ((pred - target) ** 2).sum(-1)  # (..., N)
    return float((sq * w).sum(-1).mean() / w.sum())
