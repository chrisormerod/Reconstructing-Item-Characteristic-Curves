"""Dev-set linear recalibration of predicted b onto the reference Difficulty scale."""
from __future__ import annotations

import numpy as np


class LinearRecalibrator:
    """Regress reference difficulty on predicted b; also corrects regression to the mean."""

    def __init__(self):
        self.slope: float | None = None
        self.intercept: float | None = None

    def fit(self, pred_dev, true_dev) -> "LinearRecalibrator":
        x, y = np.asarray(pred_dev, float), np.asarray(true_dev, float)
        if len(x) < 2 or np.ptp(x) == 0:
            raise ValueError("Need >= 2 dev items with varying predictions to recalibrate")
        self.slope, self.intercept = (float(v) for v in np.polyfit(x, y, 1))
        return self

    def transform(self, pred) -> np.ndarray:
        if self.slope is None:
            raise RuntimeError("LinearRecalibrator is not fitted")
        return self.slope * np.asarray(pred, float) + self.intercept
