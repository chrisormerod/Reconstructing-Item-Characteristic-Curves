"""Recover 1PL/2PL parameters from a discrete ICC (paper eq. 11)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from llm_icc.data.parameters import LogisticParameters
from llm_icc.irt.functions import logistic_2pl


class ICCFitter:
    B_RANGE = (-10.0, 10.0)

    def fit(self, p_correct: np.ndarray, theta_k: np.ndarray, omega_k: np.ndarray,
            model: str = "1PL") -> LogisticParameters:
        p = np.asarray(p_correct, float)
        th = np.asarray(theta_k, float)
        w = np.asarray(omega_k, float)

        def loss1(b):
            return float(np.sum(w * (p - logistic_2pl(th, 1.0, b)) ** 2))

        # coarse grid first: the objective can be flat in the tails
        grid = np.linspace(*self.B_RANGE, 401)
        b0 = grid[int(np.argmin([loss1(b) for b in grid]))]
        step = grid[1] - grid[0]
        res = minimize_scalar(loss1, bounds=(b0 - step, b0 + step), method="bounded",
                              options={"xatol": 1e-8})
        b1 = float(res.x)
        if model == "1PL":
            return LogisticParameters(b=b1, a=1.0, model="1PL")
        if model == "2PL":
            sw = np.sqrt(w)
            r = least_squares(lambda x: sw * (p - logistic_2pl(th, x[0], x[1])),
                              x0=[1.0, b1], bounds=([1e-3, self.B_RANGE[0]], [20.0, self.B_RANGE[1]]))
            return LogisticParameters(b=float(r.x[1]), a=float(r.x[0]), model="2PL")
        raise ValueError(f"Unknown model {model!r}")
