"""Logistic <-> NRM conversion (paper eq. 4) and the Difficulty -> model-scale transform."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from llm_icc.data.parameters import LogisticParameters, NRMParameters


def logistic_to_nrm(params: LogisticParameters, n_options: int, correct_index: int) -> NRMParameters:
    """Correct option gets (a, b - log(n-1)/a); distractors have a_i = b_i = 0."""
    if n_options < 2:
        raise ValueError("Need at least 2 options")
    a = np.zeros(n_options)
    b = np.zeros(n_options)
    a[correct_index] = params.a
    b[correct_index] = params.b - math.log(n_options - 1) / params.a
    return NRMParameters(a=a, b=b)


def nrm_to_logistic_uniform(nrm: NRMParameters, correct_index: int) -> LogisticParameters:
    """Inverse of `logistic_to_nrm` (assumes uniform distractors)."""
    n = len(nrm.a)
    a = float(nrm.a[correct_index])
    b = float(nrm.b[correct_index]) + math.log(n - 1) / a
    return LogisticParameters(b=b, a=a, model="1PL" if a == 1.0 else "2PL")


@dataclass
class DifficultyTransform:
    """Provisional map between reference Difficulty d and model-scale b.

    standardize: b = (d - shift) / scale, with shift/scale fitted as mean/SD of training d.
    linear:      b = scale * (d - shift), with shift/scale taken from config.
    Higher d means harder, matching higher b.
    """
    method: Literal["standardize", "linear"] = "standardize"
    shift: float = 0.0
    scale: float = 1.0

    def fit(self, train_difficulty: Iterable[float]) -> "DifficultyTransform":
        if self.method == "linear":
            return DifficultyTransform("linear", self.shift, self.scale)
        d = np.asarray(list(train_difficulty), dtype=float)
        sd = float(d.std())
        if sd == 0:
            raise ValueError("Cannot standardize: training difficulties have zero variance")
        return DifficultyTransform("standardize", float(d.mean()), sd)

    def to_b(self, d):
        d = np.asarray(d, dtype=float)
        return (d - self.shift) / self.scale if self.method == "standardize" else self.scale * (d - self.shift)

    def to_difficulty(self, b):
        b = np.asarray(b, dtype=float)
        return b * self.scale + self.shift if self.method == "standardize" else b / self.scale + self.shift
