"""Item parameter containers: logistic (1PL/2PL) and nominal response model."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np

from llm_icc.irt.functions import logistic_2pl, nrm_probabilities


class ItemParameters(ABC):
    @abstractmethod
    def option_probabilities(self, theta: np.ndarray) -> np.ndarray:
        """theta: (K,) -> (K, n) probabilities over options."""

    @abstractmethod
    def to_dict(self) -> dict: ...


@dataclass
class NRMParameters(ItemParameters):
    a: np.ndarray  # (n,)
    b: np.ndarray  # (n,)

    def logits(self, theta: np.ndarray) -> np.ndarray:
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        return self.a[None, :] * (theta[:, None] - self.b[None, :])

    def option_probabilities(self, theta: np.ndarray) -> np.ndarray:
        return nrm_probabilities(theta, self.a, self.b)

    def to_dict(self) -> dict:
        return {"a": self.a.tolist(), "b": self.b.tolist()}


@dataclass
class LogisticParameters(ItemParameters):
    b: float
    a: float = 1.0
    model: Literal["1PL", "2PL"] = "1PL"

    def p_correct(self, theta: np.ndarray) -> np.ndarray:
        return logistic_2pl(theta, self.a, self.b)

    def to_nrm(self, n_options: int, correct_index: int) -> NRMParameters:
        """Uniform-distractor NRM equivalent (paper eq. 4)."""
        from llm_icc.irt.conversions import logistic_to_nrm
        return logistic_to_nrm(self, n_options, correct_index)

    def option_probabilities(self, theta: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Option probabilities need the option count; use to_nrm(n, correct_index).option_probabilities")

    def to_dict(self) -> dict:
        return {"model": self.model, "a": self.a, "b": self.b}
