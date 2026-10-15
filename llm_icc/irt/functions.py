"""Basic IRT response functions (paper eqs. 1 and 3)."""
from __future__ import annotations

import numpy as np
from scipy.special import expit, softmax


def sigmoid(x):
    return expit(x)


def logistic_2pl(theta, a, b):
    """P(correct | theta) = sigmoid(a (theta - b))."""
    return expit(np.asarray(a) * (np.asarray(theta) - np.asarray(b)))


def nrm_probabilities(theta: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Nominal response model. theta (K,), a (n,), b (n,) -> (K, n) softmax of a_i (theta - b_i)."""
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    logits = np.asarray(a)[None, :] * (theta[:, None] - np.asarray(b)[None, :])
    return softmax(logits, axis=1)
