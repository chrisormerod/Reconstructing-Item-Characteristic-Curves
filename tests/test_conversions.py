import numpy as np
import pytest

from llm_icc.data.parameters import LogisticParameters
from llm_icc.irt.conversions import DifficultyTransform, logistic_to_nrm, nrm_to_logistic_uniform
from llm_icc.irt.functions import logistic_2pl

THETA = np.linspace(-4, 4, 17)


@pytest.mark.parametrize("n", range(4, 11))
@pytest.mark.parametrize("a,b", [(1.0, 0.0), (1.0, -1.3), (1.7, 0.8)])
def test_roundtrip_p_correct(n, a, b):
    for c in (0, n - 1):
        nrm = logistic_to_nrm(LogisticParameters(b=b, a=a), n, c)
        P = nrm.option_probabilities(THETA)
        assert P.shape == (len(THETA), n)
        np.testing.assert_allclose(P.sum(1), 1.0)
        np.testing.assert_allclose(P[:, c], logistic_2pl(THETA, a, b), atol=1e-12)
        # distractors are equiprobable
        d = np.delete(P, c, axis=1)
        np.testing.assert_allclose(d, d[:, :1].repeat(n - 1, 1), atol=1e-12)
        back = nrm_to_logistic_uniform(nrm, c)
        assert back.b == pytest.approx(b) and back.a == pytest.approx(a)


def test_standardize_transform_roundtrip():
    d = np.array([0.1, 0.5, 0.9, 1.3])
    t = DifficultyTransform("standardize").fit(d)
    b = t.to_b(d)
    assert b.mean() == pytest.approx(0) and b.std() == pytest.approx(1)
    np.testing.assert_allclose(t.to_difficulty(b), d)


def test_linear_transform_keeps_direction():
    t = DifficultyTransform("linear", shift=0.5, scale=2.0).fit([])
    assert t.to_b(1.0) == pytest.approx(1.0) and t.to_difficulty(1.0) == pytest.approx(1.0)
    assert t.to_b(0.9) > t.to_b(0.1)  # harder -> larger b
