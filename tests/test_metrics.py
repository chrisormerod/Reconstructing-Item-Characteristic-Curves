import numpy as np
import pytest

from llm_icc.utils import metrics as M


def test_bias_is_mean_signed_error():
    assert M.bias([1.5, 2.5], [1.0, 2.0]) == pytest.approx(0.5)
    assert M.bias([0.0, 0.0], [1.0, 3.0]) == pytest.approx(-2.0)


def test_constant_offset_has_degenerate_ci():
    ref = np.linspace(0, 1, 50)
    assert M.bootstrap_ci(ref + 0.2, ref, "bias", 500) == pytest.approx((0.2, 0.2))
    assert M.bootstrap_ci(ref + 0.2, ref, "rmse", 500) == pytest.approx((0.2, 0.2))


def test_ci_brackets_point_estimate_and_is_seeded():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=201)
    pred = ref + rng.normal(scale=0.3, size=201) + 0.1
    for stat, point in (("rmse", M.rmse(pred, ref)), ("bias", M.bias(pred, ref))):
        lo, hi = M.bootstrap_ci(pred, ref, stat, 2000, seed=3)
        assert lo < point < hi
        assert (lo, hi) == M.bootstrap_ci(pred, ref, stat, 2000, seed=3)


def test_ci_width_follows_level():
    rng = np.random.default_rng(1)
    ref, pred = rng.normal(size=100), rng.normal(size=100)
    w = lambda lv: np.subtract(*M.bootstrap_ci(pred, ref, "rmse", 2000, lv, 0)[::-1])  # noqa: E731
    assert w(0.99) > w(0.95) > w(0.50)


def test_too_few_items_is_nan():
    assert np.isnan(M.bootstrap_ci([1.0], [1.0])).all()
