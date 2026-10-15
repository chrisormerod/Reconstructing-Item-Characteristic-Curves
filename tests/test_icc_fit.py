import numpy as np
import pytest

from llm_icc.data.item import Item
from llm_icc.data.options import Option
from llm_icc.data.parameters import LogisticParameters
from llm_icc.irt.functions import logistic_2pl
from llm_icc.irt.icc_fit import ICCFitter
from llm_icc.irt.recalibration import LinearRecalibrator
from llm_icc.config import load_config

S = load_config("config/default.yaml").scale()
TH, OM = S.conditional_means(0, 1), S.weights(0, 1)


@pytest.mark.parametrize("b", [-2.5, -0.7, 0.0, 1.1, 3.0])
def test_recovers_1pl_b_noiseless(b):
    fit = ICCFitter().fit(logistic_2pl(TH, 1, b), TH, OM, "1PL")
    assert fit.b == pytest.approx(b, abs=1e-4) and fit.a == 1.0


def test_recovers_2pl():
    fit = ICCFitter().fit(logistic_2pl(TH, 1.6, 0.4), TH, OM, "2PL")
    assert fit.a == pytest.approx(1.6, abs=1e-3) and fit.b == pytest.approx(0.4, abs=1e-3)


def test_recovers_b_from_nrm_targets():
    item = Item("x", "s", [Option(k, k, k == "C") for k in "ABCDE"], LogisticParameters(b=0.6))
    P = item.target_distribution(S, 0, 1)
    fit = ICCFitter().fit(P[:, item.correct_index], TH, OM)
    assert fit.b == pytest.approx(0.6, abs=1e-4)


def test_recalibrator_fixes_shrinkage():
    true = np.linspace(0.1, 1.3, 30)
    pred = 0.4 * (true - 0.7)  # shrunken, shifted
    rc = LinearRecalibrator().fit(pred, true)
    np.testing.assert_allclose(rc.transform(pred), true, atol=1e-9)
