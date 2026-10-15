import math

import numpy as np
import pytest
import yaml
from scipy import integrate
from scipy.stats import norm

from llm_icc.config import load_config
from llm_icc.data.descriptors import AbilityDescriptor, DescriptorScale

CFG = "config/default.yaml"


def test_default_scale_is_valid_and_ordered():
    s = load_config(CFG).scale()
    assert len(s) == 20 and s.labels[0] == "Critical" and s.labels[-1] == "Exemplary"


def test_means_and_weights_match_numerical_integration():
    s = load_config(CFG).scale()
    mu, sd = 0.3, 1.2
    th, w = s.conditional_means(mu, sd), s.weights(mu, sd)
    assert w.sum() == pytest.approx(1.0)
    for d, t, om in zip(s.descriptors, th, w):
        lo, hi = max(d.lower, -40), min(d.upper, 40)
        mass, _ = integrate.quad(lambda x: norm.pdf(x, mu, sd), lo, hi)
        first, _ = integrate.quad(lambda x: x * norm.pdf(x, mu, sd), lo, hi)
        assert om == pytest.approx(mass, abs=1e-9)
        assert t == pytest.approx(first / mass, abs=1e-7)
        assert d.lower < t <= d.upper


def test_assign_uses_half_open_bounds():
    s = load_config(CFG).scale()
    assert s.label_for(-3.0) == 0          # (-inf, -3.0]
    assert s.label_for(-2.9999) == 1
    assert list(s.assign(np.array([0.0, 0.0001, 5.0]))) == [9, 10, 19]


def _mk(*bounds):
    return [AbilityDescriptor(f"d{i}", lo, hi) for i, (lo, hi) in enumerate(bounds)]


def test_validator_rejects_gap_and_overlap():
    inf = math.inf
    with pytest.raises(ValueError, match="Gap"):
        DescriptorScale(_mk((-inf, -1), (-0.5, inf)))
    with pytest.raises(ValueError, match="Overlap"):
        DescriptorScale(_mk((-inf, 0), (-0.5, inf)))
    with pytest.raises(ValueError, match="-inf"):
        DescriptorScale(_mk((-5, 0), (0, inf)))
    with pytest.raises(ValueError, match="\\+inf"):
        DescriptorScale(_mk((-inf, 0), (0, 5)))


def test_config_rejects_gap(tmp_path):
    raw = yaml.safe_load(open(CFG))
    raw["descriptors"]["Emerging"] = [-1.5, -1.2]  # the paper's Table 2 gap
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ValueError, match="Gap"):
        load_config(p)


def test_override_and_base_inheritance():
    c = load_config("config/usmle_qwen3_8b.yaml", ["training.sft.enabled=false", "population.sigma=1.5"])
    assert c.pretrained_model.name_or_path == "Qwen/Qwen3-8B"
    assert c.training.sft.enabled is False and c.population.sigma == 1.5
