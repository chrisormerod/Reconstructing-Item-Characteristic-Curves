"""Features baseline on synthetic items whose difficulty depends on stem length."""
import numpy as np
import pytest

from llm_icc.app import build_targets
from llm_icc.config import load_config
from llm_icc.data.item import Item
from llm_icc.data.item_bank import ItemBank
from llm_icc.data.options import Option
from llm_icc.modeling.features_baseline import FeaturesModel, feature_names, item_text, numeric_features

VOCAB = [f"term{i}" for i in range(30)]


def _bank(n=60):
    rng = np.random.default_rng(0)
    items = []
    for i in range(n):
        n_words = int(rng.integers(5, 60))
        stem = ". ".join(" ".join(rng.choice(VOCAB, 8)) for _ in range(max(n_words // 8, 1)))
        opts = [Option("ABCD"[j], " ".join(rng.choice(VOCAB, int(rng.integers(1, 6)))), j == 1) for j in range(4)]
        items.append(Item(str(i), stem, opts, reference_difficulty=0.3 + 0.015 * n_words + float(rng.normal(0, 0.03)),
                          metadata={"source": "train", "item_type": "Text", "exam": "S1"}))
    return ItemBank(items)


def _cfg(*extra):
    return load_config("config/usmle_features.yaml", ["features.tfidf_min_df=1", *extra])


def test_textstat_and_option_features_line_up_with_names():
    item = _bank(1)[ "0"]
    assert len(numeric_features(item)) == len(feature_names())
    assert np.isfinite(numeric_features(item)).all()


def test_empty_stem_does_not_break_features():
    item = Item("e", "", [Option("A", "x", True), Option("B", "y", False)])
    assert np.isfinite(numeric_features(item)).all()


def test_correct_answer_is_optional_in_the_text():
    item = _bank(1)["0"]
    assert "Correct answer: B" in item_text(item) and "Correct answer" not in item_text(item, False)


def test_fit_predicts_held_out_items_better_than_the_mean():
    cfg, bank = _cfg(), _bank()
    build_targets(bank, cfg)
    ids = [str(i) for i in range(60)]
    train, test = [bank[i] for i in ids[:45]], [bank[i] for i in ids[45:]]
    model = FeaturesModel(cfg).fit(train)
    pred, true = model.predict_b(test), np.array([i.parameters.b for i in test])
    base = np.full(len(test), np.mean([i.parameters.b for i in train]))
    assert pred.shape == (15,) and np.sqrt(np.mean((pred - true) ** 2)) < np.sqrt(np.mean((base - true) ** 2))


def test_unknown_run_method_is_rejected():
    with pytest.raises(Exception):
        load_config("config/default.yaml", ["run.method=nope"])
