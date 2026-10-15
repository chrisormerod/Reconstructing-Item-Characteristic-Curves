import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from llm_icc.utils.usmle import USMLEParser, describe_bank

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("usmle")
    subprocess.run([sys.executable, str(ROOT / "scripts/simulate_usmle.py"), str(d), "--n-train", "80",
                    "--n-test", "40"], check=True, capture_output=True)
    return d


def test_variable_options_and_gold_join(synth):
    bank = USMLEParser(synth).load()
    assert len(bank.ids("train")) == 80 and len(bank.ids("test")) == 40
    counts = {i.n_options for i in bank}
    assert len(counts) > 1 and all(4 <= c <= 10 for c in counts)
    gold = pd.read_excel(synth / "gold_final.xlsx").set_index("ItemNum")["Difficulty"]
    for iid in bank.ids("test"):
        assert bank[iid].reference_difficulty == pytest.approx(gold[int(iid)])
    for i in bank:
        assert [o.key for o in i.options] == list("ABCDEFGHIJ")[: i.n_options]
        assert sum(o.is_correct for o in i.options) == 1
    assert "n_items" in describe_bank(bank)


def test_gold_id_mismatch_raises(synth, tmp_path):
    for f in ("train_final.xlsx", "test_final.xlsx"):
        (tmp_path / f).write_bytes((synth / f).read_bytes())
    g = pd.read_excel(synth / "gold_final.xlsx")
    g.iloc[0, 0] = 999999
    g.to_excel(tmp_path / "gold_final.xlsx", index=False)
    with pytest.raises(ValueError, match="differ"):
        USMLEParser(tmp_path).load()


def test_answer_key_pointing_to_empty_option_raises(synth, tmp_path):
    t = pd.read_excel(synth / "train_final.xlsx")
    row = t["Answer__J"].isna().idxmax()
    t.loc[row, "Answer_Key"] = "J"
    t.to_excel(tmp_path / "train_final.xlsx", index=False)
    for f in ("test_final.xlsx", "gold_final.xlsx"):
        (tmp_path / f).write_bytes((synth / f).read_bytes())
    with pytest.raises(ValueError, match="Answer_Key"):
        USMLEParser(tmp_path).load()


def test_splits(synth):
    bank = USMLEParser(synth).load()
    (s,) = bank.split("predefined", seed=1, dev_fraction=0.1)
    assert set(s.train).isdisjoint(s.dev) and len(s.dev) == 8 and s.test == bank.ids("test")
    folds = bank.split("kfold", seed=1, k=5)
    assert sorted(x for f in folds for x in f.dev) == sorted(bank.ids("train"))
    assert all(f.test == bank.ids("test") and set(f.train).isdisjoint(f.dev) for f in folds)
