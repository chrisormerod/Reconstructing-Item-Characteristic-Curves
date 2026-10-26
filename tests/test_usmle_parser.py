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


def _renamed(synth, tmp_path, ext, with_gold):
    """Rewrite the synthetic files with other names/columns/format; returns (files, columns) config."""
    from llm_icc.config.schema import DataColumnsConfig, DataFilesConfig
    ren = {"ItemNum": "id", "ItemStem_Text": "question", "Answer_Key": "key", "Answer_Text": "key_text",
           "Difficulty": "p_hard", **{f"Answer__{k}": f"opt_{k}" for k in "ABCDEFGHIJ"}}
    write = (lambda d, n: d.to_csv(tmp_path / f"{n}.csv", index=False)) if ext == "csv" else \
        (lambda d, n: d.to_parquet(tmp_path / f"{n}.parquet"))
    train = pd.read_excel(synth / "train_final.xlsx").rename(columns=ren)
    test = pd.read_excel(synth / "test_final.xlsx").rename(columns=ren)
    gold = pd.read_excel(synth / "gold_final.xlsx").rename(columns=ren)
    write(train, "tr"), write(test if with_gold else test.merge(gold, on="id"), "te")
    if with_gold:
        write(gold, "go")
    files = DataFilesConfig(train=f"tr.{ext}", test=f"te.{ext}", gold=f"go.{ext}" if with_gold else None)
    cols = DataColumnsConfig(id="id", stem="question", option_prefix="opt_", key="key", key_text="key_text",
                             item_type=None, exam=None, difficulty="p_hard")
    return files, cols


@pytest.mark.parametrize("ext,with_gold", [("csv", True), ("csv", False), ("parquet", True)])
def test_custom_files_columns_and_formats_match_the_default_parse(synth, tmp_path, ext, with_gold):
    files, cols = _renamed(synth, tmp_path, ext, with_gold)
    ref, got = USMLEParser(synth).load(), USMLEParser(tmp_path, files, cols).load()
    assert [i.item_id for i in got] == [i.item_id for i in ref]
    for a, b in zip(ref, got):
        assert (a.stem, [(o.key, o.text, o.is_correct) for o in a.options]) == \
               (b.stem, [(o.key, o.text, o.is_correct) for o in b.options])
        assert b.reference_difficulty == pytest.approx(a.reference_difficulty)
    assert {i.metadata["item_type"] for i in got} == {""}  # optional columns switched off


def test_missing_column_and_extension_errors(synth, tmp_path):
    from llm_icc.config.schema import DataColumnsConfig, DataFilesConfig
    with pytest.raises(ValueError, match="missing columns"):
        USMLEParser(synth, columns=DataColumnsConfig(stem="nope")).load()
    (tmp_path / "x.txt").write_text("a")
    with pytest.raises(ValueError, match="unsupported extension"):
        USMLEParser(tmp_path, DataFilesConfig(train="x.txt")).load()


def test_usmle_dir_is_a_deprecated_alias_for_dir():
    from llm_icc.config import load_config
    c = load_config("config/default.yaml", ["data.usmle_dir=/somewhere"])
    assert c.data.dir == "/somewhere" and c.data.files.gold == "gold_final.xlsx"
