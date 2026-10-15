"""Write synthetic spreadsheets in the USMLE format with known Difficulty, for end-to-end tests.

Responses are not simulated: only item content and a known Difficulty are produced.
Usage: python scripts/simulate_usmle.py OUT_DIR [--n-train 60] [--n-test 30] [--seed 0]
"""
from __future__ import annotations

import argparse
import string
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = list(string.ascii_uppercase[:10])


def make_rows(n: int, start: int, rng, with_difficulty: bool):
    rows = []
    for i in range(n):
        k = int(rng.choice([4, 5, 6, 8, 10], p=[0.1, 0.7, 0.1, 0.07, 0.03]))
        key = KEYS[int(rng.integers(k))]
        row = {"ItemNum": start + i, "ItemStem_Text": f"Synthetic stem {start + i}: a patient presents with finding {rng.integers(1000)}."}
        for j, L in enumerate(KEYS):
            row[f"Answer__{L}"] = f"Option {L} for item {start + i}" if j < k else None
        row["Answer_Key"] = key
        row["Answer_Text"] = f"Option {key} for item {start + i}"
        row["ItemType"] = str(rng.choice(["Text", "PIX"], p=[0.9, 0.1]))
        row["EXAM"] = str(rng.choice(["STEP 1", "STEP 2", "STEP 3"]))
        if with_difficulty:
            row["Difficulty"] = float(np.clip(rng.normal(0.5, 0.25), 0.02, 1.4))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    p.add_argument("--n-train", type=int, default=60)
    p.add_argument("--n-test", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train = make_rows(a.n_train, 1, rng, True)
    test = make_rows(a.n_test, 100000, rng, False)
    gold = pd.DataFrame({"ItemNum": test["ItemNum"], "Difficulty": np.clip(rng.normal(0.5, 0.25, len(test)), 0.1, 1.4)})
    train["Response_Time"] = 60.0
    gold["Response_Time"] = 60.0
    train.to_excel(out / "train_final.xlsx", index=False)
    test.to_excel(out / "test_final.xlsx", index=False)
    gold.to_excel(out / "gold_final.xlsx", index=False)
    print(f"wrote {len(train)} train / {len(test)} test items to {out}")


if __name__ == "__main__":
    main()
