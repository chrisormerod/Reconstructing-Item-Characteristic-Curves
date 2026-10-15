"""Aggregate test metrics across runs/folds.

Usage: python scripts/summarize_runs.py [results_dir]   (default: results/)
Reads every <run>/<timestamp>/metrics.json written by main.py and prints one row per run.
"""
import json
import sys
from pathlib import Path

import pandas as pd


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    rows = []
    for f in sorted(root.glob("*/*/metrics.json")):
        data = json.loads(f.read_text())
        tests = [m["test"] for m in data.values()]
        df = pd.DataFrame(tests)
        row = {"run": f.parent.parent.name, "timestamp": f.parent.name, "n_splits": len(tests)}
        for c in ("pearson", "rmse", "rmse_train_mean_baseline", "icc_mse"):
            row[c] = df[c].mean()
            if len(tests) > 1:
                row[c + "_sd"] = df[c].std()
        rows.append(row)
    if not rows:
        print(f"no metrics.json found under {root}")
        return
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
