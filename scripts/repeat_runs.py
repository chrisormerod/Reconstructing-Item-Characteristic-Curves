"""Repeat one experiment over several seeds and summarize the spread of the test metrics.

Usage: python scripts/repeat_runs.py CONFIG [--seeds 1 2 3 4 5] [--results results] [--aggregate-only] [-- extra main.py args]

Each seed runs `main.py --config CONFIG --set run.seed=S --set run.name=<name>_seed<S>`. The seed changes the
train/dev split (predefined strategy draws the dev items at random) and, for neural models, initialisation and
batch order; the test set is fixed. The per-seed bootstrap CIs describe test-item sampling only, whereas the SD
across seeds describes training/split variability. Prints and writes <results>/<name>_repeats.csv.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


def run_name(config: Path) -> str:
    cfg = yaml.safe_load(config.read_text())
    if "base" in cfg and "name" not in cfg.get("run", {}):
        return run_name(config.parent / cfg["base"])
    return cfg["run"]["name"]


def latest_metrics(results: Path, name: str) -> dict | None:
    runs = sorted((results / name).glob("*/metrics.json"))
    return json.loads(runs[-1].read_text()) if runs else None


def summarize(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    cols = [c for c in ("pearson", "rmse", "rmse_ci_lo", "rmse_ci_hi", "bias", "rmse_train_mean_baseline") if c in df]
    agg = df[cols].agg(["mean", "std", "min", "max"]).T
    agg.index.name = "metric"
    return agg


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=Path)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--aggregate-only", action="store_true", help="skip running; summarize existing seed runs")
    p.add_argument("extra", nargs=argparse.REMAINDER, help="passed to main.py after `--`")
    a = p.parse_args()
    name = run_name(a.config)
    extra = [x for x in a.extra if x != "--"]

    rows = []
    for s in a.seeds:
        sname = f"{name}_seed{s}"
        if not a.aggregate_only:
            cmd = [sys.executable, "main.py", "--config", str(a.config), "--set", f"run.seed={s}",
                   "--set", f"run.name={sname}", *extra]
            print("==>", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
        m = latest_metrics(a.results, sname)
        if m is None:
            print(f"no metrics for seed {s} ({sname})")
            continue
        t = next(iter(m.values()))["test"]  # predefined strategy: one split
        row = {"seed": s, **{k: t[k] for k in ("pearson", "rmse", "bias", "rmse_train_mean_baseline") if k in t}}
        if "rmse_ci" in t:
            row["rmse_ci_lo"], row["rmse_ci_hi"] = t["rmse_ci"]
        rows.append(row)
    if not rows:
        raise SystemExit("nothing to summarize")
    per_seed, agg = pd.DataFrame(rows), summarize(rows)
    fmt = lambda x: f"{x:.4f}"  # noqa: E731
    print(f"\n{name}: {len(rows)} seeds\n{per_seed.to_string(index=False, float_format=fmt)}\n\n{agg.to_string(float_format=fmt)}")
    a.results.mkdir(exist_ok=True)
    pd.concat([per_seed.assign(kind="seed").astype({"seed": str}),
               agg.reset_index().rename(columns={"metric": "seed"}).assign(kind="summary")]).to_csv(
        a.results / f"{name}_repeats.csv", index=False)


if __name__ == "__main__":
    main()
