"""Run every model of Table 4 over several seeds on one GPU and build the table.

Usage:
  python scripts/run_all.py --preflight            # load each model + one ICC (--dry-run); logs measured GPU memory
  python scripts/run_all.py                        # all models x seeds 1-5; finished (model, seed) pairs are skipped
  python scripts/run_all.py --models qwen3_14b qwen3_32b --seeds 1 2
  python scripts/run_all.py --aggregate-only       # rebuild results/table4.{csv,md,tex} from existing runs
  python scripts/run_all.py --list
  python scripts/run_all.py --set data.dir=/path/to/usmle      # --set is passed to every main.py run

A (model, seed) run is finished when results/<name>_seed<S>/*/metrics.json exists, so an interrupted job resumes
where it stopped (a half-trained run restarts from scratch). A failure is recorded and the sweep continues;
the exit status is non-zero if any run failed. Models run cheapest first so partial tables appear early.

Table 4 columns are summarized over seeds: Pearson and bias as mean (SD), RMSE as mean (SD), and the 95% CI of the
RMSE as the mean of the per-seed bootstrap bounds. The CI covers test-item sampling; the SD covers retraining
(train/dev split, initialisation, batch order). They answer different questions and are not combined.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from repeat_runs import latest_metrics, run_name  # noqa: E402

# key: (Table 4 label, group, config, needs_gpu, approx. peak GPU GiB on the USMLE data, see docs/experiments.md)
MODELS = {
    "features":     ("Features", "Direct-regression baselines", "config/usmle_features.yaml", False, 0),
    "modernbert":   ("ModernBERT", "Direct-regression baselines", "config/usmle_modernbert_regression.yaml", True, 8),
    "roberta":      ("RoBERTa (approach of Maeda, 2023)", "Response simulation", "config/usmle_roberta_large.yaml", True, 10),
    "qwen3_1_7b":   ("Qwen-1.7B", "Response simulation", "config/usmle_qwen3_1_7b.yaml", True, 5),
    "qwen3_4b":     ("Qwen-4B", "Response simulation", "config/usmle_qwen3_4b.yaml", True, 9),
    "llama31_8b":   ("Llama-3.1-8B", "Response simulation", "config/usmle_llama31_8b.yaml", True, 17),
    "qwen3_8b":     ("Qwen-8B", "Response simulation", "config/usmle_qwen3_8b.yaml", True, 17),
    "qwen3_14b":    ("Qwen-14B", "Response simulation", "config/usmle_qwen3_14b.yaml", True, 31),
    "qwen3_32b":    ("Qwen-32B (4-bit NF4)", "Response simulation", "config/usmle_qwen3_32b.yaml", True, 24),
}
EXTRA = {  # not in the default sweep
    "qwen3_14b_nf4": ("Qwen-14B (4-bit NF4)", "Response simulation", "config/usmle_qwen3_14b_nf4.yaml", True, 14),
}
ALL = {**MODELS, **EXTRA}
PUBLISHED = [("ELECTRA (published)", 0.299), ("Ensemble, Li et al. 2025 (published)", 0.292)]
ROOT = Path(__file__).resolve().parent.parent


def run_dir_name(key: str, seed: int) -> str:
    return f"{run_name(ROOT / ALL[key][2])}_seed{seed}"


def stream(cmd: list[str], log: Path | None = None) -> int:
    """Run `cmd` from the repo root, echoing output (and appending to `log`)."""
    with subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as p:
        f = log.open("a") if log else None
        for line in p.stdout:
            sys.stdout.write(line)
            if f:
                f.write(line)
        if f:
            f.close()
    return p.returncode


def set_args(overrides: list[str]) -> list[str]:
    return [x for o in overrides for x in ("--set", o)]


def preflight(keys: list[str], results: Path, overrides: list[str]) -> int:
    """Load each model and run one forward pass (main.py --dry-run); print the measured post-load GPU memory."""
    bad = 0
    for k in keys:
        label, _, cfg, gpu, est = ALL[k]
        print(f"\n== preflight {k}: {label} (expected peak ~{est} GiB)", flush=True)
        tmp = results / "_preflight.log"
        tmp.unlink(missing_ok=True)
        rc = stream([sys.executable, "main.py", "--config", cfg, "--dry-run", "--set", "run.name=_preflight",
                         *set_args(overrides)], tmp)
        mem = re.findall(r"model loaded; (.*)", tmp.read_text()) if tmp.exists() else []
        print(f"-> {'OK' if rc == 0 else 'FAILED'} {mem[-1] if mem else ''}")
        bad += rc != 0
    return bad


def run_sweep(keys: list[str], seeds: list[int], results: Path, force: bool, overrides: list[str]) -> list[tuple[str, int, str]]:
    status, failures = [], []
    for k in keys:
        for s in seeds:
            name = run_dir_name(k, s)
            if not force and latest_metrics(results, name) is not None:
                print(f"skip {name} (finished)")
                continue
            print(f"\n==> {k} seed {s}", flush=True)
            t0 = time.time()
            rc = stream([sys.executable, "main.py", "--config", ALL[k][2], "--set", f"run.seed={s}",
                         "--set", f"run.name={name}", *set_args(overrides)])
            mins = (time.time() - t0) / 60
            status.append({"model": k, "seed": s, "ok": rc == 0, "minutes": round(mins, 1)})
            print(f"<== {k} seed {s}: {'ok' if rc == 0 else f'FAILED (exit {rc})'} in {mins:.1f} min", flush=True)
            if rc != 0:
                failures.append((k, s, f"exit {rc}"))
            results.mkdir(exist_ok=True)
            (results / "run_all_status.json").write_text(json.dumps(status, indent=1))
    return failures


def collect(keys: list[str], seeds: list[int], results: Path) -> pd.DataFrame:
    rows = []
    for k in keys:
        for s in seeds:
            m = latest_metrics(results, run_dir_name(k, s))
            if m is None:
                continue
            t = next(iter(m.values()))["test"]
            lo, hi = t.get("rmse_ci", (np.nan, np.nan))
            blo, bhi = t.get("bias_ci", (np.nan, np.nan))
            blo_r, bhi_r = t.get("rmse_train_mean_baseline_ci", (np.nan, np.nan))
            rows.append({"model": k, "seed": s, "pearson": t["pearson"], "rmse": t["rmse"], "rmse_lo": lo,
                         "rmse_hi": hi, "bias": t.get("bias", np.nan), "bias_lo": blo, "bias_hi": bhi,
                         "baseline_rmse": t["rmse_train_mean_baseline"], "baseline_lo": blo_r, "baseline_hi": bhi_r})
    return pd.DataFrame(rows)


def summarize(runs: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    g = runs.groupby("model")
    out = g.agg(n_seeds=("seed", "count"), **{f"{c}_{f}": (c, f) for c in
                ("pearson", "rmse", "rmse_lo", "rmse_hi", "bias", "baseline_rmse", "baseline_lo", "baseline_hi")
                for f in ("mean", "std")}).reset_index()
    out["label"] = out["model"].map(lambda k: ALL[k][0])
    out["group"] = out["model"].map(lambda k: ALL[k][1])
    return out.set_index("model").loc[[k for k in keys if k in set(out["model"])]].reset_index()


def pm(mean, sd, nd=3, n=1):
    if np.isnan(mean):
        return "–"
    return f"{mean:.{nd}f}" + (f" ({sd:.{nd}f})" if n > 1 and not np.isnan(sd) else "")


def table_rows(summ: pd.DataFrame) -> list[tuple[str, list[str]]]:
    rows = []
    if len(summ):
        r = summ.iloc[0]  # the train-mean baseline is nearly seed-invariant; report the first model's seed means
        rows.append(("Dummy Regressor (train mean)", ["–", pm(r.baseline_rmse_mean, r.baseline_rmse_std, 3, r.n_seeds),
                                                      f"[{r.baseline_lo_mean:.3f}, {r.baseline_hi_mean:.3f}]", "–", str(r.n_seeds)]))
    for lab, v in PUBLISHED:
        rows.append((lab, ["–", f"{v:.3f}", "–", "–", "–"]))
    for _, r in summ.iterrows():
        rows.append((r.label, [pm(r.pearson_mean, r.pearson_std, 3, r.n_seeds), pm(r.rmse_mean, r.rmse_std, 3, r.n_seeds),
                               f"[{r.rmse_lo_mean:.3f}, {r.rmse_hi_mean:.3f}]" if not np.isnan(r.rmse_lo_mean) else "–",
                               pm(r.bias_mean, r.bias_std, 3, r.n_seeds), str(int(r.n_seeds))]))
    return rows


def write_tables(runs: pd.DataFrame, summ: pd.DataFrame, results: Path) -> None:
    results.mkdir(exist_ok=True)
    runs.to_csv(results / "table4_runs.csv", index=False)
    summ.to_csv(results / "table4.csv", index=False)
    head = ["Model", "Pearson", "RMSE", "95% CI (RMSE)", "Bias", "Seeds"]
    rows = table_rows(summ)
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join([lab, *cells]) + " |" for lab, cells in rows]
    (results / "table4.md").write_text("\n".join(md) + "\n")
    tex = ["\\begin{tabular}{lccccc}", "\\toprule", " & ".join(head) + " \\\\", "\\midrule"]
    tex += [" & ".join([lab, *cells]).replace("%", "\\%") + " \\\\" for lab, cells in rows]
    (results / "table4.tex").write_text("\n".join(tex + ["\\bottomrule", "\\end{tabular}"]) + "\n")
    print("\n" + "\n".join(md))
    print(f"\nwrote {results}/table4.{{csv,md,tex}} and table4_runs.csv (mean (SD) over seeds; CI = mean of per-seed bounds)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", choices=list(ALL), default=list(MODELS))
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--results", type=Path, default=ROOT / "results")
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--aggregate-only", action="store_true")
    p.add_argument("--force", action="store_true", help="rerun finished (model, seed) pairs")
    p.add_argument("--list", action="store_true")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="config override passed to every main.py run (repeatable), e.g. --set data.dir=/data/usmle")
    a = p.parse_args()

    if a.list:
        for k, (label, _, cfg, gpu, est) in ALL.items():
            print(f"{k:14} {label:36} {cfg:45} {'GPU ~%d GiB' % est if gpu else 'CPU'}{'' if k in MODELS else '  (extra)'}")
        return
    if a.preflight:
        sys.exit(1 if preflight(a.models, a.results, a.overrides) else 0)
    failures = [] if a.aggregate_only else run_sweep(a.models, a.seeds, a.results, a.force, a.overrides)
    runs = collect(a.models, a.seeds, a.results)
    if runs.empty:
        raise SystemExit("no finished runs to summarize")
    write_tables(runs, summarize(runs, a.models), a.results)
    if failures:
        print("\nFAILED runs:", *[f"{k} seed {s} ({why})" for k, s, why in failures], sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
