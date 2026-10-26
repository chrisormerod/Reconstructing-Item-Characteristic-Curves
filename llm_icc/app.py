"""Pipeline orchestration: parse -> targets -> split -> (sft -> train -> predict -> evaluate) -> report."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from llm_icc.config import Config, load_config
from llm_icc.data.item_bank import ItemBank, Split
from llm_icc.data.parameters import LogisticParameters
from llm_icc.irt.conversions import DifficultyTransform
from llm_icc.irt.icc_fit import ICCFitter
from llm_icc.irt.recalibration import LinearRecalibrator
from llm_icc.utils import metrics as M
from llm_icc.utils import reporting, results, system
from llm_icc.utils.cli import STAGES, build_parser
from llm_icc.utils.logging_utils import make_tracker, setup_logging
from llm_icc.utils.usmle import USMLEParser, describe_bank

log = logging.getLogger("llm_icc.app")


# --------------------------------------------------------------------------- targets
def build_targets(bank: ItemBank, config: Config) -> tuple[DifficultyTransform, pd.DataFrame]:
    """Fit the Difficulty -> b transform on the training-file items, attach 1PL parameters to every item,
    and return the long-format target table.

    Test items also receive parameters, but only for evaluation (ICC-level error and plots); no test
    quantity is used for fitting the transform or for training.
    """
    scale = config.scale()
    mu, sigma = config.population.mu, config.population.sigma
    dt = config.data.difficulty_transform
    transform = DifficultyTransform(dt.method, dt.shift, dt.scale).fit(
        [i.reference_difficulty for i in bank if i.metadata["source"] == "train"])
    theta_k, omega_k = scale.conditional_means(mu, sigma), scale.weights(mu, sigma)
    frames = []
    for item in bank:
        item.parameters = LogisticParameters(b=float(transform.to_b(item.reference_difficulty)))
        t = results.icc_table(item, item.target_distribution(scale, mu, sigma), scale.labels,
                              theta_k, omega_k, "target_prob")
        t["source"] = item.metadata["source"]
        frames.append(t)
    return transform, pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- evaluation
def evaluate_split(preds: dict[str, np.ndarray], bank: ItemBank, split: Split, config: Config,
                   transform: DifficultyTransform) -> tuple[dict, pd.DataFrame]:
    """Fit item parameters from predicted ICCs, recalibrate on dev, score dev and test."""
    scale = config.scale()
    mu, sigma = config.population.mu, config.population.sigma
    theta_k, omega_k = scale.conditional_means(mu, sigma), scale.weights(mu, sigma)
    fitter = ICCFitter()
    models = config.calibration.icc_models
    primary = models[0]

    rows = []
    for role, ids in (("dev", split.dev), ("test", split.test)):
        for iid in ids:
            item, P = bank[iid], preds[iid]
            row = {"item_id": iid, "role": role, "reference_difficulty": item.reference_difficulty,
                   "target_b": float(transform.to_b(item.reference_difficulty)),
                   "icc_mse": M.weighted_mse(P, item.target_distribution(scale, mu, sigma), omega_k)}
            pc = P[:, item.correct_index]
            for m in models:
                fit = fitter.fit(pc, theta_k, omega_k, m)
                row[f"b_{m}"], row[f"a_{m}"] = fit.b, fit.a
            row["b_pred"] = row[f"b_{primary}"]
            rows.append(row)
    return score_item_parameters(pd.DataFrame(rows), bank, split, config, transform, primary)


def score_item_parameters(df: pd.DataFrame, bank: ItemBank, split: Split, config: Config,
                          transform: DifficultyTransform, primary: str) -> tuple[dict, pd.DataFrame]:
    """From per-item `b_pred` (roles dev/test): map to Difficulty, recalibrate on dev, score dev and test."""
    df["difficulty_uncalibrated"] = transform.to_difficulty(df["b_pred"].to_numpy())

    out: dict = {"primary_icc_model": primary, "split": split.name}
    dev = df["role"] == "dev"
    if config.calibration.linear_recalibration:
        rc = LinearRecalibrator().fit(df.loc[dev, "b_pred"], df.loc[dev, "reference_difficulty"])
        df["difficulty_pred"] = rc.transform(df["b_pred"].to_numpy())
        out["recalibration"] = {"slope": rc.slope, "intercept": rc.intercept}
    else:
        df["difficulty_pred"] = df["difficulty_uncalibrated"]
        out["recalibration"] = None

    train_mean = float(np.mean([bank[i].reference_difficulty for i in split.train]))
    for role in ("dev", "test"):
        d = df[df["role"] == role]
        ref = d["reference_difficulty"]
        ci = lambda pred, stat: list(M.bootstrap_ci(  # noqa: E731
            pred, ref, stat, config.report.bootstrap_n, config.report.ci_level, config.run.seed))
        baseline = np.full(len(d), train_mean)
        out[role] = {
            "n_items": int(len(d)),
            "pearson": M.pearson(d["difficulty_pred"], ref),
            "rmse": M.rmse(d["difficulty_pred"], ref),
            "rmse_uncalibrated": M.rmse(d["difficulty_uncalibrated"], ref),
            "rmse_train_mean_baseline": M.rmse(baseline, ref),
            "rmse_ci": ci(d["difficulty_pred"], "rmse"),
            "rmse_train_mean_baseline_ci": ci(baseline, "rmse"),
            "bias": M.bias(d["difficulty_pred"], ref),
            "bias_ci": ci(d["difficulty_pred"], "bias"),
            "ci_level": config.report.ci_level,
            "ci_method": f"percentile bootstrap over items, n={config.report.bootstrap_n}",
        }
        if "icc_mse" in d:
            out[role]["icc_mse"] = float(d["icc_mse"].mean())
    if config.calibration.linear_recalibration:
        out["dev"]["note"] = "dev metrics are in-sample for the recalibration line"
    return out, df


# --------------------------------------------------------------------------- per split
def make_simulator(config: Config, resume: str | None = None):
    """Build (or reload from `resume`) the simulator selected by `pretrained_model.architecture`."""
    if config.pretrained_model.architecture == "roberta_mc":
        from llm_icc.modeling.roberta_simulator import RobertaResponseSimulator as Sim
    else:
        from llm_icc.modeling.lora_simulator import LoRAResponseSimulator as Sim
    return Sim.load(Path(resume), config) if resume else Sim(config)


def run_split(split: Split, bank: ItemBank, config: Config, transform: DifficultyTransform, stage_idx: int,
              args, run_dir: Path, tracker) -> dict | None:
    from llm_icc.data.torch_datasets import DistributionDataset, SFTDataset
    from llm_icc.modeling.training_loop import DistributionTrainer, SFTTrainer

    sdir = run_dir / split.name
    sdir.mkdir(parents=True, exist_ok=True)
    scale = config.scale()
    mu, sigma = config.population.mu, config.population.sigma
    theta_k, omega_k = scale.conditional_means(mu, sigma), scale.weights(mu, sigma)
    log.info("== split %s: train=%d dev=%d test=%d", split.name, len(split.train), len(split.dev), len(split.test))

    sim = make_simulator(config, args.resume)
    log.info("model loaded; %s", system.gpu_memory_report())

    if args.dry_run:
        item = bank[split.train[0]]
        icc = sim.predict_icc(item, scale)
        log.info("dry run OK: item %s ICC shape %s, row sums in [%.4f, %.4f]", item.item_id, icc.shape,
                 icc.sum(1).min(), icc.sum(1).max())
        return None

    resumed_predict = bool(args.resume) and stage_idx >= STAGES.index("predict")
    mk = lambda ids: sim.build_examples([bank[i] for i in ids], scale, mu, sigma)  # noqa: E731
    if not resumed_predict:
        train_ex = mk(split.train)
        if config.training.sft.enabled and not args.resume:
            log.info("stage 1: SFT")
            SFTTrainer(sim, SFTDataset(train_ex, seed=config.run.seed), config, log,
                       seed=config.run.seed, tracker=tracker).train()
            sim.save(sdir / "adapter_sft")
        if stage_idx >= STAGES.index("train"):
            log.info("stage 2: distribution correction")
            res = DistributionTrainer(sim, DistributionDataset(train_ex), DistributionDataset(mk(split.dev)),
                                      config, log, checkpoint_dir=sdir / "adapter_best",
                                      seed=config.run.seed, tracker=tracker).train()
            results.write_json({"best_dev_loss": res.best_dev_loss, "best_step": res.best_step,
                                "steps": res.steps, "history": res.history}, sdir / "training.json")
            sim.save(sdir / "adapter")
            log.info("trained; %s", system.gpu_memory_report())
    if stage_idx < STAGES.index("predict"):
        return None

    log.info("predicting ICCs for dev and test items")
    preds, frames = {}, []
    for role, ids in (("dev", split.dev), ("test", split.test)):
        for iid in ids:
            P = sim.predict_icc(bank[iid], scale)
            preds[iid] = P
            t = results.icc_table(bank[iid], P, scale.labels, theta_k, omega_k, "predicted_prob")
            t["role"] = role
            frames.append(t)
    results.write_parquet(pd.concat(frames, ignore_index=True), sdir / "predicted_icc.parquet")
    if stage_idx < STAGES.index("evaluate"):
        return None

    metrics, params_df = evaluate_split(preds, bank, split, config, transform)
    results.write_csv(params_df, sdir / "item_parameters.csv")
    results.write_json(metrics, sdir / "metrics.json")
    t = params_df[params_df["role"] == "test"]
    reporting.plot_scatter(t["difficulty_pred"], t["reference_difficulty"], sdir / "scatter_test.png",
                           f"{split.name} test (r={metrics['test']['pearson']:.3f})")
    for iid in split.test[: config.report.n_icc_plots]:
        item = bank[iid]
        reporting.plot_icc(item, theta_k, item.target_distribution(scale, mu, sigma), preds[iid],
                           sdir / "icc_plots" / f"{iid}.png",
                           f"item {iid}  Difficulty={item.reference_difficulty:.2f}")
    log.info("%s test: r=%.3f rmse=%.3f (baseline %.3f)", split.name, metrics["test"]["pearson"],
             metrics["test"]["rmse"], metrics["test"]["rmse_train_mean_baseline"])
    return metrics


def run_regression_split(split: Split, bank: ItemBank, config: Config, transform: DifficultyTransform,
                         stage_idx: int, args, run_dir: Path, tracker) -> dict | None:
    """Baseline: fine-tune a one-output regression head on b with MSE, then score like the ICC pipeline."""
    from llm_icc.modeling.regression_baseline import RegressionDataset, RegressionModel
    from llm_icc.modeling.training_loop import RegressionTrainer

    sdir = run_dir / split.name
    sdir.mkdir(parents=True, exist_ok=True)
    log.info("== split %s (regression): train=%d dev=%d test=%d", split.name, len(split.train), len(split.dev),
             len(split.test))
    model = RegressionModel.load(Path(args.resume), config) if args.resume else RegressionModel(config)
    log.info("model loaded; %s", system.gpu_memory_report())
    items = lambda ids: [bank[i] for i in ids]  # noqa: E731

    if args.dry_run:
        b = model.predict_b(items(split.train[:2]))
        log.info("dry run OK: predicted b for 2 items: %s", np.round(b, 4))
        return None
    if not args.resume and stage_idx >= STAGES.index("train"):
        log.info("training regression head (MSE)")
        res = RegressionTrainer(model, RegressionDataset(model.build_examples(items(split.train))),
                                RegressionDataset(model.build_examples(items(split.dev))), config, log,
                                checkpoint_dir=sdir / "model_best", seed=config.run.seed, tracker=tracker).train()
        results.write_json({"best_dev_loss": res.best_dev_loss, "best_step": res.best_step,
                            "steps": res.steps, "history": res.history}, sdir / "training.json")
        model.save(sdir / "model")
    if stage_idx < STAGES.index("predict"):
        return None

    rows = []
    for role, ids in (("dev", split.dev), ("test", split.test)):
        for iid, b in zip(ids, model.predict_b(items(ids))):
            item = bank[iid]
            rows.append({"item_id": iid, "role": role, "reference_difficulty": item.reference_difficulty,
                         "target_b": float(transform.to_b(item.reference_difficulty)), "b_pred": float(b)})
    if stage_idx < STAGES.index("evaluate"):
        return None
    metrics, params_df = score_item_parameters(pd.DataFrame(rows), bank, split, config, transform, "regression")
    results.write_csv(params_df, sdir / "item_parameters.csv")
    results.write_json(metrics, sdir / "metrics.json")
    t = params_df[params_df["role"] == "test"]
    reporting.plot_scatter(t["difficulty_pred"], t["reference_difficulty"], sdir / "scatter_test.png",
                           f"{split.name} test (r={metrics['test']['pearson']:.3f})")
    log.info("%s test: r=%.3f rmse=%.3f (baseline %.3f)", split.name, metrics["test"]["pearson"],
             metrics["test"]["rmse"], metrics["test"]["rmse_train_mean_baseline"])
    return metrics


def run_features_split(split: Split, bank: ItemBank, config: Config, transform: DifficultyTransform,
                       stage_idx: int, args, run_dir: Path, tracker) -> dict | None:
    """Baseline: TF-IDF + textstat features -> ridge on b, then score like the ICC pipeline. Fits in seconds on a CPU."""
    from llm_icc.modeling.features_baseline import FeaturesModel

    sdir = run_dir / split.name
    sdir.mkdir(parents=True, exist_ok=True)
    log.info("== split %s (features): train=%d dev=%d test=%d", split.name, len(split.train), len(split.dev),
             len(split.test))
    if args.dry_run or stage_idx < STAGES.index("predict"):
        return None
    items = lambda ids: [bank[i] for i in ids]  # noqa: E731
    model = FeaturesModel(config).fit(items(split.train))
    log.info("ridge alpha=%.3g, %d features", model.ridge.alpha_, model.ridge.coef_.size)
    rows = []
    for role, ids in (("dev", split.dev), ("test", split.test)):
        for iid, b in zip(ids, model.predict_b(items(ids))):
            item = bank[iid]
            rows.append({"item_id": iid, "role": role, "reference_difficulty": item.reference_difficulty,
                         "target_b": float(transform.to_b(item.reference_difficulty)), "b_pred": float(b)})
    if stage_idx < STAGES.index("evaluate"):
        return None
    metrics, params_df = score_item_parameters(pd.DataFrame(rows), bank, split, config, transform, "features")
    results.write_csv(params_df, sdir / "item_parameters.csv")
    results.write_json(metrics, sdir / "metrics.json")
    t = params_df[params_df["role"] == "test"]
    reporting.plot_scatter(t["difficulty_pred"], t["reference_difficulty"], sdir / "scatter_test.png",
                           f"{split.name} test (r={metrics['test']['pearson']:.3f})")
    log.info("%s test: r=%.3f rmse=%.3f (baseline %.3f)", split.name, metrics["test"]["pearson"],
             metrics["test"]["rmse"], metrics["test"]["rmse_train_mean_baseline"])
    return metrics


# --------------------------------------------------------------------------- entry
def run(config: Config, args) -> Path:
    stage_idx = len(STAGES) - 2 if args.stage == "all" else STAGES.index(args.stage)  # all == evaluate
    system.set_seed(config.run.seed)
    run_dir = system.make_run_dir(config.run.results_dir, config.run.name)
    setup_logging(config.run.logs_dir, config.run.name)
    results.write_config(config, run_dir / "config.resolved.yaml")
    log.info("run dir %s | git %s", run_dir, system.git_commit_hash())

    bank = USMLEParser(config.data.usmle_dir).load()
    if config.data.exclude_pix:
        bank = bank.filter(lambda i: i.metadata["item_type"] != "PIX")
    log.info("parsed: %s", describe_bank(bank))
    if stage_idx < STAGES.index("targets"):
        return run_dir

    transform, targets = build_targets(bank, config)
    results.write_parquet(targets, run_dir / "targets.parquet")
    log.info("targets built (difficulty transform: %s)", transform)
    if stage_idx < STAGES.index("sft") and not args.dry_run:
        return run_dir

    sp = config.data.splits
    splits = bank.split(sp.strategy, seed=config.run.seed, dev_fraction=sp.dev_fraction, k=sp.k)
    if args.fold is not None:
        if not 0 <= args.fold < len(splits):
            raise SystemExit(f"--fold {args.fold} out of range for {len(splits)} split(s)")
        splits = [splits[args.fold]]
    tracker = make_tracker(config.run, run_dir)

    all_metrics = {}
    for split in splits:
        runner = {"regression": run_regression_split, "features": run_features_split}.get(config.run.method, run_split)
        m = runner(split, bank, config, transform, stage_idx, args, run_dir, tracker)
        if args.dry_run:
            break
        if m is not None:
            all_metrics[split.name] = m
    if all_metrics:
        table = reporting.summary_table(all_metrics)
        results.write_csv(table.reset_index(names="split"), run_dir / "summary.csv")
        results.write_json(all_metrics, run_dir / "metrics.json")
        log.info("summary (test):\n%s", table.to_string())
    return run_dir


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(load_config(args.config, args.overrides), args)
