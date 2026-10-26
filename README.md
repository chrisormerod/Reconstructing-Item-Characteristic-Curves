# Reconstructing Item Characteristic Curves with fine-tuned LLMs

Implementation of Ormerod (2026), *Reconstructing Item Characteristic Curves using Fine-Tuned Large
Language Models* (arXiv:2601.02580v2), applied to the USMLE item set.

A decoder LLM is fine-tuned with LoRA so that, given a multiple-choice item and an **ability descriptor**
("Basic", "Proficient", …), the next-token probabilities over the option keys match the option probabilities a
population at that ability level would produce. Evaluating the model at every descriptor on an unseen item gives a
discrete ICC; fitting a 1PL curve to it recovers the item difficulty, which is compared with the known `Difficulty`.
Design and details: [PROJECT.md](PROJECT.md).

| Method | Config | Notes |
|--------|--------|-------|
| LoRA LLM simulator | `usmle_qwen3_{1_7b,4b,8b,14b,32b}.yaml`, `usmle_llama31_8b.yaml` | The method. 32B is 4-bit QLoRA |
| RoBERTa simulator | `usmle_roberta_large.yaml` | Maeda (2023): ability = share of the vocabulary kept |
| ModernBERT regression | `usmle_modernbert_regression.yaml` | Head fine-tuned with MSE on `b` |
| Features regression | `usmle_features.yaml` | TF-IDF + `textstat` features, ridge (CPU) |

## Install

```bash
pip install -e .          # Python >= 3.10
pip install -e .[quant]   # bitsandbytes, only for pretrained_model.quantization (CUDA, Linux)
pytest                    # no model download needed
```

Four tests (three in `test_descriptors.py`, plus `test_sft_dataset_samples_from_target`) fail: they assume 20
descriptors and the default config has 21 (it adds `Rudimentary` to close a gap in the paper's table).

## Data

`data.usmle_dir` (default `/mnt/c/data/idm/usmle/`) must hold `train_final.xlsx`, `test_final.xlsx` and
`gold_final.xlsx` ([format](docs/usmle_format.md)). `python scripts/simulate_usmle.py OUT_DIR` writes synthetic
spreadsheets in the same format.

## Run

```bash
python main.py --config config/usmle_qwen3_4b.yaml                  # full pipeline
python main.py --config config/default.yaml --dry-run               # parse, targets, one forward pass
python main.py --config config/default.yaml --stage targets         # stop after a stage
python main.py --config config/default.yaml --set training.learning_rate=1e-5
python main.py --config config/default.yaml --stage evaluate --resume results/<run>/<ts>/predefined/adapter
```

`--stage` runs up to and including that stage (`parse`, `targets`, `sft`, `train`, `predict`, `evaluate`, `all`).
Check a tokenizer first with `python scripts/check_option_tokens.py Qwen/Qwen3-4B`. Llama is gated: accept the licence
and run `huggingface-cli login`.

### Seeds and the full sweep

```bash
python scripts/repeat_runs.py config/usmle_qwen3_8b.yaml   # one model, seeds 1-5, mean/SD table
python scripts/run_all.py --preflight                      # load every model once, print GPU memory
python scripts/run_all.py                                  # all models x 5 seeds -> results/table4.{csv,md,tex}
python scripts/run_all.py --models qwen3_14b --seeds 1 2   # subset; --list, --aggregate-only, --force also exist
```

The sweep skips finished (model, seed) runs, so it resumes after an interruption, and it continues past failures
(`results/run_all_status.json`). The seed changes the train/dev split, initialisation and batch order; the test set is
fixed.

### Memory on one 48 GB L40S (~44 GiB usable)

Peak is roughly the bf16 weights plus ~1 GiB (batch size 1, gradient checkpointing). Qwen3-4B and 8B were measured
(7.6 and 15.3 GiB after load, 16.1 GiB peak for 8B); the rest are estimates, so run `--preflight` first.

| Model | Weights (bf16) | Expected peak |
|-------|----------------|---------------|
| Qwen3-1.7B / 4B | ~4 / 7.5 GiB | ~5 / ~9 GiB |
| Qwen3-8B, Llama-3.1-8B | ~15 GiB | ~17 GiB |
| Qwen3-14B | ~27.5 GiB | ~29-31 GiB |
| Qwen3-32B | ~61 GiB, **does not fit** | ~24 GiB with `quantization: 4bit` (NF4) |

Quantization changes the model, so `usmle_qwen3_14b_nf4.yaml` trains the 14B in 4-bit to measure its effect. If a
model does not fit, lower `pretrained_model.max_length` or `predict_batch_size`, or quantize. The earlier 8B run took
about 2.4 h of training on a laptop GPU, so five seeds of every model is days of compute.

## Outputs

`results/<run.name>/<timestamp>/` holds `config.resolved.yaml`, `targets.parquet` and, per split,
`predicted_icc.parquet`, `item_parameters.csv`, `metrics.json`, `training.json`, `adapter*/`, `icc_plots/` and
`scatter_test.png`. Folds are aggregated in `summary.csv`.

`metrics.json` gives, for dev and test, on the original `Difficulty` scale: Pearson r, RMSE, `bias` (mean predicted −
true), percentile-bootstrap 95% CIs for RMSE, bias and the train-mean baseline RMSE, and ICC MSE for ICC methods.
Report the test metrics (dev is in-sample for the recalibration line). The CIs resample test items
(`report.bootstrap_n`, default 10,000), need no retraining, and capture test-item sampling only; training variability
is the SD across seeds, reported separately. In `table4.*`, Pearson, RMSE and bias are mean (SD) over seeds and the CI is
the mean of the per-seed bounds. `python scripts/summarize_runs.py` tabulates metrics across runs.

## Layout

`llm_icc/data` items, descriptors, datasets · `llm_icc/irt` NRM, ICC fitting, recalibration · `llm_icc/modeling`
simulators, baselines, prompts, losses, training loops · `llm_icc/utils` parser, CLI, metrics, results, plots ·
`llm_icc/app.py` pipeline · `config/` experiments · `scripts/` sweeps and helpers · `docs/` method, formats, config
reference, experiments.

## Status

Run end to end on synthetic data with Qwen3-0.6B (bf16 and 4-bit). On the real data, single runs exist for Qwen3-4B/8B,
ModernBERT, RoBERTa and the features baseline (five seeds); the sweep, Llama-3.1-8B and the 1.7B/14B/32B models have not
been run, and runs made before the CI code existed have no intervals. Read every RMSE against the train-mean RMSE of
0.311 (the SD of the true test difficulties is 0.31); the features baseline does not beat it.
