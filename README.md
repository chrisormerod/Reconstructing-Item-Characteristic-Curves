# Reconstructing Item Characteristic Curves with fine-tuned LLMs

Implementation of Ormerod (2026), *Reconstructing Item Characteristic Curves using Fine-Tuned Large
Language Models* (arXiv:2601.02580v2), applied to the USMLE item set.

A decoder LLM (Qwen3 by default) is fine-tuned with LoRA so that, given a multiple-choice item and an
**ability descriptor** ("Basic", "Proficient", …), the next-token probabilities over the option keys
(`A`, `B`, …) match the option probabilities a population at that ability level would produce.
Evaluating the model at every descriptor (21 in the default config) on an unseen item gives a discrete ICC; fitting a 1PL curve to
it recovers the item difficulty, which is compared with the known `Difficulty`.

```
known Difficulty ─► b (standardize) ─► NRM (uniform distractors, eq. 4) ─► targets P_ik at θ_k
                                                                          │
item + descriptor ─► LoRA LLM ─► restricted softmax over option tokens ───┴─► ω-weighted MSE
held-out item ─► LLM at every descriptor ─► discrete ICC ─► 1PL fit (eq. 11) ─► linear recalibration ─► Difficulty
```

There are no student responses in the data, so the only target mode is known-`b`. Items are text only.

Besides the LoRA LLM simulator the repo contains the comparison systems used in the article's Table 4, all scored
through the same pipeline (`run.method` / `pretrained_model.architecture` select them):

| Method | Config | What it is |
|--------|--------|------------|
| LoRA LLM simulator | `usmle_qwen3_{1_7b,4b,8b,14b,32b}.yaml`, `usmle_llama31_8b.yaml` | The method. 32B (and `qwen3_14b_nf4`) are 4-bit QLoRA |
| RoBERTa simulator | `usmle_roberta_large.yaml` | Maeda (2023) vocabulary-ablation variant, see below |
| ModernBERT regression | `usmle_modernbert_regression.yaml` | One-output head fine-tuned with MSE on `b` (no ICC) |
| Features regression | `usmle_features.yaml` | TF-IDF + `textstat` readability/option features -> ridge on `b` (CPU, no ICC) |

## Install

```bash
pip install -e .          # Python >= 3.10; torch, transformers, peft, scikit-learn, textstat, pandas, ...
pip install -e .[quant]   # adds bitsandbytes, needed only for pretrained_model.quantization (CUDA, Linux)
pytest                    # 69 tests, no model download needed
```

Four tests (three in `test_descriptors.py` and `test_sft_dataset_samples_from_target`) currently fail: they assume
20 descriptors, and the default config now has 21 (see "Decisions worth knowing"). The other 65 pass.

## Data

`/mnt/c/data/idm/usmle/` (configurable via `data.usmle_dir`) must hold `train_final.xlsx`,
`test_final.xlsx` and `gold_final.xlsx`. See [docs/usmle_format.md](docs/usmle_format.md). Without the real
data, `python scripts/simulate_usmle.py OUT_DIR` writes synthetic spreadsheets in the same format.

## Run

```bash
python main.py --config config/usmle_qwen3_4b.yaml                  # full pipeline
python main.py --config config/default.yaml --dry-run               # parse, targets, one forward pass
python main.py --config config/default.yaml --stage targets         # stop after building targets
python main.py --config config/default.yaml --fold 0 \
    --set data.splits.strategy=kfold --set training.learning_rate=1e-5
python main.py --config config/default.yaml --stage evaluate --resume results/<run>/<ts>/predefined/adapter
```

`--stage` runs the pipeline **up to and including** that stage (`parse`, `targets`, `sft`, `train`,
`predict`, `evaluate`, `all`). `--resume` loads a LoRA adapter instead of creating one; it skips SFT, and with
`--stage predict|evaluate` it skips training entirely. Before running the 4B/8B configs, check the tokenizer
with `python scripts/check_option_tokens.py Qwen/Qwen3-4B`.

### Seeds, the full sweep, and a single L40S

```bash
python scripts/repeat_runs.py config/usmle_qwen3_8b.yaml            # one model, seeds 1-5, mean/SD table
python scripts/run_all.py --list                                    # every Table 4 model, config and GPU estimate
python scripts/run_all.py --preflight                               # load each model, one forward pass, print GPU memory
python scripts/run_all.py                                           # all models x seeds 1-5, then results/table4.{csv,md,tex}
python scripts/run_all.py --models qwen3_14b qwen3_32b --seeds 1 2  # a subset
python scripts/run_all.py --aggregate-only                          # rebuild the table from finished runs
```

`run_all.py` runs models cheapest first, skips (model, seed) pairs that already have a `metrics.json` (so an
interrupted sweep resumes; a half-trained run restarts), records failures in `results/run_all_status.json` and keeps
going. Runs are named `<name>_seed<S>`. The seed changes the train/dev split (the dev items are drawn at random),
initialisation and batch order; the test set is fixed.

**Memory on a 48 GB L40S (about 44 GiB usable).** Training holds the frozen base weights plus LoRA parameters and
optimizer state (tens of MiB) and activations at batch size 1 with gradient checkpointing and the LM head run on one
position only, so peak is roughly weights + 1 GiB. Measured on the real data (24 GB laptop GPU): Qwen3-4B 7.6 GiB
after load; Qwen3-8B 15.3 GiB after load, 16.1 GiB peak in training. The rest are estimates from parameter counts:

| Model | Weights | Expected peak | Fits unquantized? |
|-------|---------|---------------|-------------------|
| Qwen3-1.7B / 4B | ~4 / 7.5 GiB | ~5 / ~9 GiB | yes |
| Qwen3-8B, Llama-3.1-8B | ~15 GiB | ~17 GiB | yes |
| Qwen3-14B | ~27.5 GiB | ~29-31 GiB | yes, with ~13 GiB spare |
| Qwen3-32B | ~61 GiB bf16 | - | **no**: use `quantization: 4bit` (NF4, ~17 GiB weights, plus ~6 GiB for the fp32 embedding and LM head that `prepare_model_for_kbit_training` upcasts, ~24 GiB peak) |

`8bit` would also fit 32B (~33 GiB) but bitsandbytes int8 training is markedly slower, so NF4 is the default.
Quantization changes the model, so the 32B row is not strictly comparable with the bf16 rows; `config/usmle_qwen3_14b_nf4.yaml`
(`run_all.py --models qwen3_14b_nf4`) trains the same 14B model in 4-bit, so the effect of quantization can be measured
and reported. Run `--preflight` first: it prints the real post-load memory for each model on your GPU, and catches
gated-model or tokenizer problems (Llama needs a licence and `huggingface-cli login`) before hours of training.
If a model does not fit, lower `pretrained_model.max_length` or `predict_batch_size`, or quantize.
Time: the earlier Qwen3-8B run took about 2.4 h of training on the laptop GPU, so 5 seeds of every model is days of
compute; 4-bit 32B will be the slowest.

## Outputs

`results/<run.name>/<timestamp>/`:

| File | Contents |
|------|----------|
| `config.resolved.yaml` | Frozen config |
| `targets.parquet` | Target option probabilities, long format (item × descriptor × option) |
| `<split>/predicted_icc.parquet` | Model option probabilities for dev and test items |
| `<split>/item_parameters.csv` | Fitted `b` per item, recalibrated and original-scale difficulty, ICC error |
| `<split>/metrics.json`, `<split>/training.json` | Per role (dev, test): Pearson r, RMSE, `bias`, percentile-bootstrap 95% CIs for RMSE, bias and the train-mean baseline RMSE, ICC MSE (ICC methods); dev-loss history |
| `<split>/adapter*/` | LoRA adapters (SFT, best, final) |
| `<split>/icc_plots/`, `scatter_test.png` | Target vs. LLM ICCs (Fig. 3 style) and a predicted-vs-reference scatter |
| `summary.csv`, `metrics.json` | Aggregate over folds |

`results/table4.{csv,md,tex}` (from `run_all.py`) and `results/<name>_repeats.csv` (from `repeat_runs.py`) hold the
seed-aggregated tables. `python scripts/summarize_runs.py` tabulates metrics across runs. Metrics are on the original `Difficulty`
scale and are reported next to a train-mean baseline RMSE, so that a result can be judged against it.

**Confidence intervals and bias.** `bias` is mean(predicted − true `Difficulty`) after dev recalibration. The 95% CIs
are percentile bootstraps over the test items (`report.bootstrap_n`, default 10,000; `report.ci_level`, 0.95), always
with the same resampled item indices (seeded by `run.seed`). They need no retraining and capture test-item sampling
only. Training/split variability is the SD across seeds (`repeat_runs.py`, `run_all.py`); the two are reported
separately and not combined. In `table4.*` the Pearson, RMSE and bias are mean (SD) over seeds and the CI is the mean of
the per-seed bounds.

## Layout

`llm_icc/data` item/descriptor/option classes and torch datasets · `llm_icc/irt` NRM, eq. 4, ICC fitting,
recalibration · `llm_icc/modeling` LoRA simulator, prompts, losses, manual training loops ·
`llm_icc/utils` USMLE parser, CLI, logging, metrics (incl. bias and bootstrap CIs), results, plots ·
`llm_icc/app.py` pipeline · `config/` YAML experiments · `scripts/` `run_all.py` (full sweep and Table 4),
`repeat_runs.py` (seeds), `summarize_runs.py`, `check_option_tokens.py`, `simulate_usmle.py` · `tests/` · `docs/`
method, formats, config reference, experiments. Besides `lora_simulator.py`, `llm_icc/modeling` has
`roberta_simulator.py`, `regression_baseline.py` (ModernBERT) and `features_baseline.py` (TF-IDF + textstat).

## Decisions worth knowing

- **Descriptor gap.** The paper's descriptor table leaves (−1.8, −1.5] without a label. The default config bridges
  it with an extra label, `Rudimentary`, giving 21 descriptors (the article's Table 1 lists 20); the validator rejects
  gaps and overlaps. The four failing tests still assume 20 and need updating, or the label should be removed. Other
  table/equation typos are listed in [docs/method.md](docs/method.md).
- **Llama prompt date.** Llama 3.1's chat template can inject the current date into the system prompt, so
  `usmle_llama31_8b.yaml` pins it with `pretrained_model.chat_template_kwargs.date_string`.
- **Quantization (QLoRA).** `pretrained_model.quantization: 4bit|8bit` loads the frozen base with bitsandbytes
  (NF4, double quantization, `dtype` as compute type) and trains 16-bit LoRA adapters on top. It needs CUDA. A 4-bit
  LoRA smoke run was verified on Qwen3-0.6B only; 14B/32B have not been run.
- **Features baseline.** TF-IDF (1-2-grams, 5,000 features) over stem, options and key, plus textstat readability
  and length measures of the stem and option statistics, ridge regression with the penalty chosen by leave-one-out CV on
  the training items. Untuned defaults; see `features:` in [docs/config_reference.md](docs/config_reference.md).
- **Prompt wording is a reconstruction.** I did not have the Appendix A text, so the system/user prompts in
  `llm_icc/modeling/prompts.py` follow the described structure but are not verbatim. The answer slot is the
  next token after `**Student Answer:**`, and option keys are single tokens (` A`…` J`).
- **Difficulty scale.** Training targets use `b = (Difficulty − mean)/SD` with mean/SD from the training file
  (a provisional map; `Difficulty` is only approximately normal and the abilities are assumed N(0,1)). The
  dev-set linear recalibration then maps predictions onto `Difficulty`. Dev metrics are in-sample for that line;
  **test metrics are the ones to report**.
- **Left padding.** Batches are left-padded so the answer slot is the last position (equivalent to "last
  non-pad position", and lets the LM head run on one position only).
- **Targets for test items** are built from gold difficulty for evaluation/plots only; nothing from the test
  set enters training or the transform.

## Status

The pipeline (SFT → distribution training → prediction → evaluation, `--resume`, `--dry-run`) was run end to end on
synthetic data with Qwen3-0.6B (bf16 and 4-bit). On the real USMLE data, single runs exist for Qwen3-4B/8B,
ModernBERT, RoBERTa and the features baseline (the last five-seeded); the multi-seed sweep, Llama-3.1-8B and the
1.7B/14B/32B models have **not** been run, and runs made before the CI code existed have no intervals. Results are
generated into `results/` (git-ignored) rather than quoted here. Observed so far, on the 201 test items: the features
baseline is indistinguishable from predicting the training mean (RMSE ≈ 0.31); the train-mean RMSE is 0.311 and the SD of
the true test difficulties is 0.31, so every RMSE should be read against that. The raw `test_final.xlsx` has 159 items
with 5 options (see §8 of PROJECT.md), unlike the article's Table 2.

## RoBERTa variant

`config/usmle_roberta_large.yaml` swaps the LoRA LLM for `FacebookAI/roberta-large` as a multiple-choice model,
following Maeda (2023, arXiv:2310.11655): ability is simulated by zeroing the word embeddings of a random share of
the vocabulary. Descriptor k keeps a fraction Phi(theta_k) of the words; the ICC is the option distribution averaged
over `roberta.ablation_samples` ablations. The encoder and head are fully fine-tuned with the same SFT and
distribution stages. Code: `llm_icc/modeling/roberta_simulator.py`.

    python main.py --config config/usmle_roberta_large.yaml
