# Reconstructing Item Characteristic Curves with fine-tuned LLMs

Implementation of Ormerod (2026), *Reconstructing Item Characteristic Curves using Fine-Tuned Large
Language Models* (arXiv:2601.02580v2), applied to the USMLE item set.

A decoder LLM (Qwen3 by default) is fine-tuned with LoRA so that, given a multiple-choice item and an
**ability descriptor** ("Basic", "Proficient", …), the next-token probabilities over the option keys
(`A`, `B`, …) match the option probabilities a population at that ability level would produce.
Evaluating the model at all 20 descriptors on an unseen item gives a discrete ICC; fitting a 1PL curve to
it recovers the item difficulty, which is compared with the known `Difficulty`.

```
known Difficulty ─► b (standardize) ─► NRM (uniform distractors, eq. 4) ─► targets P_ik at θ_k
                                                                          │
item + descriptor ─► LoRA LLM ─► restricted softmax over option tokens ───┴─► ω-weighted MSE
held-out item ─► LLM at 20 descriptors ─► discrete ICC ─► 1PL fit (eq. 11) ─► linear recalibration ─► Difficulty
```

There are no student responses in the data, so the only target mode is known-`b`. Items are text only.

## Install

```bash
pip install -e .          # Python >= 3.10; torch, transformers, peft, pandas, pyarrow, openpyxl, ...
pytest                    # 52 unit tests, no model download needed
```

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

## Outputs

`results/<run.name>/<timestamp>/`:

| File | Contents |
|------|----------|
| `config.resolved.yaml` | Frozen config |
| `targets.parquet` | Target option probabilities, long format (item × descriptor × option) |
| `<split>/predicted_icc.parquet` | Model option probabilities for dev and test items |
| `<split>/item_parameters.csv` | Fitted `b` per item, recalibrated and original-scale difficulty, ICC error |
| `<split>/metrics.json`, `<split>/training.json` | Pearson r, RMSE, ICC MSE; dev-loss history |
| `<split>/adapter*/` | LoRA adapters (SFT, best, final) |
| `<split>/icc_plots/`, `scatter_test.png` | Target vs. LLM ICCs (Fig. 3 style) and a predicted-vs-reference scatter |
| `summary.csv`, `metrics.json` | Aggregate over folds |

`python scripts/summarize_runs.py` tabulates metrics across runs. Metrics are on the original `Difficulty`
scale and are reported next to a train-mean baseline RMSE, so that a result can be judged against it.

## Layout

`llm_icc/data` item/descriptor/option classes and torch datasets · `llm_icc/irt` NRM, eq. 4, ICC fitting,
recalibration · `llm_icc/modeling` LoRA simulator, prompts, losses, manual training loops ·
`llm_icc/utils` USMLE parser, CLI, logging, metrics, results, plots · `llm_icc/app.py` pipeline ·
`config/` YAML experiments · `scripts/` helpers · `tests/` · `docs/` method, formats, config reference,
experiments.

## Decisions worth knowing

- **Descriptor gap.** Paper Table 2 leaves a gap between Minimal (ends −1.8) and Emerging (starts −1.5). The
  default config extends Emerging down to −1.8; the validator rejects gaps and overlaps. Other Table 2/eq.
  typos are listed in [docs/method.md](docs/method.md).
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

Unit tests pass. The pipeline (SFT → distribution training → prediction → evaluation, `--resume`,
`--dry-run`) was run end to end on synthetic data with Qwen3-0.6B. **It has not been run on the real USMLE data
with the 4B/8B models**, so no accuracy numbers are claimed here.

## RoBERTa variant

`config/usmle_roberta_large.yaml` swaps the LoRA LLM for `FacebookAI/roberta-large` as a multiple-choice model,
following Maeda (2023, arXiv:2310.11655): ability is simulated by zeroing the word embeddings of a random share of
the vocabulary. Descriptor k keeps a fraction Phi(theta_k) of the words; the ICC is the option distribution averaged
over `roberta.ablation_samples` ablations. The encoder and head are fully fine-tuned with the same SFT and
distribution stages. Code: `llm_icc/modeling/roberta_simulator.py`.

    python main.py --config config/usmle_roberta_large.yaml
