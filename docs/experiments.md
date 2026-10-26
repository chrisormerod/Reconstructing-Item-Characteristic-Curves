# Reproducing the USMLE experiments

1. `pip install -e . && pytest`
2. `python scripts/check_option_tokens.py Qwen/Qwen3-4B` (every key must be a single token).
3. Smoke test: `python main.py --config config/usmle_qwen3_4b.yaml --dry-run`.
4. Predefined split (train/dev from the training file, official test set):
   `python main.py --config config/usmle_qwen3_4b.yaml`
5. 5-fold CV over the training file (same test set in every fold):
   `python main.py --config config/usmle_qwen3_4b.yaml --set data.splits.strategy=kfold`
   (one fold: add `--fold 2`).
6. Llama-3.1-8B-Instruct (gated; accept the licence and `huggingface-cli login`): run `python scripts/check_option_tokens.py meta-llama/Llama-3.1-8B-Instruct`, then `config/usmle_llama31_8b.yaml`.
7. Larger models: `config/usmle_qwen3_{8b,14b}.yaml`; `usmle_qwen3_32b.yaml` is 4-bit (QLoRA) because bf16 weights alone are ~61 GiB. See the L40S memory table in the README.
8. Compare runs: `python scripts/summarize_runs.py results/`.
9. All Table 4 models over 5 seeds: `python scripts/run_all.py --preflight`, then `python scripts/run_all.py` (resumable; writes `results/table4.{csv,md,tex}`).

Things to vary: `data.exclude_pix`, `training.sft.enabled`, `lora.r`, `data.difficulty_transform`,
`calibration.linear_recalibration`. Compare RMSE with `rmse_train_mean_baseline` in `metrics.json`.

Runtime reference (Qwen3-0.6B, 54 items, one 24 GB GPU): SFT 1 epoch ≈ 6 min, ~1 GB peak memory. The real
466-item run with 4B/8B is ~8× more examples per epoch and a larger model; plan accordingly.
