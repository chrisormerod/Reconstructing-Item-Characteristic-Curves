# Reproducing the USMLE experiments

1. `pip install -e . && pytest`
2. `python scripts/check_option_tokens.py Qwen/Qwen3-4B` (every key must be a single token).
3. Smoke test: `python main.py --config config/usmle_qwen3_4b.yaml --dry-run`.
4. Predefined split (train/dev from the training file, official test set):
   `python main.py --config config/usmle_qwen3_4b.yaml`
5. 5-fold CV over the training file (same test set in every fold):
   `python main.py --config config/usmle_qwen3_4b.yaml --set data.splits.strategy=kfold`
   (one fold: add `--fold 2`).
6. Larger model: `config/usmle_qwen3_8b.yaml` (needs more GPU memory; lower `max_length` if needed).
7. Compare runs: `python scripts/summarize_runs.py results/`.

Things to vary: `data.exclude_pix`, `training.sft.enabled`, `lora.r`, `data.difficulty_transform`,
`calibration.linear_recalibration`. Compare RMSE with `rmse_train_mean_baseline` in `metrics.json`.

Runtime reference (Qwen3-0.6B, 54 items, one 24 GB GPU): SFT 1 epoch ≈ 6 min, ~1 GB peak memory. The real
466-item run with 4B/8B is ~8× more examples per epoch and a larger model; plan accordingly.
