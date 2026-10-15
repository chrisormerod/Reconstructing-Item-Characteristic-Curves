# Config reference

YAML, validated by `llm_icc/config/schema.py` (unknown keys are errors). A file may set `base: other.yaml`
(relative path) and override only what differs; `descriptors` is replaced wholesale, other sections merge.
Any key can be overridden with `--set a.b.c=value` (value parsed as YAML).
`config/default.yaml` is the annotated source of truth for defaults.

| Section | Keys |
|---------|------|
| `run` | `name`, `seed`, `results_dir`, `logs_dir`, `tensorboard`, `wandb` (both default false) |
| `data` | `usmle_dir`; `splits.{strategy: predefined\|kfold, k, dev_fraction}`; `exclude_pix`; `difficulty_transform.{method: standardize\|linear, shift, scale}` |
| `pretrained_model` | `name_or_path`, `dtype`, `max_length`, `gradient_checkpointing`, `enable_thinking`, `attn_implementation`, `predict_batch_size` |
| `lora` | `r`, `alpha`, `dropout`, `target_modules`, `bias` |
| `training` | `epochs`, `learning_rate`, `batch_size`, `gradient_accumulation_steps`, `weight_decay`, `warmup_ratio`, `max_grad_norm`, `eval_every_steps` (optimizer steps), `early_stopping_patience`, `sft.{enabled, epochs, learning_rate}` |
| `population` | `mu`, `sigma` |
| `calibration` | `icc_models` (first entry drives the metrics), `linear_recalibration` |
| `report` | `n_icc_plots` |
| `descriptors` | ordered mapping `label: [lower, upper]`; contiguous, first lower `-.inf`, last upper `.inf` |

`linear` difficulty transform: `b = scale·(Difficulty − shift)`. `standardize`: `shift`/`scale` are replaced
by the mean/SD of the training file's `Difficulty`.
