# Config reference

YAML, validated by `llm_icc/config/schema.py` (unknown keys are errors). A file may set `base: other.yaml`
(relative path) and override only what differs; `descriptors` is replaced wholesale, other sections merge.
Any key can be overridden with `--set a.b.c=value` (value parsed as YAML).
`config/default.yaml` is the annotated source of truth for defaults.

| Section | Keys |
|---------|------|
| `run` | `name`, `seed`, `results_dir`, `logs_dir`, `tensorboard`, `wandb` (both default false) |
| `data` | `dir` (alias `usmle_dir`); `files.{train, test, gold}`; `columns.{id, stem, option_prefix, key, key_text, item_type, exam, difficulty}`; `splits.{strategy: predefined\|kfold, k, dev_fraction}`; `exclude_pix`; `difficulty_transform.{method: standardize\|linear, shift, scale}` |
| `pretrained_model` | `architecture` (`causal_lm` \| `roberta_mc`), `name_or_path`, `dtype`, `quantization` (`none` \| `8bit` \| `4bit`; bitsandbytes QLoRA, CUDA only), `max_length`, `gradient_checkpointing`, `enable_thinking`, `chat_template_kwargs` (extra chat-template variables, e.g. Llama 3.1 `date_string`), `attn_implementation`, `predict_batch_size` |
| `roberta` | `ablation_samples`, `protect_special_tokens` (only for `roberta_mc`) |
| `features` | `tfidf_max_features`, `ngram_max`, `tfidf_min_df`, `numeric_weight`, `alpha_log10_min/max`, `include_correct_answer` (only for `run.method: features`) |
| `lora` | `r`, `alpha`, `dropout`, `target_modules`, `bias` |
| `training` | `epochs`, `learning_rate`, `batch_size`, `gradient_accumulation_steps`, `weight_decay`, `warmup_ratio`, `max_grad_norm`, `eval_every_steps` (optimizer steps), `early_stopping_patience`, `sft.{enabled, epochs, learning_rate}` |
| `population` | `mu`, `sigma` |
| `calibration` | `icc_models` (first entry drives the metrics), `linear_recalibration` |
| `report` | `n_icc_plots`, `bootstrap_n` (percentile-bootstrap resamples over items for the RMSE/bias CIs), `ci_level` |
| `descriptors` | ordered mapping `label: [lower, upper]`; contiguous, first lower `-.inf`, last upper `.inf` |

`linear` difficulty transform: `b = scale·(Difficulty − shift)`. `standardize`: `shift`/`scale` are replaced
by the mean/SD of the training file's `Difficulty`.
