"""Config schema for every YAML section (pydantic v2). Unknown keys are rejected."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from llm_icc.data.descriptors import DescriptorScale


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunConfig(_Base):
    name: str = "default"
    seed: int = 1234
    results_dir: str = "results/"
    logs_dir: str = "logs/"
    method: Literal["icc", "regression", "features"] = "icc"   # icc = response simulator; regression/features = item-parameter baselines
    tensorboard: bool = False
    wandb: bool = False


class SplitConfig(_Base):
    strategy: Literal["predefined", "kfold"] = "predefined"
    k: int = Field(5, ge=2)
    dev_fraction: float = Field(0.1, gt=0, lt=1)


class DifficultyTransformConfig(_Base):
    method: Literal["standardize", "linear"] = "standardize"
    shift: float = 0.0
    scale: float = 1.0

    @field_validator("scale")
    @classmethod
    def _nonzero(cls, v):
        if v == 0:
            raise ValueError("scale must be non-zero")
        return v


class DataConfig(_Base):
    usmle_dir: str = "/mnt/c/data/idm/usmle"
    splits: SplitConfig = SplitConfig()
    exclude_pix: bool = False
    difficulty_transform: DifficultyTransformConfig = DifficultyTransformConfig()


class PretrainedModelConfig(_Base):
    architecture: Literal["causal_lm", "roberta_mc"] = "causal_lm"  # causal_lm = LoRA LLM; roberta_mc = RoBERTa
    name_or_path: str = "Qwen/Qwen3-4B"
    dtype: Literal["bfloat16", "float16"] = "bfloat16"
    max_length: int = 2048
    gradient_checkpointing: bool = True
    enable_thinking: bool = False
    chat_template_kwargs: dict[str, Any] = {}     # extra template variables, e.g. Llama 3.1 `date_string`
    attn_implementation: str = "sdpa"
    predict_batch_size: int = 10


class LoraConfigSchema(_Base):
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05
    target_modules: list[str] = ["q_proj", "k_proj", "v_proj"]
    bias: Literal["none", "all", "lora_only"] = "none"


class RobertaConfigSchema(_Base):
    """Used only when pretrained_model.architecture == roberta_mc (Maeda 2023, arXiv:2310.11655)."""
    ablation_samples: int = Field(10, ge=1)       # random vocabulary ablations averaged per (item, descriptor)
    protect_special_tokens: bool = True           # never zero <s>, </s>, <pad>, <unk>, <mask>
    include_correct_answer: bool = True           # append the correct option's text to the stem


class RegressionConfigSchema(_Base):
    """Used only when run.method == regression."""
    max_length: int = 1024
    include_correct_answer: bool = True           # the key is part of the input; difficulty depends on it


class FeaturesConfigSchema(_Base):
    """Used only when run.method == features (TF-IDF + textstat features -> ridge)."""
    tfidf_max_features: int = Field(5000, ge=1)
    ngram_max: int = Field(2, ge=1)
    tfidf_min_df: int = Field(2, ge=1)
    numeric_weight: float = Field(1.0, gt=0)      # scale of the standardized numeric block relative to TF-IDF rows
    alpha_log10_min: float = -2.0                 # ridge penalty grid 10**[min, max], 13 points, chosen by LOO CV
    alpha_log10_max: float = 4.0
    include_correct_answer: bool = True


class SFTConfig(_Base):
    enabled: bool = True
    epochs: int = 1
    learning_rate: float = 5.0e-5


class TrainingConfig(_Base):
    epochs: int = 3
    learning_rate: float = 5.0e-6
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    eval_every_steps: int = 500
    early_stopping_patience: int = 3
    sft: SFTConfig = SFTConfig()


class PopulationConfig(_Base):
    mu: float = 0.0
    sigma: float = Field(1.0, gt=0)


class CalibrationConfig(_Base):
    icc_models: list[Literal["1PL", "2PL"]] = ["1PL"]
    linear_recalibration: bool = True


class ReportConfig(_Base):
    n_icc_plots: int = 12
    bootstrap_n: int = Field(10_000, ge=1)        # item-bootstrap resamples for the RMSE / bias CIs
    ci_level: float = Field(0.95, gt=0, lt=1)


class Config(_Base):
    run: RunConfig = RunConfig()
    data: DataConfig = DataConfig()
    pretrained_model: PretrainedModelConfig = PretrainedModelConfig()
    lora: LoraConfigSchema = LoraConfigSchema()
    roberta: RobertaConfigSchema = RobertaConfigSchema()
    regression: RegressionConfigSchema = RegressionConfigSchema()
    features: FeaturesConfigSchema = FeaturesConfigSchema()
    training: TrainingConfig = TrainingConfig()
    population: PopulationConfig = PopulationConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    report: ReportConfig = ReportConfig()
    descriptors: dict[str, tuple[float, float]]

    @model_validator(mode="after")
    def _check_descriptors(self):
        DescriptorScale.from_config(self.descriptors)  # raises on gaps / overlaps / bad ends
        return self

    def scale(self) -> DescriptorScale:
        return DescriptorScale.from_config(self.descriptors)
