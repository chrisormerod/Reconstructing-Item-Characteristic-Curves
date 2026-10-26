# PROJECT.md — LLM-Simulated Item Characteristic Curves

**Status:** Draft 3 / design specification
**Reference:** Ormerod, C. (2026). *Reconstructing Item Characteristic Curves using Fine-Tuned Large Language Models.* arXiv:2601.02580v2
**Language:** Python ≥ 3.10
**Data:** USMLE item set in `/mnt/c/data/idm/usmle/` (text-only multiple-choice items with known difficulty)

---

## 1. Purpose

This project implements the method of arXiv:2601.02580 on the USMLE item set. A decoder LLM (Qwen3 dense series by default) is fine-tuned with LoRA so that, when prompted with a multiple-choice item and an **ability-level descriptor** (e.g. "Basic", "Proficient"), the next-token probabilities over the option tokens (`A`, `B`, `C`, …) reproduce the option probabilities a population of students at that ability level would produce. Evaluating the fine-tuned model on unseen items at each descriptor yields a **discrete Item Characteristic Curve (ICC)**, from which a 1PL item difficulty is recovered and compared with the known difficulty.

**Target construction.** Each training item has a known Rasch/1PL difficulty `b`. It is converted to Nominal Response Model (NRM) parameters under the "uniform distractor" assumption (paper eq. 4) and evaluated at the descriptor midpoints θ_k to give a target option-probability matrix.

**Loss and training.** The LLM is trained with a **weighted MSE loss between the model's option-token probabilities and the target option probabilities**, using a hand-written PyTorch training loop (explicit optimizer, scheduler, gradient clipping, early stopping), not the Hugging Face `Trainer`.

**Scope.** The data contains no student responses or student abilities, so the project has no response-based NRM calibration. The only target mode is known-`b`. Images are not supported; all items are text.

---

## 2. Method Summary (implementation-relevant)

### 2.1 Ability descriptors → θ_k and ω_k

Descriptors `L_1 … L_N` have bounds `(c_{k-1}, c_k]` with `c_0 = -∞`, `c_N = +∞`. Assuming θ ~ N(μ, ς²) and writing α_k = (c_k − μ)/ς:

- Conditional mean (midpoint used for each descriptor):
  θ_k = μ + ς · [φ(α_{k-1}) − φ(α_k)] / [Φ(α_k) − Φ(α_{k-1})]
- Descriptor weight:
  ω_k = Φ(α_k) − Φ(α_{k-1})

μ = 0 and ς = 1 (student abilities are assumed N(0, 1); the data has none to estimate from). They are set in config.

### 2.2 NRM

For an item with n options and slope/location parameters (a_i, b_i):

P(y = v_i | θ) = softmax(a_1(θ − b_1), …, a_n(θ − b_n))_i

### 2.3 IRT → NRM conversion (paper eq. 4, inverted)

With the correct option at index c and all distractors equiprobable (a_i = b_i = 0 for i ≠ c):

- a_c = a
- b_c = b − log(n − 1) / a      (1PL: a = 1)

Items have between 4 and 10 options, so n varies per item.

### 2.4 Fine-tuning

Two stages (paper §3.2):

1. **Stage 1 — SFT (optional, on by default):** cross-entropy on a single sampled answer token per (item, descriptor), answer sampled from the target distribution; 1 epoch, lr 5e-5, batch size 1, linear schedule, gradient clipping.
2. **Stage 2 — Distribution correction (core):** prompt truncated at the answer slot; restricted softmax over the option-token logits only; loss is

   L = Σ_k ω_k · Σ_i ( p̂_ik − P_ik )²

   lr 5e-6, linear schedule, gradient clipping, early stopping on the ω-weighted dev loss. Each item appears once per descriptor per epoch (N exposures, N = number of descriptors: 21 in the default config, 20 in the article).

### 2.5 Recovering item parameters

For each held-out item, evaluate the model at all N descriptors to obtain the discrete ICC (P_j1 … P_jN) for the correct option, then fit (paper eq. 11):

b_j = argmin Σ_k ω_k (P_jk − σ(θ_k − b_j))²

(1PL, a_j fixed at 1; `ICCFitter` also supports a free a_j for 2PL, but the data has no reference `a`.) A linear post-hoc recalibration fitted on the dev set maps the predicted b onto the original `Difficulty` scale, which also corrects regression toward the mean. It is applied to test predictions.

### 2.6 Errata / ambiguities in the preprint (handled explicitly in code)

- Table 2 lists "Critical" as (∞, −3); this should be (−∞, −3).
- Table 2 has a gap: "Minimal" ends at −1.8 and "Emerging" starts at −1.5. The config validator **rejects gaps and overlaps**; the default config closes the gap by extending Emerging down to −1.8 (see §5 and §12).
- Φ(x) is written as ∫₀ˣ; it should be ∫₋∞ˣ.
- Eq. 9 writes P(c_{k-1} < θ ≤ c_k | y = 1); the intended quantity is P(y = 1 | c_{k-1} < θ ≤ c_k).
- Eq. 12 and the vector after eq. 14 have index typos; the intended vector is (P_j1, …, P_jN) over descriptors for the correct option.
- The Appendix A prompt asks for a free-text "Student Answer". For token-probability training, the prompt must present lettered options and the answer slot must be a single option token.

---

## 3. Directory Structure

```
llm-icc/
├── PROJECT.md
├── README.md
├── pyproject.toml
├── .gitignore                     # ignores results/, logs/, checkpoints, caches
├── main.py                        # thin entry point → llm_icc.app.main()
│
├── config/                        # Configuration files (YAML)
│   ├── default.yaml
│   ├── usmle_qwen3_{1_7b,4b,8b,14b,32b}.yaml   # 32b: quantization 4bit
│   ├── usmle_qwen3_14b_nf4.yaml   # 14B in 4-bit, to isolate the effect of quantization
│   ├── usmle_llama31_8b.yaml      # pins chat_template_kwargs.date_string
│   ├── usmle_roberta_large.yaml   # Maeda (2023) variant
│   ├── usmle_modernbert_regression.yaml   # run.method: regression
│   └── usmle_features.yaml        # run.method: features
│
├── llm_icc/
│   ├── __init__.py
│   ├── app.py                     # Pipeline orchestration (parse → targets → train → predict → report)
│   │
│   ├── config/                    # Config schema + loading (validates YAML)
│   │   ├── __init__.py
│   │   ├── schema.py              # dataclasses / pydantic models for every config section
│   │   └── loader.py              # YAML load, CLI overrides, validation
│   │
│   ├── data/                      # Data structures (classes) facilitating modeling
│   │   ├── __init__.py
│   │   ├── descriptors.py         # AbilityDescriptor, DescriptorScale
│   │   ├── options.py             # Option
│   │   ├── parameters.py          # ItemParameters (ABC), NRMParameters, LogisticParameters
│   │   ├── item.py                # Item
│   │   ├── item_bank.py           # ItemBank (collection + splits)
│   │   └── torch_datasets.py      # SFTDataset, DistributionDataset, collators
│   │
│   ├── modeling/                  # LoRA/RoBERTa simulators, regression baselines, fine-tuning
│   │   ├── __init__.py
│   │   ├── base.py                # ResponseSimulator (ABC)
│   │   ├── lora_simulator.py      # LoRAResponseSimulator (HF + PEFT wrapper; optional bitsandbytes 4/8-bit)
│   │   ├── roberta_simulator.py   # RobertaResponseSimulator (vocabulary-ablation ability, Maeda 2023)
│   │   ├── regression_baseline.py # RegressionModel: ModernBERT + one-output head, MSE on b
│   │   ├── features_baseline.py   # FeaturesModel: TF-IDF + textstat features -> ridge on b
│   │   ├── prompts.py             # PromptBuilder (Appendix A templates, chat template)
│   │   ├── option_tokens.py       # OptionTokenizer: option key ↔ single token id
│   │   ├── losses.py              # weighted_distribution_mse, sft_cross_entropy
│   │   ├── training_loop.py       # DistributionTrainer, SFTTrainer, RegressionTrainer (manual loops)
│   │   └── early_stopping.py      # EarlyStopping, checkpoint selection
│   │
│   ├── irt/                       # All IRT-related tools
│   │   ├── __init__.py
│   │   ├── functions.py           # sigmoid, logistic_2pl, nrm_probabilities
│   │   ├── ability.py             # truncated-normal θ_k, ω_k
│   │   ├── conversions.py         # logistic ↔ NRM (eq. 4), difficulty-scale transform
│   │   ├── icc_fit.py             # ICCFitter: 1PL (and 2PL) from discrete ICC (eq. 11)
│   │   └── recalibration.py       # LinearRecalibrator (dev-set mean-regression fix)
│   │
│   └── utils/                     # Parsing, system, CLI, results, reporting
│       ├── __init__.py
│       ├── usmle.py               # USMLEParser: configurable tabular files (USMLE layout by default) → ItemBank
│       ├── system.py              # seeding, device/dtype, GPU memory, git hash, run dirs
│       ├── cli.py                 # argparse definition
│       ├── logging_utils.py       # console + file logging, optional W&B/TensorBoard
│       ├── metrics.py             # Pearson, RMSE, bias, bootstrap_ci (percentile), weighted MSE
│       ├── results.py             # write/read parameter tables, ICC tables, predictions
│       └── reporting.py           # ICC plots (Fig. 3 style), summary tables
│
├── scripts/                       # Other tools
│   ├── simulate_usmle.py          # synthetic USMLE-format spreadsheets (known b) for end-to-end tests
│   ├── check_option_tokens.py     # verify option keys A–J are single tokens for a tokenizer
│   ├── summarize_runs.py          # aggregate metrics across runs/folds
│   ├── repeat_runs.py             # one config over several seeds -> mean/SD/min/max table
│   └── run_all.py                 # every Table 4 model x seeds on one GPU, --preflight, results/table4.{csv,md,tex}
│
├── tests/
│   ├── test_descriptors.py
│   ├── test_conversions.py
│   ├── test_icc_fit.py
│   ├── test_losses.py
│   ├── test_prompts.py
│   ├── test_metrics.py            # bias, bootstrap CI
│   ├── test_features_baseline.py
│   ├── test_regression_baseline.py
│   ├── test_roberta_simulator.py
│   ├── test_training_loop.py
│   └── test_usmle_parser.py
│
├── docs/                          # Documentation
│   ├── method.md                  # math derivations + errata (§2 expanded)
│   ├── usmle_format.md            # input format specification (§8)
│   ├── config_reference.md
│   └── experiments.md             # how to reproduce the USMLE results
│
├── results/                       # NOT tracked in git
└── logs/                          # NOT tracked in git
```

---

## 4. Data Structures and Abstract Classes

Shapes use: `n` = number of options for an item, `N` = number of descriptors, `B` = batch size, `n_max` = max options in a batch.

### 4.1 Descriptors (`llm_icc/data/descriptors.py`)

```python
@dataclass(frozen=True)
class AbilityDescriptor:
    label: str
    lower: float          # c_{k-1}, may be -inf
    upper: float          # c_k, may be +inf

class DescriptorScale:
    """Ordered, contiguous, non-overlapping set of descriptors."""
    def __init__(self, descriptors: Sequence[AbilityDescriptor]): ...  # validates order & contiguity
    @classmethod
    def from_config(cls, mapping: Mapping[str, tuple[float, float]]) -> "DescriptorScale": ...
    labels: list[str]
    def __len__(self) -> int: ...
    def label_for(self, theta: float) -> int: ...                       # index k
    def assign(self, thetas: np.ndarray) -> np.ndarray: ...             # vectorized
    def conditional_means(self, mu: float, sigma: float) -> np.ndarray: ...  # θ_k, shape (N,)
    def weights(self, mu: float, sigma: float) -> np.ndarray: ...            # ω_k, shape (N,)
```

### 4.2 Options (`llm_icc/data/options.py`)

```python
@dataclass
class Option:
    key: str              # display key, "A".."J"; must map to one token
    text: str
    is_correct: bool
```

### 4.3 Parameters (`llm_icc/data/parameters.py`)

```python
class ItemParameters(ABC):
    @abstractmethod
    def option_probabilities(self, theta: np.ndarray) -> np.ndarray:
        """theta: (K,) → (K, n) probabilities over options."""
    @abstractmethod
    def to_dict(self) -> dict: ...

@dataclass
class NRMParameters(ItemParameters):
    a: np.ndarray         # (n,)
    b: np.ndarray         # (n,)
    def logits(self, theta: np.ndarray) -> np.ndarray: ...       # (K, n)
    def option_probabilities(self, theta): ...                   # softmax(logits)

@dataclass
class LogisticParameters(ItemParameters):
    b: float
    a: float = 1.0
    model: Literal["1PL", "2PL"] = "1PL"
    def p_correct(self, theta: np.ndarray) -> np.ndarray: ...    # σ(a(θ−b))
    def to_nrm(self, n_options: int, correct_index: int) -> NRMParameters: ...   # eq. 4
    def option_probabilities(self, theta): ...                   # via to_nrm
```

`NRMParameters` is only an intermediate: it is produced from `LogisticParameters` and is not calibrated from data.

### 4.4 Items (`llm_icc/data/item.py`)

```python
@dataclass
class Item:
    item_id: str                      # str(ItemNum)
    stem: str                         # ItemStem_Text
    options: list[Option]             # 4-10 options, keys contiguous from "A"
    parameters: LogisticParameters | None            # b on the model scale (after the difficulty transform)
    reference_difficulty: float | None               # Difficulty as given in the data (original scale)
    metadata: dict                    # exam step, item type (Text/PIX), source row

    @property
    def n_options(self) -> int: ...
    @property
    def correct_index(self) -> int: ...
    @property
    def correct_option(self) -> Option: ...
    def is_ready(self) -> bool: ...   # parameters exist, so targets can be produced

    def target_distribution(self, scale: DescriptorScale, mu: float, sigma: float) -> np.ndarray:
        """(N, n) target option probabilities at θ_k via eq. 4. Raises if not ready."""
        nrm = self.parameters.to_nrm(self.n_options, self.correct_index)
        return nrm.option_probabilities(scale.conditional_means(mu, sigma))
```

### 4.5 Item bank and splits (`llm_icc/data/item_bank.py`)

```python
class ItemBank:
    items: dict[str, Item]
    def __iter__(self): ...
    def filter(self, predicate) -> "ItemBank": ...
    def split(self, strategy: Literal["predefined", "kfold"], **kw) -> list["Split"]: ...

@dataclass
class Split:
    name: str                         # e.g. "predefined" or "fold_0"
    train: list[str]; dev: list[str]; test: list[str]
```

The official test set is always the test items. `predefined` takes `dev_fraction` of the training items as dev (seeded). `kfold` partitions the training items into train/dev folds, with the same test set in every fold.

### 4.6 Torch datasets (`llm_icc/data/torch_datasets.py`)

Each training example is one (item, descriptor) pair.

```python
@dataclass
class DistributionExample:
    item_id: str
    descriptor_index: int
    input_ids: list[int]              # prompt truncated at answer slot
    option_token_ids: list[int]       # length n
    target: np.ndarray                # (n,)
    weight: float                     # ω_k

class DistributionDataset(torch.utils.data.Dataset): ...   # yields DistributionExample
class SFTDataset(torch.utils.data.Dataset): ...            # resamples one answer per (item, k) each epoch
def distribution_collate(batch, pad_id) -> dict[str, Tensor]:
    # input_ids (B, T), attention_mask (B, T), option_token_ids (B, n_max),
    # option_mask (B, n_max), target (B, n_max), weight (B,)
```

---

## 5. Configuration

One YAML file per experiment, validated by `llm_icc/config/schema.py`. Any key can be overridden from the CLI (`--set training.learning_rate=1e-5`).

```yaml
# config/default.yaml
run:
  name: default
  seed: 1234
  method: icc                  # icc | regression (ModernBERT) | features (TF-IDF + textstat -> ridge)
  results_dir: results/
  logs_dir: logs/

data:
  dir: /mnt/c/data/idm/usmle           # `usmle_dir` is a deprecated alias
  files:   {train: train_final.xlsx, test: test_final.xlsx, gold: gold_final.xlsx}   # .xlsx/.csv/.tsv/.parquet; gold: null = test file has difficulty
  columns: {id: ItemNum, stem: ItemStem_Text, option_prefix: "Answer__", key: Answer_Key, key_text: Answer_Text,
            item_type: ItemType, exam: EXAM, difficulty: Difficulty}   # key_text, item_type, exam may be null
  splits:
    strategy: predefined       # predefined | kfold
    k: 5                       # kfold only
    dev_fraction: 0.1          # fraction of the training items held out as dev
  exclude_pix: false           # true drops ItemType == PIX items (stems that refer to a missing picture)
  difficulty_transform:        # provisional Difficulty → training-target b; see §12
    method: standardize        # standardize | linear
    shift: 0.0                 # linear only: b = scale * (Difficulty - shift)
    scale: 1.0                 # standardize: shift/scale are the mean/SD of the training items' Difficulty

pretrained_model:
  architecture: causal_lm      # causal_lm (LoRA LLM) | roberta_mc
  name_or_path: Qwen/Qwen3-4B  # sized for a single 24-48 GB GPU
  dtype: bfloat16              # bfloat16 | float16 (compute dtype when quantized)
  quantization: none           # none | 8bit | 4bit (bitsandbytes QLoRA; CUDA only; 4bit = NF4 + double quant)
  max_length: 2048
  gradient_checkpointing: true
  enable_thinking: false       # Qwen3 chat-template flag
  chat_template_kwargs: {}     # extra template variables, e.g. Llama 3.1 date_string
  attn_implementation: sdpa
  predict_batch_size: 10       # descriptors per forward pass at prediction time

lora:
  r: 32
  alpha: 64
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj]
  bias: none

training:
  epochs: 3                    # maximum stage-2 epochs; early stopping may end sooner
  learning_rate: 5.0e-6        # stage 2 (distribution correction)
  batch_size: 1
  gradient_accumulation_steps: 8
  weight_decay: 0.0
  warmup_ratio: 0.03
  max_grad_norm: 1.0
  eval_every_steps: 500
  early_stopping_patience: 3
  sft:
    enabled: true
    epochs: 1
    learning_rate: 5.0e-5

population:                    # θ ~ N(mu, sigma²); assumed standard normal
  mu: 0.0
  sigma: 1.0

calibration:
  icc_models: [1PL]            # 2PL is supported by ICCFitter but has no reference `a` in the data
  linear_recalibration: true

report:
  n_icc_plots: 12
  bootstrap_n: 10000           # percentile-bootstrap resamples over test items for the RMSE / bias CIs
  ci_level: 0.95

roberta: {ablation_samples: 10, protect_special_tokens: true, include_correct_answer: true}   # roberta_mc only
regression: {max_length: 1024, include_correct_answer: true}                                   # run.method: regression
features:                      # run.method: features
  tfidf_max_features: 5000
  ngram_max: 2
  tfidf_min_df: 2
  numeric_weight: 1.0          # scale of the standardized textstat/option block relative to TF-IDF
  alpha_log10_min: -2.0        # ridge penalty grid 10**[min, max], 13 points, chosen by leave-one-out CV
  alpha_log10_max: 4.0
  include_correct_answer: true

descriptors:                   # ordered: lowest → highest; [lower, upper]
  Critical:          [-.inf, -3.0]
  Severely Limited:  [-3.0, -2.7]
  Deficient:         [-2.7, -2.4]
  Inadequate:        [-2.4, -2.1]
  Minimal:           [-2.1, -1.8]
  Rudimentary:       [-1.8, -1.5]   # paper leaves (-1.8, -1.5) unlabeled; bridged with this label (21 descriptors)
  Emerging:          [-1.5, -1.2]
  Developing:        [-1.2, -0.9]
  Approaching Basic: [-0.9, -0.6]
  Basic:             [-0.6, -0.3]
  Functional:        [-0.3, 0.0]
  Satisfactory:      [0.0, 0.3]
  Competent:         [0.3, 0.6]
  Proficient:        [0.6, 0.9]
  Accomplished:      [0.9, 1.2]
  Advanced:          [1.2, 1.5]
  Superior:          [1.5, 1.8]
  Exceptional:       [1.8, 2.1]
  Outstanding:       [2.1, 2.4]
  Distinguished:     [2.4, 2.7]
  Exemplary:         [2.7, .inf]
```

Per-experiment files set `base: default.yaml` and override only what differs. Other sections not shown (`lora` for `roberta_mc`, `training.sft`) keep their defaults. Validation rules: descriptors must be in ascending order, contiguous (each `lower` equals the previous `upper`), first `lower = -inf`, last `upper = +inf`, labels unique. YAML mapping order is preserved and is significant.

---

## 6. Modeling (`llm_icc/modeling/`)

### 6.1 Abstract simulator

```python
class ResponseSimulator(ABC):
    """Wraps a pretrained LM; exposes option-level probabilities, not text."""

    @abstractmethod
    def option_logits(self, batch: dict[str, Tensor]) -> Tensor:
        """(B, n_max) logits at the answer slot, gathered at option_token_ids; masked = -inf."""

    def option_probabilities(self, batch) -> Tensor:
        """Restricted softmax over option tokens (paper eq. 16, LHS)."""
        return torch.softmax(self.option_logits(batch), dim=-1)

    @torch.no_grad()
    def predict_icc(self, item: Item, scale: DescriptorScale) -> np.ndarray:
        """(N, n) option probabilities for one item at every descriptor."""

    @abstractmethod
    def trainable_parameters(self) -> Iterable[nn.Parameter]: ...
    @abstractmethod
    def save(self, path: Path) -> None: ...
    @classmethod
    @abstractmethod
    def load(cls, path: Path, config) -> "ResponseSimulator": ...
```

### 6.2 `LoRAResponseSimulator`

Wraps `AutoModelForCausalLM` + `peft.LoraConfig`/`get_peft_model`. Responsibilities: load base model in configured dtype; attach adapters to `target_modules`; run a forward pass and take logits at the last non-pad position of each prompt; gather logits at the option token ids; expose `option_probabilities`, `predict_icc`, and adapter save/load. Only adapter weights are saved. Batches are left-padded so the answer slot is always the last position, which lets the LM head run on that one position only (`logits_to_keep=1`).

With `pretrained_model.quantization: 4bit|8bit` the base model is loaded through bitsandbytes (`BitsAndBytesConfig`: NF4 with double quantization and `dtype` as compute type for 4-bit) with `device_map={"": 0}`, prepared with `prepare_model_for_kbit_training` (fp32 norms/embeddings, input grads, gradient checkpointing) and not moved with `.to(device)`; LoRA adapters stay 16/32-bit (QLoRA). This is what makes Qwen3-32B trainable on one 48 GB GPU (~61 GiB in bf16, ~24 GiB in NF4). It needs CUDA and the `quant` extra.

### 6.2a Other models behind the same interfaces

- **`RobertaResponseSimulator`** (`architecture: roberta_mc`): `FacebookAI/roberta-large` as a multiple-choice model, fully fine-tuned. Ability is simulated by zeroing the word embeddings of a random share of the vocabulary: descriptor k keeps a fraction Φ(θ_k); the ICC averages `roberta.ablation_samples` ablations (Maeda 2023, arXiv:2310.11655).
- **`RegressionModel`** (`run.method: regression`): ModernBERT with a one-output head fine-tuned with MSE on `b` by `RegressionTrainer`; it has no ICC and is scored only through the predicted Difficulty.
- **`FeaturesModel`** (`run.method: features`): no neural network. Text = stem + lettered options (+ key). TF-IDF (1–2-grams, sublinear tf) is stacked with standardized numeric features: textstat readability (Flesch ease, Flesch-Kincaid, Gunning fog, SMOG, ARI, Coleman-Liau, Dale-Chall), counts (syllables, words, sentences, characters, difficult words), words per sentence and syllables per word of the stem, and option statistics (number of options, mean/max option length, correct-option length and its difference from the distractor mean, option–stem word overlap, key position). `RidgeCV` picks the penalty by efficient LOO CV on the training items; the dev set is left for the shared recalibration. CPU only.

All three produce a per-item `b_pred`, which `score_item_parameters` maps to Difficulty, recalibrates on dev, and scores exactly as for ICC models.

### 6.3 Prompts and option tokens

`PromptBuilder` renders the Appendix A system and user prompts via the tokenizer's chat template, with three changes needed for token-probability training: the options are listed with keys (`A.`, `B.`, …); the "correct answer" field is shown as `Key. text`; and the assistant turn is pre-filled up to the answer slot (e.g. `**Student Answer:** `) so the next token is the option key. `enable_thinking=False` is passed for Qwen3, and `pretrained_model.chat_template_kwargs` is forwarded to the template (Llama 3.1 injects "Today Date" into the system prompt, so its config pins `date_string`). The stem is `ItemStem_Text` and each option is the `Answer__X` text. The `PIX` flag is not shown to the model. The Appendix A text was not available, so the prompts in `prompts.py` follow the described structure but are not verbatim.

`OptionTokenizer` maps option keys to token ids and **asserts each key is a single token** in the context it appears (with/without a leading space). `scripts/check_option_tokens.py` runs this check for a given tokenizer. Keys A–J must be supported (the data has up to 10 options).

### 6.4 Losses (`losses.py`)

```python
def weighted_distribution_mse(pred: Tensor, target: Tensor, option_mask: Tensor, weight: Tensor) -> Tensor:
    """Σ_b weight_b · Σ_i mask_bi · (pred_bi − target_bi)²  /  Σ_b weight_b"""

def sft_cross_entropy(logits: Tensor, labels: Tensor) -> Tensor: ...
```

Loss is computed in float32 even when the model runs in bf16.

### 6.5 Training loop (`training_loop.py`)

Manual loop, no HF `Trainer`:

```python
class DistributionTrainer:
    def __init__(self, simulator, train_ds, dev_ds, config, logger): ...
    def build_optimizer(self) -> torch.optim.AdamW: ...
    def build_scheduler(self, optimizer, num_training_steps) -> LambdaLR: ...   # linear w/ warmup
    def train(self) -> TrainingResult: ...
    def evaluate(self, ds) -> dict[str, float]: ...      # ω-weighted dev MSE (early-stop metric)
```

Each step: forward → restricted softmax → `weighted_distribution_mse` → backward with gradient accumulation → `clip_grad_norm_(max_grad_norm)` → `optimizer.step()` → `scheduler.step()` → `zero_grad()`. Every `eval_every_steps`, evaluate on dev; `EarlyStopping` keeps the best adapter checkpoint. `SFTTrainer` shares the same structure with cross-entropy loss and re-samples answers each epoch.

---

## 7. IRT (`llm_icc/irt/`)

| Module | Key API | Paper ref |
|--------|---------|-----------|
| `functions.py` | `sigmoid`, `logistic_2pl(theta, a, b)`, `nrm_probabilities(theta, a, b)` | eqs. 1, 3 |
| `ability.py` | `conditional_means(scale, mu, sigma)`, `descriptor_weights(...)` | §3.1 |
| `conversions.py` | `logistic_to_nrm(params, n, correct)`, `nrm_to_logistic_uniform(nrm, correct)`, `DifficultyTransform.fit(train_difficulty)`, `.to_b(d)`, `.to_difficulty(b)` | eq. 4 |
| `icc_fit.py` | `ICCFitter.fit(p_correct, theta_k, omega_k, model="1PL") -> LogisticParameters` | eq. 11 |
| `recalibration.py` | `LinearRecalibrator.fit(pred_dev, true_dev)`, `.transform(pred_test)` | §3.2 end |

`DifficultyTransform` gives the training targets a provisional scale: with `standardize`, `b = (Difficulty − mean)/SD`, using the training items only. This is reasonable because `Difficulty` is approximately normal and abilities are N(0, 1). The final linear map between the model's b scale and `Difficulty` is determined on the dev set: `LinearRecalibrator` regresses the reference `Difficulty` on the predicted `b` over the dev items, and applies the fitted line to test predictions. Metrics are therefore on the original `Difficulty` scale. If recalibration is switched off, `DifficultyTransform.to_difficulty` inverts the provisional map instead.

---

## 8. Utilities (`llm_icc/utils/`)

- **`usmle.py` — `USMLEParser`:** reads the train, test and (optional) gold files and produces an `ItemBank` of `Item` objects. File names (`data.files`) and column names (`data.columns`) are configurable, and `.xlsx`, `.csv`, `.tsv` and `.parquet` are read by extension; the defaults below are the USMLE layout. A missing mapped column raises an error naming it. It sits behind an `ItemSource` ABC so other formats can be added without touching the pipeline. The format is documented in `docs/usmle_format.md`.

  **USMLE format** (`/mnt/c/data/idm/usmle/`, one sheet per file):

  | File | Rows | Columns | Role |
  |------|------|---------|------|
  | `train_final.xlsx` | 466 | `ItemNum, ItemStem_Text, Answer__A … Answer__J, Answer_Key, Answer_Text, ItemType, EXAM, Difficulty` (and an unused `Response_Time`) | Training and dev items, with difficulty |
  | `test_final.xlsx` | 201 | same as train, without `Difficulty` and `Response_Time` | Test item content |
  | `gold_final.xlsx` | 201 | `ItemNum, Difficulty` (and an unused `Response_Time`) | Test difficulties, joined to `test_final` on `ItemNum` (the id sets are identical) |

  Parsing rules, checked against the files:
  - `ItemNum` is unique within and across files (train and test do not overlap) and becomes `item_id`.
  - Options are the non-empty `Answer__A … Answer__J` cells. They are contiguous from A. Train items have 4–10 options (mostly 5: 366 of 466); test items have 4–8 (159 of 201 have 5).
  - `Answer_Key` is a letter, always non-empty and always pointing to a non-empty option. `Answer_Text` repeats the correct option's text, and the parser asserts they agree. They match in 464 of 466 train rows; the 2 mismatches are logged and `Answer_Key` wins.
  - `ItemType` is `Text` or `PIX` (50 train, 22 test). PIX stems refer to a picture that is not provided. They stay in by default and can be dropped with `data.exclude_pix`. `EXAM` is `STEP 1`, `STEP 2` or `STEP 3`. Both are stored in `item.metadata`.
  - `Difficulty` is stored in `reference_difficulty` and converted to the provisional training scale with `data.difficulty_transform`. It ranges 0.02–1.38 in train (mean 0.48) and 0.10–1.36 in gold (mean 0.50). Higher means harder.
  - `Response_Time` is not used anywhere in this repo. The parser does not read it.
  - Read the sheets with `openpyxl`. Ignore any `~$*.xlsx` Excel lock files.
- **`system.py`:** `set_seed`, device/dtype selection, GPU memory reporting, git commit hash capture, run-directory creation (`results/<run.name>/<timestamp>/`).
- **`cli.py`:** argument parsing (§9).
- **`logging_utils.py`:** console + rotating file logs in `logs/`; optional TensorBoard/W&B.
- **`metrics.py`:** Pearson r, RMSE, `bias` (mean predicted − true), `bootstrap_ci` and weighted MSE, computed on the original `Difficulty` scale. `bootstrap_ci(pred, ref, stat, n_boot, level, seed)` resamples (pred, ref) pairs, i.e. items, with replacement (vectorized; `stat` is `rmse` or `bias`) and returns percentile bounds. The same seed gives the same resampled indices for every statistic. Models are not retrained, so the interval reflects test-item sampling only.
- **`results.py`:** writes `targets.parquet`, `predicted_icc.parquet`, `item_parameters.csv`, `metrics.json`, and a frozen copy of the resolved config.
- **`reporting.py`:** per-item ICC plots overlaying target and LLM curves (paper Fig. 3); summary tables across folds/models.

---

## 9. Command-Line Interface

```
python main.py --config config/default.yaml [options]

  --config PATH            YAML config (required)
  --stage STAGE            parse | targets | sft | train | predict | evaluate | all  (default: all)
  --fold INT               run a single fold (default: all folds)
  --resume PATH            resume from an adapter checkpoint
  --set KEY=VALUE          override any config key (repeatable)
  --dry-run                parse + build targets + one forward pass, no training
```

Sweeps are driven by two scripts that call `main.py` once per (model, seed) as subprocesses (so GPU memory is released between runs), with `--set run.seed=S --set run.name=<name>_seed<S>`:

```
python scripts/repeat_runs.py CONFIG [--seeds 1 2 3 4 5] [--aggregate-only]   # one model; mean/SD/min/max -> results/<name>_repeats.csv
python scripts/run_all.py [--models ...] [--seeds ...] [--preflight|--aggregate-only|--force|--list]
```

`run_all.py` holds the model registry (Table 4 rows, config, expected GPU memory), skips finished (model, seed) pairs, continues past failures (`results/run_all_status.json`, non-zero exit) and writes `results/table4.{csv,md,tex}` plus `table4_runs.csv`. In the table, Pearson, RMSE and bias are mean (SD) over seeds; the RMSE CI is the mean of the per-seed bootstrap bounds; ELECTRA and the Li et al. ensemble are the published RMSEs. `--preflight` runs `--dry-run` per model and prints the measured post-load GPU memory.

### Pipeline (`llm_icc/app.py`)

1. **Setup:** load and validate config; seed; create run directory; log environment and git hash.
2. **Parse:** `USMLEParser(config.data.dir, config.data.files, config.data.columns).load()` → `ItemBank`; report counts, option-count distribution, PIX items, and exam steps.
3. **Build targets:** convert each item's `b` to NRM (eq. 4); compute θ_k, ω_k from `population`; materialize (N, n) targets per item; save to results.
4. **Split:** build the predefined split or k-fold splits.
5. **Per split:**
   a. Build the simulator selected by `pretrained_model.architecture` (`LoRAResponseSimulator` or `RobertaResponseSimulator`). For `run.method: regression|features` a separate runner (`run_regression_split`, `run_features_split`) fits a direct regressor on `b` instead and skips steps b–d.
   b. Stage 1 SFT (if enabled).
   c. Stage 2 distribution training with MSE and early stopping on dev.
   d. Predict discrete ICCs for dev and test items.
   e. Fit 1PL `b` from the predicted ICCs (eq. 11); fit `LinearRecalibrator` on dev; apply to test.
   f. Compute metrics against the reference difficulty (gold for test) in `score_item_parameters`: Pearson, RMSE, bias, 95% bootstrap CIs for RMSE, bias and the train-mean baseline RMSE (`report.bootstrap_n`, `report.ci_level`, seeded by `run.seed`).
6. **Report:** aggregate across folds; plots and tables.

---

## 10. Dependencies

`torch`, `transformers`, `peft`, `accelerate` (device placement only), `numpy`, `scipy`, `pandas`, `pyarrow`, `openpyxl`, `pyyaml`, `pydantic`, `matplotlib`, `tqdm`, `pytest`, `scikit-learn` and `textstat` (features baseline). Optional extras: `tracking` (`tensorboard`, `wandb`) and `quant` (`bitsandbytes`, for 4/8-bit loading).

---

## 11. Testing Strategy

- **Unit:** θ_k/ω_k against numerical integration; eq. 4 round-trip (logistic → NRM → p_correct equals σ(a(θ−b))) for n = 4…10; `ICCFitter` recovers a known `b` from noiseless curves; loss masking with mixed option counts; descriptor validator rejects gaps/overlaps; option keys A–J are single tokens; `USMLEParser` handles variable option counts and the gold join.
- **Baselines and metrics:** bias of a constant offset; bootstrap CI is degenerate for a constant error, brackets the point estimate, is reproducible for a seed, widens with the level, and is NaN for n < 2; the features model beats the mean on synthetic items with a known signal and handles empty stems; the regression baseline and RoBERTa simulator train on tiny random models and round-trip checkpoints; prompt kwargs are forwarded to the chat template.
- **Known failing tests:** `test_descriptors.py` (3) and `test_training_loop.py::test_sft_dataset_samples_from_target` assume 20 descriptors; the default config now has 21 (`Rudimentary`).
- **Integration:** `scripts/simulate_usmle.py` builds synthetic spreadsheets in the USMLE format with known `b`; a `--dry-run` and a tiny-model run (e.g. Qwen3-0.6B, few steps) execute end-to-end in CI.

