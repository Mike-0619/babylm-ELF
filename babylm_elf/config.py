from __future__ import annotations

from dataclasses import MISSING, dataclass, field, replace
import json
import math
from pathlib import Path
from typing import Any, TypeAlias, get_type_hints

import yaml

from babylm_elf.modules.model import BabyLMELFConfig
from babylm_elf.training.optim import resolve_ema_decay


@dataclass(frozen=True)
class DataConfig:
    source: str = "local_text"
    hf_dataset: str | None = None
    hf_config: str | None = None
    hf_revision: str | None = None
    hf_train_split: str = "train"
    hf_text_field: str = "text"
    train_text: str | None = None
    train_path: str = "data/text_data/train_100M_tokenized.bin"
    tokenizer_path: str = "data/tokenizer_100M.json"
    tokenizer_vocab_size: int = 16_384
    manifest_path: str | None = None
    seq_length: int = 1_024
    num_workers: int = 0
    train_word_count: int | None = None


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_steps: int = 0
    log_every: int = 50
    latest_every_steps: int = 1_000
    checkpoint_by_words: bool = False
    checkpoint_word_limit: int | None = None
    word_exposure_offset: int = 0
    encoder_freeze_steps_ratio: float = 0.0
    precision: str = "bf16"
    device: str = "auto"


@dataclass(frozen=True)
class OptimizerConfig:
    type: str = "adamw"
    learning_rate: float = 1.25e-4
    aux_learning_rate: float | None = None
    encoder_lr_multiplier: float = 1.0
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1.0e-8
    max_grad_norm: float = 1.0


@dataclass(frozen=True)
class SchedulerConfig:
    type: str = "constant"
    warmup_steps: int | None = None
    warmup_epochs: float | None = 0.5
    min_lr: float = 0.0


@dataclass(frozen=True)
class EMAConfig:
    reference_decay: float = 0.9999
    reference_steps: int = 95_000
    warmup: bool = True
    scale_to_run: bool = True


@dataclass(frozen=True)
class EncoderOptimizerConfig(OptimizerConfig):
    """Legacy encoder optimization layout retained for the encoder CLI."""

    warmup_steps: int = -1
    warmup_epochs: float | None = 0.5
    schedule: str = "constant"
    min_lr: float = 0.0
    ema_reference_decay: float = 0.9999
    ema_reference_steps: int = 95_000
    ema_warmup: bool = True
    ema_scale_to_run: bool = True


@dataclass(frozen=True)
class TargetConfig:
    special_token_count: int = 16
    filter_empty_control: bool = False
    excluded_token_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class StepCorruptionConfig:
    type: str = "step10_step20"
    seed: int = 0
    segment_boundary_token_id: int = 1


@dataclass(frozen=True)
class BertCorruptionConfig:
    type: str = "bert_15_80_10_10"
    target_probability: float = 0.15
    mask_probability: float = 0.80
    random_probability: float = 0.10
    unchanged_probability: float = 0.10


MLMCorruptionConfig: TypeAlias = StepCorruptionConfig | BertCorruptionConfig


@dataclass(frozen=True)
class ELFObjectiveBase:
    type: str
    targets: TargetConfig = field(default_factory=TargetConfig)
    time_schedule: str = "logit_normal"
    denoiser_p_mean: float = -1.5
    denoiser_p_std: float = 0.8
    denoiser_noise_scale: float = 2.0
    decoder_probability: float = 0.2
    self_condition_probability: float = 0.5
    self_condition_cfg_min: float = 0.5
    self_condition_cfg_max: float = 5.0
    t_eps: float = 0.05

    @property
    def decoder_objective(self) -> str:
        return {
            "elf_noisy_ce": "official_noisy_ce",
            "elf_token_mlm": "token_mlm",
        }[self.type]


@dataclass(frozen=True)
class NoisyCEObjectiveConfig(ELFObjectiveBase):
    type: str = "elf_noisy_ce"
    decoder_noise_scale: float = 5.0
    decoder_p_mean: float = 0.8
    decoder_p_std: float = 0.8
    fixed_gaussian_seed: int = 0
    fixed_gaussian_scale: float = 5.0


@dataclass(frozen=True)
class TokenMLMObjectiveConfig(ELFObjectiveBase):
    type: str = "elf_token_mlm"
    corruption: MLMCorruptionConfig = field(default_factory=StepCorruptionConfig)
    mask_latent_seed: int = 0
    mask_latent_scale: float = 1.0


@dataclass(frozen=True)
class MDLMObjectiveConfig:
    type: str = "standard_mdlm"
    targets: TargetConfig = field(default_factory=TargetConfig)
    noise_schedule: str = "loglinear"
    sampling_eps: float = 1.0e-3
    noise_eps: float = 1.0e-3
    antithetic_sampling: bool = True
    time_conditioning: str = "t"
    sampling_steps: int = 128
    mask_latent_seed: int = 0
    mask_latent_scale: float = 1.0

    @property
    def decoder_objective(self) -> str:
        return "standard_mdlm"


ObjectiveConfig: TypeAlias = (
    NoisyCEObjectiveConfig | TokenMLMObjectiveConfig | MDLMObjectiveConfig
)


@dataclass(frozen=True)
class RunConfig:
    name: str
    seed: int
    output_dir: str
    training: TrainingConfig
    model: BabyLMELFConfig
    data: DataConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    ema: EMAConfig
    objective: ObjectiveConfig


@dataclass(frozen=True)
class ResolvedRun:
    config: RunConfig
    model: BabyLMELFConfig
    objective: ObjectiveConfig
    max_steps: int
    warmup_steps: int
    ema_decay: float
    actual_train_word_count: int


@dataclass
class EncoderModelConfig:
    base_vocab_size: int = 16_384
    sentinel_start_id: int = 16_384
    sentinel_count: int = 100
    vocab_size: int = 16_484
    d_model: int = 512
    d_ff: int = 2_048
    d_kv: int = 64
    num_layers: int = 6
    num_decoder_layers: int = 6
    num_heads: int = 8
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1.0e-6
    pad_token_id: int = 3
    eos_token_id: int = 2
    decoder_start_token_id: int = 3


@dataclass
class EncoderObjectiveConfig:
    objective: str = "t5_span_corruption"
    noise_density: float = 0.15
    mean_noise_span_length: float = 3.0
    special_token_count: int = 16


@dataclass
class EncoderTrainConfig:
    name: str = "encoder"
    seed: int = 42
    output_dir: str = "outputs/2026_10M/encoder"
    epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 64
    log_every: int = 50
    mixed_precision: bool = True
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: EncoderModelConfig = field(default_factory=EncoderModelConfig)
    objective: EncoderObjectiveConfig = field(default_factory=EncoderObjectiveConfig)
    optim: EncoderOptimizerConfig = field(
        default_factory=lambda: EncoderOptimizerConfig(
            type="adamw",
            learning_rate=1.0e-3,
            warmup_epochs=0.4,
            schedule="cosine",
            ema_reference_decay=0.0,
            ema_warmup=False,
        )
    )


def load_config(path: str | Path) -> RunConfig:
    raw = _load_mapping(Path(path))
    required = {
        "name",
        "seed",
        "output_dir",
        "training",
        "model",
        "data",
        "optimizer",
        "scheduler",
        "ema",
        "objective",
    }
    _validate_keys(raw, required, required, "RunConfig")
    config = RunConfig(
        name=str(raw["name"]),
        seed=int(raw["seed"]),
        output_dir=str(raw["output_dir"]),
        training=_from_mapping(TrainingConfig, raw["training"]),
        model=_from_mapping(BabyLMELFConfig, raw["model"]),
        data=_from_mapping(DataConfig, raw["data"]),
        optimizer=_from_mapping(OptimizerConfig, raw["optimizer"]),
        scheduler=_from_mapping(SchedulerConfig, raw["scheduler"]),
        ema=_from_mapping(EMAConfig, raw["ema"]),
        objective=_load_objective(raw["objective"]),
    )
    _validate_run_config(config)
    return config


def load_encoder_config(path: str | Path) -> EncoderTrainConfig:
    return _from_mapping(EncoderTrainConfig, _load_mapping(Path(path)))


def resolve_model_config(
    config: RunConfig,
    *,
    vocab_size: int,
    pad_token_id: int,
) -> BabyLMELFConfig:
    objective = config.objective
    mask_seed = getattr(objective, "mask_latent_seed", 0)
    mask_scale = getattr(objective, "mask_latent_scale", 1.0)
    return replace(
        config.model,
        vocab_size=vocab_size,
        base_vocab_size=vocab_size,
        sentinel_start_id=vocab_size,
        encoder_vocab_size=vocab_size + config.model.sentinel_count,
        pad_token_id=pad_token_id,
        training_objective=objective.decoder_objective,
        mlm_mask_latent_seed=mask_seed,
        mlm_mask_latent_scale=mask_scale,
    )


def resolve_objective(
    objective: ObjectiveConfig,
    *,
    excluded_token_ids: set[int],
) -> ObjectiveConfig:
    targets = replace(
        objective.targets,
        excluded_token_ids=tuple(
            sorted(set(objective.targets.excluded_token_ids) | excluded_token_ids)
        ),
    )
    return replace(objective, targets=targets)


def resolve_run(
    config: RunConfig,
    *,
    model: BabyLMELFConfig,
    objective: ObjectiveConfig,
    microbatches_per_epoch: int,
    optimizer_steps_per_epoch: int,
    actual_train_word_count: int,
) -> ResolvedRun:
    training = config.training
    max_steps = training.max_steps
    if max_steps <= 0:
        total_microbatches = microbatches_per_epoch * training.epochs
        max_steps = math.ceil(
            total_microbatches / training.gradient_accumulation_steps
        )
    warmup_steps = config.scheduler.warmup_steps
    if warmup_steps is None:
        warmup_steps = int(
            optimizer_steps_per_epoch * (config.scheduler.warmup_epochs or 0.0)
        )
    ema_decay = (
        resolve_ema_decay(
            config.ema.reference_decay,
            config.ema.reference_steps,
            max_steps,
        )
        if config.ema.scale_to_run
        else config.ema.reference_decay
    )
    return ResolvedRun(
        config=config,
        model=model,
        objective=objective,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        ema_decay=ema_decay,
        actual_train_word_count=actual_train_word_count,
    )


def _load_objective(raw: dict[str, Any]) -> ObjectiveConfig:
    raw = dict(raw)
    objective_type = raw.get("type")
    targets = _from_mapping(TargetConfig, raw.pop("targets", {}))
    targets = replace(
        targets,
        excluded_token_ids=tuple(targets.excluded_token_ids),
    )
    raw["targets"] = targets
    if objective_type == "elf_noisy_ce":
        return _from_mapping(NoisyCEObjectiveConfig, raw)
    if objective_type == "elf_token_mlm":
        corruption_raw = raw.pop("corruption", {})
        corruption_type = corruption_raw.get("type", "step10_step20")
        if corruption_type == "step10_step20":
            corruption = _from_mapping(StepCorruptionConfig, corruption_raw)
        elif corruption_type == "bert_15_80_10_10":
            corruption = _from_mapping(BertCorruptionConfig, corruption_raw)
        else:
            raise ValueError(f"Unknown MLM corruption: {corruption_type!r}.")
        raw["corruption"] = corruption
        return _from_mapping(TokenMLMObjectiveConfig, raw)
    if objective_type == "standard_mdlm":
        return _from_mapping(MDLMObjectiveConfig, raw)
    raise ValueError(f"Unknown objective.type: {objective_type!r}.")


def _validate_run_config(config: RunConfig) -> None:
    if not config.name.strip():
        raise ValueError("name must not be empty.")
    _validate_training_config(config.training)
    _validate_data_config(config.data)
    _validate_optimizer_config(config.optimizer)
    _validate_scheduler_config(config.scheduler, config.optimizer)
    _validate_ema_config(config.ema)
    _validate_model_config(config.model)
    if config.training.precision not in {"bf16", "fp32"}:
        raise ValueError("training.precision must be 'bf16' or 'fp32'.")
    if config.optimizer.type not in {"adamw", "muon"}:
        raise ValueError("optimizer.type must be 'adamw' or 'muon'.")
    objective = config.objective
    targets = objective.targets
    if targets.special_token_count < 0:
        raise ValueError("objective.targets.special_token_count must be non-negative.")
    if isinstance(objective, ELFObjectiveBase):
        probabilities = (
            objective.decoder_probability,
            objective.self_condition_probability,
        )
        if any(value < 0.0 or value > 1.0 for value in probabilities):
            raise ValueError("ELF objective probabilities must be in [0, 1].")
        if objective.denoiser_p_std < 0.0 or objective.denoiser_noise_scale < 0.0:
            raise ValueError("ELF denoiser scales must be non-negative.")
        if objective.self_condition_cfg_min < 0.0:
            raise ValueError("ELF CFG minimum must be non-negative.")
        if objective.self_condition_cfg_max < objective.self_condition_cfg_min:
            raise ValueError("ELF CFG maximum must be at least the minimum.")
        if not 0.0 < objective.t_eps < 1.0:
            raise ValueError("ELF t_eps must be in (0, 1).")
    if isinstance(objective, NoisyCEObjectiveConfig):
        if objective.decoder_noise_scale < 0.0 or objective.decoder_p_std < 0.0:
            raise ValueError("Noisy-CE decoder scales must be non-negative.")
        if objective.fixed_gaussian_scale < 0.0:
            raise ValueError("fixed_gaussian_scale must be non-negative.")
    if isinstance(objective, TokenMLMObjectiveConfig):
        corruption = objective.corruption
        if isinstance(corruption, BertCorruptionConfig):
            probabilities = (
                corruption.mask_probability,
                corruption.random_probability,
                corruption.unchanged_probability,
            )
            if not 0.0 <= corruption.target_probability <= 1.0:
                raise ValueError("BERT target_probability must be in [0, 1].")
            if any(value < 0.0 or value > 1.0 for value in probabilities):
                raise ValueError("BERT replacement probabilities must be in [0, 1].")
            if not math.isclose(sum(probabilities), 1.0, abs_tol=1.0e-8):
                raise ValueError("BERT replacement probabilities must sum to 1.")
    if isinstance(objective, MDLMObjectiveConfig):
        if config.model.embedding_source != "learnable":
            raise ValueError("standard_mdlm requires learnable embeddings.")
        if objective.noise_schedule != "loglinear":
            raise ValueError("standard_mdlm requires noise_schedule='loglinear'.")
        if not 0.0 < objective.sampling_eps < 1.0:
            raise ValueError("MDLM sampling_eps must be in (0, 1).")
        if not 0.0 <= objective.noise_eps < 1.0:
            raise ValueError("MDLM noise_eps must be in [0, 1).")
        if objective.time_conditioning != "t":
            raise ValueError("standard_mdlm requires time_conditioning='t'.")
        if objective.sampling_steps < 1:
            raise ValueError("MDLM sampling_steps must be positive.")


def _validate_training_config(config: TrainingConfig) -> None:
    positive = {
        "training.epochs": config.epochs,
        "training.batch_size": config.batch_size,
        "training.gradient_accumulation_steps": config.gradient_accumulation_steps,
        "training.log_every": config.log_every,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if config.max_steps < 0:
        raise ValueError("training.max_steps must be non-negative.")
    if config.latest_every_steps < 0:
        raise ValueError("training.latest_every_steps must be non-negative.")
    if config.checkpoint_word_limit is not None and config.checkpoint_word_limit <= 0:
        raise ValueError("training.checkpoint_word_limit must be positive.")
    if config.word_exposure_offset < 0:
        raise ValueError("training.word_exposure_offset must be non-negative.")
    if not 0.0 <= config.encoder_freeze_steps_ratio <= 1.0:
        raise ValueError("training.encoder_freeze_steps_ratio must be in [0, 1].")


def _validate_data_config(config: DataConfig) -> None:
    if config.seq_length <= 0:
        raise ValueError("data.seq_length must be positive.")
    if config.num_workers < 0:
        raise ValueError("data.num_workers must be non-negative.")
    if config.tokenizer_vocab_size <= 0:
        raise ValueError("data.tokenizer_vocab_size must be positive.")
    if config.train_word_count is not None and config.train_word_count <= 0:
        raise ValueError("data.train_word_count must be positive.")
    if config.source == "huggingface":
        required = {
            "hf_dataset": config.hf_dataset,
            "hf_revision": config.hf_revision,
            "train_word_count": config.train_word_count,
        }
        missing = sorted(
            name for name, value in required.items() if value is None or value == ""
        )
        if missing:
            raise ValueError(
                "Canonical Hugging Face data config is missing: "
                + ", ".join(missing)
            )


def _validate_optimizer_config(config: OptimizerConfig) -> None:
    if config.learning_rate <= 0.0:
        raise ValueError("optimizer.learning_rate must be positive.")
    if config.aux_learning_rate is not None and config.aux_learning_rate <= 0.0:
        raise ValueError("optimizer.aux_learning_rate must be positive.")
    if config.encoder_lr_multiplier < 0.0 or config.weight_decay < 0.0:
        raise ValueError("Optimizer multipliers and weight decay must be non-negative.")
    if not 0.0 <= config.beta1 < 1.0 or not 0.0 <= config.beta2 < 1.0:
        raise ValueError("optimizer betas must be in [0, 1).")
    if config.eps <= 0.0 or config.max_grad_norm <= 0.0:
        raise ValueError("optimizer eps and max_grad_norm must be positive.")


def _validate_scheduler_config(
    config: SchedulerConfig,
    optimizer: OptimizerConfig,
) -> None:
    if config.warmup_steps is not None and config.warmup_steps < 0:
        raise ValueError("scheduler.warmup_steps must be non-negative.")
    if config.warmup_epochs is not None and config.warmup_epochs < 0.0:
        raise ValueError("scheduler.warmup_epochs must be non-negative.")
    if config.type not in {"constant", "cosine"}:
        raise ValueError("scheduler.type must be 'constant' or 'cosine'.")
    if config.min_lr < 0.0 or config.min_lr > optimizer.learning_rate:
        raise ValueError("scheduler.min_lr must be in [0, optimizer.learning_rate].")


def _validate_ema_config(config: EMAConfig) -> None:
    if not 0.0 <= config.reference_decay < 1.0:
        raise ValueError("ema.reference_decay must be in [0, 1).")
    if config.reference_steps <= 0:
        raise ValueError("ema.reference_steps must be positive.")


def _validate_model_config(config: BabyLMELFConfig) -> None:
    dimensions = {
        "vocab_size": config.vocab_size,
        "base_vocab_size": config.base_vocab_size,
        "embedding_size": config.embedding_size,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "max_position_embeddings": config.max_position_embeddings,
        "bottleneck_size": config.bottleneck_size,
    }
    for name, value in dimensions.items():
        if value <= 0:
            raise ValueError(f"model.{name} must be positive.")
    if config.hidden_size % config.num_attention_heads:
        raise ValueError("model.hidden_size must be divisible by num_attention_heads.")
    if config.embedding_rms <= 0.0:
        raise ValueError("model.embedding_rms must be positive.")


def _load_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = (
            yaml.safe_load(handle)
            if path.suffix.lower() in {".yml", ".yaml"}
            else json.load(handle)
        )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a mapping.")
    return value


def _from_mapping(cls, data: dict[str, Any]):
    if not isinstance(data, dict):
        raise ValueError(f"{cls.__name__} must be a mapping.")
    known_fields = set(cls.__dataclass_fields__)
    _validate_keys(data, known_fields, set(), cls.__name__)
    type_hints = get_type_hints(cls)
    kwargs = {}
    for field_info in cls.__dataclass_fields__.values():
        if field_info.name in data:
            value = data[field_info.name]
        elif field_info.default is not MISSING:
            value = field_info.default
        elif field_info.default_factory is not MISSING:
            value = field_info.default_factory()
        else:
            raise ValueError(f"Missing required config field: {field_info.name}")
        field_type = type_hints.get(field_info.name, field_info.type)
        if hasattr(field_type, "__dataclass_fields__") and isinstance(value, dict):
            value = _from_mapping(field_type, value)
        kwargs[field_info.name] = value
    return cls(**kwargs)


def _validate_keys(
    data: dict[str, Any],
    allowed: set[str],
    required: set[str],
    label: str,
) -> None:
    unknown = set(data) - allowed
    missing = required - set(data)
    if unknown:
        raise ValueError(f"Unknown fields for {label}: {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"Missing fields for {label}: {', '.join(sorted(missing))}")
