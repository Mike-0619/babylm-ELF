from __future__ import annotations

from dataclasses import MISSING, dataclass, field
from pathlib import Path
from typing import Any, get_type_hints
import json

import yaml

from babylm_elf.modeling.model import BabyLMELFConfig


@dataclass
class DataConfig:
    source: str = "local_text"
    hf_dataset: str | None = None
    hf_config: str | None = None
    hf_train_split: str = "train"
    hf_valid_split: str | None = None
    hf_text_field: str = "text"
    train_text: str | None = None
    valid_text: str | None = None
    train_path: str = "data/text_data/train_100M_tokenized.bin"
    valid_path: str | None = "data/text_data/dev_tokenized.bin"
    tokenizer_path: str = "data/tokenizer_100M.json"
    tokenizer_vocab_size: int = 16384
    seq_length: int = 1024
    num_workers: int = 0
    train_word_count: int | None = None


@dataclass
class OptimConfig:
    optimizer: str = "muon"
    learning_rate: float = 1.25e-4
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1.0e-8
    max_grad_norm: float = 1.0
    warmup_steps: int = -1
    warmup_epochs: float | None = 0.5
    lr_schedule: str = "constant"
    min_lr: float = 0.0
    ema_decay: float = 0.9999


@dataclass
class DiffusionConfig:
    prediction_type: str = "x0"
    time_schedule: str = "logit_normal"
    denoiser_p_mean: float = -1.5
    denoiser_p_std: float = 0.8
    denoiser_noise_scale: float = 2.0
    decoder_objective: str = "mlm"
    decoder_probability: float = 0.2
    decoder_p_mean: float = 0.8
    decoder_p_std: float = 0.8
    decoder_noise_scale: float = 5.0
    mlm_mask_probability: float = 0.15
    mlm_special_token_count: int = 16
    mlm_min_masks_per_sequence: int = 1
    self_condition_probability: float = 0.5
    self_condition_cfg_min: float = 0.5
    self_condition_cfg_max: float = 5.0
    t_eps: float = 0.05


@dataclass
class TrainConfig:
    name: str = "babylm_elf"
    seed: int = 42
    output_dir: str = "outputs"
    max_steps: int = 0
    epochs: int = 5
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    log_every: int = 50
    validate_every: int = 1000
    save_every: int = 1000
    checkpoint_by_words: bool = False
    checkpoint_word_limit: int | None = None
    word_exposure_offset: int = 0
    mixed_precision: bool = True
    device: str = "auto"
    model: BabyLMELFConfig = field(default_factory=BabyLMELFConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)


@dataclass
class EncoderModelConfig:
    base_vocab_size: int = 16384
    sentinel_start_id: int = 16384
    sentinel_count: int = 100
    vocab_size: int = 16484
    d_model: int = 512
    d_ff: int = 2048
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
    optim: OptimConfig = field(
        default_factory=lambda: OptimConfig(
            optimizer="adamw",
            learning_rate=1.0e-3,
            weight_decay=0.0,
            beta1=0.9,
            beta2=0.999,
            eps=1.0e-8,
            max_grad_norm=1.0,
            warmup_steps=-1,
            warmup_epochs=0.4,
            lr_schedule="cosine",
            min_lr=0.0,
            ema_decay=0.0,
        )
    )


def load_config(path: str | Path) -> TrainConfig:
    path = Path(path)
    raw = _load_mapping(path)
    return _from_mapping(TrainConfig, raw)


def load_encoder_config(path: str | Path) -> EncoderTrainConfig:
    path = Path(path)
    raw = _load_mapping(path)
    return _from_mapping(EncoderTrainConfig, raw)


def _load_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() in {".yml", ".yaml"}:
            return yaml.safe_load(handle) or {}
        return json.load(handle)


def _from_mapping(cls, data: dict[str, Any]):
    kwargs = {}
    type_hints = get_type_hints(cls)
    known_fields = set(cls.__dataclass_fields__)
    unknown_fields = set(data) - known_fields
    if unknown_fields:
        raise ValueError(
            f"Unknown fields for {cls.__name__}: {', '.join(sorted(unknown_fields))}"
        )
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
