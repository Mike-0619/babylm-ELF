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
    tokenizer_min_frequency: int = 2
    seq_length: int = 512
    num_workers: int = 0


@dataclass
class OptimConfig:
    optimizer: str = "adamw"
    learning_rate: float = 5.0e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1.0e-8
    max_grad_norm: float = 2.0
    warmup_steps: int = 1000
    ema_decay: float = 0.999


@dataclass
class DiffusionConfig:
    prediction_type: str = "x0"
    time_schedule: str = "linear"
    noise_scale: float = 1.0
    flow_loss_weight: float = 1.0
    decode_loss_weight: float = 0.25


@dataclass
class TrainConfig:
    name: str = "babylm_elf"
    seed: int = 42
    output_dir: str = "outputs"
    max_steps: int = 1000
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    log_every: int = 50
    validate_every: int = 1000
    save_every: int = 1000
    mixed_precision: bool = True
    device: str = "auto"
    model: BabyLMELFConfig = field(default_factory=BabyLMELFConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)


def load_config(path: str | Path) -> TrainConfig:
    path = Path(path)
    raw = _load_mapping(path)
    return _from_mapping(TrainConfig, raw)


def _load_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() in {".yml", ".yaml"}:
            return yaml.safe_load(handle) or {}
        return json.load(handle)


def _from_mapping(cls, data: dict[str, Any]):
    kwargs = {}
    type_hints = get_type_hints(cls)
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
