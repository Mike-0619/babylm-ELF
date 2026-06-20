from __future__ import annotations

import json
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

from babylm_elf.config import TrainConfig


def export_checkpoint_to_hf(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    config: TrainConfig,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state = checkpoint.get("model", checkpoint)
    torch.save(_hf_model_state(model_state), output_dir / "pytorch_model.bin")

    tokenizer_path = Path(config.data.tokenizer_path)
    if tokenizer_path.exists():
        shutil.copy2(tokenizer_path, output_dir / "tokenizer.json")

    checkpoint_config = checkpoint.get("config", {})
    model_config = _model_config(checkpoint_config, config, model_state)
    hf_config = {
        "model_type": "babylm_elf",
        "architectures": ["BabyLMELFHFModel"],
        "auto_map": {
            "AutoConfig": "configuration_babylm_elf.BabyLMELFHFConfig",
            "AutoModel": "modeling_babylm_elf.BabyLMELFHFModel",
            "AutoModelForMaskedLM": "modeling_babylm_elf.BabyLMELFHFModel",
        },
        "babylm_elf_config": model_config,
        "training_metadata": _training_metadata(checkpoint_config, config),
    }
    (output_dir / "config.json").write_text(json.dumps(hf_config, indent=2), encoding="utf-8")

    _copy_export_code(output_dir)


def _model_config(
    checkpoint_config: Any,
    config: TrainConfig,
    model_state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    if isinstance(checkpoint_config, dict) and isinstance(checkpoint_config.get("model"), dict):
        model_config = dict(checkpoint_config["model"])
    elif is_dataclass(config.model):
        model_config = asdict(config.model)
    else:
        model_config = dict(config.model)

    vocab_size = _infer_vocab_size(model_state)
    if vocab_size is not None:
        model_config["vocab_size"] = vocab_size
    return model_config


def _training_metadata(checkpoint_config: Any, config: TrainConfig) -> dict[str, Any]:
    if isinstance(checkpoint_config, dict):
        data_config = checkpoint_config.get("data", {})
        return {
            "name": checkpoint_config.get("name", config.name),
            "tokenizer_path": str(data_config.get("tokenizer_path", config.data.tokenizer_path)),
            "train_path": str(data_config.get("train_path", config.data.train_path)),
            "valid_path": str(data_config.get("valid_path", config.data.valid_path)),
            "max_steps": checkpoint_config.get("max_steps", config.max_steps),
        }
    return {
        "name": config.name,
        "tokenizer_path": str(config.data.tokenizer_path),
        "train_path": str(config.data.train_path),
        "valid_path": str(config.data.valid_path),
        "max_steps": config.max_steps,
    }


def _infer_vocab_size(model_state: dict[str, torch.Tensor]) -> int | None:
    for key in ("token_embedding.weight", "babylm_elf.token_embedding.weight"):
        weight = model_state.get(key)
        if weight is not None:
            return int(weight.shape[0])
    return None


def _hf_model_state(model_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if all(key.startswith("babylm_elf.") for key in model_state):
        return model_state
    return {f"babylm_elf.{key}": value for key, value in model_state.items()}


def _copy_export_code(output_dir: Path) -> None:
    package_root = Path(__file__).resolve().parents[1]
    shutil.copy2(package_root / "export" / "hf_config.py", output_dir / "configuration_babylm_elf.py")
    shutil.copy2(package_root / "export" / "hf_model.py", output_dir / "modeling_babylm_elf.py")

    package_copy = output_dir / "babylm_elf"
    if package_copy.exists():
        shutil.rmtree(package_copy)

    modeling_copy = package_copy / "modeling"
    modeling_copy.mkdir(parents=True, exist_ok=True)
    shutil.copy2(package_root / "__init__.py", package_copy / "__init__.py")
    shutil.copy2(package_root / "modeling" / "__init__.py", modeling_copy / "__init__.py")
    for filename in ("model.py", "layers.py", "heads.py"):
        shutil.copy2(package_root / "modeling" / filename, modeling_copy / filename)
