from __future__ import annotations

import json
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer

from babylm_elf.config import TrainConfig

SPECIAL_TOKENS = {
    "unk_token": "<unk>",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
    "mask_token": "<mask>",
    "cls_token": "<s>",
    "sep_token": "</s>",
}


def export_checkpoint_to_hf(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    config: TrainConfig,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_state = checkpoint.get("model", checkpoint)
    safe_state = {
        name: tensor.detach().contiguous()
        for name, tensor in _hf_model_state(model_state).items()
    }
    save_file(safe_state, output_dir / "model.safetensors")
    legacy_weights = output_dir / "pytorch_model.bin"
    if legacy_weights.exists():
        legacy_weights.unlink()

    tokenizer_path = Path(config.data.tokenizer_path)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    shutil.copy2(tokenizer_path, output_dir / "tokenizer.json")
    token_ids = _write_tokenizer_metadata(
        tokenizer_path,
        output_dir,
        config.model.max_position_embeddings,
    )

    checkpoint_config = checkpoint.get("config", {})
    model_config = _model_config(checkpoint_config, config, model_state)
    hf_config = {
        "model_type": "babylm_elf",
        "architectures": ["BabyLMELFForMaskedLM"],
        "auto_map": {
            "AutoConfig": "configuration_babylm_elf.BabyLMELFHFConfig",
            "AutoModel": "modeling_babylm_elf.BabyLMELFHFModel",
            "AutoModelForMaskedLM": "modeling_babylm_elf.BabyLMELFForMaskedLM",
        },
        "hidden_size": model_config["hidden_size"],
        "vocab_size": model_config["vocab_size"],
        "max_position_embeddings": model_config["max_position_embeddings"],
        "pad_token_id": token_ids["pad_token"],
        "bos_token_id": token_ids["bos_token"],
        "eos_token_id": token_ids["eos_token"],
        "mask_token_id": token_ids["mask_token"],
        "cls_token_id": token_ids["cls_token"],
        "sep_token_id": token_ids["sep_token"],
        "babylm_elf_config": model_config,
        "diffusion_config": _diffusion_config(checkpoint_config, config),
        "evaluation_config": {
            "adapter": "continuous_noise_pseudo_likelihood",
            "mc_samples": 4,
            "seed": config.seed,
        },
        "diagnostic_generation_config": _diagnostic_generation_config(
            checkpoint_config,
            config,
        ),
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


def _diffusion_config(checkpoint_config: Any, config: TrainConfig) -> dict[str, Any]:
    if isinstance(checkpoint_config, dict) and isinstance(
        checkpoint_config.get("diffusion"), dict
    ):
        return dict(checkpoint_config["diffusion"])
    return asdict(config.diffusion) if is_dataclass(config.diffusion) else dict(config.diffusion)


def _diagnostic_generation_config(
    checkpoint_config: Any,
    config: TrainConfig,
) -> dict[str, Any]:
    diffusion = _diffusion_config(checkpoint_config, config)
    return {
        "purpose": "open_ended_diagnostic_not_babylm_official_scoring",
        "sampling_method": "sde",
        "num_steps": 64,
        "time_schedule": diffusion.get("time_schedule", "logit_normal"),
        "self_cond_cfg_scale": 3.0,
        "sde_gamma": 1.0,
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


def _write_tokenizer_metadata(
    tokenizer_path: Path,
    output_dir: Path,
    model_max_length: int,
) -> dict[str, int]:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    token_ids: dict[str, int] = {}
    for role, token in SPECIAL_TOKENS.items():
        token_id = tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"Tokenizer is missing required {role}: {token}")
        token_ids[role] = token_id

    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": model_max_length,
        "padding_side": "right",
        "truncation_side": "right",
        **SPECIAL_TOKENS,
    }
    special_tokens_map = dict(SPECIAL_TOKENS)
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2),
        encoding="utf-8",
    )
    (output_dir / "special_tokens_map.json").write_text(
        json.dumps(special_tokens_map, indent=2),
        encoding="utf-8",
    )
    return token_ids


def _copy_export_code(output_dir: Path) -> None:
    package_root = Path(__file__).resolve().parents[1]
    shutil.copy2(package_root / "export" / "hf_config.py", output_dir / "configuration_babylm_elf.py")
    shutil.copy2(package_root / "export" / "hf_model.py", output_dir / "modeling_babylm_elf.py")
    shutil.copy2(package_root / "modeling" / "model.py", output_dir / "modeling_core.py")
    shutil.copy2(package_root / "modeling" / "layers.py", output_dir / "layers.py")
