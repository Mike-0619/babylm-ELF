from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer

from babylm_elf.config import TrainConfig
from babylm_elf.training.checkpointing import ModelWeights, select_model_weights

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
    *,
    weights: ModelWeights = "ema",
) -> None:
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_state = select_model_weights(checkpoint, weights=weights)
    safe_state = _dedupe_shared_tensors({
        name: tensor.detach().contiguous()
        for name, tensor in _hf_model_state(model_state).items()
    })
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
        "evaluation_config": _evaluation_config(checkpoint_config, config),
        "diagnostic_generation_config": _diagnostic_generation_config(
            checkpoint_config,
            config,
        ),
        "training_metadata": _training_metadata(
            checkpoint_config,
            config,
            checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {},
            weights,
        ),
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
    diffusion = _diffusion_config(checkpoint_config, config)
    model_config.setdefault(
        "mlm_mask_latent_seed",
        int(diffusion.get("mlm_mask_latent_seed", 0)),
    )
    model_config.setdefault(
        "mlm_mask_latent_scale",
        float(diffusion.get("mlm_mask_latent_scale", 1.0)),
    )

    vocab_size = _infer_vocab_size(model_state)
    if vocab_size is not None:
        model_config["vocab_size"] = vocab_size
        if model_config.get("embedding_source") == "scratch_t5_encoder":
            model_config["base_vocab_size"] = vocab_size
            model_config["sentinel_start_id"] = vocab_size
            model_config["encoder_vocab_size"] = vocab_size + int(
                model_config.get("sentinel_count", 100)
            )
    if model_config.get("embedding_source") == "scratch_t5_encoder":
        model_config["encoder_checkpoint_path"] = None
        model_config["latent_stats_path"] = None
    return model_config


def _training_metadata(
    checkpoint_config: Any,
    config: TrainConfig,
    checkpoint_metadata: dict[str, Any],
    weights: ModelWeights,
) -> dict[str, Any]:
    if isinstance(checkpoint_config, dict):
        data_config = checkpoint_config.get("data", {})
        return {
            "name": checkpoint_config.get("name", config.name),
            "tokenizer_path": str(data_config.get("tokenizer_path", config.data.tokenizer_path)),
            "train_path": str(data_config.get("train_path", config.data.train_path)),
            "valid_path": str(data_config.get("valid_path", config.data.valid_path)),
            "max_steps": checkpoint_config.get("max_steps", config.max_steps),
            "word_exposure_offset": checkpoint_config.get(
                "word_exposure_offset",
                config.word_exposure_offset,
            ),
            "weights_variant": weights,
            "checkpoint_metadata": dict(checkpoint_metadata),
        }
    return {
        "name": config.name,
        "tokenizer_path": str(config.data.tokenizer_path),
        "train_path": str(config.data.train_path),
        "valid_path": str(config.data.valid_path),
        "max_steps": config.max_steps,
        "word_exposure_offset": config.word_exposure_offset,
        "weights_variant": weights,
        "checkpoint_metadata": dict(checkpoint_metadata),
    }


def _diffusion_config(checkpoint_config: Any, config: TrainConfig) -> dict[str, Any]:
    if isinstance(checkpoint_config, dict) and isinstance(
        checkpoint_config.get("diffusion"), dict
    ):
        diffusion = dict(checkpoint_config["diffusion"])
    else:
        diffusion = asdict(config.diffusion) if is_dataclass(config.diffusion) else dict(config.diffusion)
    return diffusion


def _evaluation_config(checkpoint_config: Any, config: TrainConfig) -> dict[str, Any]:
    diffusion = _diffusion_config(checkpoint_config, config)
    adapter = diffusion.get("mlm_eval_adapter", "mlm_mask_latent")
    if adapter != "mlm_mask_latent":
        raise ValueError(
            "BabyLM-ELF exports only support diffusion.mlm_eval_adapter="
            f"'mlm_mask_latent'; got {adapter!r}."
        )
    return {
        "backend": "mlm",
        "adapter": "mlm_mask_latent",
        "mc_samples": 1,
        "seed": config.seed,
        "mask_latent_seed": int(diffusion.get("mlm_mask_latent_seed", 0)),
        "mask_latent_scale": float(diffusion.get("mlm_mask_latent_scale", 1.0)),
    }


def _diagnostic_generation_config(
    checkpoint_config: Any,
    config: TrainConfig,
) -> dict[str, Any]:
    diffusion = _diffusion_config(checkpoint_config, config)
    return {
        "purpose": "optional_debug_only_not_babylm_official_scoring",
        "sampling_method": "sde",
        "num_steps": 64,
        "time_schedule": diffusion.get("time_schedule", "logit_normal"),
        "self_cond_cfg_scale": 3.0,
        "sde_gamma": 1.0,
    }


def _infer_vocab_size(model_state: dict[str, torch.Tensor]) -> int | None:
    for key in ("codebook.weight", "babylm_elf.codebook.weight"):
        weight = model_state.get(key)
        if weight is not None:
            return int(weight.shape[1])
    for key in ("codebook.bias", "babylm_elf.codebook.bias"):
        bias = model_state.get(key)
        if bias is not None:
            return int(bias.shape[0])
    for key in (
        "codebook.embedding.weight",
        "babylm_elf.codebook.embedding.weight",
    ):
        weight = model_state.get(key)
        if weight is not None:
            return int(weight.shape[0])
    return None


def _hf_model_state(model_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if all(key.startswith("babylm_elf.") for key in model_state):
        return model_state
    return {f"babylm_elf.{key}": value for key, value in model_state.items()}


def _dedupe_shared_tensors(
    model_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    deduped: dict[str, torch.Tensor] = {}
    seen: dict[tuple[int, int, tuple[int, ...], tuple[int, ...]], str] = {}
    for name, tensor in model_state.items():
        storage_key = (
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
        )
        existing_name = seen.get(storage_key)
        if existing_name is not None:
            if _prefer_shared_tensor_name(name, existing_name):
                deduped.pop(existing_name)
                deduped[name] = tensor
                seen[storage_key] = name
            continue
        seen[storage_key] = name
        deduped[name] = tensor
    return deduped


def _prefer_shared_tensor_name(candidate: str, current: str) -> bool:
    if candidate.endswith(".encoder.embed_tokens.weight") and current.endswith(".shared.weight"):
        return True
    return False


def _write_tokenizer_metadata(
    tokenizer_path: Path,
    output_dir: Path,
    model_max_length: int,
) -> dict[str, int]:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    extra_ids = [
        token
        for token in (f"<extra_id_{index}>" for index in range(100))
        if tokenizer.token_to_id(token) is not None
    ]
    if extra_ids:
        raise ValueError(
            "Exported BabyLM-ELF tokenizer must not contain T5 extra_id tokens: "
            f"{', '.join(extra_ids[:5])}"
        )
    token_ids: dict[str, int] = {}
    for role, token in SPECIAL_TOKENS.items():
        token_id = tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"Tokenizer is missing required {role}: {token}")
        token_ids[role] = token_id

    export_model_max_length = max(
        model_max_length,
        int(os.environ.get("BABYLM_ELF_EXPORT_TOKENIZER_MAX_LENGTH", model_max_length)),
    )
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": export_model_max_length,
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
    shutil.copy2(package_root / "modeling" / "codebook.py", output_dir / "codebook.py")
    shutil.copy2(package_root / "modeling" / "mask_latent.py", output_dir / "mask_latent.py")
    shutil.copy2(package_root / "modeling" / "positions.py", output_dir / "positions.py")
    layers_source = (package_root / "modeling" / "layers.py").read_text(encoding="utf-8")
    layers_source = layers_source.replace(
        "try:\n"
        "    from .sdpa import sdpa_attention\n"
        "except ImportError:\n"
        "    from babylm_elf.modeling.sdpa import sdpa_attention\n",
        "def sdpa_attention(\n"
        "    query: torch.Tensor,\n"
        "    key: torch.Tensor,\n"
        "    value: torch.Tensor,\n"
        "    *,\n"
        "    attn_mask: torch.Tensor | None,\n"
        "    dropout_p: float,\n"
        ") -> torch.Tensor:\n"
        "    return F.scaled_dot_product_attention(\n"
        "        query,\n"
        "        key,\n"
        "        value,\n"
        "        attn_mask=attn_mask,\n"
        "        dropout_p=dropout_p,\n"
        "    )\n",
    )
    (output_dir / "layers.py").write_text(layers_source, encoding="utf-8")
    shutil.copy2(package_root / "modeling" / "sdpa.py", output_dir / "sdpa.py")
