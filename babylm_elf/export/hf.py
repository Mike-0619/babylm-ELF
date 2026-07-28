from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
from typing import Any
import uuid

from safetensors.torch import save_file
import torch
from tokenizers import Tokenizer

from babylm_elf.config import RunConfig, load_config
from babylm_elf.modules.model import BabyLMELF, BabyLMELFConfig
from babylm_elf.training.checkpoint import (
    ModelWeights,
    infer_track,
    required_revisions,
    select_model_weights,
)


SPECIAL_TOKENS = {
    "unk_token": "<unk>",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
    "mask_token": "<mask>",
    "cls_token": "<s>",
    "sep_token": "</s>",
}


def export_from_config(
    config_path: Path,
    *,
    checkpoint: Path | None,
    output_dir: Path | None,
    weights: ModelWeights,
    all_revisions: bool,
    track: str | None,
) -> None:
    config = load_config(config_path)
    run_dir = Path(config.output_dir) / config.name
    if all_revisions:
        if checkpoint is not None:
            raise ValueError("--checkpoint cannot be used with --all-revisions.")
        _export_all_revisions(
            run_dir,
            output_dir or run_dir / "hf_revisions",
            config,
            track=track,
            weights=weights,
        )
        return
    export_checkpoint_to_hf(
        checkpoint or _default_checkpoint(run_dir),
        output_dir or run_dir / "hf",
        config,
        weights=weights,
    )


def _export_all_revisions(
    run_dir: Path,
    output_root: Path,
    config: RunConfig,
    *,
    track: str | None,
    weights: ModelWeights,
) -> None:
    track = track or infer_track(config.data.train_word_count)
    required_dir = run_dir / "checkpoints" / "babylm_required"
    checkpoints = [
        required_dir / f"{revision}.pt"
        for revision in required_revisions(track)
    ]
    missing = [path.stem for path in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required BabyLM checkpoints: " + ", ".join(missing)
        )
    main = run_dir / "checkpoints" / "final.pt"
    if not main.exists():
        main = _default_checkpoint(run_dir)
    export_checkpoint_to_hf(
        main,
        output_root / f"{config.name}_hf",
        config,
        weights=weights,
    )
    for path in checkpoints:
        export_checkpoint_to_hf(
            path,
            output_root / path.stem,
            config,
            weights=weights,
        )


def _default_checkpoint(run_dir: Path) -> Path:
    required = list(
        (run_dir / "checkpoints" / "babylm_required").glob("chck_*.pt")
    )
    if required:
        return max(required, key=_checkpoint_exposure)
    for name in ("final.pt", "latest.pt"):
        candidate = run_dir / "checkpoints" / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No checkpoint under {run_dir / 'checkpoints'}.")


def _checkpoint_exposure(path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint.get("metadata", {})
    return int(metadata.get("target_words", metadata.get("words_seen", 0)))


def export_checkpoint_to_hf(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    config: RunConfig,
    *,
    weights: ModelWeights = "ema",
) -> None:
    output_dir = Path(output_dir)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    expected_config = asdict(config)
    saved_config = checkpoint.get("resolved_config", {}).get("config")
    if saved_config != expected_config:
        raise ValueError(
            "Export config differs from the checkpoint resolved config. "
            "Use the exact YAML that created this checkpoint."
        )
    model_state = select_model_weights(checkpoint, weights=weights)
    resolved = checkpoint["resolved_config"]
    model_config = dict(resolved["model"])
    if model_config.get("embedding_source") == "scratch_t5_encoder":
        model_config["encoder_checkpoint_path"] = None
        model_config["latent_stats_path"] = None
    verification_model = BabyLMELF(BabyLMELFConfig(**model_config))
    verification_model.load_state_dict(model_state, strict=True)
    del verification_model

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.with_name(
        f".{output_dir.name}.{uuid.uuid4().hex}.tmp"
    )
    backup = output_dir.with_name(
        f".{output_dir.name}.{uuid.uuid4().hex}.old"
    )
    temporary.mkdir()
    try:
        _write_hf_export(
            checkpoint,
            model_state,
            temporary,
            config,
            weights=weights,
            model_config=model_config,
        )
        _validate_export_directory(temporary)
        if output_dir.exists():
            os.replace(output_dir, backup)
        try:
            os.replace(temporary, output_dir)
        except BaseException:
            if backup.exists():
                if output_dir.exists():
                    shutil.rmtree(output_dir)
                os.replace(backup, output_dir)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _write_hf_export(
    checkpoint: dict,
    model_state: dict[str, torch.Tensor],
    output_dir: Path,
    config: RunConfig,
    *,
    weights: ModelWeights,
    model_config: dict[str, Any],
) -> None:
    safe_state = _dedupe_shared_tensors(
        {
            f"babylm_elf.{name}": tensor.detach().contiguous()
            for name, tensor in model_state.items()
        }
    )
    save_file(safe_state, output_dir / "model.safetensors")

    tokenizer_path = Path(config.data.tokenizer_path)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    shutil.copy2(tokenizer_path, output_dir / "tokenizer.json")

    resolved = checkpoint["resolved_config"]
    token_ids = _write_tokenizer_metadata(
        tokenizer_path,
        output_dir,
        int(model_config["max_position_embeddings"]),
    )
    objective = dict(resolved["objective"])
    hf_config = {
        "model_type": "babylm_elf",
        "architectures": ["BabyLMELFForMaskedLM"],
        "auto_map": {
            "AutoConfig": "configuration_babylm_elf.BabyLMELFHFConfig",
            "AutoModel": "modeling_babylm_elf.BabyLMELFHFModel",
            "AutoModelForMaskedLM": (
                "modeling_babylm_elf.BabyLMELFForMaskedLM"
            ),
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
        "objective_config": objective,
        "evaluation_config": _evaluation_config(objective),
        "training_metadata": _training_metadata(
            resolved,
            checkpoint["metadata"],
            weights,
        ),
    }
    (output_dir / "config.json").write_text(
        json.dumps(hf_config, indent=2),
        encoding="utf-8",
    )
    _copy_export_code(output_dir)


def _validate_export_directory(output_dir: Path) -> None:
    required = {
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "configuration_babylm_elf.py",
        "modeling_babylm_elf.py",
        "modeling_core.py",
        "encoder_modeling.py",
        "layers.py",
    }
    missing = sorted(name for name in required if not (output_dir / name).is_file())
    if missing:
        raise RuntimeError("Incomplete HF export: " + ", ".join(missing))
    with (output_dir / "config.json").open("r", encoding="utf-8") as handle:
        exported_config = json.load(handle)
    if exported_config.get("model_type") != "babylm_elf":
        raise RuntimeError("HF export config has an invalid model_type.")


def _evaluation_config(objective: dict[str, Any]) -> dict[str, Any]:
    objective_type = objective["type"]
    targets = objective.get("targets", {})
    evaluation: dict[str, Any] = {
        "backend": "mlm",
        "special_token_count": int(targets.get("special_token_count", 16)),
        "excluded_token_ids": list(targets.get("excluded_token_ids", ())),
    }
    if objective_type == "elf_noisy_ce":
        evaluation.update(
            {
                "adapter": "fixed_gaussian_v1",
                "fixed_gaussian_seed": int(
                    objective.get("fixed_gaussian_seed", 0)
                ),
                "fixed_gaussian_scale": float(
                    objective.get("fixed_gaussian_scale", 5.0)
                ),
                "distribution_shift": (
                    "training uses target-leaking noisy interpolation; "
                    "BabyLM masks use deterministic target-free Gaussian latents"
                ),
            }
        )
    elif objective_type == "standard_mdlm":
        evaluation.update(
            {
                "adapter": "mdlm_subs_v1",
                "noise_schedule": objective.get("noise_schedule", "loglinear"),
                "mdlm_sampling_eps": float(
                    objective.get("sampling_eps", 1.0e-3)
                ),
                "mdlm_noise_eps": float(objective.get("noise_eps", 1.0e-3)),
                "mdlm_sampling_steps": int(
                    objective.get("sampling_steps", 128)
                ),
                "subs_mask_suppression": True,
                "subs_carry_over": True,
            }
        )
    elif objective_type == "elf_token_mlm":
        evaluation["adapter"] = "mlm_mask_latent"
        corruption = objective["corruption"]
        if corruption["type"] == "bert_15_80_10_10":
            evaluation.update(
                {
                    "training_corruption": "bert_15_80_10_10",
                    "training_target_probability": float(
                        corruption["target_probability"]
                    ),
                    "training_replacement_probabilities": {
                        "mask_latent": float(corruption["mask_probability"]),
                        "random_token": float(corruption["random_probability"]),
                        "unchanged": float(
                            corruption["unchanged_probability"]
                        ),
                    },
                    "evaluation_corruption": "all_mask_latent",
                    "distribution_shift": (
                        "training uses 15% BERT 80/10/10 corruption; BabyLM "
                        "evaluation replaces every mask with the learned latent"
                    ),
                }
            )
    else:
        raise ValueError(f"Unsupported objective type: {objective_type!r}.")
    return evaluation


def _training_metadata(
    resolved: dict[str, Any],
    checkpoint_metadata: dict[str, Any],
    weights: ModelWeights,
) -> dict[str, Any]:
    run = resolved["config"]
    data = run["data"]
    return {
        "name": run["name"],
        "tokenizer_path": str(data["tokenizer_path"]),
        "train_path": str(data["train_path"]),
        "max_steps": int(resolved["max_steps"]),
        "word_exposure_offset": int(
            run["training"]["word_exposure_offset"]
        ),
        "weights_variant": weights,
        "checkpoint_metadata": dict(checkpoint_metadata),
    }


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
        existing = seen.get(storage_key)
        if existing is not None:
            if (
                name.endswith(".encoder.embed_tokens.weight")
                and existing.endswith(".shared.weight")
            ):
                deduped.pop(existing)
                deduped[name] = tensor
                seen[storage_key] = name
            continue
        seen[storage_key] = name
        deduped[name] = tensor
    return deduped


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
            "Export tokenizer must not contain T5 extra_id tokens: "
            + ", ".join(extra_ids[:5])
        )
    token_ids: dict[str, int] = {}
    for role, token in SPECIAL_TOKENS.items():
        token_id = tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"Tokenizer is missing required {role}: {token}")
        token_ids[role] = token_id

    export_length = max(
        model_max_length,
        int(
            os.environ.get(
                "BABYLM_ELF_EXPORT_TOKENIZER_MAX_LENGTH",
                model_max_length,
            )
        ),
    )
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": export_length,
        "padding_side": "right",
        "truncation_side": "right",
        **SPECIAL_TOKENS,
    }
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2),
        encoding="utf-8",
    )
    (output_dir / "special_tokens_map.json").write_text(
        json.dumps(SPECIAL_TOKENS, indent=2),
        encoding="utf-8",
    )
    return token_ids


def _copy_export_code(output_dir: Path) -> None:
    package_root = Path(__file__).resolve().parents[1]
    copies = {
        package_root / "export" / "configuration.py": (
            output_dir / "configuration_babylm_elf.py"
        ),
        package_root / "export" / "modeling.py": (
            output_dir / "modeling_babylm_elf.py"
        ),
        package_root / "modules" / "model.py": output_dir / "modeling_core.py",
        package_root / "modules" / "encoder.py": (
            output_dir / "encoder_modeling.py"
        ),
        package_root / "modules" / "layers.py": output_dir / "layers.py",
    }
    for source, destination in copies.items():
        shutil.copy2(source, destination)
