from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from src.config import RunConfig
    from src.modules.model import BabyLMELFConfig


def build_scratch_encoder(config: BabyLMELFConfig):
    from transformers import T5Config, T5EncoderModel

    t5_config = T5Config(
        vocab_size=config.encoder_vocab_size,
        d_model=config.embedding_size,
        d_ff=config.encoder_d_ff,
        d_kv=config.encoder_d_kv,
        num_layers=config.encoder_num_layers,
        num_decoder_layers=config.encoder_num_layers,
        num_heads=config.encoder_num_heads,
        dropout_rate=config.encoder_dropout_rate,
        layer_norm_epsilon=config.layer_norm_eps,
        feed_forward_proj="relu",
        pad_token_id=config.pad_token_id,
        eos_token_id=2,
        decoder_start_token_id=config.pad_token_id,
    )
    return T5EncoderModel(t5_config)


def load_scratch_encoder(
    scratch_encoder: nn.Module | None,
    checkpoint_path: str | None,
) -> None:
    if scratch_encoder is None or not checkpoint_path:
        return
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Scratch encoder checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("encoder", checkpoint.get("model", checkpoint))
    try:
        scratch_encoder.load_state_dict(state)
    except RuntimeError:
        scratch_encoder.encoder.load_state_dict(
            _strip_prefix_if_present(state, "encoder.")
        )


def load_latent_stats(
    config: BabyLMELFConfig,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
) -> None:
    if config.embedding_source != "scratch_t5_encoder" or not config.latent_stats_path:
        return
    path = Path(config.latent_stats_path)
    if not path.exists():
        raise FileNotFoundError(f"Scratch encoder latent stats not found: {path}")
    stats = torch.load(path, map_location="cpu", weights_only=True)
    mean = stats["mean"].float()
    std = stats["std"].float().clamp_min(1.0e-6)
    if mean.shape != latent_mean.shape or std.shape != latent_std.shape:
        raise ValueError(
            "Latent stats shape mismatch: expected "
            f"{tuple(latent_mean.shape)}, got mean={tuple(mean.shape)}, "
            f"std={tuple(std.shape)}"
        )
    latent_mean.copy_(mean)
    latent_std.copy_(std)


def initialize_gaussian_embedding(
    token_embedding: nn.Embedding | None,
    standard_deviation: float,
) -> None:
    if token_embedding is None:
        return
    nn.init.normal_(
        token_embedding.weight,
        mean=0.0,
        std=standard_deviation,
    )
    with torch.no_grad():
        weight = token_embedding.weight
        weight.sub_(weight.mean(dim=-1, keepdim=True))
        std = weight.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1.0e-6)
        weight.div_(std)
        weight.mul_(standard_deviation)


def set_scratch_encoder_trainability(
    scratch_encoder: nn.Module | None,
    trainable: bool,
) -> None:
    if scratch_encoder is None:
        return
    scratch_encoder.train(trainable)
    if not trainable:
        scratch_encoder.eval()
    for parameter in scratch_encoder.parameters():
        parameter.requires_grad_(trainable)


def validate_embedding_source(config: RunConfig) -> None:
    model = config.model
    if model.embedding_source != "scratch_t5_encoder":
        return
    if model.scratch_encoder_trainable:
        return
    if not model.encoder_checkpoint_path:
        raise ValueError(
            "scratch_t5_encoder requires model.encoder_checkpoint_path."
        )
    encoder_checkpoint = Path(model.encoder_checkpoint_path)
    if not encoder_checkpoint.exists():
        raise FileNotFoundError(
            f"Scratch encoder checkpoint not found: {encoder_checkpoint}"
        )
    if not model.latent_stats_path:
        raise ValueError("scratch_t5_encoder requires model.latent_stats_path.")
    latent_stats = Path(model.latent_stats_path)
    if not latent_stats.exists():
        raise FileNotFoundError(
            f"Scratch encoder latent stats not found: {latent_stats}"
        )


def scratch_encoder_parameter_ids(model) -> set[int]:
    if model.scratch_encoder is None:
        return set()
    return {
        id(parameter)
        for parameter in model.scratch_encoder.parameters()
    }


def scratch_encoder_should_train(
    config: RunConfig,
    base_trainable: bool,
    max_steps: int,
    step: int,
) -> bool:
    if (
        config.model.embedding_source != "scratch_t5_encoder"
        or not base_trainable
    ):
        return base_trainable
    freeze_steps = math.ceil(
        max(0.0, config.training.encoder_freeze_steps_ratio) * max_steps
    )
    return step >= freeze_steps


def set_model_scratch_encoder_trainability(model, trainable: bool) -> None:
    set_scratch_encoder_trainability(model.scratch_encoder, trainable)


def zero_scratch_encoder_optimizer_lrs(
    optimizer: torch.optim.Optimizer,
    scratch_encoder_param_ids: set[int],
) -> None:
    if not scratch_encoder_param_ids:
        return
    for group in optimizer.param_groups:
        if any(
            id(parameter) in scratch_encoder_param_ids
            for parameter in group["params"]
        ):
            group["lr"] = 0.0


def embed_with_scratch_encoder(
    scratch_encoder: nn.Module | None,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    segment_ids: torch.Tensor | None,
    *,
    config: BabyLMELFConfig,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    if scratch_encoder is None:
        raise RuntimeError("Scratch encoder is not initialized.")
    if attention_mask is None:
        attention_mask = input_ids.ne(config.pad_token_id).to(torch.long)
    encoder_attention = scratch_encoder_attention_mask(
        attention_mask,
        segment_ids,
        dtype=next(scratch_encoder.parameters()).dtype,
    )
    encoder_ids = replace_mask_spans_with_sentinels(
        input_ids,
        attention_mask,
        mask_token_id=config.mask_token_id,
        sentinel_start_id=config.sentinel_start_id,
        sentinel_count=config.sentinel_count,
    )
    if config.scratch_encoder_trainable:
        outputs = scratch_encoder(
            input_ids=encoder_ids,
            attention_mask=encoder_attention,
        ).last_hidden_state
    else:
        was_training = scratch_encoder.training
        scratch_encoder.eval()
        try:
            with torch.no_grad():
                outputs = scratch_encoder(
                    input_ids=encoder_ids,
                    attention_mask=encoder_attention,
                ).last_hidden_state
        finally:
            if was_training:
                scratch_encoder.train()
    normalized = (outputs.float() - latent_mean) / latent_std.clamp_min(1.0e-6)
    return normalized.to(dtype=output_dtype)


def scratch_encoder_attention_mask(
    attention_mask: torch.Tensor,
    segment_ids: torch.Tensor | None,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    active = attention_mask.bool()
    if segment_ids is None:
        return active
    if segment_ids.shape != active.shape:
        raise ValueError(
            "segment_ids and attention_mask must have the same shape; "
            f"got {tuple(segment_ids.shape)} and {tuple(active.shape)}."
        )
    same_segment = segment_ids.unsqueeze(-1).eq(segment_ids.unsqueeze(-2))
    allowed = active.unsqueeze(-1) & active.unsqueeze(-2) & same_segment
    # Padding queries are irrelevant downstream, but a self edge keeps the T5
    # softmax finite without allowing them to read any lexical token.
    padding_diagonal = torch.diag_embed(~active)
    allowed = (allowed | padding_diagonal).unsqueeze(1)
    additive = torch.zeros(
        allowed.shape,
        device=allowed.device,
        dtype=dtype,
    )
    return additive.masked_fill(~allowed, torch.finfo(dtype).min)


def replace_mask_spans_with_sentinels(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    mask_token_id: int | None,
    sentinel_start_id: int,
    sentinel_count: int,
) -> torch.Tensor:
    if mask_token_id is None:
        return input_ids
    mapped = input_ids.clone()
    mask_positions = input_ids.eq(mask_token_id) & attention_mask.bool()
    for batch_idx in range(input_ids.size(0)):
        sentinel_offset = 0
        in_span = False
        for token_idx in range(input_ids.size(1)):
            if not bool(mask_positions[batch_idx, token_idx]):
                in_span = False
                continue
            if not in_span:
                mapped[batch_idx, token_idx] = min(
                    sentinel_start_id + sentinel_offset,
                    sentinel_start_id + sentinel_count - 1,
                )
                sentinel_offset += 1
                in_span = True
            else:
                mapped[batch_idx, token_idx] = mapped[batch_idx, token_idx - 1]
    return mapped


def _strip_prefix_if_present(
    state: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    if not any(key.startswith(prefix) for key in state):
        return state
    return {
        key.removeprefix(prefix): value
        for key, value in state.items()
        if key.startswith(prefix)
    }
