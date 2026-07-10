from __future__ import annotations

import math

import torch

try:
    from .codebook import safe_unit_vectors
except ImportError:
    from babylm_elf.modeling.codebook import safe_unit_vectors


def build_embedding_stats_mask_latent(
    token_embedding_weight: torch.Tensor,
    *,
    embedding_size: int,
    embedding_rms: float,
    pad_token_id: int | None,
    seed: int,
    scale: float,
) -> torch.Tensor:
    """Initialize the learned continuous mask latent from embedding statistics."""
    if token_embedding_weight.size(-1) != embedding_size:
        raise ValueError(
            "token_embedding_weight last dimension must match embedding_size; "
            f"got {token_embedding_weight.size(-1)} and {embedding_size}."
        )

    with torch.no_grad():
        weight = safe_unit_vectors(token_embedding_weight.detach())
        weight = weight * (math.sqrt(embedding_size) * embedding_rms)
        if pad_token_id is not None:
            keep = torch.ones(weight.size(0), device=weight.device, dtype=torch.bool)
            if 0 <= pad_token_id < weight.size(0):
                keep[pad_token_id] = False
            weight_for_stats = weight[keep]
        else:
            weight_for_stats = weight
        if weight_for_stats.numel() == 0:
            raise ValueError("Cannot build mask latent from an empty embedding table.")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        fixed_eps = torch.randn(
            embedding_size,
            generator=generator,
            dtype=torch.float32,
        ).to(weight_for_stats.device)
        emb_mean = weight_for_stats.mean(dim=0)
        emb_std = weight_for_stats.std(dim=0, unbiased=False).clamp_min(1.0e-6)
        return (emb_mean + float(scale) * emb_std * fixed_eps).detach()
