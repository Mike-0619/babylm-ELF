from __future__ import annotations

import torch


def add_noise(
    clean: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
    noise_scale: float,
) -> torch.Tensor:
    """Paper rectified-flow path: t=0 is noise and t=1 is clean data."""
    t = t.view(-1, 1, 1)
    return t * clean + (1.0 - t) * noise * noise_scale


def prediction_to_velocity(
    prediction: torch.Tensor,
    z_t: torch.Tensor,
    t: torch.Tensor,
    t_eps: float,
) -> torch.Tensor:
    denominator = (1.0 - t.view(-1, 1, 1)).clamp_min(t_eps)
    return (prediction - z_t) / denominator
