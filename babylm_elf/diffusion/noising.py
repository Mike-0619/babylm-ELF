from __future__ import annotations

from dataclasses import dataclass

import torch

from babylm_elf.diffusion.schedules import alpha_sigma, sample_timesteps
from babylm_elf.diffusion.targets import make_target


@dataclass
class DiffusionBatch:
    z_t: torch.Tensor
    t: torch.Tensor
    eps: torch.Tensor
    alpha: torch.Tensor
    sigma: torch.Tensor
    target: torch.Tensor


def add_noise(
    x0: torch.Tensor,
    prediction_type: str,
    time_schedule: str,
    noise_scale: float,
) -> DiffusionBatch:
    t = sample_timesteps(x0.size(0), x0.device, x0.dtype)
    eps = torch.randn_like(x0)
    alpha, sigma = alpha_sigma(t, time_schedule, noise_scale)
    alpha_b = alpha.view(-1, 1, 1)
    sigma_b = sigma.view(-1, 1, 1)
    z_t = alpha_b * x0 + sigma_b * eps

    target = make_target(prediction_type, x0, eps, alpha_b, sigma_b)

    return DiffusionBatch(z_t=z_t, t=t, eps=eps, alpha=alpha, sigma=sigma, target=target)
