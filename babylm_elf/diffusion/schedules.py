from __future__ import annotations

import torch


def sample_timesteps(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    min_t: float = 1.0e-4,
    max_t: float = 1.0,
) -> torch.Tensor:
    return torch.empty(batch_size, device=device, dtype=dtype).uniform_(min_t, max_t)


def alpha_sigma(
    t: torch.Tensor,
    time_schedule: str,
    noise_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if time_schedule == "linear":
        return 1.0 - t, t * noise_scale
    if time_schedule == "cosine":
        theta = t * torch.pi / 2.0
        return torch.cos(theta), torch.sin(theta) * noise_scale
    raise ValueError(f"Unknown time_schedule: {time_schedule}")

