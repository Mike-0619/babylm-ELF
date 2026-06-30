from __future__ import annotations

import torch


def sample_timesteps(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    p_mean: float,
    p_std: float,
    schedule: str,
) -> torch.Tensor:
    if schedule == "logit_normal":
        logits = torch.randn(batch_size, device=device, dtype=dtype)
        return torch.sigmoid(logits * p_std + p_mean)
    if schedule == "uniform":
        return torch.rand(batch_size, device=device, dtype=dtype)
    raise ValueError(f"Unknown time schedule: {schedule}")


def sample_cfg_scale(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    minimum: float,
    maximum: float,
) -> torch.Tensor:
    uniform = torch.rand(batch_size, device=device, dtype=dtype)
    low = 1.0 + minimum
    high = 1.0 + maximum
    return low * torch.exp(uniform * torch.log(torch.tensor(high / low, device=device))) - 1.0
