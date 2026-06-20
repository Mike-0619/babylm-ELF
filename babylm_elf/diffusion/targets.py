from __future__ import annotations

import torch


def make_target(
    prediction_type: str,
    x0: torch.Tensor,
    eps: torch.Tensor,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    if prediction_type == "x0":
        return x0
    if prediction_type == "epsilon":
        return eps
    if prediction_type == "velocity":
        return alpha * eps - sigma * x0
    raise ValueError(f"Unknown prediction_type: {prediction_type}")


def prediction_to_x0(prediction, diffusion, prediction_type: str) -> torch.Tensor:
    if prediction_type == "x0":
        return prediction
    alpha = diffusion.alpha.view(-1, 1, 1).clamp_min(1.0e-6)
    sigma = diffusion.sigma.view(-1, 1, 1)
    if prediction_type == "epsilon":
        return (diffusion.z_t - sigma * prediction) / alpha
    if prediction_type == "velocity":
        return alpha * diffusion.z_t - sigma * prediction
    raise ValueError(f"Unknown prediction_type: {prediction_type}")
