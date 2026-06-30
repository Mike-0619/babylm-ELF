from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from babylm_elf.diffusion.noising import add_noise, prediction_to_velocity
from babylm_elf.diffusion.schedules import sample_cfg_scale, sample_timesteps


@dataclass
class StepOutput:
    loss: torch.Tensor
    metrics: dict[str, float]


def train_step(model, batch: dict[str, torch.Tensor], config) -> StepOutput:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    valid_mask = attention_mask.to(torch.float32)
    batch_size, sequence_length = input_ids.shape
    dtype = model.token_embedding.weight.dtype
    device = input_ids.device

    clean = model.embed_tokens(input_ids)
    t = sample_timesteps(
        batch_size,
        device,
        dtype,
        config.denoiser_p_mean,
        config.denoiser_p_std,
        config.time_schedule,
    )
    noise = torch.randn_like(clean)
    denoiser_z = add_noise(clean, noise, t, config.denoiser_noise_scale)
    velocity_target = (clean - denoiser_z) / (
        1.0 - t.view(-1, 1, 1)
    ).clamp_min(config.t_eps)

    decoder_active = (
        torch.rand(batch_size, device=device) < config.decoder_probability
    ).to(dtype)
    decoder_mask_3d = decoder_active.view(-1, 1, 1)
    decoder_mask_2d = decoder_active.view(-1, 1)
    decoder_mix = torch.sigmoid(
        torch.randn(
            batch_size,
            sequence_length,
            1,
            device=device,
            dtype=dtype,
        )
        * config.decoder_p_std
        + config.decoder_p_mean
    )
    decoder_noise = torch.randn_like(clean) * config.decoder_noise_scale
    decoder_z = decoder_mix * clean + (1.0 - decoder_mix) * decoder_noise

    mixed_z = decoder_mask_3d * decoder_z + (1.0 - decoder_mask_3d) * denoiser_z
    mixed_t = decoder_active + (1.0 - decoder_active) * t
    self_condition_mask = (
        torch.rand(batch_size, 1, 1, device=device)
        < config.self_condition_probability
    ).to(dtype)
    cfg_scale = sample_cfg_scale(
        batch_size,
        device,
        dtype,
        config.self_condition_cfg_min,
        config.self_condition_cfg_max,
    )

    zero_condition = torch.zeros_like(clean)
    with torch.no_grad():
        unconditioned_prediction, _ = model(
            torch.cat((denoiser_z, zero_condition), dim=-1),
            t,
            attention_mask=attention_mask,
            self_cond_cfg_scale=cfg_scale,
            decoder_step_active=torch.zeros_like(decoder_active),
        )
        unconditioned_velocity = prediction_to_velocity(
            unconditioned_prediction,
            denoiser_z,
            t,
            config.t_eps,
        )
        conditioned_prediction, _ = model(
            torch.cat((denoiser_z, unconditioned_prediction), dim=-1),
            t,
            attention_mask=attention_mask,
            self_cond_cfg_scale=cfg_scale,
            decoder_step_active=torch.zeros_like(decoder_active),
        )
        conditioned_velocity = prediction_to_velocity(
            conditioned_prediction,
            denoiser_z,
            t,
            config.t_eps,
        )
        guidance = (1.0 - 1.0 / cfg_scale.view(-1, 1, 1)) * (
            conditioned_velocity - unconditioned_velocity
        )
        velocity_target = velocity_target + guidance * self_condition_mask

    self_condition = unconditioned_prediction * self_condition_mask
    self_condition = self_condition * (1.0 - decoder_mask_3d)
    prediction, decoder_logits = model(
        torch.cat((mixed_z, self_condition), dim=-1),
        mixed_t,
        attention_mask=attention_mask,
        self_cond_cfg_scale=cfg_scale,
        decoder_step_active=decoder_active,
    )

    ce_per_token = F.cross_entropy(
        decoder_logits.float().reshape(-1, decoder_logits.size(-1)),
        input_ids.reshape(-1),
        reduction="none",
    ).view_as(input_ids)
    predicted_velocity = prediction_to_velocity(
        prediction,
        denoiser_z,
        t,
        config.t_eps,
    )
    l2_per_token = (predicted_velocity - velocity_target.detach()).square().mean(dim=-1)

    ce_mask = valid_mask * decoder_mask_2d
    l2_mask = valid_mask * (1.0 - decoder_mask_2d)
    denominator = valid_mask.sum().clamp_min(1.0)
    loss = (
        (ce_per_token * ce_mask).sum() + (l2_per_token * l2_mask).sum()
    ) / denominator

    ce = (ce_per_token * ce_mask).sum() / ce_mask.sum().clamp_min(1.0)
    flow = (l2_per_token * l2_mask).sum() / l2_mask.sum().clamp_min(1.0)
    correct = decoder_logits.argmax(dim=-1).eq(input_ids).to(torch.float32)
    accuracy = (correct * ce_mask).sum() / ce_mask.sum().clamp_min(1.0)
    return StepOutput(
        loss=loss,
        metrics={
            "loss": float(loss.detach()),
            "flow": float(flow.detach()),
            "ce": float(ce.detach()),
            "acc": float(accuracy.detach()),
            "decode_frac": float(decoder_active.float().mean()),
        },
    )


@torch.no_grad()
def eval_step(model, batch: dict[str, torch.Tensor], diffusion_config) -> dict[str, float]:
    return train_step(model, batch, diffusion_config).metrics
