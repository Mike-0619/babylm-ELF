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
    device = input_ids.device
    decoder_objective = getattr(config, "decoder_objective", "continuous")
    if decoder_objective not in {"continuous", "mlm"}:
        raise ValueError(
            "diffusion.decoder_objective must be 'continuous' or 'mlm', "
            f"got {decoder_objective!r}"
        )

    clean = model.embed_tokens(input_ids, attention_mask=attention_mask)
    dtype = clean.dtype
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
    if decoder_objective == "mlm":
        masked_input_ids, mlm_mask = make_mlm_inputs(
            input_ids,
            attention_mask,
            mask_token_id=model.config.mask_token_id,
            mask_probability=config.mlm_mask_probability,
            special_token_count=config.mlm_special_token_count,
            min_masks_per_sequence=config.mlm_min_masks_per_sequence,
        )
        decoder_z = model.embed_tokens(masked_input_ids, attention_mask=attention_mask)
        mlm_mask_float = mlm_mask.to(torch.float32)
    else:
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
        mlm_mask_float = valid_mask

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

    decoder_valid_mask = valid_mask * decoder_mask_2d
    ce_mask = decoder_valid_mask * mlm_mask_float
    l2_mask = valid_mask * (1.0 - decoder_mask_2d)
    denominator = valid_mask.sum().clamp_min(1.0)

    ce = (ce_per_token * ce_mask).sum() / ce_mask.sum().clamp_min(1.0)
    flow = (l2_per_token * l2_mask).sum() / l2_mask.sum().clamp_min(1.0)
    if decoder_objective == "mlm":
        decoder_weight = decoder_valid_mask.sum() / denominator
        flow_weight = l2_mask.sum() / denominator
        loss = ce * decoder_weight + flow * flow_weight
    else:
        loss = (
            (ce_per_token * ce_mask).sum() + (l2_per_token * l2_mask).sum()
        ) / denominator
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


def make_mlm_inputs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    mask_token_id: int | None,
    mask_probability: float,
    special_token_count: int,
    min_masks_per_sequence: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mask_token_id is None:
        raise ValueError("MLM decoder objective requires model.config.mask_token_id.")
    if not 0.0 <= mask_probability <= 1.0:
        raise ValueError("mlm_mask_probability must be in [0, 1].")

    maskable = attention_mask.bool() & input_ids.ge(special_token_count)
    sampled = torch.rand(input_ids.shape, device=input_ids.device) < mask_probability
    mlm_mask = sampled & maskable
    min_masks = max(0, int(min_masks_per_sequence))
    if min_masks > 0:
        for batch_idx in range(input_ids.size(0)):
            candidates = maskable[batch_idx].nonzero(as_tuple=False).flatten()
            if candidates.numel() == 0:
                continue
            current = int(mlm_mask[batch_idx].sum().item())
            required = min(min_masks, candidates.numel())
            if current >= required:
                continue
            unmasked_candidates = candidates[~mlm_mask[batch_idx, candidates]]
            if unmasked_candidates.numel() == 0:
                continue
            fill_count = min(required - current, unmasked_candidates.numel())
            order = torch.randperm(unmasked_candidates.numel(), device=input_ids.device)
            chosen = unmasked_candidates[order[:fill_count]]
            mlm_mask[batch_idx, chosen] = True

    masked_input_ids = input_ids.clone()
    masked_input_ids[mlm_mask] = mask_token_id
    return masked_input_ids, mlm_mask


@torch.no_grad()
def eval_step(model, batch: dict[str, torch.Tensor], diffusion_config) -> dict[str, float]:
    return train_step(model, batch, diffusion_config).metrics
