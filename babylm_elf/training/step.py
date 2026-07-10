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


def unwrap_model(model):
    return getattr(model, "module", model)


def train_step(
    model,
    batch: dict[str, torch.Tensor],
    config,
    current_epoch: int = 0,
) -> StepOutput:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    sequence_ids = batch.get("sequence_id")
    valid_mask = attention_mask.to(torch.float32)
    decoder_objective = getattr(config, "decoder_objective", "token_mlm")
    if decoder_objective != "token_mlm":
        raise ValueError(
            "diffusion.decoder_objective must be 'token_mlm', "
            f"got {decoder_objective!r}"
        )

    return train_step_token_mlm(
        model,
        input_ids,
        attention_mask,
        valid_mask,
        config,
        sequence_ids=sequence_ids,
        current_epoch=current_epoch,
    )


def train_step_token_mlm(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    config,
    *,
    sequence_ids: torch.Tensor | None = None,
    current_epoch: int = 0,
) -> StepOutput:
    batch_size = input_ids.size(0)
    device = input_ids.device
    base_model = unwrap_model(model)

    if config.decoder_probability >= 1.0:
        return run_decoder_only_step(
            model,
            input_ids,
            attention_mask,
            config,
            sequence_ids=sequence_ids,
            current_epoch=current_epoch,
        )

    clean = base_model.embed_tokens(input_ids, attention_mask=attention_mask)
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
    decoder_rows = decoder_active.bool().nonzero(as_tuple=False).flatten()
    denoiser_rows = (~decoder_active.bool()).nonzero(as_tuple=False).flatten()

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
    cfg_scale = torch.where(
        decoder_active.bool(),
        torch.ones_like(cfg_scale),
        cfg_scale,
    )

    if denoiser_rows.numel() > 0:
        zero_condition = torch.zeros_like(clean)
        with torch.no_grad():
            unconditioned_prediction, _ = base_model(
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
            conditioned_prediction, _ = base_model(
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
    else:
        unconditioned_prediction = torch.zeros_like(clean)

    self_condition = unconditioned_prediction * self_condition_mask

    z_parts: list[torch.Tensor] = []
    self_condition_parts: list[torch.Tensor] = []
    t_parts: list[torch.Tensor] = []
    attention_parts: list[torch.Tensor] = []
    cfg_parts: list[torch.Tensor] = []
    decoder_active_parts: list[torch.Tensor] = []

    decoder_count = 0
    decoder_target_ids = input_ids.new_empty((0, input_ids.size(1)))
    decoder_attention_mask = attention_mask.new_empty((0, attention_mask.size(1)))
    ce_token_mask = torch.zeros(
        (0, input_ids.size(1)),
        device=device,
        dtype=torch.float32,
    )
    if decoder_rows.numel() > 0:
        (
            decoder_z,
            ce_token_mask,
            decoder_target_ids,
            decoder_attention_mask,
            _,
        ) = build_mlm_decoder_inputs(
            model,
            input_ids.index_select(0, decoder_rows),
            attention_mask.index_select(0, decoder_rows),
            config,
            sequence_ids=(
                sequence_ids.to(device=device).index_select(0, decoder_rows)
                if sequence_ids is not None
                else None
            ),
            current_epoch=current_epoch,
        )
        decoder_count = decoder_z.size(0)
        if decoder_count > 0:
            z_parts.append(decoder_z)
            self_condition_parts.append(torch.zeros_like(decoder_z))
            t_parts.append(torch.ones(decoder_count, device=device, dtype=dtype))
            attention_parts.append(decoder_attention_mask)
            cfg_parts.append(torch.ones(decoder_count, device=device, dtype=dtype))
            decoder_active_parts.append(
                torch.ones(decoder_count, device=device, dtype=dtype)
            )

    if denoiser_rows.numel() > 0:
        z_parts.append(denoiser_z.index_select(0, denoiser_rows))
        self_condition_parts.append(self_condition.index_select(0, denoiser_rows))
        t_parts.append(t.index_select(0, denoiser_rows))
        attention_parts.append(attention_mask.index_select(0, denoiser_rows))
        cfg_parts.append(cfg_scale.index_select(0, denoiser_rows))
        decoder_active_parts.append(
            torch.zeros(denoiser_rows.numel(), device=device, dtype=dtype)
        )

    if not z_parts:
        zero = torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)
        return StepOutput(
            loss=zero,
            metrics={
                "loss": 0.0,
                "flow": 0.0,
                "ce": 0.0,
                "acc": 0.0,
                "decode_frac": float(decoder_active.float().mean()),
            },
        )

    mixed_z = torch.cat(z_parts, dim=0)
    mixed_self_condition = torch.cat(self_condition_parts, dim=0)
    mixed_t = torch.cat(t_parts, dim=0)
    mixed_attention_mask = torch.cat(attention_parts, dim=0)
    mixed_cfg_scale = torch.cat(cfg_parts, dim=0)
    mixed_decoder_active = torch.cat(decoder_active_parts, dim=0)

    prediction, decoder_logits = model(
        torch.cat((mixed_z, mixed_self_condition), dim=-1),
        mixed_t,
        attention_mask=mixed_attention_mask,
        self_cond_cfg_scale=mixed_cfg_scale,
        decoder_step_active=mixed_decoder_active,
    )

    zero_from_forward = prediction.sum() * 0.0
    if decoder_count > 0:
        decoder_logits_slice = decoder_logits[:decoder_count]
        ce_mask = decoder_attention_mask.to(torch.float32) * ce_token_mask
        ce, accuracy = ce_loss_and_accuracy(
            decoder_target_ids,
            decoder_logits_slice,
            ce_mask,
        )
    else:
        ce = zero_from_forward
        accuracy = zero_from_forward

    if denoiser_rows.numel() > 0:
        denoiser_prediction = prediction[decoder_count:]
        predicted_velocity = prediction_to_velocity(
            denoiser_prediction,
            denoiser_z.index_select(0, denoiser_rows),
            t.index_select(0, denoiser_rows),
            config.t_eps,
        )
        l2_per_token = (
            predicted_velocity
            - velocity_target.index_select(0, denoiser_rows).detach()
        ).square().mean(dim=-1)
        l2_mask = valid_mask.index_select(0, denoiser_rows)
        flow = (l2_per_token * l2_mask).sum() / l2_mask.sum().clamp_min(1.0)
    else:
        flow = zero_from_forward

    decoder_mask_2d = decoder_active.view(-1, 1)
    decoder_valid_mask = valid_mask * decoder_mask_2d
    l2_mask_original = valid_mask * (1.0 - decoder_mask_2d)
    denominator = valid_mask.sum().clamp_min(1.0)
    decoder_weight = decoder_valid_mask.sum() / denominator
    flow_weight = l2_mask_original.sum() / denominator
    loss = ce * decoder_weight + flow * flow_weight

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


def run_decoder_only_step(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    config,
    *,
    sequence_ids: torch.Tensor | None = None,
    current_epoch: int = 0,
) -> StepOutput:
    device = input_ids.device
    (
        decoder_z,
        ce_token_mask,
        decoder_target_ids,
        decoder_attention_mask,
        _,
    ) = build_mlm_decoder_inputs(
        model,
        input_ids,
        attention_mask,
        config,
        sequence_ids=sequence_ids,
        current_epoch=current_epoch,
    )
    if decoder_z.size(0) == 0:
        zero = torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)
        return StepOutput(
            loss=zero,
            metrics={
                "loss": 0.0,
                "flow": 0.0,
                "ce": 0.0,
                "acc": 0.0,
                "decode_frac": 1.0,
            },
        )

    dtype = decoder_z.dtype
    decoder_active = torch.ones(decoder_z.size(0), device=device, dtype=dtype)
    _, decoder_logits = model(
        torch.cat((decoder_z, torch.zeros_like(decoder_z)), dim=-1),
        torch.ones_like(decoder_active),
        attention_mask=decoder_attention_mask,
        self_cond_cfg_scale=torch.ones_like(decoder_active),
        decoder_step_active=decoder_active,
    )
    ce_mask = decoder_attention_mask.to(torch.float32) * ce_token_mask
    ce, accuracy = ce_loss_and_accuracy(
        decoder_target_ids,
        decoder_logits,
        ce_mask,
    )
    return StepOutput(
        loss=ce,
        metrics={
            "loss": float(ce.detach()),
            "flow": 0.0,
            "ce": float(ce.detach()),
            "acc": float(accuracy.detach()),
            "decode_frac": 1.0,
        },
    )


def build_mlm_decoder_inputs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    config,
    *,
    sequence_ids: torch.Tensor | None = None,
    current_epoch: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    base_model = unwrap_model(model)
    _masked_ids_for_selection, mlm_mask, row_indices = make_mlm_input(
        input_ids,
        attention_mask,
        mask_token_id=base_model.config.mask_token_id,
        special_token_count=config.mlm_special_token_count,
        excluded_token_ids=getattr(config, "mlm_excluded_token_ids", ()),
        strategy=getattr(
            config,
            "mlm_mask_strategy",
            "one_per_segment_step10_then_step20",
        ),
        mask_schedule=getattr(config, "mlm_mask_schedule", "random"),
        mask_seed=int(getattr(config, "mlm_mask_seed", 0)),
        sequence_ids=sequence_ids,
        current_epoch=current_epoch,
        segment_boundary_token_id=int(
            getattr(config, "mlm_segment_boundary_token_id", 1)
        ),
    )
    decoder_attention_mask = attention_mask.index_select(0, row_indices)
    target_input_ids = input_ids.index_select(0, row_indices)
    decoder_base = base_model.embed_tokens(
        target_input_ids,
        attention_mask=decoder_attention_mask,
    )
    decoder_base = apply_mlm_mask_latent(decoder_base, mlm_mask, base_model)
    return (
        decoder_base,
        mlm_mask.to(torch.float32),
        target_input_ids,
        decoder_attention_mask,
        row_indices,
    )


def apply_mlm_mask_latent(
    clean: torch.Tensor,
    mlm_mask: torch.Tensor,
    model,
) -> torch.Tensor:
    if not hasattr(model, "mlm_mask_latent_value"):
        raise ValueError("mlm_mask_latent requires a BabyLMELF model instance.")
    latent = model.mlm_mask_latent_value(
        device=clean.device,
        dtype=clean.dtype,
    )
    apply_mask = mlm_mask.to(device=clean.device, dtype=torch.bool).unsqueeze(-1)
    return torch.where(apply_mask, latent.view(1, 1, -1), clean)


def ce_loss_and_accuracy(
    input_ids: torch.Tensor,
    decoder_logits: torch.Tensor,
    ce_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    ce_per_token = F.cross_entropy(
        decoder_logits.float().reshape(-1, decoder_logits.size(-1)),
        input_ids.reshape(-1),
        reduction="none",
    ).view_as(input_ids)
    ce_losses = ce_per_token[ce_mask.bool()]
    ce = ce_losses.mean() if ce_losses.numel() > 0 else ce_per_token.sum() * 0.0
    correct = decoder_logits.argmax(dim=-1).eq(input_ids).to(torch.float32)
    accuracy = (correct * ce_mask).sum() / ce_mask.sum().clamp_min(1.0)
    return ce, accuracy


def _make_one_per_segment_mlm_input(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    mask_token_id: int | None,
    special_token_count: int,
    excluded_token_ids: tuple[int, ...] | list[int] = (),
    mask_schedule: str = "random",
    mask_seed: int = 0,
    sequence_ids: torch.Tensor | None = None,
    current_epoch: int = 0,
    segment_boundary_token_id: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mask_token_id is None:
        raise ValueError("MLM decoder objective requires model.config.mask_token_id.")

    device = input_ids.device
    sequence_length = input_ids.size(1)
    mask_schedule = str(mask_schedule or "random").lower()
    if mask_schedule not in {"cyclic", "random"}:
        raise ValueError(
            "diffusion.mlm_mask_schedule must be one of: 'cyclic', 'random', "
            f"got {mask_schedule!r}"
        )
    if sequence_ids is None:
        sequence_ids = torch.arange(input_ids.size(0), device=device, dtype=torch.long)
    else:
        sequence_ids = sequence_ids.to(device=device, dtype=torch.long)
    maskable = attention_mask.bool() & input_ids.ge(special_token_count)
    if excluded_token_ids:
        excluded = torch.as_tensor(
            tuple(excluded_token_ids),
            device=device,
            dtype=input_ids.dtype,
        )
        maskable = maskable & ~torch.isin(input_ids, excluded)

    selected_input_ids: list[torch.Tensor] = []
    selected_masks: list[torch.Tensor] = []
    row_indices: list[torch.Tensor] = []
    for batch_idx in range(input_ids.size(0)):
        candidates = maskable[batch_idx].nonzero(as_tuple=False).flatten()
        if candidates.numel() == 0:
            continue

        candidate_groups = _candidate_segment_groups(
            input_ids[batch_idx],
            candidates,
            segment_boundary_token_id=segment_boundary_token_id,
        )
        if not candidate_groups:
            continue

        selected_per_segment: list[torch.Tensor] = []
        sequence_id = int(sequence_ids[batch_idx].item())
        for segment_idx, group in enumerate(candidate_groups):
            target_count = min(
                _step10_then_step20_mask_count(group.numel()),
                group.numel(),
            )
            if target_count <= 0:
                continue
            token_indices = _select_segment_token_indices(
                group,
                target_count,
                schedule=mask_schedule,
                current_epoch=current_epoch,
                sequence_id=sequence_id,
                segment_id=segment_idx,
                seed=mask_seed,
            )
            selected_per_segment.append(group[token_indices])

        if not selected_per_segment:
            continue
        selected = torch.cat(selected_per_segment)

        rows = input_ids[batch_idx].unsqueeze(0).clone()
        masks = torch.zeros(
            1,
            sequence_length,
            device=device,
            dtype=torch.bool,
        )
        masks[:, candidates[selected]] = True
        rows[masks] = mask_token_id
        selected_input_ids.append(rows)
        selected_masks.append(masks)
        row_indices.append(
            torch.full((1,), batch_idx, device=device, dtype=torch.long)
        )

    if not selected_input_ids:
        return (
            input_ids.new_empty((0, sequence_length)),
            torch.zeros((0, sequence_length), device=device, dtype=torch.bool),
            torch.empty((0,), device=device, dtype=torch.long),
        )

    return (
        torch.cat(selected_input_ids, dim=0),
        torch.cat(selected_masks, dim=0),
        torch.cat(row_indices, dim=0),
    )


def make_one_per_segment_step10_then_step20_mlm_input(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    mask_token_id: int | None,
    special_token_count: int,
    excluded_token_ids: tuple[int, ...] | list[int] = (),
    mask_schedule: str = "random",
    mask_seed: int = 0,
    sequence_ids: torch.Tensor | None = None,
    current_epoch: int = 0,
    segment_boundary_token_id: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _make_one_per_segment_mlm_input(
        input_ids,
        attention_mask,
        mask_token_id=mask_token_id,
        special_token_count=special_token_count,
        excluded_token_ids=excluded_token_ids,
        mask_schedule=mask_schedule,
        mask_seed=mask_seed,
        sequence_ids=sequence_ids,
        current_epoch=current_epoch,
        segment_boundary_token_id=segment_boundary_token_id,
    )


def _step10_then_step20_mask_count(n: int) -> int:
    n = int(n)
    if n <= 0:
        return 0
    if n <= 40:
        return (n + 9) // 10
    return 4 + ((n - 40 + 19) // 20)


def _candidate_segment_groups(
    input_ids: torch.Tensor,
    candidates: torch.Tensor,
    *,
    segment_boundary_token_id: int,
) -> list[torch.Tensor]:
    if candidates.numel() == 0:
        return []
    boundaries = input_ids.eq(segment_boundary_token_id).nonzero(
        as_tuple=False
    ).flatten()
    if boundaries.numel() == 0:
        return [torch.arange(candidates.numel(), device=candidates.device)]

    candidate_indices = torch.arange(candidates.numel(), device=candidates.device)
    segment_ids = torch.bucketize(candidates, boundaries, right=True)
    groups: list[torch.Tensor] = []
    for segment_id in segment_ids.unique(sorted=True):
        group = candidate_indices[segment_ids.eq(segment_id)]
        if group.numel() > 0:
            groups.append(group)
    return groups


def _select_segment_token_indices(
    group: torch.Tensor,
    target_count: int,
    *,
    schedule: str,
    current_epoch: int,
    sequence_id: int,
    segment_id: int,
    seed: int,
) -> torch.Tensor:
    if schedule == "random":
        return torch.randperm(group.numel(), device=group.device)[:target_count]
    group_len = group.numel()
    base_offsets = (
        torch.arange(target_count, device=group.device, dtype=torch.long)
        * group_len
    ) // target_count
    start = (
        _cyclic_mask_start(
            group_len,
            sequence_id=sequence_id,
            segment_id=segment_id,
            seed=seed,
        )
        + int(current_epoch)
    ) % group_len
    return (start + base_offsets) % group_len


def _cyclic_mask_start(
    group_len: int,
    *,
    sequence_id: int,
    segment_id: int,
    seed: int,
) -> int:
    raw = (
        int(sequence_id) * 1_000_003
        + int(segment_id) * 97_531
        + int(seed) * 17_917
    )
    return raw % int(group_len)


def make_mlm_input(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    mask_token_id: int | None,
    special_token_count: int,
    excluded_token_ids: tuple[int, ...] | list[int] = (),
    strategy: str | None = None,
    mask_schedule: str = "random",
    mask_seed: int = 0,
    sequence_ids: torch.Tensor | None = None,
    current_epoch: int = 0,
    segment_boundary_token_id: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    strategy = str(strategy or "one_per_segment_step10_then_step20").lower()
    if strategy == "one_per_segment_step10_then_step20":
        return make_one_per_segment_step10_then_step20_mlm_input(
            input_ids,
            attention_mask,
            mask_token_id=mask_token_id,
            special_token_count=special_token_count,
            excluded_token_ids=excluded_token_ids,
            mask_schedule=mask_schedule,
            mask_seed=mask_seed,
            sequence_ids=sequence_ids,
            current_epoch=current_epoch,
            segment_boundary_token_id=segment_boundary_token_id,
        )
    raise ValueError(
        "diffusion.mlm_mask_strategy must be "
        "'one_per_segment_step10_then_step20', "
        f"got {strategy!r}"
    )


@torch.no_grad()
def eval_step(model, batch: dict[str, torch.Tensor], diffusion_config) -> dict[str, float]:
    return train_step(model, batch, diffusion_config).metrics
