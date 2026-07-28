from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F


@dataclass
class StepOutput:
    loss: torch.Tensor
    metrics: dict[str, float]


def _distributed_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _distributed_world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def unwrap_model(model):
    return getattr(model, "module", model)


def forward_model(
    model,
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    attention_mask: torch.Tensor,
    segment_ids: torch.Tensor | None,
    self_cond_cfg_scale: torch.Tensor,
    decoder_step_active: torch.Tensor,
):
    kwargs = {
        "attention_mask": attention_mask,
        "self_cond_cfg_scale": self_cond_cfg_scale,
        "decoder_step_active": decoder_step_active,
    }
    if segment_ids is not None:
        kwargs["segment_ids"] = segment_ids
    return model(x, t, **kwargs)


def make_target_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    special_token_count: int,
    excluded_token_ids: tuple[int, ...] | list[int] = (),
) -> torch.Tensor:
    """Return positions that may contribute a lexical-token training target."""
    target_mask = attention_mask.bool() & input_ids.ge(special_token_count)
    if excluded_token_ids:
        excluded = torch.as_tensor(
            tuple(excluded_token_ids),
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        target_mask = target_mask & ~torch.isin(input_ids, excluded)
    return target_mask


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


@dataclass(frozen=True)
class BertMLMCorruption:
    corrupted_input_ids: torch.Tensor
    target_mask: torch.Tensor
    mask_positions: torch.Tensor
    random_positions: torch.Tensor
    unchanged_positions: torch.Tensor


def make_bert_15_80_10_10_mlm_input(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    vocab_size: int,
    special_token_count: int,
    excluded_token_ids: tuple[int, ...] | list[int] = (),
    target_probability: float = 0.15,
    mask_probability: float = 0.80,
    random_probability: float = 0.10,
    unchanged_probability: float = 0.10,
) -> BertMLMCorruption:
    probabilities = (
        float(target_probability),
        float(mask_probability),
        float(random_probability),
        float(unchanged_probability),
    )
    if any(probability < 0.0 or probability > 1.0 for probability in probabilities):
        raise ValueError("BERT corruption probabilities must be in [0, 1].")
    replacement_total = probabilities[1] + probabilities[2] + probabilities[3]
    if abs(replacement_total - 1.0) > 1.0e-8:
        raise ValueError(
            "BERT mask/random/unchanged probabilities must sum to 1.0."
        )
    eligible = make_target_mask(
        input_ids,
        attention_mask,
        special_token_count=special_token_count,
        excluded_token_ids=excluded_token_ids,
    )
    target_mask = eligible & (
        torch.rand(input_ids.shape, device=input_ids.device)
        < float(target_probability)
    )
    replacement_draw = torch.rand(input_ids.shape, device=input_ids.device)
    mask_cutoff = float(mask_probability)
    random_cutoff = mask_cutoff + float(random_probability)
    mask_positions = target_mask & (replacement_draw < mask_cutoff)
    random_positions = (
        target_mask
        & (replacement_draw >= mask_cutoff)
        & (replacement_draw < random_cutoff)
    )
    unchanged_positions = target_mask & ~mask_positions & ~random_positions

    corrupted_input_ids = input_ids.clone()
    random_count = int(random_positions.sum().item())
    if random_count > 0:
        legal_vocabulary = torch.arange(
            int(special_token_count),
            int(vocab_size),
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        excluded = {
            int(token_id)
            for token_id in excluded_token_ids
            if int(special_token_count) <= int(token_id) < int(vocab_size)
        }
        if excluded:
            legal_mask = torch.ones(
                legal_vocabulary.numel(),
                device=input_ids.device,
                dtype=torch.bool,
            )
            for token_id in excluded:
                legal_mask &= legal_vocabulary.ne(token_id)
            legal_vocabulary = legal_vocabulary[legal_mask]
        if legal_vocabulary.numel() == 0:
            raise ValueError(
                "BERT random-token replacement has no legal lexical tokens."
            )
        sampled_indices = torch.randint(
            legal_vocabulary.numel(),
            (random_count,),
            device=input_ids.device,
        )
        corrupted_input_ids[random_positions] = legal_vocabulary[sampled_indices]

    return BertMLMCorruption(
        corrupted_input_ids=corrupted_input_ids,
        target_mask=target_mask,
        mask_positions=mask_positions,
        random_positions=random_positions,
        unchanged_positions=unchanged_positions,
    )


def make_one_per_segment_step10_then_step20_mlm_input(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    mask_token_id: int | None,
    special_token_count: int,
    excluded_token_ids: tuple[int, ...] | list[int] = (),
    mask_seed: int = 0,
    sequence_ids: torch.Tensor | None = None,
    current_epoch: int = 0,
    segment_boundary_token_id: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mask_token_id is None:
        raise ValueError("MLM decoder objective requires model.config.mask_token_id.")

    device = input_ids.device
    sequence_length = input_ids.size(1)
    if sequence_ids is None:
        sequence_ids = torch.arange(input_ids.size(0), device=device, dtype=torch.long)
    else:
        sequence_ids = sequence_ids.to(device=device, dtype=torch.long)
    maskable = make_target_mask(
        input_ids,
        attention_mask,
        special_token_count=special_token_count,
        excluded_token_ids=excluded_token_ids,
    )

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
    current_epoch: int,
    sequence_id: int,
    segment_id: int,
    seed: int,
) -> torch.Tensor:
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
    return low * torch.exp(
        uniform * torch.log(torch.tensor(high / low, device=device))
    ) - 1.0


def train_step_elf(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    config,
    *,
    segment_ids: torch.Tensor | None = None,
    sequence_ids: torch.Tensor | None = None,
    current_epoch: int = 0,
) -> StepOutput:
    batch_size = input_ids.size(0)
    device = input_ids.device
    base_model = unwrap_model(model)
    decoder_objective = config.decoder_objective
    loss_target_mask = make_target_mask(
        input_ids,
        attention_mask,
        special_token_count=config.targets.special_token_count,
        excluded_token_ids=config.targets.excluded_token_ids,
    )
    decoder_eligible_rows = loss_target_mask.any(dim=-1)

    clean = base_model.embed_tokens(
        input_ids,
        attention_mask=attention_mask,
        segment_ids=segment_ids,
    )
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
        (torch.rand(batch_size, device=device) < config.decoder_probability)
        & decoder_eligible_rows
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

    self_condition = torch.zeros_like(clean)
    self_condition_rows = (
        self_condition_mask.view(batch_size).bool() & ~decoder_active.bool()
    ).nonzero(as_tuple=False).flatten()
    if self_condition_rows.numel() > 0:
        self_condition_z = denoiser_z.index_select(0, self_condition_rows)
        self_condition_t = t.index_select(0, self_condition_rows)
        self_condition_attention = attention_mask.index_select(
            0,
            self_condition_rows,
        )
        self_condition_segments = (
            segment_ids.index_select(0, self_condition_rows)
            if segment_ids is not None
            else None
        )
        self_condition_cfg = cfg_scale.index_select(0, self_condition_rows)
        auxiliary_decoder_active = torch.zeros(
            self_condition_rows.numel(),
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            unconditioned_prediction, _ = forward_model(
                base_model,
                torch.cat(
                    (self_condition_z, torch.zeros_like(self_condition_z)),
                    dim=-1,
                ),
                self_condition_t,
                attention_mask=self_condition_attention,
                segment_ids=self_condition_segments,
                self_cond_cfg_scale=self_condition_cfg,
                decoder_step_active=auxiliary_decoder_active,
            )
            unconditioned_velocity = prediction_to_velocity(
                unconditioned_prediction,
                self_condition_z,
                self_condition_t,
                config.t_eps,
            )
            conditioned_prediction, _ = forward_model(
                base_model,
                torch.cat((self_condition_z, unconditioned_prediction), dim=-1),
                self_condition_t,
                attention_mask=self_condition_attention,
                segment_ids=self_condition_segments,
                self_cond_cfg_scale=self_condition_cfg,
                decoder_step_active=auxiliary_decoder_active,
            )
            conditioned_velocity = prediction_to_velocity(
                conditioned_prediction,
                self_condition_z,
                self_condition_t,
                config.t_eps,
            )
            guidance = (1.0 - 1.0 / self_condition_cfg.view(-1, 1, 1)) * (
                conditioned_velocity - unconditioned_velocity
            )
        self_condition.index_copy_(
            0,
            self_condition_rows,
            unconditioned_prediction,
        )
        guidance_target = torch.zeros_like(velocity_target)
        guidance_target.index_copy_(0, self_condition_rows, guidance)
        velocity_target = velocity_target + guidance_target

    z_parts: list[torch.Tensor] = []
    self_condition_parts: list[torch.Tensor] = []
    t_parts: list[torch.Tensor] = []
    attention_parts: list[torch.Tensor] = []
    segment_parts: list[torch.Tensor] = []
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
        decoder_input_ids = input_ids.index_select(0, decoder_rows)
        decoder_input_attention = attention_mask.index_select(0, decoder_rows)
        decoder_input_segments = (
            segment_ids.index_select(0, decoder_rows)
            if segment_ids is not None
            else None
        )
        if decoder_objective == "official_noisy_ce":
            (
                decoder_z,
                ce_token_mask,
                decoder_target_ids,
                decoder_attention_mask,
                selected_decoder_rows,
            ) = build_official_decoder_inputs(
                model,
                decoder_input_ids,
                decoder_input_attention,
                config,
                segment_ids=decoder_input_segments,
            )
        else:
            (
                decoder_z,
                ce_token_mask,
                decoder_target_ids,
                decoder_attention_mask,
                selected_decoder_rows,
            ) = build_mlm_decoder_inputs(
                model,
                decoder_input_ids,
                decoder_input_attention,
                config,
                segment_ids=decoder_input_segments,
                sequence_ids=(
                    sequence_ids.to(device=device).index_select(0, decoder_rows)
                    if sequence_ids is not None
                    else None
                ),
                current_epoch=current_epoch,
            )
        decoder_count = decoder_z.size(0)
        if decoder_count > 0:
            selected_global_rows = decoder_rows.index_select(
                0,
                selected_decoder_rows,
            )
            z_parts.append(decoder_z)
            self_condition_parts.append(torch.zeros_like(decoder_z))
            t_parts.append(torch.ones(decoder_count, device=device, dtype=dtype))
            attention_parts.append(decoder_attention_mask)
            if segment_ids is not None:
                segment_parts.append(
                    segment_ids.index_select(0, selected_global_rows)
                )
            cfg_parts.append(torch.ones(decoder_count, device=device, dtype=dtype))
            decoder_active_parts.append(
                torch.ones(decoder_count, device=device, dtype=dtype)
            )

    if denoiser_rows.numel() > 0:
        z_parts.append(denoiser_z.index_select(0, denoiser_rows))
        self_condition_parts.append(self_condition.index_select(0, denoiser_rows))
        t_parts.append(t.index_select(0, denoiser_rows))
        attention_parts.append(attention_mask.index_select(0, denoiser_rows))
        if segment_ids is not None:
            segment_parts.append(segment_ids.index_select(0, denoiser_rows))
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
    mixed_segment_ids = (
        torch.cat(segment_parts, dim=0) if segment_ids is not None else None
    )
    mixed_cfg_scale = torch.cat(cfg_parts, dim=0)
    mixed_decoder_active = torch.cat(decoder_active_parts, dim=0)

    prediction, decoder_logits = forward_model(
        model,
        torch.cat((mixed_z, mixed_self_condition), dim=-1),
        mixed_t,
        attention_mask=mixed_attention_mask,
        segment_ids=mixed_segment_ids,
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
        flow_target_mask = loss_target_mask.to(torch.float32)
        l2_mask = flow_target_mask.index_select(0, denoiser_rows)
        flow = (l2_per_token * l2_mask).sum() / l2_mask.sum().clamp_min(1.0)
    else:
        flow = zero_from_forward

    lexical_mask = loss_target_mask.to(torch.float32)
    decoder_mask_2d = decoder_active.view(-1, 1)
    decoder_valid_mask = lexical_mask * decoder_mask_2d
    l2_mask_original = lexical_mask * (1.0 - decoder_mask_2d)
    denominator = lexical_mask.sum().clamp_min(1.0)
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


def build_mlm_decoder_inputs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    config,
    *,
    segment_ids: torch.Tensor | None = None,
    sequence_ids: torch.Tensor | None = None,
    current_epoch: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    base_model = unwrap_model(model)
    corruption = config.corruption
    corruption_name = corruption.type
    if corruption_name == "bert_15_80_10_10":
        vocab_size = getattr(base_model.config, "base_vocab_size", None)
        if vocab_size is None:
            vocab_size = getattr(base_model.config, "vocab_size", None)
        if vocab_size is None:
            vocab_size = base_model.vocab_size
        bert_corruption = make_bert_15_80_10_10_mlm_input(
            input_ids,
            attention_mask,
            vocab_size=int(vocab_size),
            special_token_count=config.targets.special_token_count,
            excluded_token_ids=config.targets.excluded_token_ids,
            target_probability=float(corruption.target_probability),
            mask_probability=float(corruption.mask_probability),
            random_probability=float(corruption.random_probability),
            unchanged_probability=float(corruption.unchanged_probability),
        )
        row_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        decoder_base = base_model.embed_tokens(
            bert_corruption.corrupted_input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
        )
        decoder_base = apply_mlm_mask_latent(
            decoder_base,
            bert_corruption.mask_positions,
            base_model,
        )
        return (
            decoder_base,
            bert_corruption.target_mask.to(torch.float32),
            input_ids,
            attention_mask,
            row_indices,
        )
    if corruption_name != "step10_step20":
        raise ValueError(
            "objective.corruption.type must be one of "
            "'step10_step20' or 'bert_15_80_10_10'; "
            f"got {corruption_name!r}."
        )

    _masked_ids_for_selection, mlm_mask, row_indices = (
        make_one_per_segment_step10_then_step20_mlm_input(
            input_ids,
            attention_mask,
            mask_token_id=base_model.config.mask_token_id,
            special_token_count=config.targets.special_token_count,
            excluded_token_ids=config.targets.excluded_token_ids,
            mask_seed=int(corruption.seed),
            sequence_ids=sequence_ids,
            current_epoch=current_epoch,
            segment_boundary_token_id=int(corruption.segment_boundary_token_id),
        )
    )
    decoder_attention_mask = attention_mask.index_select(0, row_indices)
    decoder_segment_ids = (
        segment_ids.index_select(0, row_indices)
        if segment_ids is not None
        else None
    )
    target_input_ids = input_ids.index_select(0, row_indices)
    decoder_base = base_model.embed_tokens(
        target_input_ids,
        attention_mask=decoder_attention_mask,
        segment_ids=decoder_segment_ids,
    )
    decoder_base = apply_mlm_mask_latent(decoder_base, mlm_mask, base_model)
    return (
        decoder_base,
        mlm_mask.to(torch.float32),
        target_input_ids,
        decoder_attention_mask,
        row_indices,
    )


def build_official_decoder_inputs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    config,
    *,
    segment_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the official ELF final-step decoder corruption.

    Every lexical target receives independent logit-normal Gaussian corruption.
    CE is evaluated at all eligible lexical positions in selected decoder rows.
    """
    base_model = unwrap_model(model)
    clean = base_model.embed_tokens(
        input_ids,
        attention_mask=attention_mask,
        segment_ids=segment_ids,
    )
    logits = (
        torch.randn(
            (*clean.shape[:-1], 1),
            device=clean.device,
            dtype=clean.dtype,
        )
        * float(config.decoder_p_std)
        + float(config.decoder_p_mean)
    )
    interpolation = torch.sigmoid(logits)
    noise = torch.randn_like(clean) * float(config.decoder_noise_scale)
    decoder_z = interpolation * clean + (1.0 - interpolation) * noise
    ce_mask = make_target_mask(
        input_ids,
        attention_mask,
        special_token_count=config.targets.special_token_count,
        excluded_token_ids=config.targets.excluded_token_ids,
    )
    row_indices = torch.arange(input_ids.size(0), device=input_ids.device)
    return (
        decoder_z,
        ce_mask.to(torch.float32),
        input_ids,
        attention_mask,
        row_indices,
    )


def sample_mdlm_timesteps(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    sampling_eps: float,
    antithetic: bool,
    rank: int = 0,
    world_size: int = 1,
) -> torch.Tensor:
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if not 0.0 < sampling_eps < 1.0:
        raise ValueError("mdlm_sampling_eps must be in (0, 1).")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank {rank} is invalid for world_size {world_size}.")

    if antithetic:
        global_batch = batch_size * world_size
        strata = rank * batch_size + torch.arange(
            batch_size,
            device=device,
            dtype=torch.float32,
        )
        unit = (strata + torch.rand(batch_size, device=device)) / global_batch
    else:
        unit = torch.rand(batch_size, device=device, dtype=torch.float32)
    return (sampling_eps + (1.0 - sampling_eps) * unit).to(dtype=dtype)


def loglinear_mask_probability(
    t: torch.Tensor,
    *,
    noise_eps: float,
) -> torch.Tensor:
    if not 0.0 <= noise_eps < 1.0:
        raise ValueError("mdlm_noise_eps must be in [0, 1).")
    return (1.0 - noise_eps) * t


def suppress_mask_logits(
    logits: torch.Tensor,
    *,
    mask_token_id: int,
) -> torch.Tensor:
    constrained = logits.float().clone()
    constrained[..., mask_token_id] = torch.finfo(constrained.dtype).min
    return constrained


def apply_subs_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    mask_positions: torch.Tensor,
    *,
    mask_token_id: int,
    special_token_count: int,
) -> torch.Tensor:
    constrained = logits.float().clone()
    minimum = torch.finfo(constrained.dtype).min
    constrained[..., mask_token_id] = minimum
    if special_token_count > 0:
        constrained[..., :special_token_count] = torch.where(
            mask_positions.unsqueeze(-1),
            torch.full_like(
                constrained[..., :special_token_count],
                minimum,
            ),
            constrained[..., :special_token_count],
        )
    carried = torch.full_like(constrained, minimum)
    carried.scatter_(-1, input_ids.unsqueeze(-1), 0.0)
    return torch.where(mask_positions.unsqueeze(-1), constrained, carried)


def train_step_standard_mdlm(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    config,
    *,
    segment_ids: torch.Tensor | None = None,
) -> StepOutput:
    if config.noise_schedule != "loglinear":
        raise ValueError("standard_mdlm requires noise_schedule='loglinear'.")
    if config.time_conditioning != "t":
        raise ValueError("standard_mdlm requires time_conditioning='t'.")

    base_model = unwrap_model(model)
    if base_model.flow_head is not None or base_model.self_cond_projection is not None:
        raise RuntimeError(
            "standard_mdlm must not instantiate flow or self-conditioning modules."
        )

    clean = base_model.embed_tokens(
        input_ids,
        attention_mask=attention_mask,
        segment_ids=segment_ids,
    )
    batch_size = input_ids.size(0)
    t = sample_mdlm_timesteps(
        batch_size,
        input_ids.device,
        clean.dtype,
        sampling_eps=float(config.sampling_eps),
        antithetic=bool(config.antithetic_sampling),
        rank=_distributed_rank(),
        world_size=_distributed_world_size(),
    )
    eligible = make_target_mask(
        input_ids,
        attention_mask,
        special_token_count=config.targets.special_token_count,
        excluded_token_ids=config.targets.excluded_token_ids,
    )
    mask_probability = loglinear_mask_probability(
        t.float(),
        noise_eps=float(config.noise_eps),
    )
    masked_positions = (
        torch.rand(input_ids.shape, device=input_ids.device)
        < mask_probability.unsqueeze(-1)
    ) & eligible
    corrupted = apply_mlm_mask_latent(clean, masked_positions, base_model)
    decoder_active = torch.ones(
        batch_size,
        device=input_ids.device,
        dtype=clean.dtype,
    )
    prediction, logits = forward_model(
        model,
        base_model.prepare_decoder_input(corrupted),
        t,
        attention_mask=attention_mask,
        segment_ids=segment_ids,
        self_cond_cfg_scale=torch.ones_like(decoder_active),
        decoder_step_active=decoder_active,
    )
    if prediction is not None:
        raise RuntimeError("standard_mdlm unexpectedly produced a flow prediction.")

    constrained_logits = suppress_mask_logits(
        logits,
        mask_token_id=base_model.config.mask_token_id,
    )
    ce_per_token = F.cross_entropy(
        constrained_logits.reshape(-1, constrained_logits.size(-1)),
        input_ids.reshape(-1),
        reduction="none",
    ).view_as(input_ids)
    mask_float = masked_positions.to(torch.float32)
    eligible_count = eligible.sum().to(torch.float32)
    masked_count = mask_float.sum()
    inverse_t = t.float().reciprocal().unsqueeze(-1)
    connected_zero = logits.float().sum() * 0.0
    mdlm_nelbo = (
        (ce_per_token * mask_float * inverse_t).sum()
        / eligible_count.clamp_min(1.0)
    ) + connected_zero
    masked_ce = (
        (ce_per_token * mask_float).sum() / masked_count.clamp_min(1.0)
    ) + connected_zero
    correct = constrained_logits.argmax(dim=-1).eq(input_ids).to(torch.float32)
    masked_accuracy = (correct * mask_float).sum() / masked_count.clamp_min(1.0)
    mask_rate = masked_count / eligible_count.clamp_min(1.0)

    return StepOutput(
        loss=mdlm_nelbo,
        metrics={
            "loss": float(mdlm_nelbo.detach()),
            "flow": 0.0,
            "ce": float(masked_ce.detach()),
            "acc": float(masked_accuracy.detach()),
            "decode_frac": 1.0,
            "mdlm_nelbo": float(mdlm_nelbo.detach()),
            "masked_ce": float(masked_ce.detach()),
            "masked_acc": float(masked_accuracy.detach()),
            "mask_rate": float(mask_rate.detach()),
        },
    )


def train_step(
    model,
    batch: dict[str, torch.Tensor],
    config,
    current_epoch: int = 0,
) -> StepOutput:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    segment_ids = batch.get("segment_ids")
    objective = config.decoder_objective
    if objective == "standard_mdlm":
        return train_step_standard_mdlm(
            model,
            input_ids,
            attention_mask,
            config,
            segment_ids=segment_ids,
        )
    if objective not in {"token_mlm", "official_noisy_ce"}:
        raise ValueError(f"Unsupported training objective: {objective!r}.")
    return train_step_elf(
        model,
        input_ids,
        attention_mask,
        config,
        segment_ids=segment_ids,
        sequence_ids=batch.get("sequence_id"),
        current_epoch=current_epoch,
    )
