from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SpanCorruptionBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def make_t5_span_corruption_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    pad_token_id: int,
    eos_token_id: int,
    sentinel_start_id: int,
    sentinel_count: int,
    noise_density: float,
    mean_noise_span_length: float,
    special_token_count: int,
    generator: torch.Generator | None = None,
) -> SpanCorruptionBatch:
    """Create T5-style span-corruption inputs and labels.

    The base tokenizer never stores exported ``<extra_id_*>`` tokens. Sentinel
    ids are internal ids above the base vocab and are used only by the scratch
    encoder pretraining objective.
    """

    corrupted_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    for row_ids, row_mask in zip(input_ids.cpu(), attention_mask.cpu()):
        valid = row_mask.bool()
        clean = row_ids[valid].long()
        noise_mask = _sample_noise_mask(
            clean,
            noise_density=noise_density,
            mean_noise_span_length=mean_noise_span_length,
            special_token_count=special_token_count,
            generator=generator,
        )
        corrupted, labels = _corrupt_one_sequence(
            clean,
            noise_mask,
            eos_token_id=eos_token_id,
            sentinel_start_id=sentinel_start_id,
            sentinel_count=sentinel_count,
        )
        corrupted_rows.append(corrupted)
        label_rows.append(labels)

    max_input_len = max(row.numel() for row in corrupted_rows)
    max_label_len = max(row.numel() for row in label_rows)
    batch_input = input_ids.new_full(
        (input_ids.size(0), max_input_len),
        pad_token_id,
    )
    batch_attention = attention_mask.new_zeros((input_ids.size(0), max_input_len))
    batch_labels = input_ids.new_full((input_ids.size(0), max_label_len), -100)

    for index, row in enumerate(corrupted_rows):
        batch_input[index, : row.numel()] = row.to(batch_input.device)
        batch_attention[index, : row.numel()] = 1
    for index, row in enumerate(label_rows):
        batch_labels[index, : row.numel()] = row.to(batch_labels.device)

    return SpanCorruptionBatch(
        input_ids=batch_input.to(input_ids.device),
        attention_mask=batch_attention.to(attention_mask.device),
        labels=batch_labels.to(input_ids.device),
    )


def _sample_noise_mask(
    input_ids: torch.Tensor,
    *,
    noise_density: float,
    mean_noise_span_length: float,
    special_token_count: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    maskable = input_ids.ge(special_token_count)
    maskable_indices = maskable.nonzero(as_tuple=False).flatten()
    noise_mask = torch.zeros(input_ids.shape, dtype=torch.bool)
    if maskable_indices.numel() == 0:
        return noise_mask

    target_tokens = max(1, int(round(maskable_indices.numel() * noise_density)))
    span_length = max(1, int(round(mean_noise_span_length)))
    max_spans = max(1, min(100, int(round(target_tokens / span_length)) + 1))
    perm = torch.randperm(maskable_indices.numel(), generator=generator)
    spans = 0
    for start in maskable_indices[perm].tolist():
        if noise_mask.sum().item() >= target_tokens or spans >= max_spans:
            break
        if noise_mask[start]:
            continue
        end = start
        while (
            end < input_ids.numel()
            and bool(maskable[end])
            and not bool(noise_mask[end])
            and end - start < span_length
            and noise_mask.sum().item() < target_tokens
        ):
            noise_mask[end] = True
            end += 1
        spans += 1
    return noise_mask


def _corrupt_one_sequence(
    input_ids: torch.Tensor,
    noise_mask: torch.Tensor,
    *,
    eos_token_id: int,
    sentinel_start_id: int,
    sentinel_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    corrupted: list[int] = []
    labels: list[int] = []
    sentinel_offset = 0
    index = 0
    while index < input_ids.numel():
        if not bool(noise_mask[index]):
            corrupted.append(int(input_ids[index]))
            index += 1
            continue

        sentinel_id = min(
            sentinel_start_id + sentinel_offset,
            sentinel_start_id + sentinel_count - 1,
        )
        corrupted.append(sentinel_id)
        labels.append(sentinel_id)
        while index < input_ids.numel() and bool(noise_mask[index]):
            labels.append(int(input_ids[index]))
            index += 1
        sentinel_offset += 1

    corrupted.append(eos_token_id)
    labels.append(eos_token_id)
    return (
        torch.tensor(corrupted, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )
