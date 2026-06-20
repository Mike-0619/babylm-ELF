from __future__ import annotations

import torch
import torch.nn.functional as F


def flow_loss(prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if not valid_mask.any():
        return prediction.sum() * 0.0
    return F.mse_loss(prediction[valid_mask].float(), target[valid_mask].float())


def decode_ce_loss(
    decoder_logits: torch.Tensor,
    input_ids: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if not valid_mask.any():
        return decoder_logits.sum() * 0.0
    return F.cross_entropy(decoder_logits[valid_mask].float(), input_ids[valid_mask])


@torch.no_grad()
def token_accuracy(
    decoder_logits: torch.Tensor,
    input_ids: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if not valid_mask.any():
        return decoder_logits.new_tensor(0.0)
    return decoder_logits[valid_mask].argmax(dim=-1).eq(input_ids[valid_mask]).float().mean()
