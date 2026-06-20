from __future__ import annotations

from dataclasses import dataclass

import torch

from babylm_elf.diffusion.noising import add_noise
from babylm_elf.diffusion.targets import prediction_to_x0
from babylm_elf.training.losses import decode_ce_loss, flow_loss, token_accuracy


@dataclass
class StepOutput:
    loss: torch.Tensor
    metrics: dict[str, float]


def train_step(model, batch: dict[str, torch.Tensor], diffusion_config) -> StepOutput:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    valid_mask = attention_mask.bool()

    x0 = model.embed_tokens(input_ids)
    diffusion = add_noise(
        x0,
        prediction_type=diffusion_config.prediction_type,
        time_schedule=diffusion_config.time_schedule,
        noise_scale=diffusion_config.noise_scale,
    )
    prediction = model.denoise(diffusion.z_t, diffusion.t, attention_mask=attention_mask)
    x0_hat = prediction_to_x0(prediction, diffusion, diffusion_config.prediction_type)
    decoder_logits = model.decode_embeddings(x0_hat)

    flow = flow_loss(prediction, diffusion.target, valid_mask)
    ce = decode_ce_loss(decoder_logits, input_ids, valid_mask)
    loss = diffusion_config.flow_loss_weight * flow + diffusion_config.decode_loss_weight * ce

    accuracy = token_accuracy(decoder_logits, input_ids, valid_mask)

    return StepOutput(
        loss=loss,
        metrics={
            "loss": float(loss.detach()),
            "flow": float(flow.detach()),
            "ce": float(ce.detach()),
            "acc": float(accuracy.detach()),
        },
    )


@torch.no_grad()
def eval_step(model, batch: dict[str, torch.Tensor], diffusion_config) -> dict[str, float]:
    output = train_step(model, batch, diffusion_config)
    return output.metrics
