from __future__ import annotations

import math
import random
from dataclasses import dataclass
from contextlib import contextmanager
import torch

from babylm_elf.training.muon_utils import muon_with_aux_adam


@dataclass
class TrainState:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LambdaLR
    ema: "ExponentialMovingAverage | None" = None
    step: int = 0
    microbatches_seen: int = 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; refusing to train on CPU.")
        return torch.device("cuda")
    return torch.device(device)


def create_optimizer(model: torch.nn.Module, config) -> torch.optim.Optimizer:
    optimizer_name = config.optimizer.lower()
    if optimizer_name == "muon":
        return muon_with_aux_adam(
            model,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )

    no_decay = ("bias", "norm", "embedding")
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name.lower() for nd in no_decay):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            groups,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )
    raise ValueError(f"Unknown optimizer: {config.optimizer}. Choose 'adamw' or 'muon'.")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    schedule: str = "cosine",
    min_lr: float = 0.0,
):
    warmup_steps = max(1, warmup_steps)
    max_steps = max(warmup_steps + 1, max_steps)
    base_lr = optimizer.param_groups[0]["lr"]
    min_factor = min_lr / base_lr if base_lr > 0 else 0.0
    schedule = schedule.lower()

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        if schedule == "constant":
            return 1.0
        if schedule != "cosine":
            raise ValueError(f"Unknown LR schedule: {schedule}. Choose 'constant' or 'cosine'.")
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_factor + (1.0 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(parameter.detach(), 1.0 - self.decay)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.shadow = state["shadow"]

    def model_state_dict(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        state = model.state_dict()
        return {
            name: self.shadow.get(name, value).detach().clone()
            for name, value in state.items()
        }

    @contextmanager
    def average_parameters(self, model: torch.nn.Module):
        backup = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if name in self.shadow
        }
        try:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in self.shadow:
                        parameter.copy_(self.shadow[name])
            yield
        finally:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in backup:
                        parameter.copy_(backup[name])
