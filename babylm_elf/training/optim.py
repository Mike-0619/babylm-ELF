from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable

import torch

from babylm_elf.training.lamb import Lamb


@dataclass
class TrainState:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LambdaLR
    ema: "ExponentialMovingAverage | None" = None
    step: int = 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def create_optimizer(model: torch.nn.Module, config) -> torch.optim.Optimizer:
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
    if config.optimizer.lower() == "lamb":
        return Lamb(groups, lr=config.learning_rate, betas=(config.beta1, config.beta2), eps=config.eps)
    return torch.optim.AdamW(
        groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
    )


def create_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, max_steps: int):
    warmup_steps = max(1, warmup_steps)
    max_steps = max(warmup_steps + 1, max_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class ExponentialMovingAverage:
    def __init__(self, parameters: Iterable[torch.nn.Parameter], decay: float) -> None:
        self.decay = decay
        self.shadow = [p.detach().clone() for p in parameters if p.requires_grad]

    @torch.no_grad()
    def update(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        params = [p for p in parameters if p.requires_grad]
        for shadow, param in zip(self.shadow, params):
            shadow.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.shadow = state["shadow"]
