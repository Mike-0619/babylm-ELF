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


def runtime_seed_for_rank(base_seed: int, rank: int) -> int:
    if rank < 0:
        raise ValueError(f"rank must be non-negative, got {rank}.")
    return int(base_seed) + int(rank)


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
            aux_lr=getattr(config, "aux_learning_rate", None),
            encoder_lr_multiplier=getattr(config, "encoder_lr_multiplier", 1.0),
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )

    no_decay = ("bias", "norm", "embedding", "latent")
    grouped_params: dict[tuple[bool, bool], list[torch.nn.Parameter]] = {
        (False, False): [],
        (False, True): [],
        (True, False): [],
        (True, True): [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_encoder = name.startswith("scratch_encoder.")
        use_no_decay = any(nd in name.lower() for nd in no_decay)
        grouped_params[(is_encoder, use_no_decay)].append(param)
    encoder_lr_multiplier = getattr(config, "encoder_lr_multiplier", 1.0)
    groups = []
    for (is_encoder, use_no_decay), params in grouped_params.items():
        if not params:
            continue
        groups.append(
            {
                "params": params,
                "lr": config.learning_rate * (encoder_lr_multiplier if is_encoder else 1.0),
                "weight_decay": 0.0 if use_no_decay else config.weight_decay,
            }
        )
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
    def __init__(
        self,
        model: torch.nn.Module,
        decay: float,
        *,
        warmup: bool = True,
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"EMA decay must be in [0, 1), got {decay}.")
        self.decay = float(decay)
        self.warmup = bool(warmup)
        self.num_updates = 0
        self.current_decay = 0.0
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.num_updates += 1
        self.current_decay = self.decay_for_update(self.num_updates)
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(
                    parameter.detach(),
                    1.0 - self.current_decay,
                )

    def decay_for_update(self, num_updates: int) -> float:
        if num_updates <= 0:
            raise ValueError(
                f"EMA num_updates must be positive, got {num_updates}."
            )
        if not self.warmup:
            return self.decay
        warmup_decay = (1.0 + num_updates) / (10.0 + num_updates)
        return min(self.decay, warmup_decay)

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "warmup": self.warmup,
            "num_updates": self.num_updates,
            "current_decay": self.current_decay,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state["decay"])
        self.warmup = bool(state.get("warmup", False))
        self.num_updates = int(state.get("num_updates", 0))
        self.current_decay = float(
            state.get(
                "current_decay",
                self.decay if self.num_updates > 0 else 0.0,
            )
        )
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


def resolve_ema_decay(
    reference_decay: float,
    reference_steps: int,
    total_optimizer_steps: int,
) -> float:
    if not 0.0 <= reference_decay < 1.0:
        raise ValueError(
            "EMA reference_decay must be in [0, 1), "
            f"got {reference_decay}."
        )
    if reference_steps <= 0:
        raise ValueError(
            f"EMA reference_steps must be positive, got {reference_steps}."
        )
    if total_optimizer_steps <= 0:
        raise ValueError(
            "EMA total_optimizer_steps must be positive, "
            f"got {total_optimizer_steps}."
        )
    return float(reference_decay) ** (
        float(reference_steps) / float(total_optimizer_steps)
    )
