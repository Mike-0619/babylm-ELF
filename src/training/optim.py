from __future__ import annotations

from dataclasses import dataclass
import math
import random

import torch
import torch.distributed as dist
import torch.nn as nn


def muon_with_aux_adam(
    model: nn.Module,
    lr: float,
    aux_lr: float | None = None,
    encoder_lr_multiplier: float = 1.0,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """ELF-style Muon optimizer: 2D params use Muon, others use auxiliary Adam."""
    aux_lr = lr if aux_lr is None else aux_lr
    try:
        import muon as _muon_module
        from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
    except ImportError as exc:
        raise ImportError(
            "Muon optimizer requires the 'muon-optimizer' package. "
            "Install it with: pip install muon-optimizer>=0.1.0"
        ) from exc

    def _nesterov_adam_update(grad, mu, nu, step, betas, eps):
        b1, b2 = betas
        mu.lerp_(grad, 1 - b1)
        nu.lerp_(grad.square(), 1 - b2)
        mu_hat = b1 * (mu / (1 - b1 ** (step + 1))) + (1 - b1) * (grad / (1 - b1 ** step))
        nu_hat = nu / (1 - b2 ** step)
        return mu_hat / (nu_hat.sqrt() + eps)

    _muon_module.adam_update = _nesterov_adam_update

    def _zeropower_via_newtonschulz5_fp32(grad, steps):
        a, b, c = (3.4445, -4.7750, 2.0315)
        x = grad.to(torch.float32)
        if grad.size(-2) > grad.size(-1):
            x = x.mT
        x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-8)
        for _ in range(steps):
            gram = x @ x.mT
            polynomial = b * gram + c * gram @ gram
            x = a * x + polynomial @ x
        if grad.size(-2) > grad.size(-1):
            x = x.mT
        return x

    _muon_module.zeropower_via_newtonschulz5 = _zeropower_via_newtonschulz5_fp32

    def _muon_update_optax(
        grad,
        momentum,
        step,
        beta=0.95,
        ns_steps=5,
        nesterov=True,
        flax_layout=False,
        update_momentum=True,
    ):
        if update_momentum:
            momentum.lerp_(grad, 1 - beta)
        if nesterov:
            mu_corr = momentum / (1 - beta ** (step + 1))
            g_corr = grad / (1 - beta ** step)
            update = beta * mu_corr + (1 - beta) * g_corr
        else:
            update = momentum / (1 - beta ** step)
        if update.ndim == 4:
            update = update.view(len(update), -1)
        update = _muon_module.zeropower_via_newtonschulz5(update, steps=ns_steps)
        rows, cols = grad.size(-2), grad.size(-1)
        if flax_layout:
            update *= max(1, cols / rows) ** 0.5
        else:
            update *= max(1, rows / cols) ** 0.5
        return update

    linear_weight_ids = set()
    for module in model.modules():
        if isinstance(module, nn.Linear) and module.weight is not None:
            linear_weight_ids.add(id(module.weight))

    muon_params = []
    adam_params = []
    encoder_muon_params = []
    encoder_adam_params = []
    muon_flax_layout = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_encoder = name.startswith("scratch_encoder.")
        if param.ndim == 2:
            if is_encoder:
                encoder_muon_params.append(param)
            else:
                muon_params.append(param)
            muon_flax_layout[id(param)] = id(param) not in linear_weight_ids
        else:
            if is_encoder:
                encoder_adam_params.append(param)
            else:
                adam_params.append(param)

    distributed = dist.is_available() and dist.is_initialized()
    base_cls = MuonWithAuxAdam if distributed else SingleDeviceMuonWithAuxAdam

    class _SafeMuonWithAuxAdam(base_cls):
        @torch.no_grad()
        def step(self, closure=None):
            loss = closure() if closure is not None else None
            for group in self.param_groups:
                for param in group["params"]:
                    if param.grad is None:
                        param.grad = torch.zeros_like(param)

            for group in self.param_groups:
                if group["use_muon"]:
                    params = group["params"]
                    if distributed:
                        world_size = dist.get_world_size()
                        rank = dist.get_rank()
                        for param_i, param in enumerate(params):
                            owner = param_i % world_size
                            state = self._muon_state(param)
                            state["step"] += 1
                            state["momentum_buffer"].lerp_(
                                param.grad,
                                1 - group["momentum"],
                            )
                            if rank == owner:
                                self._muon_step_one(
                                    param,
                                    group,
                                    state=state,
                                    momentum_already_updated=True,
                                    increment_step=False,
                                )
                            # Muon parameters have heterogeneous shapes, so they
                            # cannot share one all_gather collective. The owner
                            # updates each parameter, then broadcasts it in place.
                            dist.broadcast(param, src=owner)
                    else:
                        for param in params:
                            self._muon_step_one(param, group)
                else:
                    for param in group["params"]:
                        state = self.state[param]
                        if len(state) == 0:
                            state["exp_avg"] = torch.zeros_like(param)
                            state["exp_avg_sq"] = torch.zeros_like(param)
                            state["step"] = 0
                        state["step"] += 1
                        update = _muon_module.adam_update(
                            param.grad,
                            state["exp_avg"],
                            state["exp_avg_sq"],
                            state["step"],
                            group["betas"],
                            group["eps"],
                        )
                        param.mul_(1 - group["lr"] * group["weight_decay"])
                        param.add_(update, alpha=-group["lr"])
            return loss

        def _muon_state(self, param):
            state = self.state[param]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(param)
                state["step"] = 0
            return state

        def _muon_step_one(
            self,
            param,
            group,
            *,
            state=None,
            momentum_already_updated=False,
            increment_step=True,
        ):
            state = self._muon_state(param) if state is None else state
            if increment_step:
                state["step"] += 1
            update = _muon_update_optax(
                param.grad,
                state["momentum_buffer"],
                state["step"],
                beta=group["momentum"],
                flax_layout=muon_flax_layout.get(id(param), False),
                update_momentum=not momentum_already_updated,
            )
            param.mul_(1 - group["lr"] * group["weight_decay"])
            param.add_(update, alpha=-group["lr"])

    encoder_lr = lr * encoder_lr_multiplier
    encoder_aux_lr = aux_lr * encoder_lr_multiplier
    param_groups = []
    if muon_params:
        param_groups.append(
            {
                "params": muon_params,
                "lr": lr,
                "momentum": 0.95,
                "weight_decay": 0.0,
                "use_muon": True,
            }
        )
    if adam_params:
        param_groups.append(
            {
                "params": adam_params,
                "lr": aux_lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": 0.0,
                "use_muon": False,
            }
        )
    if encoder_muon_params:
        param_groups.append(
            {
                "params": encoder_muon_params,
                "lr": encoder_lr,
                "momentum": 0.95,
                "weight_decay": 0.0,
                "use_muon": True,
            }
        )
    if encoder_adam_params:
        param_groups.append(
            {
                "params": encoder_adam_params,
                "lr": encoder_aux_lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": 0.0,
                "use_muon": False,
            }
        )
    print(
        "Using MuonWithAuxAdam: "
        f"{len(muon_params)} non-encoder 2D params use Muon; "
        f"{len(adam_params)} non-encoder other params use Nesterov-Adam; "
        f"{len(encoder_muon_params)} encoder 2D params use Muon; "
        f"{len(encoder_adam_params)} encoder other params use Nesterov-Adam "
        f"(muon_lr={lr:g}, aux_lr={aux_lr:g}, "
        f"encoder_muon_lr={encoder_lr:g}, encoder_aux_lr={encoder_aux_lr:g}, "
        f"betas={betas}, "
        f"eps={eps:g}, weight_decay=0)"
    )
    return _SafeMuonWithAuxAdam(param_groups)


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
    optimizer_name = config.type.lower()
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
    raise ValueError(f"Unknown optimizer: {config.type}. Choose 'adamw' or 'muon'.")


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
