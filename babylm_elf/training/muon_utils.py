from __future__ import annotations

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

    def _muon_update_optax(grad, momentum, step, beta=0.95, ns_steps=5, nesterov=True, flax_layout=False):
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
                        pad_n = (-len(params)) % world_size
                        params_pad = params + [torch.empty_like(params[-1])] * pad_n
                        for base_i in range(0, len(params), world_size):
                            if base_i + dist.get_rank() < len(params):
                                param = params[base_i + dist.get_rank()]
                                self._muon_step_one(param, group)
                            dist.all_gather(
                                params_pad[base_i:base_i + world_size],
                                params_pad[base_i + dist.get_rank()],
                            )
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

        def _muon_step_one(self, param, group):
            state = self.state[param]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(param)
                state["step"] = 0
            state["step"] += 1
            update = _muon_update_optax(
                param.grad,
                state["momentum_buffer"],
                state["step"],
                beta=group["momentum"],
                flax_layout=muon_flax_layout.get(id(param), False),
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
