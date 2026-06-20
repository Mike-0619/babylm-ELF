from __future__ import annotations

from pathlib import Path
from dataclasses import asdict, is_dataclass

import torch


def save_checkpoint(path: str | Path, state, config) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": state.model.state_dict(),
            "optimizer": state.optimizer.state_dict(),
            "scheduler": state.scheduler.state_dict(),
            "ema": state.ema.state_dict() if state.ema is not None else None,
            "step": state.step,
            "config": asdict(config) if is_dataclass(config) else config,
        },
        path,
    )


def load_model_weights(path: str | Path, model: torch.nn.Module, map_location: str | torch.device = "cpu") -> dict:
    checkpoint = torch.load(path, map_location=map_location)
    weights = checkpoint.get("model", checkpoint)
    model.load_state_dict(weights)
    return checkpoint
