from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from babylm_elf.config import TrainConfig
from babylm_elf.data.datasets import build_dataloader
from babylm_elf.data.tokenizer import load_tokenizer
from babylm_elf.modeling.model import BabyLMELF
from babylm_elf.training.checkpointing import save_checkpoint
from babylm_elf.training.optim import (
    ExponentialMovingAverage,
    TrainState,
    create_optimizer,
    create_scheduler,
    resolve_device,
    seed_everything,
)
from babylm_elf.training.step import eval_step, train_step
from babylm_elf.utils.logging import format_metrics


def train_from_config(config: TrainConfig) -> None:
    seed_everything(config.seed)
    device = resolve_device(config.device)

    tokenizer = load_tokenizer(config.data.tokenizer_path)
    config.model.vocab_size = tokenizer.get_vocab_size()
    pad_token_id = tokenizer.token_to_id("<pad>")
    if pad_token_id is not None:
        config.model.pad_token_id = pad_token_id

    train_loader = build_dataloader(
        config.data.train_path,
        tokenizer,
        config.data.seq_length,
        config.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
    )
    valid_loader = None
    if config.data.valid_path:
        valid_loader = build_dataloader(
            config.data.valid_path,
            tokenizer,
            config.data.seq_length,
            config.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )

    if config.max_steps <= 0:
        config.max_steps = len(train_loader) * 10

    model = BabyLMELF(config.model).to(device)
    optimizer = create_optimizer(model, config.optim)
    scheduler = create_scheduler(optimizer, config.optim.warmup_steps, config.max_steps)
    ema = ExponentialMovingAverage(model.parameters(), config.optim.ema_decay)
    state = TrainState(model=model, optimizer=optimizer, scheduler=scheduler, ema=ema)

    checkpoint_dir = Path(config.output_dir) / config.name / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    progress = tqdm(total=config.max_steps, desc="train")
    train_iter = iter(train_loader)

    while state.step < config.max_steps:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accum_metrics: dict[str, float] = {}

        for _ in range(config.gradient_accumulation_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            batch = move_batch(batch, device)

            with autocast_context(device, config.mixed_precision):
                output = train_step(model, batch, config.diffusion)
                loss = output.loss / config.gradient_accumulation_steps
            loss.backward()
            for key, value in output.metrics.items():
                accum_metrics[key] = accum_metrics.get(key, 0.0) + value / config.gradient_accumulation_steps

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.max_grad_norm)
        optimizer.step()
        scheduler.step()
        ema.update(model.parameters())
        state.step += 1
        progress.update(1)

        if state.step % config.log_every == 0:
            progress.write(f"step {state.step} | {format_metrics(accum_metrics)}")

        if valid_loader is not None and state.step % config.validate_every == 0:
            metrics = validate(model, valid_loader, config, device)
            progress.write(f"validation {state.step} | {format_metrics(metrics)}")

        if state.step % config.save_every == 0:
            save_checkpoint(checkpoint_dir / f"step_{state.step}.pt", state, config)
            save_checkpoint(checkpoint_dir / "latest.pt", state, config)

    save_checkpoint(checkpoint_dir / "latest.pt", state, config)
    save_checkpoint(checkpoint_dir / "final.pt", state, config)


def validate(model, valid_loader, config: TrainConfig, device: torch.device) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in valid_loader:
        batch = move_batch(batch, device)
        with autocast_context(device, config.mixed_precision):
            metrics = eval_step(model, batch, config.diffusion)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    return {key: value / max(1, count) for key, value in totals.items()}


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def autocast_context(device: torch.device, enabled: bool):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=enabled and device.type == "cuda",
    )
