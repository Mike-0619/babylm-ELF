from __future__ import annotations

import math
from pathlib import Path

import torch
from tqdm import tqdm

from babylm_elf.config import TrainConfig
from babylm_elf.data.datasets import build_dataloader
from babylm_elf.data.text import count_clean_words
from babylm_elf.data.tokenizer import load_tokenizer
from babylm_elf.modeling.model import BabyLMELF
from babylm_elf.training.checkpointing import CheckpointManager
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
    base_vocab_size = tokenizer.get_vocab_size()
    config.model.vocab_size = base_vocab_size
    config.model.base_vocab_size = base_vocab_size
    config.model.sentinel_start_id = base_vocab_size
    config.model.encoder_vocab_size = base_vocab_size + config.model.sentinel_count
    pad_token_id = tokenizer.token_to_id("<pad>")
    if pad_token_id is not None:
        config.model.pad_token_id = pad_token_id
    _validate_embedding_source(config)

    train_generator = torch.Generator()
    train_loader = build_dataloader(
        config.data.train_path,
        tokenizer,
        config.data.seq_length,
        config.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        generator=train_generator,
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
    else:
        print(
            "No validation dataset configured. Select hyperparameters using "
            "BabyLM fast evaluation at a fixed word-exposure checkpoint."
        )

    microbatches_per_epoch = len(train_loader)
    optimizer_steps_per_epoch = max(
        1,
        math.ceil(microbatches_per_epoch / config.gradient_accumulation_steps),
    )
    actual_train_word_count = _count_actual_train_words(config)
    configured_max_steps = config.max_steps
    if config.max_steps <= 0:
        total_microbatches = microbatches_per_epoch * config.epochs
        config.max_steps = math.ceil(
            total_microbatches / config.gradient_accumulation_steps
        )
    else:
        total_microbatches = config.max_steps * config.gradient_accumulation_steps
    warmup_steps = config.optim.warmup_steps
    if warmup_steps < 0:
        warmup_steps = int(
            optimizer_steps_per_epoch * (config.optim.warmup_epochs or 0.0)
        )

    model = BabyLMELF(config.model).to(device)
    optimizer = create_optimizer(model, config.optim)
    scheduler = create_scheduler(
        optimizer,
        warmup_steps,
        config.max_steps,
        schedule=config.optim.lr_schedule,
        min_lr=config.optim.min_lr,
    )
    ema = ExponentialMovingAverage(model, config.optim.ema_decay)
    state = TrainState(model=model, optimizer=optimizer, scheduler=scheduler, ema=ema)

    checkpoint_dir = Path(config.output_dir) / config.name / "checkpoints"
    steps_per_epoch = optimizer_steps_per_epoch
    checkpoints = CheckpointManager(
        checkpoint_dir,
        config,
        steps_per_epoch,
        microbatches_per_epoch=microbatches_per_epoch,
        actual_train_word_count=actual_train_word_count,
        run_word_limit=(
            total_microbatches * config.data.train_word_count // microbatches_per_epoch
            if config.data.train_word_count is not None
            else None
        ),
    )
    progress = tqdm(total=config.max_steps, desc="train")
    current_epoch = 0
    train_iter = _epoch_iterator(
        train_loader,
        train_generator,
        config.seed,
        current_epoch,
    )

    while state.microbatches_seen < total_microbatches:
        group_size = _next_group_size(
            total_microbatches,
            state.microbatches_seen,
            config.gradient_accumulation_steps,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accum_metrics: dict[str, float] = {}

        for group_microbatch in range(group_size):
            try:
                batch = next(train_iter)
            except StopIteration:
                current_epoch += 1
                train_iter = _epoch_iterator(
                    train_loader,
                    train_generator,
                    config.seed,
                    current_epoch,
                )
                batch = next(train_iter)

            batch = move_batch(batch, device)
            with autocast_context(device, config.mixed_precision):
                output = train_step(model, batch, config.diffusion)
                loss = output.loss / config.gradient_accumulation_steps
            _assert_finite_step_output(
                output.loss,
                output.metrics,
                optimizer_step=state.step + 1,
                microbatch=state.microbatches_seen + 1,
                group_microbatch=group_microbatch + 1,
            )
            if not torch.isfinite(loss).all().item():
                raise FloatingPointError(
                    "Non-finite scaled loss before backward at "
                    f"optimizer step {state.step + 1}, "
                    f"microbatch {state.microbatches_seen + 1}."
                )
            loss.backward()
            state.microbatches_seen += 1
            for key, value in output.metrics.items():
                accum_metrics[key] = accum_metrics.get(key, 0.0) + value / group_size

        _clip_gradients(
            model,
            config.optim.max_grad_norm,
            optimizer_step=state.step + 1,
        )
        optimizer.step()
        _assert_finite_parameters(model, optimizer_step=state.step + 1)
        scheduler.step()
        ema.update(model)
        state.step += 1
        progress.update(1)
        accum_metrics["lr"] = scheduler.get_last_lr()[0]
        accum_metrics["effective_batch"] = group_size * config.batch_size

        if state.step % config.log_every == 0:
            progress.write(
                f"step {state.step} | {format_metrics(accum_metrics)}"
            )

        if (
            valid_loader is not None
            and state.step % config.validate_every == 0
        ):
            with ema.average_parameters(model):
                metrics = validate(model, valid_loader, config, device)
            progress.write(
                f"validation {state.step} | {format_metrics(metrics)}"
            )

        checkpoints.save_if_due(state, progress)

    checkpoints.save_final(state)
    if configured_max_steps <= 0 and state.step != config.max_steps:
        raise RuntimeError(
            f"Training ended at step {state.step}, expected {config.max_steps}."
        )


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


def _assert_finite_step_output(
    loss: torch.Tensor,
    metrics: dict[str, float],
    *,
    optimizer_step: int,
    microbatch: int,
    group_microbatch: int,
) -> None:
    nonfinite_metrics = {
        key: value
        for key, value in metrics.items()
        if not math.isfinite(value)
    }
    if torch.isfinite(loss).all().item() and not nonfinite_metrics:
        return

    metric_text = ", ".join(
        f"{key}={value}" for key, value in sorted(metrics.items())
    )
    raise FloatingPointError(
        "Non-finite training output before backward at "
        f"optimizer step {optimizer_step}, microbatch {microbatch} "
        f"(accumulation microbatch {group_microbatch}): {metric_text}"
    )


@torch.no_grad()
def _assert_finite_parameters(model: torch.nn.Module, *, optimizer_step: int) -> None:
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not named_parameters:
        return

    finite_checks = torch.stack(
        [torch.isfinite(parameter).all() for _, parameter in named_parameters]
    )
    if finite_checks.all().item():
        return

    bad_names = [
        name
        for (name, _), is_finite in zip(named_parameters, finite_checks.tolist())
        if not is_finite
    ]
    raise FloatingPointError(
        "Non-finite model parameters after optimizer step "
        f"{optimizer_step}: {', '.join(bad_names)}"
    )


def _clip_gradients(
    model: torch.nn.Module,
    max_grad_norm: float,
    *,
    optimizer_step: int,
) -> None:
    try:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
            error_if_nonfinite=True,
        )
    except RuntimeError as exc:
        raise FloatingPointError(
            f"Non-finite gradient norm before optimizer step {optimizer_step}."
        ) from exc


def _count_actual_train_words(config: TrainConfig) -> int | None:
    if not config.data.train_text:
        return None
    train_text = Path(config.data.train_text)
    if not train_text.exists():
        print(
            f"Actual word count unavailable: training text does not exist at {train_text}"
        )
        return None

    actual_word_count = count_clean_words(train_text)
    nominal_word_count = config.data.train_word_count
    if nominal_word_count is None:
        print(f"Training corpus: {actual_word_count:,} cleaned whitespace words")
    else:
        print(
            "Training corpus: "
            f"{nominal_word_count:,} nominal words; "
            f"{actual_word_count:,} cleaned whitespace words"
        )
    return actual_word_count


def _validate_embedding_source(config: TrainConfig) -> None:
    if config.model.embedding_source != "scratch_t5_encoder":
        return
    if not config.model.encoder_checkpoint_path:
        raise ValueError("scratch_t5_encoder requires model.encoder_checkpoint_path.")
    encoder_checkpoint = Path(config.model.encoder_checkpoint_path)
    if not encoder_checkpoint.exists():
        raise FileNotFoundError(f"Scratch encoder checkpoint not found: {encoder_checkpoint}")
    if not config.model.latent_stats_path:
        raise ValueError("scratch_t5_encoder requires model.latent_stats_path.")
    latent_stats = Path(config.model.latent_stats_path)
    if not latent_stats.exists():
        raise FileNotFoundError(f"Scratch encoder latent stats not found: {latent_stats}")


def _next_group_size(
    total_microbatches: int,
    microbatches_seen: int,
    accumulation_steps: int,
) -> int:
    return min(accumulation_steps, total_microbatches - microbatches_seen)


def _epoch_iterator(
    train_loader,
    train_generator: torch.Generator,
    seed: int,
    epoch: int,
):
    train_generator.manual_seed(seed + epoch)
    return iter(train_loader)
