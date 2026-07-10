from __future__ import annotations

from contextlib import nullcontext
import math
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from babylm_elf.config import TrainConfig, sync_model_mlm_mask_latent_config
from babylm_elf.data.datasets import build_dataloader
from babylm_elf.data.manifest import validate_training_data_manifest
from babylm_elf.data.tokenizer import load_tokenizer
from babylm_elf.modeling.sdpa import sdpa_backend_status
from babylm_elf.modeling.model import BabyLMELF
from babylm_elf.training.checkpointing import CheckpointManager
from babylm_elf.training.optim import (
    ExponentialMovingAverage,
    TrainState,
    create_optimizer,
    create_scheduler,
    resolve_ema_decay,
    resolve_device,
    runtime_seed_for_rank,
    seed_everything,
)
from babylm_elf.training.step import eval_step, train_step
from babylm_elf.utils.distributed import (
    cleanup_distributed,
    get_local_rank,
    get_rank,
    get_world_size,
    init_distributed_from_env,
    is_main_process,
)
from babylm_elf.utils.logging import format_metrics


def train_from_config(config: TrainConfig) -> None:
    distributed = init_distributed_from_env()
    seed_everything(config.seed)
    device = (
        torch.device("cuda", get_local_rank())
        if distributed
        else resolve_device(config.device)
    )
    try:
        if is_main_process():
            _log_runtime_acceleration(device)
            if distributed:
                print(
                    "Distributed training: "
                    f"world_size={get_world_size()}; rank={get_rank()}; "
                    f"local_rank={get_local_rank()}",
                    flush=True,
                )

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
        sync_model_mlm_mask_latent_config(config)
        data_manifest = validate_training_data_manifest(config.data, tokenizer)
        actual_train_word_count = _manifest_train_word_count(config, data_manifest)

        train_generator = torch.Generator()
        train_loader = build_dataloader(
            config.data.train_path,
            tokenizer,
            config.data.seq_length,
            config.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            generator=train_generator,
            distributed=distributed,
            rank=get_rank(),
            world_size=get_world_size(),
            seed=config.seed,
            drop_incomplete=True,
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
        elif is_main_process():
            print(
                "No validation dataset configured. Select hyperparameters using "
                "BabyLM fast evaluation at a fixed word-exposure checkpoint."
            )

        microbatches_per_epoch = len(train_loader)
        optimizer_steps_per_epoch = max(
            1,
            math.ceil(microbatches_per_epoch / config.gradient_accumulation_steps),
        )
        _validate_manifest_loader_shape(
            config,
            data_manifest,
            train_loader,
            microbatches_per_epoch,
            optimizer_steps_per_epoch,
        )
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
        ema_decay = resolve_ema_decay(
            config.optim.ema_reference_decay,
            config.optim.ema_reference_steps,
            config.max_steps,
        )
        runtime_seed = runtime_seed_for_rank(config.seed, get_rank())
        if is_main_process():
            print(
                "EMA: "
                f"reference_decay={config.optim.ema_reference_decay:.7f}; "
                f"reference_steps={config.optim.ema_reference_steps:,}; "
                f"total_optimizer_steps={config.max_steps:,}; "
                f"resolved_decay={ema_decay:.9f}; "
                f"warmup={config.optim.ema_warmup}",
                flush=True,
            )
            print(
                "Random seeds: "
                f"model_init={config.seed}; "
                "runtime=base_seed+global_rank; "
                f"rank0_runtime={runtime_seed}",
                flush=True,
            )

        model = BabyLMELF(config.model).to(device)
        train_model = _wrap_for_distributed(model, distributed)
        # DDP starts from the same model on every rank. Training stochasticity
        # must diverge only after model construction and DDP synchronization.
        seed_everything(runtime_seed)
        optimizer = create_optimizer(model, config.optim)
        scheduler = create_scheduler(
            optimizer,
            warmup_steps,
            config.max_steps,
            schedule=config.optim.lr_schedule,
            min_lr=config.optim.min_lr,
        )
        ema = ExponentialMovingAverage(
            model,
            ema_decay,
            warmup=config.optim.ema_warmup,
        )
        state = TrainState(model=model, optimizer=optimizer, scheduler=scheduler, ema=ema)
        base_scratch_encoder_trainable = bool(config.model.scratch_encoder_trainable)
        scratch_encoder_param_ids = _scratch_encoder_parameter_ids(model)

        checkpoint_dir = Path(config.output_dir) / config.name / "checkpoints"
        steps_per_epoch = optimizer_steps_per_epoch
        checkpoints = None
        if is_main_process():
            checkpoints = CheckpointManager(
                checkpoint_dir,
                config,
                steps_per_epoch,
                microbatches_per_epoch=microbatches_per_epoch,
                actual_train_word_count=actual_train_word_count,
                run_word_limit=(
                    total_microbatches
                    * config.data.train_word_count
                    // microbatches_per_epoch
                    if config.data.train_word_count is not None
                    else None
                ),
            )
        progress = tqdm(
            total=config.max_steps,
            desc="train",
            disable=not is_main_process(),
        )
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
            train_model.train()
            scratch_encoder_trainable = _scratch_encoder_should_train(
                config,
                base_scratch_encoder_trainable,
                config.max_steps,
                state.step,
            )
            _set_scratch_encoder_trainability(model, scratch_encoder_trainable)
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
                sync_context = _gradient_sync_context(
                    train_model,
                    distributed,
                    group_microbatch,
                    group_size,
                )
                with sync_context:
                    with autocast_context(device, config.mixed_precision):
                        output = train_step(
                            train_model,
                            batch,
                            config.diffusion,
                            current_epoch=current_epoch,
                        )
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
            if not scratch_encoder_trainable:
                _zero_scratch_encoder_optimizer_lrs(optimizer, scratch_encoder_param_ids)
            optimizer.step()
            _assert_finite_parameters(model, optimizer_step=state.step + 1)
            scheduler.step()
            ema.update(model)
            state.step += 1

            if is_main_process():
                progress.update(1)
                accum_metrics["lr"] = scheduler.get_last_lr()[0]
                accum_metrics["effective_batch"] = (
                    group_size * config.batch_size * get_world_size()
                )

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

                if checkpoints is not None:
                    checkpoints.save_if_due(state, progress)

        if is_main_process() and checkpoints is not None:
            checkpoints.save_final(state)
        if configured_max_steps <= 0 and state.step != config.max_steps:
            raise RuntimeError(
                f"Training ended at step {state.step}, expected {config.max_steps}."
            )
    finally:
        cleanup_distributed()


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


def _wrap_for_distributed(
    model: torch.nn.Module,
    distributed: bool,
) -> torch.nn.Module:
    if not distributed:
        return model
    local_rank = get_local_rank()
    return DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )


def _gradient_sync_context(
    model: torch.nn.Module,
    distributed: bool,
    group_microbatch: int,
    group_size: int,
):
    if distributed and group_microbatch < group_size - 1:
        return model.no_sync()
    return nullcontext()


def _log_runtime_acceleration(device: torch.device) -> None:
    if device.type != "cuda":
        return
    status = sdpa_backend_status()
    print(
        "Runtime acceleration: "
        f"torch={torch.__version__}; "
        f"cuda={torch.version.cuda}; "
        f"device={device}; "
        f"{status['env']}={status['selected']}; "
        f"flash_available={status['flash_available']}; "
        f"flash_enabled={status['flash_enabled']}; "
        f"efficient_enabled={status['efficient_enabled']}; "
        f"math_enabled={status['math_enabled']}",
        flush=True,
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


def _manifest_train_word_count(config: TrainConfig, manifest: dict) -> int:
    actual_word_count = int(manifest["normalization"]["words"])
    nominal_word_count = config.data.train_word_count
    if is_main_process():
        nominal_text = (
            f"{nominal_word_count:,} official words; "
            if nominal_word_count is not None
            else ""
        )
        print(
            f"Training corpus: {nominal_text}"
            f"{actual_word_count:,} normalized usable words"
        )
    return actual_word_count


def _validate_manifest_loader_shape(
    config: TrainConfig,
    manifest: dict | None,
    train_loader,
    microbatches_per_epoch: int,
    optimizer_steps_per_epoch: int,
) -> None:
    if manifest is None:
        return
    packing = manifest.get("packing", {})
    sampler = train_loader.sampler
    samples_per_rank = len(sampler)
    expected = {
        "world_size": get_world_size(),
        "distributed_sampler_drop_last": bool(
            getattr(sampler, "drop_last", False)
        ),
        "samples_per_rank": samples_per_rank,
        "distributed_chunks_per_epoch": samples_per_rank * get_world_size(),
        "per_device_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "dataloader_batches_per_rank": microbatches_per_epoch,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
    }
    mismatches = {
        key: (packing.get(key), value)
        for key, value in expected.items()
        if packing.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: manifest={found!r}, runtime={runtime!r}"
            for key, (found, runtime) in mismatches.items()
        )
        raise ValueError(f"Data manifest/runtime dataloader mismatch: {details}.")


def _validate_embedding_source(config: TrainConfig) -> None:
    if config.model.embedding_source != "scratch_t5_encoder":
        return
    if config.model.scratch_encoder_trainable:
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


def _scratch_encoder_parameter_ids(model: BabyLMELF) -> set[int]:
    if model.scratch_encoder is None:
        return set()
    return {id(parameter) for parameter in model.scratch_encoder.parameters()}


def _scratch_encoder_should_train(
    config: TrainConfig,
    base_scratch_encoder_trainable: bool,
    max_steps: int,
    step: int,
) -> bool:
    if (
        config.model.embedding_source != "scratch_t5_encoder"
        or not base_scratch_encoder_trainable
    ):
        return base_scratch_encoder_trainable
    freeze_steps = math.ceil(max(0.0, config.encoder_freeze_steps_ratio) * max_steps)
    return step >= freeze_steps


def _set_scratch_encoder_trainability(model: BabyLMELF, trainable: bool) -> None:
    if model.scratch_encoder is None:
        return
    model.config.scratch_encoder_trainable = trainable
    model.scratch_encoder.train(trainable)
    if not trainable:
        model.scratch_encoder.eval()
    for parameter in model.scratch_encoder.parameters():
        parameter.requires_grad_(trainable)


def _zero_scratch_encoder_optimizer_lrs(
    optimizer: torch.optim.Optimizer,
    scratch_encoder_param_ids: set[int],
) -> None:
    if not scratch_encoder_param_ids:
        return
    for group in optimizer.param_groups:
        if any(id(parameter) in scratch_encoder_param_ids for parameter in group["params"]):
            group["lr"] = 0.0


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
    sampler = getattr(train_loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
    else:
        train_generator.manual_seed(seed + epoch)
    return iter(train_loader)
