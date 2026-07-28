from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from babylm_elf.config import (
    RunConfig,
    resolve_model_config,
    resolve_objective,
    resolve_run,
)
from babylm_elf.data.dataset import build_dataloader, validate_training_data_manifest
from babylm_elf.data.prepare import empty_control_token_ids, load_tokenizer
from babylm_elf.modules.encoder import (
    scratch_encoder_parameter_ids,
    scratch_encoder_should_train,
    set_model_scratch_encoder_trainability,
    validate_embedding_source,
    zero_scratch_encoder_optimizer_lrs,
)
from babylm_elf.modules.layers import sdpa_backend_status
from babylm_elf.modules.model import BabyLMELF
from babylm_elf.training.checkpoint import (
    CheckpointManager,
    load_training_checkpoint,
    restore_rank_rng_state,
)
from babylm_elf.training.objectives import train_step
from babylm_elf.training.optim import (
    ExponentialMovingAverage,
    TrainState,
    create_optimizer,
    create_scheduler,
    resolve_device,
    runtime_seed_for_rank,
    seed_everything,
)


def distributed_requested() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def init_distributed_from_env() -> bool:
    if not distributed_requested():
        return False
    if not dist.is_available():
        raise RuntimeError(
            "Distributed training requested, but torch.distributed is unavailable."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA because NCCL was requested.")
    torch.cuda.set_device(get_local_rank())
    if not is_distributed():
        dist.init_process_group(backend="nccl")
    return True


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def format_metrics(metrics: dict[str, float]) -> str:
    return " | ".join(f"{key}: {value:.4f}" for key, value in metrics.items())


def reduce_metrics(metrics: dict[str, float], device: torch.device) -> dict[str, float]:
    if not is_distributed() or not metrics:
        return metrics
    keys = sorted(metrics)
    values = torch.tensor(
        [metrics[key] for key in keys],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values.div_(get_world_size())
    return dict(zip(keys, values.cpu().tolist()))


def move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def autocast_context(device: torch.device, enabled: bool):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=enabled and device.type == "cuda",
    )


def wrap_for_distributed(
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


def gradient_sync_context(
    model: torch.nn.Module,
    distributed: bool,
    group_microbatch: int,
    group_size: int,
):
    if distributed and group_microbatch < group_size - 1:
        return model.no_sync()
    return nullcontext()


def log_runtime_acceleration(device: torch.device) -> None:
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


def assert_finite_step_output(
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
def assert_finite_parameters(
    model: torch.nn.Module,
    *,
    optimizer_step: int,
) -> None:
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not named_parameters:
        return
    finite_checks = torch.stack(
        [
            torch.isfinite(parameter).all()
            for _, parameter in named_parameters
        ]
    )
    if finite_checks.all().item():
        return
    bad_names = [
        name
        for (name, _), is_finite in zip(
            named_parameters,
            finite_checks.tolist(),
        )
        if not is_finite
    ]
    raise FloatingPointError(
        "Non-finite model parameters after optimizer step "
        f"{optimizer_step}: {', '.join(bad_names)}"
    )


def clip_gradients(
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


def resolve_resume_path(
    config: RunConfig,
    requested: str | Path | None,
) -> Path | None:
    checkpoint_dir = Path(config.output_dir) / config.name / "checkpoints"
    latest = checkpoint_dir / "latest.pt"
    existing = tuple(checkpoint_dir.rglob("*.pt")) if checkpoint_dir.exists() else ()
    if requested == "auto":
        if latest.is_file():
            return latest
        if existing:
            raise FileExistsError(
                f"{checkpoint_dir} contains checkpoints but no latest.pt; "
                "refusing to restart and overwrite the run."
            )
        return None
    if requested is None:
        if existing:
            raise FileExistsError(
                f"{checkpoint_dir} already contains checkpoints. Use "
                "--resume auto or --resume PATH."
            )
        return None
    path = Path(requested)
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    return path


def train_from_config(
    config: RunConfig,
    *,
    resume_path: str | Path | None = None,
) -> None:
    resume_path = resolve_resume_path(config, resume_path)
    distributed = init_distributed_from_env()
    seed_everything(config.seed)
    device = (
        torch.device("cuda", get_local_rank())
        if distributed
        else resolve_device(config.training.device)
    )
    progress = None
    try:
        if is_main_process():
            log_runtime_acceleration(device)
            if distributed:
                print(
                    "Distributed training: "
                    f"world_size={get_world_size()}; rank={get_rank()}; "
                    f"local_rank={get_local_rank()}",
                    flush=True,
                )

        tokenizer = load_tokenizer(config.data.tokenizer_path)
        base_vocab_size = tokenizer.get_vocab_size()
        pad_token_id = tokenizer.token_to_id("<pad>")
        if pad_token_id is None:
            pad_token_id = config.model.pad_token_id
        validate_embedding_source(config)
        model_config = resolve_model_config(
            config,
            vocab_size=base_vocab_size,
            pad_token_id=pad_token_id,
        )
        excluded_token_ids = (
            empty_control_token_ids(tokenizer)
            if config.objective.targets.filter_empty_control
            else set()
        )
        objective = resolve_objective(
            config.objective,
            excluded_token_ids=excluded_token_ids,
        )
        data_manifest = validate_training_data_manifest(config.data, tokenizer)
        actual_train_word_count = _manifest_train_word_count(config, data_manifest)

        train_generator = torch.Generator()
        train_loader = build_dataloader(
            config.data.train_path,
            tokenizer,
            config.data.seq_length,
            config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            generator=train_generator,
            distributed=distributed,
            rank=get_rank(),
            world_size=get_world_size(),
            seed=config.seed,
            drop_incomplete=True,
        )
        microbatches_per_epoch = len(train_loader)
        optimizer_steps_per_epoch = max(
            1,
            math.ceil(
                microbatches_per_epoch
                / config.training.gradient_accumulation_steps
            ),
        )
        _validate_manifest_loader_shape(
            config,
            data_manifest,
            train_loader,
            microbatches_per_epoch,
            optimizer_steps_per_epoch,
        )
        if config.training.max_steps <= 0:
            total_microbatches = microbatches_per_epoch * config.training.epochs
        else:
            total_microbatches = (
                config.training.max_steps
                * config.training.gradient_accumulation_steps
            )
        resolved = resolve_run(
            config,
            model=model_config,
            objective=objective,
            microbatches_per_epoch=microbatches_per_epoch,
            optimizer_steps_per_epoch=optimizer_steps_per_epoch,
            actual_train_word_count=actual_train_word_count,
        )
        runtime_seed = runtime_seed_for_rank(config.seed, get_rank())
        if is_main_process():
            print(
                "EMA: "
                f"reference_decay={config.ema.reference_decay:.7f}; "
                f"reference_steps={config.ema.reference_steps:,}; "
                f"total_optimizer_steps={resolved.max_steps:,}; "
                f"resolved_decay={resolved.ema_decay:.9f}; "
                f"warmup={config.ema.warmup}",
                flush=True,
            )
            print(
                "Random seeds: "
                f"model_init={config.seed}; "
                "runtime=base_seed+global_rank; "
                f"rank0_runtime={runtime_seed}",
                flush=True,
            )

        model = BabyLMELF(resolved.model).to(device)
        train_model = wrap_for_distributed(model, distributed)
        # DDP starts from the same model on every rank. Training stochasticity
        # must diverge only after model construction and DDP synchronization.
        seed_everything(runtime_seed)
        optimizer = create_optimizer(model, config.optimizer)
        scheduler = create_scheduler(
            optimizer,
            resolved.warmup_steps,
            resolved.max_steps,
            schedule=config.scheduler.type,
            min_lr=config.scheduler.min_lr,
        )
        ema = ExponentialMovingAverage(
            model,
            resolved.ema_decay,
            warmup=config.ema.warmup,
        )
        state = TrainState(model=model, optimizer=optimizer, scheduler=scheduler, ema=ema)
        resume_checkpoint = None
        if resume_path is not None:
            resume_checkpoint = load_training_checkpoint(
                resume_path,
                state,
                map_location=device,
            )
            _validate_resume_config(resolved, resume_checkpoint["resolved_config"])
            _validate_resume_progress(
                resume_checkpoint,
                state,
                microbatches_per_epoch=microbatches_per_epoch,
                total_microbatches=total_microbatches,
            )
            if is_main_process():
                print(
                    f"Resumed {resume_path}: step={state.step:,}; "
                    f"microbatches={state.microbatches_seen:,}; "
                    f"epoch={resume_checkpoint['progress']['epoch']}; "
                    "per-rank RNG will be restored after dataloader reconstruction.",
                    flush=True,
                )
        base_scratch_encoder_trainable = bool(
            resolved.model.scratch_encoder_trainable
        )
        scratch_encoder_param_ids = scratch_encoder_parameter_ids(model)

        checkpoint_dir = Path(config.output_dir) / config.name / "checkpoints"
        steps_per_epoch = optimizer_steps_per_epoch
        checkpoints = CheckpointManager(
            checkpoint_dir,
            resolved,
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
            writer=is_main_process(),
        )
        if resume_checkpoint is not None:
            checkpoints.restore_progress(state)
        progress = tqdm(
            total=resolved.max_steps,
            initial=state.step,
            desc="train",
            disable=not is_main_process(),
        )
        current_epoch, microbatch_in_epoch = divmod(
            state.microbatches_seen,
            microbatches_per_epoch,
        )
        train_iter = _epoch_iterator(
            train_loader,
            train_generator,
            config.seed,
            current_epoch,
        )
        for _ in range(microbatch_in_epoch):
            try:
                next(train_iter)
            except StopIteration as exc:
                raise RuntimeError(
                    "Resume progress exceeds the reconstructed epoch dataloader."
                ) from exc
        if resume_checkpoint is not None:
            restore_rank_rng_state(
                resume_checkpoint,
                rank=get_rank(),
                world_size=get_world_size(),
            )

        while state.microbatches_seen < total_microbatches:
            group_size = _next_group_size(
                total_microbatches,
                state.microbatches_seen,
                config.training.gradient_accumulation_steps,
            )
            train_model.train()
            scratch_encoder_trainable = scratch_encoder_should_train(
                config,
                base_scratch_encoder_trainable,
                resolved.max_steps,
                state.step,
            )
            set_model_scratch_encoder_trainability(model, scratch_encoder_trainable)
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
                sync_context = gradient_sync_context(
                    train_model,
                    distributed,
                    group_microbatch,
                    group_size,
                )
                with sync_context:
                    with autocast_context(
                        device,
                        config.training.precision == "bf16",
                    ):
                        output = train_step(
                            train_model,
                            batch,
                            resolved.objective,
                            current_epoch=current_epoch,
                        )
                        loss = (
                            output.loss
                            / config.training.gradient_accumulation_steps
                        )
                    assert_finite_step_output(
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

            clip_gradients(
                model,
                config.optimizer.max_grad_norm,
                optimizer_step=state.step + 1,
            )
            if not scratch_encoder_trainable:
                zero_scratch_encoder_optimizer_lrs(
                    optimizer,
                    scratch_encoder_param_ids,
                )
            optimizer.step()
            assert_finite_parameters(model, optimizer_step=state.step + 1)
            scheduler.step()
            ema.update(model)
            state.step += 1

            accum_metrics["lr"] = scheduler.get_last_lr()[0]
            accum_metrics["effective_batch"] = (
                group_size
                * config.training.batch_size
                * get_world_size()
            )
            accum_metrics = reduce_metrics(accum_metrics, device)
            if is_main_process():
                progress.update(1)
                if state.step % config.training.log_every == 0:
                    progress.write(
                        f"step {state.step} | {format_metrics(accum_metrics)}"
                    )

            checkpoints.save_if_due(
                state,
                progress if is_main_process() else None,
            )

        checkpoints.save_final(state)
        if config.training.max_steps <= 0 and state.step != resolved.max_steps:
            raise RuntimeError(
                f"Training ended at step {state.step}, expected {resolved.max_steps}."
            )
    finally:
        if progress is not None:
            progress.close()
        cleanup_distributed()




def _manifest_train_word_count(config: RunConfig, manifest: dict) -> int:
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
    config: RunConfig,
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
        "per_device_batch_size": config.training.batch_size,
        "gradient_accumulation_steps": (
            config.training.gradient_accumulation_steps
        ),
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
    dataset = getattr(train_loader, "dataset", None)
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
    sampler = getattr(train_loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
    else:
        train_generator.manual_seed(seed + epoch)
    return iter(train_loader)


def _validate_resume_config(
    config,
    saved_config: dict,
) -> None:
    current = asdict(config)
    if saved_config != current:
        raise ValueError("Resume config differs from the resolved checkpoint config.")


def _validate_resume_progress(
    checkpoint: dict,
    state: TrainState,
    *,
    microbatches_per_epoch: int,
    total_microbatches: int,
) -> None:
    progress = checkpoint["progress"]
    max_steps = int(checkpoint["resolved_config"]["max_steps"])
    if state.step < 0 or state.step > max_steps:
        raise ValueError(f"Invalid resumed optimizer step: {state.step}.")
    if (
        state.microbatches_seen < 0
        or state.microbatches_seen > total_microbatches
    ):
        raise ValueError(
            "Invalid resumed microbatch count: "
            f"{state.microbatches_seen} not in [0, {total_microbatches}]."
        )
    epoch, within_epoch = divmod(
        state.microbatches_seen,
        microbatches_per_epoch,
    )
    if (
        int(progress["epoch"]) != epoch
        or int(progress["microbatch_in_epoch"]) != within_epoch
    ):
        raise ValueError(
            "Resume checkpoint epoch/microbatch metadata is inconsistent "
            "with microbatches_seen."
        )
