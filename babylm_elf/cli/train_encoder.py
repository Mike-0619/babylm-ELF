from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import T5Config, T5ForConditionalGeneration

from babylm_elf.config import EncoderTrainConfig, load_encoder_config
from babylm_elf.data.datasets import build_dataloader
from babylm_elf.data.manifest import (
    load_data_manifest,
    validate_training_data_manifest,
)
from babylm_elf.data.tokenizer import load_tokenizer
from babylm_elf.encoder.span_corruption import make_t5_span_corruption_batch
from babylm_elf.training.checkpointing import _atomic_torch_save
from babylm_elf.training.optim import create_scheduler, resolve_device, seed_everything
from babylm_elf.training.trainer import autocast_context, move_batch
from babylm_elf.utils.logging import format_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a scratch T5 encoder for BabyLM-ELF.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_encoder_config(args.config)
    train_encoder_from_config(config)


def train_encoder_from_config(config: EncoderTrainConfig) -> None:
    if config.objective.objective != "t5_span_corruption":
        raise ValueError("Only objective=t5_span_corruption is currently supported.")

    seed_everything(config.seed)
    device = resolve_device(config.device)
    tokenizer = load_tokenizer(config.data.tokenizer_path)
    _sync_vocab_settings(config, tokenizer)
    data_manifest = validate_training_data_manifest(config.data, tokenizer)
    _validate_encoder_train_word_count(config, data_manifest)

    train_generator = torch.Generator()
    train_generator.manual_seed(config.seed)
    train_loader = build_dataloader(
        config.data.train_path,
        tokenizer,
        config.data.seq_length,
        config.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        generator=train_generator,
    )
    model = T5ForConditionalGeneration(_make_t5_config(config)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optim.learning_rate,
        betas=(config.optim.beta1, config.optim.beta2),
        eps=config.optim.eps,
        weight_decay=config.optim.weight_decay,
    )

    microbatches_per_epoch = len(train_loader)
    total_microbatches = microbatches_per_epoch * config.epochs
    max_steps = math.ceil(total_microbatches / config.gradient_accumulation_steps)
    warmup_steps = config.optim.warmup_steps
    if warmup_steps < 0:
        warmup_steps = int(
            max(1, math.ceil(microbatches_per_epoch / config.gradient_accumulation_steps))
            * (config.optim.warmup_epochs or 0.0)
        )
    scheduler = create_scheduler(
        optimizer,
        warmup_steps,
        max_steps,
        schedule=config.optim.lr_schedule,
        min_lr=config.optim.min_lr,
    )

    output_dir = Path(config.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    progress = tqdm(total=max_steps, desc="train_encoder")
    state_step = 0
    microbatches_seen = 0
    current_epoch = 0
    train_iter = _epoch_iterator(train_loader, train_generator, config.seed, current_epoch)
    span_generator = torch.Generator()
    span_generator.manual_seed(config.seed + 17)

    while microbatches_seen < total_microbatches:
        group_size = min(
            config.gradient_accumulation_steps,
            total_microbatches - microbatches_seen,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accum_metrics: dict[str, float] = {}
        for _ in range(group_size):
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
            corrupted = make_t5_span_corruption_batch(
                batch["input_ids"],
                batch["attention_mask"],
                pad_token_id=config.model.pad_token_id,
                eos_token_id=config.model.eos_token_id,
                sentinel_start_id=config.model.sentinel_start_id,
                sentinel_count=config.model.sentinel_count,
                noise_density=config.objective.noise_density,
                mean_noise_span_length=config.objective.mean_noise_span_length,
                special_token_count=config.objective.special_token_count,
                generator=span_generator,
            )
            with autocast_context(device, config.mixed_precision):
                output = model(
                    input_ids=corrupted.input_ids,
                    attention_mask=corrupted.attention_mask,
                    labels=corrupted.labels,
                )
                loss = output.loss / config.gradient_accumulation_steps
            if not torch.isfinite(loss).all().item():
                raise FloatingPointError(
                    f"Non-finite encoder loss at optimizer step {state_step + 1}."
                )
            loss.backward()
            microbatches_seen += 1
            accum_metrics["loss"] = accum_metrics.get("loss", 0.0) + float(output.loss.detach()) / group_size

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.max_grad_norm)
        optimizer.step()
        scheduler.step()
        state_step += 1
        progress.update(1)
        accum_metrics["lr"] = scheduler.get_last_lr()[0]
        if state_step % config.log_every == 0:
            progress.write(f"step {state_step} | {format_metrics(accum_metrics)}")

    progress.close()
    _save_encoder_checkpoint(checkpoint_dir / "final.pt", model, config, state_step)
    _compute_and_save_latent_stats(output_dir / "latent_stats.pt", model, config, tokenizer, device)


def _sync_vocab_settings(config: EncoderTrainConfig, tokenizer) -> None:
    base_vocab_size = tokenizer.get_vocab_size()
    config.model.base_vocab_size = base_vocab_size
    config.model.sentinel_start_id = base_vocab_size
    config.model.vocab_size = base_vocab_size + config.model.sentinel_count
    pad_token_id = tokenizer.token_to_id("<pad>")
    eos_token_id = tokenizer.token_to_id("</s>")
    if pad_token_id is not None:
        config.model.pad_token_id = pad_token_id
        config.model.decoder_start_token_id = pad_token_id
    if eos_token_id is not None:
        config.model.eos_token_id = eos_token_id


def _validate_encoder_train_word_count(
    config: EncoderTrainConfig,
    manifest: dict,
) -> None:
    expected = config.data.train_word_count
    actual = int(manifest["normalization"]["words"])
    print(
        "Encoder training corpus: "
        f"{expected:,} official words; {actual:,} normalized usable words"
    )


def _make_t5_config(config: EncoderTrainConfig) -> T5Config:
    return T5Config(
        vocab_size=config.model.vocab_size,
        d_model=config.model.d_model,
        d_ff=config.model.d_ff,
        d_kv=config.model.d_kv,
        num_layers=config.model.num_layers,
        num_decoder_layers=config.model.num_decoder_layers,
        num_heads=config.model.num_heads,
        dropout_rate=config.model.dropout_rate,
        layer_norm_epsilon=config.model.layer_norm_epsilon,
        feed_forward_proj="relu",
        pad_token_id=config.model.pad_token_id,
        eos_token_id=config.model.eos_token_id,
        decoder_start_token_id=config.model.decoder_start_token_id,
    )


def _epoch_iterator(loader, generator: torch.Generator, seed: int, epoch: int):
    generator.manual_seed(seed + epoch)
    return iter(loader)


def _save_encoder_checkpoint(
    path: Path,
    model: T5ForConditionalGeneration,
    config: EncoderTrainConfig,
    step: int,
) -> None:
    checkpoint = {
        "encoder": model.encoder.state_dict(),
        "config": {
            "model": vars(config.model),
            "objective": vars(config.objective),
            "data": vars(config.data),
            "seed": config.seed,
            "epochs": config.epochs,
            "step": step,
        },
        "metadata": {
            "checkpoint_type": "scratch_t5_encoder",
            "step": step,
            "encoder_epochs": config.epochs,
            "train_word_count": config.data.train_word_count,
            "words_seen": (
                config.epochs * config.data.train_word_count
                if config.data.train_word_count is not None
                else None
            ),
        },
    }
    _atomic_torch_save(checkpoint, path)


@torch.no_grad()
def _compute_and_save_latent_stats(
    path: Path,
    model: T5ForConditionalGeneration,
    config: EncoderTrainConfig,
    tokenizer,
    device: torch.device,
) -> None:
    loader = build_dataloader(
        config.data.train_path,
        tokenizer,
        config.data.seq_length,
        config.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    model.eval()
    total = 0
    sum_x = torch.zeros(config.model.d_model, device=device, dtype=torch.float64)
    sum_x2 = torch.zeros(config.model.d_model, device=device, dtype=torch.float64)
    for batch in tqdm(loader, desc="latent_stats"):
        batch = move_batch(batch, device)
        with autocast_context(device, config.mixed_precision):
            hidden = model.encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).last_hidden_state.float()
        mask = batch["attention_mask"].bool()
        values = hidden[mask].to(torch.float64)
        if values.numel() == 0:
            continue
        total += values.size(0)
        sum_x += values.sum(dim=0)
        sum_x2 += values.square().sum(dim=0)
    if total == 0:
        raise ValueError("Cannot compute latent stats from an empty corpus.")
    mean = sum_x / total
    variance = (sum_x2 / total - mean.square()).clamp_min(1.0e-12)
    stats = {
        "mean": mean.float().cpu(),
        "std": variance.sqrt().float().cpu(),
        "count": total,
        "metadata": {
            "train_word_count": config.data.train_word_count,
            "actual_train_word_count": (
                int(
                    load_data_manifest(config.data.manifest_path)["normalization"][
                        "words"
                    ]
                )
                if config.data.manifest_path
                else None
            ),
        },
    }
    _atomic_torch_save(stats, path)


if __name__ == "__main__":
    main()
