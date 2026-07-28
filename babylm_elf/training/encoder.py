from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    PreTrainedTokenizerFast,
    T5Config,
    T5ForConditionalGeneration,
)

from babylm_elf.config import (
    EncoderTrainConfig,
    load_config,
    resolve_model_config,
)
from babylm_elf.data.dataset import (
    build_dataloader,
    load_data_manifest,
    validate_training_data_manifest,
)
from babylm_elf.data.prepare import load_tokenizer
from babylm_elf.modules.encoder import scratch_encoder_attention_mask
from babylm_elf.modules.model import BabyLMELF
from babylm_elf.training.checkpoint import atomic_torch_save, select_model_weights
from babylm_elf.training.optim import create_scheduler, resolve_device, seed_everything
from babylm_elf.training.train import autocast_context, format_metrics, move_batch


@dataclass
class SpanCorruptionBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def make_t5_span_corruption_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    pad_token_id: int,
    eos_token_id: int,
    sentinel_start_id: int,
    sentinel_count: int,
    noise_density: float,
    mean_noise_span_length: float,
    special_token_count: int,
    generator: torch.Generator | None = None,
    segment_boundary_token_id: int | None = None,
) -> SpanCorruptionBatch:
    """Create T5-style span-corruption inputs and labels.

    The base tokenizer never stores exported ``<extra_id_*>`` tokens. Sentinel
    ids are internal ids above the base vocab and are used only by the scratch
    encoder pretraining objective.
    """

    corrupted_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    for row_ids, row_mask in zip(input_ids.cpu(), attention_mask.cpu()):
        valid = row_mask.bool()
        clean = row_ids[valid].long()
        for segment in _split_packed_segments(clean, segment_boundary_token_id):
            noise_mask = _sample_noise_mask(
                segment,
                noise_density=noise_density,
                mean_noise_span_length=mean_noise_span_length,
                special_token_count=special_token_count,
                generator=generator,
            )
            corrupted, labels = _corrupt_one_sequence(
                segment,
                noise_mask,
                eos_token_id=eos_token_id,
                sentinel_start_id=sentinel_start_id,
                sentinel_count=sentinel_count,
            )
            corrupted_rows.append(corrupted)
            label_rows.append(labels)

    if not corrupted_rows:
        raise ValueError("Cannot span-corrupt a batch with no active tokens.")

    max_input_len = max(row.numel() for row in corrupted_rows)
    max_label_len = max(row.numel() for row in label_rows)
    batch_input = input_ids.new_full(
        (len(corrupted_rows), max_input_len),
        pad_token_id,
    )
    batch_attention = attention_mask.new_zeros(
        (len(corrupted_rows), max_input_len)
    )
    batch_labels = input_ids.new_full((len(corrupted_rows), max_label_len), -100)

    for index, row in enumerate(corrupted_rows):
        batch_input[index, : row.numel()] = row.to(batch_input.device)
        batch_attention[index, : row.numel()] = 1
    for index, row in enumerate(label_rows):
        batch_labels[index, : row.numel()] = row.to(batch_labels.device)

    return SpanCorruptionBatch(
        input_ids=batch_input.to(input_ids.device),
        attention_mask=batch_attention.to(attention_mask.device),
        labels=batch_labels.to(input_ids.device),
    )


def _split_packed_segments(
    input_ids: torch.Tensor,
    boundary_token_id: int | None,
) -> list[torch.Tensor]:
    if boundary_token_id is None or input_ids.numel() == 0:
        return [input_ids]
    starts = input_ids.eq(boundary_token_id).nonzero(as_tuple=False).flatten().tolist()
    if not starts:
        return [input_ids]
    boundaries = ([0] if starts[0] != 0 else []) + starts + [input_ids.numel()]
    return [
        input_ids[start:end]
        for start, end in zip(boundaries, boundaries[1:])
        if end > start
    ]


def _sample_noise_mask(
    input_ids: torch.Tensor,
    *,
    noise_density: float,
    mean_noise_span_length: float,
    special_token_count: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    maskable = input_ids.ge(special_token_count)
    maskable_indices = maskable.nonzero(as_tuple=False).flatten()
    noise_mask = torch.zeros(input_ids.shape, dtype=torch.bool)
    if maskable_indices.numel() == 0:
        return noise_mask

    target_tokens = max(1, int(round(maskable_indices.numel() * noise_density)))
    span_length = max(1, int(round(mean_noise_span_length)))
    max_spans = max(1, min(100, int(round(target_tokens / span_length)) + 1))
    perm = torch.randperm(maskable_indices.numel(), generator=generator)
    spans = 0
    for start in maskable_indices[perm].tolist():
        if noise_mask.sum().item() >= target_tokens or spans >= max_spans:
            break
        if noise_mask[start]:
            continue
        end = start
        while (
            end < input_ids.numel()
            and bool(maskable[end])
            and not bool(noise_mask[end])
            and end - start < span_length
            and noise_mask.sum().item() < target_tokens
        ):
            noise_mask[end] = True
            end += 1
        spans += 1
    return noise_mask


def _corrupt_one_sequence(
    input_ids: torch.Tensor,
    noise_mask: torch.Tensor,
    *,
    eos_token_id: int,
    sentinel_start_id: int,
    sentinel_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    corrupted: list[int] = []
    labels: list[int] = []
    sentinel_offset = 0
    index = 0
    while index < input_ids.numel():
        if not bool(noise_mask[index]):
            corrupted.append(int(input_ids[index]))
            index += 1
            continue

        sentinel_id = min(
            sentinel_start_id + sentinel_offset,
            sentinel_start_id + sentinel_count - 1,
        )
        corrupted.append(sentinel_id)
        labels.append(sentinel_id)
        while index < input_ids.numel() and bool(noise_mask[index]):
            labels.append(int(input_ids[index]))
            index += 1
        sentinel_offset += 1

    corrupted.append(eos_token_id)
    labels.append(eos_token_id)
    return (
        torch.tensor(corrupted, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def train_encoder_from_config(config: EncoderTrainConfig) -> None:
    if config.objective.objective != "t5_span_corruption":
        raise ValueError("Only objective=t5_span_corruption is currently supported.")

    seed_everything(config.seed)
    device = resolve_device(config.device)
    tokenizer = load_tokenizer(config.data.tokenizer_path)
    _sync_vocab_settings(config, tokenizer)
    bos_token_id = tokenizer.token_to_id("<s>")
    if bos_token_id is None:
        raise ValueError("Scratch encoder training requires tokenizer token <s>.")
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
        schedule=config.optim.schedule,
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
                segment_boundary_token_id=bos_token_id,
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
            accum_metrics["loss"] = (
                accum_metrics.get("loss", 0.0)
                + float(output.loss.detach()) / group_size
            )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.optim.max_grad_norm,
        )
        optimizer.step()
        scheduler.step()
        state_step += 1
        progress.update(1)
        accum_metrics["lr"] = scheduler.get_last_lr()[0]
        if state_step % config.log_every == 0:
            progress.write(f"step {state_step} | {format_metrics(accum_metrics)}")

    progress.close()
    _save_encoder_checkpoint(checkpoint_dir / "final.pt", model, config, state_step)
    _compute_and_save_latent_stats(
        output_dir / "latent_stats.pt",
        model,
        config,
        tokenizer,
        device,
    )


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
    atomic_torch_save(checkpoint, path)


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
                attention_mask=scratch_encoder_attention_mask(
                    batch["attention_mask"],
                    batch.get("segment_ids"),
                    dtype=next(model.parameters()).dtype,
                ),
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
    atomic_torch_save(stats, path)


DEFAULT_PAIRS = [
    (
        "river_vs_account",
        "She sat by the river bank and watched the water.",
        "He opened a new bank account downtown.",
        "bank",
    ),
    (
        "river_vs_loan",
        "The child played on the river bank after school.",
        "The bank approved the loan yesterday.",
        "bank",
    ),
]


def run_contextuality(
    config_path: Path,
    *,
    checkpoint: Path | None,
    device_name: str,
) -> None:
    config = load_config(config_path)
    if config.model.embedding_source != "scratch_t5_encoder":
        raise ValueError("Encoder contextuality check requires scratch_t5_encoder config.")

    tokenizer_core = load_tokenizer(config.data.tokenizer_path)
    pad_token_id = tokenizer_core.token_to_id("<pad>")
    model_config = resolve_model_config(
        config,
        vocab_size=tokenizer_core.get_vocab_size(),
        pad_token_id=(
            config.model.pad_token_id
            if pad_token_id is None
            else pad_token_id
        ),
    )
    device = resolve_diagnostic_device(device_name)
    model = BabyLMELF(model_config).to(device)
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(select_model_weights(payload, weights="ema"))
    model.eval()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(config.data.tokenizer_path),
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        mask_token="<mask>",
        cls_token="<s>",
        sep_token="</s>",
    )
    cls_id = tokenizer.convert_tokens_to_ids("<s>")

    print(f"config: {config_path}")
    print(f"checkpoint: {checkpoint or 'config encoder checkpoint only'}")
    print("pair\tword\tcosine\ttokens_a\ttokens_b")
    for name, text_a, text_b, word in DEFAULT_PAIRS:
        vec_a, tokens_a = encode_word(model, tokenizer, cls_id, text_a, word, device)
        vec_b, tokens_b = encode_word(model, tokenizer, cls_id, text_b, word, device)
        cosine = F.cosine_similarity(vec_a, vec_b, dim=0).item()
        print(f"{name}\t{word}\t{cosine:.4f}\t{tokens_a}\t{tokens_b}")


@torch.no_grad()
def encode_word(
    model: BabyLMELF,
    tokenizer: PreTrainedTokenizerFast,
    cls_id: int,
    text: str,
    word: str,
    device: torch.device,
) -> tuple[torch.Tensor, list[str]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = torch.tensor([[cls_id, *encoded["input_ids"]]], device=device)
    attention_mask = torch.ones_like(input_ids)
    offsets = [(0, 0), *encoded["offset_mapping"]]
    word_span = find_word_span(text, word)
    token_indices = [
        index
        for index, (start, end) in enumerate(offsets)
        if start < word_span[1] and end > word_span[0]
    ]
    if not token_indices:
        raise RuntimeError(f"No tokens overlapped {word!r} in: {text}")
    embeddings = model.embed_tokens(input_ids, attention_mask=attention_mask)
    vector = embeddings[0, token_indices].float().mean(dim=0).cpu()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0, token_indices].tolist())
    return vector, tokens


def find_word_span(text: str, word: str) -> tuple[int, int]:
    match = re.search(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Could not find word {word!r} in: {text}")
    return match.span()


def resolve_diagnostic_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return resolved
