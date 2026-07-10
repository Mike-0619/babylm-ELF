from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    BertConfig,
    BertForMaskedLM,
    PreTrainedTokenizerFast,
    get_cosine_schedule_with_warmup,
)

from babylm_elf.config import load_config
from babylm_elf.data.datasets import build_dataloader
from babylm_elf.data.tokenizer import load_tokenizer
from babylm_elf.training.trainer import autocast_context, move_batch
from babylm_elf.training.optim import resolve_device, seed_everything


SPECIAL_TOKENS = {
    "unk_token": "<unk>",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
    "mask_token": "<mask>",
    "cls_token": "<s>",
    "sep_token": "</s>",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a plain token MLM baseline with the BabyLM-ELF tokenizer/data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/"
            "2026_10M_learnable_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_bert_head_scaled_size33m_gb128_adamw_lr4e-4_4gpu.yml"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/2026_10M/token_mlm_baseline_bert_seq128/hf_revisions/chck_100M"),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=float, default=0.5)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--special_token_count", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mixed_precision", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_config = load_config(args.config)
    seed_everything(train_config.seed)
    device = resolve_device(args.device)

    raw_tokenizer = load_tokenizer(train_config.data.tokenizer_path)
    hf_tokenizer = build_hf_tokenizer(train_config.data.tokenizer_path)
    vocab_size = raw_tokenizer.get_vocab_size()
    token_ids = {
        role: raw_tokenizer.token_to_id(token)
        for role, token in SPECIAL_TOKENS.items()
    }
    missing_tokens = [
        token for role, token in SPECIAL_TOKENS.items() if token_ids[role] is None
    ]
    if missing_tokens:
        raise ValueError(f"Tokenizer is missing required tokens: {missing_tokens}")

    loader = build_dataloader(
        train_config.data.train_path,
        raw_tokenizer,
        train_config.data.seq_length,
        args.batch_size,
        shuffle=True,
        num_workers=train_config.data.num_workers,
        generator=torch.Generator().manual_seed(train_config.seed),
    )
    microbatches_per_epoch = len(loader)
    total_microbatches = microbatches_per_epoch * args.epochs
    total_steps = math.ceil(total_microbatches / args.gradient_accumulation_steps)
    warmup_steps = int(
        math.ceil(microbatches_per_epoch / args.gradient_accumulation_steps)
        * args.warmup_epochs
    )

    bert_config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=train_config.model.hidden_size,
        num_hidden_layers=train_config.model.num_hidden_layers,
        num_attention_heads=train_config.model.num_attention_heads,
        intermediate_size=train_config.model.intermediate_size,
        hidden_dropout_prob=train_config.model.hidden_dropout_prob,
        attention_probs_dropout_prob=train_config.model.hidden_dropout_prob,
        max_position_embeddings=train_config.model.max_position_embeddings,
        pad_token_id=token_ids["pad_token"],
        bos_token_id=token_ids["bos_token"],
        eos_token_id=token_ids["eos_token"],
        mask_token_id=token_ids["mask_token"],
        cls_token_id=token_ids["cls_token"],
        sep_token_id=token_ids["sep_token"],
        type_vocab_size=1,
    )
    model = BertForMaskedLM(bert_config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    progress = tqdm(total=total_steps, desc="token-mlm")
    iterator = iter(loader)
    microbatches_seen = 0
    step = 0
    while microbatches_seen < total_microbatches:
        group_size = min(
            args.gradient_accumulation_steps,
            total_microbatches - microbatches_seen,
        )
        optimizer.zero_grad(set_to_none=True)
        metrics = {"loss": 0.0, "acc": 0.0}
        for _ in range(group_size):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = move_batch(batch, device)
            corrupted, labels, mlm_mask = make_mlm_batch(
                batch["input_ids"],
                batch["attention_mask"],
                mask_token_id=token_ids["mask_token"],
                vocab_size=vocab_size,
                special_token_count=args.special_token_count,
                mlm_probability=args.mlm_probability,
            )
            with autocast_context(device, args.mixed_precision):
                output = model(
                    input_ids=corrupted,
                    attention_mask=batch["attention_mask"],
                    labels=labels,
                )
                loss = output.loss / group_size
            loss.backward()
            metrics["loss"] += float(output.loss.detach()) / group_size
            metrics["acc"] += masked_accuracy(output.logits.detach(), labels, mlm_mask) / group_size
            microbatches_seen += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1
        progress.update(1)
        if step % 50 == 0:
            progress.write(
                f"step {step} | loss: {metrics['loss']:.4f} | "
                f"acc: {metrics['acc']:.4f} | lr: {scheduler.get_last_lr()[0]:.6g}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    save_hf_tokenizer(hf_tokenizer, args.output_dir, train_config.model.max_position_embeddings)
    print(f"Saved token MLM baseline to {args.output_dir}")


def build_hf_tokenizer(tokenizer_path: str | Path) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        mask_token="<mask>",
        cls_token="<s>",
        sep_token="</s>",
    )


def save_hf_tokenizer(
    tokenizer: PreTrainedTokenizerFast,
    output_dir: Path,
    model_max_length: int,
) -> None:
    tokenizer.save_pretrained(output_dir)
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": model_max_length,
        "padding_side": "right",
        "truncation_side": "right",
        **SPECIAL_TOKENS,
    }
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2),
        encoding="utf-8",
    )
    (output_dir / "special_tokens_map.json").write_text(
        json.dumps(SPECIAL_TOKENS, indent=2),
        encoding="utf-8",
    )


def make_mlm_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    mask_token_id: int,
    vocab_size: int,
    special_token_count: int,
    mlm_probability: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    maskable = attention_mask.bool() & input_ids.ge(special_token_count)
    mlm_mask = torch.rand(input_ids.shape, device=input_ids.device) < mlm_probability
    mlm_mask = mlm_mask & maskable
    for row in range(input_ids.size(0)):
        if bool(maskable[row].any()) and not bool(mlm_mask[row].any()):
            first_maskable = torch.nonzero(maskable[row], as_tuple=False)[0, 0]
            mlm_mask[row, first_maskable] = True

    labels = input_ids.clone()
    labels[~mlm_mask] = -100
    corrupted = input_ids.clone()
    replace_prob = torch.rand(input_ids.shape, device=input_ids.device)
    replace_with_mask = mlm_mask & (replace_prob < 0.8)
    replace_with_random = mlm_mask & (replace_prob >= 0.8) & (replace_prob < 0.9)
    corrupted[replace_with_mask] = mask_token_id
    random_tokens = torch.randint(
        special_token_count,
        vocab_size,
        input_ids.shape,
        device=input_ids.device,
    )
    corrupted[replace_with_random] = random_tokens[replace_with_random]
    return corrupted, labels, mlm_mask


def masked_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mlm_mask: torch.Tensor,
) -> float:
    if not bool(mlm_mask.any()):
        return 0.0
    predictions = logits.argmax(dim=-1)
    return float(predictions[mlm_mask].eq(labels[mlm_mask]).float().mean())


if __name__ == "__main__":
    main()
