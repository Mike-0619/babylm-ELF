from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

from babylm_elf.config import load_config
from babylm_elf.modeling.model import BabyLMELF


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether the scratch encoder gives context-specific token latents."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a scratch_t5_encoder ELF config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional ELF checkpoint. Use this for joint-trained encoder checks.",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if config.model.embedding_source != "scratch_t5_encoder":
        raise ValueError("Encoder contextuality check requires scratch_t5_encoder config.")

    device = resolve_diagnostic_device(args.device)
    model = BabyLMELF(config.model).to(device)
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        state = checkpoint.get("model", checkpoint)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"missing keys: {len(missing)}")
        if unexpected:
            print(f"unexpected keys: {len(unexpected)}")
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

    print(f"config: {args.config}")
    print(f"checkpoint: {args.checkpoint or 'config encoder checkpoint only'}")
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


if __name__ == "__main__":
    main()
