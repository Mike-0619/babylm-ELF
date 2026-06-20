from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>", "<mask>"] + [
    f"<special_{i}>" for i in range(11)
]


def load_tokenizer(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def train_bpe_tokenizer(
    input_path: str | Path,
    output_path: str | Path,
    vocab_size: int = 16384,
    min_frequency: int = 2,
) -> Tokenizer:
    tokenizer = build_bpe_tokenizer()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(_line_iterator(Path(input_path)), trainer)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    return tokenizer


def build_bpe_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>", byte_fallback=False))
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.Prepend(" "),
            normalizers.NFKC(),
            normalizers.Replace(Regex("\n"), "\n "),
            normalizers.Replace(Regex(" *\n"), "\n"),
        ]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            pre_tokenizers.Split(Regex(".{1,24}"), behavior="isolated"),
        ]
    )
    tokenizer.decoder = decoders.Sequence(
        [
            decoders.ByteLevel(add_prefix_space=False, use_regex=False),
            decoders.Strip(" ", 1, 0),
            decoders.Replace("\n ", "\n"),
        ]
    )
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A",
        special_tokens=[("<s>", 1)],
    )
    return tokenizer


def _line_iterator(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield line
