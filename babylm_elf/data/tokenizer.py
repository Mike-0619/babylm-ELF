from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from babylm_elf.data.text import iter_documents


SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>", "<mask>"] + [
    f"<special_{i}>" for i in range(11)
]


def load_tokenizer(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def train_bpe_tokenizer(
    input_path: str | Path,
    output_path: str | Path,
    vocab_size: int = 16384,
) -> Tokenizer:
    tokenizer = build_bpe_tokenizer()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    iterator = iter_documents(Path(input_path))
    tokenizer.train_from_iterator(iterator, trainer)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    _strip_byte_alphabet_added_tokens(output_path)
    tokenizer = load_tokenizer(output_path)
    return tokenizer


def build_bpe_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(
        BPE(
            unk_token="<unk>",
            byte_fallback=False,
            fuse_unk=False,
            ignore_merges=True,
        )
    )
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
            pre_tokenizers.Split(
                Regex(
                    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|"
                    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|"
                    " ?\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                ),
                behavior="isolated",
                invert=False,
            ),
            pre_tokenizers.ByteLevel(
                add_prefix_space=False,
                use_regex=False,
                trim_offsets=True,
            ),
            pre_tokenizers.Split(
                Regex(".{1,24}"),
                behavior="isolated",
                invert=False,
            ),
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
        pair="<s> $A <s> $B",
        special_tokens=[("<s>", 1)],
    )
    return tokenizer


def _strip_byte_alphabet_added_tokens(path: Path) -> None:
    with path.open("r", encoding="utf-8") as handle:
        tokenizer_json = json.load(handle)
    added_tokens = tokenizer_json.get("added_tokens", [])
    if len(added_tokens) >= 256:
        tokenizer_json["added_tokens"] = added_tokens[:-256]
    with path.open("w", encoding="utf-8") as handle:
        json.dump(tokenizer_json, handle, ensure_ascii=False, indent=4)
