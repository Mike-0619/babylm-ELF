from __future__ import annotations

import argparse
from pathlib import Path

import torch

from babylm_elf.config import load_config
from babylm_elf.data.datasets import export_hf_split_to_text
from babylm_elf.data.tokenizer import load_tokenizer, train_bpe_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare local BabyLM Strict text for BabyLM-ELF.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source", choices=("local_text", "huggingface"))
    parser.add_argument("--hf_dataset")
    parser.add_argument("--hf_config")
    parser.add_argument("--hf_train_split")
    parser.add_argument("--hf_valid_split")
    parser.add_argument("--hf_text_field")
    parser.add_argument("--train_text", type=Path)
    parser.add_argument("--valid_text", type=Path)
    parser.add_argument("--tokenizer_path", type=Path)
    parser.add_argument("--vocab_size", type=int)
    parser.add_argument("--min_frequency", type=int)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--train_tokenizer", action="store_true")
    parser.add_argument("--refresh_raw_text", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = prepare_plan_from_args(args)
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    materialize_raw_text_if_needed(plan)

    if plan.train_tokenizer or not plan.tokenizer_path.exists():
        tokenizer = train_bpe_tokenizer(
            plan.train_text,
            plan.tokenizer_path,
            vocab_size=plan.vocab_size,
            min_frequency=plan.min_frequency,
        )
    else:
        tokenizer = load_tokenizer(plan.tokenizer_path)

    tokenize_text_file(tokenizer, plan.train_text, plan.train_output_path)
    if plan.valid_text is not None and plan.valid_output_path is not None:
        tokenize_text_file(tokenizer, plan.valid_text, plan.valid_output_path)


class PreparePlan:
    def __init__(
        self,
        source: str,
        hf_dataset: str | None,
        hf_config: str | None,
        hf_train_split: str,
        hf_valid_split: str | None,
        hf_text_field: str,
        train_text: Path,
        valid_text: Path | None,
        tokenizer_path: Path,
        output_dir: Path,
        train_output_path: Path,
        valid_output_path: Path | None,
        vocab_size: int,
        min_frequency: int,
        train_tokenizer: bool,
        refresh_raw_text: bool,
    ) -> None:
        self.source = source
        self.hf_dataset = hf_dataset
        self.hf_config = hf_config
        self.hf_train_split = hf_train_split
        self.hf_valid_split = hf_valid_split
        self.hf_text_field = hf_text_field
        self.train_text = train_text
        self.valid_text = valid_text
        self.tokenizer_path = tokenizer_path
        self.output_dir = output_dir
        self.train_output_path = train_output_path
        self.valid_output_path = valid_output_path
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.train_tokenizer = train_tokenizer
        self.refresh_raw_text = refresh_raw_text


def prepare_plan_from_args(args: argparse.Namespace) -> PreparePlan:
    config = load_config(args.config) if args.config is not None else None
    data = config.data if config is not None else None
    model = config.model if config is not None else None

    source = args.source or (data.source if data is not None else "local_text")
    hf_dataset = args.hf_dataset or (data.hf_dataset if data is not None else None)
    hf_config = args.hf_config or (data.hf_config if data is not None else None)
    hf_train_split = args.hf_train_split or (
        data.hf_train_split if data is not None else "train"
    )
    hf_valid_split = args.hf_valid_split or (
        data.hf_valid_split if data is not None else None
    )
    hf_text_field = args.hf_text_field or (data.hf_text_field if data is not None else "text")

    train_text = args.train_text or _path_from_config(data.train_text if data else None)
    if train_text is None:
        raise ValueError("Provide --train_text or set data.train_text in --config.")

    valid_text = args.valid_text or _path_from_config(data.valid_text if data else None)
    tokenizer_path = args.tokenizer_path or _path_from_config(
        data.tokenizer_path if data else None
    )
    if tokenizer_path is None:
        tokenizer_path = Path("data/tokenizer_100M.json")

    train_output_path = _path_from_config(data.train_path if data else None)
    valid_output_path = _path_from_config(data.valid_path if data else None)

    output_dir = args.output_dir
    if output_dir is None and train_output_path is not None:
        output_dir = train_output_path.parent
    if output_dir is None:
        output_dir = Path("data/text_data")

    if train_output_path is None:
        train_output_path = output_dir / f"{train_text.stem}_tokenized.bin"
    if valid_text is not None and valid_output_path is None:
        valid_output_path = output_dir / f"{valid_text.stem}_tokenized.bin"

    vocab_size = args.vocab_size
    if vocab_size is None and data is not None:
        vocab_size = data.tokenizer_vocab_size
    if vocab_size is None and model is not None:
        vocab_size = model.vocab_size
    if vocab_size is None:
        vocab_size = 16384

    min_frequency = args.min_frequency
    if min_frequency is None and data is not None:
        min_frequency = data.tokenizer_min_frequency
    if min_frequency is None:
        min_frequency = 2

    return PreparePlan(
        source=source,
        hf_dataset=hf_dataset,
        hf_config=hf_config,
        hf_train_split=hf_train_split,
        hf_valid_split=hf_valid_split,
        hf_text_field=hf_text_field,
        train_text=train_text,
        valid_text=valid_text,
        tokenizer_path=tokenizer_path,
        output_dir=output_dir,
        train_output_path=train_output_path,
        valid_output_path=valid_output_path,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        train_tokenizer=args.train_tokenizer,
        refresh_raw_text=args.refresh_raw_text,
    )


def _path_from_config(value: str | Path | None) -> Path | None:
    if value in {None, ""}:
        return None
    return Path(value)


def materialize_raw_text_if_needed(plan: PreparePlan) -> None:
    if plan.source == "local_text":
        return
    if plan.source != "huggingface":
        raise ValueError(f"Unknown data source: {plan.source}")
    if plan.hf_dataset in {None, ""}:
        raise ValueError("Set data.hf_dataset for Hugging Face data preparation.")

    if plan.refresh_raw_text or not plan.train_text.exists():
        export_hf_split_to_text(
            dataset_name=plan.hf_dataset,
            config_name=plan.hf_config,
            split=plan.hf_train_split,
            output_path=plan.train_text,
            text_field=plan.hf_text_field,
        )
    else:
        print(f"Using cached train text at {plan.train_text}")

    if plan.valid_text is not None and plan.hf_valid_split:
        if plan.refresh_raw_text or not plan.valid_text.exists():
            export_hf_split_to_text(
                dataset_name=plan.hf_dataset,
                config_name=plan.hf_config,
                split=plan.hf_valid_split,
                output_path=plan.valid_text,
                text_field=plan.hf_text_field,
            )
        else:
            print(f"Using cached valid text at {plan.valid_text}")


def tokenize_text_file(tokenizer, input_path: Path, output_path: Path) -> None:
    documents = []
    with input_path.open("r", encoding="utf-8") as handle:
        text = handle.read()
    for document in text.split("\n\n"):
        document = document.strip()
        if document:
            ids = tokenizer.encode(document, add_special_tokens=False).ids
            documents.append(torch.tensor(ids, dtype=torch.int32))
    torch.save(documents, output_path)
    print(f"Saved {len(documents)} tokenized documents to {output_path}")


if __name__ == "__main__":
    main()
