from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset

from babylm_elf.data.collate import collate_tokenized_batch


class TokenizedTextDataset(Dataset):
    """Fixed-length chunks from torch-saved tokenized documents."""

    def __init__(
        self,
        path: str | Path,
        seq_length: int,
        cls_token_id: int,
        pad_token_id: int,
    ) -> None:
        self.path = Path(path)
        self.seq_length = seq_length
        self.cls_token_id = cls_token_id
        self.pad_token_id = pad_token_id
        documents = torch.load(self.path, map_location="cpu")
        self.segments = list(self._make_segments(documents))
        if not self.segments:
            raise ValueError(f"No usable token segments found in {self.path}")

    def _make_segments(self, documents: Iterable[torch.Tensor]) -> Iterable[torch.Tensor]:
        payload_len = self.seq_length - 1
        for document in documents:
            if len(document) == 0:
                continue
            document = document.long()
            for offset in range(0, len(document), payload_len):
                chunk = document[offset : offset + payload_len]
                if len(chunk) > 0:
                    yield torch.cat([torch.tensor([self.cls_token_id]), chunk])

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        input_ids = self.segments[index][: self.seq_length]
        attention_mask = torch.ones_like(input_ids)
        pad_len = self.seq_length - input_ids.numel()
        if pad_len > 0:
            input_ids = torch.cat(
                [input_ids, input_ids.new_full((pad_len,), self.pad_token_id)]
            )
            attention_mask = torch.cat([attention_mask, attention_mask.new_zeros(pad_len)])
        return {"input_ids": input_ids.long(), "attention_mask": attention_mask.long()}


def build_dataloader(
    path: str | Path,
    tokenizer,
    seq_length: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    cls_token_id = _required_token_id(tokenizer, "<s>")
    pad_token_id = _required_token_id(tokenizer, "<pad>")
    dataset = TokenizedTextDataset(path, seq_length, cls_token_id, pad_token_id)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_tokenized_batch,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
    )


def _required_token_id(tokenizer, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"Tokenizer is missing required token: {token}")
    return token_id


def export_hf_split_to_text(
    dataset_name: str,
    split: str,
    output_path: str | Path,
    text_field: str = "text",
    config_name: str | None = None,
) -> None:
    """Materialize one Hugging Face dataset split as paragraph-separated text."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install the 'datasets' package to prepare Hugging Face BabyLM data."
        ) from exc

    if not dataset_name:
        raise ValueError("Set data.hf_dataset before preparing a Hugging Face source.")

    dataset = load_dataset(dataset_name, config_name, split=split)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in dataset:
            if text_field not in row:
                raise ValueError(
                    f"Field '{text_field}' was not found in dataset row. "
                    f"Available fields: {sorted(row.keys())}"
                )
            text = str(row[text_field]).strip()
            if not text:
                continue
            handle.write(text)
            handle.write("\n\n")
            count += 1
    print(f"Saved {count} Hugging Face rows from {dataset_name}:{split} to {output_path}")
