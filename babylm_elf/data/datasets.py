from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from babylm_elf.data.collate import collate_tokenized_batch
from babylm_elf.data.schema import DISTRIBUTED_SAMPLER_DROP_LAST
from babylm_elf.data.token_stream import open_token_stream


class TokenizedTextDataset(Dataset):
    """Epoch-shifted views over a BOS-delimited memory-mapped token stream."""

    def __init__(
        self,
        path: str | Path,
        seq_length: int,
        pad_token_id: int,
        bos_token_id: int = 1,
        seed: int = 0,
        drop_incomplete: bool = False,
    ) -> None:
        if seq_length <= 0:
            raise ValueError(f"seq_length must be positive, got {seq_length}.")
        self.path = Path(path)
        self.seq_length = seq_length
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.seed = int(seed)
        self.epoch = 0
        self.drop_incomplete = drop_incomplete
        self.stream = open_token_stream(self.path)
        self.total_tokens = self.stream.numel()
        self.total_chunks = (
            (
                max(1, (self.total_tokens - 1) // self.seq_length)
                if self.total_tokens >= self.seq_length
                else 0
            )
            if self.drop_incomplete
            else math.ceil(self.total_tokens / self.seq_length)
        )
        if self.total_chunks == 0:
            raise ValueError(
                f"No full {self.seq_length}-token chunks found in {self.path}"
            )

    def __len__(self) -> int:
        return self.total_chunks

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}.")
        self.epoch = int(epoch)

    @property
    def epoch_offset(self) -> int:
        if not self.drop_incomplete:
            return 0
        slack = self.total_tokens - self.total_chunks * self.seq_length
        modulus = slack + 1
        if modulus <= 1:
            return 0
        stride = _coprime_stride(modulus, self.seed)
        return (self.epoch * stride) % modulus

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = self.epoch_offset + index * self.seq_length
        input_ids = self.stream[start : start + self.seq_length].long()
        attention_mask = torch.ones(input_ids.numel(), dtype=torch.long)
        pad_len = self.seq_length - input_ids.numel()
        if pad_len > 0:
            input_ids = torch.cat(
                [input_ids, input_ids.new_full((pad_len,), self.pad_token_id)]
            )
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_zeros(pad_len)]
            )
        segment_ids = _record_segment_ids(
            input_ids,
            attention_mask,
            bos_token_id=self.bos_token_id,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "segment_ids": segment_ids,
            "sequence_id": torch.tensor(start, dtype=torch.long),
        }


def build_dataloader(
    path: str | Path,
    tokenizer,
    seq_length: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    generator: torch.Generator | None = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
    drop_incomplete: bool = False,
) -> DataLoader:
    bos_token_id = _required_token_id(tokenizer, "<s>")
    pad_token_id = _required_token_id(tokenizer, "<pad>")
    dataset = TokenizedTextDataset(
        path,
        seq_length,
        pad_token_id,
        bos_token_id=bos_token_id,
        seed=seed,
        drop_incomplete=drop_incomplete,
    )
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            # Never pad a rank with repeated chunks. If the number of chunks is
            # not divisible by world size, a shuffled remainder is omitted and
            # rotates across epochs through DistributedSampler.set_epoch().
            drop_last=DISTRIBUTED_SAMPLER_DROP_LAST,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_tokenized_batch,
        pin_memory=torch.cuda.is_available(),
        generator=generator if sampler is None else None,
        drop_last=False,
    )


def _required_token_id(tokenizer, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"Tokenizer is missing required token: {token}")
    return int(token_id)


def _record_segment_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    bos_token_id: int,
) -> torch.Tensor:
    active = attention_mask.bool()
    starts = input_ids.eq(bos_token_id) & active
    segment_ids = starts.long().cumsum(dim=-1)
    return segment_ids.masked_fill(~active, -1)


def _coprime_stride(modulus: int, seed: int) -> int:
    if modulus <= 1:
        return 0
    stride = int(modulus * 0.6180339887498949) % modulus
    if int(seed) % 2:
        stride = (-stride) % modulus
    stride = max(1, stride)
    while math.gcd(stride, modulus) != 1:
        stride = (stride + 1) % modulus
        if stride == 0:
            stride = 1
    return stride
