from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource

import torch.distributed as dist

from babylm_elf.config import load_config
from babylm_elf.data.datasets import build_dataloader
from babylm_elf.data.manifest import validate_training_data_manifest
from babylm_elf.data.tokenizer import load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read canonical mmap data from every torchrun rank."
    )
    parser.add_argument("--config", type=Path, action="append", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    try:
        reports = [_smoke_config(path, rank, world_size) for path in args.config]
        gathered: list[list[dict] | None] = [None] * world_size
        dist.all_gather_object(gathered, reports)
        if rank == 0:
            print(json.dumps(gathered, indent=2), flush=True)
    finally:
        dist.destroy_process_group()


def _smoke_config(config_path: Path, rank: int, world_size: int) -> dict:
    config = load_config(config_path)
    tokenizer = load_tokenizer(config.data.tokenizer_path)
    manifest = validate_training_data_manifest(config.data, tokenizer)
    loader = build_dataloader(
        config.data.train_path,
        tokenizer,
        seq_length=config.data.seq_length,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        distributed=True,
        rank=rank,
        world_size=world_size,
        drop_incomplete=True,
    )
    expected_batches = manifest["packing"]["dataloader_batches_per_rank"]
    if len(loader) != expected_batches:
        raise ValueError(
            f"{config_path}: rank {rank} has {len(loader):,} batches; "
            f"manifest expects {expected_batches:,}."
        )
    batch = next(iter(loader))
    expected_shape = (config.batch_size, config.data.seq_length)
    if tuple(batch["input_ids"].shape) != expected_shape:
        raise ValueError(
            f"{config_path}: rank {rank} batch shape "
            f"{tuple(batch['input_ids'].shape)} != {expected_shape}."
        )
    if tuple(batch["segment_ids"].shape) != expected_shape:
        raise ValueError(
            f"{config_path}: rank {rank} segment shape "
            f"{tuple(batch['segment_ids'].shape)} != {expected_shape}."
        )
    active = batch["attention_mask"].bool()
    if (batch["segment_ids"][active] < 0).any():
        raise ValueError(f"{config_path}: active tokens have invalid segment IDs.")
    if (batch["segment_ids"][~active] != -1).any():
        raise ValueError(f"{config_path}: padding tokens must use segment ID -1.")
    epoch_zero_offset = loader.dataset.epoch_offset
    loader.dataset.set_epoch(1)
    epoch_one_offset = loader.dataset.epoch_offset
    offset_slack = (
        loader.dataset.total_tokens
        - loader.dataset.total_chunks * loader.dataset.seq_length
    )
    if offset_slack and epoch_one_offset == epoch_zero_offset:
        raise ValueError(f"{config_path}: epoch-wise packing offset did not advance.")
    loader.dataset.set_epoch(0)
    return {
        "config": str(config_path),
        "rank": rank,
        "pid": os.getpid(),
        "batches": len(loader),
        "batch_shape": list(batch["input_ids"].shape),
        "input_dtype": str(batch["input_ids"].dtype),
        "segment_dtype": str(batch["segment_ids"].dtype),
        "epoch_offsets": [epoch_zero_offset, epoch_one_offset],
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


if __name__ == "__main__":
    main()
