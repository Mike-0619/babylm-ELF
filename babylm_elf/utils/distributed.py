from __future__ import annotations

import os

import torch
import torch.distributed as dist


def distributed_requested() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def init_distributed_from_env() -> bool:
    if not distributed_requested():
        return False
    if not dist.is_available():
        raise RuntimeError("Distributed training requested, but torch.distributed is unavailable.")
    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA because NCCL was requested.")
    torch.cuda.set_device(get_local_rank())
    if not is_distributed():
        dist.init_process_group(backend="nccl")
    return True


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0
