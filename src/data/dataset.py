from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler


NORMALIZATION_NAME = "official_format_only"


TOKEN_STREAM_FORMAT = "flat_int16_le_v1"
TOKEN_STREAM_DTYPE = "int16"
TOKEN_STREAM_ENDIANNESS = "little"
TOKEN_BYTES = 2
MAX_TOKEN_ID = 32_767
DISTRIBUTED_SAMPLER_DROP_LAST = True


def open_token_stream(path: str | Path) -> torch.Tensor:
    path = Path(path)
    token_count = validate_token_stream_file(path)
    if sys.byteorder != TOKEN_STREAM_ENDIANNESS:
        raise RuntimeError(
            f"{TOKEN_STREAM_FORMAT} requires a little-endian host, "
            f"got {sys.byteorder}."
        )
    return torch.from_file(
        str(path),
        shared=False,
        size=token_count,
        dtype=torch.int16,
    )


def validate_token_stream_file(
    path: str | Path,
    *,
    expected_tokens: int | None = None,
) -> int:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Token stream not found: {path}")
    byte_count = path.stat().st_size
    if byte_count == 0 or byte_count % TOKEN_BYTES:
        raise ValueError(
            f"Invalid {TOKEN_STREAM_FORMAT} byte size for {path}: "
            f"{byte_count}."
        )
    token_count = byte_count // TOKEN_BYTES
    if expected_tokens is not None and token_count != expected_tokens:
        raise ValueError(
            f"Token stream length mismatch for {path}: "
            f"found {token_count:,}, expected {expected_tokens:,}."
        )
    return token_count


def collate_tokenized_batch(batch):
    return default_collate(batch)


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
    bos_token_id = required_token_id(tokenizer, "<s>")
    pad_token_id = required_token_id(tokenizer, "<pad>")
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


def required_token_id(tokenizer, token: str) -> int:
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


@dataclass(frozen=True)
class HFExportStats:
    dataset: str
    config: str | None
    split: str
    fingerprint: str | None
    source_rows: int
    source_words: int
    usable_rows: int
    dropped_rows: int
    normalized_words: int
    revision: str | None = None


@dataclass(frozen=True)
class TextAudit:
    rows: int
    words: int


@dataclass(frozen=True)
class TokenizationStats:
    rows: int
    words: int
    subwords: int
    unk_tokens: int
    vocab_size: int
    emitted_vocab_size: int
    pieces_frequency_lt_10: int
    pieces_frequency_lt_100: int

    @property
    def bos_tokens(self) -> int:
        return self.rows

    @property
    def stream_tokens(self) -> int:
        return self.subwords + self.bos_tokens

    @property
    def tokens_per_word(self) -> float:
        return self.subwords / self.words if self.words else 0.0


MANIFEST_SCHEMA_VERSION = 3
PACKING_STRATEGY = "bos_segmented_epoch_offset_v1"
SEGMENT_IDENTITY = "usable_official_row_bos"
ATTENTION_BOUNDARY = "bos_record_block_diagonal"
EPOCH_OFFSET_POLICY = "deterministic_within_unexposed_tail"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_entry(
    actual_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, str | int]:
    actual_path = Path(actual_path)
    return {
        "path": str(manifest_path),
        "bytes": actual_path.stat().st_size,
        "sha256": file_sha256(actual_path),
    }


def build_data_manifest(
    *,
    export_stats: HFExportStats,
    raw_stats: TextAudit,
    tokenization_stats: TokenizationStats,
    seq_length: int,
    world_size: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    epochs: int,
    artifacts: dict[str, dict[str, str | int]],
) -> dict[str, Any]:
    stream_tokens = tokenization_stats.stream_tokens
    chunks = (
        max(1, (stream_tokens - 1) // seq_length)
        if stream_tokens >= seq_length
        else 0
    )
    if DISTRIBUTED_SAMPLER_DROP_LAST:
        samples_per_rank = chunks // world_size
    else:
        samples_per_rank = math.ceil(chunks / world_size)
    if samples_per_rank == 0:
        raise ValueError(
            f"Only {chunks:,} full chunks are available for world size {world_size}."
        )
    distributed_chunks = samples_per_rank * world_size
    batches_per_rank = math.ceil(samples_per_rank / batch_size)
    steps_per_epoch = math.ceil(batches_per_rank / gradient_accumulation_steps)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source": {
            "hf_dataset": export_stats.dataset,
            "hf_config": export_stats.config,
            "revision": export_stats.revision,
            "split": export_stats.split,
            "fingerprint": export_stats.fingerprint,
            "rows": export_stats.source_rows,
            "words": export_stats.source_words,
        },
        "normalization": {
            "name": NORMALIZATION_NAME,
            "usable_rows": raw_stats.rows,
            "dropped_rows": export_stats.dropped_rows,
            "words": raw_stats.words,
        },
        "tokenization": {
            **asdict(tokenization_stats),
            "bos_tokens": tokenization_stats.bos_tokens,
            "stream_tokens": stream_tokens,
            "bos_fraction": tokenization_stats.bos_tokens / stream_tokens,
            "tokens_per_word": tokenization_stats.tokens_per_word,
            "storage_format": TOKEN_STREAM_FORMAT,
            "dtype": TOKEN_STREAM_DTYPE,
            "endianness": TOKEN_STREAM_ENDIANNESS,
            "bytes_per_token": TOKEN_BYTES,
        },
        "packing": {
            "strategy": PACKING_STRATEGY,
            "segment_identity": SEGMENT_IDENTITY,
            "attention_boundary": ATTENTION_BOUNDARY,
            "epoch_offset_policy": EPOCH_OFFSET_POLICY,
            "seq_length": seq_length,
            "drop_incomplete": True,
            "chunks": chunks,
            "world_size": world_size,
            "distributed_sampler_drop_last": DISTRIBUTED_SAMPLER_DROP_LAST,
            "samples_per_rank": samples_per_rank,
            "distributed_chunks_per_epoch": distributed_chunks,
            "distributed_chunks_dropped_per_epoch": chunks - distributed_chunks,
            "per_device_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "dataloader_batches_per_rank": batches_per_rank,
            "optimizer_steps_per_epoch": steps_per_epoch,
            "epochs": epochs,
            "total_optimizer_steps": steps_per_epoch * epochs,
        },
        "artifacts": artifacts,
    }


def write_data_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def load_data_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Data manifest not found: {path}. Re-run the canonical prepare job."
        )
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported data manifest schema in {path}: "
            f"{manifest.get('schema_version')!r}; "
            f"expected {MANIFEST_SCHEMA_VERSION}."
        )
    return manifest


def validate_manifest_artifacts(
    manifest: dict[str, Any],
    actual_paths: dict[str, str | Path],
    *,
    verify_hash: bool = False,
) -> None:
    artifacts = manifest.get("artifacts", {})
    for name, raw_path in actual_paths.items():
        path = Path(raw_path)
        entry = artifacts.get(name)
        if not entry:
            raise ValueError(f"Data manifest is missing artifacts.{name}.")
        if Path(str(entry.get("path", ""))) != path:
            raise ValueError(
                f"Manifest path mismatch for {name}: "
                f"config={path}, manifest={entry.get('path')!r}."
            )
        if path.stat().st_size != entry.get("bytes"):
            raise ValueError(f"Manifest byte size mismatch for {name}: {path}")
        if verify_hash and file_sha256(path) != entry.get("sha256"):
            raise ValueError(f"Manifest SHA-256 mismatch for {name}: {path}")


def validate_training_data_manifest(data_config, tokenizer) -> dict[str, Any]:
    """Validate immutable preparation metadata without rescanning raw text."""
    if not data_config.manifest_path:
        raise ValueError("Canonical training requires data.manifest_path.")
    manifest = load_data_manifest(data_config.manifest_path)
    source = manifest.get("source", {})
    normalization = manifest.get("normalization", {})
    tokenization = manifest.get("tokenization", {})
    packing = manifest.get("packing", {})
    if normalization.get("name") != NORMALIZATION_NAME:
        raise ValueError(
            "Data manifest normalization mismatch: expected "
            f"{NORMALIZATION_NAME!r}, found {normalization.get('name')!r}."
        )
    expected_source = {"words": data_config.train_word_count}
    if data_config.source == "huggingface":
        expected_source.update(
            {
                "hf_dataset": data_config.hf_dataset,
                "hf_config": data_config.hf_config,
                "revision": data_config.hf_revision,
                "split": data_config.hf_train_split,
            }
        )
    source_mismatches = {
        key: (source.get(key), expected)
        for key, expected in expected_source.items()
        if source.get(key) != expected
    }
    if source_mismatches:
        details = ", ".join(
            f"{key}: manifest={found!r}, expected={expected!r}"
            for key, (found, expected) in source_mismatches.items()
        )
        raise ValueError(
            "Official dataset identity differs from the pinned config: "
            + details
            + "."
        )
    expected_storage = {
        "storage_format": TOKEN_STREAM_FORMAT,
        "dtype": TOKEN_STREAM_DTYPE,
        "endianness": TOKEN_STREAM_ENDIANNESS,
        "bytes_per_token": TOKEN_BYTES,
    }
    for key, expected in expected_storage.items():
        if tokenization.get(key) != expected:
            raise ValueError(
                f"Token stream {key} mismatch: "
                f"manifest={tokenization.get(key)!r}, expected={expected!r}."
            )
    expected_packing = {
        "strategy": PACKING_STRATEGY,
        "segment_identity": SEGMENT_IDENTITY,
        "attention_boundary": ATTENTION_BOUNDARY,
        "epoch_offset_policy": EPOCH_OFFSET_POLICY,
    }
    for key, expected in expected_packing.items():
        if packing.get(key) != expected:
            raise ValueError(
                f"Data packing {key} mismatch: "
                f"manifest={packing.get(key)!r}, expected={expected!r}."
            )

    vocab_size = tokenizer.get_vocab_size()
    if tokenization.get("vocab_size") != vocab_size:
        raise ValueError(
            "Tokenizer vocabulary differs from the data manifest: "
            f"loaded={vocab_size:,}, manifest={tokenization.get('vocab_size')!r}."
        )
    tokenizer_artifact = manifest.get("artifacts", {}).get("tokenizer")
    if not tokenizer_artifact:
        raise ValueError("Data manifest is missing artifacts.tokenizer.")
    actual_tokenizer_hash = file_sha256(data_config.tokenizer_path)
    if actual_tokenizer_hash != tokenizer_artifact.get("sha256"):
        raise ValueError(
            "Tokenizer hash differs from the data manifest: "
            f"loaded={actual_tokenizer_hash}, "
            f"manifest={tokenizer_artifact.get('sha256')}."
        )

    actual_paths = {
        "tokenizer": data_config.tokenizer_path,
        "tokenized": data_config.train_path,
    }
    if data_config.train_text:
        actual_paths["raw"] = data_config.train_text
    validate_manifest_artifacts(manifest, actual_paths, verify_hash=True)
    validate_token_stream_file(
        data_config.train_path,
        expected_tokens=int(tokenization["stream_tokens"]),
    )
    return manifest
