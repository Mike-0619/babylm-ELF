from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from babylm_elf.data.schema import (
    DISTRIBUTED_SAMPLER_DROP_LAST,
    HFExportStats,
    TextAudit,
    TokenizationStats,
)
from babylm_elf.data.text import NORMALIZATION_NAME
from babylm_elf.data.token_stream import (
    TOKEN_BYTES,
    TOKEN_STREAM_DTYPE,
    TOKEN_STREAM_ENDIANNESS,
    TOKEN_STREAM_FORMAT,
    validate_token_stream_file,
)


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
    if source.get("words") != data_config.train_word_count:
        raise ValueError(
            "Official source budget mismatch between config and manifest: "
            f"config={data_config.train_word_count!r}, "
            f"manifest={source.get('words')!r}."
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
    validate_manifest_artifacts(manifest, actual_paths)
    validate_token_stream_file(
        data_config.train_path,
        expected_tokens=int(tokenization["stream_tokens"]),
    )
    return manifest
