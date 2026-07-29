from __future__ import annotations

from array import array
from dataclasses import dataclass, replace
import json
import os
import re
from collections.abc import Iterator
from pathlib import Path
import resource
import shutil
import sys
import unicodedata

import numpy as np
import torch.distributed as dist
from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from src.config import load_config
from src.data.dataset import (
    HFExportStats,
    MAX_TOKEN_ID,
    NORMALIZATION_NAME,
    TOKEN_BYTES,
    TOKEN_STREAM_ENDIANNESS,
    TextAudit,
    TokenizationStats,
    artifact_entry,
    build_data_manifest,
    build_dataloader,
    required_token_id,
    validate_manifest_artifacts,
    validate_token_stream_file,
    validate_training_data_manifest,
    write_data_manifest,
)

_FORMAT_CHAR_TO_SPACE_RE = re.compile("[\u200b\ufeff]")
_FORMAT_CHAR_TO_DELETE_RE = re.compile(
    "[\u00ad\u061c\u200c\u200d\u200e\u200f\u202a-\u202e\u2060\u2066-\u2069]"
)


def normalize_document(document: str) -> str:
    """Apply the sole normalization used by the 2026 official-data route."""
    document = document.replace("\r\n", "\n").replace("\r", "\n")
    document = document.replace("\t", " ")
    document = _FORMAT_CHAR_TO_SPACE_RE.sub(" ", document)
    document = _FORMAT_CHAR_TO_DELETE_RE.sub("", document)
    return document.strip()


def iter_documents(
    path: str | Path,
    chunk_size: int = 1024 * 1024,
) -> Iterator[str]:
    """Yield normalized, non-empty paragraph-delimited source rows."""
    path = Path(path)
    pending = ""
    with path.open("r", encoding="utf-8") as handle:
        while chunk := handle.read(chunk_size):
            pending += chunk
            documents = pending.split("\n\n")
            pending = documents.pop()
            for document in documents:
                document = normalize_document(document)
                if document:
                    yield document

    document = normalize_document(pending)
    if document:
        yield document


def count_words(path: str | Path) -> int:
    return sum(len(document.split()) for document in iter_documents(path))


SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>", "<mask>"] + [
    f"<special_{i}>" for i in range(11)
]


def load_tokenizer(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def empty_control_token_ids(tokenizer: Tokenizer) -> set[int]:
    excluded: set[int] = set()
    for token, token_id in tokenizer.get_vocab().items():
        normalized = token.replace("Ġ", "").replace("▁", "").strip()
        if normalized == "" or any(
            unicodedata.category(char) in {"Cc", "Cf"}
            for char in normalized
        ):
            excluded.add(int(token_id))
    return excluded


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
                    "[^\\r\\n\\p{L}\\p{N}]?"
                    "[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*"
                    "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|"
                    "[^\\r\\n\\p{L}\\p{N}]?"
                    "[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+"
                    "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|"
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


DEFAULT_BUFFER_TOKENS = 1 << 20


def export_hf_split_to_text(
    dataset_name: str,
    split: str,
    output_path: str | Path,
    text_field: str = "text",
    config_name: str | None = None,
    expected_source_words: int | None = None,
    revision: str | None = None,
) -> HFExportStats:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install the 'datasets' package to prepare Hugging Face data."
        ) from exc
    if not dataset_name:
        raise ValueError("Set data.hf_dataset before preparing data.")
    dataset = load_dataset(
        dataset_name,
        config_name,
        split=split,
        revision=revision,
    )
    fingerprint = getattr(dataset, "_fingerprint", None)
    advertised_rows = getattr(dataset, "num_rows", None)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_rows = source_words = usable_rows = dropped_rows = 0
    normalized_words = 0
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in dataset:
                if text_field not in row:
                    raise ValueError(
                        f"Missing dataset text field {text_field!r}."
                    )
                source_text = str(row[text_field])
                source_rows += 1
                source_words += len(source_text.split())
                text = normalize_document(source_text)
                if not text:
                    dropped_rows += 1
                    continue
                handle.write(text)
                handle.write("\n\n")
                normalized_words += len(text.split())
                usable_rows += 1
        if advertised_rows is not None and source_rows != int(advertised_rows):
            raise ValueError("Hugging Face row count changed while exporting.")
        if (
            expected_source_words is not None
            and source_words != expected_source_words
        ):
            raise ValueError(
                "Official source word budget mismatch: "
                f"{source_words:,} != {expected_source_words:,}."
            )
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"Saved {usable_rows:,}/{source_rows:,} rows from "
        f"{dataset_name}:{split} to {output_path}; "
        f"source words={source_words:,}, normalized words={normalized_words:,}, "
        f"dropped rows={dropped_rows:,}"
    )
    return HFExportStats(
        dataset=dataset_name,
        config=config_name,
        split=split,
        fingerprint=str(fingerprint) if fingerprint is not None else None,
        source_rows=source_rows,
        source_words=source_words,
        usable_rows=usable_rows,
        dropped_rows=dropped_rows,
        normalized_words=normalized_words,
        revision=revision,
    )


def audit_text(path: str | Path) -> TextAudit:
    rows = words = 0
    for document in iter_documents(path):
        rows += 1
        words += len(document.split())
    return TextAudit(rows=rows, words=words)


def write_token_stream(
    tokenizer,
    input_path: str | Path,
    output_path: str | Path,
    *,
    buffer_tokens: int = DEFAULT_BUFFER_TOKENS,
) -> TokenizationStats:
    if buffer_tokens <= 0:
        raise ValueError("buffer_tokens must be positive.")
    if array("h").itemsize != TOKEN_BYTES:
        raise RuntimeError("flat_int16_le_v1 requires a 2-byte signed short.")
    bos_token_id = required_token_id(tokenizer, "<s>")
    unk_token_id = required_token_id(tokenizer, "<unk>")
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size - 1 > MAX_TOKEN_ID:
        raise ValueError(
            f"Vocabulary size {vocab_size:,} exceeds int16 capacity."
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    buffer = array("h")
    frequencies = np.zeros(vocab_size, dtype=np.int64)
    rows = words = subwords = unk_tokens = 0
    try:
        with temporary.open("wb") as handle:
            for document in iter_documents(input_path):
                ids = tokenizer.encode(document, add_special_tokens=False).ids
                _validate_token_ids(ids, vocab_size)
                buffer.append(bos_token_id)
                buffer.extend(ids)
                rows += 1
                words += len(document.split())
                subwords += len(ids)
                unk_tokens += ids.count(unk_token_id)
                if len(buffer) >= buffer_tokens:
                    _flush_buffer(handle, buffer, frequencies)
            _flush_buffer(handle, buffer, frequencies)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    frequencies[bos_token_id] -= rows
    if frequencies[bos_token_id] < 0:
        raise ValueError("BOS frequency accounting became negative.")
    emitted = frequencies > 0
    stats = TokenizationStats(
        rows=rows,
        words=words,
        subwords=subwords,
        unk_tokens=unk_tokens,
        vocab_size=vocab_size,
        emitted_vocab_size=int(np.count_nonzero(emitted)),
        pieces_frequency_lt_10=int(
            np.count_nonzero(emitted & (frequencies < 10))
        ),
        pieces_frequency_lt_100=int(
            np.count_nonzero(emitted & (frequencies < 100))
        ),
    )
    validate_token_stream_file(
        output_path,
        expected_tokens=stats.stream_tokens,
    )
    print(
        f"Saved {stats.rows:,} rows as {stats.stream_tokens:,} flat int16 "
        f"tokens ({stats.subwords:,} subwords + {stats.bos_tokens:,} BOS) "
        f"to {output_path}"
    )
    return stats


def _flush_buffer(
    handle,
    buffer: array,
    frequencies: np.ndarray,
) -> None:
    if not buffer:
        return
    values = np.frombuffer(buffer, dtype=np.int16)
    frequencies += np.bincount(values, minlength=frequencies.size)
    del values
    if sys.byteorder == TOKEN_STREAM_ENDIANNESS:
        buffer.tofile(handle)
    else:
        swapped = array("h", buffer)
        swapped.byteswap()
        swapped.tofile(handle)
    del buffer[:]


def _validate_token_ids(ids: list[int], vocab_size: int) -> None:
    if not ids:
        return
    minimum = min(ids)
    maximum = max(ids)
    if minimum < 0 or maximum >= vocab_size or maximum > MAX_TOKEN_ID:
        raise ValueError(
            "Tokenizer emitted an out-of-range ID: "
            f"min={minimum}, max={maximum}, vocab_size={vocab_size}."
        )


@dataclass(frozen=True)
class PreparePlan:
    hf_dataset: str
    hf_config: str | None
    hf_revision: str
    hf_train_split: str
    hf_text_field: str
    train_text: Path
    tokenizer_path: Path
    train_output_path: Path
    manifest_path: Path
    canonical_root: Path
    staging_root: Path | None
    vocab_size: int
    source_word_budget: int
    seq_length: int
    world_size: int
    batch_size: int
    gradient_accumulation_steps: int
    epochs: int


def prepare_from_config(
    config_path: Path,
    world_size: int,
    *,
    staging: bool,
) -> None:
    if world_size <= 0:
        raise ValueError("--world-size must be positive.")
    canonical = prepare_plan(config_path, world_size)
    plan = staged_plan(canonical) if staging else canonical
    if staging:
        reset_staging_root(plan)
    build_dataset(plan)
    if staging:
        promote_staging_root(plan)


def smoke_data(config_paths: list[Path]) -> None:
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    try:
        reports = [
            _smoke_config(path, rank, world_size)
            for path in config_paths
        ]
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
        batch_size=config.training.batch_size,
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
    expected_shape = (
        config.training.batch_size,
        config.data.seq_length,
    )
    for key in ("input_ids", "segment_ids"):
        if tuple(batch[key].shape) != expected_shape:
            raise ValueError(
                f"{config_path}: {key} shape "
                f"{tuple(batch[key].shape)} != {expected_shape}."
            )
    active = batch["attention_mask"].bool()
    if (batch["segment_ids"][active] < 0).any():
        raise ValueError(f"{config_path}: active tokens have invalid segments.")
    if (batch["segment_ids"][~active] != -1).any():
        raise ValueError(f"{config_path}: padding segments must be -1.")
    epoch_zero_offset = loader.dataset.epoch_offset
    loader.dataset.set_epoch(1)
    epoch_one_offset = loader.dataset.epoch_offset
    slack = (
        loader.dataset.total_tokens
        - loader.dataset.total_chunks * loader.dataset.seq_length
    )
    if slack and epoch_one_offset == epoch_zero_offset:
        raise ValueError(f"{config_path}: epoch packing offset did not advance.")
    loader.dataset.set_epoch(0)
    return {
        "config": str(config_path),
        "rank": rank,
        "pid": os.getpid(),
        "batches": len(loader),
        "batch_shape": list(batch["input_ids"].shape),
        "epoch_offsets": [epoch_zero_offset, epoch_one_offset],
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def prepare_plan(config_path: Path, world_size: int) -> PreparePlan:
    config = load_config(config_path)
    data = config.data
    if data.source != "huggingface":
        raise ValueError(
            "The canonical 2026 route requires data.source='huggingface'."
        )
    required = {
        "hf_dataset": data.hf_dataset,
        "hf_revision": data.hf_revision,
        "train_text": data.train_text,
        "manifest_path": data.manifest_path,
        "train_word_count": data.train_word_count,
    }
    missing = [name for name, value in required.items() if value in {None, ""}]
    if missing:
        raise ValueError(
            "Canonical prepare config is missing: " + ", ".join(sorted(missing))
        )

    manifest_path = Path(data.manifest_path)
    canonical_root = manifest_path.parent
    paths = {
        "train_text": Path(data.train_text),
        "tokenizer_path": Path(data.tokenizer_path),
        "train_output_path": Path(data.train_path),
    }
    for name, path in paths.items():
        try:
            path.relative_to(canonical_root)
        except ValueError as exc:
            raise ValueError(
                f"{name}={path} must be inside canonical root {canonical_root}."
            ) from exc

    return PreparePlan(
        hf_dataset=str(data.hf_dataset),
        hf_config=data.hf_config,
        hf_revision=str(data.hf_revision),
        hf_train_split=data.hf_train_split,
        hf_text_field=data.hf_text_field,
        train_text=paths["train_text"],
        tokenizer_path=paths["tokenizer_path"],
        train_output_path=paths["train_output_path"],
        manifest_path=manifest_path,
        canonical_root=canonical_root,
        staging_root=None,
        vocab_size=data.tokenizer_vocab_size,
        source_word_budget=int(data.train_word_count),
        seq_length=data.seq_length,
        world_size=world_size,
        batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        epochs=config.training.epochs,
    )


def staged_plan(plan: PreparePlan) -> PreparePlan:
    staging_root = plan.canonical_root.with_name(
        f".{plan.canonical_root.name}.staging"
    )

    def staged(path: Path) -> Path:
        return staging_root / path.relative_to(plan.canonical_root)

    return replace(
        plan,
        train_text=staged(plan.train_text),
        tokenizer_path=staged(plan.tokenizer_path),
        train_output_path=staged(plan.train_output_path),
        manifest_path=staged(plan.manifest_path),
        staging_root=staging_root,
    )


def reset_staging_root(plan: PreparePlan) -> None:
    if plan.staging_root is None:
        raise ValueError("reset_staging_root requires a staged plan.")
    if plan.staging_root.exists():
        shutil.rmtree(plan.staging_root)
    plan.staging_root.mkdir(parents=True)


def build_dataset(plan: PreparePlan) -> dict:
    export_stats = export_hf_split_to_text(
        dataset_name=plan.hf_dataset,
        config_name=plan.hf_config,
        split=plan.hf_train_split,
        output_path=plan.train_text,
        text_field=plan.hf_text_field,
        expected_source_words=plan.source_word_budget,
        revision=plan.hf_revision,
    )
    raw_stats = audit_text(plan.train_text)
    _validate_export(plan, export_stats, raw_stats)
    tokenizer = train_bpe_tokenizer(
        plan.train_text,
        plan.tokenizer_path,
        vocab_size=plan.vocab_size,
    )
    tokenization_stats = write_token_stream(
        tokenizer,
        plan.train_text,
        plan.train_output_path,
    )
    _validate_tokenization(plan, raw_stats, tokenization_stats)

    actual_paths = {
        "raw": plan.train_text,
        "tokenizer": plan.tokenizer_path,
        "tokenized": plan.train_output_path,
    }
    artifacts = {
        name: artifact_entry(path, _canonical_path(plan, path))
        for name, path in actual_paths.items()
    }
    manifest = build_data_manifest(
        export_stats=export_stats,
        raw_stats=raw_stats,
        tokenization_stats=tokenization_stats,
        seq_length=plan.seq_length,
        world_size=plan.world_size,
        batch_size=plan.batch_size,
        gradient_accumulation_steps=plan.gradient_accumulation_steps,
        epochs=plan.epochs,
        artifacts=artifacts,
    )
    validate_manifest_artifacts(manifest, actual_paths)
    write_data_manifest(plan.manifest_path, manifest)
    print(f"Wrote audited data manifest to {plan.manifest_path}")
    return manifest


def promote_staging_root(plan: PreparePlan) -> None:
    if plan.staging_root is None:
        raise ValueError("promote_staging_root requires a staged plan.")
    backup_root = plan.canonical_root.with_name(f".{plan.canonical_root.name}.old")
    if backup_root.exists():
        shutil.rmtree(backup_root)
    if plan.canonical_root.exists():
        plan.canonical_root.replace(backup_root)
    try:
        plan.staging_root.replace(plan.canonical_root)
    except BaseException:
        if backup_root.exists() and not plan.canonical_root.exists():
            backup_root.replace(plan.canonical_root)
        raise
    if backup_root.exists():
        shutil.rmtree(backup_root)
    print(f"Promoted audited data to {plan.canonical_root}")


def _validate_export(
    plan: PreparePlan,
    export_stats: HFExportStats,
    raw_stats: TextAudit,
) -> None:
    if export_stats.revision != plan.hf_revision:
        raise ValueError(
            "Hugging Face revision mismatch: "
            f"found {export_stats.revision!r}, expected {plan.hf_revision!r}."
        )
    if (raw_stats.rows, raw_stats.words) != (
        export_stats.usable_rows,
        export_stats.normalized_words,
    ):
        raise ValueError(
            "Export and raw-text audit disagree: "
            f"export=({export_stats.usable_rows:,} rows, "
            f"{export_stats.normalized_words:,} words), "
            f"audit=({raw_stats.rows:,} rows, {raw_stats.words:,} words)."
        )


def _validate_tokenization(
    plan: PreparePlan,
    raw_stats: TextAudit,
    stats: TokenizationStats,
) -> None:
    if (stats.rows, stats.words) != (raw_stats.rows, raw_stats.words):
        raise ValueError(
            "Tokenizer input differs from the raw-text audit: "
            f"tokenizer=({stats.rows:,} rows, {stats.words:,} words), "
            f"raw=({raw_stats.rows:,} rows, {raw_stats.words:,} words)."
        )
    if stats.vocab_size != plan.vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary has {stats.vocab_size:,} entries; "
            f"expected {plan.vocab_size:,}."
        )
    if stats.unk_tokens:
        raise ValueError(
            f"Tokenizer produced {stats.unk_tokens:,} <unk> tokens; expected zero."
        )


def _canonical_path(plan: PreparePlan, path: Path) -> Path:
    source_root = plan.staging_root or plan.canonical_root
    return plan.canonical_root / path.relative_to(source_root)
