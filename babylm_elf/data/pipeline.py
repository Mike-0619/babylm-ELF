from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import shutil

from babylm_elf.config import load_config
from babylm_elf.data.export import audit_text, export_hf_split_to_text
from babylm_elf.data.manifest import (
    artifact_entry,
    build_data_manifest,
    validate_manifest_artifacts,
    write_data_manifest,
)
from babylm_elf.data.schema import HFExportStats, TextAudit, TokenizationStats
from babylm_elf.data.token_stream import write_token_stream
from babylm_elf.data.tokenizer import train_bpe_tokenizer


@dataclass(frozen=True)
class PreparePlan:
    hf_dataset: str
    hf_config: str | None
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
    expected_normalized_words: int | None
    expected_dropped_rows: int | None
    expected_subwords: int | None
    expected_stream_tokens: int | None
    seq_length: int
    world_size: int
    batch_size: int
    gradient_accumulation_steps: int
    epochs: int


def prepare_plan(config_path: Path, world_size: int) -> PreparePlan:
    config = load_config(config_path)
    data = config.data
    if data.source != "huggingface":
        raise ValueError(
            "The canonical 2026 route requires data.source='huggingface'."
        )
    required = {
        "hf_dataset": data.hf_dataset,
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
        expected_normalized_words=data.expected_normalized_word_count,
        expected_dropped_rows=data.expected_dropped_rows,
        expected_subwords=data.expected_subword_count,
        expected_stream_tokens=data.expected_stream_tokens,
        seq_length=data.seq_length,
        world_size=world_size,
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        epochs=config.epochs,
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
    if (
        plan.expected_normalized_words is not None
        and raw_stats.words != plan.expected_normalized_words
    ):
        raise ValueError(
            "Normalized official corpus word count changed: "
            f"found {raw_stats.words:,}, "
            f"expected {plan.expected_normalized_words:,}."
        )
    if (
        plan.expected_dropped_rows is not None
        and export_stats.dropped_rows != plan.expected_dropped_rows
    ):
        raise ValueError(
            "Dropped control-only row count changed: "
            f"found {export_stats.dropped_rows:,}, "
            f"expected {plan.expected_dropped_rows:,}."
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
    if plan.expected_subwords is not None and stats.subwords != plan.expected_subwords:
        raise ValueError(
            f"Subword count changed: found {stats.subwords:,}, "
            f"expected {plan.expected_subwords:,}."
        )
    if (
        plan.expected_stream_tokens is not None
        and stats.stream_tokens != plan.expected_stream_tokens
    ):
        raise ValueError(
            f"Logical token stream changed: found {stats.stream_tokens:,}, "
            f"expected {plan.expected_stream_tokens:,}."
        )


def _canonical_path(plan: PreparePlan, path: Path) -> Path:
    source_root = plan.staging_root or plan.canonical_root
    return plan.canonical_root / path.relative_to(source_root)
