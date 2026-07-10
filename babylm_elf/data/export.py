from __future__ import annotations

from pathlib import Path

from babylm_elf.data.schema import HFExportStats, TextAudit
from babylm_elf.data.text import iter_documents, normalize_document


def export_hf_split_to_text(
    dataset_name: str,
    split: str,
    output_path: str | Path,
    text_field: str = "text",
    config_name: str | None = None,
    expected_source_words: int | None = None,
) -> HFExportStats:
    """Export every official row in order using format-only normalization."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install the 'datasets' package to prepare Hugging Face BabyLM data."
        ) from exc

    if not dataset_name:
        raise ValueError("Set data.hf_dataset before preparing a Hugging Face source.")
    if expected_source_words is not None and expected_source_words <= 0:
        raise ValueError(
            "expected_source_words must be positive, "
            f"got {expected_source_words}."
        )

    dataset = load_dataset(dataset_name, config_name, split=split)
    fingerprint = getattr(dataset, "_fingerprint", None)
    advertised_rows = getattr(dataset, "num_rows", None)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_rows = 0
    source_words = 0
    usable_rows = 0
    dropped_rows = 0
    normalized_words = 0
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            for row in dataset:
                if text_field not in row:
                    raise ValueError(
                        f"Field '{text_field}' was not found in dataset row. "
                        f"Available fields: {sorted(row.keys())}"
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
            raise ValueError(
                "Hugging Face row count changed while exporting: consumed "
                f"{source_rows:,}, dataset reports {int(advertised_rows):,}."
            )
        if expected_source_words is not None and source_words != expected_source_words:
            raise ValueError(
                "Official source word budget mismatch before normalization: "
                f"found {source_words:,}, expected {expected_source_words:,}."
            )
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(
        f"Saved {usable_rows:,}/{source_rows:,} Hugging Face rows from "
        f"{dataset_name}:{split} to {output_path}; source words={source_words:,}, "
        f"normalized words={normalized_words:,}, dropped rows={dropped_rows:,}"
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
    )


def audit_text(path: str | Path) -> TextAudit:
    rows = 0
    words = 0
    for document in iter_documents(path):
        rows += 1
        words += len(document.split())
    return TextAudit(rows=rows, words=words)
