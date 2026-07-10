from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path


NORMALIZATION_NAME = "official_format_only"
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
