from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import re


_DOCUMENT_HEADER = re.compile(r"= = = [^\n]*? = = =")


def clean_document(document: str) -> str:
    document = document.strip()
    document = _DOCUMENT_HEADER.sub("", document)
    return document.strip()


def iter_clean_documents(
    path: str | Path,
    chunk_size: int = 1024 * 1024,
) -> Iterator[str]:
    path = Path(path)
    pending = ""
    with path.open("r", encoding="utf-8") as handle:
        while chunk := handle.read(chunk_size):
            pending += chunk
            documents = pending.split("\n\n")
            pending = documents.pop()
            for document in documents:
                document = clean_document(document)
                if document:
                    yield document

    document = clean_document(pending)
    if document:
        yield document


def count_clean_words(path: str | Path) -> int:
    return sum(len(document.split()) for document in iter_clean_documents(path))
