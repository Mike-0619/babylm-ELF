from __future__ import annotations

from array import array
import os
from pathlib import Path
import sys

import numpy as np
import torch

from babylm_elf.data.schema import TokenizationStats
from babylm_elf.data.text import iter_documents


TOKEN_STREAM_FORMAT = "flat_int16_le_v1"
TOKEN_STREAM_DTYPE = "int16"
TOKEN_STREAM_ENDIANNESS = "little"
TOKEN_BYTES = 2
MAX_TOKEN_ID = 32767
DEFAULT_BUFFER_TOKENS = 1 << 20


def write_token_stream(
    tokenizer,
    input_path: str | Path,
    output_path: str | Path,
    *,
    buffer_tokens: int = DEFAULT_BUFFER_TOKENS,
) -> TokenizationStats:
    """Encode rows into one BOS-delimited, little-endian int16 stream."""
    if buffer_tokens <= 0:
        raise ValueError("buffer_tokens must be positive.")
    if array("h").itemsize != TOKEN_BYTES:
        raise RuntimeError("flat_int16_le_v1 requires a 2-byte signed short.")
    bos_token_id = _required_token_id(tokenizer, "<s>")
    unk_token_id = _required_token_id(tokenizer, "<unk>")
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size - 1 > MAX_TOKEN_ID:
        raise ValueError(
            f"Vocabulary size {vocab_size:,} exceeds signed int16 capacity."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    buffer = array("h")
    frequencies = np.zeros(vocab_size, dtype=np.int64)
    rows = 0
    words = 0
    subwords = 0
    unk_tokens = 0

    try:
        with temporary_path.open("wb") as handle:
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
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

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
    validate_token_stream_file(output_path, expected_tokens=stats.stream_tokens)
    print(
        f"Saved {stats.rows:,} rows as {stats.stream_tokens:,} flat int16 tokens "
        f"({stats.subwords:,} subwords + {stats.bos_tokens:,} BOS) to {output_path}"
    )
    return stats


def open_token_stream(path: str | Path) -> torch.Tensor:
    path = Path(path)
    token_count = validate_token_stream_file(path)
    if sys.byteorder != TOKEN_STREAM_ENDIANNESS:
        raise RuntimeError(
            f"{TOKEN_STREAM_FORMAT} requires a little-endian host, got {sys.byteorder}."
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
            f"Invalid {TOKEN_STREAM_FORMAT} byte size for {path}: {byte_count}."
        )
    token_count = byte_count // TOKEN_BYTES
    if expected_tokens is not None and token_count != expected_tokens:
        raise ValueError(
            f"Token stream length mismatch for {path}: "
            f"found {token_count:,}, expected {expected_tokens:,}."
        )
    return token_count


def _flush_buffer(handle, buffer: array, frequencies: np.ndarray) -> None:
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


def _required_token_id(tokenizer, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"Tokenizer is missing required token: {token}")
    return int(token_id)
