from __future__ import annotations

from dataclasses import dataclass


DISTRIBUTED_SAMPLER_DROP_LAST = True


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
