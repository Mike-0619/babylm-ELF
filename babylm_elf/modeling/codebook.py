from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_unit_vectors(
    values: torch.Tensor,
    *,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Normalize rows without amplifying near-zero-row gradients.

    Clamping a tiny norm still exposes the row to a ``1 / eps`` derivative.
    The identity fallback has a bounded derivative and lets optimization move
    an accidentally tiny row back into the regular branch.
    """

    values = values.float()
    norms = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    normalized = values / norms.clamp_min(eps)
    return torch.where(norms > eps, normalized, values)


class SphericalCodebook(nn.Module):
    """Learnable unit-sphere token lookup with tied unembedding."""

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        *,
        embedding_rms: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.lookup_scale = math.sqrt(embedding_size) * float(embedding_rms)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # No padding_idx: pad is an ordinary non-zero codebook row. Attention,
        # not a structurally zero embedding, determines whether it participates.
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.bias)

    def normalized_weight(self) -> torch.Tensor:
        return safe_unit_vectors(self.embedding.weight)

    def lookup(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.normalized_weight()) * self.lookup_scale

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden.float() @ self.normalized_weight().T

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.project(hidden) + self.bias.float()


class UntiedCodebook(nn.Module):
    """Alternative-source lookup and independent output codebook.

    Gaussian and scratch-encoder routes are retained without leaking their
    special cases into the core decoder path.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        *,
        with_embedding: bool,
    ) -> None:
        super().__init__()
        self.embedding = (
            nn.Embedding(vocab_size, embedding_size) if with_embedding else None
        )
        self.weight = nn.Parameter(torch.empty(embedding_size, vocab_size))
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.embedding is not None:
            nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0)
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def lookup(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.embedding is None:
            raise RuntimeError("This codebook uses contextual encoder inputs.")
        return self.embedding(input_ids).float()

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden.float() @ self.weight.float()

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.project(hidden) + self.bias.float()
