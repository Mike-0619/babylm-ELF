from __future__ import annotations

import torch
import torch.nn as nn


class TiedDecoderHead(nn.Module):
    """Vocabulary decoder tied to the input token embedding table."""

    def __init__(self, hidden_size: int, vocab_size: int, eps: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=eps)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, embeddings: torch.Tensor, embedding_weight: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(embeddings)
        return hidden @ embedding_weight.T + self.bias
