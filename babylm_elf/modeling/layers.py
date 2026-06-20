from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmbedder(nn.Module):
    """Sinusoidal scalar timestep embedding followed by a small MLP."""

    def __init__(self, hidden_size: int, frequency_size: int = 256) -> None:
        super().__init__()
        self.frequency_size = frequency_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.frequency_size // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / max(1, half)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_size % 2:
            emb = torch.cat([emb, emb.new_zeros(emb.size(0), 1)], dim=-1)
        return self.mlp(emb)


class SwiGLUFFN(nn.Module):
    """Transformer feed-forward block with SwiGLU gating."""

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, 2 * intermediate_size)
        self.out_proj = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.gate_proj(x).chunk(2, dim=-1)
        x = value * F.silu(gate)
        return self.dropout(self.out_proj(x))


class ELFBlock(nn.Module):
    """Bidirectional denoising Transformer block with timestep modulation."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ffn = SwiGLUFFN(hidden_size, intermediate_size, dropout)
        self.time_mod = nn.Linear(hidden_size, 4 * hidden_size)
        nn.init.zeros_(self.time_mod.weight)
        nn.init.zeros_(self.time_mod.bias)

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_attn, scale_attn, shift_ffn, scale_ffn = self.time_mod(time_emb).chunk(4, dim=-1)

        h = self.norm_attn(x)
        h = modulate(h, shift_attn, scale_attn)
        attn_out, _ = self.attn(
            h,
            h,
            h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out

        h = self.norm_ffn(x)
        h = modulate(h, shift_ffn, scale_ffn)
        x = x + self.ffn(h)
        return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
