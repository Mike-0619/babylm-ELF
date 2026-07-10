from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .sdpa import sdpa_attention
except ImportError:
    from babylm_elf.modeling.sdpa import sdpa_attention

try:
    from .positions import PreparedAttention
except ImportError:
    from babylm_elf.modeling.positions import PreparedAttention


def _init_linear(layer: nn.Linear, zero: bool = False) -> nn.Linear:
    if zero:
        nn.init.zeros_(layer.weight)
    else:
        nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        variance = x.float().square().mean(dim=-1, keepdim=True)
        return self.weight.to(dtype) * (x * torch.rsqrt(variance + self.eps).to(dtype))


class BottleneckProjection(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bottleneck_size: int) -> None:
        super().__init__()
        self.down = _init_linear(nn.Linear(input_size, bottleneck_size, bias=False))
        self.up = _init_linear(nn.Linear(bottleneck_size, hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_size: int = 256) -> None:
        super().__init__()
        self.frequency_size = frequency_size
        self.linear1 = nn.Linear(frequency_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        nn.init.normal_(self.linear1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.linear2.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.frequency_size // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(1, half)
        )
        args = t.float().unsqueeze(-1) * frequencies.unsqueeze(0)
        embedding = torch.cat((args.cos(), args.sin()), dim=-1)
        if self.frequency_size % 2:
            embedding = F.pad(embedding, (0, 1))
        return self.linear2(F.silu(self.linear1(embedding)))


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout
        self.qkv = _init_linear(nn.Linear(hidden_size, 3 * hidden_size))
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.out = _init_linear(nn.Linear(hidden_size, hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        attention: PreparedAttention,
    ) -> torch.Tensor:
        batch, length, hidden = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        q = apply_rotary(self.q_norm(q), attention.rope_cos, attention.rope_sin)
        k = apply_rotary(self.k_norm(k), attention.rope_cos, attention.rope_sin)
        output = sdpa_attention(
            q,
            k,
            v,
            attn_mask=attention.attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        output = output.transpose(1, 2).reshape(batch, length, hidden)
        return self.out(output)

class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        inner_size = int(hidden_size * mlp_ratio * 2.0 / 3.0)
        self.input = _init_linear(nn.Linear(hidden_size, 2 * inner_size))
        self.output = _init_linear(nn.Linear(inner_size, hidden_size))
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.input(x).chunk(2, dim=-1)
        x = F.silu(gate) * value
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.output(x)


class ELFBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        eps: float,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps)
        self.attention = Attention(hidden_size, num_heads, dropout)
        self.norm2 = RMSNorm(hidden_size, eps)
        self.mlp = SwiGLU(hidden_size, mlp_ratio, dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention: PreparedAttention,
    ) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), attention)
        x = x + self.mlp(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, eps: float) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps)
        self.linear = _init_linear(nn.Linear(hidden_size, output_size), zero=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))
