from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_length: int, prefix_length: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_length = max_length
        self.prefix_length = prefix_length
        self.theta = theta
        cos, sin = self._build_buffers()
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _build_buffers(self) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (
            self.theta
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.float32)[: self.dim // 2]
                / self.dim
            )
        )
        positions = torch.arange(self.max_length, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq).repeat_interleave(2, dim=-1)
        cos = torch.cat(
            (torch.ones(self.prefix_length, self.dim), freqs.cos()),
            dim=0,
        )
        sin = torch.cat(
            (torch.zeros(self.prefix_length, self.dim), freqs.sin()),
            dim=0,
        )
        return cos, sin

    def reset_buffers(self) -> None:
        cos, sin = self._build_buffers()
        self.cos = cos.to(device=self.cos.device)
        self.sin = sin.to(device=self.sin.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(-2)
        cos = self.cos[:length].to(device=x.device, dtype=x.dtype)
        sin = self.sin[:length].to(device=x.device, dtype=x.dtype)
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
        rope: RotaryEmbedding,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, length, hidden = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        q = rope(self.q_norm(q))
        k = rope(self.k_norm(k))
        mask = None
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :].bool()
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
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
        rope: RotaryEmbedding,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), rope, attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, eps: float) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps)
        self.linear = _init_linear(nn.Linear(hidden_size, output_size), zero=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))
