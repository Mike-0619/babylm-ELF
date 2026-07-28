from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


SDPA_BACKEND_ENV = "BABYLM_ELF_SDPA_BACKEND"
_VALID_BACKENDS = {"auto", "flash", "efficient", "math"}
_warned_flash_mask_fallback = False


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


def selected_sdpa_backend() -> str:
    backend = os.environ.get(SDPA_BACKEND_ENV, "auto").strip().lower() or "auto"
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"{SDPA_BACKEND_ENV} must be one of auto, flash, efficient, math; "
            f"got {backend!r}."
        )
    return backend


def flash_attention_available() -> bool:
    if not torch.cuda.is_available():
        return False
    checker = getattr(torch.backends.cuda, "is_flash_attention_available", None)
    return bool(checker()) if checker is not None else True


def sdpa_backend_status() -> dict[str, str | bool | None]:
    cuda = torch.cuda.is_available()
    return {
        "env": SDPA_BACKEND_ENV,
        "selected": selected_sdpa_backend(),
        "flash_available": flash_attention_available(),
        "flash_enabled": torch.backends.cuda.flash_sdp_enabled() if cuda else None,
        "efficient_enabled": (
            torch.backends.cuda.mem_efficient_sdp_enabled() if cuda else None
        ),
        "math_enabled": torch.backends.cuda.math_sdp_enabled() if cuda else None,
    }


def sdpa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
) -> torch.Tensor:
    backend = selected_sdpa_backend()
    if backend == "flash" and attn_mask is not None:
        _warn_flash_mask_fallback()
        return _sdpa_attention_with_backend(
            "auto",
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
        )
    try:
        return _sdpa_attention_with_backend(
            backend,
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
        )
    except RuntimeError as exc:
        if backend == "auto":
            raise
        raise RuntimeError(
            _format_sdpa_failure(backend, query, key, value, attn_mask)
        ) from exc


def _sdpa_attention_with_backend(
    backend: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
) -> torch.Tensor:
    with _sdpa_kernel_context(backend):
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
        )


def _warn_flash_mask_fallback() -> None:
    global _warned_flash_mask_fallback
    if _warned_flash_mask_fallback:
        return
    warnings.warn(
        f"{SDPA_BACKEND_ENV}=flash does not support explicit attention masks; "
        "using the fastest compatible SDPA backend for masked batches.",
        RuntimeWarning,
        stacklevel=2,
    )
    _warned_flash_mask_fallback = True


def _sdpa_kernel_context(backend: str):
    if backend == "auto":
        return nullcontext()

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError as exc:
        raise RuntimeError(
            f"{SDPA_BACKEND_ENV}={backend!r} requires PyTorch SDPA backend "
            "selection support."
        ) from exc

    if backend == "flash":
        _require_flash_available()

    backend_map = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
    }
    return sdpa_kernel(backend_map[backend])


def _require_flash_available() -> None:
    if not flash_attention_available():
        raise RuntimeError(
            f"{SDPA_BACKEND_ENV}=flash requested, but PyTorch does not report "
            "FlashAttention as available on this process."
        )


def _format_sdpa_failure(
    backend: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
) -> str:
    return (
        f"SDPA backend {backend!r} failed for "
        f"q={tuple(query.shape)}, k={tuple(key.shape)}, v={tuple(value.shape)}, "
        f"mask={None if attn_mask is None else tuple(attn_mask.shape)}, "
        f"dtype={query.dtype}, device={query.device}."
    )


@dataclass(frozen=True)
class PreparedAttention:
    attention_mask: torch.Tensor | None
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor


class PositionAttention(nn.Module):
    """Build record-aware positions, attention mask, and RoPE once per forward."""

    def __init__(
        self,
        head_dim: int,
        max_text_length: int,
        prefix_length: int,
        theta: float = 10_000.0,
    ) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE head dimension must be even.")
        self.head_dim = head_dim
        self.max_text_length = max_text_length
        self.prefix_length = prefix_length
        self.theta = theta
        cos, sin = self._build_buffers()
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _build_buffers(self) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (
            self.theta
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32)
                / self.head_dim
            )
        )
        text_positions = torch.arange(self.max_text_length, dtype=torch.float32)
        frequencies = torch.outer(text_positions, inv_freq).repeat_interleave(2, -1)
        return (
            torch.cat(
                (torch.ones(self.prefix_length, self.head_dim), frequencies.cos())
            ),
            torch.cat(
                (torch.zeros(self.prefix_length, self.head_dim), frequencies.sin())
            ),
        )

    def reset_buffers(self) -> None:
        cos, sin = self._build_buffers()
        self.cos = cos.to(device=self.cos.device)
        self.sin = sin.to(device=self.sin.device)

    def forward(
        self,
        *,
        attention_mask: torch.Tensor | None,
        segment_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        batch_size: int,
        text_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> PreparedAttention:
        if text_length > self.max_text_length:
            raise ValueError(
                f"Text length {text_length} exceeds max_position_embeddings "
                f"{self.max_text_length}."
            )
        active = self._active_mask(
            attention_mask,
            batch_size=batch_size,
            text_length=text_length,
            device=device,
        )
        segments = self._segments(
            segment_ids,
            active,
            batch_size=batch_size,
            text_length=text_length,
            device=device,
        )
        if segments is not None and active is None:
            active = segments.ge(0)
        text_positions = self._text_positions(
            active,
            segments,
            position_ids,
            batch_size=batch_size,
            text_length=text_length,
            device=device,
        )
        if position_ids is not None and torch.any(
            text_positions >= self.max_text_length
        ):
            raise ValueError(
                "position_ids must be smaller than max_position_embeddings."
            )
        prefix_positions = torch.arange(
            self.prefix_length,
            device=device,
            dtype=torch.long,
        ).expand(batch_size, -1)
        full_positions = torch.cat(
            (prefix_positions, text_positions + self.prefix_length),
            dim=1,
        )
        cos = self.cos[full_positions].to(device=device, dtype=dtype).unsqueeze(1)
        sin = self.sin[full_positions].to(device=device, dtype=dtype).unsqueeze(1)

        prepared_mask = self._attention_mask(active, segments)
        return PreparedAttention(prepared_mask, cos, sin)

    @staticmethod
    def _active_mask(
        attention_mask: torch.Tensor | None,
        *,
        batch_size: int,
        text_length: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        if attention_mask.shape != (batch_size, text_length):
            raise ValueError(
                "attention_mask must have shape "
                f"{(batch_size, text_length)}, got {tuple(attention_mask.shape)}"
            )
        return attention_mask.to(device=device, dtype=torch.bool)

    @staticmethod
    def _text_positions(
        active: torch.Tensor | None,
        segments: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        *,
        batch_size: int,
        text_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        if position_ids is not None:
            if position_ids.shape != (batch_size, text_length):
                raise ValueError(
                    "position_ids must have shape "
                    f"{(batch_size, text_length)}, got {tuple(position_ids.shape)}"
                )
            positions = position_ids.to(device=device, dtype=torch.long)
            if torch.any(positions < 0):
                raise ValueError("position_ids must be non-negative.")
            return positions
        if segments is not None:
            indices = torch.arange(text_length, device=device).expand(batch_size, -1)
            starts = torch.zeros_like(indices)
            starts[:, 1:] = torch.where(
                segments[:, 1:].ne(segments[:, :-1]),
                indices[:, 1:],
                0,
            )
            latest_start = starts.cummax(dim=-1).values
            positions = indices - latest_start
            return positions.masked_fill(~active, 0)
        if active is None:
            return torch.arange(text_length, device=device).expand(batch_size, -1)
        compact = active.long().cumsum(dim=-1) - 1
        return compact.masked_fill(~active, 0)

    @staticmethod
    def _segments(
        segment_ids: torch.Tensor | None,
        active: torch.Tensor | None,
        *,
        batch_size: int,
        text_length: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if segment_ids is None:
            return None
        if segment_ids.shape != (batch_size, text_length):
            raise ValueError(
                "segment_ids must have shape "
                f"{(batch_size, text_length)}, got {tuple(segment_ids.shape)}"
            )
        segments = segment_ids.to(device=device, dtype=torch.long)
        if active is None:
            active = segments.ge(0)
        if torch.any(segments[active] < 0):
            raise ValueError("Active tokens must have non-negative segment_ids.")
        return segments.masked_fill(~active, -1)

    def _attention_mask(
        self,
        active: torch.Tensor | None,
        segments: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if segments is None:
            if active is None or bool(active.all()):
                return None
            prefix_mask = torch.ones(
                active.size(0),
                self.prefix_length,
                device=active.device,
                dtype=torch.bool,
            )
            return torch.cat((prefix_mask, active), dim=1)[:, None, None, :]

        if active is None:
            active = torch.ones_like(segments, dtype=torch.bool)
        batch_size, text_length = segments.shape
        total_length = self.prefix_length + text_length
        allowed = torch.zeros(
            batch_size,
            total_length,
            total_length,
            device=segments.device,
            dtype=torch.bool,
        )
        allowed[:, : self.prefix_length, : self.prefix_length] = True
        allowed[:, self.prefix_length :, : self.prefix_length] = active.unsqueeze(-1)
        same_segment = segments.unsqueeze(-1).eq(segments.unsqueeze(-2))
        text_attention = (
            same_segment
            & active.unsqueeze(-1)
            & active.unsqueeze(-2)
        )
        allowed[:, self.prefix_length :, self.prefix_length :] = text_attention
        return allowed.unsqueeze(1)


def _init_linear(layer: nn.Linear, zero: bool = False) -> nn.Linear:
    if zero:
        nn.init.zeros_(layer.weight)
    else:
        nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


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
