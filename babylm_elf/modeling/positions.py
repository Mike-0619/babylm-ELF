from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


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
