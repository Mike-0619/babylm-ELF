from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class PreparedAttention:
    padding_mask: torch.Tensor | None
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor


class PositionAttention(nn.Module):
    """Build compact positions, padding mask, and RoPE once per forward."""

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
        text_positions = self._text_positions(
            active,
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

        padding_mask = None
        if active is not None and not bool(active.all()):
            prefix_mask = torch.ones(
                batch_size,
                self.prefix_length,
                device=device,
                dtype=torch.bool,
            )
            padding_mask = torch.cat((prefix_mask, active), dim=1)[:, None, None, :]
        return PreparedAttention(padding_mask, cos, sin)

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
        if active is None:
            return torch.arange(text_length, device=device).expand(batch_size, -1)
        compact = active.long().cumsum(dim=-1) - 1
        return compact.masked_fill(~active, 0)
