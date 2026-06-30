from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    BottleneckProjection,
    ELFBlock,
    FinalLayer,
    RotaryEmbedding,
    TimestepEmbedder,
)


@dataclass
class BabyLMELFConfig:
    vocab_size: int = 16384
    embedding_size: int = 512
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    max_position_embeddings: int = 1024
    hidden_dropout_prob: float = 0.0
    layer_norm_eps: float = 1.0e-6
    bottleneck_size: int = 128
    num_time_tokens: int = 4
    num_self_cond_cfg_tokens: int = 4
    num_model_mode_tokens: int = 4
    pad_token_id: int = 3
    embedding_rms: float = 1.0


class BabyLMELF(nn.Module):
    """BabyLM-compliant ELF-B using the paper's learnable-embedding variant."""

    def __init__(self, config: BabyLMELFConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_size,
            padding_idx=config.pad_token_id,
        )
        self.self_cond_projection = nn.Linear(2 * config.embedding_size, config.embedding_size)
        self.text_projection = BottleneckProjection(
            config.embedding_size,
            config.hidden_size,
            config.bottleneck_size,
        )
        self.time_embedder = TimestepEmbedder(config.hidden_size)
        self.time_tokens = nn.Parameter(
            torch.empty(1, config.num_time_tokens, config.hidden_size)
        )
        self.self_cond_cfg_embedder = TimestepEmbedder(config.hidden_size)
        self.self_cond_cfg_tokens = nn.Parameter(
            torch.empty(1, config.num_self_cond_cfg_tokens, config.hidden_size)
        )
        self.mode_tokens = nn.Parameter(
            torch.empty(1, config.num_model_mode_tokens, config.hidden_size)
        )

        prefix_length = (
            config.num_time_tokens
            + config.num_self_cond_cfg_tokens
            + config.num_model_mode_tokens
        )
        self.rope = RotaryEmbedding(
            config.hidden_size // config.num_attention_heads,
            config.max_position_embeddings,
            prefix_length,
        )
        mlp_ratio = config.intermediate_size / config.hidden_size
        q1 = config.num_hidden_layers // 4
        q3 = 3 * config.num_hidden_layers // 4
        self.blocks = nn.ModuleList(
            ELFBlock(
                config.hidden_size,
                config.num_attention_heads,
                mlp_ratio,
                config.hidden_dropout_prob if q1 <= index < q3 else 0.0,
                config.layer_norm_eps,
            )
            for index in range(config.num_hidden_layers)
        )
        self.flow_head = FinalLayer(
            config.hidden_size,
            config.embedding_size,
            config.layer_norm_eps,
        )
        self.decoder_projection = nn.Linear(config.hidden_size, config.embedding_size)
        self.decoder_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.self_cond_projection.weight)
        nn.init.zeros_(self.self_cond_projection.bias)
        nn.init.xavier_uniform_(self.decoder_projection.weight)
        nn.init.zeros_(self.decoder_projection.bias)
        nn.init.normal_(self.time_tokens, mean=0.0, std=0.02)
        nn.init.normal_(self.self_cond_cfg_tokens, mean=0.0, std=0.02)
        nn.init.normal_(self.mode_tokens, mean=0.0, std=0.02)
        if self.config.pad_token_id is not None:
            with torch.no_grad():
                self.token_embedding.weight[self.config.pad_token_id].zero_()

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(
            self.token_embedding(input_ids).float(),
            dim=-1,
        )
        return embeddings * (math.sqrt(self.config.embedding_size) * self.config.embedding_rms)

    def normalized_embedding_weight(self) -> torch.Tensor:
        return F.normalize(self.token_embedding.weight.float(), dim=-1)

    def build_context(
        self,
        t: torch.Tensor,
        self_cond_cfg_scale: torch.Tensor,
    ) -> torch.Tensor:
        batch = t.size(0)
        time = self.time_tokens.expand(batch, -1, -1)
        time = time + self.time_embedder(t).unsqueeze(1)
        cfg = self.self_cond_cfg_tokens.expand(batch, -1, -1)
        cfg = cfg + self.self_cond_cfg_embedder(self_cond_cfg_scale).unsqueeze(1)
        return torch.cat((time, cfg), dim=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        self_cond_cfg_scale: torch.Tensor | None = None,
        decoder_step_active: torch.Tensor | bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.forward_hidden(
            x,
            t,
            attention_mask=attention_mask,
            self_cond_cfg_scale=self_cond_cfg_scale,
            decoder_step_active=decoder_step_active,
        )
        with torch.autocast(device_type=x.device.type, enabled=False):
            prediction = self.flow_head(hidden.float())
            decoder_hidden = F.gelu(
                self.decoder_projection(hidden.float()),
                approximate="tanh",
            )
            logits = decoder_hidden @ self.normalized_embedding_weight().T + self.decoder_bias
        return prediction, logits

    def forward_hidden(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        self_cond_cfg_scale: torch.Tensor | None = None,
        decoder_step_active: torch.Tensor | bool | None = None,
    ) -> torch.Tensor:
        batch = x.size(0)
        if self_cond_cfg_scale is None:
            self_cond_cfg_scale = torch.ones(batch, device=x.device, dtype=x.dtype)
        if decoder_step_active is None:
            decoder_step_active = torch.zeros(batch, device=x.device, dtype=x.dtype)
        elif not isinstance(decoder_step_active, torch.Tensor):
            decoder_step_active = torch.full(
                (batch,), float(decoder_step_active), device=x.device, dtype=x.dtype
            )

        with torch.autocast(device_type=x.device.type, enabled=False):
            if x.size(-1) == 2 * self.config.embedding_size:
                x = self.self_cond_projection(x.float())
            elif x.size(-1) != self.config.embedding_size:
                raise ValueError(
                    f"Expected embedding width {self.config.embedding_size} or "
                    f"{2 * self.config.embedding_size}, got {x.size(-1)}"
                )
            x = self.text_projection(x.float())
            context = self.build_context(t, self_cond_cfg_scale)

        mode = self.mode_tokens.expand(batch, -1, -1)
        mode = mode * decoder_step_active.to(mode.dtype).view(-1, 1, 1)
        x = torch.cat((context, mode, x), dim=1)
        prefix_length = context.size(1) + mode.size(1)
        if attention_mask is not None:
            prefix_mask = torch.ones(
                batch,
                prefix_length,
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat((prefix_mask, attention_mask), dim=1)

        for block in self.blocks:
            x = block(x, self.rope, attention_mask)
        return x[:, prefix_length:]

    def denoise(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prediction, _ = self(
            z_t,
            t,
            attention_mask=attention_mask,
            decoder_step_active=False,
        )
        return prediction

    def decode_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings.float() @ self.normalized_embedding_weight().T + self.decoder_bias
