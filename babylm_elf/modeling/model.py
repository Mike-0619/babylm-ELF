from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from babylm_elf.modeling.heads import TiedDecoderHead
from babylm_elf.modeling.layers import ELFBlock, TimestepEmbedder


@dataclass
class BabyLMELFConfig:
    vocab_size: int = 16384
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    max_position_embeddings: int = 512
    hidden_dropout_prob: float = 0.1
    layer_norm_eps: float = 1.0e-6
    pad_token_id: int = 3


class BabyLMELF(nn.Module):
    """Strict-compliant ELF-style continuous embedding diffusion model."""

    def __init__(self, config: BabyLMELFConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.time_embedder = TimestepEmbedder(config.hidden_size)
        self.blocks = nn.ModuleList(
            [
                ELFBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    dropout=config.hidden_dropout_prob,
                    layer_norm_eps=config.layer_norm_eps,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.flow_head = nn.Linear(config.hidden_size, config.hidden_size)
        self.decoder = TiedDecoderHead(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            eps=config.layer_norm_eps,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 0.02
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=std)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=std)
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)
        if self.config.pad_token_id is not None:
            with torch.no_grad():
                self.token_embedding.weight[self.config.pad_token_id].zero_()

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_position_embeddings "
                f"{self.config.max_position_embeddings}"
            )
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        return self.dropout(x)

    def denoise(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask.eq(0)
        time_emb = self.time_embedder(t)
        h = z_t
        for block in self.blocks:
            h = block(h, time_emb, key_padding_mask=key_padding_mask)
        h = self.final_norm(h)
        return self.flow_head(h)

    def decode_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.decoder(embeddings, self.token_embedding.weight)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        decode_from: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prediction = self.denoise(z_t, t, attention_mask)
        decoder_input = prediction if decode_from is None else decode_from
        decoder_logits = self.decode_embeddings(decoder_input)
        return prediction, decoder_logits
