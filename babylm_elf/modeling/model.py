from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

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
    base_vocab_size: int = 16384
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
    mask_token_id: int = 4
    embedding_rms: float = 1.0
    embedding_source: str = "learnable"
    encoder_checkpoint_path: str | None = None
    latent_stats_path: str | None = None
    encoder_vocab_size: int = 16484
    sentinel_start_id: int = 16384
    sentinel_count: int = 100
    encoder_d_ff: int = 2048
    encoder_d_kv: int = 64
    encoder_num_layers: int = 6
    encoder_num_heads: int = 8
    encoder_dropout_rate: float = 0.1


class BabyLMELF(nn.Module):
    """BabyLM-compliant ELF-B with learnable or frozen scratch-encoder embeddings."""

    def __init__(self, config: BabyLMELFConfig) -> None:
        super().__init__()
        self.config = config
        if config.embedding_source not in {"learnable", "scratch_t5_encoder"}:
            raise ValueError(
                "embedding_source must be 'learnable' or 'scratch_t5_encoder', "
                f"got {config.embedding_source!r}"
            )
        self.token_embedding = (
            nn.Embedding(
                config.vocab_size,
                config.embedding_size,
                padding_idx=config.pad_token_id,
            )
            if config.embedding_source == "learnable"
            else None
        )
        self.scratch_encoder = (
            self._build_scratch_encoder(config)
            if config.embedding_source == "scratch_t5_encoder"
            else None
        )
        latent_mean = torch.zeros(config.embedding_size, dtype=torch.float32)
        latent_std = torch.ones(config.embedding_size, dtype=torch.float32)
        self.register_buffer("latent_mean", latent_mean)
        self.register_buffer("latent_std", latent_std)
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
        self.unembed_kernel = (
            nn.Parameter(torch.empty(config.embedding_size, config.base_vocab_size))
            if config.embedding_source == "scratch_t5_encoder"
            else None
        )
        self.decoder_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.reset_parameters()
        self._load_scratch_encoder_if_configured()
        self._load_latent_stats_if_configured()
        self._freeze_scratch_encoder()

    def reset_parameters(self) -> None:
        if self.token_embedding is not None:
            nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.self_cond_projection.weight)
        nn.init.zeros_(self.self_cond_projection.bias)
        nn.init.xavier_uniform_(self.decoder_projection.weight)
        nn.init.zeros_(self.decoder_projection.bias)
        if self.unembed_kernel is not None:
            nn.init.xavier_uniform_(self.unembed_kernel)
        nn.init.normal_(self.time_tokens, mean=0.0, std=0.02)
        nn.init.normal_(self.self_cond_cfg_tokens, mean=0.0, std=0.02)
        nn.init.normal_(self.mode_tokens, mean=0.0, std=0.02)
        if self.token_embedding is not None and self.config.pad_token_id is not None:
            with torch.no_grad():
                self.token_embedding.weight[self.config.pad_token_id].zero_()

    @property
    def embedding_dtype(self) -> torch.dtype:
        return self.decoder_projection.weight.dtype

    def embed_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.config.embedding_source == "scratch_t5_encoder":
            return self._embed_with_scratch_encoder(input_ids, attention_mask)
        if self.token_embedding is None:
            raise RuntimeError("Learnable token embedding is not initialized.")
        embeddings = F.normalize(
            self.token_embedding(input_ids).float(),
            dim=-1,
        )
        return embeddings * (math.sqrt(self.config.embedding_size) * self.config.embedding_rms)

    def normalized_embedding_weight(self) -> torch.Tensor:
        if self.token_embedding is None:
            raise RuntimeError("Scratch encoder mode uses independent unembedding.")
        return F.normalize(self.token_embedding.weight.float(), dim=-1)

    def decode_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            decoder_hidden = F.gelu(
                self.decoder_projection(hidden.float()),
                approximate="tanh",
            )
            if self.unembed_kernel is not None:
                return decoder_hidden @ self.unembed_kernel.float() + self.decoder_bias
            return decoder_hidden @ self.normalized_embedding_weight().T + self.decoder_bias

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
            logits = self.decode_hidden(hidden)
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
        if self.unembed_kernel is not None:
            return embeddings.float() @ self.unembed_kernel.float() + self.decoder_bias
        return embeddings.float() @ self.normalized_embedding_weight().T + self.decoder_bias

    def _build_scratch_encoder(self, config: BabyLMELFConfig) -> T5EncoderModel:
        from transformers import T5Config, T5EncoderModel

        t5_config = T5Config(
            vocab_size=config.encoder_vocab_size,
            d_model=config.embedding_size,
            d_ff=config.encoder_d_ff,
            d_kv=config.encoder_d_kv,
            num_layers=config.encoder_num_layers,
            num_decoder_layers=config.encoder_num_layers,
            num_heads=config.encoder_num_heads,
            dropout_rate=config.encoder_dropout_rate,
            layer_norm_epsilon=config.layer_norm_eps,
            feed_forward_proj="relu",
            pad_token_id=config.pad_token_id,
            eos_token_id=2,
            decoder_start_token_id=config.pad_token_id,
        )
        return T5EncoderModel(t5_config)

    def _load_scratch_encoder_if_configured(self) -> None:
        if self.scratch_encoder is None or not self.config.encoder_checkpoint_path:
            return
        checkpoint_path = Path(self.config.encoder_checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Scratch encoder checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = checkpoint.get("encoder", checkpoint.get("model", checkpoint))
        try:
            self.scratch_encoder.load_state_dict(state)
        except RuntimeError:
            encoder_state = _strip_prefix_if_present(state, "encoder.")
            self.scratch_encoder.encoder.load_state_dict(encoder_state)

    def _load_latent_stats_if_configured(self) -> None:
        if self.config.embedding_source != "scratch_t5_encoder" or not self.config.latent_stats_path:
            return
        stats_path = Path(self.config.latent_stats_path)
        if not stats_path.exists():
            raise FileNotFoundError(f"Scratch encoder latent stats not found: {stats_path}")
        stats = torch.load(stats_path, map_location="cpu", weights_only=True)
        mean = stats["mean"].float()
        std = stats["std"].float().clamp_min(1.0e-6)
        if mean.shape != self.latent_mean.shape or std.shape != self.latent_std.shape:
            raise ValueError(
                "Latent stats shape mismatch: expected "
                f"{tuple(self.latent_mean.shape)}, got mean={tuple(mean.shape)}, "
                f"std={tuple(std.shape)}"
            )
        self.latent_mean.copy_(mean)
        self.latent_std.copy_(std)

    def _freeze_scratch_encoder(self) -> None:
        if self.scratch_encoder is None:
            return
        self.scratch_encoder.eval()
        for parameter in self.scratch_encoder.parameters():
            parameter.requires_grad_(False)

    def _embed_with_scratch_encoder(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.scratch_encoder is None:
            raise RuntimeError("Scratch encoder is not initialized.")
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).to(torch.long)
        encoder_ids = self._replace_mask_spans_with_sentinels(input_ids, attention_mask)
        was_training = self.scratch_encoder.training
        self.scratch_encoder.eval()
        try:
            with torch.no_grad():
                outputs = self.scratch_encoder(
                    input_ids=encoder_ids,
                    attention_mask=attention_mask,
                ).last_hidden_state
        finally:
            if was_training:
                self.scratch_encoder.train()
        normalized = (outputs.float() - self.latent_mean) / self.latent_std.clamp_min(1.0e-6)
        return normalized.to(dtype=self.embedding_dtype)

    def _replace_mask_spans_with_sentinels(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.mask_token_id is None:
            return input_ids
        mapped = input_ids.clone()
        mask_positions = input_ids.eq(self.config.mask_token_id) & attention_mask.bool()
        for batch_idx in range(input_ids.size(0)):
            sentinel_offset = 0
            in_span = False
            for token_idx in range(input_ids.size(1)):
                if not bool(mask_positions[batch_idx, token_idx]):
                    in_span = False
                    continue
                if not in_span:
                    mapped[batch_idx, token_idx] = min(
                        self.config.sentinel_start_id + sentinel_offset,
                        self.config.sentinel_start_id + self.config.sentinel_count - 1,
                    )
                    sentinel_offset += 1
                    in_span = True
                else:
                    mapped[batch_idx, token_idx] = mapped[batch_idx, token_idx - 1]
        return mapped


def _strip_prefix_if_present(
    state: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    if not any(key.startswith(prefix) for key in state):
        return state
    return {
        key.removeprefix(prefix): value
        for key, value in state.items()
        if key.startswith(prefix)
    }
