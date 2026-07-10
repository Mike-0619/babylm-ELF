from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codebook import SphericalCodebook, UntiedCodebook
from .layers import (
    BottleneckProjection,
    ELFBlock,
    FinalLayer,
    TimestepEmbedder,
)
from .mask_latent import build_embedding_stats_mask_latent
from .positions import PositionAttention


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
    scratch_encoder_trainable: bool = False
    gaussian_embedding_std: float = 1.0
    decoder_head_type: str = "gelu"
    mlm_mask_latent_seed: int = 0
    mlm_mask_latent_scale: float = 1.0
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
        if config.embedding_source not in {"learnable", "scratch_t5_encoder", "gaussian"}:
            raise ValueError(
                "embedding_source must be 'learnable', 'scratch_t5_encoder', "
                "or 'gaussian', "
                f"got {config.embedding_source!r}"
            )
        if config.decoder_head_type not in {"gelu", "bert_mlm", "bert_mlm_scaled"}:
            raise ValueError(
                "decoder_head_type must be 'gelu', 'bert_mlm', "
                f"or 'bert_mlm_scaled', got {config.decoder_head_type!r}"
            )
        self.codebook = (
            SphericalCodebook(
                config.vocab_size,
                config.embedding_size,
                embedding_rms=config.embedding_rms,
            )
            if config.embedding_source == "learnable"
            else UntiedCodebook(
                config.vocab_size,
                config.embedding_size,
                with_embedding=config.embedding_source == "gaussian",
            )
        )
        self.mlm_mask_latent = (
            nn.Parameter(torch.empty(config.embedding_size))
            if self.codebook.embedding is not None
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
        self.denoise_mode_tokens = nn.Parameter(
            torch.empty(1, config.num_model_mode_tokens, config.hidden_size)
        )
        self.decode_mode_tokens = nn.Parameter(
            torch.empty(1, config.num_model_mode_tokens, config.hidden_size)
        )
        prefix_length = (
            config.num_time_tokens
            + config.num_self_cond_cfg_tokens
            + config.num_model_mode_tokens
        )
        self.position_attention = PositionAttention(
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
        self.decoder_layer_norm = (
            nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
            if config.decoder_head_type in {"bert_mlm", "bert_mlm_scaled"}
            else None
        )
        self.decoder_logit_scale = (
            nn.Parameter(torch.zeros(()))
            if config.decoder_head_type == "bert_mlm_scaled"
            else None
        )
        self.reset_parameters()
        self._load_scratch_encoder_if_configured()
        self._load_latent_stats_if_configured()
        self._configure_embedding_trainability()

    def reset_parameters(self) -> None:
        self.codebook.reset_parameters()
        if self.token_embedding is not None:
            if self.config.embedding_source == "gaussian":
                self._init_gaussian_embedding()
        nn.init.xavier_uniform_(self.self_cond_projection.weight)
        nn.init.zeros_(self.self_cond_projection.bias)
        nn.init.xavier_uniform_(self.decoder_projection.weight)
        nn.init.zeros_(self.decoder_projection.bias)
        if self.decoder_layer_norm is not None:
            nn.init.ones_(self.decoder_layer_norm.weight)
            nn.init.zeros_(self.decoder_layer_norm.bias)
        if self.decoder_logit_scale is not None:
            nn.init.zeros_(self.decoder_logit_scale)
        nn.init.normal_(self.time_tokens, mean=0.0, std=0.02)
        nn.init.normal_(self.self_cond_cfg_tokens, mean=0.0, std=0.02)
        nn.init.normal_(self.denoise_mode_tokens, mean=0.0, std=0.02)
        nn.init.normal_(self.decode_mode_tokens, mean=0.0, std=0.02)
        self._init_mlm_mask_latent()

    @property
    def embedding_dtype(self) -> torch.dtype:
        return self.decoder_projection.weight.dtype

    @property
    def token_embedding(self) -> nn.Embedding | None:
        return self.codebook.embedding

    def embed_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.config.embedding_source == "scratch_t5_encoder":
            return self._embed_with_scratch_encoder(input_ids, attention_mask)
        return self.codebook.lookup(input_ids).to(dtype=self.embedding_dtype)

    def decode_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            decoder_hidden = F.gelu(
                self.decoder_projection(hidden.float()),
                approximate="tanh",
            )
            if self.decoder_layer_norm is not None:
                decoder_hidden = self.decoder_layer_norm(decoder_hidden)
            scores = self.codebook.project(decoder_hidden)
            if self.decoder_logit_scale is not None:
                scores = scores * self.decoder_logit_scale.exp()
            return scores + self.codebook.bias.float()

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
        position_ids: torch.Tensor | None = None,
        self_cond_cfg_scale: torch.Tensor | None = None,
        decoder_step_active: torch.Tensor | bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.forward_hidden(
            x,
            t,
            attention_mask=attention_mask,
            position_ids=position_ids,
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
        position_ids: torch.Tensor | None = None,
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

        mode = self._build_mode_tokens(
            decoder_step_active,
            batch=batch,
            device=x.device,
            dtype=x.dtype,
        )
        x = torch.cat((context, mode, x), dim=1)
        prefix_length = context.size(1) + mode.size(1)
        if prefix_length != self.position_attention.prefix_length:
            raise RuntimeError("Runtime prefix length does not match model configuration.")
        prepared_attention = self.position_attention(
            attention_mask=attention_mask,
            position_ids=position_ids,
            batch_size=batch,
            text_length=x.size(1) - prefix_length,
            device=x.device,
            dtype=x.dtype,
        )

        for block in self.blocks:
            x = block(x, prepared_attention)
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
        return self.codebook.decode(embeddings)

    def mlm_mask_latent_value(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.mlm_mask_latent is None:
            raise ValueError(
                "mlm_mask_latent requires a learned token embedding table. "
                "Scratch-encoder BabyLM-ELF models do not expose one."
            )
        return self.mlm_mask_latent.to(device=device, dtype=dtype)

    def _build_mode_tokens(
        self,
        decoder_step_active: torch.Tensor,
        *,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        active = decoder_step_active.to(device=device, dtype=dtype).reshape(batch, 1, 1)
        denoise_mode = self.denoise_mode_tokens.to(device=device, dtype=dtype)
        decode_mode = self.decode_mode_tokens.to(device=device, dtype=dtype)
        return denoise_mode.expand(batch, -1, -1).lerp(
            decode_mode.expand(batch, -1, -1),
            active,
        )

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

    def _init_gaussian_embedding(self) -> None:
        if self.token_embedding is None:
            return
        nn.init.normal_(
            self.token_embedding.weight,
            mean=0.0,
            std=self.config.gaussian_embedding_std,
        )
        with torch.no_grad():
            weight = self.token_embedding.weight
            weight.sub_(weight.mean(dim=-1, keepdim=True))
            std = weight.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1.0e-6)
            weight.div_(std)
            weight.mul_(self.config.gaussian_embedding_std)

    def _init_mlm_mask_latent(self) -> None:
        if self.token_embedding is None or self.mlm_mask_latent is None:
            return
        if self.token_embedding.weight.is_meta or self.mlm_mask_latent.is_meta:
            return
        latent = build_embedding_stats_mask_latent(
            self.token_embedding.weight,
            embedding_size=self.config.embedding_size,
            embedding_rms=self.config.embedding_rms,
            pad_token_id=self.config.pad_token_id,
            seed=self.config.mlm_mask_latent_seed,
            scale=self.config.mlm_mask_latent_scale,
        )
        with torch.no_grad():
            self.mlm_mask_latent.copy_(latent.to(self.mlm_mask_latent))

    def _configure_embedding_trainability(self) -> None:
        if self.config.embedding_source == "gaussian" and self.token_embedding is not None:
            self.token_embedding.weight.requires_grad_(False)
        self._configure_scratch_encoder_trainability()

    def _configure_scratch_encoder_trainability(self) -> None:
        if self.scratch_encoder is None:
            return
        trainable = bool(self.config.scratch_encoder_trainable)
        if not trainable:
            self.scratch_encoder.eval()
        for parameter in self.scratch_encoder.parameters():
            parameter.requires_grad_(trainable)

    def _freeze_scratch_encoder(self) -> None:
        if self.scratch_encoder is None:
            return
        self.config.scratch_encoder_trainable = False
        self._configure_scratch_encoder_trainability()

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
        if self.config.scratch_encoder_trainable:
            outputs = self.scratch_encoder(
                input_ids=encoder_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
        else:
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
