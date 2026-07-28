from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .encoder_modeling import (
        build_scratch_encoder,
        embed_with_scratch_encoder,
        initialize_gaussian_embedding,
        load_latent_stats,
        load_scratch_encoder,
        set_scratch_encoder_trainability,
    )
except ImportError:
    from babylm_elf.modules.encoder import (
        build_scratch_encoder,
        embed_with_scratch_encoder,
        initialize_gaussian_embedding,
        load_latent_stats,
        load_scratch_encoder,
        set_scratch_encoder_trainability,
    )

from .layers import (
    BottleneckProjection,
    ELFBlock,
    FinalLayer,
    PositionAttention,
    TimestepEmbedder,
)


def safe_unit_vectors(
    values: torch.Tensor,
    *,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Normalize rows without amplifying near-zero-row gradients.

    Clamping a tiny norm still exposes the row to a ``1 / eps`` derivative.
    The identity fallback has a bounded derivative and lets optimization move
    an accidentally tiny row back into the regular branch.
    """

    values = values.float()
    norms = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    normalized = values / norms.clamp_min(eps)
    return torch.where(norms > eps, normalized, values)


def build_embedding_stats_mask_latent(
    token_embedding_weight: torch.Tensor,
    *,
    embedding_size: int,
    embedding_rms: float,
    pad_token_id: int | None,
    seed: int,
    scale: float,
) -> torch.Tensor:
    """Initialize a continuous mask latent from embedding statistics."""
    if token_embedding_weight.size(-1) != embedding_size:
        raise ValueError(
            "token_embedding_weight last dimension must match embedding_size; "
            f"got {token_embedding_weight.size(-1)} and {embedding_size}."
        )

    with torch.no_grad():
        weight = safe_unit_vectors(token_embedding_weight.detach())
        weight = weight * (math.sqrt(embedding_size) * embedding_rms)
        if pad_token_id is not None:
            keep = torch.ones(weight.size(0), device=weight.device, dtype=torch.bool)
            if 0 <= pad_token_id < weight.size(0):
                keep[pad_token_id] = False
            weight_for_stats = weight[keep]
        else:
            weight_for_stats = weight
        if weight_for_stats.numel() == 0:
            raise ValueError("Cannot build mask latent from an empty embedding table.")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        fixed_eps = torch.randn(
            embedding_size,
            generator=generator,
            dtype=torch.float32,
        ).to(weight_for_stats.device)
        emb_mean = weight_for_stats.mean(dim=0)
        emb_std = weight_for_stats.std(dim=0, unbiased=False).clamp_min(1.0e-6)
        return (emb_mean + float(scale) * emb_std * fixed_eps).detach()


class SphericalCodebook(nn.Module):
    """Learnable unit-sphere token lookup with tied unembedding."""

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        *,
        embedding_rms: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.lookup_scale = math.sqrt(embedding_size) * float(embedding_rms)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # No padding_idx: pad is an ordinary non-zero codebook row. Attention,
        # not a structurally zero embedding, determines whether it participates.
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.bias)

    def normalized_weight(self) -> torch.Tensor:
        return safe_unit_vectors(self.embedding.weight)

    def lookup(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.normalized_weight()) * self.lookup_scale

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden.float() @ self.normalized_weight().T

class UntiedCodebook(nn.Module):
    """Alternative-source lookup and independent output codebook.

    Gaussian and scratch-encoder routes are retained without leaking their
    special cases into the core decoder path.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        *,
        with_embedding: bool,
    ) -> None:
        super().__init__()
        self.embedding = (
            nn.Embedding(vocab_size, embedding_size) if with_embedding else None
        )
        self.weight = nn.Parameter(torch.empty(embedding_size, vocab_size))
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.embedding is not None:
            nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0)
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def lookup(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.embedding is None:
            raise RuntimeError("This codebook uses contextual encoder inputs.")
        return self.embedding(input_ids).float()

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden.float() @ self.weight.float()

@dataclass(frozen=True)
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
    training_objective: str = "token_mlm"
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
    """BabyLM ELF-S/ELF-B model with learnable or scratch-derived embeddings."""

    def __init__(self, config: BabyLMELFConfig) -> None:
        super().__init__()
        self.config = config
        if config.embedding_source not in {"learnable", "scratch_t5_encoder", "gaussian"}:
            raise ValueError(
                "embedding_source must be 'learnable', 'scratch_t5_encoder', "
                "or 'gaussian', "
                f"got {config.embedding_source!r}"
            )
        if config.training_objective not in {
            "official_noisy_ce",
            "token_mlm",
            "standard_mdlm",
        }:
            raise ValueError(
                "training_objective must be 'official_noisy_ce', 'token_mlm', "
                f"or 'standard_mdlm', got {config.training_objective!r}"
            )
        if (
            config.training_objective == "standard_mdlm"
            and config.embedding_source != "learnable"
        ):
            raise ValueError("standard_mdlm requires embedding_source='learnable'.")
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
            if (
                self.codebook.embedding is not None
                or config.embedding_source == "scratch_t5_encoder"
            )
            else None
        )
        self.scratch_encoder = (
            build_scratch_encoder(config)
            if config.embedding_source == "scratch_t5_encoder"
            else None
        )
        latent_mean = torch.zeros(config.embedding_size, dtype=torch.float32)
        latent_std = torch.ones(config.embedding_size, dtype=torch.float32)
        self.register_buffer("latent_mean", latent_mean)
        self.register_buffer("latent_std", latent_std)
        self.self_cond_projection = nn.Linear(2 * config.embedding_size, config.embedding_size)
        self.decoder_input_projection = None
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
        self.reset_parameters()
        self._configure_objective_modules()
        load_scratch_encoder(
            self.scratch_encoder,
            self.config.encoder_checkpoint_path,
        )
        load_latent_stats(self.config, self.latent_mean, self.latent_std)
        self._configure_embedding_trainability()

    def reset_parameters(self) -> None:
        self.codebook.reset_parameters()
        if self.token_embedding is not None:
            if self.config.embedding_source == "gaussian":
                initialize_gaussian_embedding(
                    self.token_embedding,
                    self.config.gaussian_embedding_std,
                )
        nn.init.xavier_uniform_(self.self_cond_projection.weight)
        nn.init.zeros_(self.self_cond_projection.bias)
        nn.init.xavier_uniform_(self.decoder_projection.weight)
        nn.init.zeros_(self.decoder_projection.bias)
        nn.init.normal_(self.time_tokens, mean=0.0, std=0.02)
        nn.init.normal_(self.self_cond_cfg_tokens, mean=0.0, std=0.02)
        nn.init.normal_(self.mode_tokens, mean=0.0, std=0.02)
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
        segment_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.config.embedding_source == "scratch_t5_encoder":
            return embed_with_scratch_encoder(
                self.scratch_encoder,
                input_ids,
                attention_mask,
                segment_ids,
                config=self.config,
                latent_mean=self.latent_mean,
                latent_std=self.latent_std,
                output_dtype=self.embedding_dtype,
            )
        return self.codebook.lookup(input_ids).to(dtype=self.embedding_dtype)

    def decode_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            decoder_hidden = F.gelu(
                self.decoder_projection(hidden.float()),
                approximate="tanh",
            )
            scores = self.codebook.project(decoder_hidden)
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
        segment_ids: torch.Tensor | None = None,
        self_cond_cfg_scale: torch.Tensor | None = None,
        decoder_step_active: torch.Tensor | bool | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        hidden = self.forward_hidden(
            x,
            t,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            position_ids=position_ids,
            self_cond_cfg_scale=self_cond_cfg_scale,
            decoder_step_active=decoder_step_active,
        )
        with torch.autocast(device_type=x.device.type, enabled=False):
            prediction = (
                self.flow_head(hidden.float())
                if self.flow_head is not None
                else None
            )
            logits = self.decode_hidden(hidden)
        return prediction, logits

    def forward_hidden(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
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
                if self.self_cond_projection is None:
                    raise ValueError(
                        "This objective does not instantiate self-conditioning."
                    )
                x = self.self_cond_projection(x.float())
            elif x.size(-1) == self.config.embedding_size:
                if self.decoder_input_projection is not None:
                    x = self.decoder_input_projection(x.float())
            else:
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
            segment_ids=segment_ids,
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
        if self.flow_head is None:
            raise RuntimeError("denoise() is unavailable for standard_mdlm.")
        prediction, _ = self(
            z_t,
            t,
            attention_mask=attention_mask,
            decoder_step_active=False,
        )
        if prediction is None:
            raise RuntimeError("Flow prediction is unavailable for this objective.")
        return prediction

    def prepare_decoder_input(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.decoder_input_projection is not None:
            return embeddings
        return torch.cat((embeddings, torch.zeros_like(embeddings)), dim=-1)

    def mlm_mask_latent_value(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.mlm_mask_latent is None:
            raise ValueError(
                "This objective does not instantiate an MLM mask latent."
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
        return self.mode_tokens.to(device=device, dtype=dtype).expand(
            batch, -1, -1
        ) * active

    def _init_mlm_mask_latent(self) -> None:
        if self.mlm_mask_latent is None:
            return
        if self.mlm_mask_latent.is_meta:
            return
        if self.token_embedding is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(self.config.mlm_mask_latent_seed))
            latent = torch.randn(
                self.config.embedding_size,
                generator=generator,
                dtype=torch.float32,
            ) * float(self.config.mlm_mask_latent_scale)
        else:
            if self.token_embedding.weight.is_meta:
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

    def _configure_objective_modules(self) -> None:
        objective = self.config.training_objective
        if objective == "official_noisy_ce":
            self.mlm_mask_latent = None
            return
        if objective != "standard_mdlm":
            return

        if self.self_cond_projection is None:
            raise RuntimeError("Canonical self-conditioning projection is missing.")
        rng_state = torch.get_rng_state()
        projection = nn.Linear(
            self.config.embedding_size,
            self.config.embedding_size,
        )
        torch.set_rng_state(rng_state)
        with torch.no_grad():
            projection.weight.copy_(
                self.self_cond_projection.weight[:, : self.config.embedding_size]
            )
            projection.bias.copy_(self.self_cond_projection.bias)
        self.decoder_input_projection = projection
        self.self_cond_projection = None
        self.flow_head = None

    def _configure_embedding_trainability(self) -> None:
        if self.config.embedding_source == "gaussian" and self.token_embedding is not None:
            self.token_embedding.weight.requires_grad_(False)
        set_scratch_encoder_trainability(
            self.scratch_encoder,
            bool(self.config.scratch_encoder_trainable),
        )
