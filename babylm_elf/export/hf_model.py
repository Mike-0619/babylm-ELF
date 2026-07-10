from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput

# Transformers' local remote-code loader copies only direct relative imports from
# this entrypoint into HF_MODULES_CACHE. Keep this import even though the helper is
# used by modeling_core.py, otherwise fresh-cache AutoModel loading misses the file.
try:
    from .codebook import SphericalCodebook as _hf_cache_codebook_import
    from .mask_latent import (
        build_embedding_stats_mask_latent as _hf_cache_mask_latent_import,
    )
    from .positions import PositionAttention as _hf_cache_positions_import
except ImportError:
    from babylm_elf.modeling.codebook import (
        SphericalCodebook as _hf_cache_codebook_import,
    )
    from babylm_elf.modeling.mask_latent import (
        build_embedding_stats_mask_latent as _hf_cache_mask_latent_import,
    )
    from babylm_elf.modeling.positions import (
        PositionAttention as _hf_cache_positions_import,
    )

try:
    from .layers import RMSNorm as _BabyLMELFRMSNorm
except ImportError:
    _BabyLMELFRMSNorm = None

try:
    from .configuration_babylm_elf import BabyLMELFHFConfig
except ImportError:
    from babylm_elf.export.hf_config import BabyLMELFHFConfig

try:
    from .modeling_core import BabyLMELF, BabyLMELFConfig
except ImportError:
    from babylm_elf.modeling.model import BabyLMELF, BabyLMELFConfig

def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


class BabyLMELFHFModel(PreTrainedModel):
    """ELF encoder interface used by BabyLM finetuning tasks."""

    config_class = BabyLMELFHFConfig
    base_model_prefix = "babylm_elf"
    _tied_weights_keys: list[str] = []
    all_tied_weights_keys: dict[str, list[str]] = {}
    _keys_to_ignore_on_load_missing = [
        r"babylm_elf\.scratch_encoder\.shared\.weight",
    ]
    _supports_assign_param_buffer = False

    def __init__(self, config: BabyLMELFHFConfig):
        super().__init__(config)
        model_config = BabyLMELFConfig(**config.babylm_elf_config)
        self.babylm_elf = BabyLMELF(model_config)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        model.babylm_elf._configure_embedding_trainability()
        return model

    def get_input_embeddings(self):
        return self.babylm_elf.token_embedding

    def set_input_embeddings(self, value):
        if self.babylm_elf.token_embedding is None:
            raise ValueError("Scratch-encoder BabyLM-ELF does not expose input embeddings.")
        self.babylm_elf.codebook.embedding = value

    def _embed_input_ids(self, input_ids, attention_mask=None):
        return self.babylm_elf.embed_tokens(
            input_ids,
            attention_mask=attention_mask,
        )

    def _apply_context_ablation(self, embeddings, input_ids, attention_mask=None):
        mode = os.environ.get("BABYLM_ELF_CONTEXT_ABLATION", "").strip().lower()
        if mode in {"", "none", "off"}:
            return embeddings
        if input_ids is None:
            return embeddings

        mask_token_id = self.babylm_elf.config.mask_token_id
        if attention_mask is None:
            active = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            active = attention_mask.bool()
        if mask_token_id is None:
            mlm_mask = torch.zeros_like(active)
        else:
            mlm_mask = input_ids.eq(mask_token_id) & active
        visible_mask = active & ~mlm_mask

        if mode == "zero_visible":
            ablated = embeddings.clone()
            ablated[visible_mask] = 0.0
            return ablated
        if mode == "shuffle_visible":
            if embeddings.size(0) < 2:
                return embeddings
            ablated = embeddings.clone()
            shuffled = embeddings.roll(shifts=1, dims=0)
            ablated[visible_mask] = shuffled[visible_mask]
            return ablated
        raise ValueError(
            "Unsupported BABYLM_ELF_CONTEXT_ABLATION="
            f"{mode!r}. Expected one of: none, off, zero_visible, "
            "shuffle_visible."
        )

    def tie_weights(self, *args, **kwargs):
        result = super().tie_weights(*args, **kwargs)
        self.babylm_elf.position_attention.reset_buffers()
        self.babylm_elf._configure_embedding_trainability()
        return result

    def _decoder_hidden(
        self,
        z_t: torch.Tensor,
        attention_mask: torch.Tensor | None,
        t: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = z_t.size(0)
        if t is None:
            t = torch.ones(batch_size, device=z_t.device, dtype=z_t.dtype)
        return self.babylm_elf.forward_hidden(
            torch.cat((z_t, torch.zeros_like(z_t)), dim=-1),
            t,
            attention_mask=attention_mask,
            position_ids=position_ids,
            self_cond_cfg_scale=torch.ones(
                batch_size, device=z_t.device, dtype=z_t.dtype
            ),
            decoder_step_active=torch.ones(
                batch_size, device=z_t.device, dtype=z_t.dtype
            ),
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        z_t=None,
        t=None,
        position_ids=None,
        return_dict=None,
        **_,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if z_t is None:
            if input_ids is None:
                raise ValueError("Provide either input_ids or z_t.")
            z_t = self._embed_input_ids(
                input_ids,
                attention_mask=attention_mask,
            )
            z_t = self._apply_context_ablation(z_t, input_ids, attention_mask)
        elif t is None:
            raise ValueError("Provide t when passing z_t directly.")

        hidden = self._decoder_hidden(z_t, attention_mask, t, position_ids)
        if not return_dict:
            return (hidden,)
        return BaseModelOutput(last_hidden_state=hidden)


class BabyLMELFForMaskedLM(BabyLMELFHFModel):
    """Masked-token adapter for BabyLM MLM evaluation."""

    _SUPPORTED_MLM_ADAPTER = "mlm_mask_latent"

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        z_t=None,
        t=None,
        position_ids=None,
        labels=None,
        return_dict=None,
        **_,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is None:
            if z_t is None:
                raise ValueError("Provide input_ids for MLM evaluation or z_t directly.")
            logits = self._direct_logits_from_embeddings(
                z_t, attention_mask, t, position_ids
            )
        else:
            self._validate_mlm_adapter()
            embeddings = self._embed_input_ids(
                input_ids,
                attention_mask=attention_mask,
            )
            embeddings = self._apply_context_ablation(
                embeddings,
                input_ids,
                attention_mask,
            )
            mask_positions = self._mlm_mask_positions(input_ids, attention_mask)
            if mask_positions.any():
                embeddings = self._replace_mask_embeddings_with_latent(
                    embeddings,
                    mask_positions,
                )
            logits = self._direct_logits_from_embeddings(
                embeddings, attention_mask, t, position_ids
            )

        loss = None
        if labels is not None:
            loss = F.nll_loss(
                logits.float().log_softmax(dim=-1).view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            return ((loss, logits) if loss is not None else (logits,))
        return MaskedLMOutput(loss=loss, logits=logits)

    def _decode_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.babylm_elf.decode_hidden(hidden)

    def _direct_logits_from_embeddings(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
        t: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self._decoder_hidden(
            embeddings,
            attention_mask,
            t,
            position_ids,
        )
        return self._decode_hidden(hidden)

    def _validate_mlm_adapter(self) -> None:
        evaluation_config = getattr(self.config, "evaluation_config", {}) or {}
        adapter = str(evaluation_config.get("adapter", self._SUPPORTED_MLM_ADAPTER))
        if adapter != self._SUPPORTED_MLM_ADAPTER:
            raise ValueError(
                "BabyLM-ELF MLM scoring only supports "
                f"{self._SUPPORTED_MLM_ADAPTER!r}; got {adapter!r}. "
                "Re-export this checkpoint with evaluation_config.adapter="
                f"{self._SUPPORTED_MLM_ADAPTER!r}."
            )

    def _mlm_mask_positions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        mask_token_id = self.config.mask_token_id
        if mask_token_id is None:
            return torch.zeros_like(input_ids, dtype=torch.bool)
        if attention_mask is None:
            active = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            active = attention_mask.bool()
        return input_ids.eq(mask_token_id) & active

    def _replace_mask_embeddings_with_latent(
        self,
        embeddings: torch.Tensor,
        mask_positions: torch.Tensor,
    ) -> torch.Tensor:
        latent = self.babylm_elf.mlm_mask_latent_value(
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        return torch.where(
            mask_positions.unsqueeze(-1),
            latent.view(1, 1, -1),
            embeddings,
        )

    @torch.no_grad()
    def diagnostic_generate(
        self,
        batch_size: int = 1,
        sequence_length: int | None = None,
        num_steps: int = 64,
        attention_mask=None,
        sampling_method: str = "sde",
        self_cond_cfg_scale: float = 3.0,
        sde_gamma: float = 1.0,
        time_schedule: str | None = None,
        **_,
    ):
        """Optional ELF-style debug sampler; BabyLM scoring uses forward()."""
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1.")
        if sampling_method not in {"ode", "sde"}:
            raise ValueError("sampling_method must be 'ode' or 'sde'.")
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        sequence_length = sequence_length or self.babylm_elf.config.max_position_embeddings
        diffusion = self.config.diffusion_config
        noise_scale = float(diffusion.get("denoiser_noise_scale", 2.0))
        t_eps = float(diffusion.get("t_eps", 0.05))
        generation = getattr(self.config, "diagnostic_generation_config", {})
        if time_schedule is None:
            time_schedule = generation.get(
                "time_schedule",
                diffusion.get("time_schedule", "logit_normal"),
            )
        if time_schedule not in {"logit_normal", "uniform"}:
            raise ValueError("time_schedule must be 'logit_normal' or 'uniform'.")
        z = torch.randn(
            batch_size,
            sequence_length,
            self.babylm_elf.config.embedding_size,
            device=device,
            dtype=dtype,
        ) * noise_scale
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, sequence_length, device=device, dtype=torch.long
            )
        previous_prediction = torch.zeros_like(z)
        cfg_scale = torch.full(
            (batch_size,),
            float(self_cond_cfg_scale),
            device=device,
            dtype=dtype,
        )
        if time_schedule == "logit_normal":
            interior = torch.sigmoid(
                torch.randn(num_steps - 1, device=device, dtype=dtype)
                * float(diffusion.get("denoiser_p_std", 0.8))
                + float(diffusion.get("denoiser_p_mean", -1.5))
            ).sort().values
            times = torch.cat(
                (
                    torch.zeros(1, device=device, dtype=dtype),
                    interior,
                    torch.ones(1, device=device, dtype=dtype),
                )
            )
        else:
            times = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)
        for index in range(num_steps):
            next_t = times[index + 1]
            current_t = times[index]
            if sampling_method == "sde" and sde_gamma > 0.0:
                step_size = float(next_t - current_t)
                alpha = max(0.0, min(1.0, 1.0 - sde_gamma * step_size))
                model_t = current_t * alpha
                z = alpha * z + (1.0 - alpha) * torch.randn_like(z) * noise_scale
            else:
                model_t = current_t
            t = model_t.expand(batch_size)
            prediction, _ = self.babylm_elf(
                torch.cat((z, previous_prediction), dim=-1),
                t,
                attention_mask=attention_mask,
                self_cond_cfg_scale=cfg_scale,
                decoder_step_active=False,
            )
            velocity = (prediction - z) / (
                1.0 - t.view(-1, 1, 1)
            ).clamp_min(t_eps)
            z = z + (next_t - model_t) * velocity
            previous_prediction = prediction

        final_t = torch.ones(batch_size, device=device, dtype=dtype)
        _, logits = self.babylm_elf(
            torch.cat((z, torch.zeros_like(z)), dim=-1),
            final_t,
            attention_mask=attention_mask,
            self_cond_cfg_scale=cfg_scale,
            decoder_step_active=True,
        )
        return logits.argmax(dim=-1)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        """Backward-compatible diagnostic alias; not used by BabyLM scoring."""
        return self.diagnostic_generate(*args, **kwargs)
