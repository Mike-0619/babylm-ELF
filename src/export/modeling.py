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
    from .layers import PositionAttention as _hf_cache_layers_import
    from .encoder_modeling import build_scratch_encoder as _hf_cache_encoder_import
except ImportError:
    from src.modules.encoder import (
        build_scratch_encoder as _hf_cache_encoder_import,
    )
    from src.modules.layers import PositionAttention as _hf_cache_layers_import

try:
    from .layers import RMSNorm as _BabyLMELFRMSNorm
except ImportError:
    _BabyLMELFRMSNorm = None

try:
    from .configuration_babylm_elf import BabyLMELFHFConfig
except ImportError:
    from src.export.configuration import BabyLMELFHFConfig

try:
    from .modeling_core import BabyLMELF, BabyLMELFConfig
except ImportError:
    from src.modules.model import BabyLMELF, BabyLMELFConfig

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

    def _uses_time_conditioning(self) -> bool:
        evaluation = getattr(self.config, "evaluation_config", {}) or {}
        if evaluation.get("adapter") != "mdlm_subs_v1":
            return True
        return bool(evaluation.get("time_conditioning", False))

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
        if not self._uses_time_conditioning():
            t = torch.zeros_like(t)
        return self.babylm_elf.forward_hidden(
            self.babylm_elf.prepare_decoder_input(z_t),
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
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        if z_t is None:
            if input_ids is None:
                raise ValueError("Provide either input_ids or z_t.")
            z_t = self._embed_input_ids(
                input_ids,
                attention_mask=attention_mask,
            )
            z_t = self._apply_context_ablation(z_t, input_ids, attention_mask)
        elif t is None and self._uses_time_conditioning():
            raise ValueError("Provide t when passing z_t directly.")

        hidden = self._decoder_hidden(z_t, attention_mask, t, position_ids)
        if not return_dict:
            return (hidden,)
        return BaseModelOutput(last_hidden_state=hidden)


class BabyLMELFForMaskedLM(BabyLMELFHFModel):
    """Masked-token adapter for BabyLM MLM evaluation."""

    _SUPPORTED_MLM_ADAPTERS = frozenset(
        {"mlm_mask_latent", "fixed_gaussian_v1", "mdlm_subs_v1"}
    )

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
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        if input_ids is None:
            if z_t is None:
                raise ValueError("Provide input_ids for MLM evaluation or z_t directly.")
            logits = self._direct_logits_from_embeddings(
                z_t, attention_mask, t, position_ids
            )
        else:
            adapter = self._validate_mlm_adapter()
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
                embeddings = self._replace_mask_embeddings(
                    embeddings,
                    mask_positions,
                    adapter,
                )
            logits = self._direct_logits_from_embeddings(
                embeddings, attention_mask, t, position_ids
            )
            if adapter == "mdlm_subs_v1":
                logits = self._apply_mdlm_subs(
                    logits,
                    input_ids,
                    mask_positions,
                )

        loss = None
        if labels is not None:
            active_labels = labels.ne(-100)
            if active_labels.any():
                loss = F.nll_loss(
                    logits.float().log_softmax(dim=-1).view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
            else:
                loss = logits.float().sum() * 0.0

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

    def _validate_mlm_adapter(self) -> str:
        evaluation_config = getattr(self.config, "evaluation_config", {}) or {}
        adapter = str(evaluation_config.get("adapter", "mlm_mask_latent"))
        if adapter not in self._SUPPORTED_MLM_ADAPTERS:
            raise ValueError(
                "BabyLM-ELF MLM scoring only supports adapters "
                f"{sorted(self._SUPPORTED_MLM_ADAPTERS)!r}; got {adapter!r}."
            )
        return adapter

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

    def _replace_mask_embeddings(
        self,
        embeddings: torch.Tensor,
        mask_positions: torch.Tensor,
        adapter: str,
    ) -> torch.Tensor:
        if adapter in {"mlm_mask_latent", "mdlm_subs_v1"}:
            latent = self.babylm_elf.mlm_mask_latent_value(
                device=embeddings.device,
                dtype=embeddings.dtype,
            )
            replacement = latent.view(1, 1, -1)
        else:
            evaluation_config = getattr(self.config, "evaluation_config", {}) or {}
            generator = torch.Generator(device="cpu").manual_seed(
                int(evaluation_config.get("fixed_gaussian_seed", 0))
            )
            replacement = torch.randn(
                embeddings.size(1),
                embeddings.size(2),
                generator=generator,
                dtype=torch.float32,
            )
            replacement.mul_(
                float(evaluation_config.get("fixed_gaussian_scale", 5.0))
            )
            replacement = replacement.to(
                device=embeddings.device,
                dtype=embeddings.dtype,
            ).unsqueeze(0)
        return torch.where(
            mask_positions.unsqueeze(-1),
            replacement,
            embeddings,
        )

    def _apply_mdlm_subs(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        mask_positions: torch.Tensor,
    ) -> torch.Tensor:
        evaluation = getattr(self.config, "evaluation_config", {}) or {}
        special_token_count = int(evaluation.get("special_token_count", 16))
        constrained = logits.float().clone()
        minimum = torch.finfo(constrained.dtype).min
        constrained[..., self.config.mask_token_id] = minimum
        if special_token_count > 0:
            constrained[..., :special_token_count] = torch.where(
                mask_positions.unsqueeze(-1),
                torch.full_like(
                    constrained[..., :special_token_count],
                    minimum,
                ),
                constrained[..., :special_token_count],
            )
        carried = torch.full_like(constrained, minimum)
        carried.scatter_(-1, input_ids.unsqueeze(-1), 0.0)
        return torch.where(mask_positions.unsqueeze(-1), constrained, carried)

    @torch.no_grad()
    def generate_mdlm(
        self,
        batch_size: int = 1,
        sequence_length: int | None = None,
        num_steps: int = 128,
        seed: int | None = None,
    ) -> torch.Tensor:
        if self._validate_mlm_adapter() != "mdlm_subs_v1":
            raise ValueError("generate_mdlm requires adapter='mdlm_subs_v1'.")
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1.")
        sequence_length = (
            sequence_length
            or self.babylm_elf.config.max_position_embeddings
        )
        if sequence_length < 2:
            raise ValueError("sequence_length must leave room for BOS and EOS.")

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        mask_token_id = int(self.config.mask_token_id)
        bos_token_id = int(self.config.bos_token_id)
        eos_token_id = int(self.config.eos_token_id)
        evaluation = getattr(self.config, "evaluation_config", {}) or {}
        sampling_eps = float(evaluation.get("mdlm_sampling_eps", 1.0e-3))
        noise_eps = float(evaluation.get("mdlm_noise_eps", 1.0e-3))

        input_ids = torch.full(
            (batch_size, sequence_length),
            mask_token_id,
            device=device,
            dtype=torch.long,
        )
        input_ids[:, 0] = bos_token_id
        input_ids[:, -1] = eos_token_id
        attention_mask = torch.ones_like(input_ids)
        times = torch.linspace(
            1.0,
            sampling_eps,
            num_steps + 1,
            device=device,
            dtype=torch.float32,
        )
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(int(seed))

        for index in range(num_steps):
            masked = input_ids.eq(mask_token_id)
            masked[:, 0] = False
            masked[:, -1] = False
            if not bool(masked.any()):
                break
            t_value = times[index]
            s_value = times[index + 1]
            embeddings = self._embed_input_ids(
                input_ids,
                attention_mask=attention_mask,
            )
            embeddings = self._replace_mask_embeddings(
                embeddings,
                masked,
                "mdlm_subs_v1",
            )
            logits = self._direct_logits_from_embeddings(
                embeddings,
                attention_mask,
                t_value.expand(batch_size).to(dtype=dtype),
            )
            logits = self._apply_mdlm_subs(logits, input_ids, masked)
            token_probabilities = logits[masked].softmax(dim=-1)
            p_t = (1.0 - noise_eps) * t_value
            p_s = (1.0 - noise_eps) * s_value
            reveal_probability = ((p_t - p_s) / p_t.clamp_min(1.0e-12)).clamp(
                0.0,
                1.0,
            )
            probabilities = token_probabilities * reveal_probability
            probabilities[:, mask_token_id] = 1.0 - reveal_probability
            sampled = torch.multinomial(
                probabilities,
                num_samples=1,
                generator=generator,
            ).squeeze(-1)
            input_ids[masked] = sampled

        residual = input_ids.eq(mask_token_id)
        residual[:, 0] = False
        residual[:, -1] = False
        if bool(residual.any()):
            embeddings = self._embed_input_ids(
                input_ids,
                attention_mask=attention_mask,
            )
            embeddings = self._replace_mask_embeddings(
                embeddings,
                residual,
                "mdlm_subs_v1",
            )
            logits = self._direct_logits_from_embeddings(
                embeddings,
                attention_mask,
                torch.full(
                    (batch_size,),
                    sampling_eps,
                    device=device,
                    dtype=dtype,
                ),
            )
            logits = self._apply_mdlm_subs(logits, input_ids, residual)
            input_ids[residual] = logits[residual].argmax(dim=-1)
        input_ids[:, 0] = bos_token_id
        input_ids[:, -1] = eos_token_id
        return input_ids
