from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput

try:
    from .configuration_babylm_elf import BabyLMELFHFConfig
except ImportError:
    from babylm_elf.export.hf_config import BabyLMELFHFConfig

try:
    from .modeling_core import BabyLMELF, BabyLMELFConfig
except ImportError:
    from babylm_elf.modeling.model import BabyLMELF, BabyLMELFConfig


class BabyLMELFHFModel(PreTrainedModel):
    """ELF encoder interface used by BabyLM finetuning tasks."""

    config_class = BabyLMELFHFConfig
    base_model_prefix = "babylm_elf"
    _tied_weights_keys: list[str] = []
    all_tied_weights_keys: dict[str, list[str]] = {}
    _supports_assign_param_buffer = False

    def __init__(self, config: BabyLMELFHFConfig):
        super().__init__(config)
        model_config = BabyLMELFConfig(**config.babylm_elf_config)
        self.babylm_elf = BabyLMELF(model_config)

    def get_input_embeddings(self):
        return self.babylm_elf.token_embedding

    def set_input_embeddings(self, value):
        self.babylm_elf.token_embedding = value

    def tie_weights(self, *args, **kwargs):
        result = super().tie_weights(*args, **kwargs)
        self.babylm_elf.rope.reset_buffers()
        return result

    def _decoder_hidden(
        self,
        z_t: torch.Tensor,
        attention_mask: torch.Tensor | None,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = z_t.size(0)
        if t is None:
            t = torch.ones(batch_size, device=z_t.device, dtype=z_t.dtype)
        return self.babylm_elf.forward_hidden(
            torch.cat((z_t, torch.zeros_like(z_t)), dim=-1),
            t,
            attention_mask=attention_mask,
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
        return_dict=None,
        **_,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if z_t is None:
            if input_ids is None:
                raise ValueError("Provide either input_ids or z_t.")
            z_t = self.babylm_elf.embed_tokens(input_ids)
        elif t is None:
            raise ValueError("Provide t when passing z_t directly.")

        hidden = self._decoder_hidden(z_t, attention_mask, t)
        if not return_dict:
            return (hidden,)
        return BaseModelOutput(last_hidden_state=hidden)


class BabyLMELFForMaskedLM(BabyLMELFHFModel):
    """Continuous-noise pseudo-likelihood adapter for BabyLM MLM evaluation."""

    def _mask_positions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        mask_token_id = getattr(self.config, "mask_token_id", None)
        if mask_token_id is None:
            raise ValueError("The exported tokenizer/config must define mask_token_id.")
        mask = input_ids.eq(mask_token_id)
        if attention_mask is not None:
            mask = mask & attention_mask.bool()
        return mask

    def _continuous_noised_embeddings(
        self,
        clean_embeddings: torch.Tensor,
        mask_positions: torch.Tensor,
        sample_index: int,
    ) -> torch.Tensor:
        evaluation = self.config.evaluation_config
        seed = int(evaluation.get("seed", 42)) + sample_index
        generator = torch.Generator(device=clean_embeddings.device)
        generator.manual_seed(seed)
        noise = torch.randn(
            clean_embeddings.shape,
            generator=generator,
            device=clean_embeddings.device,
            dtype=clean_embeddings.dtype,
        )
        noise_scale = float(
            self.config.diffusion_config.get("decoder_noise_scale", 5.0)
        )
        return torch.where(
            mask_positions.unsqueeze(-1),
            noise * noise_scale,
            clean_embeddings,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        z_t=None,
        t=None,
        labels=None,
        return_dict=None,
        **_,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is None:
            if z_t is None:
                raise ValueError("Provide input_ids for MLM evaluation or z_t directly.")
            hidden = self._decoder_hidden(z_t, attention_mask, t)
            logits = self._decode_hidden(hidden)
        else:
            clean_embeddings = self.babylm_elf.embed_tokens(input_ids)
            mask_positions = self._mask_positions(input_ids, attention_mask)
            sample_count = max(
                1, int(self.config.evaluation_config.get("mc_samples", 4))
            )
            if mask_positions.any():
                probabilities = []
                for sample_index in range(sample_count):
                    z_sample = self._continuous_noised_embeddings(
                        clean_embeddings,
                        mask_positions,
                        sample_index,
                    )
                    hidden = self._decoder_hidden(z_sample, attention_mask)
                    probabilities.append(
                        self._decode_hidden(hidden).float().softmax(dim=-1)
                    )
                logits = torch.stack(probabilities).mean(dim=0).clamp_min(1.0e-12).log()
            else:
                hidden = self._decoder_hidden(clean_embeddings, attention_mask, t)
                logits = self._decode_hidden(hidden)

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
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            decoder_hidden = F.gelu(
                self.babylm_elf.decoder_projection(hidden.float()),
                approximate="tanh",
            )
            return (
                decoder_hidden @ self.babylm_elf.normalized_embedding_weight().T
                + self.babylm_elf.decoder_bias
            )

    @torch.no_grad()
    def generate(
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
        """ELF diagnostic sampler; BabyLM official scoring uses forward()."""
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
