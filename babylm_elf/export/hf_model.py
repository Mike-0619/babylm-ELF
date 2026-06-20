from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput

try:
    from babylm_elf.export.hf_config import BabyLMELFHFConfig
except ImportError:  # when copied into an exported HF folder
    from configuration_babylm_elf import BabyLMELFHFConfig

from babylm_elf.modeling.model import BabyLMELF, BabyLMELFConfig


class BabyLMELFHFModel(PreTrainedModel):
    config_class = BabyLMELFHFConfig
    base_model_prefix = "babylm_elf"
    _tied_weights_keys: list[str] = []
    all_tied_weights_keys: dict[str, list[str]] = {}

    def __init__(self, config: BabyLMELFHFConfig):
        super().__init__(config)
        model_config = BabyLMELFConfig(**config.babylm_elf_config)
        self.babylm_elf = BabyLMELF(model_config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        z_t=None,
        t=None,
        decode_from=None,
        labels=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if z_t is None:
            if input_ids is None:
                raise ValueError("Provide either input_ids or z_t.")
            z_t = self.babylm_elf.embed_tokens(input_ids)
            if t is None:
                t = torch.zeros(input_ids.size(0), device=input_ids.device, dtype=z_t.dtype)
        elif t is None:
            raise ValueError("Provide t when passing z_t directly.")

        prediction, logits = self.babylm_elf(
            z_t,
            t,
            attention_mask=attention_mask,
            decode_from=decode_from,
        )
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            output = (logits, prediction)
            return ((loss,) + output) if loss is not None else output
        return MaskedLMOutput(loss=loss, logits=logits, hidden_states=(prediction,))
