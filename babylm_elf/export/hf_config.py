from __future__ import annotations

from transformers import PretrainedConfig


class BabyLMELFHFConfig(PretrainedConfig):
    model_type = "babylm_elf"

    def __init__(self, **kwargs):
        self.babylm_elf_config = kwargs.pop("babylm_elf_config", {})
        self.diffusion_config = kwargs.pop("diffusion_config", {})
        self.evaluation_config = kwargs.pop("evaluation_config", {})
        self.diagnostic_generation_config = kwargs.pop(
            "diagnostic_generation_config",
            {},
        )
        self.training_metadata = kwargs.pop("training_metadata", {})
        super().__init__(**kwargs)
