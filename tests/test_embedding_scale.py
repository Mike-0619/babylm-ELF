from __future__ import annotations

import unittest

import torch

from babylm_elf.modeling.model import BabyLMELF, BabyLMELFConfig


class EmbeddingScaleTest(unittest.TestCase):
    def test_embedded_tokens_have_configured_rms(self) -> None:
        config = BabyLMELFConfig(
            vocab_size=32,
            embedding_size=16,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=16,
            bottleneck_size=8,
            embedding_rms=1.0,
        )
        model = BabyLMELF(config)
        embeddings = model.embed_tokens(torch.tensor([[1, 2, 4]]))
        rms = embeddings.square().mean(dim=-1).sqrt()

        torch.testing.assert_close(rms, torch.ones_like(rms))


if __name__ == "__main__":
    unittest.main()
