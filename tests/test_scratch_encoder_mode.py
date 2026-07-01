from __future__ import annotations

import unittest

import torch

try:
    import transformers  # noqa: F401
    HAS_TRANSFORMERS = True
except ModuleNotFoundError:
    HAS_TRANSFORMERS = False

try:
    from babylm_elf.modeling.model import BabyLMELF, BabyLMELFConfig
    from babylm_elf.training.optim import create_optimizer
except ModuleNotFoundError as exc:
    if exc.name != "transformers":
        raise
    BabyLMELF = None
    BabyLMELFConfig = None
    create_optimizer = None


class OptimStub:
    optimizer = "adamw"
    learning_rate = 1.0e-3
    weight_decay = 0.0
    beta1 = 0.9
    beta2 = 0.999
    eps = 1.0e-8


@unittest.skipIf(not HAS_TRANSFORMERS or BabyLMELF is None, "transformers is not installed")
class ScratchEncoderModeTest(unittest.TestCase):
    def test_scratch_encoder_is_frozen_and_decodes_to_base_vocab(self) -> None:
        config = BabyLMELFConfig(
            vocab_size=32,
            base_vocab_size=32,
            embedding_size=16,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=16,
            bottleneck_size=8,
            pad_token_id=3,
            mask_token_id=4,
            embedding_source="scratch_t5_encoder",
            encoder_vocab_size=132,
            sentinel_start_id=32,
            sentinel_count=100,
            encoder_d_ff=32,
            encoder_d_kv=4,
            encoder_num_layers=1,
            encoder_num_heads=4,
            encoder_dropout_rate=0.0,
        )
        model = BabyLMELF(config)
        input_ids = torch.tensor([[1, 4, 4, 7, 4, 8, 3]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0]])

        mapped = model._replace_mask_spans_with_sentinels(input_ids, attention_mask)
        self.assertEqual(mapped[0, 1].item(), 32)
        self.assertEqual(mapped[0, 2].item(), 32)
        self.assertEqual(mapped[0, 4].item(), 33)

        embeddings = model.embed_tokens(input_ids, attention_mask=attention_mask)
        self.assertEqual(tuple(embeddings.shape), (1, 7, 16))
        self.assertTrue(torch.isfinite(embeddings).all())

        _, logits = model(
            embeddings,
            torch.ones(1),
            attention_mask=attention_mask,
            decoder_step_active=True,
        )
        self.assertEqual(tuple(logits.shape), (1, 7, 32))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(model.unembed_kernel.requires_grad)
        self.assertTrue(all(not p.requires_grad for p in model.scratch_encoder.parameters()))

        optimizer = create_optimizer(
            model,
            OptimStub(),
        )
        optimizer_params = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        encoder_params = {id(parameter) for parameter in model.scratch_encoder.parameters()}
        self.assertTrue(optimizer_params.isdisjoint(encoder_params))
        self.assertIn(id(model.unembed_kernel), optimizer_params)


if __name__ == "__main__":
    unittest.main()
