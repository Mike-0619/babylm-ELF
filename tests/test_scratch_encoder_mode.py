from __future__ import annotations

import unittest
import importlib.util

import torch

try:
    import transformers  # noqa: F401
    HAS_TRANSFORMERS = True
except ModuleNotFoundError:
    HAS_TRANSFORMERS = False

try:
    from babylm_elf.config import TrainConfig
    from babylm_elf.modeling.codebook import SphericalCodebook, UntiedCodebook
    from babylm_elf.modeling.model import BabyLMELF, BabyLMELFConfig
    from babylm_elf.training.trainer import (
        _scratch_encoder_should_train,
        _set_scratch_encoder_trainability,
        _validate_embedding_source,
    )
    from babylm_elf.training.optim import create_optimizer
except ModuleNotFoundError as exc:
    if exc.name != "transformers":
        raise
    TrainConfig = None
    BabyLMELF = None
    BabyLMELFConfig = None
    _scratch_encoder_should_train = None
    _set_scratch_encoder_trainability = None
    _validate_embedding_source = None
    create_optimizer = None


class OptimStub:
    optimizer = "adamw"
    learning_rate = 1.0e-3
    encoder_lr_multiplier = 1.0
    aux_learning_rate = None
    weight_decay = 0.0
    beta1 = 0.9
    beta2 = 0.999
    eps = 1.0e-8


class MuonOptimStub:
    optimizer = "muon"
    learning_rate = 3.5e-4
    aux_learning_rate = 1.0e-4
    encoder_lr_multiplier = 1.0
    weight_decay = 0.0
    beta1 = 0.9
    beta2 = 0.999
    eps = 1.0e-8


@unittest.skipIf(not HAS_TRANSFORMERS or BabyLMELF is None, "transformers is not installed")
class ScratchEncoderModeTest(unittest.TestCase):
    @unittest.skipIf(importlib.util.find_spec("muon") is None, "muon is not installed")
    def test_muon_and_aux_adam_can_use_separate_learning_rates(self) -> None:
        config = BabyLMELFConfig(
            vocab_size=32,
            base_vocab_size=32,
            embedding_size=16,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            max_position_embeddings=16,
            bottleneck_size=8,
            pad_token_id=3,
            mask_token_id=4,
            embedding_source="learnable",
        )
        model = BabyLMELF(config)
        optimizer = create_optimizer(model, MuonOptimStub())
        group_lrs = {
            group["use_muon"]: group["lr"]
            for group in optimizer.param_groups
        }
        self.assertEqual(group_lrs[True], MuonOptimStub.learning_rate)
        self.assertEqual(group_lrs[False], MuonOptimStub.aux_learning_rate)

    def test_learnable_embedding_is_trainable(self) -> None:
        config = BabyLMELFConfig(
            vocab_size=32,
            base_vocab_size=32,
            embedding_size=16,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            max_position_embeddings=16,
            bottleneck_size=8,
            pad_token_id=3,
            mask_token_id=4,
            embedding_source="learnable",
        )
        model = BabyLMELF(config)
        self.assertIsNotNone(model.token_embedding)
        self.assertIsInstance(model.codebook, SphericalCodebook)
        self.assertTrue(model.token_embedding.weight.requires_grad)

        optimizer = create_optimizer(model, OptimStub())
        optimizer_params = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertIn(id(model.token_embedding.weight), optimizer_params)

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
        self.assertIsInstance(model.codebook, UntiedCodebook)
        self.assertTrue(model.codebook.weight.requires_grad)
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
        self.assertIn(id(model.codebook.weight), optimizer_params)

    def test_scratch_encoder_can_be_joint_trained(self) -> None:
        config = BabyLMELFConfig(
            vocab_size=32,
            base_vocab_size=32,
            embedding_size=16,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            max_position_embeddings=16,
            bottleneck_size=8,
            pad_token_id=3,
            mask_token_id=4,
            embedding_source="scratch_t5_encoder",
            scratch_encoder_trainable=True,
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
        input_ids = torch.tensor([[1, 4, 7, 8]])
        attention_mask = torch.ones_like(input_ids)
        embeddings = model.embed_tokens(input_ids, attention_mask=attention_mask)
        self.assertTrue(embeddings.requires_grad)
        embeddings.sum().backward()
        encoder_grads = [
            parameter.grad
            for parameter in model.scratch_encoder.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(encoder_grads)
        self.assertTrue(any(grad is not None for grad in encoder_grads))

        optimizer = create_optimizer(model, OptimStub())
        optimizer_params = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        encoder_params = {id(parameter) for parameter in model.scratch_encoder.parameters()}
        self.assertFalse(optimizer_params.isdisjoint(encoder_params))

    def test_trainable_scratch_encoder_can_start_without_checkpoint_or_stats(self) -> None:
        config = TrainConfig()
        config.model.embedding_source = "scratch_t5_encoder"
        config.model.scratch_encoder_trainable = True
        config.model.encoder_checkpoint_path = None
        config.model.latent_stats_path = None
        _validate_embedding_source(config)

        config.model.scratch_encoder_trainable = False
        with self.assertRaises(ValueError):
            _validate_embedding_source(config)

    def test_scratch_encoder_freeze_ratio_switches_trainability(self) -> None:
        train_config = TrainConfig()
        train_config.model.embedding_source = "scratch_t5_encoder"
        train_config.model.scratch_encoder_trainable = True
        train_config.encoder_freeze_steps_ratio = 0.15
        self.assertFalse(_scratch_encoder_should_train(train_config, True, 100, 14))
        self.assertTrue(_scratch_encoder_should_train(train_config, True, 100, 15))

        model_config = BabyLMELFConfig(
            vocab_size=32,
            base_vocab_size=32,
            embedding_size=16,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            max_position_embeddings=16,
            bottleneck_size=8,
            pad_token_id=3,
            mask_token_id=4,
            embedding_source="scratch_t5_encoder",
            scratch_encoder_trainable=True,
            encoder_vocab_size=132,
            sentinel_start_id=32,
            sentinel_count=100,
            encoder_d_ff=32,
            encoder_d_kv=4,
            encoder_num_layers=1,
            encoder_num_heads=4,
            encoder_dropout_rate=0.0,
        )
        model = BabyLMELF(model_config)

        _set_scratch_encoder_trainability(model, False)
        self.assertFalse(model.config.scratch_encoder_trainable)
        self.assertTrue(all(not p.requires_grad for p in model.scratch_encoder.parameters()))

        _set_scratch_encoder_trainability(model, True)
        self.assertTrue(model.config.scratch_encoder_trainable)
        self.assertTrue(all(p.requires_grad for p in model.scratch_encoder.parameters()))

    def test_joint_encoder_optimizer_uses_scaled_encoder_learning_rate(self) -> None:
        class EncoderOptimStub(OptimStub):
            encoder_lr_multiplier = 0.1

        config = BabyLMELFConfig(
            vocab_size=32,
            base_vocab_size=32,
            embedding_size=16,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            max_position_embeddings=16,
            bottleneck_size=8,
            pad_token_id=3,
            mask_token_id=4,
            embedding_source="scratch_t5_encoder",
            scratch_encoder_trainable=True,
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
        optimizer = create_optimizer(model, EncoderOptimStub())
        encoder_params = {id(parameter) for parameter in model.scratch_encoder.parameters()}
        encoder_lrs = {
            group["lr"]
            for group in optimizer.param_groups
            if any(id(parameter) in encoder_params for parameter in group["params"])
        }
        self.assertEqual(encoder_lrs, {EncoderOptimStub.learning_rate * 0.1})

    def test_gaussian_embedding_is_frozen_with_trainable_unembedding(self) -> None:
        config = BabyLMELFConfig(
            vocab_size=32,
            base_vocab_size=32,
            embedding_size=16,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            max_position_embeddings=16,
            bottleneck_size=8,
            pad_token_id=3,
            mask_token_id=4,
            embedding_source="gaussian",
            gaussian_embedding_std=1.0,
        )
        model = BabyLMELF(config)
        self.assertIsNotNone(model.token_embedding)
        self.assertIsInstance(model.codebook, UntiedCodebook)
        self.assertFalse(model.token_embedding.weight.requires_grad)
        self.assertTrue(model.codebook.weight.requires_grad)

        token_rows = model.token_embedding.weight.detach()[torch.tensor([0, 1, 2, 4, 5, 6, 7, 8])]
        torch.testing.assert_close(
            token_rows.mean(dim=-1),
            torch.zeros(8),
            atol=1.0e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            token_rows.std(dim=-1, unbiased=False),
            torch.ones(8),
            atol=1.0e-5,
            rtol=0.0,
        )

        input_ids = torch.tensor([[1, 4, 7, 3]])
        attention_mask = torch.tensor([[1, 1, 1, 0]])
        embeddings = model.embed_tokens(input_ids, attention_mask=attention_mask)
        _, logits = model(
            embeddings,
            torch.ones(1),
            attention_mask=attention_mask,
            decoder_step_active=True,
        )
        self.assertEqual(tuple(logits.shape), (1, 4, 32))
        self.assertTrue(torch.isfinite(logits).all())

        optimizer = create_optimizer(model, OptimStub())
        optimizer_params = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertNotIn(id(model.token_embedding.weight), optimizer_params)
        self.assertIn(id(model.codebook.weight), optimizer_params)


if __name__ == "__main__":
    unittest.main()
