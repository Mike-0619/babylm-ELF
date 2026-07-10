from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

from babylm_elf.config import DiffusionConfig
from babylm_elf.modeling.mask_latent import build_embedding_stats_mask_latent
from babylm_elf.training.step import (
    apply_mlm_mask_latent,
    build_mlm_decoder_inputs,
    make_target_mask,
    make_one_per_segment_step10_then_step20_mlm_input,
    train_step,
)


class EchoInputTokenModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 32) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            mask_token_id=4,
            pad_token_id=3,
            embedding_size=vocab_size,
            embedding_rms=1.0,
        )
        self.vocab_size = vocab_size
        self.embedding_size = vocab_size
        self.token_embedding = torch.nn.Embedding(
            vocab_size,
            vocab_size,
            padding_idx=self.config.pad_token_id,
        )
        with torch.no_grad():
            self.token_embedding.weight.copy_(torch.eye(vocab_size))
        self.mlm_mask_latent = torch.nn.Parameter(
            build_embedding_stats_mask_latent(
                self.token_embedding.weight,
                embedding_size=self.config.embedding_size,
                embedding_rms=self.config.embedding_rms,
                pad_token_id=self.config.pad_token_id,
                seed=0,
                scale=1.0,
            )
        )
        self.embedded_input_ids: list[torch.Tensor] = []
        self.seen_cfg_scales: list[torch.Tensor] = []
        self.forward_batch_sizes: list[int] = []
        self.forward_calls = 0

    def embed_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_mask
        self.embedded_input_ids.append(input_ids.detach().cpu())
        return F.one_hot(input_ids.clamp(max=self.vocab_size - 1), self.vocab_size).float()

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
        self_cond_cfg_scale: torch.Tensor | None = None,
        decoder_step_active: torch.Tensor | bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del t, attention_mask, segment_ids, decoder_step_active
        if self_cond_cfg_scale is not None:
            self.seen_cfg_scales.append(self_cond_cfg_scale.detach().cpu())
        self.forward_batch_sizes.append(x.size(0))
        self.forward_calls += 1
        current_ids = x[..., : self.embedding_size].argmax(dim=-1)
        logits = x.new_full((*current_ids.shape, self.vocab_size), -5.0)
        logits.scatter_(-1, current_ids.unsqueeze(-1), 5.0)
        prediction = torch.zeros_like(x[..., : self.embedding_size])
        return prediction, logits

    def mlm_mask_latent_value(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.mlm_mask_latent.to(device=device, dtype=dtype)


class MLMDecoderObjectiveTest(unittest.TestCase):
    def test_target_mask_excludes_padding_and_reserved_special_tokens(self) -> None:
        input_ids = torch.tensor([[1, 16, 2, 17, 3]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 0]])

        target_mask = make_target_mask(
            input_ids,
            attention_mask,
            special_token_count=16,
        )

        self.assertTrue(
            torch.equal(
                target_mask,
                torch.tensor([[False, True, False, True, False]]),
            )
        )

    def test_step10_then_step20_masks_one_token_per_short_segment(self) -> None:
        torch.manual_seed(0)
        input_ids = torch.tensor([[1, 16, 17, 1, 18, 19, 1, 20, 21, 1, 22, 23, 2]])
        attention_mask = torch.ones_like(input_ids)

        masked_ids, mlm_mask, row_indices = (
            make_one_per_segment_step10_then_step20_mlm_input(
                input_ids,
                attention_mask,
                mask_token_id=4,
                special_token_count=16,
                segment_boundary_token_id=1,
            )
        )

        self.assertEqual(masked_ids.shape, (1, 13))
        self.assertEqual(row_indices.tolist(), [0])
        self.assertEqual(mlm_mask.sum(dim=1).tolist(), [4])
        self.assertEqual(int(mlm_mask[0, 1:3].sum()), 1)
        self.assertEqual(int(mlm_mask[0, 4:6].sum()), 1)
        self.assertEqual(int(mlm_mask[0, 7:9].sum()), 1)
        self.assertEqual(int(mlm_mask[0, 10:12].sum()), 1)
        self.assertTrue(input_ids[row_indices][mlm_mask].ge(16).all())
        self.assertTrue(masked_ids[mlm_mask].eq(4).all())

    def test_step10_then_step20_masks_every_maskable_segment(self) -> None:
        torch.manual_seed(0)
        input_ids = torch.tensor([[1, 16, 17, 18, 1, 19, 20, 21, 22, 2]])
        attention_mask = torch.ones_like(input_ids)

        masked_ids, mlm_mask, row_indices = (
            make_one_per_segment_step10_then_step20_mlm_input(
                input_ids,
                attention_mask,
                mask_token_id=4,
                special_token_count=16,
                segment_boundary_token_id=1,
            )
        )

        self.assertEqual(masked_ids.shape, (1, 10))
        self.assertEqual(row_indices.tolist(), [0])
        self.assertEqual(mlm_mask.sum(dim=1).tolist(), [2])
        self.assertEqual(int(mlm_mask[0, 1:4].sum()), 1)
        self.assertEqual(int(mlm_mask[0, 5:9].sum()), 1)
        self.assertTrue(input_ids[row_indices][mlm_mask].ge(16).all())
        self.assertTrue(masked_ids[mlm_mask].eq(4).all())

    def test_step10_then_step20_uses_expected_counts(self) -> None:
        cases = (
            (1, 1),
            (10, 1),
            (11, 2),
            (20, 2),
            (21, 3),
            (30, 3),
            (31, 4),
            (40, 4),
            (41, 5),
            (60, 5),
            (61, 6),
            (80, 6),
            (81, 7),
            (100, 7),
        )
        for length, expected_masks in cases:
            with self.subTest(length=length):
                input_ids = torch.cat(
                    [torch.tensor([1]), torch.arange(16, 16 + length), torch.tensor([2])]
                ).unsqueeze(0)
                attention_mask = torch.ones_like(input_ids)

                masked_ids, mlm_mask, row_indices = (
                    make_one_per_segment_step10_then_step20_mlm_input(
                        input_ids,
                        attention_mask,
                        mask_token_id=4,
                        special_token_count=16,
                        mask_schedule="cyclic",
                        mask_seed=0,
                        sequence_ids=torch.tensor([0]),
                        current_epoch=0,
                        segment_boundary_token_id=1,
                    )
                )

                self.assertEqual(row_indices.tolist(), [0])
                self.assertEqual(int(mlm_mask.sum()), expected_masks)
                self.assertTrue(masked_ids[mlm_mask].eq(4).all())

    def test_step10_then_step20_counts_filtered_maskable_tokens_only(self) -> None:
        input_ids = torch.tensor([[1, *range(16, 28), 2]])
        attention_mask = torch.ones_like(input_ids)

        _, mlm_mask, row_indices = make_one_per_segment_step10_then_step20_mlm_input(
            input_ids,
            attention_mask,
            mask_token_id=4,
            special_token_count=16,
            excluded_token_ids=[16, 17],
            mask_schedule="cyclic",
            mask_seed=0,
            sequence_ids=torch.tensor([0]),
            current_epoch=0,
            segment_boundary_token_id=1,
        )

        self.assertEqual(row_indices.tolist(), [0])
        self.assertEqual(int(mlm_mask.sum()), 1)
        self.assertFalse(mlm_mask[0, 1])
        self.assertFalse(mlm_mask[0, 2])

    def test_cyclic_schedule_is_evenly_spaced_in_maskable_order(self) -> None:
        input_ids = torch.tensor([[1, *range(16, 35), 2]])
        attention_mask = torch.ones_like(input_ids)

        _, epoch0_mask, _ = make_one_per_segment_step10_then_step20_mlm_input(
            input_ids,
            attention_mask,
            mask_token_id=4,
            special_token_count=16,
            mask_schedule="cyclic",
            mask_seed=0,
            sequence_ids=torch.tensor([0]),
            current_epoch=0,
            segment_boundary_token_id=1,
        )
        _, epoch1_mask, _ = make_one_per_segment_step10_then_step20_mlm_input(
            input_ids,
            attention_mask,
            mask_token_id=4,
            special_token_count=16,
            mask_schedule="cyclic",
            mask_seed=0,
            sequence_ids=torch.tensor([0]),
            current_epoch=1,
            segment_boundary_token_id=1,
        )

        self.assertTrue(epoch0_mask[0, 1])
        self.assertTrue(epoch0_mask[0, 10])
        self.assertEqual(int(epoch0_mask.sum()), 2)
        self.assertTrue(epoch1_mask[0, 2])
        self.assertTrue(epoch1_mask[0, 11])
        self.assertEqual(int(epoch1_mask.sum()), 2)

    def test_cyclic_mask_schedule_is_deterministic_by_sequence_epoch_and_seed(
        self,
    ) -> None:
        input_ids = torch.tensor([[1, *range(16, 32), 2]])
        attention_mask = torch.ones_like(input_ids)
        sequence_ids = torch.tensor([123])

        torch.manual_seed(0)
        first_ids, first_mask, _ = make_one_per_segment_step10_then_step20_mlm_input(
            input_ids,
            attention_mask,
            mask_token_id=4,
            special_token_count=16,
            sequence_ids=sequence_ids,
            current_epoch=2,
            mask_schedule="cyclic",
            mask_seed=7,
            segment_boundary_token_id=1,
        )
        torch.manual_seed(999)
        second_ids, second_mask, _ = make_one_per_segment_step10_then_step20_mlm_input(
            input_ids,
            attention_mask,
            mask_token_id=4,
            special_token_count=16,
            sequence_ids=sequence_ids,
            current_epoch=2,
            mask_schedule="cyclic",
            mask_seed=7,
            segment_boundary_token_id=1,
        )

        self.assertTrue(torch.equal(first_ids, second_ids))
        self.assertTrue(torch.equal(first_mask, second_mask))

    def test_cyclic_mask_schedule_advances_between_epochs(self) -> None:
        input_ids = torch.tensor([[1, *range(16, 32), 2]])
        attention_mask = torch.ones_like(input_ids)
        sequence_ids = torch.tensor([123])

        _, epoch0_mask, _ = make_one_per_segment_step10_then_step20_mlm_input(
            input_ids,
            attention_mask,
            mask_token_id=4,
            special_token_count=16,
            sequence_ids=sequence_ids,
            current_epoch=0,
            mask_schedule="cyclic",
            mask_seed=7,
            segment_boundary_token_id=1,
        )
        _, epoch1_mask, _ = make_one_per_segment_step10_then_step20_mlm_input(
            input_ids,
            attention_mask,
            mask_token_id=4,
            special_token_count=16,
            sequence_ids=sequence_ids,
            current_epoch=1,
            mask_schedule="cyclic",
            mask_seed=7,
            segment_boundary_token_id=1,
        )

        self.assertEqual(int(epoch0_mask.sum()), 2)
        self.assertEqual(int(epoch1_mask.sum()), 2)
        self.assertEqual(int((epoch0_mask & epoch1_mask).sum()), 0)

    def test_cyclic_mask_schedule_is_stable_under_batch_reordering(self) -> None:
        rows = torch.tensor(
            [
                [1, 16, 17, 18, 19, 2],
                [1, 20, 21, 22, 23, 2],
            ]
        )
        attention_mask = torch.ones_like(rows)

        _, masks, _ = make_one_per_segment_step10_then_step20_mlm_input(
            rows,
            attention_mask,
            mask_token_id=4,
            special_token_count=16,
            sequence_ids=torch.tensor([10, 20]),
            current_epoch=1,
            mask_schedule="cyclic",
            mask_seed=0,
            segment_boundary_token_id=1,
        )
        _, reordered_masks, _ = make_one_per_segment_step10_then_step20_mlm_input(
            rows.flip(0),
            attention_mask.flip(0),
            mask_token_id=4,
            special_token_count=16,
            sequence_ids=torch.tensor([20, 10]),
            current_epoch=1,
            mask_schedule="cyclic",
            mask_seed=0,
            segment_boundary_token_id=1,
        )

        self.assertTrue(torch.equal(masks[0], reordered_masks[1]))
        self.assertTrue(torch.equal(masks[1], reordered_masks[0]))

    def test_random_mask_schedule_preserves_rng_driven_selection(self) -> None:
        input_ids = torch.tensor([[1, *range(16, 32), 2]])
        attention_mask = torch.ones_like(input_ids)

        torch.manual_seed(0)
        _, first_mask, _ = make_one_per_segment_step10_then_step20_mlm_input(
            input_ids,
            attention_mask,
            mask_token_id=4,
            special_token_count=16,
            mask_schedule="random",
            segment_boundary_token_id=1,
        )
        torch.manual_seed(0)
        _, second_mask, _ = make_one_per_segment_step10_then_step20_mlm_input(
            input_ids,
            attention_mask,
            mask_token_id=4,
            special_token_count=16,
            mask_schedule="random",
            segment_boundary_token_id=1,
        )

        self.assertTrue(torch.equal(first_mask, second_mask))

    def test_step10_then_step20_masks_all_available_tokens_in_tiny_segment(self) -> None:
        torch.manual_seed(0)
        input_ids = torch.tensor([[1, 16, 1, 17, 18, 2]])
        attention_mask = torch.ones_like(input_ids)

        masked_ids, mlm_mask, row_indices = (
            make_one_per_segment_step10_then_step20_mlm_input(
                input_ids,
                attention_mask,
                mask_token_id=4,
                special_token_count=16,
                segment_boundary_token_id=1,
            )
        )

        self.assertEqual(row_indices.tolist(), [0])
        self.assertEqual(mlm_mask.sum(dim=1).tolist(), [2])
        self.assertEqual(int(mlm_mask[0, 1:2].sum()), 1)
        self.assertEqual(int(mlm_mask[0, 3:5].sum()), 1)
        self.assertTrue(masked_ids[mlm_mask].eq(4).all())

    def test_step10_then_step20_skips_segments_without_maskable_tokens(self) -> None:
        torch.manual_seed(0)
        input_ids = torch.tensor([[1, 2, 1, 16, 17, 1, 2, 3]])
        attention_mask = torch.ones_like(input_ids)

        masked_ids, mlm_mask, row_indices = (
            make_one_per_segment_step10_then_step20_mlm_input(
                input_ids,
                attention_mask,
                mask_token_id=4,
                special_token_count=16,
                segment_boundary_token_id=1,
            )
        )

        self.assertEqual(masked_ids.shape, (1, 8))
        self.assertEqual(row_indices.tolist(), [0])
        self.assertEqual(mlm_mask.sum(dim=1).tolist(), [1])
        self.assertEqual(int(mlm_mask[0, 3:5].sum()), 1)
        self.assertTrue(masked_ids[mlm_mask].eq(4).all())

    def test_step10_then_step20_skips_sequences_without_maskable_tokens(self) -> None:
        input_ids = torch.tensor([[1, 2, 3, 3], [1, 4, 15, 3]])
        attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])

        masked_ids, mlm_mask, row_indices = (
            make_one_per_segment_step10_then_step20_mlm_input(
                input_ids,
                attention_mask,
                mask_token_id=4,
                special_token_count=16,
                segment_boundary_token_id=1,
            )
        )

        self.assertEqual(masked_ids.shape, (0, 4))
        self.assertEqual(mlm_mask.shape, (0, 4))
        self.assertEqual(row_indices.shape, (0,))

    def test_step10_then_step20_excludes_configured_token_ids(self) -> None:
        torch.manual_seed(0)
        input_ids = torch.tensor([[1, 16, 17, 18, 1, 19, 2]])
        attention_mask = torch.ones_like(input_ids)

        masked_ids, mlm_mask, row_indices = (
            make_one_per_segment_step10_then_step20_mlm_input(
                input_ids,
                attention_mask,
                mask_token_id=4,
                special_token_count=16,
                excluded_token_ids=[16, 17, 19],
                segment_boundary_token_id=1,
            )
        )

        self.assertEqual(row_indices.tolist(), [0])
        self.assertEqual(mlm_mask.sum(dim=1).tolist(), [1])
        self.assertTrue(mlm_mask[0, 3])
        self.assertFalse(mlm_mask[0, 1])
        self.assertFalse(mlm_mask[0, 2])
        self.assertFalse(mlm_mask[0, 5])
        self.assertTrue(masked_ids[mlm_mask].eq(4).all())

    def test_step10_then_step20_keeps_digits_when_not_excluded(self) -> None:
        torch.manual_seed(0)
        input_ids = torch.tensor([[1, 31, 2]])
        attention_mask = torch.ones_like(input_ids)

        masked_ids, mlm_mask, row_indices = (
            make_one_per_segment_step10_then_step20_mlm_input(
                input_ids,
                attention_mask,
                mask_token_id=4,
                special_token_count=16,
                excluded_token_ids=[],
                segment_boundary_token_id=1,
            )
        )

        self.assertEqual(row_indices.tolist(), [0])
        self.assertEqual(mlm_mask.sum(dim=1).tolist(), [1])
        self.assertTrue(mlm_mask[0, 1])
        self.assertTrue(masked_ids[mlm_mask].eq(4).all())

    def test_decoder_inputs_use_learned_mask_latent_not_mask_or_target_embeddings(self) -> None:
        torch.manual_seed(0)
        model = EchoInputTokenModel(vocab_size=32)
        input_ids = torch.tensor([[1, 16, 17, 1, 18, 19, 2]])
        attention_mask = torch.ones_like(input_ids)
        config = DiffusionConfig(
            mlm_special_token_count=16,
            mlm_mask_latent_seed=0,
            mlm_mask_latent_scale=1.0,
        )

        decoder_z, mlm_mask, target_ids, decoder_attention_mask, row_indices = (
            build_mlm_decoder_inputs(
                model,
                input_ids,
                attention_mask,
                config,
            )
        )

        clean = F.one_hot(target_ids, model.vocab_size).float()
        mask_embedding = F.one_hot(
            torch.full_like(target_ids, model.config.mask_token_id),
            model.vocab_size,
        ).float()
        latent = model.mlm_mask_latent_value(
            device=decoder_z.device,
            dtype=decoder_z.dtype,
        )

        self.assertEqual(row_indices.tolist(), [0])
        self.assertTrue(torch.equal(target_ids, input_ids))
        self.assertTrue(torch.equal(decoder_attention_mask, attention_mask))
        self.assertEqual(len(model.embedded_input_ids), 1)
        self.assertTrue(torch.equal(model.embedded_input_ids[0], input_ids))
        self.assertTrue(
            torch.allclose(decoder_z[~mlm_mask.bool()], clean[~mlm_mask.bool()])
        )
        self.assertTrue(
            torch.allclose(
                decoder_z[mlm_mask.bool()],
                latent.expand_as(decoder_z[mlm_mask.bool()]),
            )
        )
        self.assertFalse(
            torch.allclose(
                decoder_z[mlm_mask.bool()],
                mask_embedding[mlm_mask.bool()],
            )
        )
        self.assertFalse(
            torch.allclose(
                decoder_z[mlm_mask.bool()],
                clean[mlm_mask.bool()],
            )
        )
        if int(mlm_mask.sum().item()) > 1:
            first_masked = decoder_z[mlm_mask.bool()][0]
            self.assertTrue(
                torch.allclose(
                    decoder_z[mlm_mask.bool()],
                    first_masked.expand_as(decoder_z[mlm_mask.bool()]),
                )
            )

    def test_apply_mlm_mask_latent_changes_only_mask_positions(self) -> None:
        clean = torch.zeros(2, 4, 32)
        clean[..., 0] = 1.0
        mlm_mask = torch.tensor(
            [
                [False, True, False, False],
                [False, False, True, False],
            ]
        )
        model = EchoInputTokenModel(vocab_size=32)
        with_latent = apply_mlm_mask_latent(clean, mlm_mask, model)
        latent = model.mlm_mask_latent_value(
            device=clean.device,
            dtype=clean.dtype,
        )

        self.assertTrue(torch.allclose(with_latent[~mlm_mask], clean[~mlm_mask]))
        self.assertTrue(
            torch.allclose(
                with_latent[mlm_mask],
                latent.expand_as(with_latent[mlm_mask]),
            )
        )
        self.assertFalse(torch.allclose(with_latent[mlm_mask], clean[mlm_mask]))

    def test_mask_latent_changes_with_seed_and_scale(self) -> None:
        model = EchoInputTokenModel(vocab_size=32)

        seed_zero = build_embedding_stats_mask_latent(
            model.token_embedding.weight,
            embedding_size=model.config.embedding_size,
            embedding_rms=model.config.embedding_rms,
            pad_token_id=model.config.pad_token_id,
            seed=0,
            scale=1.0,
        )
        seed_one = build_embedding_stats_mask_latent(
            model.token_embedding.weight,
            embedding_size=model.config.embedding_size,
            embedding_rms=model.config.embedding_rms,
            pad_token_id=model.config.pad_token_id,
            seed=1,
            scale=1.0,
        )
        scale_two = build_embedding_stats_mask_latent(
            model.token_embedding.weight,
            embedding_size=model.config.embedding_size,
            embedding_rms=model.config.embedding_rms,
            pad_token_id=model.config.pad_token_id,
            seed=0,
            scale=2.0,
        )

        self.assertFalse(torch.allclose(seed_zero, seed_one))
        self.assertFalse(torch.allclose(seed_zero, scale_two))

    def test_train_step_mlm_decoder_scores_only_masked_positions(self) -> None:
        torch.manual_seed(0)
        model = EchoInputTokenModel()
        batch = {
            "input_ids": torch.tensor([[1, 16, 17, 18, 2], [1, 19, 20, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]),
        }
        config = DiffusionConfig(
            decoder_objective="token_mlm",
            decoder_probability=1.0,
            mlm_special_token_count=16,
            self_condition_probability=0.0,
        )

        output = train_step(model, batch, config)

        self.assertTrue(torch.isfinite(output.loss))
        self.assertEqual(output.metrics["decode_frac"], 1.0)
        self.assertGreaterEqual(output.metrics["acc"], 0.0)
        self.assertLessEqual(output.metrics["acc"], 1.0)
        self.assertEqual(model.forward_calls, 1)
        self.assertEqual(len(model.embedded_input_ids), 1)
        embedded_input_ids = model.embedded_input_ids[0]
        self.assertEqual(embedded_input_ids.shape, (2, 5))
        self.assertTrue(torch.equal(embedded_input_ids, batch["input_ids"]))
        self.assertEqual(embedded_input_ids.eq(4).sum().item(), 0)

    def test_decoder_ineligible_rows_fall_back_to_flow_instead_of_disappearing(
        self,
    ) -> None:
        torch.manual_seed(0)
        model = EchoInputTokenModel()
        batch = {
            "input_ids": torch.tensor([[1, 16, 17, 2], [1, 2, 3, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
        }
        config = DiffusionConfig(
            decoder_objective="token_mlm",
            decoder_probability=1.0,
            mlm_special_token_count=16,
            self_condition_probability=0.0,
        )

        output = train_step(model, batch, config)

        self.assertTrue(torch.isfinite(output.loss))
        self.assertEqual(output.metrics["decode_frac"], 0.5)
        self.assertGreater(output.metrics["ce"], 0.0)
        self.assertEqual(output.metrics["flow"], 0.0)
        self.assertEqual(model.forward_batch_sizes[-1], 2)
        self.assertEqual(len(model.embedded_input_ids), 2)
        self.assertTrue(
            torch.equal(model.embedded_input_ids[1], batch["input_ids"][:1])
        )

    def test_unsupported_decoder_objective_is_rejected(self) -> None:
        model = EchoInputTokenModel()
        batch = {
            "input_ids": torch.tensor([[1, 16, 17, 18, 2]]),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
        }
        for objective in ("continuous", "mlm"):
            with self.subTest(objective=objective):
                config = DiffusionConfig(
                    decoder_objective=objective,
                    decoder_probability=1.0,
                    self_condition_probability=0.0,
                )
                with self.assertRaisesRegex(ValueError, "token_mlm"):
                    train_step(model, batch, config)

    def test_train_step_honors_one_per_segment_step10_then_step20_mask_strategy(self) -> None:
        torch.manual_seed(0)
        model = EchoInputTokenModel()
        batch = {
            "input_ids": torch.tensor([[1, 16, 17, 1, 18, 19, 2]]),
            "attention_mask": torch.ones(1, 7, dtype=torch.long),
        }
        config = DiffusionConfig(
            decoder_objective="token_mlm",
            decoder_probability=1.0,
            mlm_mask_strategy="one_per_segment_step10_then_step20",
            mlm_special_token_count=16,
            mlm_segment_boundary_token_id=1,
            self_condition_probability=0.0,
        )

        output = train_step(model, batch, config)

        self.assertTrue(torch.isfinite(output.loss))
        self.assertEqual(model.forward_calls, 1)
        embedded_input_ids = model.embedded_input_ids[0]
        self.assertEqual(embedded_input_ids.shape, (1, 7))
        self.assertTrue(torch.equal(embedded_input_ids, batch["input_ids"]))
        self.assertEqual(embedded_input_ids.eq(4).sum().item(), 0)

    def test_train_step_keeps_selected_ce_rows_once(self) -> None:
        torch.manual_seed(0)
        model = EchoInputTokenModel(vocab_size=256)
        input_ids = torch.tensor(
            [[1, *range(16 + row * 4, 20 + row * 4), 2] for row in range(32)]
        )
        batch = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }
        config = DiffusionConfig(
            decoder_objective="token_mlm",
            decoder_probability=0.5,
            mlm_special_token_count=16,
            self_condition_probability=0.0,
        )

        output = train_step(model, batch, config)

        active_count = round(output.metrics["decode_frac"] * input_ids.size(0))
        self.assertTrue(torch.isfinite(output.loss))
        self.assertGreater(output.metrics["flow"], 0.0)
        self.assertGreater(output.metrics["ce"], 0.0)
        self.assertGreater(active_count, 0)
        self.assertLess(active_count, input_ids.size(0))
        self.assertEqual(len(model.embedded_input_ids), 2)
        embedded_input_ids = model.embedded_input_ids[1]
        self.assertEqual(embedded_input_ids.shape[0], active_count)
        self.assertEqual(embedded_input_ids.eq(4).sum().item(), 0)

    def test_train_step_token_mlm_ce20_mixes_decoder_and_denoiser_losses(self) -> None:
        torch.manual_seed(0)
        model = EchoInputTokenModel(vocab_size=256)
        input_ids = torch.tensor(
            [[1, *range(16 + row * 4, 20 + row * 4), 2] for row in range(32)]
        )
        batch = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }
        config = DiffusionConfig(
            decoder_objective="token_mlm",
            decoder_probability=0.2,
            mlm_special_token_count=16,
            self_condition_probability=0.0,
        )

        output = train_step(model, batch, config)

        self.assertTrue(torch.isfinite(output.loss))
        self.assertGreater(output.metrics["flow"], 0.0)
        self.assertGreater(output.metrics["ce"], 0.0)
        self.assertGreater(output.metrics["decode_frac"], 0.0)
        self.assertLess(output.metrics["decode_frac"], 0.5)
        self.assertEqual(model.forward_calls, 1)

    def test_self_conditioning_auxiliary_forwards_only_use_active_flow_rows(
        self,
    ) -> None:
        torch.manual_seed(0)
        model = EchoInputTokenModel(vocab_size=256)
        input_ids = torch.tensor(
            [[1, *range(16 + row * 4, 20 + row * 4), 2] for row in range(32)]
        )
        batch = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }
        config = DiffusionConfig(
            decoder_objective="token_mlm",
            decoder_probability=0.5,
            mlm_special_token_count=16,
            self_condition_probability=1.0,
        )

        output = train_step(model, batch, config)

        decoder_count = round(output.metrics["decode_frac"] * input_ids.size(0))
        flow_count = input_ids.size(0) - decoder_count
        self.assertGreater(decoder_count, 0)
        self.assertGreater(flow_count, 0)
        self.assertEqual(model.forward_batch_sizes, [flow_count, flow_count, 32])

if __name__ == "__main__":
    unittest.main()
