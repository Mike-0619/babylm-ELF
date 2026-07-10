from __future__ import annotations

import math
import unittest

import torch
import torch.nn.functional as F

from babylm_elf.modeling.model import BabyLMELF, BabyLMELFConfig


def _small_config(decoder_head_type: str = "gelu") -> BabyLMELFConfig:
    return BabyLMELFConfig(
        vocab_size=32,
        base_vocab_size=32,
        embedding_size=16,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=16,
        bottleneck_size=8,
        decoder_head_type=decoder_head_type,
    )


class DecoderHeadTest(unittest.TestCase):
    def test_pad_row_is_an_ordinary_nonzero_codebook_entry(self) -> None:
        model = BabyLMELF(_small_config())
        embedding = model.token_embedding
        self.assertIsNotNone(embedding)
        self.assertIsNone(embedding.padding_idx)
        self.assertGreater(
            float(embedding.weight[model.config.pad_token_id].detach().norm()),
            0.0,
        )

    def test_zero_pad_codebook_row_cannot_dominate_tied_head_gradient(self) -> None:
        model = BabyLMELF(_small_config("bert_mlm"))
        pad_id = model.config.pad_token_id
        self.assertGreater(
            float(model.token_embedding.weight[pad_id].detach().norm()),
            0.0,
        )
        with torch.no_grad():
            model.token_embedding.weight[pad_id].zero_()

        hidden = torch.randn(2, 3, model.config.hidden_size)
        labels = torch.tensor([[5, 6, 7], [8, 9, 10]])
        loss = F.cross_entropy(
            model.decode_hidden(hidden).reshape(-1, model.config.vocab_size),
            labels.reshape(-1),
        )
        loss.backward()

        gradients = model.token_embedding.weight.grad
        self.assertTrue(torch.isfinite(gradients).all())
        self.assertLess(
            float(gradients[pad_id].norm() / gradients.norm()),
            0.25,
        )
        self.assertGreater(float(gradients[5:].norm()), 0.0)

    def test_near_zero_codebook_row_has_no_inverse_epsilon_gradient(self) -> None:
        model = BabyLMELF(_small_config("bert_mlm"))
        pad_id = model.config.pad_token_id
        with torch.no_grad():
            model.token_embedding.weight[pad_id].fill_(1.0e-12)
        loss = model.decode_hidden(torch.randn(2, 3, model.config.hidden_size)).square().mean()
        loss.backward()
        gradient = model.token_embedding.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertLess(float(gradient.norm()), 1.0e3)

    def test_left_padding_does_not_change_valid_hidden_states(self) -> None:
        model = BabyLMELF(_small_config())
        model.eval()
        unpadded_ids = torch.tensor([[1, 7, 4, 2]])
        padded_ids = torch.tensor([[3, 3, 1, 7, 4, 2]])
        unpadded_mask = torch.ones_like(unpadded_ids)
        padded_mask = padded_ids.ne(model.config.pad_token_id).long()

        with torch.no_grad():
            unpadded = model.forward_hidden(
                model.embed_tokens(unpadded_ids),
                torch.ones(1),
                attention_mask=unpadded_mask,
                decoder_step_active=True,
            )
            padded = model.forward_hidden(
                model.embed_tokens(padded_ids),
                torch.ones(1),
                attention_mask=padded_mask,
                decoder_step_active=True,
            )

        torch.testing.assert_close(unpadded, padded[:, -unpadded_ids.size(1) :])

    def test_right_padding_and_batch_partner_do_not_change_valid_logits(self) -> None:
        model = BabyLMELF(_small_config("bert_mlm"))
        model.eval()
        ids = torch.tensor([[1, 7, 4, 2]])
        mask = torch.ones_like(ids)
        padded_ids = torch.tensor(
            [
                [1, 7, 4, 2, 3, 3],
                [1, 8, 9, 10, 11, 2],
            ]
        )
        padded_mask = padded_ids.ne(model.config.pad_token_id).long()

        with torch.no_grad():
            single_hidden = model.forward_hidden(
                model.embed_tokens(ids),
                torch.ones(1),
                attention_mask=mask,
                decoder_step_active=True,
            )
            batch_hidden = model.forward_hidden(
                model.embed_tokens(padded_ids),
                torch.ones(2),
                attention_mask=padded_mask,
                decoder_step_active=True,
            )
        torch.testing.assert_close(single_hidden, batch_hidden[:1, :4])
        torch.testing.assert_close(
            model.decode_hidden(single_hidden),
            model.decode_hidden(batch_hidden[:1, :4]),
        )

    def test_explicit_compact_positions_match_implicit_positions(self) -> None:
        model = BabyLMELF(_small_config())
        model.eval()
        ids = torch.tensor([[3, 3, 1, 7, 4, 2]])
        mask = ids.ne(model.config.pad_token_id).long()
        positions = torch.tensor([[0, 0, 0, 1, 2, 3]])
        embeddings = model.embed_tokens(ids)
        with torch.no_grad():
            implicit = model.forward_hidden(
                embeddings,
                torch.ones(1),
                attention_mask=mask,
                decoder_step_active=True,
            )
            explicit = model.forward_hidden(
                embeddings,
                torch.ones(1),
                attention_mask=mask,
                position_ids=positions,
                decoder_step_active=True,
            )
        torch.testing.assert_close(implicit[:, 2:], explicit[:, 2:])

    def test_segment_boundaries_block_cross_record_attention(self) -> None:
        model = BabyLMELF(_small_config("bert_mlm"))
        model.eval()
        first = torch.tensor([[1, 7, 8, 1, 9, 10]])
        changed_partner = torch.tensor([[1, 7, 8, 1, 11, 12]])
        attention_mask = torch.ones_like(first)
        segment_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])

        with torch.no_grad():
            first_hidden = model.forward_hidden(
                model.embed_tokens(first),
                torch.ones(1),
                attention_mask=attention_mask,
                segment_ids=segment_ids,
                decoder_step_active=True,
            )
            changed_hidden = model.forward_hidden(
                model.embed_tokens(changed_partner),
                torch.ones(1),
                attention_mask=attention_mask,
                segment_ids=segment_ids,
                decoder_step_active=True,
            )

        torch.testing.assert_close(first_hidden[:, :3], changed_hidden[:, :3])
        self.assertFalse(torch.allclose(first_hidden[:, 3:], changed_hidden[:, 3:]))

    def test_segment_positions_reset_at_each_record(self) -> None:
        model = BabyLMELF(_small_config())
        model.eval()
        ids = torch.tensor([[1, 7, 8, 1, 9, 10]])
        attention_mask = torch.ones_like(ids)
        segment_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])
        position_ids = torch.tensor([[0, 1, 2, 0, 1, 2]])
        embeddings = model.embed_tokens(ids)

        with torch.no_grad():
            implicit = model.forward_hidden(
                embeddings,
                torch.ones(1),
                attention_mask=attention_mask,
                segment_ids=segment_ids,
                decoder_step_active=True,
            )
            explicit = model.forward_hidden(
                embeddings,
                torch.ones(1),
                attention_mask=attention_mask,
                segment_ids=segment_ids,
                position_ids=position_ids,
                decoder_step_active=True,
            )

        torch.testing.assert_close(implicit, explicit)

    def test_segmented_padding_produces_finite_hidden_states(self) -> None:
        model = BabyLMELF(_small_config())
        model.eval()
        ids = torch.tensor([[1, 7, 3, 3]])
        attention_mask = torch.tensor([[1, 1, 0, 0]])
        segment_ids = torch.tensor([[0, 0, -1, -1]])

        with torch.no_grad():
            hidden = model.forward_hidden(
                model.embed_tokens(ids),
                torch.ones(1),
                attention_mask=attention_mask,
                segment_ids=segment_ids,
                decoder_step_active=True,
            )

        self.assertTrue(torch.isfinite(hidden).all())

    def test_model_uses_explicit_mode_tokens_and_learned_mask_latent(self) -> None:
        model = BabyLMELF(_small_config())
        state = model.state_dict()

        self.assertIn("denoise_mode_tokens", state)
        self.assertIn("decode_mode_tokens", state)
        self.assertIn("mlm_mask_latent", state)
        self.assertNotIn("mode_tokens", state)
        self.assertTrue(model.mlm_mask_latent.requires_grad)

        with torch.no_grad():
            model.denoise_mode_tokens.zero_()
            model.decode_mode_tokens.fill_(1.0)
        active = torch.tensor([0.0, 1.0, 0.5])
        mode = model._build_mode_tokens(
            active,
            batch=active.numel(),
            device=active.device,
            dtype=torch.float32,
        )

        torch.testing.assert_close(mode[0], torch.zeros_like(mode[0]))
        torch.testing.assert_close(mode[1], torch.ones_like(mode[1]))
        torch.testing.assert_close(mode[2], torch.full_like(mode[2], 0.5))

        embeddings = torch.randn(1, 4, model.config.embedding_size)
        attention_mask = torch.ones(1, 4, dtype=torch.long)
        denoise_hidden = model.forward_hidden(
            torch.cat((embeddings, torch.zeros_like(embeddings)), dim=-1),
            torch.ones(1),
            attention_mask=attention_mask,
            decoder_step_active=False,
        )
        decode_hidden = model.forward_hidden(
            torch.cat((embeddings, torch.zeros_like(embeddings)), dim=-1),
            torch.ones(1),
            attention_mask=attention_mask,
            decoder_step_active=True,
        )
        self.assertFalse(torch.allclose(denoise_hidden, decode_hidden))

    def test_default_decoder_head_keeps_legacy_parameters(self) -> None:
        model = BabyLMELF(_small_config())

        self.assertIsNone(model.decoder_layer_norm)
        self.assertIsNone(model.decoder_logit_scale)
        self.assertNotIn("decoder_layer_norm.weight", model.state_dict())
        self.assertNotIn("decoder_logit_scale", model.state_dict())

    def test_bert_mlm_head_adds_layer_norm_only(self) -> None:
        model = BabyLMELF(_small_config("bert_mlm"))

        self.assertIsNotNone(model.decoder_layer_norm)
        self.assertIsNone(model.decoder_logit_scale)
        self.assertIn("decoder_layer_norm.weight", model.state_dict())
        self.assertNotIn("decoder_logit_scale", model.state_dict())

        hidden = torch.randn(2, 5, model.config.hidden_size)
        logits = model.decode_hidden(hidden)
        self.assertEqual(tuple(logits.shape), (2, 5, model.config.vocab_size))
        self.assertTrue(torch.isfinite(logits).all())

    def test_scaled_bert_mlm_head_scales_scores_before_bias(self) -> None:
        model = BabyLMELF(_small_config("bert_mlm_scaled"))
        self.assertIsNotNone(model.decoder_layer_norm)
        self.assertIsNotNone(model.decoder_logit_scale)

        hidden = torch.randn(2, 5, model.config.hidden_size)
        with torch.no_grad():
            model.decoder_logit_scale.fill_(math.log(2.0))
            decoder_hidden = F.gelu(
                model.decoder_projection(hidden.float()),
                approximate="tanh",
            )
            decoder_hidden = model.decoder_layer_norm(decoder_hidden)
            expected = (
                2.0 * (decoder_hidden @ model.codebook.normalized_weight().T)
                + model.codebook.bias
            )

        torch.testing.assert_close(model.decode_hidden(hidden), expected)

    def test_unknown_decoder_head_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "decoder_head_type"):
            BabyLMELF(_small_config("not_a_head"))


if __name__ == "__main__":
    unittest.main()
