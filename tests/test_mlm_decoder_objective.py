from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

from babylm_elf.config import DiffusionConfig
from babylm_elf.training.step import make_mlm_inputs, train_step


class EchoInputTokenModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 32) -> None:
        super().__init__()
        self.config = SimpleNamespace(mask_token_id=4)
        self.vocab_size = vocab_size
        self.embedding_size = vocab_size
        self.embedded_input_ids: list[torch.Tensor] = []

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
        self_cond_cfg_scale: torch.Tensor | None = None,
        decoder_step_active: torch.Tensor | bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del t, attention_mask, self_cond_cfg_scale, decoder_step_active
        current_ids = x[..., : self.embedding_size].argmax(dim=-1)
        logits = x.new_full((*current_ids.shape, self.vocab_size), -5.0)
        logits.scatter_(-1, current_ids.unsqueeze(-1), 5.0)
        prediction = torch.zeros_like(x[..., : self.embedding_size])
        return prediction, logits


class MLMDecoderObjectiveTest(unittest.TestCase):
    def test_mlm_sampler_masks_only_non_special_non_pad_tokens(self) -> None:
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 15, 16, 17]])
        attention_mask = torch.ones_like(input_ids)

        masked_ids, mlm_mask = make_mlm_inputs(
            input_ids,
            attention_mask,
            mask_token_id=4,
            mask_probability=1.0,
            special_token_count=16,
            min_masks_per_sequence=1,
        )

        self.assertEqual(mlm_mask.tolist(), [[False, False, False, False, False, False, True, True]])
        self.assertEqual(masked_ids.tolist(), [[1, 2, 3, 4, 5, 15, 4, 4]])

    def test_mlm_sampler_ensures_minimum_masks_per_nonempty_sequence(self) -> None:
        torch.manual_seed(0)
        input_ids = torch.tensor([[1, 16, 17, 3], [1, 2, 3, 3]])
        attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

        masked_ids, mlm_mask = make_mlm_inputs(
            input_ids,
            attention_mask,
            mask_token_id=4,
            mask_probability=0.0,
            special_token_count=16,
            min_masks_per_sequence=1,
        )

        self.assertEqual(int(mlm_mask[0].sum().item()), 1)
        self.assertEqual(int(mlm_mask[1].sum().item()), 0)
        self.assertTrue(input_ids[0, mlm_mask[0]].ge(16).all())
        self.assertTrue(masked_ids[0, mlm_mask[0]].eq(4).all())
        self.assertTrue(masked_ids[attention_mask.eq(0)].eq(3).all())

    def test_train_step_mlm_decoder_scores_only_masked_positions(self) -> None:
        torch.manual_seed(0)
        model = EchoInputTokenModel()
        batch = {
            "input_ids": torch.tensor([[1, 16, 17, 18, 2], [1, 19, 20, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]),
        }
        config = DiffusionConfig(
            decoder_objective="mlm",
            decoder_probability=1.0,
            mlm_mask_probability=0.0,
            mlm_special_token_count=16,
            mlm_min_masks_per_sequence=1,
            self_condition_probability=0.0,
        )

        output = train_step(model, batch, config)

        self.assertTrue(torch.isfinite(output.loss))
        self.assertEqual(output.metrics["decode_frac"], 1.0)
        self.assertEqual(output.metrics["acc"], 0.0)
        self.assertEqual(len(model.embedded_input_ids), 2)
        torch.testing.assert_close(model.embedded_input_ids[0], batch["input_ids"])
        masked_input_ids = model.embedded_input_ids[1]
        mlm_mask = masked_input_ids.eq(4) & batch["input_ids"].ge(16)
        self.assertEqual(mlm_mask.sum(dim=1).tolist(), [1, 1])


if __name__ == "__main__":
    unittest.main()
