from __future__ import annotations

import unittest

import torch

from babylm_elf.encoder.span_corruption import make_t5_span_corruption_batch


class SpanCorruptionTest(unittest.TestCase):
    def test_internal_sentinels_and_label_padding(self) -> None:
        input_ids = torch.tensor(
            [
                [1, 16, 17, 18, 19, 20, 21, 2],
                [1, 16, 17, 18, 3, 3, 3, 3],
            ]
        )
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 0, 0],
            ]
        )
        generator = torch.Generator()
        generator.manual_seed(7)

        batch = make_t5_span_corruption_batch(
            input_ids,
            attention_mask,
            pad_token_id=3,
            eos_token_id=2,
            sentinel_start_id=16384,
            sentinel_count=100,
            noise_density=0.5,
            mean_noise_span_length=2.0,
            special_token_count=16,
            generator=generator,
        )

        self.assertEqual(batch.input_ids.shape, batch.attention_mask.shape)
        self.assertTrue(batch.input_ids.ge(0).all())
        self.assertTrue((batch.input_ids >= 16384).any())
        self.assertTrue((batch.labels >= 16384).any())
        self.assertTrue((batch.labels == -100).any())
        self.assertTrue(torch.isin(torch.tensor(1), batch.input_ids[0]))
        self.assertTrue(torch.isin(torch.tensor(2), batch.input_ids[0]))


if __name__ == "__main__":
    unittest.main()
