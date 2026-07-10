from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from babylm_elf.data.datasets import TokenizedTextDataset, build_dataloader


class _FakeTokenizer:
    def token_to_id(self, token: str):
        return {"<s>": 1, "<pad>": 3}.get(token)


class TokenizedTextDatasetTest(unittest.TestCase):
    def test_drop_incomplete_removes_padded_tail_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tokens.bin"
            np.asarray([1, *range(16, 26)], dtype="<i2").tofile(path)

            padded = TokenizedTextDataset(
                path,
                seq_length=4,
                pad_token_id=3,
            )
            full_only = TokenizedTextDataset(
                path,
                seq_length=4,
                pad_token_id=3,
                drop_incomplete=True,
            )

            self.assertEqual(len(padded), 3)
            self.assertEqual(len(full_only), 2)
            self.assertEqual(
                padded[2]["input_ids"].tolist(),
                [23, 24, 25, 3],
            )
            self.assertEqual(
                padded[2]["attention_mask"].tolist(),
                [1, 1, 1, 0],
            )
            self.assertEqual(int(padded[2]["sequence_id"]), 2)
            for index in range(len(full_only)):
                self.assertEqual(int(full_only[index]["sequence_id"]), index)
                self.assertEqual(
                    full_only[index]["attention_mask"].tolist(),
                    [1] * 4,
                )

    def test_distributed_sampler_never_pads_with_duplicate_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tokens.bin"
            np.arange(20, dtype="<i2").tofile(path)
            loaders = [
                build_dataloader(
                    path,
                    _FakeTokenizer(),
                    seq_length=4,
                    batch_size=2,
                    shuffle=True,
                    distributed=True,
                    rank=rank,
                    world_size=4,
                    seed=7,
                    drop_incomplete=True,
                )
                for rank in range(4)
            ]

            epoch_indices = []
            for epoch in (0, 1):
                indices = []
                for loader in loaders:
                    loader.sampler.set_epoch(epoch)
                    indices.extend(list(loader.sampler))
                self.assertEqual(len(indices), 4)
                self.assertEqual(len(indices), len(set(indices)))
                self.assertTrue(set(indices).issubset(range(5)))
                epoch_indices.append(set(indices))
            self.assertNotEqual(epoch_indices[0], epoch_indices[1])


if __name__ == "__main__":
    unittest.main()
