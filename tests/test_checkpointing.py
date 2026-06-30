from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from babylm_elf.config import TrainConfig
from babylm_elf.training.checkpointing import save_checkpoint
from babylm_elf.training.optim import (
    ExponentialMovingAverage,
    TrainState,
    create_scheduler,
)


class CheckpointingTest(unittest.TestCase):
    def test_checkpoint_is_atomically_replaced(self) -> None:
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        scheduler = create_scheduler(optimizer, warmup_steps=1, max_steps=10)
        ema = ExponentialMovingAverage(model, decay=0.9)
        state = TrainState(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            step=1,
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(
                path,
                state,
                TrainConfig(),
            )
            state.step = 2
            save_checkpoint(
                path,
                state,
                TrainConfig(),
            )
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)

            self.assertEqual(checkpoint["step"], 2)
            self.assertEqual(list(Path(directory).glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
