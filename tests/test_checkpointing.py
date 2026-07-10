from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from babylm_elf.config import TrainConfig
from babylm_elf.training.checkpointing import (
    load_model_weights,
    save_checkpoint,
    select_model_weights,
)
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

    def test_checkpoint_saves_raw_and_ema_with_stability_metadata(self) -> None:
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        scheduler = create_scheduler(optimizer, warmup_steps=1, max_steps=10)
        ema = ExponentialMovingAverage(model, decay=0.9, warmup=False)
        with torch.no_grad():
            model.weight.add_(1.0)
            model.bias.add_(2.0)
        ema.update(model)
        state = TrainState(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            step=1,
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, state, TrainConfig(seed=17))
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)

        self.assertEqual(checkpoint["checkpoint_format_version"], 2)
        self.assertIn("model", checkpoint)
        self.assertIn("model_ema", checkpoint)
        self.assertIn("model_raw", checkpoint)
        self.assertIs(checkpoint["model"], checkpoint["model_ema"])
        self.assertFalse(
            torch.equal(
                checkpoint["model_raw"]["weight"],
                checkpoint["model_ema"]["weight"],
            )
        )
        metadata = checkpoint["metadata"]
        self.assertEqual(metadata["model_init_seed"], 17)
        self.assertEqual(metadata["runtime_seed_policy"], "base_seed_plus_global_rank")
        self.assertEqual(metadata["runtime_seed_rank0"], 17)
        self.assertEqual(metadata["ema_reference_decay"], 0.9999)
        self.assertEqual(metadata["ema_reference_steps"], 95_000)
        self.assertEqual(metadata["ema_resolved_decay"], 0.9)
        self.assertEqual(metadata["ema_current_decay"], 0.9)
        self.assertEqual(metadata["ema_num_updates"], 1)
        self.assertFalse(metadata["ema_warmup"])

    def test_weight_selection_and_legacy_compatibility(self) -> None:
        ema_state = {"weight": torch.full((2, 3), 1.0), "bias": torch.ones(2)}
        raw_state = {"weight": torch.full((2, 3), 2.0), "bias": torch.ones(2)}
        checkpoint = {
            "model": ema_state,
            "model_ema": ema_state,
            "model_raw": raw_state,
        }
        self.assertIs(select_model_weights(checkpoint), ema_state)
        self.assertIs(select_model_weights(checkpoint, weights="raw"), raw_state)

        legacy = {"model": ema_state}
        self.assertIs(select_model_weights(legacy), ema_state)
        with self.assertRaisesRegex(KeyError, "no 'model_raw'"):
            select_model_weights(legacy, weights="raw")

        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(checkpoint, path)
            loaded = torch.nn.Linear(3, 2)
            load_model_weights(path, loaded)
            torch.testing.assert_close(loaded.weight, ema_state["weight"])
            load_model_weights(path, loaded, weights="raw")
            torch.testing.assert_close(loaded.weight, raw_state["weight"])


if __name__ == "__main__":
    unittest.main()
