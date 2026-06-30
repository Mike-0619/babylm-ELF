from __future__ import annotations

import unittest

import torch

from babylm_elf.training.trainer import (
    _assert_finite_parameters,
    _assert_finite_step_output,
    _clip_gradients,
    _next_group_size,
)


class TrainingSafetyTest(unittest.TestCase):
    def test_accumulation_crosses_epoch_boundaries(self) -> None:
        total = 6
        seen = 0
        groups = []
        while seen < total:
            group_size = _next_group_size(total, seen, accumulation_steps=2)
            groups.append(group_size)
            seen += group_size

        self.assertEqual(groups, [2, 2, 2])

    def test_nonfinite_loss_fails_before_backward(self) -> None:
        loss = torch.tensor(float("nan"), requires_grad=True)

        with self.assertRaisesRegex(
            FloatingPointError,
            "optimizer step 7, microbatch 11",
        ):
            _assert_finite_step_output(
                loss,
                {"loss": float("nan"), "ce": 1.0},
                optimizer_step=7,
                microbatch=11,
                group_microbatch=3,
            )

        self.assertIsNone(loss.grad)

    def test_nonfinite_gradient_fails(self) -> None:
        model = torch.nn.Linear(2, 2)
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, float("inf"))

        with self.assertRaisesRegex(
            FloatingPointError,
            "before optimizer step 4",
        ):
            _clip_gradients(model, 1.0, optimizer_step=4)

    def test_nonfinite_parameter_fails(self) -> None:
        model = torch.nn.Linear(2, 2)
        with torch.no_grad():
            model.weight[0, 0] = float("nan")

        with self.assertRaisesRegex(
            FloatingPointError,
            "weight",
        ):
            _assert_finite_parameters(model, optimizer_step=5)


if __name__ == "__main__":
    unittest.main()
