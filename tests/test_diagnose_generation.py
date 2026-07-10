from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
try:
    from transformers import AutoConfig
except ModuleNotFoundError as exc:
    if exc.name != "transformers":
        raise
    AutoConfig = None

from babylm_elf.cli.diagnose_generation import run_diagnostics
from babylm_elf.config import DataConfig, DiffusionConfig, TrainConfig
from babylm_elf.export.convert_checkpoint import export_checkpoint_to_hf
from babylm_elf.modeling.model import BabyLMELF, BabyLMELFConfig


def _test_tokenizer_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    smoke_path = project_root / "data/smoke/tokenizer/tokenizer.json"
    if smoke_path.exists():
        return smoke_path
    return project_root / "data/2026_10M/tokenizer/tokenizer.json"


@unittest.skipIf(AutoConfig is None, "transformers is not installed")
class DiagnoseGenerationTest(unittest.TestCase):
    def test_diagnose_generation_writes_samples_and_denoise_report(self) -> None:
        tokenizer_path = _test_tokenizer_path()
        model_config = BabyLMELFConfig(
            vocab_size=384,
            embedding_size=16,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=16,
            bottleneck_size=8,
        )
        config = TrainConfig(
            name="diagnose_smoke",
            model=model_config,
            data=DataConfig(
                tokenizer_path=str(tokenizer_path),
                tokenizer_vocab_size=384,
                seq_length=16,
            ),
            diffusion=DiffusionConfig(
                decoder_objective="token_mlm",
                decoder_probability=0.2,
            ),
        )
        model = BabyLMELF(model_config)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "checkpoint.pt"
            export_dir = root / "hf"
            tokenized_path = root / "train.bin"
            torch.save({"model": model.state_dict()}, checkpoint_path)
            np.asarray([1, 16, 17, 18, 19, 20, 21], dtype="<i2").tofile(
                tokenized_path
            )
            export_checkpoint_to_hf(checkpoint_path, export_dir, config)

            with patch.dict(
                "os.environ",
                {"HF_MODULES_CACHE": str(root / "hf_modules")},
            ), patch(
                "transformers.dynamic_module_utils.HF_MODULES_CACHE",
                str(root / "hf_modules"),
            ):
                run_dir = run_diagnostics(
                    export_dir,
                    output_dir=root / "diagnostics",
                    seed=0,
                    num_samples=2,
                    sequence_length=8,
                    steps=(2,),
                    methods=("ode",),
                    denoise_times=(0.0, 0.5),
                    tokenized_path=tokenized_path,
                    denoise_batch_size=1,
                    denoise_batches=1,
                )

            self.assertTrue((run_dir / "samples_ode_2.txt").exists())
            self.assertTrue((run_dir / "summary.md").exists())
            report = json.loads((run_dir / "denoise_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "ok")
            self.assertEqual(len(report["time_points"]), 2)
            for row in report["time_points"]:
                self.assertIn("mse", row)
                self.assertIn("token_accuracy", row)
                self.assertIn("maskable_token_ce", row)


if __name__ == "__main__":
    unittest.main()
