from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from babylm_elf.config import (
    DataConfig,
    DiffusionConfig,
    OptimConfig,
    TrainConfig,
)
from babylm_elf.data.export import audit_text
from babylm_elf.data.manifest import (
    artifact_entry,
    build_data_manifest,
    write_data_manifest,
)
from babylm_elf.data.schema import HFExportStats
from babylm_elf.data.token_stream import write_token_stream
from babylm_elf.data.tokenizer import load_tokenizer
from babylm_elf.modeling.model import BabyLMELFConfig
from babylm_elf.training.trainer import train_from_config
from babylm_elf.training.optim import (
    ExponentialMovingAverage,
    resolve_ema_decay,
    runtime_seed_for_rank,
    seed_everything,
)


def _ddp_seed_worker(
    rank: int,
    world_size: int,
    init_path: str,
    result_dir: str,
    base_seed: int,
) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        seed_everything(base_seed)
        model = DistributedDataParallel(torch.nn.Linear(4, 3))
        initial_parameters = torch.cat(
            [parameter.detach().flatten() for parameter in model.parameters()]
        )

        runtime_seed = runtime_seed_for_rank(base_seed, rank)
        seed_everything(runtime_seed)
        stochastic_values = torch.cat(
            [
                torch.randn(8),
                torch.rand(8),
                torch.randperm(8).float(),
            ]
        )
        torch.save(
            {
                "initial_parameters": initial_parameters,
                "runtime_seed": runtime_seed,
                "stochastic_values": stochastic_values,
            },
            Path(result_dir) / f"rank_{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


class TrainingStabilityTest(unittest.TestCase):
    def test_scaled_ema_targets_match_current_runs(self) -> None:
        decay_10m = resolve_ema_decay(0.9999, 95_000, 9_670)
        decay_100m = resolve_ema_decay(0.9999, 95_000, 49_130)

        self.assertAlmostEqual(decay_10m, 0.999018, places=6)
        self.assertAlmostEqual(decay_100m, 0.999807, places=6)

    def test_ema_warmup_is_monotonic_and_updates_once_per_call(self) -> None:
        model = torch.nn.Linear(2, 2)
        ema = ExponentialMovingAverage(model, decay=0.999, warmup=True)
        observed_decays = []
        for _ in range(20):
            with torch.no_grad():
                model.weight.add_(0.1)
            ema.update(model)
            observed_decays.append(ema.current_decay)

        self.assertEqual(ema.num_updates, 20)
        self.assertEqual(observed_decays, sorted(observed_decays))
        self.assertTrue(all(decay <= ema.decay for decay in observed_decays))
        self.assertAlmostEqual(observed_decays[0], 2.0 / 11.0)

    def test_ema_state_round_trip_preserves_warmup_progress(self) -> None:
        model = torch.nn.Linear(2, 2)
        source = ExponentialMovingAverage(model, decay=0.99, warmup=True)
        source.update(model)
        source.update(model)

        restored = ExponentialMovingAverage(model, decay=0.5, warmup=False)
        restored.load_state_dict(source.state_dict())

        self.assertEqual(restored.decay, source.decay)
        self.assertEqual(restored.warmup, source.warmup)
        self.assertEqual(restored.num_updates, source.num_updates)
        self.assertEqual(restored.current_decay, source.current_decay)

    def test_short_cpu_training_logs_and_checkpoint_stability_state(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        source_tokenizer = project_root / "data/2026_10M/tokenizer/tokenizer.json"
        if not source_tokenizer.exists():
            self.skipTest("canonical 10M tokenizer is unavailable")

        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "raw/train.txt"
            tokenizer_path = root / "tokenizer/tokenizer.json"
            tokenized_path = root / "tokenized/train.bin"
            manifest_path = root / "manifest.json"
            raw_path.parent.mkdir(parents=True)
            tokenizer_path.parent.mkdir(parents=True)
            raw_path.write_text(
                "\n".join(
                    f"row {index} has enough words for the smoke test"
                    for index in range(32)
                )
                + "\n",
                encoding="utf-8",
            )
            shutil.copy2(source_tokenizer, tokenizer_path)
            tokenizer = load_tokenizer(tokenizer_path)
            tokenization_stats = write_token_stream(
                tokenizer,
                raw_path,
                tokenized_path,
                buffer_tokens=32,
            )
            raw_stats = audit_text(raw_path)
            export_stats = HFExportStats(
                dataset="smoke",
                config=None,
                split="train",
                fingerprint="smoke",
                source_rows=raw_stats.rows,
                source_words=raw_stats.words,
                usable_rows=raw_stats.rows,
                dropped_rows=0,
                normalized_words=raw_stats.words,
            )
            artifacts = {
                name: artifact_entry(path, path)
                for name, path in {
                    "raw": raw_path,
                    "tokenizer": tokenizer_path,
                    "tokenized": tokenized_path,
                }.items()
            }
            manifest = build_data_manifest(
                export_stats=export_stats,
                raw_stats=raw_stats,
                tokenization_stats=tokenization_stats,
                seq_length=8,
                world_size=1,
                batch_size=2,
                gradient_accumulation_steps=2,
                epochs=1,
                artifacts=artifacts,
            )
            # This smoke test intentionally exercises the non-DDP loader. The
            # production manifests use DistributedSampler(drop_last=True).
            manifest["packing"]["distributed_sampler_drop_last"] = False
            write_data_manifest(manifest_path, manifest)

            config = TrainConfig(
                name="stability_smoke",
                seed=23,
                output_dir=str(root / "outputs"),
                max_steps=2,
                epochs=1,
                batch_size=2,
                gradient_accumulation_steps=2,
                log_every=1,
                validate_every=100,
                save_every=1,
                mixed_precision=False,
                device="cpu",
                model=BabyLMELFConfig(
                    vocab_size=tokenizer.get_vocab_size(),
                    base_vocab_size=tokenizer.get_vocab_size(),
                    embedding_size=16,
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=1,
                    num_attention_heads=4,
                    max_position_embeddings=8,
                    bottleneck_size=8,
                    hidden_dropout_prob=0.0,
                ),
                data=DataConfig(
                    train_text=str(raw_path),
                    train_path=str(tokenized_path),
                    valid_path=None,
                    tokenizer_path=str(tokenizer_path),
                    manifest_path=str(manifest_path),
                    tokenizer_vocab_size=tokenizer.get_vocab_size(),
                    seq_length=8,
                    train_word_count=raw_stats.words,
                ),
                optim=OptimConfig(
                    optimizer="adamw",
                    learning_rate=1.0e-4,
                    warmup_steps=1,
                    warmup_epochs=None,
                    ema_reference_decay=0.9999,
                    ema_reference_steps=95_000,
                    ema_warmup=True,
                ),
                diffusion=DiffusionConfig(decoder_probability=0.0),
            )
            output = io.StringIO()
            with redirect_stdout(output):
                train_from_config(config)

            checkpoint_path = (
                root
                / "outputs/stability_smoke/checkpoints/final.pt"
            )
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            self.assertIn("total_optimizer_steps=2", output.getvalue())
            self.assertIn("runtime=base_seed+global_rank", output.getvalue())
            self.assertEqual(checkpoint["step"], 2)
            self.assertEqual(checkpoint["metadata"]["microbatches_seen"], 4)
            self.assertEqual(checkpoint["metadata"]["ema_num_updates"], 2)
            self.assertIn("model_ema", checkpoint)
            self.assertIn("model_raw", checkpoint)

    @unittest.skipUnless(
        dist.is_available() and dist.is_gloo_available(),
        "torch.distributed with gloo is required",
    )
    def test_ddp_initialization_matches_but_runtime_rng_is_rank_specific(self) -> None:
        first = self._run_seed_probe()
        second = self._run_seed_probe()

        torch.testing.assert_close(
            first[0]["initial_parameters"],
            first[1]["initial_parameters"],
        )
        self.assertEqual(first[0]["runtime_seed"], 1234)
        self.assertEqual(first[1]["runtime_seed"], 1235)
        self.assertFalse(
            torch.equal(
                first[0]["stochastic_values"],
                first[1]["stochastic_values"],
            )
        )
        for rank in range(2):
            torch.testing.assert_close(
                first[rank]["initial_parameters"],
                second[rank]["initial_parameters"],
            )
            torch.testing.assert_close(
                first[rank]["stochastic_values"],
                second[rank]["stochastic_values"],
            )

    def _run_seed_probe(self) -> dict[int, dict]:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            init_path = root / "distributed_init"
            mp.spawn(
                _ddp_seed_worker,
                args=(2, str(init_path), str(root), 1234),
                nprocs=2,
                join=True,
            )
            return {
                rank: torch.load(
                    root / f"rank_{rank}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                for rank in range(2)
            }


if __name__ == "__main__":
    unittest.main()
