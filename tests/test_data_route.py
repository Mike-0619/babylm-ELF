from __future__ import annotations

import math
from pathlib import Path
from tempfile import TemporaryDirectory
import types
import unittest

import yaml

from babylm_elf.data.datasets import build_dataloader
from babylm_elf.data.export import audit_text
from babylm_elf.data.manifest import (
    artifact_entry,
    build_data_manifest,
    load_data_manifest,
    validate_manifest_artifacts,
    write_data_manifest,
)
from babylm_elf.data.pipeline import PreparePlan, staged_plan
from babylm_elf.data.schema import HFExportStats
from babylm_elf.data.token_stream import write_token_stream


class _FakeTokenizer:
    def token_to_id(self, token: str):
        return {"<unk>": 0, "<s>": 1, "<pad>": 3}.get(token)

    def get_vocab_size(self):
        return 16

    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return types.SimpleNamespace(
            ids=[5 + index % 10 for index, _ in enumerate(text.split())]
        )


class CanonicalDataRouteTest(unittest.TestCase):
    def test_every_training_config_uses_one_canonical_data_route(self) -> None:
        config_paths = sorted(Path("configs").glob("2026_*.yml"))
        self.assertTrue(config_paths)
        for path in config_paths:
            with self.subTest(path=path):
                raw = yaml.safe_load(path.read_text(encoding="utf-8"))
                data = raw["data"]
                scale = "100M" if "2026_100M" in path.name else "10M"
                self.assertEqual(
                    data["train_text"],
                    f"data/2026_{scale}/raw/train_{scale}.txt",
                )
                self.assertEqual(
                    data["train_path"],
                    f"data/2026_{scale}/tokenized/train_{scale}.bin",
                )
                self.assertEqual(
                    data["tokenizer_path"],
                    f"data/2026_{scale}/tokenizer/tokenizer.json",
                )
                self.assertEqual(
                    data["manifest_path"],
                    f"data/2026_{scale}/manifest.json",
                )
                self.assertNotIn("tokenizer_type", data)
                self.assertNotIn("normalization_profile", data)
                self.assertNotIn("train_fill_text", data)
                self.assertNotIn("allow_nominal_word_count_mismatch", data)
                self.assertNotIn("official_format_only", path.name)
                optim = raw["optim"]
                self.assertNotIn("ema_decay", optim)
                self.assertEqual(optim["ema_reference_decay"], 0.9999)
                self.assertEqual(optim["ema_reference_steps"], 95_000)
                self.assertTrue(optim["ema_warmup"])

    def test_manifest_steps_match_actual_dataloader_length(self) -> None:
        tokenizer = _FakeTokenizer()
        with TemporaryDirectory() as directory:
            root = Path(directory) / "2026_10M"
            raw_path = root / "raw/train.txt"
            tokenizer_path = root / "tokenizer/tokenizer.json"
            tokenized_path = root / "tokenized/train.bin"
            manifest_path = root / "manifest.json"
            raw_path.parent.mkdir(parents=True)
            tokenizer_path.parent.mkdir(parents=True)
            raw_path.write_text(
                "one two three four five\n\nsix seven eight nine ten eleven\n",
                encoding="utf-8",
            )
            tokenizer_path.write_text("{}", encoding="utf-8")
            tokenization_stats = write_token_stream(
                tokenizer,
                raw_path,
                tokenized_path,
                buffer_tokens=4,
            )
            raw_stats = audit_text(raw_path)
            export_stats = HFExportStats(
                dataset="fake",
                config=None,
                split="train",
                fingerprint="fingerprint",
                source_rows=2,
                source_words=11,
                usable_rows=2,
                dropped_rows=0,
                normalized_words=11,
            )
            actual_paths = {
                "raw": raw_path,
                "tokenizer": tokenizer_path,
                "tokenized": tokenized_path,
            }
            manifest = build_data_manifest(
                export_stats=export_stats,
                raw_stats=raw_stats,
                tokenization_stats=tokenization_stats,
                seq_length=4,
                world_size=1,
                batch_size=2,
                gradient_accumulation_steps=2,
                epochs=3,
                artifacts={
                    name: artifact_entry(path, path)
                    for name, path in actual_paths.items()
                },
            )
            write_data_manifest(manifest_path, manifest)
            validate_manifest_artifacts(load_data_manifest(manifest_path), actual_paths)

            loader = build_dataloader(
                tokenized_path,
                tokenizer,
                seq_length=4,
                batch_size=2,
                shuffle=False,
                drop_incomplete=True,
            )
            packing = manifest["packing"]
            self.assertEqual(
                packing["strategy"],
                "bos_segmented_epoch_offset_v1",
            )
            self.assertEqual(
                packing["attention_boundary"],
                "bos_record_block_diagonal",
            )
            self.assertEqual(
                packing["segment_identity"],
                "usable_official_row_bos",
            )
            self.assertTrue(packing["distributed_sampler_drop_last"])
            self.assertEqual(packing["distributed_chunks_per_epoch"], 3)
            self.assertEqual(packing["distributed_chunks_dropped_per_epoch"], 0)
            self.assertEqual(packing["dataloader_batches_per_rank"], len(loader))
            self.assertEqual(
                packing["optimizer_steps_per_epoch"],
                math.ceil(len(loader) / 2),
            )
            self.assertEqual(
                packing["total_optimizer_steps"],
                math.ceil(len(loader) / 2) * 3,
            )

    def test_staged_paths_stay_beside_canonical_root(self) -> None:
        root = Path("data/2026_10M")
        plan = PreparePlan(
            hf_dataset="fake",
            hf_config=None,
            hf_train_split="train",
            hf_text_field="text",
            train_text=root / "raw/train.txt",
            tokenizer_path=root / "tokenizer/tokenizer.json",
            train_output_path=root / "tokenized/train.bin",
            manifest_path=root / "manifest.json",
            canonical_root=root,
            staging_root=None,
            vocab_size=16384,
            source_word_budget=10_000_000,
            expected_normalized_words=9_999_993,
            expected_dropped_rows=3,
            expected_subwords=14_735_674,
            expected_stream_tokens=15_839_777,
            seq_length=128,
            world_size=4,
            batch_size=32,
            gradient_accumulation_steps=1,
            epochs=10,
        )
        staged = staged_plan(plan)
        self.assertEqual(staged.staging_root, Path("data/.2026_10M.staging"))
        self.assertEqual(
            staged.train_output_path,
            Path("data/.2026_10M.staging/tokenized/train.bin"),
        )


if __name__ == "__main__":
    unittest.main()
