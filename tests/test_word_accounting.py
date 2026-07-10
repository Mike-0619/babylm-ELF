from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import patch

import numpy as np

from babylm_elf.config import TrainConfig
from babylm_elf.data.export import export_hf_split_to_text
from babylm_elf.data.manifest import (
    artifact_entry,
    build_data_manifest,
    validate_training_data_manifest,
    write_data_manifest,
)
from babylm_elf.data.schema import HFExportStats, TextAudit, TokenizationStats
from babylm_elf.data.text import count_words, iter_documents, normalize_document
from babylm_elf.training.checkpointing import CheckpointManager
from babylm_elf.training.trainer import _manifest_train_word_count


class WordAccountingTest(unittest.TestCase):
    def test_format_only_preserves_content_and_inner_whitespace(self) -> None:
        text = (
            "  *MOT:\tlook  at &lt;br&gt;\r\n"
            "= = Wiki header = =\r"
            "keep &amp; stay <ref>keep this</ref>  "
        )

        self.assertEqual(
            normalize_document(text),
            "*MOT: look  at &lt;br&gt;\n"
            "= = Wiki header = =\n"
            "keep &amp; stay <ref>keep this</ref>",
        )

    def test_format_only_handles_every_configured_control_character(self) -> None:
        text = (
            " one\u200btwo\ufeffthree\tend\r\n"
            "soft\u00adhyphen\u061c\u200c\u200d\u200e\u200f"
            "\u202a\u202b\u202c\u202d\u202e\u2060"
            "\u2066\u2067\u2068\u2069done "
        )

        self.assertEqual(
            normalize_document(text),
            "one two three end\nsofthyphendone",
        )

    def test_iterator_and_word_count_use_the_same_normalization(self) -> None:
        text = "one \u2060 two\n\nthree \u200e four\n\nfive\u200bsix"
        with TemporaryDirectory() as directory:
            path = Path(directory) / "train.txt"
            path.write_text(text, encoding="utf-8")

            self.assertEqual(
                list(iter_documents(path)),
                ["one  two", "three  four", "five six"],
            )
            self.assertEqual(count_words(path), 6)

    def test_hf_export_tracks_source_and_usable_accounting(self) -> None:
        class FakeDataset(list):
            _fingerprint = "fake-fingerprint"

            @property
            def num_rows(self):
                return len(self)

        rows = FakeDataset(
            [
                {"text": "  *MOT:\tone  two <ref>x</ref> "},
                {"text": "\u2060"},
                {"text": "= = Header = =\r\nthree\u200bfour"},
            ]
        )
        fake_datasets = types.ModuleType("datasets")
        fake_datasets.load_dataset = lambda *_args, **_kwargs: rows

        with TemporaryDirectory() as directory:
            path = Path(directory) / "train.txt"
            with patch.dict(sys.modules, {"datasets": fake_datasets}):
                stats = export_hf_split_to_text(
                    dataset_name="fake",
                    split="train",
                    output_path=path,
                    expected_source_words=11,
                )

            self.assertEqual(stats.fingerprint, "fake-fingerprint")
            self.assertEqual(stats.source_rows, 3)
            self.assertEqual(stats.source_words, 11)
            self.assertEqual(stats.usable_rows, 2)
            self.assertEqual(stats.dropped_rows, 1)
            self.assertEqual(stats.normalized_words, 11)
            self.assertEqual(count_words(path), 11)
            self.assertEqual(
                list(iter_documents(path)),
                [
                    "*MOT: one  two <ref>x</ref>",
                    "= = Header = =\nthree four",
                ],
            )

    def test_hf_export_rejects_changed_official_budget(self) -> None:
        fake_datasets = types.ModuleType("datasets")
        fake_datasets.load_dataset = lambda *_args, **_kwargs: [{"text": "one two"}]

        with TemporaryDirectory() as directory:
            path = Path(directory) / "train.txt"
            with patch.dict(sys.modules, {"datasets": fake_datasets}):
                with self.assertRaisesRegex(ValueError, "source word budget mismatch"):
                    export_hf_split_to_text(
                        dataset_name="fake",
                        split="train",
                        output_path=path,
                        expected_source_words=3,
                    )
            self.assertFalse(path.exists())

    def test_training_rejects_unmanifested_data(self) -> None:
        config = TrainConfig()
        config.checkpoint_by_words = True
        config.data.train_word_count = 3

        with self.assertRaisesRegex(ValueError, "manifest_path"):
            validate_training_data_manifest(config.data, object())

    def test_manifest_authorizes_exact_source_normalized_dual_accounting(self) -> None:
        class FakeTokenizer:
            def get_vocab_size(self):
                return 17

        config = TrainConfig()
        config.checkpoint_by_words = True
        config.data.train_word_count = 10

        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "train.txt"
            raw_path.write_text(
                "one two three four five six seven eight nine\n",
                encoding="utf-8",
            )
            tokenizer_path = root / "tokenizer.json"
            tokenizer_path.write_text("{}", encoding="utf-8")
            tokenized_path = root / "train.bin"
            np.asarray([1, 5, 6], dtype="<i2").tofile(tokenized_path)
            manifest_path = root / "manifest.json"
            actual_paths = {
                "raw": raw_path,
                "tokenizer": tokenizer_path,
                "tokenized": tokenized_path,
            }
            manifest = build_data_manifest(
                export_stats=HFExportStats(
                    dataset="fake",
                    config=None,
                    split="train",
                    fingerprint="fingerprint",
                    source_rows=1,
                    source_words=10,
                    usable_rows=1,
                    dropped_rows=0,
                    normalized_words=9,
                ),
                raw_stats=TextAudit(rows=1, words=9),
                tokenization_stats=TokenizationStats(
                    rows=1,
                    words=9,
                    subwords=2,
                    unk_tokens=0,
                    vocab_size=17,
                    emitted_vocab_size=2,
                    pieces_frequency_lt_10=2,
                    pieces_frequency_lt_100=2,
                ),
                seq_length=2,
                world_size=1,
                batch_size=1,
                gradient_accumulation_steps=1,
                epochs=1,
                artifacts={
                    name: artifact_entry(path, path)
                    for name, path in actual_paths.items()
                },
            )
            write_data_manifest(manifest_path, manifest)
            config.data.train_text = str(raw_path)
            config.data.train_path = str(tokenized_path)
            config.data.tokenizer_path = str(tokenizer_path)
            config.data.manifest_path = str(manifest_path)

            loaded_manifest = validate_training_data_manifest(
                config.data,
                FakeTokenizer(),
            )
            actual = _manifest_train_word_count(config, loaded_manifest)
            self.assertEqual(actual, 9)

    def test_metadata_keeps_nominal_and_actual_counts(self) -> None:
        config = TrainConfig()
        config.max_steps = 100
        config.checkpoint_by_words = True
        config.checkpoint_word_limit = 100_000_000
        config.data.train_word_count = 10_000_000

        with TemporaryDirectory() as directory:
            manager = CheckpointManager(
                directory,
                config,
                steps_per_epoch=10,
                microbatches_per_epoch=20,
                actual_train_word_count=9_999_993,
            )
            metadata = manager.metadata(step=100, microbatches_seen=200)

        self.assertEqual(manager.word_targets[-1], 100_000_000)
        self.assertEqual(metadata["words_seen"], 100_000_000)
        self.assertEqual(metadata["nominal_words_seen"], 100_000_000)
        self.assertEqual(metadata["nominal_corpus_word_count"], 10_000_000)
        self.assertEqual(metadata["actual_corpus_word_count"], 9_999_993)
        self.assertEqual(metadata["actual_words_seen"], 99_999_930)

    def test_strict_targets_still_end_at_one_billion(self) -> None:
        config = TrainConfig()
        config.max_steps = 100
        config.checkpoint_by_words = True
        config.checkpoint_word_limit = 1_000_000_000
        config.data.train_word_count = 100_000_000

        with TemporaryDirectory() as directory:
            manager = CheckpointManager(
                directory,
                config,
                steps_per_epoch=10,
                actual_train_word_count=99_999_996,
            )

        self.assertEqual(manager.word_targets[-1], 1_000_000_000)

    def test_encoder_offset_counts_toward_babylm_exposure(self) -> None:
        config = TrainConfig()
        config.max_steps = 70
        config.checkpoint_by_words = True
        config.checkpoint_word_limit = 100_000_000
        config.word_exposure_offset = 30_000_000
        config.data.train_word_count = 10_000_000

        with TemporaryDirectory() as directory:
            manager = CheckpointManager(
                directory,
                config,
                steps_per_epoch=10,
                microbatches_per_epoch=20,
                run_word_limit=70_000_000,
            )
            metadata = manager.metadata(step=70, microbatches_seen=140)

        self.assertEqual(manager.word_targets[0], 40_000_000)
        self.assertEqual(manager.word_targets[-1], 100_000_000)
        self.assertNotIn(30_000_000, manager.word_targets)
        self.assertEqual(metadata["words_seen"], 100_000_000)
        self.assertEqual(metadata["word_exposure_offset"], 30_000_000)
        self.assertEqual(metadata["elf_words_seen"], 70_000_000)
        self.assertEqual(metadata["elf_epochs_completed"], 7.0)
        self.assertEqual(metadata["epochs_completed"], 10.0)


if __name__ == "__main__":
    unittest.main()
