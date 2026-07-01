from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from babylm_elf.config import TrainConfig
from babylm_elf.data.text import count_clean_words, iter_clean_documents
from babylm_elf.training.checkpointing import CheckpointManager


class WordAccountingTest(unittest.TestCase):
    def test_clean_word_count_matches_tokenization_input(self) -> None:
        text = (
            "first document\nwith four words\n\n"
            "= = = source/file.txt = = =\n"
            "second document\n\n\n"
            "third"
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "train.txt"
            path.write_text(text, encoding="utf-8")

            self.assertEqual(
                list(iter_clean_documents(path)),
                [
                    "first document\nwith four words",
                    "second document",
                    "third",
                ],
            )
            self.assertEqual(count_clean_words(path), 8)

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
                actual_train_word_count=9_863_479,
            )
            metadata = manager.metadata(
                step=100,
                microbatches_seen=200,
            )

        self.assertEqual(manager.word_targets[-1], 100_000_000)
        self.assertEqual(metadata["words_seen"], 100_000_000)
        self.assertEqual(metadata["nominal_words_seen"], 100_000_000)
        self.assertEqual(metadata["nominal_corpus_word_count"], 10_000_000)
        self.assertEqual(metadata["actual_corpus_word_count"], 9_863_479)
        self.assertEqual(metadata["actual_words_seen"], 98_634_790)

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
                actual_train_word_count=98_969_109,
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
            metadata = manager.metadata(
                step=70,
                microbatches_seen=140,
            )

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
