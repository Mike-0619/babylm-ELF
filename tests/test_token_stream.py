from __future__ import annotations

from pathlib import Path
import random
import struct
from tempfile import TemporaryDirectory
import types
import unittest

import torch

from babylm_elf.data.datasets import TokenizedTextDataset
from babylm_elf.data.token_stream import (
    MAX_TOKEN_ID,
    TOKEN_BYTES,
    open_token_stream,
    validate_token_stream_file,
    write_token_stream,
)


class _IntegerWordTokenizer:
    def __init__(self, vocab_size: int = 64) -> None:
        self.vocab_size = vocab_size

    def token_to_id(self, token: str):
        return {"<unk>": 0, "<s>": 1, "<pad>": 3}.get(token)

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def encode(self, text: str, add_special_tokens: bool = False):
        self.assert_no_special_tokens(add_special_tokens)
        return types.SimpleNamespace(
            ids=[int(word.removeprefix("w")) for word in text.split()]
        )

    @staticmethod
    def assert_no_special_tokens(value: bool) -> None:
        if value:
            raise AssertionError("The canonical stream inserts only row BOS tokens.")


class TokenStreamTest(unittest.TestCase):
    def test_binary_round_trip_bos_order_size_and_endianness(self) -> None:
        tokenizer = _IntegerWordTokenizer()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "train.txt"
            output_path = root / "train.bin"
            raw_path.write_text("w5 w6\n\nw7 w8 w9\n", encoding="utf-8")

            stats = write_token_stream(
                tokenizer,
                raw_path,
                output_path,
                buffer_tokens=3,
            )

            expected = [1, 5, 6, 1, 7, 8, 9]
            self.assertEqual(open_token_stream(output_path).tolist(), expected)
            self.assertEqual(output_path.stat().st_size, len(expected) * TOKEN_BYTES)
            self.assertEqual(validate_token_stream_file(output_path), len(expected))
            self.assertEqual(output_path.read_bytes()[:4], struct.pack("<hh", 1, 5))
            self.assertEqual(stats.rows, 2)
            self.assertEqual(stats.words, 5)
            self.assertEqual(stats.subwords, 5)
            self.assertEqual(stats.stream_tokens, 7)
            self.assertEqual(stats.unk_tokens, 0)
            self.assertEqual(stats.emitted_vocab_size, 5)

    def test_mmap_chunks_match_the_previous_logical_stream_bit_for_bit(self) -> None:
        tokenizer = _IntegerWordTokenizer()
        generator = random.Random(17)
        documents = [
            [generator.randrange(4, 64) for _ in range(length)]
            for length in (11, 3, 17, 1, 8)
        ]
        legacy_stream = [value for document in documents for value in [1, *document]]

        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "train.txt"
            output_path = root / "train.bin"
            raw_path.write_text(
                "\n\n".join(" ".join(f"w{value}" for value in row) for row in documents),
                encoding="utf-8",
            )
            write_token_stream(tokenizer, raw_path, output_path, buffer_tokens=7)
            dataset = TokenizedTextDataset(
                output_path,
                seq_length=8,
                pad_token_id=3,
                drop_incomplete=False,
            )

            for index in range(len(dataset)):
                start = index * 8
                expected = legacy_stream[start : start + 8]
                expected += [3] * (8 - len(expected))
                self.assertTrue(
                    torch.equal(
                        dataset[index]["input_ids"],
                        torch.tensor(expected, dtype=torch.long),
                    )
                )

            self.assertEqual(dataset[1]["input_ids"][4].item(), 1)
            full_only = TokenizedTextDataset(
                output_path,
                seq_length=8,
                pad_token_id=3,
                drop_incomplete=True,
            )
            self.assertEqual(len(full_only), len(legacy_stream) // 8)

    def test_rejects_out_of_range_token_ids_and_malformed_files(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "train.txt"
            output_path = root / "train.bin"
            raw_path.write_text("w5", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "signed int16 capacity"):
                write_token_stream(
                    _IntegerWordTokenizer(MAX_TOKEN_ID + 2),
                    raw_path,
                    output_path,
                )
            self.assertFalse(output_path.exists())

            output_path.write_bytes(b"\x01")
            with self.assertRaisesRegex(ValueError, "byte size"):
                validate_token_stream_file(output_path)


if __name__ == "__main__":
    unittest.main()
