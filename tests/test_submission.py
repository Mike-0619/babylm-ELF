from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from babylm_elf.submission.artifacts import (
    validate_hub_repository,
    validate_revision_tree,
)
from babylm_elf.submission.contract import (
    HF_ARTIFACT_FILES,
    OFFICIAL_EVALUATOR_COMMIT,
    checkpoint_targets,
    required_revisions,
    revision_for_words,
)
from babylm_elf.submission.evaluator import validate_collated_submission


def _write_artifact(path: Path) -> None:
    path.mkdir(parents=True)
    for name in HF_ARTIFACT_FILES:
        (path / name).write_text("{}", encoding="utf-8")


class SubmissionTest(unittest.TestCase):
    def test_contract_has_exact_official_revisions(self) -> None:
        small = required_revisions("strict-small")
        strict = required_revisions("strict")
        self.assertEqual(len(small), 19)
        self.assertEqual(len(strict), 28)
        self.assertEqual(small[0], "chck_1M")
        self.assertEqual(small[-1], "chck_100M")
        self.assertEqual(strict[-1], "chck_1000M")
        self.assertEqual(
            checkpoint_targets(100_000_000, word_exposure_offset=30_000_000)[0],
            40_000_000,
        )
        self.assertEqual(revision_for_words(1_000_000), "chck_1M")
        self.assertEqual(len(OFFICIAL_EVALUATOR_COMMIT), 40)

    def test_validate_revision_tree_is_fail_closed(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            main = root / "model_hf"
            _write_artifact(main)
            for revision in required_revisions("strict-small"):
                _write_artifact(root / revision)

            tree = validate_revision_tree(root, "strict-small")
            self.assertEqual(tree.main, main)
            self.assertEqual(
                tuple(name for name, _ in tree.revisions),
                required_revisions("strict-small"),
            )
            (root / "chck_1M" / "model.safetensors").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "model.safetensors"):
                validate_revision_tree(root, "strict-small")

    def test_missing_main_or_revision_fails_closed(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "exactly one main"):
                validate_revision_tree(root, "strict-small")
            _write_artifact(root / "main_export")
            with self.assertRaisesRegex(FileNotFoundError, "chck_1M"):
                validate_revision_tree(root, "strict-small")

    def test_hub_validation_checks_every_revision_and_file(self) -> None:
        class FakeApi:
            def __init__(self):
                self.revisions = []

            def model_info(self, _repo_id, *, revision):
                self.revisions.append(revision)
                files = HF_ARTIFACT_FILES
                if revision == "chck_2M":
                    files = files - {"config.json"}
                return SimpleNamespace(
                    siblings=[SimpleNamespace(rfilename=name) for name in files]
                )

        api = FakeApi()
        with self.assertRaisesRegex(FileNotFoundError, "config.json"):
            validate_hub_repository("org/model", "strict-small", api=api)
        self.assertEqual(api.revisions[:3], ["main", "chck_1M", "chck_2M"])

    def test_collated_validator_rejects_missing_none_nan_and_bad_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing tasks"):
            validate_collated_submission({}, "strict-small")

        task_names = {
            "blimp",
            "blimp_supplement",
            "ewok",
            "entity_tracking_filtered",
            "comps",
            "global_piqa_parallel",
            "global_piqa_nonparallel",
            "reading",
            "aoa_surprisals",
            "aoa",
            "glue",
            "fast_eval_results",
        }
        for invalid in (None, float("nan"), float("inf")):
            payload = {name: {} for name in task_names}
            payload["blimp"] = invalid
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "None|non-finite"):
                    validate_collated_submission(payload, "strict-small")

        payload = {name: {} for name in task_names}
        with self.assertRaisesRegex(ValueError, "blimp has 0 predictions"):
            validate_collated_submission(payload, "strict-small")


if __name__ == "__main__":
    unittest.main()
