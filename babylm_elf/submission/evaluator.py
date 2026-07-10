from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from .artifacts import validate_hub_repository
from .contract import (
    AOA_SAMPLES_PER_REVISION,
    FAST_TASKS,
    FULL_PREDICTION_COUNTS,
    FULL_TASKS,
    GLUE_SAMPLE_COUNTS,
    OFFICIAL_EVALUATOR_COMMIT,
    OFFICIAL_EVALUATOR_REPOSITORY,
    Track,
    required_revisions,
)


_EVALUATOR_EXCLUDES = (
    ".env",
    "strict/results/",
    "strict/models/",
    "strict/wandb/",
)


def prepare_official_evaluator(
    work_dir: str | Path,
    *,
    download_data: bool = False,
    python: str = sys.executable,
) -> Path:
    work_dir = Path(work_dir).expanduser().resolve()
    if not work_dir.exists():
        work_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", OFFICIAL_EVALUATOR_REPOSITORY, str(work_dir)])
    if not (work_dir / ".git").is_dir():
        raise RuntimeError(f"Evaluator work directory is not a Git checkout: {work_dir}")
    _require_clean_checkout(work_dir)
    if not _has_commit(work_dir, OFFICIAL_EVALUATOR_COMMIT):
        _run(["git", "fetch", "origin", OFFICIAL_EVALUATOR_COMMIT], cwd=work_dir)
    _run(["git", "checkout", "--detach", OFFICIAL_EVALUATOR_COMMIT], cwd=work_dir)
    _configure_local_excludes(work_dir)
    verify_official_evaluator(work_dir)

    if download_data:
        strict_dir = work_dir / "strict"
        _run([python, "-m", "scripts.download_evals"], cwd=strict_dir)
        _run([python, "-m", "evaluation_pipeline.global_piqa.dl"], cwd=strict_dir)
        verify_official_evaluator(work_dir)
    return work_dir


def verify_official_evaluator(work_dir: str | Path) -> None:
    work_dir = Path(work_dir).expanduser().resolve()
    commit = _capture(["git", "rev-parse", "HEAD"], cwd=work_dir).strip()
    if commit != OFFICIAL_EVALUATOR_COMMIT:
        raise RuntimeError(
            f"Official evaluator must be at {OFFICIAL_EVALUATOR_COMMIT}; found {commit}."
        )
    _require_clean_checkout(work_dir)
    if not (work_dir / "strict" / "evaluation_pipeline" / "collate_preds.py").is_file():
        raise FileNotFoundError("Official strict evaluator source is incomplete.")


def run_official_evaluation(
    repo_id: str,
    track: Track | str,
    work_dir: str | Path,
    *,
    backend: str = "mlm",
) -> Path:
    if backend != "mlm":
        raise ValueError("BabyLM-ELF official evaluation requires backend='mlm'.")
    _require_public_repo_id(repo_id)
    work_dir = Path(work_dir).expanduser().resolve()
    verify_official_evaluator(work_dir)
    validate_hub_repository(repo_id, track, require_public=True)
    strict_dir = work_dir / "strict"
    _validate_evaluation_data(strict_dir)

    env_file = work_dir / ".env"
    created_env = not env_file.exists()
    if created_env:
        env_file.touch()
    env = {
        **os.environ,
        "TOKENIZERS_PARALLELISM": "false",
        "WANDB_DISABLED": "true",
    }
    commands = (
        ["bash", "scripts/eval_zero_shot.sh", repo_id, backend],
        ["bash", "scripts/eval_finetuning.sh", "--model_path", repo_id],
        [
            "bash",
            "scripts/eval_zero_shot_fast_all_revisions.sh",
            repo_id,
            backend,
            str(track),
        ],
        [
            "bash",
            "scripts/eval_zero_shot_global_piqa.sh",
            repo_id,
            backend,
            str(track),
        ],
        [
            "bash",
            "scripts/eval_aoa.sh",
            repo_id,
            backend,
            str(track),
        ],
        ["bash", "scripts/collate_preds.sh", repo_id, backend, str(track)],
    )
    try:
        for command in commands:
            _run(command, cwd=strict_dir, env=env)
    finally:
        if created_env:
            env_file.unlink(missing_ok=True)

    result = (
        strict_dir
        / "results"
        / Path(repo_id).name
        / f"all_full_preds_and_fast_scores_{backend}.json"
    )
    validate_collated_submission(result, track)
    verify_official_evaluator(work_dir)
    return result


def validate_collated_submission(
    path_or_payload: str | Path | dict[str, Any],
    track: Track | str,
) -> dict[str, Any]:
    if isinstance(path_or_payload, dict):
        payload = path_or_payload
    else:
        path = Path(path_or_payload)
        if not path.is_file():
            raise FileNotFoundError(f"Collated submission does not exist: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))

    missing = FULL_TASKS - payload.keys()
    if missing:
        raise ValueError(f"Collated submission is missing tasks: {', '.join(sorted(missing))}")
    _validate_values(payload, "submission")

    for task, expected in FULL_PREDICTION_COUNTS.items():
        actual = _prediction_count(payload[task])
        if actual != expected:
            raise ValueError(f"{task} has {actual} predictions; expected {expected}.")

    glue = payload["glue"]
    if not isinstance(glue, dict):
        raise ValueError("glue must be an object.")
    missing_glue = GLUE_SAMPLE_COUNTS.keys() - glue.keys()
    if missing_glue:
        raise ValueError(f"glue is missing tasks: {', '.join(sorted(missing_glue))}")
    for task, expected in GLUE_SAMPLE_COUNTS.items():
        actual = _prediction_count(glue[task])
        if actual != expected:
            raise ValueError(f"glue/{task} has {actual} predictions; expected {expected}.")

    aoa = payload["aoa_surprisals"]
    aoa_results = aoa.get("results") if isinstance(aoa, dict) else None
    expected_aoa = AOA_SAMPLES_PER_REVISION * len(required_revisions(track))
    if not isinstance(aoa_results, list) or len(aoa_results) != expected_aoa:
        actual = None if not isinstance(aoa_results, list) else len(aoa_results)
        raise ValueError(f"AoA has {actual} rows; expected {expected_aoa}.")

    fast = payload["fast_eval_results"]
    if not isinstance(fast, dict):
        raise ValueError("fast_eval_results must be an object.")
    missing_fast = FAST_TASKS - fast.keys()
    if missing_fast:
        raise ValueError(f"Fast evaluation is missing: {', '.join(sorted(missing_fast))}")
    revision_count = len(required_revisions(track))
    for task in FAST_TASKS:
        scores = fast[task]
        if not isinstance(scores, list) or len(scores) != revision_count:
            actual = None if not isinstance(scores, list) else len(scores)
            raise ValueError(
                f"Fast task {task} has {actual} revisions; expected {revision_count}."
            )
    return payload


def _validate_evaluation_data(strict_dir: Path) -> None:
    required = (
        "evaluation_data/full_eval/blimp_filtered",
        "evaluation_data/full_eval/supplement_filtered",
        "evaluation_data/full_eval/entity_tracking",
        "evaluation_data/full_eval/comps",
        "evaluation_data/full_eval/reading/reading_data.csv",
        "evaluation_data/full_eval/glue_filtered",
        "evaluation_data/full_eval/aoa/cdi_childes.json",
        "evaluation_data/full_eval/global_piqa_parallel/eng_latn.jsonl",
        "evaluation_data/full_eval/global_piqa_nonparallel/eng_latn.jsonl",
        "evaluation_data/fast_eval/blimp_fast",
        "evaluation_data/fast_eval/supplement_fast",
        "evaluation_data/fast_eval/entity_tracking_fast",
        "evaluation_data/fast_eval/reading/reading_data.csv",
        "evaluation_data/fast_eval/global_piqa_parallel/eng_latn.jsonl",
        "evaluation_data/fast_eval/global_piqa_nonparallel/eng_latn.jsonl",
    )
    missing = [relative for relative in required if not (strict_dir / relative).exists()]
    ewok = strict_dir / "evaluation_data/full_eval/ewok_filtered"
    ewok_fast = strict_dir / "evaluation_data/fast_eval/ewok_fast"
    if not ewok.exists() or not ewok_fast.exists():
        raise FileNotFoundError(
            "EWoK data is missing. Accept the ewok-core dataset terms, log in to "
            "Hugging Face, then run from the evaluator strict/ directory: "
            "python -m evaluation_pipeline.ewok.dl_and_filter"
        )
    if missing:
        raise FileNotFoundError(
            "Official evaluation data is incomplete: " + ", ".join(missing)
        )


def _prediction_count(value: Any) -> int:
    if isinstance(value, dict):
        predictions = value.get("predictions")
        if isinstance(predictions, list):
            return len(predictions)
        return sum(_prediction_count(child) for child in value.values())
    return 0


def _validate_values(value: Any, path: str) -> None:
    if value is None:
        raise ValueError(f"{path} contains None.")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite number.")
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_values(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_values(child, f"{path}[{index}]")


def _require_public_repo_id(repo_id: str) -> None:
    if Path(repo_id).expanduser().exists() or repo_id.count("/") != 1 or "://" in repo_id:
        raise ValueError(
            "Official full evaluation only accepts a public Hugging Face repo id "
            "in the form namespace/model."
        )


def _configure_local_excludes(work_dir: Path) -> None:
    exclude_path = work_dir / ".git" / "info" / "exclude"
    current = exclude_path.read_text(encoding="utf-8") if exclude_path.exists() else ""
    additions = [entry for entry in _EVALUATOR_EXCLUDES if entry not in current.splitlines()]
    if additions:
        separator = "" if not current or current.endswith("\n") else "\n"
        exclude_path.write_text(
            current + separator + "\n".join(additions) + "\n",
            encoding="utf-8",
        )


def _require_clean_checkout(work_dir: Path) -> None:
    status = _capture(["git", "status", "--porcelain"], cwd=work_dir).strip()
    if status:
        raise RuntimeError(f"Official evaluator working tree is not clean:\n{status}")


def _has_commit(work_dir: Path, commit: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=work_dir,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _run(command: list[str], *, cwd: Path | None = None, env=None) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _capture(command: list[str], *, cwd: Path) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
