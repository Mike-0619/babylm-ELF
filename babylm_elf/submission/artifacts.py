from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

from .contract import HF_ARTIFACT_FILES, Track, required_revisions


@dataclass(frozen=True)
class RevisionTree:
    main: Path
    revisions: tuple[tuple[str, Path], ...]


def validate_revision_tree(root: str | Path, track: Track | str) -> RevisionTree:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Revision directory does not exist: {root}")
    main_candidates = [
        child
        for child in root.iterdir()
        if child.is_dir() and not child.name.startswith("chck_")
        and (child / "config.json").is_file()
    ]
    if len(main_candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one main HF export under {root}; "
            f"found {len(main_candidates)}."
        )
    main = main_candidates[0]
    _validate_artifact(main, "main")

    revisions: list[tuple[str, Path]] = []
    for revision in required_revisions(track):
        artifact = root / revision
        _validate_artifact(artifact, revision)
        revisions.append((revision, artifact))
    return RevisionTree(main, tuple(revisions))


def validate_hub_repository(
    repo_id: str,
    track: Track | str,
    *,
    api: HfApi | None = None,
    require_public: bool = False,
) -> None:
    api = api or HfApi()
    for revision in ("main", *required_revisions(track)):
        try:
            info = api.model_info(repo_id, revision=revision)
        except Exception as exc:
            raise RuntimeError(f"Missing or inaccessible {repo_id}@{revision}.") from exc
        if (
            require_public
            and revision == "main"
            and bool(getattr(info, "private", False))
        ):
            raise ValueError(
                f"Official full evaluation requires a public Hub repository: {repo_id}"
            )
        files = {sibling.rfilename for sibling in info.siblings or ()}
        missing = HF_ARTIFACT_FILES - files
        if missing:
            raise FileNotFoundError(
                f"{repo_id}@{revision} is missing: {', '.join(sorted(missing))}"
            )


def smoke_local_artifact(path: str | Path) -> None:
    path = str(path)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    input_ids = torch.tensor(
        [[tokenizer.bos_token_id, tokenizer.mask_token_id, tokenizer.eos_token_id]]
    )
    attention_mask = torch.ones_like(input_ids)
    encoder = AutoModel.from_pretrained(path, trust_remote_code=True)
    masked_lm = AutoModelForMaskedLM.from_pretrained(path, trust_remote_code=True)
    with torch.no_grad():
        hidden = encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = masked_lm(input_ids=input_ids, attention_mask=attention_mask).logits
    if hidden.shape[:2] != input_ids.shape or not torch.isfinite(hidden).all():
        raise RuntimeError(f"AutoModel smoke failed for {path}.")
    if logits.shape[:2] != input_ids.shape or not torch.isfinite(logits).all():
        raise RuntimeError(f"AutoModelForMaskedLM smoke failed for {path}.")


def smoke_hub_revision(repo_id: str, revision: str) -> None:
    with TemporaryDirectory(prefix="babylm-elf-hf-smoke-") as directory:
        local_dir = snapshot_download(repo_id, revision=revision, local_dir=directory)
        smoke_local_artifact(local_dir)


def _validate_artifact(path: Path, revision: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing required HF revision {revision}: {path}")
    files = {child.name for child in path.iterdir() if child.is_file()}
    missing = HF_ARTIFACT_FILES - files
    if missing:
        raise FileNotFoundError(
            f"HF artifact {revision} is missing: {', '.join(sorted(missing))}"
        )
