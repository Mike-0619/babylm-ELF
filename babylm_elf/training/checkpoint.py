from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import random
from typing import Literal
import uuid

import torch
import torch.distributed as dist

Track = Literal["strict-small", "strict"]

STRICT_SMALL_MILLIONS = (*range(1, 10), *range(10, 101, 10))
STRICT_MILLIONS = (*STRICT_SMALL_MILLIONS, *range(200, 1001, 100))


def required_revisions(track: Track | str) -> tuple[str, ...]:
    return tuple(f"chck_{value}M" for value in _millions_for_track(track))


def required_word_targets(track: Track | str) -> tuple[int, ...]:
    return tuple(value * 1_000_000 for value in _millions_for_track(track))


def checkpoint_targets(
    word_limit: int,
    *,
    word_exposure_offset: int = 0,
) -> list[int]:
    track: Track = "strict" if word_limit > 100_000_000 else "strict-small"
    return [
        target
        for target in required_word_targets(track)
        if word_exposure_offset < target <= word_limit
    ]


def revision_for_words(words: int) -> str:
    if words <= 0 or words % 1_000_000:
        raise ValueError(
            "BabyLM checkpoint exposure must be a positive whole number of "
            f"millions, got {words}."
        )
    revision = f"chck_{words // 1_000_000}M"
    if revision not in required_revisions("strict"):
        raise ValueError(f"{revision} is not an official BabyLM revision.")
    return revision


def words_for_revision(revision: str) -> int:
    if revision not in required_revisions("strict"):
        raise ValueError(f"Unknown official BabyLM revision: {revision!r}.")
    return int(revision.removeprefix("chck_").removesuffix("M")) * 1_000_000


def infer_track(train_word_count: int | None) -> Track:
    if train_word_count == 10_000_000:
        return "strict-small"
    if train_word_count == 100_000_000:
        return "strict"
    raise ValueError(
        "Cannot infer BabyLM track: data.train_word_count must be exactly "
        "10,000,000 or 100,000,000. Pass --track explicitly."
    )


def _millions_for_track(track: Track | str) -> tuple[int, ...]:
    if track == "strict-small":
        return STRICT_SMALL_MILLIONS
    if track == "strict":
        return STRICT_MILLIONS
    raise ValueError(f"Unknown BabyLM track {track!r}.")
FORMAT_VERSION = 4
ModelWeights = Literal["ema", "raw"]


class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: str | Path,
        resolved,
        steps_per_epoch: int,
        *,
        microbatches_per_epoch: int,
        actual_train_word_count: int,
        run_word_limit: int | None = None,
        writer: bool = True,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.resolved = resolved
        self.run = resolved.config
        self.steps_per_epoch = steps_per_epoch
        self.microbatches_per_epoch = microbatches_per_epoch
        self.actual_train_word_count = actual_train_word_count
        self.word_exposure_offset = max(
            0,
            int(self.run.training.word_exposure_offset),
        )
        self.word_targets = _build_word_checkpoint_targets(
            resolved,
            steps_per_epoch,
            run_word_limit=run_word_limit,
        )
        self.next_word_checkpoint = 0
        self.writer = writer
        if self.writer:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def word_checkpoints_enabled(self) -> bool:
        return bool(self.word_targets)

    def save_if_due(self, state, progress=None) -> bool:
        saved = False
        words_seen = self.words_seen(state.step, state.microbatches_seen)
        while (
            self.next_word_checkpoint < len(self.word_targets)
            and words_seen >= self.word_targets[self.next_word_checkpoint]
        ):
            target_words = self.word_targets[self.next_word_checkpoint]
            revision = revision_for_words(target_words)
            if self.writer:
                save_checkpoint(
                    self.checkpoint_dir / "babylm_required" / f"{revision}.pt",
                    state,
                    self.resolved,
                    self.metadata(
                        state.step,
                        target_words,
                        microbatches_seen=state.microbatches_seen,
                    ),
                    kind="revision",
                    **self._progress(state),
                )
            if self.writer and progress is not None:
                progress.write(
                    f"checkpoint {revision}.pt | step {state.step} | "
                    f"words_seen: {words_seen}"
                )
            self.next_word_checkpoint += 1
            saved = True

        every = self.run.training.latest_every_steps
        if every > 0 and state.step % every == 0:
            self._save_latest(
                state,
                self.metadata(
                    state.step,
                    microbatches_seen=state.microbatches_seen,
                ),
            )
            saved = True
        return saved

    def save_final(self, state) -> None:
        metadata = self.metadata(
            state.step,
            microbatches_seen=state.microbatches_seen,
        )
        self._save_latest(state, metadata)
        if self.writer and not self.word_checkpoints_enabled:
            save_checkpoint(
                self.checkpoint_dir / "final.pt",
                state,
                self.resolved,
                metadata,
                kind="revision",
                **self._progress(state),
            )

    def restore_progress(self, state) -> None:
        words_seen = self.words_seen(state.step, state.microbatches_seen)
        self.next_word_checkpoint = sum(
            target <= words_seen for target in self.word_targets
        )

    def _save_latest(self, state, metadata: dict) -> None:
        rng_by_rank = collect_rng_states()
        if self.writer:
            save_training_checkpoint(
                self.checkpoint_dir / "latest.pt",
                state,
                self.resolved,
                metadata,
                rng_by_rank=rng_by_rank,
                **self._progress(state),
            )

    def _progress(self, state) -> dict[str, int]:
        epoch, microbatch_in_epoch = divmod(
            state.microbatches_seen,
            self.microbatches_per_epoch,
        )
        return {
            "epoch": epoch,
            "microbatch_in_epoch": microbatch_in_epoch,
        }

    def metadata(
        self,
        step: int,
        target_words: int | None = None,
        microbatches_seen: int | None = None,
    ) -> dict[str, int | float | str]:
        nominal_count = self.run.data.train_word_count
        words_seen = self.words_seen(step, microbatches_seen)
        stage_words = self._stage_words_seen_for_count(
            nominal_count or 0,
            step,
            microbatches_seen,
        )
        metadata: dict[str, int | float | str] = {
            "words_seen": words_seen,
            "nominal_words_seen": words_seen,
            "word_exposure_offset": self.word_exposure_offset,
            "elf_words_seen": stage_words,
            "steps_per_epoch": self.steps_per_epoch,
            "microbatches_per_epoch": self.microbatches_per_epoch,
        }
        if microbatches_seen is not None:
            metadata["microbatches_seen"] = microbatches_seen
        if nominal_count is not None:
            metadata.update(
                {
                    "corpus_word_count": nominal_count,
                    "nominal_corpus_word_count": nominal_count,
                    "epochs_completed": words_seen / nominal_count,
                    "elf_epochs_completed": self._stage_epochs_completed(
                        step,
                        microbatches_seen,
                    ),
                    "exposure_unit": "whitespace_words",
                }
            )
        actual_offset = self._actual_word_exposure_offset()
        metadata.update(
            {
                "actual_corpus_word_count": self.actual_train_word_count,
                "actual_word_exposure_offset": actual_offset,
                "actual_words_seen": (
                    actual_offset
                    + self._stage_words_seen_for_count(
                        self.actual_train_word_count,
                        step,
                        microbatches_seen,
                    )
                ),
                "elf_actual_words_seen": self._stage_words_seen_for_count(
                    self.actual_train_word_count,
                    step,
                    microbatches_seen,
                ),
            }
        )
        if target_words is not None:
            metadata["target_words"] = target_words
            metadata["revision_name"] = revision_for_words(target_words)
        return metadata

    def words_seen(
        self,
        step: int,
        microbatches_seen: int | None = None,
    ) -> int:
        word_count = self.run.data.train_word_count
        if word_count is None:
            return 0
        return self.word_exposure_offset + self._stage_words_seen_for_count(
            word_count,
            step,
            microbatches_seen,
        )

    def _stage_words_seen_for_count(
        self,
        word_count: int,
        step: int,
        microbatches_seen: int | None,
    ) -> int:
        if microbatches_seen is not None:
            complete_epochs, within_epoch = divmod(
                microbatches_seen,
                self.microbatches_per_epoch,
            )
            return (
                complete_epochs * word_count
                + within_epoch * word_count // self.microbatches_per_epoch
            )
        return step * word_count // max(1, self.steps_per_epoch)

    def _stage_epochs_completed(
        self,
        step: int,
        microbatches_seen: int | None,
    ) -> float:
        if microbatches_seen is not None:
            return microbatches_seen / self.microbatches_per_epoch
        return step / max(1, self.steps_per_epoch)

    def _actual_word_exposure_offset(self) -> int:
        nominal = self.run.data.train_word_count
        if not nominal:
            return self.word_exposure_offset
        return (
            self.word_exposure_offset * self.actual_train_word_count
        ) // nominal


def save_checkpoint(
    path: str | Path,
    state,
    resolved,
    metadata: dict | None = None,
    *,
    kind: str = "revision",
    epoch: int = 0,
    microbatch_in_epoch: int = 0,
) -> None:
    payload = _checkpoint_payload(
        state,
        resolved,
        metadata,
        kind=kind,
        epoch=epoch,
        microbatch_in_epoch=microbatch_in_epoch,
    )
    atomic_torch_save(payload, _prepare_path(path))


def save_training_checkpoint(
    path: str | Path,
    state,
    resolved,
    metadata: dict | None = None,
    *,
    epoch: int,
    microbatch_in_epoch: int,
    rng_by_rank: list[dict] | None = None,
) -> None:
    payload = _checkpoint_payload(
        state,
        resolved,
        metadata,
        kind="latest",
        epoch=epoch,
        microbatch_in_epoch=microbatch_in_epoch,
    )
    payload["training_state"] = {
        "optimizer": state.optimizer.state_dict(),
        "scheduler": (
            state.scheduler.state_dict() if state.scheduler is not None else None
        ),
        "ema": state.ema.state_dict() if state.ema is not None else None,
        "rng_by_rank": rng_by_rank or [capture_rng_state()],
    }
    atomic_torch_save(payload, _prepare_path(path))


def _checkpoint_payload(
    state,
    resolved,
    metadata: dict | None,
    *,
    kind: str,
    epoch: int,
    microbatch_in_epoch: int,
) -> dict:
    if kind not in {"revision", "latest"}:
        raise ValueError(f"Unknown checkpoint kind: {kind!r}.")
    model = _unwrap_model(state.model)
    raw = model.state_dict()
    ema = state.ema.model_state_dict(model) if state.ema is not None else raw
    resolved_metadata = dict(metadata or {})
    resolved_metadata.update(_training_stability_metadata(state, resolved))
    return {
        "format_version": FORMAT_VERSION,
        "kind": kind,
        "resolved_config": asdict(resolved),
        "weights": {"raw": raw, "ema": ema},
        "progress": {
            "step": int(state.step),
            "microbatches_seen": int(state.microbatches_seen),
            "epoch": int(epoch),
            "microbatch_in_epoch": int(microbatch_in_epoch),
        },
        "metadata": resolved_metadata,
    }


def load_training_checkpoint(
    path: str | Path,
    state,
    *,
    map_location: str | torch.device = "cpu",
) -> dict:
    checkpoint = _load_v4(path, map_location=map_location, kind="latest")
    training = checkpoint.get("training_state")
    if not isinstance(training, dict):
        raise ValueError("Latest checkpoint is missing training_state.")
    rng_by_rank = training.get("rng_by_rank")
    if not isinstance(rng_by_rank, list) or not rng_by_rank:
        raise ValueError(
            "Latest format-v4 checkpoint has no per-rank RNG state and cannot "
            "be resumed exactly. It remains valid for model export."
        )
    model = _unwrap_model(state.model)
    model.load_state_dict(checkpoint["weights"]["raw"])
    state.optimizer.load_state_dict(training["optimizer"])
    if state.scheduler is not None:
        if training["scheduler"] is None:
            raise ValueError("Latest checkpoint has no scheduler state.")
        state.scheduler.load_state_dict(training["scheduler"])
    if state.ema is not None:
        if training["ema"] is None:
            raise ValueError("Latest checkpoint has no EMA state.")
        state.ema.load_state_dict(training["ema"])
    state.step = int(checkpoint["progress"]["step"])
    state.microbatches_seen = int(
        checkpoint["progress"]["microbatches_seen"]
    )
    return checkpoint


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": (
            torch.cuda.get_rng_state().cpu() if torch.cuda.is_available() else None
        ),
    }


def collect_rng_states() -> list[dict]:
    local_state = capture_rng_state()
    if not (dist.is_available() and dist.is_initialized()):
        return [local_state]
    gathered: list[dict | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_state)
    return [state for state in gathered if state is not None]


def restore_rank_rng_state(
    checkpoint: dict,
    *,
    rank: int,
    world_size: int,
) -> None:
    training = checkpoint.get("training_state", {})
    rng_by_rank = training.get("rng_by_rank")
    if not isinstance(rng_by_rank, list) or not rng_by_rank:
        raise ValueError(
            "Latest format-v4 checkpoint has no per-rank RNG state and cannot "
            "be resumed exactly. It remains valid for model export."
        )
    if len(rng_by_rank) != world_size:
        raise ValueError(
            "Resume world size differs from checkpoint RNG state: "
            f"checkpoint={len(rng_by_rank)}, runtime={world_size}."
        )
    state = rng_by_rank[rank]
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise ValueError(
                "Checkpoint contains CUDA RNG state, but CUDA is unavailable."
            )
        torch.cuda.set_rng_state(cuda_state.cpu())


def load_model_weights(
    path: str | Path,
    model: torch.nn.Module,
    map_location: str | torch.device = "cpu",
    *,
    weights: ModelWeights = "ema",
) -> dict:
    checkpoint = _load_v4(path, map_location=map_location)
    model.load_state_dict(select_model_weights(checkpoint, weights=weights))
    return checkpoint


def select_model_weights(
    checkpoint: dict,
    *,
    weights: ModelWeights = "ema",
) -> dict[str, torch.Tensor]:
    if weights not in {"ema", "raw"}:
        raise ValueError(f"Unknown checkpoint weights: {weights!r}.")
    _validate_v4(checkpoint)
    return checkpoint["weights"][weights]


def _load_v4(
    path: str | Path,
    *,
    map_location: str | torch.device,
    kind: str | None = None,
) -> dict:
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    _validate_v4(checkpoint, kind=kind)
    return checkpoint


def _validate_v4(checkpoint: dict, *, kind: str | None = None) -> None:
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 4:
        raise ValueError("BabyLM-ELF requires a format-v4 checkpoint.")
    required = {"kind", "resolved_config", "weights", "progress", "metadata"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError("Checkpoint is missing: " + ", ".join(missing))
    if kind is not None and checkpoint["kind"] != kind:
        raise ValueError(
            f"Expected checkpoint kind {kind!r}, got {checkpoint['kind']!r}."
        )
    if set(checkpoint["weights"]) != {"raw", "ema"}:
        raise ValueError("Checkpoint weights must contain exactly raw and ema.")


def atomic_torch_save(checkpoint: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(checkpoint, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_path(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _unwrap_model(model):
    return getattr(model, "module", model)


def _training_stability_metadata(state, resolved) -> dict:
    ema = resolved.config.ema
    metadata = {
        "model_init_seed": int(resolved.config.seed),
        "runtime_seed_policy": "base_seed_plus_global_rank",
        "runtime_seed_rank0": int(resolved.config.seed),
    }
    if state.ema is not None:
        metadata.update(
            {
                "ema_reference_decay": float(ema.reference_decay),
                "ema_reference_steps": int(ema.reference_steps),
                "ema_resolved_decay": float(state.ema.decay),
                "ema_current_decay": float(state.ema.current_decay),
                "ema_num_updates": int(state.ema.num_updates),
                "ema_warmup": bool(state.ema.warmup),
            }
        )
    return metadata


def _build_word_checkpoint_targets(
    resolved,
    steps_per_epoch: int,
    *,
    run_word_limit: int | None = None,
) -> list[int]:
    run = resolved.config
    training = run.training
    word_count = run.data.train_word_count
    if not training.checkpoint_by_words:
        return []
    if word_count is None:
        raise ValueError(
            "data.train_word_count is required for word checkpoints."
        )
    if run_word_limit is None:
        run_word_limit = resolved.max_steps * word_count // max(1, steps_per_epoch)
    offset = max(0, training.word_exposure_offset)
    competition_limit = word_count * 10
    total_limit = offset + run_word_limit
    if total_limit > competition_limit:
        raise ValueError(
            f"Configured exposure {total_limit:,} exceeds {competition_limit:,}."
        )
    word_limit = training.checkpoint_word_limit or competition_limit
    if word_limit > competition_limit:
        raise ValueError(
            f"checkpoint_word_limit {word_limit:,} exceeds {competition_limit:,}."
        )
    return checkpoint_targets(
        min(word_limit, total_limit),
        word_exposure_offset=offset,
    )
