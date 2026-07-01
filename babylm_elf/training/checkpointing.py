from __future__ import annotations

from pathlib import Path
from dataclasses import asdict, is_dataclass
import os
import uuid

import torch


class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: str | Path,
        config,
        steps_per_epoch: int,
        microbatches_per_epoch: int | None = None,
        actual_train_word_count: int | None = None,
        run_word_limit: int | None = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config = config
        self.steps_per_epoch = steps_per_epoch
        self.microbatches_per_epoch = microbatches_per_epoch
        self.actual_train_word_count = actual_train_word_count
        self.word_exposure_offset = max(
            0,
            int(getattr(config, "word_exposure_offset", 0) or 0),
        )
        self.word_targets = _build_word_checkpoint_targets(
            config,
            steps_per_epoch,
            run_word_limit=run_word_limit,
        )
        self.next_word_checkpoint = 0
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def word_checkpoints_enabled(self) -> bool:
        return bool(self.word_targets)

    def save_if_due(self, state, progress=None) -> bool:
        saved = False
        if not self.word_targets:
            if self.config.save_every > 0 and state.step % self.config.save_every == 0:
                metadata = self.metadata(
                    state.step,
                    microbatches_seen=state.microbatches_seen,
                )
                save_checkpoint(
                    self.checkpoint_dir / f"step_{state.step}.pt",
                    state,
                    self.config,
                    metadata,
                )
                save_checkpoint(
                    self.checkpoint_dir / "latest.pt",
                    state,
                    self.config,
                    metadata,
                )
                saved = True
            return saved

        words_seen = self.words_seen(state.step, state.microbatches_seen)
        while (
            self.next_word_checkpoint < len(self.word_targets)
            and words_seen >= self.word_targets[self.next_word_checkpoint]
        ):
            target_words = self.word_targets[self.next_word_checkpoint]
            revision = _format_babylm_revision(target_words)
            save_checkpoint(
                self.checkpoint_dir / "babylm_required" / f"{revision}.pt",
                state,
                self.config,
                self.metadata(
                    state.step,
                    target_words,
                    microbatches_seen=state.microbatches_seen,
                ),
            )
            if progress is not None:
                progress.write(
                    f"checkpoint {revision}.pt | step {state.step} | "
                    f"words_seen: {words_seen}"
                )
            self.next_word_checkpoint += 1
            saved = True
        return saved

    def save_final(self, state) -> None:
        if self.word_checkpoints_enabled:
            return
        final_metadata = self.metadata(
            state.step,
            microbatches_seen=state.microbatches_seen,
        )
        save_checkpoint(
            self.checkpoint_dir / "latest.pt",
            state,
            self.config,
            final_metadata,
        )
        save_checkpoint(
            self.checkpoint_dir / "final.pt",
            state,
            self.config,
            final_metadata,
        )

    def metadata(
        self,
        step: int,
        target_words: int | None = None,
        checkpoint_type: str | None = None,
        microbatches_seen: int | None = None,
    ) -> dict[str, int | float | str]:
        words_seen = self.words_seen(step, microbatches_seen)
        metadata: dict[str, int | float | str] = {
            "checkpoint_type": checkpoint_type
            or ("babylm_word_exposure" if self.word_checkpoints_enabled else "step"),
            "words_seen": words_seen,
            "nominal_words_seen": words_seen,
            "word_exposure_offset": self.word_exposure_offset,
            "elf_words_seen": self._stage_words_seen_for_count(
                self.config.data.train_word_count or 0,
                step,
                microbatches_seen,
            ),
            "steps_per_epoch": self.steps_per_epoch,
        }
        if self.microbatches_per_epoch is not None:
            metadata["microbatches_per_epoch"] = self.microbatches_per_epoch
        if microbatches_seen is not None:
            metadata["microbatches_seen"] = microbatches_seen
        if self.config.data.train_word_count is not None:
            metadata.update(
                {
                    "corpus_word_count": self.config.data.train_word_count,
                    "nominal_corpus_word_count": self.config.data.train_word_count,
                    "epochs_completed": words_seen / self.config.data.train_word_count,
                    "elf_epochs_completed": self._stage_epochs_completed(
                        step,
                        microbatches_seen,
                    ),
                    "stage_epochs_completed": self._stage_epochs_completed(
                        step,
                        microbatches_seen,
                    ),
                    "exposure_unit": "whitespace_words",
                }
            )
        if self.actual_train_word_count is not None:
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
            metadata["revision_name"] = _format_babylm_revision(target_words)
        return metadata

    def words_seen(
        self,
        step: int,
        microbatches_seen: int | None = None,
    ) -> int:
        word_count = self.config.data.train_word_count
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
        microbatches_seen: int | None = None,
    ) -> int:
        if microbatches_seen is not None and self.microbatches_per_epoch:
            complete_epochs, within_epoch = divmod(
                microbatches_seen,
                self.microbatches_per_epoch,
            )
            return (
                complete_epochs * word_count
                + within_epoch * word_count // self.microbatches_per_epoch
            )
        return (step * word_count) // max(1, self.steps_per_epoch)

    def _stage_epochs_completed(
        self,
        step: int,
        microbatches_seen: int | None = None,
    ) -> float:
        if microbatches_seen is not None and self.microbatches_per_epoch:
            return microbatches_seen / self.microbatches_per_epoch
        return step / max(1, self.steps_per_epoch)

    def _actual_word_exposure_offset(self) -> int:
        nominal_word_count = self.config.data.train_word_count
        if not nominal_word_count or self.actual_train_word_count is None:
            return self.word_exposure_offset
        return (
            self.word_exposure_offset * self.actual_train_word_count
        ) // nominal_word_count


def save_checkpoint(
    path: str | Path,
    state,
    config,
    metadata: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata or {})
    raw_model_state = state.model.state_dict()
    ema_model_state = (
        state.ema.model_state_dict(state.model) if state.ema is not None else raw_model_state
    )
    checkpoint = {
        # Evaluation/export uses EMA, matching the paper.
        "model": ema_model_state,
        "step": state.step,
        "config": asdict(config) if is_dataclass(config) else config,
        "metadata": metadata,
    }
    _atomic_torch_save(checkpoint, path)


def _atomic_torch_save(checkpoint: dict, path: Path) -> None:
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("wb") as handle:
            torch.save(checkpoint, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_model_weights(path: str | Path, model: torch.nn.Module, map_location: str | torch.device = "cpu") -> dict:
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    weights = checkpoint.get("model", checkpoint)
    model.load_state_dict(weights)
    return checkpoint


def _build_word_checkpoint_targets(
    config,
    steps_per_epoch: int,
    run_word_limit: int | None = None,
) -> list[int]:
    if not config.checkpoint_by_words:
        return []
    if config.data.train_word_count is None:
        raise ValueError("Set data.train_word_count when checkpoint_by_words is enabled.")

    if run_word_limit is None:
        run_word_limit = (
            config.max_steps * config.data.train_word_count
        ) // max(1, steps_per_epoch)
    word_exposure_offset = max(0, int(getattr(config, "word_exposure_offset", 0) or 0))
    competition_limit = config.data.train_word_count * 10
    total_run_word_limit = word_exposure_offset + run_word_limit
    if total_run_word_limit > competition_limit:
        raise ValueError(
            "BabyLM competition runs may not exceed 10 epochs: "
            f"configured exposure is {total_run_word_limit:,} words "
            f"({word_exposure_offset:,} offset + {run_word_limit:,} ELF), limit is "
            f"{competition_limit:,} words."
        )

    word_limit = config.checkpoint_word_limit or competition_limit
    if word_limit > competition_limit:
        raise ValueError(
            "checkpoint_word_limit exceeds the BabyLM 10-epoch exposure limit: "
            f"{word_limit:,} > {competition_limit:,}."
        )
    word_limit = min(word_limit, total_run_word_limit)

    # BabyLM 2026 fast evaluation revisions:
    # 1M-9M, 10M-100M by 10M, and Strict 200M-1B by 100M.
    targets = [
        *range(1_000_000, 10_000_000, 1_000_000),
        *range(10_000_000, 100_000_000 + 1, 10_000_000),
        *range(200_000_000, word_limit + 1, 100_000_000),
    ]
    if word_limit % 100_000_000 != 0 and word_limit > 100_000_000:
        targets.append(word_limit)
    return [
        target
        for target in targets
        if word_exposure_offset < target <= word_limit
    ]


def _format_babylm_revision(words: int) -> str:
    if words % 1_000_000 != 0:
        raise ValueError(
            "BabyLM checkpoint exposure must be a whole number of millions: "
            f"{words}"
        )
    return f"chck_{words // 1_000_000}M"
