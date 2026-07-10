from __future__ import annotations

from typing import Literal


Track = Literal["strict-small", "strict"]

OFFICIAL_EVALUATOR_REPOSITORY = "https://github.com/babylm-org/babylm-eval"
OFFICIAL_EVALUATOR_COMMIT = "3d57ddc8c40ee795c0b5e41b3a20251a9457a593"

STRICT_SMALL_MILLIONS = (*range(1, 10), *range(10, 101, 10))
STRICT_MILLIONS = (*STRICT_SMALL_MILLIONS, *range(200, 1001, 100))

HF_ARTIFACT_FILES = frozenset(
    {
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "configuration_babylm_elf.py",
        "modeling_babylm_elf.py",
        "modeling_core.py",
        "layers.py",
        "codebook.py",
        "positions.py",
        "mask_latent.py",
    }
)

FULL_TASKS = frozenset(
    {
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
)
FAST_TASKS = frozenset(
    {
        "blimp",
        "blimp_supplement",
        "ewok",
        "entity_tracking_filtered",
        "global_piqa_parallel",
        "global_piqa_nonparallel",
        "reading",
    }
)
GLUE_SAMPLE_COUNTS = {
    "boolq": 1_635,
    "mnli": 4_908,
    "mrpc": 204,
    "multirc": 2_424,
    "qqp": 20_215,
    "rte": 139,
    "wsc": 52,
}
FULL_PREDICTION_COUNTS = {
    "blimp": 59_875,
    "blimp_supplement": 5_218,
    "ewok": 7_618,
    "entity_tracking_filtered": 6_780,
    "comps": 91_028,
    "global_piqa_parallel": 103,
    "global_piqa_nonparallel": 100,
    "reading": 1_726,
}
AOA_SAMPLES_PER_REVISION = 8_005


def required_revisions(track: Track | str) -> tuple[str, ...]:
    millions = _millions_for_track(track)
    return tuple(f"chck_{value}M" for value in millions)


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
