from __future__ import annotations

import argparse
from pathlib import Path

from babylm_elf.data.pipeline import (
    build_dataset,
    prepare_plan,
    promote_staging_root,
    reset_staging_root,
    staged_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one canonical BabyLM 2026 official-data route."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument(
        "--staging",
        action="store_true",
        help="Build beside the canonical data root and promote only after audit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.world_size <= 0:
        raise ValueError("--world_size must be positive.")
    canonical_plan = prepare_plan(args.config, args.world_size)
    plan = staged_plan(canonical_plan) if args.staging else canonical_plan
    if args.staging:
        reset_staging_root(plan)
    build_dataset(plan)
    if args.staging:
        promote_staging_root(plan)


if __name__ == "__main__":
    main()
