from __future__ import annotations

import argparse
from pathlib import Path

import torch

from babylm_elf.config import load_config
from babylm_elf.export.convert_checkpoint import export_checkpoint_to_hf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export BabyLM-ELF checkpoint to HF-style artifacts.")
    parser.add_argument("--config", type=Path, default=Path("configs/2026_100M_adamW.yml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument(
        "--all-revisions",
        action="store_true",
        help="Export <run_name>_hf plus every BabyLM chck_*M checkpoint into revision directories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_dir = Path(config.output_dir) / config.name
    if args.all_revisions:
        if args.checkpoint is not None:
            raise ValueError("--checkpoint cannot be combined with --all-revisions.")
        output_root = args.output_dir or run_dir / "hf_revisions"
        export_all_revisions(run_dir, output_root, config)
        return

    checkpoint = args.checkpoint or default_checkpoint(run_dir)
    output_dir = args.output_dir or run_dir / "hf"
    export_checkpoint_to_hf(checkpoint, output_dir, config)


def export_all_revisions(run_dir: Path, output_root: Path, config) -> None:
    final_checkpoint = run_dir / "checkpoints" / "final.pt"
    main_checkpoint = (
        final_checkpoint if final_checkpoint.exists() else default_checkpoint(run_dir)
    )
    export_checkpoint_to_hf(
        main_checkpoint,
        output_root / _main_export_name(config),
        config,
    )
    required_dir = run_dir / "checkpoints" / "babylm_required"
    checkpoints = sorted(required_dir.glob("chck_*M.pt"), key=_checkpoint_exposure)
    if not checkpoints:
        raise FileNotFoundError(f"No chck_*M.pt checkpoints found under {required_dir}")
    for checkpoint in checkpoints:
        export_checkpoint_to_hf(
            checkpoint,
            output_root / checkpoint.stem,
            config,
        )


def default_checkpoint(run_dir: Path) -> Path:
    required_dir = run_dir / "checkpoints" / "babylm_required"
    required_checkpoints = [
        *required_dir.glob("chck_*.pt"),
        # Backward compatibility for runs produced before the BabyLM revision
        # naming was applied directly to local checkpoint files.
        *required_dir.glob("words_*.pt"),
    ]
    if required_checkpoints:
        return max(required_checkpoints, key=_checkpoint_exposure)

    final_checkpoint = run_dir / "checkpoints" / "final.pt"
    if final_checkpoint.exists():
        return final_checkpoint
    latest_checkpoint = run_dir / "checkpoints" / "latest.pt"
    if latest_checkpoint.exists():
        return latest_checkpoint
    raise FileNotFoundError(f"No checkpoint found under {run_dir / 'checkpoints'}")


def _checkpoint_exposure(path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {}
    return int(metadata.get("target_words", metadata.get("words_seen", 0)))


def _main_export_name(config) -> str:
    name = getattr(config, "name", "model") or "model"
    return f"{name}_hf"


if __name__ == "__main__":
    main()
