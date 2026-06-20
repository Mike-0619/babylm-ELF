from __future__ import annotations

import argparse
from pathlib import Path

from babylm_elf.config import load_config
from babylm_elf.export.convert_checkpoint import export_checkpoint_to_hf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export BabyLM-ELF checkpoint to HF-style artifacts.")
    parser.add_argument("--config", type=Path, default=Path("configs/babylm2026_elf_base.yml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_dir = Path(config.output_dir) / config.name
    checkpoint = args.checkpoint or default_checkpoint(run_dir)
    output_dir = args.output_dir or run_dir / "hf"
    export_checkpoint_to_hf(checkpoint, output_dir, config)


def default_checkpoint(run_dir: Path) -> Path:
    final_checkpoint = run_dir / "checkpoints" / "final.pt"
    if final_checkpoint.exists():
        return final_checkpoint
    return run_dir / "checkpoints" / "latest.pt"


if __name__ == "__main__":
    main()
