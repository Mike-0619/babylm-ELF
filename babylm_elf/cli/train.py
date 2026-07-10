from __future__ import annotations

import argparse
from pathlib import Path

from babylm_elf.config import load_config
from babylm_elf.training.trainer import train_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BabyLM-ELF.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/"
            "2026_10M_learnable_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_bert_head_scaled_size33m_gb128_adamw_lr4e-4_4gpu.yml"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_from_config(config)


if __name__ == "__main__":
    main()
