from __future__ import annotations

import argparse
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m babylm_elf")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Train an ELF experiment.")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument(
        "--resume",
        metavar="auto|PATH",
        help="Resume latest.pt automatically or load an explicit checkpoint.",
    )

    prepare = commands.add_parser("prepare", help="Prepare canonical data.")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--world-size", type=int, default=4)
    prepare.add_argument("--staging", action="store_true")

    smoke = commands.add_parser(
        "smoke-data",
        help="Read canonical mmap data from every torchrun rank.",
    )
    smoke.add_argument("--config", type=Path, action="append", required=True)

    export = commands.add_parser("export", help="Export checkpoints to HF.")
    export.add_argument("--config", type=Path, required=True)
    export.add_argument("--checkpoint", type=Path)
    export.add_argument("--output-dir", type=Path)
    export.add_argument("--weights", choices=("ema", "raw"), default="ema")
    export.add_argument("--all-revisions", action="store_true")
    export.add_argument("--track", choices=("strict-small", "strict"))

    train_encoder = commands.add_parser(
        "train-encoder",
        help="Train a scratch T5 encoder.",
    )
    train_encoder.add_argument("--config", type=Path, required=True)

    contextuality = commands.add_parser(
        "contextuality",
        help="Inspect scratch-encoder contextuality.",
    )
    contextuality.add_argument("--config", type=Path, required=True)
    contextuality.add_argument("--checkpoint", type=Path)
    contextuality.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "train":
        from babylm_elf.config import load_config
        from babylm_elf.training.train import train_from_config

        train_from_config(load_config(args.config), resume_path=args.resume)
    elif args.command == "prepare":
        from babylm_elf.data.prepare import prepare_from_config

        prepare_from_config(args.config, args.world_size, staging=args.staging)
    elif args.command == "smoke-data":
        from babylm_elf.data.prepare import smoke_data

        smoke_data(args.config)
    elif args.command == "export":
        from babylm_elf.export.hf import export_from_config

        export_from_config(
            args.config,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            weights=args.weights,
            all_revisions=args.all_revisions,
            track=args.track,
        )
    elif args.command == "train-encoder":
        from babylm_elf.config import load_encoder_config
        from babylm_elf.training.encoder import train_encoder_from_config

        train_encoder_from_config(load_encoder_config(args.config))
    elif args.command == "contextuality":
        from babylm_elf.training.encoder import run_contextuality

        run_contextuality(
            args.config,
            checkpoint=args.checkpoint,
            device_name=args.device,
        )


if __name__ == "__main__":
    main()
