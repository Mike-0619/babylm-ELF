#!/usr/bin/env bash
set -euo pipefail

python -m babylm_elf.cli.train_encoder --config configs/2026_10M_encoder.yml
