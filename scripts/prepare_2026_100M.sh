#!/usr/bin/env bash
set -euo pipefail

python -m babylm_elf.cli.prepare_data \
  --config configs/2026_100M_adamW.yml \
  --train_tokenizer
