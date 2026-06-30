#!/usr/bin/env bash
set -euo pipefail

python -m babylm_elf.cli.prepare_data \
  --config configs/2025.yml \
  --train_tokenizer
