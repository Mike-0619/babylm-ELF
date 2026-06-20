#!/usr/bin/env bash
set -euo pipefail

python -m babylm_elf.cli.prepare_data \
  --config configs/babylm2025_elf_base.yml \
  --train_tokenizer
