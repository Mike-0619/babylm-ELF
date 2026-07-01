#!/usr/bin/env bash
set -euo pipefail

rm -rf data/smoke outputs/smoke
mkdir -p data/smoke/raw data/smoke/tokenizer data/smoke/tokenized
printf '%s\n\n%s\n\n%s\n\n%s\n' \
  "the child sees the red ball" \
  "a small dog runs through the garden" \
  "children learn words from simple stories" \
  "the bright sun warms the little house" \
  > data/smoke/raw/train.txt
printf '%s\n\n%s\n' \
  "a friend brings milk and bread" \
  "the parent reads a book at night" \
  > data/smoke/raw/valid.txt

python -m babylm_elf.cli.prepare_data \
  --config configs/smoke.yml \
  --train_tokenizer

python -m babylm_elf.cli.train --config configs/smoke.yml
