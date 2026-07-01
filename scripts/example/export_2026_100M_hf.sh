#!/usr/bin/env bash
set -euo pipefail

python -m babylm_elf.cli.export_hf \
  --config configs/2026_100M_scratch_encoder.yml \
  --all-revisions
