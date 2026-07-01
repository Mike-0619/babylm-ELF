#!/usr/bin/env bash
set -euo pipefail

python -m babylm_elf.cli.train --config configs/2026_10M_scratch_encoder.yml
