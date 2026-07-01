#!/usr/bin/env bash
set -euo pipefail

cd /dss/dssfs05/lwp-dss-0003/pn39je/pn39je-dss-0004/go46wuw2_2/projects/diffusion-lm/new-project/babylm-ELF

source /dss/dsshome1/07/go46wuw2/miniconda3/etc/profile.d/conda.sh
conda activate babylm-elf

python -u -m babylm_elf.cli.export_hf \
  --config configs/2026_100M_scratch_encoder.yml \
  --all-revisions
