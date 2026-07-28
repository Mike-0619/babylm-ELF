#!/bin/bash

#SBATCH --job-name=prepare_2026_10M
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:45:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
python -m babylm_elf prepare \
  --config configs/10m/elf_noisy.yml \
  --world-size 4 \
  --staging
