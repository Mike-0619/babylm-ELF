#!/bin/bash

#SBATCH --job-name=smoke_2026_data
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
torchrun --standalone --nproc_per_node=4 \
  -m babylm_elf smoke-data \
  --config configs/10m/elf_noisy.yml \
  --config configs/100m/elf_noisy.yml
