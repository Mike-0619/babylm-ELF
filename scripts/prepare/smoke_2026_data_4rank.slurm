#!/bin/bash

#SBATCH --job-name=smoke_2026_data
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclude=gpu-nvidia-h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${CONDA_ROOT:-$HOME/miniforge3}/etc/profile.d/conda.sh"
conda activate babylm-elf
mkdir -p logs

torchrun --standalone --nproc_per_node=4 \
  -m babylm_elf smoke-data \
  --config configs/10m/elf_noisy.yml \
  --config configs/100m/elf_noisy.yml
