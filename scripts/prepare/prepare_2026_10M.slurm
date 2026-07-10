#!/bin/bash

#SBATCH --job-name=prepare_2026_10M
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclude=gpu-nvidia-h100
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:45:00

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${CONDA_ROOT:-$HOME/miniforge3}/etc/profile.d/conda.sh"
conda activate babylm-elf
mkdir -p logs

python -m babylm_elf.cli.prepare_data \
  --config configs/data/2026_10M.yml \
  --world_size 4 \
  --staging
