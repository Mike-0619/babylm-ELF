#!/bin/bash

# Baseline experiment: Standard MDLM objective with the ELF-S backbone.

#SBATCH --job-name=elf10-mdlm
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclude=gpu-nvidia-h100
#SBATCH --gres=gpu:nvidia:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${CONDA_ROOT:-$HOME/miniforge3}/etc/profile.d/conda.sh"
conda activate babylm-elf

export BABYLM_ELF_SDPA_BACKEND=flash
mkdir -p logs

srun torchrun --standalone --nproc_per_node=4 \
  -m babylm_elf train \
  --config configs/10m/elf_mdlm.yml \
  --resume auto
