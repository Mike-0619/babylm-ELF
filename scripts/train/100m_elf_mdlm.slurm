#!/bin/bash

#SBATCH --job-name=elf100-mdlm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
srun torchrun --standalone --nproc_per_node=4 \
  -m babylm_elf train \
  --config configs/100m/elf_mdlm.yml \
  --resume auto
