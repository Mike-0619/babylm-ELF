#!/bin/bash

#SBATCH --job-name=elf10-mdlm
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

export OMP_NUM_THREADS=2

cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
srun torchrun --standalone --nproc_per_node=4 \
  -m src train \
  --config configs/10m/elf_mdlm.yml \
  --resume auto
