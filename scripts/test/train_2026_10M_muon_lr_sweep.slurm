#!/bin/bash

#SBATCH --job-name=train_10M_muon_lr
#SBATCH --output=/dss/dssfs05/lwp-dss-0003/pn39je/pn39je-dss-0004/go46wuw2_2/projects/diffusion-lm/new-project/babylm-ELF/logs/%A_%a.out
#SBATCH --error=/dss/dssfs05/lwp-dss-0003/pn39je/pn39je-dss-0004/go46wuw2_2/projects/diffusion-lm/new-project/babylm-ELF/logs/%A_%a.err
#SBATCH --array=0-2

#SBATCH --partition=lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00

set -euo pipefail

cd /dss/dssfs05/lwp-dss-0003/pn39je/pn39je-dss-0004/go46wuw2_2/projects/diffusion-lm/new-project/babylm-ELF

source /dss/dsshome1/07/go46wuw2/miniconda3/etc/profile.d/conda.sh
conda activate babylm-elf

mkdir -p logs

CONFIGS=(
  "configs/test/2026_10M_muon_lr6.25e-5.yml"
  "configs/test/2026_10M_muon_lr1.25e-4.yml"
  "configs/test/2026_10M_muon_lr2.5e-4.yml"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "Array task ${SLURM_ARRAY_TASK_ID}: ${CONFIG}"

srun python -u -m babylm_elf.cli.train --config "${CONFIG}"
