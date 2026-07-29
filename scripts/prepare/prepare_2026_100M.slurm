#!/bin/bash

#SBATCH --job-name=prepare_2026_100M
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:10:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m src prepare \
  --config configs/100m/elf_noisy.yml \
  --world-size 4 \
  --staging
