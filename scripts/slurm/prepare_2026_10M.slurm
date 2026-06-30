#!/bin/bash

#SBATCH --job-name=prepare_2026_10M
#SBATCH --output=/dss/dssfs05/lwp-dss-0003/pn39je/pn39je-dss-0004/go46wuw2_2/projects/diffusion-lm/new-project/babylm-ELF/logs/%j.out
#SBATCH --error=/dss/dssfs05/lwp-dss-0003/pn39je/pn39je-dss-0004/go46wuw2_2/projects/diffusion-lm/new-project/babylm-ELF/logs/%j.err

#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=00:10:00


cd /dss/dssfs05/lwp-dss-0003/pn39je/pn39je-dss-0004/go46wuw2_2/projects/diffusion-lm/new-project/babylm-ELF

source /dss/dsshome1/07/go46wuw2/miniconda3/etc/profile.d/conda.sh
conda activate babylm-elf

mkdir -p logs

python -m babylm_elf.cli.prepare_data \
  --config configs/2026_10M_adamW.yml \
  --train_tokenizer
