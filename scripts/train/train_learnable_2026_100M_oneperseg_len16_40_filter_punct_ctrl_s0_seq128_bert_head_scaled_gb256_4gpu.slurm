#!/bin/bash

# 100M mainline: 96M params, gb256, one-mask-per-segment 16/40, s0 seq128.

#SBATCH --job-name=100m_l16_g256
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclude=gpu-nvidia-h100
#SBATCH --gres=gpu:nvidia:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

source "${CONDA_ROOT:-$HOME/miniforge3}/etc/profile.d/conda.sh"
conda activate babylm-elf

export BABYLM_ELF_SDPA_BACKEND=flash

mkdir -p logs

srun torchrun --standalone --nproc_per_node=4 -m babylm_elf.cli.train --config configs/2026_100M_learnable_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_bert_head_scaled_gb256_4gpu.yml
