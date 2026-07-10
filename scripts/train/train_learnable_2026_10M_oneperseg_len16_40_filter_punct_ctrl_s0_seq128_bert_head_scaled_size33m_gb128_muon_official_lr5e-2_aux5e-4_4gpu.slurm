#!/bin/bash

# 10M Muon aux ablation: 33M params, gb128, lr=5e-2, aux=5e-4, one-mask-per-segment 16/40.

#SBATCH --job-name=s33m5e2a5e4
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclude=gpu-nvidia-h100
#SBATCH --gres=gpu:nvidia:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

source "${CONDA_ROOT:-$HOME/miniforge3}/etc/profile.d/conda.sh"
conda activate babylm-elf

export BABYLM_ELF_SDPA_BACKEND=flash

mkdir -p logs

srun torchrun --standalone --nproc_per_node=4 -m babylm_elf.cli.train --config configs/2026_10M_learnable_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_bert_head_scaled_size33m_gb128_muon_official_lr5e-2_aux5e-4_4gpu.yml
