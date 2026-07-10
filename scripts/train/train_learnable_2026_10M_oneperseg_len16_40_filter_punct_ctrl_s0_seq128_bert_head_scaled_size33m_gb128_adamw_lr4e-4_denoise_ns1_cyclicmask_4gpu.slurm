#!/bin/bash

# 10M AdamW mainline ablation: lr=4e-4, Step10 cyclic MLM masking,
# learned mask latent, denoiser_noise_scale=1.0.

#SBATCH --job-name=10maw4_cns1
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

srun torchrun --standalone --nproc_per_node=4 -m babylm_elf.cli.train --config configs/2026_10M_learnable_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_bert_head_scaled_size33m_gb128_adamw_lr4e-4_denoise_ns1_cyclicmask_4gpu.yml
