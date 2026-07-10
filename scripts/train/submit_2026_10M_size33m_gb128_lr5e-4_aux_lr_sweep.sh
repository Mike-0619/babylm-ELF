#!/bin/bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

sbatch scripts/train/train_learnable_2026_10M_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_bert_head_scaled_size33m_gb128_lr5e-4_aux2e-4_4gpu.slurm
sbatch scripts/train/train_learnable_2026_10M_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_bert_head_scaled_size33m_gb128_lr5e-4_aux3e-4_4gpu.slurm
