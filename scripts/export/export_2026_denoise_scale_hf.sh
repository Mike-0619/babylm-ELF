#!/usr/bin/env bash
set -euo pipefail

# Export the maintained ns1 cyclic mainline checkpoints.

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

source "${CONDA_ROOT:-$HOME/miniforge3}/etc/profile.d/conda.sh"
conda activate babylm-elf

export BABYLM_ELF_EXPORT_TOKENIZER_MAX_LENGTH=512

runs=(
  "configs/2026_10M_learnable_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_bert_head_scaled_size33m_gb128_adamw_lr4e-4_denoise_ns1_cyclicmask_4gpu.yml|outputs/2026_10M/learnable_token_mlm_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_ce20_bert_head_scaled_size33m_gb128_adamw_lr4e-4_denoise_ns1_cyclicmask_4gpu/checkpoints/babylm_required/chck_100M.pt"
  "configs/2026_100M_learnable_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_bert_head_scaled_gb256_denoise_ns1_cyclicmask_4gpu.yml|outputs/2026_100M/learnable_token_mlm_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_ce20_bert_head_scaled_muon_gb256_denoise_ns1_cyclicmask_4gpu/checkpoints/babylm_required/chck_1000M.pt"
)

for item in "${runs[@]}"; do
  IFS='|' read -r config final_checkpoint <<< "$item"
  echo "Checking ${final_checkpoint}"
  if [[ ! -f "${final_checkpoint}" ]]; then
    echo "Missing final BabyLM checkpoint: ${final_checkpoint}" >&2
    exit 1
  fi

  echo "Exporting ${config}"
  python -u -m babylm_elf.cli.export_hf \
    --config "${config}" \
    --all-revisions
done
