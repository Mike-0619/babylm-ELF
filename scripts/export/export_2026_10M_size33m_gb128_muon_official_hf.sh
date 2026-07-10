#!/usr/bin/env bash
set -euo pipefail

# Export 33M gb128 Muon official-LR runs.

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

source "${CONDA_ROOT:-$HOME/miniforge3}/etc/profile.d/conda.sh"
conda activate babylm-elf

export BABYLM_ELF_EXPORT_TOKENIZER_MAX_LENGTH=512

runs=(
  "configs/2026_10M_learnable_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_bert_head_scaled_size33m_gb128_muon_official_lr5e-3_4gpu.yml|learnable_token_mlm_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_ce20_bert_head_scaled_muon_size33m_gb128_official_lr5e-3_4gpu"
  "configs/2026_10M_learnable_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_bert_head_scaled_size33m_gb128_muon_official_lr1e-2_4gpu.yml|learnable_token_mlm_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_ce20_bert_head_scaled_muon_size33m_gb128_official_lr1e-2_4gpu"
)

for item in "${runs[@]}"; do
  IFS='|' read -r config run_name <<< "$item"
  checkpoint="outputs/2026_10M/${run_name}/checkpoints/babylm_required/chck_100M.pt"
  echo "Checking ${run_name}"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing final BabyLM checkpoint: ${checkpoint}" >&2
    exit 1
  fi

  echo "Exporting ${run_name}"
  python -u -m babylm_elf.cli.export_hf \
    --config "${config}" \
    --all-revisions
done
