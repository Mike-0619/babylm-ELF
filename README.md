# BabyLM ELF

Strict-compliant BabyLM continuous embedding diffusion prototype.

The target implementation adapts ELF-style continuous diffusion to BabyLM
Strict by replacing ELF's pretrained T5 latent encoder with a from-scratch BPE
tokenizer and trainable token embeddings:

```text
token ids -> token embeddings x -> noisy z_t -> encoder denoiser
          -> clean embedding prediction -> tied vocab decoder -> token logits
```

Default training loss:

```text
L = flow_loss_weight * MSE(prediction, target)
  + decode_loss_weight * CE(token_logits, input_ids)
```

## Layout

```text
babylm_elf/cli/             prepare/train/export command-line entry points
babylm_elf/data/            tokenizer, datasets, collation
babylm_elf/modeling/        model, layers, decoder heads
babylm_elf/diffusion/       noising, schedules, target conversions
babylm_elf/training/        trainer, step, losses, optimizer, checkpoints
babylm_elf/export/          Hugging Face-compatible export helpers
configs/                    2025, 2026, and smoke YAML configs
scripts/                    thin prepare/train/export/SLURM wrappers
paper/                      paper draft, figures, tables
```

## Setup

```bash
conda activate babylm-elf
pip install -r requirements.txt
```

## Smoke Test

```bash
scripts/smoke_test.sh
```

This generates a tiny toy tokenizer/dataset and runs two ELF training steps.

Generated files are grouped by run name:

```text
data/smoke/{raw,tokenizer,tokenized}/
outputs/smoke/checkpoints/
outputs/smoke/hf/

data/babylm2025/{raw,tokenizer,tokenized}/
outputs/babylm2025/checkpoints/
outputs/babylm2025/hf/

data/babylm2026/{raw,tokenizer,tokenized}/
outputs/babylm2026/checkpoints/
outputs/babylm2026/hf/
```

## Train

Prepare year-specific Hugging Face data, tokenizer, and tokenized files first:

```bash
scripts/prepare_babylm2025.sh
scripts/prepare_babylm2026.sh
```

Then run:

```bash
scripts/train_babylm2025.sh
scripts/train_babylm2026.sh
```

or on SLURM:

```bash
sbatch scripts/slurm/train_babylm2025.slurm
sbatch scripts/slurm/train_babylm2026.slurm
```

Export Hugging Face-compatible artifacts for the official BabyLM evaluation
repositories:

```bash
scripts/export_babylm2025_hf.sh
scripts/export_babylm2026_hf.sh
```

The BabyLM 2025 Hugging Face dataset id is currently left as a placeholder, so
the 2025 prepare script should be treated as inactive unless local 2025 text is
placed under `data/babylm2025/raw/`. The 2026 config already points to
`BabyLM-community/BabyLM-2026-Strict`.
