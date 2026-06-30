# BabyLM ELF

BabyLM Strict adaptation of Embedded Language Flows (ELF).

The target implementation adapts ELF-style continuous diffusion to BabyLM
Strict by replacing ELF's pretrained T5 latent encoder with a from-scratch BPE
tokenizer and trainable token embeddings:

```text
token ids -> 512-d learnable embeddings -> 128-d bottleneck -> ELF-B
          -> continuous x-prediction / final-step token decoding
```

The implementation follows the paper's mixed-batch objective:

```text
80% of examples: logit-normal flow matching with velocity-equivalent MSE
20% of examples: t=1 token-level corruption with cross-entropy decoding
```

It includes RMSNorm, RoPE, qk-norm, in-context time/CFG/mode tokens,
50% self-conditioning, training-time self-conditioning CFG, Muon with
auxiliary Nesterov-Adam, 0.5-epoch warmup, and EMA checkpoints.

The intentional departure from the paper's main OpenWebText experiment is the
embedding encoder. BabyLM Strict cannot use pretrained T5-small, so this
project uses the paper's learnable tied-embedding ablation rather than the
paper's default pretrained contextual T5 latents. The single-GPU configs use
effective batch 32 and linearly scale Muon's learning rate from `0.002` to
`0.000125`. Learnable token embeddings are normalized to unit RMS before flow
corruption so their scale is compatible with the configured noise scales.

The exported Hugging Face model has two inference surfaces. Official BabyLM
evaluation should use the MLM-compatible `forward()` adapter, which scores
masked positions with continuous-noise pseudo-likelihood. Open-ended
`generate()` runs ELF-style ODE/SDE sampling from Gaussian noise and is meant
for diagnostics and qualitative examples, not as a replacement for the BabyLM
official evaluation pipeline. Its default diagnostic settings are 64-step SDE,
logit-normal time schedule, self-conditioning CFG scale 3, and gamma 1.

The main model must not use pretrained T5/BERT/GPT/RoBERTa weights, pretrained
T5 tokenizers, ELF pretrained checkpoints, or off-the-shelf learned language
tools. Any learned ancillary language model or tool would count toward the
BabyLM word-exposure budget.

## Layout

```text
babylm_elf/cli/             prepare/train/export command-line entry points
babylm_elf/data/            tokenizer, datasets, collation
babylm_elf/modeling/        ELF-B model and layers
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

data/2026_100M/{raw,tokenizer,tokenized}/
outputs/2026_100M/{adamW,muon}/checkpoints/
outputs/2026_100M/{adamW,muon}/hf/

data/2026_10M/{raw,tokenizer,tokenized}/
outputs/2026_10M/{adamW,muon}/checkpoints/
outputs/2026_10M/{adamW,muon}/hf/
```

## Train

Prepare year-specific Hugging Face data, tokenizer, and tokenized files first:

```bash
scripts/prepare_2025.sh
scripts/prepare_2026_100M.sh
scripts/prepare_2026_10M.sh
```

Then run:

```bash
scripts/train_2025.sh
scripts/train_2026_100M.sh
scripts/train_2026_10M.sh
```

or on SLURM:

```bash
sbatch scripts/slurm/train_2025.slurm
sbatch scripts/slurm/train_2026_100M_adamW.slurm
sbatch scripts/slurm/train_2026_10M_adamW.slurm
```

Official `chck_*M.pt` files are compact EMA model checkpoints. Checkpoint
writes use a temporary file followed by atomic replacement, so an interrupted
save cannot leave a partially written checkpoint at the final path.

The official 2026 configs do not use evaluation data as training validation.
Choose Muon learning rates by comparing BabyLM fast-evaluation results at the
same word-exposure checkpoint, rather than by training loss alone.

Export Hugging Face-compatible artifacts for the official BabyLM evaluation
repositories:

```bash
scripts/export_2025_hf.sh
scripts/export_2026_100M_hf.sh
scripts/export_2026_10M_hf.sh
```

The BabyLM 2025 Hugging Face dataset id is currently left as a placeholder, so
the 2025 prepare script should be treated as inactive unless local 2025 text is
placed under `data/babylm2025/raw/`. The 2026 100M config points to
`BabyLM-community/BabyLM-2026-Strict`; the 2026 10M config points to
`BabyLM-community/BabyLM-2026-Strict-Small`.
