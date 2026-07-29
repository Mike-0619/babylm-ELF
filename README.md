# BabyLM ELF

ELF and masked diffusion language-model experiments for BabyLM 2026.
The repository contains five Strict-Small 10M runs and two Strict 100M runs.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install a PyTorch build that matches the CUDA driver on the target machine.

## Experiments

| Track | Route | Optimizer | Config | Slurm |
| --- | --- | --- | --- | --- |
| 10M | ELF noisy-CE | AdamW | `configs/10m/elf_noisy.yml` | `scripts/train/10m_elf_noisy.slurm` |
| 10M | ELF noisy-CE | Muon | `configs/10m/elf_noisy_muon.yml` | `scripts/train/10m_elf_noisy_muon.slurm` |
| 10M | ELF cyclic MLM | AdamW | `configs/10m/elf_mlm_cyclic.yml` | `scripts/train/10m_elf_mlm_cyclic.slurm` |
| 10M | ELF BERT15 MLM | AdamW | `configs/10m/elf_mlm_bert15.yml` | `scripts/train/10m_elf_mlm_bert15.slurm` |
| 10M | Standard MDLM | AdamW | `configs/10m/elf_mdlm.yml` | `scripts/train/10m_elf_mdlm.slurm` |
| 100M | ELF noisy-CE | AdamW | `configs/100m/elf_noisy.yml` | `scripts/train/100m_elf_noisy.slurm` |
| 100M | Standard MDLM | AdamW | `configs/100m/elf_mdlm.yml` | `scripts/train/100m_elf_mdlm.slurm` |

The four 10M AdamW runs are the main objective comparison. The 10M Muon run
is an optimizer control.

## Data

Prepare the 10M and 100M token streams:

```bash
python -m src prepare \
  --config configs/10m/elf_noisy.yml \
  --world-size 4 \
  --staging

python -m src prepare \
  --config configs/100m/elf_noisy.yml \
  --world-size 4 \
  --staging
```

The generated schema-v3 manifests record corpus statistics and artifact
hashes. To check both streams from four ranks:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m src smoke-data \
  --config configs/10m/elf_noisy.yml \
  --config configs/100m/elf_noisy.yml
```

CPU Slurm templates are available under `scripts/prepare/`.

## Training

Run an experiment directly:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m src train \
  --config configs/10m/elf_mlm_cyclic.yml
```

Or submit a Slurm template from the repository root:

```bash
sbatch --partition=YOUR_PARTITION scripts/train/10m_elf_noisy.slurm
```

Resume from `latest.pt`:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m src train \
  --config configs/10m/elf_mlm_cyclic.yml \
  --resume auto
```

Revision checkpoints are for evaluation and cannot be used for training
resume. Checkpoints use format v4 and training semantics version 2.

ELF loss combines flow MSE and decoder CE. MDLM reports its `CE/t` NELBO, so
loss and accuracy values from different objectives are not directly
comparable.

## Export

Export EMA weights from the latest exposure checkpoint:

```bash
python -m src export \
  --config configs/10m/elf_mlm_cyclic.yml
```

Export the main model and all required BabyLM revisions:

```bash
python -m src export \
  --config configs/10m/elf_mlm_cyclic.yml \
  --all-revisions \
  --track strict-small
```

Use `--track strict` for a 100M run and `--weights raw` to export raw weights.
Exports support `AutoModel` and `AutoModelForMaskedLM`; MDLM exports also
provide `generate_mdlm()`.

## Layout

```text
src/
├── config.py
├── data/
├── modules/
├── training/
└── export/
```

Run `python -m src --help` for the full command list. Scratch T5, Gaussian
embeddings, encoder pretraining, and contextuality tools remain available.
