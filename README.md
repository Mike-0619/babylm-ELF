# BabyLM ELF

ELF-style language-model experiments for BabyLM 2026, with five Strict-Small
10M routes and two Strict 100M ELF-B routes. Four 10M AdamW routes form the
objective-controlled comparison; the fifth is an optimizer-controlled Muon
noisy-CE comparison.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use a PyTorch build compatible with the target CUDA driver.

## Prepare Data

Build and validate the canonical schema-v3 manifests, tokenizers, and
`flat_int16_le_v1` streams:

```bash
python -m src prepare \
  --config configs/10m/elf_noisy.yml --world-size 4 --staging
python -m src prepare \
  --config configs/100m/elf_noisy.yml --world-size 4 --staging
```

Data preparation and training share the same audited experiment configs:

```text
configs/10m/elf_noisy.yml
configs/100m/elf_noisy.yml
```

These canonical representatives pin each track's dataset revision and artifact
paths. The generated schema-v3 manifest is the single source of truth for the
dataset fingerprint, corpus statistics, and artifact hashes.

An optional four-rank data smoke is available:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m src smoke-data \
  --config configs/10m/elf_noisy.yml \
  --config configs/100m/elf_noisy.yml
```

Cluster-agnostic Slurm templates are available at:

```text
scripts/prepare/prepare_2026_10M.slurm
scripts/prepare/prepare_2026_100M.slurm
```

The package has one command surface:

```text
python -m src {train,prepare,smoke-data,export,train-encoder,contextuality}
```

Run `python -m src COMMAND --help` for command-specific arguments.

## Source Layout

```text
src/
├── config.py       # strict YAML parsing and resolved runs
├── modules/        # ELF model, layers, attention, and embeddings
├── training/       # objectives, loop, optimizers, checkpoints, encoder jobs
├── data/           # mmap datasets, manifests, tokenization, and preparation
└── export/         # checkpoint conversion and HF remote code
```

## Train

The five maintained 10M experiments are:

| Route | Config | Slurm |
| --- | --- | --- |
| ELF noisy-CE | `configs/10m/elf_noisy.yml` | `scripts/train/10m_elf_noisy.slurm` |
| ELF noisy-CE (empirical Muon LR) | `configs/10m/elf_noisy_muon.yml` | `scripts/train/10m_elf_noisy_muon.slurm` |
| ELF cyclic token-MLM | `configs/10m/elf_mlm_cyclic.yml` | `scripts/train/10m_elf_mlm_cyclic.slurm` |
| ELF BERT15 token-MLM | `configs/10m/elf_mlm_bert15.yml` | `scripts/train/10m_elf_mlm_bert15.slurm` |
| Standard MDLM | `configs/10m/elf_mdlm.yml` | `scripts/train/10m_elf_mdlm.slurm` |

The two 100M ELF-B experiments are:

| Route | Config | Slurm |
| --- | --- | --- |
| ELF noisy-CE | `configs/100m/elf_noisy.yml` | `scripts/train/100m_elf_noisy.slurm` |
| Standard MDLM | `configs/100m/elf_mdlm.yml` | `scripts/train/100m_elf_mdlm.slurm` |

The tracked templates contain only portable resource requests. Activate a
working Python environment, submit from the repository root, and provide
site-specific scheduling options through `sbatch`:

```bash
sbatch --partition=YOUR_PARTITION scripts/train/10m_elf_noisy.slurm
```

For a direct launch, pass the experiment config explicitly:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m src train \
  --config configs/10m/elf_mlm_cyclic.yml
```

`--config` is required; training never selects an implicit experiment.

Training logs report 50-step window averages with token-count-weighted
decoder/MDLM metrics across all ranks. ELF `loss` mixes decoder CE and flow MSE;
MDLM reports its `CE/t` NELBO. Noisy-CE accuracy, BERT15 target accuracy, and
MDLM masked-token accuracy use different corruptions and must not be compared
as the same metric.

## Resume

Resume automatically from format-v4 `latest.pt`:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m src train \
  --config configs/10m/elf_mlm_cyclic.yml \
  --resume auto
```

Revision checkpoints are lightweight evaluation artifacts and are not accepted
for training resume. `auto` starts only when the run directory is empty; it
refuses to overwrite revision artifacts when `latest.pt` is missing. An
explicit checkpoint path remains supported. Checkpoints also carry training
semantics version 2; older semantics are rejected even when the container
format is v4.

## Export

Export EMA weights for the main checkpoint:

```bash
python -m src export \
  --config configs/10m/elf_mlm_cyclic.yml
```

Export the main checkpoint plus every required exposure revision:

```bash
python -m src export \
  --config configs/10m/elf_mlm_cyclic.yml \
  --all-revisions --track strict-small
```

Use `--track strict` with a 100M config to export all 28 revisions through
`chck_1000M`.

Use `--weights raw` to export raw weights. Remote code supports `AutoModel` and
`AutoModelForMaskedLM`; Standard MDLM also exposes `generate_mdlm()`.

Scratch T5, Gaussian embeddings, encoder pretraining/contextuality, and their
format-v4 HF export remain supported interfaces. They are not additional
entries in the seven-experiment matrix.
