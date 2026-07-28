# BabyLM ELF

BabyLM 2026 ELF experiments with five Strict-Small 10M routes and two Strict
100M ELF-B routes. The official source tree at `../ELF` is a read-only
reference; all BabyLM code and configuration live here. Four AdamW routes form
the objective-controlled main comparison; the fifth is an optimizer-controlled
Muon noisy-CE comparison.

The complete model, objective, optimizer, exposure, checkpoint, and adapter
specification is in [project.md](project.md).

## Install

```bash
conda activate babylm-elf
pip install -r requirements.txt
```

Install a PyTorch build compatible with the cluster's CUDA driver.

## Prepare Data

Build and validate the canonical schema-v3 manifests, tokenizers, and
`flat_int16_le_v1` streams:

```bash
python -m babylm_elf prepare \
  --config configs/10m/elf_noisy.yml --world-size 4 --staging
python -m babylm_elf prepare \
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
  -m babylm_elf smoke-data \
  --config configs/10m/elf_noisy.yml \
  --config configs/100m/elf_noisy.yml
```

The package has one command surface:

```text
python -m babylm_elf {train,prepare,smoke-data,export,train-encoder,contextuality}
```

Run `python -m babylm_elf COMMAND --help` for command-specific arguments.

## Source Layout

```text
babylm_elf/
├── config.py       # strict YAML parsing and resolved runs
├── modules/        # ELF model, layers, attention, and embeddings
├── training/       # objectives, loop, optimizers, checkpoints, encoder jobs
├── data/           # mmap datasets, manifests, tokenization, and preparation
└── export/         # checkpoint conversion and HF remote code
```

## Train

The five maintained 10M experiments are:

| Route | Config |
| --- | --- |
| ELF noisy-CE | `configs/10m/elf_noisy.yml` |
| ELF noisy-CE (empirical Muon LR) | `configs/10m/elf_noisy_muon.yml` |
| ELF cyclic token-MLM | `configs/10m/elf_mlm_cyclic.yml` |
| ELF BERT15 token-MLM | `configs/10m/elf_mlm_bert15.yml` |
| Standard MDLM | `configs/10m/elf_mdlm.yml` |

The two 100M ELF-B experiments are:

| Route | Config |
| --- | --- |
| ELF noisy-CE | `configs/100m/elf_noisy.yml` |
| Standard MDLM | `configs/100m/elf_mdlm.yml` |

Cluster submission scripts are intentionally local and ignored under
`scripts/`; they may contain site-specific partitions, hosts, and environment
paths without entering Git history.

For a direct launch, pass the experiment config explicitly:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m babylm_elf train \
  --config configs/10m/elf_mlm_cyclic.yml
```

`--config` is required; training never selects an implicit experiment.

## Resume

Resume automatically from format-v4 `latest.pt`:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m babylm_elf train \
  --config configs/10m/elf_mlm_cyclic.yml \
  --resume auto
```

Revision checkpoints are lightweight evaluation artifacts and are not accepted
for training resume. `auto` starts only when the run directory is empty; it
refuses to overwrite revision artifacts when `latest.pt` is missing. An
explicit checkpoint path remains supported.

## Export

Export EMA weights for the main checkpoint:

```bash
python -m babylm_elf export \
  --config configs/10m/elf_mlm_cyclic.yml
```

Export the main checkpoint plus every required exposure revision:

```bash
python -m babylm_elf export \
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
