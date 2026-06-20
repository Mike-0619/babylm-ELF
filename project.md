# BabyLM 100M: ELF-style Continuous Diffusion

## 1. Project Goal

本项目目标是实现一个合规的 **BabyLM 100M ELF-style continuous diffusion language model**。核心实验是把 ELF-style continuous embedding diffusion 改造成 BabyLM Strict 约束下可训练、可导出、可评测的 from-scratch 语言模型。

核心想法：

```text
GPT-2 baseline: token -> next token
Masked DLM:     token -> masked token -> token
Our model:      token -> embedding -> noisy embedding -> clean embedding -> token
```

研究问题：

```text
在 BabyLM 100M 数据限制下，continuous embedding diffusion
是否比 autoregressive GPT-2 和 masked discrete diffusion 更有效？
```

代码主目录：

```text
new-project/babylm-ELF/
```

This repository is a training codebase. It prepares BabyLM data, trains BabyLM-ELF checkpoints, and exports Hugging Face-compatible artifacts for the official BabyLM evaluation repositories. It does **not** implement the full BabyLM evaluation pipeline internally.

## 2. Benchmark Settings

最终论文/报告分成两个不可混用的 benchmark 设置。GPT-2 与 masked-DLM 结果只引用官方/论文 reported results；本项目只训练 BabyLM-ELF。

Important rule:

```text
Do not directly compare 2025 scores and 2026 scores as if they were the same benchmark.
Report each year with its matching data and evaluation pipeline.
```

### Comparison A: BabyLM 2025 Setting

```text
data:       BabyLM 2025 English 100M
evaluation: BabyLM 2025 evaluation pipeline/tasks
track:      100M / strict-style English setting
```

| Model | Training Objective | Data | Evaluation | Role |
| --- | --- | --- | --- | --- |
| 2025 BabyLM GPT-2 100M official baseline | autoregressive next-token prediction | BabyLM 2025 100M | BabyLM 2025 eval | Reported official baseline |
| 2025 BabyLM masked DLM winning paper | masked discrete diffusion / masked token recovery | BabyLM 2025 100M | BabyLM 2025 eval | Reported prior winner |
| BabyLM-ELF 2025 | continuous embedding diffusion / ELF-style flow | BabyLM 2025 100M | BabyLM 2025 eval | Our experiment |

### Comparison B: BabyLM 2026 Setting

```text
data:       BabyLM 2026 English Strict 100M
evaluation: BabyLM 2026 evaluation pipeline/tasks
track:      English Strict 100M
```

| Model | Training Objective | Data | Evaluation | Role |
| --- | --- | --- | --- | --- |
| 2026 BabyLM GPT-2 100M official baseline | autoregressive next-token prediction | BabyLM 2026 100M | BabyLM 2026 eval | Reported official baseline |
| BabyLM-ELF 2026 | continuous embedding diffusion / ELF-style flow | BabyLM 2026 100M | BabyLM 2026 eval | Our experiment / challenge model |

Baseline result sources for paper comparison:

```text
2025 GPT-2 official baseline: use reported BabyLM 2025 baseline results.
2025 masked-DLM winner: use reported results from the 2025 winning paper.
2026 GPT-2 official baseline: use reported BabyLM 2026 official baseline results.
```

## 3. Compliance and Data Rules

两个 ELF run 使用同一个模型架构，但分别使用对应年份的数据、tokenizer、checkpoint 和官方 evaluation repo。

```text
2025 run:
  data: BabyLM 2025 English 100M
  tokenizer: trained from scratch on BabyLM 2025 only
  model: initialized from scratch

2026 run:
  data: BabyLM 2026 English Strict 100M
  tokenizer: trained from scratch on BabyLM 2026 only
  model: initialized from scratch
```

Both runs follow the same safe route:

```text
BabyLM data for that year only
  -> tokenizer trained from scratch on that year's data
  -> model initialized from scratch
  -> ELF-style continuous diffusion objective
  -> <= 10 epochs
  -> Hugging Face-compatible export for official evaluation
```

Not allowed in the main model:

```text
pretrained T5 encoder
pretrained T5 tokenizer
pretrained BERT/GPT/RoBERTa weights
off-the-shelf parser / tagger / reranker / sentence embedder
ELF_PyTorch pretrained checkpoints
```

Current code path:

```text
Hugging Face BabyLM dataset
  -> local raw text cache under data/<run>/raw/
  -> babylm_elf.cli.prepare_data
  -> year-specific BPE tokenizer under data/<run>/tokenizer/
  -> torch-saved tokenized documents under data/<run>/tokenized/
```

Current status:

```text
2026: uses BabyLM-community/BabyLM-2026-Strict from Hugging Face.
2025: HF dataset id is currently unavailable; config keeps a placeholder and
      should not be used for HF download until the id is confirmed.
```

2026 dataset reference:

```python
from datasets import load_dataset

ds = load_dataset("BabyLM-community/BabyLM-2026-Strict")
```

Dataset page:

```text
https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict
```

## 4. Model Architecture

BabyLM-ELF is a from-scratch, Strict-compliant version of ELF-style continuous embedding diffusion. The architecture is shared by the 2025 and 2026 runs; only data, tokenizer, checkpoint path, and external evaluation repo differ.

```text
BabyLM text
  -> year-specific BPE tokenizer trained from scratch
  -> token ids
  -> trainable token embeddings x0
  -> continuous Gaussian noising z_t
  -> time-conditioned denoising Transformer
  -> predicted clean embedding / velocity / epsilon
  -> tied vocabulary projection head
  -> token logits
```

Initial objective:

```text
L = flow_loss_weight * L_flow + decode_loss_weight * L_ce

default:
  flow_loss_weight = 1.0
  decode_loss_weight = 0.25
  prediction_type = x0
  time_schedule = linear
  noise_scale = 1.0
```

Core training step:

```text
input_ids
  -> x0 = token_embedding(input_ids)
  -> t = sample_timestep()
  -> eps = sample_gaussian_noise()
  -> z_t = add_noise(x0, eps, t)
  -> prediction, decoder_logits = model(z_t, t, attention_mask)
  -> L_flow = flow_loss(prediction, target)
  -> L_ce = decode_ce(decoder_logits, input_ids)
  -> L = L_flow + decode_loss_weight * L_ce
```

Key difference from original ELF:

```text
ELF_PyTorch:
  text -> pretrained/frozen text encoder latent x0 -> flow model -> decoder

BabyLM-ELF:
  text -> from-scratch BPE -> trainable token embedding x0 -> flow model -> tied vocab decoder
```

## 5. Component Sources

| Component | Choice | Source / Inspiration | From Scratch? |
| --- | --- | --- | --- |
| Training data | BabyLM 2025 100M or BabyLM 2026 Strict 100M | BabyLM official | N/A |
| Tokenizer | Separate BPE tokenizer trained on each year's BabyLM data | `babylm-diffusion` tokenizer scripts | Yes |
| Embedding table | Trainable token embeddings | BERT / masked-DLM embedding form | Yes |
| Backbone | LTG-BERT-style encoder-only Transformer | `babylm-diffusion` / LTG-BERT | Yes |
| Time conditioning | timestep conditioning | ELF + diffusion code | Yes |
| Noise space | continuous embedding space | ELF | N/A |
| Main objective | flow / denoising loss | ELF | N/A |
| Decoder objective | continuous-to-discrete decoding | ELF-style reconstruction | Yes |
| Decoder head | tied vocab projection / MLM-style head | `babylm-diffusion` implementation form | Yes |
| Training loop | checkpoint, EMA, distributed training | `babylm-diffusion` infrastructure | N/A |

Source-use policy:

```text
Borrow from ELF_PyTorch:
  continuous noising, timestep-conditioned denoising, flow/denoising loss,
  decoder branch concept.

Reuse/adapt from babylm-diffusion:
  tokenizer/data scripts, checkpoint/EMA/distributed training utilities,
  LTG-BERT-style implementation patterns, tied vocab projection, SLURM scaffolding.

Do not use:
  pretrained T5 encoder/tokenizer, pretrained ELF checkpoints,
  discrete masked-token corruption as the main objective.
```

## 6. Training-Only Project Architecture

The design follows two references:

```text
ELF_PyTorch:
  continuous embedding noising
  timestep-conditioned denoising Transformer
  flow / denoising loss
  decoder branch for token recovery

babylm-diffusion:
  BabyLM tokenizer and tokenized-data workflow
  from-scratch Transformer pretraining infrastructure
  checkpoint / EMA / distributed training utilities
  LAMB optimizer and SLURM training patterns
```

Target layout:

```text
babylm-ELF/
├── README.md
├── LICENSE
├── requirements.txt
├── project.md
│
├── configs/
│   ├── babylm2025_elf_base.yml
│   ├── babylm2026_elf_base.yml
│   └── smoke.yml
│
├── scripts/
│   ├── prepare_babylm2025.sh
│   ├── prepare_babylm2026.sh
│   ├── train_babylm2025.sh
│   ├── train_babylm2026.sh
│   ├── export_babylm2025_hf.sh
│   ├── export_babylm2026_hf.sh
│   └── slurm/
│       ├── train_babylm2025.slurm
│       └── train_babylm2026.slurm
│
├── babylm_elf/
│   ├── cli/
│   │   ├── prepare_data.py
│   │   ├── train.py
│   │   └── export_hf.py
│   ├── config.py
│   ├── data/
│   │   ├── tokenizer.py
│   │   ├── datasets.py
│   │   └── collate.py
│   ├── modeling/
│   │   ├── model.py
│   │   ├── layers.py
│   │   └── heads.py
│   ├── diffusion/
│   │   ├── noising.py
│   │   ├── schedules.py
│   │   └── targets.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── step.py
│   │   ├── losses.py
│   │   ├── optim.py
│   │   └── checkpointing.py
│   ├── export/
│   │   ├── hf_config.py
│   │   ├── hf_model.py
│   │   └── convert_checkpoint.py
│   └── utils/
│       ├── logging.py
│       ├── distributed.py
│       └── seed.py
│
└── paper/
    ├── babylm-EFL.tex
    ├── paper.md
    ├── figures/
    └── tables/
```

Module responsibilities:

```text
configs/
  Year-specific run settings: data version, tokenizer path, model size,
  sequence length, loss weights, training budget, and output path.

scripts/
  Thin shell wrappers for data preparation, training, HF export, and SLURM jobs.

babylm_elf/cli/
  Command-line entry points. These files parse config and call library code;
  they should not contain core model or training logic.

babylm_elf/data/
  BabyLM text loading, year-specific tokenizer training, tokenized datasets,
  padding, attention masks, and dataloader collation.

babylm_elf/modeling/
  The from-scratch BabyLM-ELF model: token embeddings, denoising Transformer,
  timestep conditioning, embedding prediction head, and tied vocab decoder.

babylm_elf/diffusion/
  ELF continuous diffusion math: timestep sampling, Gaussian noising,
  noise schedules, and x0 / epsilon / velocity target conversion.

babylm_elf/training/
  Training loop, one-batch step, flow and decode losses, optimizer setup,
  gradient clipping, EMA, and checkpointing.

babylm_elf/export/
  Conversion from training checkpoints to Hugging Face-compatible artifacts
  for later use in the official BabyLM evaluation repository.
```

Non-target components:

```text
Not implemented in this repository:
  full BabyLM evaluation pipeline
  discrete <mask> corruption as the main objective
  frequency-informed masking as the main objective
  pretrained T5 encoder/tokenizer
  pretrained ELF checkpoints
```

## 7. Run Commands

Use Python 3.12 through the project conda environment:

```bash
conda activate babylm-elf
pip install -r requirements.txt
```

All prepare, train, export, and smoke-test commands should be run from the project root.

Generated data and outputs are grouped by run name:

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

Smoke test:

```bash
scripts/smoke_test.sh
```

BabyLM 2025 run:

```bash
python -m babylm_elf.cli.prepare_data --config configs/babylm2025_elf_base.yml
python -m babylm_elf.cli.train --config configs/babylm2025_elf_base.yml
python -m babylm_elf.cli.export_hf --config configs/babylm2025_elf_base.yml
```

BabyLM 2026 run:

```bash
python -m babylm_elf.cli.prepare_data --config configs/babylm2026_elf_base.yml
python -m babylm_elf.cli.train --config configs/babylm2026_elf_base.yml
python -m babylm_elf.cli.export_hf --config configs/babylm2026_elf_base.yml
```

Shell wrappers:

```bash
scripts/prepare_babylm2025.sh
scripts/train_babylm2025.sh
scripts/export_babylm2025_hf.sh

scripts/prepare_babylm2026.sh
scripts/train_babylm2026.sh
scripts/export_babylm2026_hf.sh
```

SLURM training:

```bash
sbatch scripts/slurm/train_babylm2025.slurm
sbatch scripts/slurm/train_babylm2026.slurm
```

## 8. Training Plan

### Stage 1: Data and Tokenizer

```text
1. Prepare BabyLM 2025 100M data for the 2025 comparison run.
2. Prepare BabyLM 2026 Strict 100M data for the 2026 challenge run.
3. Train one BPE tokenizer per year, using only that year's 100M corpus.
4. Tokenize train/dev data separately for 2025 and 2026.
5. Record tokenizer provenance, data version, word counts, and token counts.
```

Default:

```text
vocab_size = 16384
seq_length = 512 first, 1024 if memory allows
```

### Stage 2: Smoke Test

Run a tiny training job:

```text
100-500 steps for real debugging
2 steps for shape/integration smoke test
small batch size
objective = elf_flow
```

Success criteria:

```text
loss is finite
L_flow is finite
L_ce is finite
checkpoint saves and reloads
no shape mismatch in embedding -> noise -> denoise -> decode
```

### Stage 3: Main 100M Training

```text
2025 ELF run:
  data year: 2025
  model size: around 100M parameters
  objective: elf_flow
  decode_loss_weight: 0.25
  EMA: enabled
  mixed precision: enabled if stable

2026 ELF run:
  data year: 2026
  model size: around 100M parameters
  epochs: up to 10
  objective: elf_flow
  decode_loss_weight: 0.25
  EMA: enabled
  mixed precision: enabled if stable
```

Checkpoint policy:

```text
For each year:
  chck_1M, chck_2M, ..., chck_10M
  chck_20M, chck_30M, ..., chck_100M
  final checkpoint
```

## 9. Export and Evaluation

This repository stops at Hugging Face-compatible export. Full BabyLM evaluation is run later in the official evaluation repositories.

```text
BabyLM-ELF 2025 checkpoint
  -> export HF artifacts
  -> evaluate in official BabyLM 2025 evaluation repo
  -> compare with reported 2025 GPT-2 and masked-DLM results

BabyLM-ELF 2026 checkpoint
  -> export HF artifacts
  -> evaluate in official BabyLM 2026 evaluation repo
  -> compare with reported 2026 GPT-2 baseline
```

Export requirements:

```text
tokenizer files
model config
model weights
HF wrapper / modeling file
checkpoint metadata: data year, tokenizer provenance, training tokens, epoch count
```

## 10. Experiments

Experiments we actually run:

```text
exp2025_elf: BabyLM-ELF trained/exported with BabyLM 2025 data
exp2026_elf: BabyLM-ELF trained/exported with BabyLM 2026 data
```

Their evaluation scores are produced later with the matching official BabyLM evaluation repository.

Reported comparison results used in the paper:

```text
exp2025_gpt2_official: 2025 BabyLM GPT-2 100M official baseline result
exp2025_masked_dlm_winner: 2025 BabyLM masked-DLM winning paper result
exp2026_gpt2_official: 2026 BabyLM GPT-2 100M official baseline results
```

These reported baselines are not retrained in this project.

Optional ablations:

```text
decode_loss_weight = 0
decode_loss_weight = 0.1
decode_loss_weight = 0.5
different time schedules
prediction_type = x0 / epsilon / velocity
seq_length 512 vs 1024
```

## 11. Paper Story

Main narrative:

```text
GPT-2 learns left-to-right token prediction.
Masked DLM learns discrete token recovery.
Our model learns continuous denoising trajectories in embedding space.
```

Contribution:

```text
We adapt ELF-style continuous embedding diffusion to BabyLM 100M
without relying on pretrained text encoders, making it compatible
with the Strict track data constraint.
```

Result narrative:

```text
First, we train BabyLM-ELF under the 2025 data/evaluation setting and compare
it against reported 2025 GPT-2 and masked-DLM results.

Second, we train the same BabyLM-ELF architecture under the 2026 Strict 100M
setup and compare it against the reported 2026 official GPT-2 baseline.
```

## 12. Immediate Next Steps

```text
1. Keep the training-only package layout stable.
2. Verify year-aware Hugging Face data preparation and tokenizer training.
3. Run smoke training before each larger experiment.
4. Train/export BabyLM-ELF 2025.
5. Train/export BabyLM-ELF 2026.
6. Evaluate exported models later in the official BabyLM evaluation repositories.
7. Write paper results without mixing 2025 and 2026 benchmark numbers.
```
