# BabyLM: ELF-style Continuous Diffusion

> **Implementation status (June 2026):** The executable code now follows the
> ELF paper's mixed denoiser/decoder objective, rectified-flow direction,
> logit-normal schedules, decoder corruption, self-conditioning CFG,
> RMSNorm/RoPE/qk-norm architecture, MuonWithAuxAdam, warmup, and EMA export.
> Official BabyLM competition configs use 10 epochs; shorter ELF
> paper-alignment and learning-rate sweeps are treated as ablations.
> BabyLM Strict uses the paper's learnable tied-embedding ablation instead of
> pretrained T5-small.

## 1. Project Goal

本项目目标是实现一个合规的 **BabyLM ELF-style continuous diffusion language model**。核心实验是把 ELF-style continuous embedding diffusion 改造成 BabyLM Strict 约束下可训练、可导出、可评测的 from-scratch 语言模型。当前主线是 BabyLM 2026 Strict 100M；计划同时加入 BabyLM 2026 Strict-Small 10M track。

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

Secondary 10M question:

```text
在 BabyLM 2026 Strict-Small 10M 数据限制下，同一套 BabyLM-ELF 架构和
tokenizer/data-processing recipe 是否仍然稳定有效？
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

### Comparison C: BabyLM 2026 Strict-Small 10M Setting

```text
data:       BabyLM 2026 English Strict-Small 10M
evaluation: BabyLM 2026 evaluation pipeline/tasks
track:      English Strict-Small 10M
status:     configured; data preparation and training pending
```

| Model | Training Objective | Data | Evaluation | Role |
| --- | --- | --- | --- | --- |
| BabyLM-ELF 2026 10M | continuous embedding diffusion / ELF-style flow | BabyLM 2026 Strict-Small 10M | BabyLM 2026 eval | 10M track experiment |

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

2026 10M run:
  data: BabyLM 2026 English Strict-Small 10M
  tokenizer: trained from scratch on BabyLM 2026 Strict-Small only
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

2026 compliance checklist:

```text
Training exposure:
  Do not exceed the official exposure budget.
  STRICT-SMALL / 10M: at most 100M whitespace-separated input words.
  STRICT / 100M: at most 1B whitespace-separated input words.
  In this project, max_steps: 0 makes the trainer run 10 epochs, which matches
  the intended BabyLM exposure budget for the official 10M and 100M corpora.

Data source:
  Do not train the tokenizer or model on dev/test/eval data.
  Tokenizers must be trained only from the official train corpus for that track.
  Re-tokenized or cleaned local data is allowed only when it is derived from the
  same official train corpus and does not add external text.

Checkpoint requirements:
  Save intermediate checkpoints by word exposure, not arbitrary step count.
  Save every 1M words through 10M.
  Save every 10M words through 100M.
  For non-STRICT-SMALL tracks, also save every 100M words through 1B.
```

Current implementation status:

```text
2026 100M full config:
  max_steps: 0
  checkpoint_by_words: true
  save_every: 0
  required checkpoints: 1M..10M, 20M..100M, 200M..1B

2026 10M full config:
  max_steps: 0
  checkpoint_by_words: true
  save_every: 0
  required checkpoints: 1M..10M, 20M..100M

Checkpoint files:
  outputs/<run>/checkpoints/babylm_required/chck_*.pt
```

The word-based checkpoints include metadata such as `words_seen`,
`target_words`, and `steps_per_epoch`. They are model-only checkpoints so the
output directory stays compact. Short test configs may still use step-based
checkpointing, but final BabyLM runs should use the word-based policy above.

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
  -> year-specific byte-level BPE tokenizer under data/<run>/tokenizer/
  -> torch-saved tokenized documents under data/<run>/tokenized/
```

Current 2026 data-processing decision:

```text
tokenizer model       = BPE
vocab_size            = 16384
special tokens        = <unk>, <s>, </s>, <pad>, <mask>, <special_0>...<special_10>
BPE settings          = byte_fallback=False, fuse_unk=False, ignore_merges=True
initial alphabet      = ByteLevel.alphabet()
normalizer            = prepend one space, NFKC, newline spacing cleanup
pre-tokenizer         = Unicode letter/number/punctuation regex split
                      + ByteLevel(add_prefix_space=False, use_regex=False, trim_offsets=True)
                      + Split(Regex(".{1,24}"))
post-processor        = single: "<s> $A"; pair: "<s> $A <s> $B"
tokenizer JSON cleanup = remove the final 256 byte-alphabet added_tokens entries
document cleanup      = strip, remove one-line Wiki-style "= = = ... = = =" headings, strip again
encoded dtype         = torch.int16
seq_length            = 1024 tokens per training example
```

This mirrors the tokenizer/data-preparation recipe used in the 2025
BabyLM diffusion winner's released code, but it is implemented directly in this
repository. The 2026 configs now use the standard path
`data/2026_100M/{tokenizer,tokenized}/`; the earlier
`data/babylm2026_maskeddlm/` directory was only a comparison artifact.

Current status:

```text
2026 100M: uses BabyLM-community/BabyLM-2026-Strict from Hugging Face.
2026 10M: configured; uses BabyLM-community/BabyLM-2026-Strict-Small from Hugging Face.
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

2026 Strict-Small 10M dataset reference:

```python
from datasets import load_dataset

ds = load_dataset("BabyLM-community/BabyLM-2026-Strict-Small")
```

Dataset page:

```text
https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict-Small
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

Training objective:

```text
Each example independently selects one branch:
  probability 0.8 -> denoiser mode, velocity-space MSE
  probability 0.2 -> decoder mode, token cross-entropy

The branch masks share one valid-token denominator, so in expectation:
  L = 0.8 * mean(L_flow) + 0.2 * mean(L_ce)
```

Core training step:

```text
input_ids
  -> x0 = token_embedding(input_ids)
  -> t = sample_timestep()
  -> eps = sample_gaussian_noise()
  -> build denoiser z_t and decoder-corrupted z_decoder
  -> choose denoiser/decoder mode independently per example
  -> apply ELF self-conditioning and training-time CFG to denoiser rows
  -> prediction, decoder_logits = shared_model(mixed_z, mixed_t, mode)
  -> mask velocity MSE to denoiser rows
  -> mask token CE to decoder rows
  -> combine both masked sums with one valid-token denominator
```

Key difference from original ELF:

```text
ELF_PyTorch:
  text -> pretrained/frozen text encoder latent x0 -> flow model -> decoder

BabyLM-ELF:
  text -> from-scratch BPE -> trainable token embedding x0 -> flow model -> tied vocab decoder
```

This is an intentional BabyLM-compliant version of ELF's learnable
tied-embedding ablation. It is not the paper's default pretrained T5
contextual-embedding setup, because pretrained T5 weights and tokenizer would
violate the Strict from-scratch route used here.

Inference surfaces:

```text
Official BabyLM evaluation:
  AutoModelForMaskedLM.forward(...)
  -> continuous-noise pseudo-likelihood for masked positions
  -> used by the official MLM backend/checkpoint revisions

Diagnostic ELF generation:
  BabyLMELFForMaskedLM.generate(...)
  -> Gaussian noise -> ODE/SDE continuous denoising -> final decode
  -> qualitative/debugging signal only, not the official BabyLM score
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
  SLURM training patterns
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
│   ├── 2025.yml
│   ├── 2026_100M_adamW.yml
│   ├── 2026_100M_muon.yml
│   ├── 2026_10M_adamW.yml
│   ├── 2026_10M_muon.yml
│   └── smoke.yml
│
├── scripts/
│   ├── prepare_2025.sh
│   ├── prepare_2026_100M.sh
│   ├── prepare_2026_10M.sh
│   ├── train_2025.sh
│   ├── train_2026_100M.sh
│   ├── train_2026_10M.sh
│   ├── export_2025_hf.sh
│   ├── export_2026_100M_hf.sh
│   ├── export_2026_10M_hf.sh
│   └── slurm/
│       ├── train_2025.slurm
│       └── train_2026_100M_adamW.slurm
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

data/2026_100M/{raw,tokenizer,tokenized}/
outputs/2026_100M/{adamW,muon}/checkpoints/
outputs/2026_100M/{adamW,muon}/hf/

data/2026_10M/{raw,tokenizer,tokenized}/
outputs/2026_10M/{adamW,muon}/checkpoints/
outputs/2026_10M/{adamW,muon}/hf/
```

Smoke test:

```bash
scripts/smoke_test.sh
```

BabyLM 2025 run:

```bash
python -m babylm_elf.cli.prepare_data --config configs/2025.yml
python -m babylm_elf.cli.train --config configs/2025.yml
python -m babylm_elf.cli.export_hf --config configs/2025.yml
```

BabyLM 2026 run:

```bash
python -m babylm_elf.cli.prepare_data --config configs/2026_100M_adamW.yml
python -m babylm_elf.cli.train --config configs/2026_100M_adamW.yml
python -m babylm_elf.cli.export_hf --config configs/2026_100M_adamW.yml --all-revisions
```

BabyLM 2026 Strict-Small 10M run:

```bash
python -m babylm_elf.cli.prepare_data --config configs/2026_10M_adamW.yml
python -m babylm_elf.cli.train --config configs/2026_10M_adamW.yml
python -m babylm_elf.cli.export_hf --config configs/2026_10M_adamW.yml --all-revisions
```

Shell wrappers:

```bash
scripts/prepare_2025.sh
scripts/train_2025.sh
scripts/export_2025_hf.sh

scripts/prepare_2026_100M.sh
scripts/prepare_2026_10M.sh
scripts/train_2026_100M.sh
scripts/train_2026_10M.sh
scripts/export_2026_100M_hf.sh
scripts/export_2026_10M_hf.sh
```

SLURM training:

```bash
sbatch scripts/slurm/train_2025.slurm
sbatch scripts/slurm/train_2026_100M_adamW.slurm
sbatch scripts/slurm/train_2026_10M_adamW.slurm
```

## 8. Training Plan

### Stage 1: Data and Tokenizer

```text
1. Prepare BabyLM 2025 100M data for the 2025 comparison run.
2. Prepare BabyLM 2026 Strict 100M data for the 2026 challenge run.
3. Prepare BabyLM 2026 Strict-Small 10M data for the 10M track.
4. Train one BPE tokenizer per run, using only that run's allowed corpus.
5. Tokenize train/dev data separately for 2025, 2026 100M, and 2026 10M.
6. Record tokenizer provenance, data version, word counts, and token counts.
```

Default:

```text
vocab_size = 16384
seq_length = 1024
tokenization = byte-level BPE with Unicode regex split, ByteLevel trim_offsets,
               `.{1,24}` chunk split, and Wiki-style heading cleanup
```

Rationale:

```text
The tokenizer/data-processing recipe follows the released 2025 masked-DLM
implementation. The ELF architecture uses sequence length 1024 to match the
official ELF setup; sequence length is independent of the BPE training recipe.
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
  decoder_probability: 0.2
  EMA: enabled
  mixed precision: enabled if stable

2026 ELF run:
  data year: 2026
  data size: Strict 100M
  model size: around 100M parameters
  epochs: up to 10
  objective: elf_flow
  decoder_probability: 0.2
  EMA: enabled
  mixed precision: enabled if stable

2026 ELF small run:
  data year: 2026
  data size: Strict-Small 10M
  status: configured; data preparation and training pending
  tokenizer: trained from scratch on Strict-Small only
  objective: elf_flow
  decoder_probability: 0.2
  EMA: enabled
  mixed precision: enabled if stable
```

Checkpoint policy:

```text
BabyLM 2026 100M / STRICT:
  checkpoints/babylm_required/chck_1M.pt, ..., chck_10M.pt
  checkpoints/babylm_required/chck_20M.pt, ..., chck_100M.pt
  checkpoints/babylm_required/chck_200M.pt, ..., chck_1000M.pt

BabyLM 2026 10M / STRICT-SMALL:
  checkpoints/babylm_required/chck_1M.pt, ..., chck_10M.pt
  checkpoints/babylm_required/chck_20M.pt, ..., chck_100M.pt
```

The 2026 full-run configs enable `checkpoint_by_words: true`, so checkpoint
targets are based on estimated whitespace-word exposure rather than raw training
steps. Each checkpoint includes metadata with `words_seen`, `target_words`, and
`steps_per_epoch`. BabyLM-required checkpoints are model-only to keep outputs
compact. No duplicate `final.pt` or periodic step checkpoints are written for
word-checkpoint runs. All checkpoint files use temporary-file writes followed
by atomic replacement, so an interrupted save cannot replace a valid checkpoint
with a partial file.
Export automatically selects the official checkpoint with the largest exposure.

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
exp2026_100M_elf: BabyLM-ELF trained/exported with BabyLM 2026 Strict 100M data
exp2026_10M_elf: BabyLM-ELF run with BabyLM 2026 Strict-Small 10M data
```

Current 2026 experiment settings:

```text
config:       configs/2026_100M_adamW.yml
test config:  configs/2026_100M_adamW.yml
tokenizer:    data/2026_100M/tokenizer/tokenizer.json
tokenized:    data/2026_100M/tokenized/train_100M.bin
seq_length:   1024
```

2026 Strict-Small 10M settings:

```text
dataset:      BabyLM-community/BabyLM-2026-Strict-Small
config:       configs/2026_10M_adamW.yml
test config:  configs/2026_10M_adamW.yml
tokenizer:    data/2026_10M/tokenizer/tokenizer.json
tokenized:    data/2026_10M/tokenized/train_10M.bin
seq_length:   1024
```

Their evaluation scores are produced later with the matching official BabyLM evaluation repository.

Reported comparison results used in the paper:

```text
exp2025_gpt2_official: 2025 BabyLM GPT-2 100M official baseline result
exp2025_masked_dlm_winner: 2025 BabyLM masked-DLM winning paper result
exp2026_100M_gpt2_official: 2026 BabyLM GPT-2 100M official baseline results
```

These reported baselines are not retrained in this project.

Optional ablations:

```text
decoder_probability = 0.1
decoder_probability = 0.2
decoder_probability = 0.3
different time schedules
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
