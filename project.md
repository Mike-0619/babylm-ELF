# BabyLM 2026 ELF Experiment Specification

## 1. Research Question

The primary study asks:

```text
Under the BabyLM 2026 Strict-Small 10M data and 100M-word exposure limit,
how do the official ELF noisy decoder, cyclic sparse token-MLM, BERT15
token-MLM, and a Standard MDLM baseline compare when the ELF-S backbone,
shared initialization, data, optimizer, and training budget are held fixed?
```

All four models are trained from scratch. The official ELF repository is a
read-only reference; this repository owns all BabyLM integration.

## 2. Experiments

The primary objective-controlled 10M comparison contains four AdamW runs:

| Run | Decoder input and target | Evaluation adapter |
| --- | --- | --- |
| `official_noisy_ce` | `sigmoid(N(0.8,0.8))*x0 + (1-p)*N(0,5^2)`; CE on eligible lexical tokens | `fixed_gaussian_v1` |
| `token_mlm` | cyclic Step10/Step20 targets replaced by the learned mask latent; CE only on masked tokens | learned mask latent |
| `token_mlm_bert15` | independent 15% targets; 80% mask latent, 10% random lexical token, 10% unchanged; CE on all targets | learned mask latent |
| `standard_mdlm` | log-linear absorbing masks; masked-token `CE/t` normalized by all eligible tokens | `mdlm_subs_v1` |

The four YAML configs are:

```text
configs/10m/elf_noisy.yml
configs/10m/elf_mlm_cyclic.yml
configs/10m/elf_mlm_bert15.yml
configs/10m/elf_mdlm.yml
```

An additional optimizer-controlled run keeps the complete `official_noisy_ce`
model, data, objective, initialization, batch, EMA, and exposure contract fixed,
and replaces only AdamW with an empirically tuned MuonWithAuxAdam:

```text
configs/10m/elf_noisy_muon.yml
```

The shared backbone and optimizer/data settings are controlled. Route-specific
flow, self-conditioning, and mask modules are instantiated only when used.

## 3. Shared Model

ELF-S uses:

```text
vocab_size:             16,384
embedding_size:            384
hidden_size:               512
intermediate_size:       2,048
num_hidden_layers:           8
num_attention_heads:         8
bottleneck_size:            96
sequence_length:           128
```

All routes retain RMSNorm, RoPE, QK norm, SwiGLU, the 96-dimensional
bottleneck, decoder, and official single gated mode-token group.
Embedding lookup and unembedding share the same learned matrix. Both use its
row-wise unit-normalized directions; lookup additionally applies a `sqrt(384)`
scale so token embeddings have RMS 1.0, while unembedding uses the normalized
directions directly. The decoder projection learns the output magnitude that
sets the logits temperature.

| Route | Objective-specific modules | Trainable parameters |
| --- | --- | ---: |
| `official_noisy_ce` (AdamW or Muon) | flow head and full self-conditioning projection | 33,092,048 |
| `token_mlm` | flow head, full self-conditioning projection, mask latent | 33,092,432 |
| `token_mlm_bert15` | flow head, full self-conditioning projection, mask latent | 33,092,432 |
| `standard_mdlm` | mask latent and direct decoder-input projection | 32,747,472 |

Standard MDLM does not construct a flow head or self-conditioning projection.
Its direct projection is initialized from the canonical projection's first 384
columns and bias. Construction uses a common canonical initialization before
route pruning, so all same-named shared tensors are exactly equal at seed 42.

## 4. Shared Training Contract

```text
seed:                         42
epochs:                       10
optimizer steps:              9,670
per-device batch:             32
world size:                   4
global batch:                 128
gradient accumulation:        1
precision:                    BF16
optimizer:                    AdamW
learning rate:                4e-4
betas:                        (0.9, 0.999)
epsilon:                      1e-8
weight decay:                 0
warmup:                       0.5 epoch
schedule:                     cosine
minimum learning rate:        4e-5
gradient clipping:            1
EMA reference:                0.9999 at 95,000 steps
EMA scaling and warmup:       enabled
```

The optimizer-controlled official ELF run uses an empirical LR pair:

```yaml
optimizer:
  type: muon
  learning_rate: 5.0e-2
  aux_learning_rate: 5.0e-3

scheduler:
  type: constant
  warmup_epochs: 0.5
```

Eligible two-dimensional parameters use Muon; all remaining parameters use the
official-compatible auxiliary Nesterov-Adam update. Linear warmup scales both
group learning rates proportionally. The values `5e-2 / 5e-3` are an
empirically supported starting point from an older global-batch-128 sweep,
where the final-20-step mean total loss was approximately 1.2347. That sweep
used an older token-MLM/head/data route, so it is not evidence that this pair is
optimal for the current noisy-CE experiment and is not reported as a current
result. The run name includes `muon_empirical` so these values cannot be
mistaken for the official ELF optimizer defaults.

The three ELF objectives use the shared denoiser with `P_mean=-1.5`,
`P_std=0.8`, noise scale 2,
`t_eps=0.05`, 80% flow MSE, 20% decoder CE, 50% self-conditioning, and CFG
sampled from `[0.5,5]`.

The cyclic route chooses Step10/Step20 targets per packed BOS segment. BERT15
instead selects each eligible token independently with probability 0.15 and
uses 80% mask-latent, 10% uniform legal lexical-token, and 10% unchanged
replacement. Every selected position contributes CE, including random and
unchanged positions; rows with no selected targets remain decoder rows with
zero CE.

Standard MDLM instead uses 100% absorbing-mask diffusion with log-linear mask
probability `(1-1e-3)t`, rank-aware stratified antithetic time samples on
`[1e-3,1]`, and the FP32 NELBO estimator `sum(masked CE/t) / eligible tokens`.
It does not use flow MSE, Step10/Step20 masking, CFG, self-conditioning, or loss
softening. SUBS suppresses `<mask>` output and carries visible tokens unchanged.

Token IDs below 16, padding, and empty/control tokens do not contribute to CE,
MSE, or branch weighting. Punctuation remains eligible.

## 5. Data And Packing

The only primary training corpus is
`BabyLM-community/BabyLM-2026-Strict-Small`. Data preparation creates:

```text
data/2026_10M/raw/train_10M.txt
data/2026_10M/tokenizer/tokenizer.json
data/2026_10M/tokenized/train_10M.bin
data/2026_10M/manifest.json
```

The schema-v3 manifest and token stream must match:

```text
source words:                 10,000,000
usable normalized words:       9,999,993
subwords:                     14,735,674
stream tokens:                15,839,777
stream format:                flat_int16_le_v1
chunks:                          123,748
samples per rank:                 30,937
batches per rank per epoch:          967
```

Training verifies every artifact against the SHA-256 recorded in the manifest.
BOS tokens define packed segments, attention is block diagonal between
segments, and RoPE positions restart per segment. The distributed sampler
never pads with duplicate chunks. Each config pins the Hugging Face dataset
commit; the manifest records the resulting fingerprint, corpus statistics, and
artifact hashes. Strict-Small uses revision `c92ab16b...`, and Strict uses
revision `9e57baaa...`.

Nominal exposure is based on the official 10M source-word count; metadata also
records usable-word exposure. Configurations exceeding 100M nominal exposure
are rejected.

## 6. Checkpoint, Resume, And Export

Each 10M run produces the 19 Strict-Small exposure revisions:

```text
chck_1M ... chck_9M
chck_10M, chck_20M, ... chck_100M
```

Revision files contain raw and EMA weights plus config/exposure metadata.
`latest.pt` is atomically overwritten every 500 steps and at completion with
raw weights, EMA state, optimizer, scheduler, step, epoch, and microbatch.
The latest checkpoint also stores Python, Torch CPU, and CUDA RNG state for
every rank.
Both use checkpoint format v4; older checkpoint formats are intentionally not
loaded by the refactored training or export code.

Training accepts `--resume auto|PATH`; local cluster wrappers should use
`auto`. Automatic resume loads `latest.pt`, starts only in an empty run
directory, and refuses to overwrite orphaned revisions. Resume validates
model, data, optimizer, objective, and world size, reconstructs and skips the
dataloader, then restores the saved per-rank random streams.

HF remote code supports `AutoModel`, `AutoModelForMaskedLM`, EMA/raw selection,
the main export, and all 19 revisions. Both Token-MLM routes use the trained
mask latent. BERT15 export metadata records that training uses 15% 80/10/10
corruption while evaluator masks are all represented by the mask latent.
Noisy-CE uses deterministic target-free Gaussian latents with seed 0 and scale
5, with the distribution shift declared in metadata. Standard MDLM uses
`mdlm_subs_v1`; `generate_mdlm()` starts with masked interior tokens, preserves
BOS/EOS, and uses 128 log-linear ancestral reverse steps by default.

## 7. Strict 100M ELF-B Routes

Two approximately 96M-parameter runs repeat the main objective comparison at
ELF-B scale:

| Route | Parameters | Native HF adapter |
| --- | ---: | --- |
| Official noisy-CE | 96,517,632 | `fixed_gaussian_v1` |
| Standard MDLM | 95,861,504 | `mdlm_subs_v1` |

Both use the BabyLM 2026 Strict 100M corpus, global batch 256, 10 epochs,
49,130 optimizer steps, and a 1B-word exposure limit. AdamW uses peak LR
`3e-4`, cosine decay to `3e-5`, 0.5 epoch warmup, zero weight decay, and the
same scaled-warmup EMA. The resolved warmup is 2,456 steps and the resolved EMA
decay is approximately 0.9998066445.

Each 100M run produces all 28 Strict revisions, from `chck_1M` through
`chck_1000M`.

The configs are `configs/100m/elf_noisy.yml` and
`configs/100m/elf_mdlm.yml`. Scratch T5, Gaussian embedding, encoder
pretraining/contextuality, and format-v4 HF export remain supported interfaces
but are not experiment entries.

## 8. Acceptance Criteria

- The four primary configs load with the same AdamW, schedule, EMA, data, and
  exposure settings.
- The Muon noisy-CE config differs from the AdamW noisy-CE config only in run
  name, optimizer, and scheduler, with Muon/auxiliary LRs `5e-2 / 5e-3` and a
  constant schedule.
- Route parameter totals are exactly 33,092,048, 33,092,432, 33,092,432, and
  32,747,472.
- All same-named shared parameters initialize identically; no route keeps
  objective-unused placeholder parameters.
- Noisy corruption, cyclic masking, BERT15 corruption, token filtering, mixed
  CE/MSE weighting, packed attention, and RoPE reset have unit coverage.
- CPU, single-GPU, and four-rank smoke runs produce finite loss/gradients and
  synchronized parameters.
- All 19 Strict-Small and 28 Strict revisions contain complete raw/EMA
  metadata; `latest.pt` restores optimizer, scheduler, EMA, dataloader
  progress, and per-rank RNG streams.
- Fresh-cache HF loading works for all native adapters and selected revisions.
- The two 100M configs reproduce noisy-CE and Standard MDLM on the shared
  ELF-B backbone and Strict data route.
- The official ELF working tree has no tracked changes.
