# BabyLM ELF

BabyLM 2026 Strict/Strict-Small implementation of Embedded Language Flows.

The main 2026 recipe is:

```text
16K BabyLM BPE ids
-> learnable token embeddings
-> ELF-B continuous flow model
-> final-step MLM logits over the base 16K vocab
```

The tokenizer is trained only on the allowed BabyLM corpus. Exported tokenizer
files do not contain T5 `<extra_id_*>` tokens. Encoder pretraining uses internal
sentinel ids above the base vocab:

```text
base_vocab_size = 16384
sentinel_start_id = 16384
sentinel_count = 100
encoder_vocab_size = 16484
```

Encoder exposure counts toward the BabyLM 10-epoch budget when encoder routes
are used. The current 10M mainline is the 33M learnable-token AdamW4e-4 model
with global batch 128, cyclic Step10/20 masking, learned continuous mask latent,
`denoiser_noise_scale: 1.0`, and 128-token packed sequences. It trains with 20%
decode CE and 80% denoising MSE. For each decode row, the CE branch filters
punctuation/control tokens as prediction targets, replaces selected positions
with the learned mask latent, runs ELF decode mode at `t=1`, unembeds logits,
and applies CE only at masked positions.

The training objective is `token_mlm`; the BabyLM evaluation backend still
remains `mlm`. Old random-mask, CE100/CE90, split, and optimizer
ablation results remain only as archived outputs/results.

## Layout

```text
babylm_elf/cli/       prepare, encoder-train, ELF-train, HF-export entry points
babylm_elf/data/      HF export, BPE, flat token stream, manifest, mmap Dataset
babylm_elf/encoder/   T5-style span corruption utilities
babylm_elf/modeling/  ELF-B model with frozen scratch encoder mode
babylm_elf/training/  trainer, optimizer, checkpoint exposure accounting
babylm_elf/export/    Hugging Face remote-code export for AutoModel/MLM
babylm_elf/submission/ one contract for revisions, publication, and official evaluation
configs/              current 10M AdamW4e-4 mainline and ablation configs
scripts/              thin local and SLURM wrappers; training jobs live flat in scripts/train/
tests/                tokenizer/span/encoder/export/accounting smoke tests
```

## Setup

```bash
conda activate babylm-elf
pip install -r requirements.txt
```

PyTorch should match the cluster CUDA driver; see `requirements.txt`.

## 10M Run

```bash
sbatch scripts/prepare/prepare_2026_10M.slurm
sbatch scripts/train/train_learnable_2026_10M_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_bert_head_scaled_size33m_gb128_adamw_lr4e-4_4gpu.slurm
```

The prepare job consumes the complete official split, applies format-only
normalization, trains a track-specific 16K BPE tokenizer, tokenizes the corpus,
audits `manifest.json`, and only then promotes the staged data directory.

Tokenization has one canonical storage format, `flat_int16_le_v1`:

```text
<s> row1_ids <s> row2_ids <s> row3_ids ...
```

It is a raw little-endian signed-int16 stream. BOS preserves each official row
boundary, while fixed-length packing and attention continue across row
boundaries. Training memory-maps this file and copies only the current batch to
`torch.long`; it does not deserialize a Python list of document tensors. The
schema-v2 manifest records logical counts, tokenizer diagnostics, storage
metadata, packing/step counts, byte sizes, and preparation-time SHA-256 hashes.
Training startup checks sizes and immutable metadata without rescanning raw text.

Distributed training never pads ranks with repeated chunks. When the full-chunk
count is not divisible by world size, `DistributedSampler(drop_last=True)`
omits a shuffled remainder that rotates across epochs. For the current 100M
route this drops one chunk instead of repeating three; batch and optimizer-step
counts stay unchanged. Model initialization uses the same seed on every rank;
after DDP synchronization, training RNG is reseeded with `seed + global_rank`.

EMA uses warmup and scales the configured `0.9999 @ 95,000 optimizer steps`
reference to the actual run length. Checkpoints store both `model_ema` and
`model_raw`; the backward-compatible `model` key aliases EMA.

The 100M route uses the same pipeline:

```bash
sbatch scripts/prepare/prepare_2026_100M.slurm
```

After both routes are prepared, the four-rank mmap/dataloader check is:

```bash
sbatch scripts/prepare/smoke_2026_data_4rank.slurm
```

Aux LR sweep:

```bash
bash scripts/train/submit_2026_10M_size33m_gb128_lr5e-4_aux_lr_sweep.sh
```

Export:

```bash
python -m babylm_elf.cli.export_hf --all-revisions --track strict-small
```

Export selects EMA by default. Use `--weights raw` only for an explicit raw/EMA
comparison. Export is fail-closed: all 19 Strict-Small or all 28 Strict
checkpoints must exist before an all-revision export starts.

Validate and publish the final model plus every required BabyLM revision:

```bash
python -m babylm_elf.cli.submission check \
  --revisions-dir outputs/.../hf_revisions \
  --track strict-small \
  --smoke

python -m babylm_elf.cli.submission publish \
  --revisions-dir outputs/.../hf_revisions \
  --repo-id YOUR_ORG/YOUR_MODEL \
  --track strict-small \
  --dry-run
```

Remove `--dry-run` to publish. The command uploads `main` and every required
`chck_*M` branch, then fresh-cache smoke-tests each branch. Official scoring
uses `AutoModelForMaskedLM.forward(...)` logits; sampling is not involved.

Prepare the exact official evaluator checkout without modifying its source:

```bash
python -m babylm_elf.cli.submission prepare-eval \
  --work-dir .cache/babylm-eval \
  --download-data
```

Full evaluation intentionally accepts only a public Hub repository:

```bash
python -m babylm_elf.cli.submission evaluate \
  --work-dir .cache/babylm-eval \
  --repo-id YOUR_ORG/YOUR_MODEL \
  --track strict-small
```

The checkout is pinned to the commit in `submission/contract.py`, verified
clean before and after evaluation, and run through the official scripts. The
final validator rejects missing tasks or revisions, wrong sample counts,
`None`, NaN, and infinity. Full EWoK remains gated; if unavailable, preflight
stops with the exact command needed after accepting its terms.

The maintained denoise route uses `denoiser_noise_scale: 1.0`. Older
noise-scale-2.0 runs are archived as historical artifacts, not public training
or export entry points.

Optional diagnostics:

```bash
python -m babylm_elf.cli.diagnose_generation <hf_model_dir>
```

This script can write open-ended samples and denoising reports for debugging
the ELF trajectory, but it is not required for training, export, BLiMP, or
official BabyLM evaluation.

## Expected Artifacts

```text
data/2026_10M/tokenizer/tokenizer.json
data/2026_10M/tokenized/train_10M.bin
data/2026_10M/manifest.json
data/2026_100M/tokenizer/tokenizer.json
data/2026_100M/tokenized/train_100M.bin
data/2026_100M/manifest.json
outputs/2026_10M/learnable_token_mlm_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_ce20_bert_head_scaled_size33m_gb128_adamw_lr4e-4_4gpu/hf_revisions/chck_100M/
outputs/2026_10M/learnable_token_mlm_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_ce20_bert_head_scaled_muon_size33m_gb128_lr5e-4_aux2e-4_4gpu/hf_revisions/chck_100M/
outputs/2026_10M/learnable_token_mlm_oneperseg_len16_40_filter_punct_ctrl_s0_seq128_ce20_bert_head_scaled_muon_size33m_gb128_lr5e-4_aux3e-4_4gpu/hf_revisions/chck_100M/
```

For 10M scratch-encoder routes, checkpoint metadata reports total exposure as
`encoder warm-up words + ELF words`: joint uses a `10M` offset, while frozen
scratch uses a `20M` offset.

## Tests

```bash
python -m unittest discover tests
```

For Hugging Face remote-code tests on systems with a read-only home cache:

```bash
HF_HOME=/tmp/babylm-elf-hf-test \
TRANSFORMERS_CACHE=/tmp/babylm-elf-hf-test/transformers \
python -m unittest tests.test_hf_export
```

## Notes

- Raw data preparation stays shared between 10M and 100M.
- The official BabyLM evaluation backend should be `mlm`.
- Sampling/generation is optional debug only; it is not used by BabyLM/BLiMP
  scores.
- The 10M main methods use `decoder_objective: token_mlm` with
  a learned continuous mask latent at selected mask positions, ELF decode mode
  at `t=1`, unembed logits, and masked-position CE.
- MLM-style decoder objectives mask positions in the original CE row. The
  maintained route does not duplicate CE rows, traverse positions sequentially
  like scoring, or use a global cursor.
- Supported training objective is `token_mlm`.
- `AutoModel` returns `BaseModelOutput(last_hidden_state=...)`.
- `AutoModelForMaskedLM` returns `MaskedLMOutput(logits=...)`.
- The scratch encoder is frozen during ELF training and excluded from the
  optimizer and EMA trainable shadow unless `scratch_encoder_trainable: true`.
