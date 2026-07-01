# BabyLM ELF

BabyLM 2026 Strict/Strict-Small implementation of Embedded Language Flows
with a from-scratch contextual encoder.

The main 2026 recipe is:

```text
16K BabyLM BPE ids
-> scratch T5-small-style span-corruption encoder
-> channel-wise normalized 512-d contextual latents
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

Encoder exposure counts toward the BabyLM 10-epoch budget. The default split is
3 encoder epochs plus 7 ELF epochs.

## Layout

```text
babylm_elf/cli/       prepare, encoder-train, ELF-train, HF-export entry points
babylm_elf/encoder/   T5-style span corruption utilities
babylm_elf/modeling/  ELF-B model with frozen scratch encoder mode
babylm_elf/training/  trainer, optimizer, checkpoint exposure accounting
babylm_elf/export/    Hugging Face remote-code export for AutoModel/MLM
configs/              10M/100M encoder and scratch-ELF configs
scripts/              thin local and SLURM wrappers
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
scripts/prepare_2026_10M.sh
scripts/train_encoder_2026_10M.sh
scripts/train_2026_10M.sh
scripts/export_2026_10M_hf.sh
```

On SLURM:

```bash
sbatch scripts/slurm/prepare_2026_10M.slurm
sbatch scripts/slurm/train_encoder_2026_10M.slurm
sbatch scripts/slurm/train_2026_10M.slurm
bash scripts/slurm/export_2026_10M_hf.txt
```

## 100M Run

```bash
scripts/prepare_2026_100M.sh
scripts/train_encoder_2026_100M.sh
scripts/train_2026_100M.sh
scripts/export_2026_100M_hf.sh
```

On SLURM:

```bash
sbatch scripts/slurm/prepare_2026_100M.slurm
sbatch scripts/slurm/train_encoder_2026_100M.slurm
sbatch scripts/slurm/train_2026_100M.slurm
bash scripts/slurm/export_2026_100M_hf.txt
```

## Expected Artifacts

```text
data/2026_10M/tokenizer/tokenizer.json
data/2026_10M/tokenized/train_10M.bin
outputs/2026_10M/encoder/checkpoints/final.pt
outputs/2026_10M/encoder/latent_stats.pt
outputs/2026_10M/scratch_encoder_muon/checkpoints/babylm_required/chck_*M.pt
outputs/2026_10M/scratch_encoder_muon/hf_revisions/
outputs/2026_10M/scratch_encoder_muon/hf_revisions/scratch_encoder_muon_hf/

data/2026_100M/tokenizer/tokenizer.json
data/2026_100M/tokenized/train_100M.bin
outputs/2026_100M/encoder/checkpoints/final.pt
outputs/2026_100M/encoder/latent_stats.pt
outputs/2026_100M/scratch_encoder_muon/checkpoints/babylm_required/chck_*M.pt
outputs/2026_100M/scratch_encoder_muon/hf_revisions/
outputs/2026_100M/scratch_encoder_muon/hf_revisions/scratch_encoder_muon_hf/
```

For 10M, checkpoint metadata reports total exposure as
`30M encoder words + ELF words`, so required revisions start after the encoder
offset. For 100M, the offset is `300M`.

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
- The official BabyLM backend should be `mlm`.
- `AutoModel` returns `BaseModelOutput(last_hidden_state=...)`.
- `AutoModelForMaskedLM` returns `MaskedLMOutput(logits=...)`.
- The scratch encoder is frozen during ELF training and excluded from the
  optimizer and EMA trainable shadow.
