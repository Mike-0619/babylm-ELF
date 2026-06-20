#!/usr/bin/env bash
set -euo pipefail

python -m babylm_elf.cli.export_hf --config configs/babylm2026_elf_base.yml
