#!/usr/bin/env bash
set -euo pipefail

python -m babylm_elf.cli.export_hf --config configs/babylm2025_elf_base.yml
