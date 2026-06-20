#!/usr/bin/env bash
set -euo pipefail

python -m babylm_elf.cli.train --config configs/babylm2025_elf_base.yml
