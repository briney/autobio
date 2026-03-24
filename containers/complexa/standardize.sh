#!/usr/bin/env bash
# standardize.sh — Phase 3: transform Proteina-Complexa raw outputs to schema format
set -euo pipefail

WORKSPACE="$1"

python3 /opt/tool/standardize.py \
    --workspace "$WORKSPACE"
