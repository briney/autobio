#!/usr/bin/env bash
# run.sh — Phase 2: score sequences using ProteinMPNN/LigandMPNN
set -euo pipefail

WORKSPACE="$1"

echo "[mpnn-score] Running via Python API..."
python3 /opt/tool/score_sequences.py --workspace "$WORKSPACE" 2>&1 | tee "$WORKSPACE/logs/tool.log"
