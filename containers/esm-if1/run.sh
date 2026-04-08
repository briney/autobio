#!/usr/bin/env bash
# run.sh — Phase 2: execute ESM-IF1 inverse folding or scoring
set -euo pipefail

WORKSPACE="$1"

echo "[esm-if1] Running via Python API..."
python3 /opt/tool/run_esm_if1.py --workspace "$WORKSPACE" 2>&1 | tee "$WORKSPACE/logs/tool.log"
