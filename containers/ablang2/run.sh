#!/usr/bin/env bash
# run.sh — Phase 2: execute AbLang2 embedding extraction or PLL scoring
set -euo pipefail

WORKSPACE="$1"

echo "[ablang2] Running inference via Python API..."
python3 /opt/tool/run_ablang2.py --workspace "$WORKSPACE" 2>&1 | tee "$WORKSPACE/logs/tool.log"
