#!/usr/bin/env bash
# run.sh — Phase 2: execute ESMFold structure prediction
set -euo pipefail

WORKSPACE="$1"
CONFIG="$WORKSPACE/config.json"

export HF_HOME=$(jq -r '.hf_cache // "/app/esmfold/hf_cache"' "$CONFIG")
export TRANSFORMERS_OFFLINE=1

echo "[esmfold] Running structure prediction via Python API..."
python3 /opt/tool/run_esmfold.py --workspace "$WORKSPACE" 2>&1 | tee "$WORKSPACE/logs/tool.log"
