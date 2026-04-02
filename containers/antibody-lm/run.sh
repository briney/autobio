#!/usr/bin/env bash
# run.sh — Phase 2: execute antibody LM embedding extraction or PLL scoring
set -euo pipefail

WORKSPACE="$1"
CONFIG="$WORKSPACE/config.json"

# Set HuggingFace cache from config (pre-cached weights in container)
export HF_HOME=$(jq -r '.hf_cache // "/app/antibody-lm/hf_cache"' "$CONFIG")
export TRANSFORMERS_OFFLINE=1

echo "[antibody-lm] Running inference via Python API..."
python3 /opt/tool/run_antibody_lm.py --workspace "$WORKSPACE" 2>&1 | tee "$WORKSPACE/logs/tool.log"
