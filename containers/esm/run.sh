#!/usr/bin/env bash
# run.sh — Phase 2: execute ESM embedding extraction
#
# Delegates to a Python script because HuggingFace ESM has no built-in
# "extract embeddings to files" CLI. The Python script provides full
# control over layer selection, pooling, and batching.
set -euo pipefail

WORKSPACE="$1"
CONFIG="$WORKSPACE/config.json"

# Set HuggingFace cache from config (pre-cached weights in container)
export HF_HOME=$(jq -r '.hf_cache // "/app/esm/hf_cache"' "$CONFIG")
export TRANSFORMERS_OFFLINE=1

echo "[esm] Running embedding extraction via Python API..."
python3 /opt/tool/run_esm.py --workspace "$WORKSPACE" 2>&1 | tee "$WORKSPACE/logs/tool.log"
