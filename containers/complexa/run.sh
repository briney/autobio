#!/usr/bin/env bash
# run.sh — Phase 2: execute Proteina-Complexa generation
set -euo pipefail

WORKSPACE="$1"

# Activate the UV venv (PATH is set in Dockerfile, but be explicit)
if [ -f /app/.venv/bin/activate ]; then
    source /app/.venv/bin/activate
fi

# Mark environment as initialized (bypasses `complexa init` check)
export COMPLEXA_INIT=1

python3 /opt/tool/run_complexa.py \
    --workspace "$WORKSPACE" \
    2>&1 | tee "$WORKSPACE/logs/tool.log"
