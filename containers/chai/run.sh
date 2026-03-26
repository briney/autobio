#!/usr/bin/env bash
# run.sh — Phase 2: execute Chai-1 structure prediction
#
# Delegates to a Python wrapper because the chai-lab fold CLI does not expose
# all inference parameters (num_diffn_samples, seed, constraint_path, etc.).
# The Python API (chai_lab.chai1.run_inference) provides full control.
set -euo pipefail

WORKSPACE="$1"
CONFIG="$WORKSPACE/config.json"

# Set downloads directory from config (pre-cached weights in container)
export CHAI_DOWNLOADS_DIR=$(jq -r '.downloads_dir // "/app/chai/downloads"' "$CONFIG")

echo "[chai] Running inference via Python API..."
python3 /opt/tool/run_chai.py --workspace "$WORKSPACE" 2>&1 | tee "$WORKSPACE/logs/tool.log"
