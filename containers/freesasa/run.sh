#!/usr/bin/env bash
# run.sh — Phase 2: execute FreeSASA SASA/BSA calculation
set -euo pipefail

WORKSPACE="$1"
CONFIG="$WORKSPACE/config.json"
OUTPUT_DIR="$WORKSPACE/outputs/raw"

# --- Build command ----------------------------------------------------------
CMD=(
    python /opt/tool/run_freesasa.py
    --config "$CONFIG"
    --output-dir "$OUTPUT_DIR"
)

echo "[freesasa] Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "$WORKSPACE/logs/tool.log"

# --- Verify output ----------------------------------------------------------
if [ ! -f "$OUTPUT_DIR/output.json" ]; then
    echo "[freesasa] ERROR: output.json not found in $OUTPUT_DIR" >&2
    exit 1
fi

echo "[freesasa] Execution complete."
