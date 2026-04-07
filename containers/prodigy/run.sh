#!/usr/bin/env bash
# run.sh — Phase 2: execute PRODIGY binding affinity prediction
set -euo pipefail

WORKSPACE="$1"
CONFIG="$WORKSPACE/config.json"
OUTPUT_DIR="$WORKSPACE/outputs/raw"

# --- Build command ----------------------------------------------------------
CMD=(
    python /opt/tool/run_prodigy.py
    --config "$CONFIG"
    --output-dir "$OUTPUT_DIR"
)

echo "[prodigy] Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "$WORKSPACE/logs/tool.log"

# --- Verify output ----------------------------------------------------------
if [ ! -f "$OUTPUT_DIR/output.json" ]; then
    echo "[prodigy] ERROR: output.json not found in $OUTPUT_DIR" >&2
    exit 1
fi

echo "[prodigy] Execution complete."
