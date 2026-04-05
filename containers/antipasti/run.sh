#!/usr/bin/env bash
# run.sh — Phase 2: execute ANTIPASTI binding affinity prediction
set -euo pipefail

WORKSPACE="$1"
CONFIG="$WORKSPACE/config.json"
OUTPUT_DIR="$WORKSPACE/outputs/raw"

# --- Required fields --------------------------------------------------------
PDB_PATH=$(jq -r '.pdb_path' "$CONFIG")
HEAVY_CHAIN=$(jq -r '.heavy_chain' "$CONFIG")
LIGHT_CHAIN=$(jq -r '.light_chain' "$CONFIG")
ANTIGEN_CHAINS=$(jq -c '.antigen_chains' "$CONFIG")
CHECKPOINT_PATH=$(jq -r '.checkpoint_path' "$CONFIG")
ANTIPASTI_DIR=$(jq -r '.antipasti_dir' "$CONFIG")

# --- Optional fields with defaults ------------------------------------------
MODES=$(jq -r '.modes // "all"' "$CONFIG")

# --- Build command ----------------------------------------------------------
CMD=(
    python /opt/tool/inference.py
    --pdb_path "$PDB_PATH"
    --heavy_chain "$HEAVY_CHAIN"
    --light_chain "$LIGHT_CHAIN"
    --antigen_chains "$ANTIGEN_CHAINS"
    --checkpoint_path "$CHECKPOINT_PATH"
    --antipasti_dir "$ANTIPASTI_DIR"
    --output_dir "$OUTPUT_DIR"
    --modes "$MODES"
)

echo "[antipasti] Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "$WORKSPACE/logs/tool.log"

# --- Verify output ----------------------------------------------------------
if [ ! -f "$OUTPUT_DIR/output.json" ]; then
    echo "[antipasti] ERROR: output.json not found in $OUTPUT_DIR" >&2
    exit 1
fi

echo "[antipasti] Execution complete."
