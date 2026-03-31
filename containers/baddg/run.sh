#!/usr/bin/env bash
# run.sh — Phase 2: execute BA-ddG binding ddG prediction
set -euo pipefail

WORKSPACE="$1"
CONFIG="$WORKSPACE/config.json"
OUTPUT_DIR="$WORKSPACE/outputs/raw"

# --- Required fields --------------------------------------------------------
PDB_PATH=$(jq -r '.pdb_path' "$CONFIG")
MUTATIONS=$(jq -r '.mutations' "$CONFIG")
CHAINS=$(jq -r '.chains' "$CONFIG")
MPNN_CHECKPOINT=$(jq -r '.mpnn_checkpoint_path' "$CONFIG")
DDG_CHECKPOINT=$(jq -r '.ddg_checkpoint_path' "$CONFIG")

# --- Device resolution ------------------------------------------------------
DEVICE=$(jq -r '.device // "auto"' "$CONFIG")
if [ "$DEVICE" = "auto" ]; then
    if nvidia-smi > /dev/null 2>&1; then
        DEVICE="cuda"
    else
        DEVICE="cpu"
    fi
    echo "[baddg] Auto-detected device: $DEVICE"
fi

# --- Optional fields with defaults ------------------------------------------
N_FOLDS=$(jq -r '.n_folds // 3' "$CONFIG")
SEED=$(jq -r '.seed // 0' "$CONFIG")

# --- Build command ----------------------------------------------------------
CMD=(
    python /opt/tool/inference.py
    --pdb_path "$PDB_PATH"
    --mutations "$MUTATIONS"
    --chains "$CHAINS"
    --mpnn_checkpoint "$MPNN_CHECKPOINT"
    --ddg_checkpoint "$DDG_CHECKPOINT"
    --output_dir "$OUTPUT_DIR"
    --device "$DEVICE"
    --n_folds "$N_FOLDS"
    --seed "$SEED"
)

echo "[baddg] Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "$WORKSPACE/logs/tool.log"

# Verify output was produced
if [ ! -f "$OUTPUT_DIR/output.csv" ]; then
    echo "[baddg] ERROR: No output.csv found in $OUTPUT_DIR" >&2
    exit 1
fi

echo "[baddg] Execution complete."
