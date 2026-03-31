#!/usr/bin/env bash
# run.sh — Phase 2: execute StaB-ddG binding ddG prediction
set -euo pipefail

WORKSPACE="$1"
CONFIG="$WORKSPACE/config.json"
OUTPUT_DIR="$WORKSPACE/outputs/raw"

# --- Required fields --------------------------------------------------------
PDB_PATH=$(jq -r '.pdb_path' "$CONFIG")
MUTATIONS=$(jq -r '.mutations' "$CONFIG")
CHAINS=$(jq -r '.chains' "$CONFIG")
CHECKPOINT_PATH=$(jq -r '.checkpoint_path' "$CONFIG")

# --- Device resolution ------------------------------------------------------
DEVICE=$(jq -r '.device // "auto"' "$CONFIG")
if [ "$DEVICE" = "auto" ]; then
    if nvidia-smi > /dev/null 2>&1; then
        DEVICE="cuda"
    else
        DEVICE="cpu"
    fi
    echo "[stabddg] Auto-detected device: $DEVICE"
fi

# --- Optional fields with defaults ------------------------------------------
MC_SAMPLES=$(jq -r '.mc_samples // 20' "$CONFIG")
NOISE_LEVEL=$(jq -r '.noise_level // 0.1' "$CONFIG")
BATCH_SIZE=$(jq -r '.batch_size // 10000' "$CONFIG")
TRIALS=$(jq -r '.trials // 1' "$CONFIG")
SEED=$(jq -r '.seed // 0' "$CONFIG")

# --- Build command ----------------------------------------------------------
CMD=(
    python /app/stabddg/run_stabddg.py
    --pdb_path "$PDB_PATH"
    --mutation "$MUTATIONS"
    --chains "$CHAINS"
    --checkpoint "$CHECKPOINT_PATH"
    --output_dir "$OUTPUT_DIR"
    --device "$DEVICE"
    --mc_samples "$MC_SAMPLES"
    --noise_level "$NOISE_LEVEL"
    --batch_size "$BATCH_SIZE"
    --trials "$TRIALS"
    --seed "$SEED"
)

echo "[stabddg] Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "$WORKSPACE/logs/tool.log"

# --- Collect output CSV to raw/ --------------------------------------------
# StaB-ddG in single-PDB mode writes to {pdb_basename}_output/ next to the
# input PDB, ignoring --output_dir. Find and copy the CSV to the expected
# raw output location.
if [ ! -f "$OUTPUT_DIR/output.csv" ]; then
    echo "[stabddg] Output not in expected location, searching workspace..."
    CSV_FILE=$(find "$WORKSPACE" -name "output.csv" -type f 2>/dev/null | head -1)
    if [ -n "$CSV_FILE" ]; then
        echo "[stabddg] Found output at: $CSV_FILE"
        cp "$CSV_FILE" "$OUTPUT_DIR/output.csv"
    else
        echo "[stabddg] ERROR: No output.csv found anywhere in workspace" >&2
        exit 1
    fi
fi

echo "[stabddg] Execution complete."
