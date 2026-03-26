#!/usr/bin/env bash
# run.sh — Phase 2: score a structure with Rosetta score_jd2
set -euo pipefail

WORKSPACE="$1"
CONFIG="$WORKSPACE/config.json"
OUTPUT_DIR="$WORKSPACE/outputs/raw"

# Read parameters from config.json
STRUCTURE_PATH=$(jq -r '.structure_path' "$CONFIG")
DATABASE_PATH=$(jq -r '.database_path' "$CONFIG")
SCORE_FUNCTION=$(jq -r '.score_function // "ref2015"' "$CONFIG")

# Build command
CMD=(
    score_jd2
    -database "$DATABASE_PATH"
    -in:file:s "$STRUCTURE_PATH"
    -out:file:scorefile "$OUTPUT_DIR/score.sc"
    -scorefxn "$SCORE_FUNCTION"
    -out:no_nstruct_label
)

# Optional: extra rotamer sampling
EX1=$(jq -r '.ex1 // empty' "$CONFIG" 2>/dev/null || true)
if [ -n "$EX1" ] && [ "$EX1" = "true" ]; then
    CMD+=(-ex1)
fi

EX2=$(jq -r '.ex2 // empty' "$CONFIG" 2>/dev/null || true)
if [ -n "$EX2" ] && [ "$EX2" = "true" ]; then
    CMD+=(-ex2)
fi

echo "[rosetta-score] Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "$WORKSPACE/logs/tool.log"
