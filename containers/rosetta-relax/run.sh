#!/usr/bin/env bash
# run.sh — Phase 2: relax a structure with Rosetta FastRelax
set -euo pipefail

WORKSPACE="$1"
CONFIG="$WORKSPACE/config.json"
OUTPUT_DIR="$WORKSPACE/outputs/raw"

# Read parameters from config.json
STRUCTURE_PATH=$(jq -r '.structure_path' "$CONFIG")
DATABASE_PATH=$(jq -r '.database_path' "$CONFIG")
XML_PATH=$(jq -r '.xml_path' "$CONFIG")
NSTRUCT=$(jq -r '.nstruct // 5' "$CONFIG")
SCORE_FUNCTION=$(jq -r '.score_function // "ref2015"' "$CONFIG")

# Build command
CMD=(
    rosetta_scripts
    -database "$DATABASE_PATH"
    -in:file:s "$STRUCTURE_PATH"
    -parser:protocol "$XML_PATH"
    -out:path:pdb "$OUTPUT_DIR"
    -out:file:scorefile "$OUTPUT_DIR/score.sc"
    -nstruct "$NSTRUCT"
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

echo "[rosetta-relax] Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "$WORKSPACE/logs/tool.log"
