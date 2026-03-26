#!/usr/bin/env bash
# run.sh — Phase 2: minimize a structure with Rosetta MinMover
set -euo pipefail

WORKSPACE="$1"
CONFIG="$WORKSPACE/config.json"
OUTPUT_DIR="$WORKSPACE/outputs/raw"

# Read parameters from config.json
STRUCTURE_PATH=$(jq -r '.structure_path' "$CONFIG")
DATABASE_PATH=$(jq -r '.database_path' "$CONFIG")
XML_PATH=$(jq -r '.xml_path' "$CONFIG")
NSTRUCT=$(jq -r '.nstruct // 1' "$CONFIG")

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

echo "[rosetta-minimize] Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "$WORKSPACE/logs/tool.log"
