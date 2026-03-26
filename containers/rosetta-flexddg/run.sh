#!/usr/bin/env bash
# run.sh — Phase 2: run flex-ddG protocol for binding DDG prediction
set -euo pipefail

WORKSPACE="$1"
CONFIG="$WORKSPACE/config.json"
OUTPUT_DIR="$WORKSPACE/outputs/raw"

# Read parameters from config.json
STRUCTURE_PATH=$(jq -r '.structure_path' "$CONFIG")
DATABASE_PATH=$(jq -r '.database_path' "$CONFIG")
XML_PATH=$(jq -r '.xml_path' "$CONFIG")
CHAINS_TO_MOVE=$(jq -r '.chains_to_move' "$CONFIG")
NSTRUCT=$(jq -r '.nstruct // 35' "$CONFIG")
BACKRUB_TRIALS=$(jq -r '.backrub_trials // 35000' "$CONFIG")
MAX_MIN_ITER=$(jq -r '.max_minimization_iter // 5000' "$CONFIG")

# Generate resfile from mutations if no custom resfile was provided
RESFILE_PATH=$(jq -r '.resfile_path // empty' "$CONFIG" 2>/dev/null || true)
if [ -z "$RESFILE_PATH" ] || [ "$RESFILE_PATH" = "null" ]; then
    # Generate resfile from mutation_list
    python3 -c "
import json
config = json.load(open('$CONFIG'))
mutations = config.get('mutation_list', config.get('mutations', []))
lines = ['NATAA', 'start']
for m in mutations:
    wt = m[0]
    mut = m[-1]
    resnum = m[1:-1]
    lines.append(f'{resnum} A PIKAA {mut}')
with open('$WORKSPACE/inputs/mutations.resfile', 'w') as f:
    f.write('\n'.join(lines) + '\n')
"
    RESFILE_PATH="$WORKSPACE/inputs/mutations.resfile"
fi

# Build command
CMD=(
    rosetta_scripts
    -database "$DATABASE_PATH"
    -s "$STRUCTURE_PATH"
    -parser:protocol "$XML_PATH"
    -parser:script_vars
        "number_backrub_trials=$BACKRUB_TRIALS"
        "max_minimization_iter=$MAX_MIN_ITER"
        "chainstomove=$CHAINS_TO_MOVE"
    -resfile "$RESFILE_PATH"
    -nstruct "$NSTRUCT"
    -ex1 -ex2
    -out:path:pdb "$OUTPUT_DIR"
    -out:file:scorefile "$OUTPUT_DIR/score.sc"
    -out:no_nstruct_label
)

echo "[rosetta-flexddg] Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "$WORKSPACE/logs/tool.log"
