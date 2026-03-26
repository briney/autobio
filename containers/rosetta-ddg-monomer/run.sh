#!/usr/bin/env bash
# run.sh — Phase 2: predict stability DDG with Rosetta ddg_monomer
set -euo pipefail

WORKSPACE="$1"
CONFIG="$WORKSPACE/config.json"
OUTPUT_DIR="$WORKSPACE/outputs/raw"

# Read parameters from config.json
STRUCTURE_PATH=$(jq -r '.structure_path' "$CONFIG")
DATABASE_PATH=$(jq -r '.database_path' "$CONFIG")
ITERATIONS=$(jq -r '.ddg_iterations // 50' "$CONFIG")
LOCAL_OPT_ONLY=$(jq -r '.local_opt_only // "false"' "$CONFIG")

# Generate the mutation file from the mutation_list in config
# Format: total_mutations\n<wt_aa> <resnum> <mut_aa>\n...
python3 -c "
import json, sys
config = json.load(open('$CONFIG'))
mutations = config.get('mutation_list', config.get('mutations', []))
out = ['total ' + str(len(mutations))]
for m in mutations:
    # Parse 'A42F' format: first char = wt, last char = mut, middle = resnum
    wt = m[0]
    mut = m[-1]
    resnum = m[1:-1]
    out.append(f'{wt} {resnum} {mut}')
    out.append('')  # blank line between mutations
with open('$WORKSPACE/inputs/mutations.mut', 'w') as f:
    f.write('\n'.join(out) + '\n')
"

# Build command
CMD=(
    ddg_monomer
    -database "$DATABASE_PATH"
    -in:file:s "$STRUCTURE_PATH"
    -ddg::mut_file "$WORKSPACE/inputs/mutations.mut"
    -ddg::iterations "$ITERATIONS"
    -ddg::dump_pdbs false
    -ddg::local_opt_only "$LOCAL_OPT_ONLY"
    -ddg::min_cst true
    -ddg::mean false
    -ddg::min true
    -ddg::sc_min_only false
    -ddg::ramp_repulsive true
    -mute all
    -unmute apps.public.ddg.ddg_monomer protocols.jd2
    -out:path:pdb "$OUTPUT_DIR"
    -out:file:scorefile "$OUTPUT_DIR/score.sc"
)

echo "[rosetta-ddg-monomer] Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "$WORKSPACE/logs/tool.log"
