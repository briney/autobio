#!/usr/bin/env bash
# run.sh — Phase 2: flex-ddG multi-step workflow for binding DDG prediction
#
# Three steps sharing a single backrub ensemble:
#   1. Generate backrub conformational ensemble from WT structure
#   2. Score WT: repack (NATAA) + minimize each ensemble member
#   3. Score mutant: apply mutations + repack + minimize each ensemble member
#
# Scoring both WT and mutant from the same ensemble is critical for
# noise cancellation in DDG computation.
set -euo pipefail

WORKSPACE="$1"
CONFIG="$WORKSPACE/config.json"
OUTPUT_DIR="$WORKSPACE/outputs/raw"
ENSEMBLE_DIR="$OUTPUT_DIR/ensemble"

# Read parameters from config.json
STRUCTURE_PATH=$(jq -r '.structure_path' "$CONFIG")
DATABASE_PATH=$(jq -r '.database_path' "$CONFIG")
CHAINS_TO_MOVE=$(jq -r '.chains_to_move' "$CONFIG")
NSTRUCT=$(jq -r '.nstruct // 35' "$CONFIG")
BACKRUB_TRIALS=$(jq -r '.backrub_trials // 35000' "$CONFIG")
MAX_MIN_ITER=$(jq -r '.max_minimization_iter // 5000' "$CONFIG")

# Generate mutation resfile if no custom resfile was provided
RESFILE_PATH=$(jq -r '.resfile_path // empty' "$CONFIG" 2>/dev/null || true)
if [ -z "$RESFILE_PATH" ] || [ "$RESFILE_PATH" = "null" ]; then
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

# Generate NATAA-only resfile for WT scoring (repack without mutations)
WT_RESFILE="$WORKSPACE/inputs/wt.resfile"
printf 'NATAA\nstart\n' > "$WT_RESFILE"

mkdir -p "$ENSEMBLE_DIR"

# ── Step 1/3: Generate backrub ensemble ──────────────────────────────────────
echo "[flex-ddg] Step 1/3: Generating backrub ensemble (nstruct=$NSTRUCT, trials=$BACKRUB_TRIALS)"
CMD_ENSEMBLE=(
    rosetta_scripts
    -database "$DATABASE_PATH"
    -s "$STRUCTURE_PATH"
    -parser:protocol /opt/tool/xml/backrub_ensemble.xml
    -parser:script_vars
        "number_backrub_trials=$BACKRUB_TRIALS"
    -nstruct "$NSTRUCT"
    -out:path:pdb "$ENSEMBLE_DIR"
    -out:file:scorefile "$ENSEMBLE_DIR/backrub_score.sc"
    -out:no_nstruct_label
)

echo "[flex-ddg] Running: ${CMD_ENSEMBLE[*]}"
"${CMD_ENSEMBLE[@]}" 2>&1 | tee "$WORKSPACE/logs/step1_backrub.log"

# Collect ensemble PDBs into a list file for downstream steps
ENSEMBLE_LIST="$WORKSPACE/inputs/ensemble.list"
ls "$ENSEMBLE_DIR"/*.pdb | sort > "$ENSEMBLE_LIST"
ENSEMBLE_COUNT=$(wc -l < "$ENSEMBLE_LIST")
echo "[flex-ddg] Generated $ENSEMBLE_COUNT ensemble members"

# ── Step 2/3: Score wild-type ────────────────────────────────────────────────
echo "[flex-ddg] Step 2/3: Scoring wild-type ensemble"
CMD_WT=(
    rosetta_scripts
    -database "$DATABASE_PATH"
    -l "$ENSEMBLE_LIST"
    -parser:protocol /opt/tool/xml/repack_minimize.xml
    -parser:script_vars
        "max_minimization_iter=$MAX_MIN_ITER"
    -resfile "$WT_RESFILE"
    -nstruct 1
    -ex1 -ex2
    -out:prefix wt_backrub_
    -out:path:pdb "$OUTPUT_DIR"
    -out:file:scorefile "$OUTPUT_DIR/wt_score.sc"
    -out:no_nstruct_label
)

echo "[flex-ddg] Running: ${CMD_WT[*]}"
"${CMD_WT[@]}" 2>&1 | tee "$WORKSPACE/logs/step2_wt.log"

# ── Step 3/3: Score mutant ───────────────────────────────────────────────────
echo "[flex-ddg] Step 3/3: Scoring mutant ensemble"
CMD_MUT=(
    rosetta_scripts
    -database "$DATABASE_PATH"
    -l "$ENSEMBLE_LIST"
    -parser:protocol /opt/tool/xml/repack_minimize.xml
    -parser:script_vars
        "max_minimization_iter=$MAX_MIN_ITER"
    -resfile "$RESFILE_PATH"
    -nstruct 1
    -ex1 -ex2
    -out:prefix mut_backrub_
    -out:path:pdb "$OUTPUT_DIR"
    -out:file:scorefile "$OUTPUT_DIR/mut_score.sc"
    -out:no_nstruct_label
)

echo "[flex-ddg] Running: ${CMD_MUT[*]}"
"${CMD_MUT[@]}" 2>&1 | tee "$WORKSPACE/logs/step3_mut.log"

echo "[flex-ddg] All steps complete."
