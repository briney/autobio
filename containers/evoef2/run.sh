#!/usr/bin/env bash
# run.sh — Phase 2: execute EvoEF2 command
#
# NOTE: EvoEF2 writes output files (e.g., *_Repair.pdb, *_Model_*.pdb) to the
# current working directory, NOT next to the input PDB. We `cd` to the raw
# output directory before running so output files land there directly.
set -euo pipefail

WORKSPACE="$1"
CONFIG="$WORKSPACE/config.json"
OUTPUT_DIR="$WORKSPACE/outputs/raw"

# Read common parameters from config.json
COMMAND=$(jq -r '.command' "$CONFIG")
STRUCTURE_PATH=$(jq -r '.structure_path' "$CONFIG")
EVOEF2_BIN=$(jq -r '.evoef2_bin' "$CONFIG")

echo "[evoef2] Command: $COMMAND"
echo "[evoef2] Structure: $STRUCTURE_PATH"

# ---------------------------------------------------------------------------
# RepairStructure
# ---------------------------------------------------------------------------
if [ "$COMMAND" = "RepairStructure" ]; then
    cd "$OUTPUT_DIR"

    CMD=("$EVOEF2_BIN" --command=RepairStructure --pdb="$STRUCTURE_PATH")

    echo "[evoef2] Running: ${CMD[*]}"
    "${CMD[@]}" 2>&1 | tee "$WORKSPACE/logs/tool.log"

# ---------------------------------------------------------------------------
# ComputeBinding
# ---------------------------------------------------------------------------
elif [ "$COMMAND" = "ComputeBinding" ]; then
    REPAIR=$(jq -r '.repair // true' "$CONFIG")
    SPLIT_CHAINS=$(jq -r '.split_chains // empty' "$CONFIG")

    BINDING_PDB="$STRUCTURE_PATH"

    # Step 1: Optionally repair structure before computing binding energy
    if [ "$REPAIR" = "true" ]; then
        cd "$OUTPUT_DIR"
        echo "[evoef2] Running RepairStructure before ComputeBinding..."
        "$EVOEF2_BIN" --command=RepairStructure --pdb="$STRUCTURE_PATH" \
            2>&1 | tee "$WORKSPACE/logs/repair.log"

        # Find the repaired PDB (written to CWD = OUTPUT_DIR)
        PDBID=$(basename "$STRUCTURE_PATH" .pdb)
        REPAIRED="$OUTPUT_DIR/${PDBID}_Repair.pdb"
        if [ -f "$REPAIRED" ]; then
            BINDING_PDB="$REPAIRED"
            echo "[evoef2] Using repaired structure: $REPAIRED"
        else
            echo "[evoef2] WARNING: RepairStructure did not produce expected output, using original" >&2
        fi
    fi

    # Step 2: Compute binding energy (stay in OUTPUT_DIR)
    cd "$OUTPUT_DIR"
    CMD=("$EVOEF2_BIN" --command=ComputeBinding --pdb="$BINDING_PDB")

    if [ -n "$SPLIT_CHAINS" ] && [ "$SPLIT_CHAINS" != "null" ]; then
        CMD+=(--split_chains="$SPLIT_CHAINS")
    fi

    echo "[evoef2] Running: ${CMD[*]}"
    "${CMD[@]}" 2>&1 | tee "$OUTPUT_DIR/binding_output.txt"

# ---------------------------------------------------------------------------
# BuildMutant
# ---------------------------------------------------------------------------
elif [ "$COMMAND" = "BuildMutant" ]; then
    MUTANT_FILE=$(jq -r '.mutant_file' "$CONFIG")
    cd "$OUTPUT_DIR"

    CMD=("$EVOEF2_BIN" --command=BuildMutant --pdb="$STRUCTURE_PATH" --mutant_file="$MUTANT_FILE")

    echo "[evoef2] Running: ${CMD[*]}"
    "${CMD[@]}" 2>&1 | tee "$WORKSPACE/logs/tool.log"

else
    echo "[evoef2] ERROR: Unknown command: $COMMAND" >&2
    exit 1
fi

echo "[evoef2] Done."
