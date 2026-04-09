#!/usr/bin/env bash
# validate_config.sh — Phase 1: verify config.json for MPNN scoring
set -euo pipefail

CONFIG_FILE="$1"

# Required fields
for field in structure_path model_type checkpoint_path mode; do
    if ! jq -e ".$field" "$CONFIG_FILE" > /dev/null 2>&1; then
        echo "ERROR: config.json missing required field '$field'" >&2
        exit 1
    fi
done

# Validate mode
MODE=$(jq -r '.mode' "$CONFIG_FILE")
if [ "$MODE" != "score" ]; then
    echo "ERROR: mode must be 'score', got '$MODE'" >&2
    exit 1
fi

# Validate model_type
MODEL_TYPE=$(jq -r '.model_type' "$CONFIG_FILE")
if [[ "$MODEL_TYPE" != "protein_mpnn" && "$MODEL_TYPE" != "ligand_mpnn" ]]; then
    echo "ERROR: model_type must be 'protein_mpnn' or 'ligand_mpnn', got '$MODEL_TYPE'" >&2
    exit 1
fi

# Validate structure file exists
STRUCTURE_PATH=$(jq -r '.structure_path' "$CONFIG_FILE")
if [ ! -f "$STRUCTURE_PATH" ]; then
    echo "ERROR: structure file not found: $STRUCTURE_PATH" >&2
    exit 1
fi

# Validate checkpoint file exists
CHECKPOINT_PATH=$(jq -r '.checkpoint_path' "$CONFIG_FILE")
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: checkpoint file not found: $CHECKPOINT_PATH" >&2
    exit 1
fi

echo "Config validation passed."
exit 0
