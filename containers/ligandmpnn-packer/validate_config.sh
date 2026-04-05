#!/usr/bin/env bash
# validate_config.sh — Phase 1: verify config.json for LigandMPNN packer
set -euo pipefail

CONFIG_FILE="$1"

# Required fields
for field in structure_path mutations checkpoint_sc; do
    if ! jq -e ".$field" "$CONFIG_FILE" > /dev/null 2>&1; then
        echo "ERROR: config.json missing required field '$field'" >&2
        exit 1
    fi
done

# Validate mutations is a non-empty array
MUTATION_COUNT=$(jq '.mutations | length' "$CONFIG_FILE")
if [ "$MUTATION_COUNT" -eq 0 ]; then
    echo "ERROR: mutations array must not be empty" >&2
    exit 1
fi

# Validate structure file exists
STRUCTURE_PATH=$(jq -r '.structure_path' "$CONFIG_FILE")
if [ ! -f "$STRUCTURE_PATH" ]; then
    echo "ERROR: structure file not found: $STRUCTURE_PATH" >&2
    exit 1
fi

# Validate checkpoint files exist
CHECKPOINT_SC=$(jq -r '.checkpoint_sc' "$CONFIG_FILE")
if [ ! -f "$CHECKPOINT_SC" ]; then
    echo "ERROR: sidechain packing checkpoint not found: $CHECKPOINT_SC" >&2
    exit 1
fi

echo "Config validation passed."
exit 0
