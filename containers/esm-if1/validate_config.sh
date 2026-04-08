#!/usr/bin/env bash
# validate_config.sh — Phase 1: verify config.json for ESM-IF1 tools
set -euo pipefail

CONFIG_FILE="$1"

# Required field: mode
if ! jq -e '.mode' "$CONFIG_FILE" > /dev/null 2>&1; then
    echo "ERROR: config.json missing required field 'mode'" >&2
    exit 1
fi

MODE=$(jq -r '.mode' "$CONFIG_FILE")
if [[ "$MODE" != "design" && "$MODE" != "score" ]]; then
    echo "ERROR: mode must be 'design' or 'score', got '$MODE'" >&2
    exit 1
fi

# Required field: structure_path (both modes)
if ! jq -e '.structure_path' "$CONFIG_FILE" > /dev/null 2>&1; then
    echo "ERROR: config.json missing required field 'structure_path'" >&2
    exit 1
fi

STRUCTURE_PATH=$(jq -r '.structure_path' "$CONFIG_FILE")
if [ ! -f "$STRUCTURE_PATH" ]; then
    echo "ERROR: structure file not found: $STRUCTURE_PATH" >&2
    exit 1
fi

# Mode-specific validation
if [[ "$MODE" == "design" ]]; then
    # num_sequences must be a positive integer
    NUM_SEQ=$(jq -r '.num_sequences // 1' "$CONFIG_FILE")
    if ! [[ "$NUM_SEQ" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: num_sequences must be a positive integer, got '$NUM_SEQ'" >&2
        exit 1
    fi

    # temperature must be positive
    TEMP=$(jq -r '.temperature // 0.1' "$CONFIG_FILE")
    if ! echo "$TEMP > 0" | bc -l > /dev/null 2>&1; then
        echo "ERROR: temperature must be a positive number, got '$TEMP'" >&2
        exit 1
    fi
elif [[ "$MODE" == "score" ]]; then
    # sequences is required for score mode
    if ! jq -e '.sequences' "$CONFIG_FILE" > /dev/null 2>&1; then
        echo "ERROR: config.json missing required field 'sequences' for score mode" >&2
        exit 1
    fi
fi

echo "Config validation passed."
exit 0
