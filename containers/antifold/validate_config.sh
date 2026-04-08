#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="$1"

# Required: mode
MODE=$(jq -r '.mode' "$CONFIG_FILE")
if [[ "$MODE" != "design" && "$MODE" != "score" ]]; then
    echo "ERROR: mode must be 'design' or 'score', got '$MODE'" >&2
    exit 1
fi

# Required: structure_path
STRUCTURE_PATH=$(jq -r '.structure_path' "$CONFIG_FILE")
if [ ! -f "$STRUCTURE_PATH" ]; then
    echo "ERROR: structure file not found: $STRUCTURE_PATH" >&2
    exit 1
fi

# Required: at least one of heavy_chain, light_chain
HC=$(jq -r '.heavy_chain // empty' "$CONFIG_FILE" 2>/dev/null || true)
LC=$(jq -r '.light_chain // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -z "$HC" ] && [ -z "$LC" ]; then
    echo "ERROR: at least one of 'heavy_chain' or 'light_chain' must be provided" >&2
    exit 1
fi

# Optional: validate regions
if jq -e '.regions' "$CONFIG_FILE" > /dev/null 2>&1; then
    VALID_REGIONS="CDRH1 CDRH2 CDRH3 CDRL1 CDRL2 CDRL3 FWH1 FWH2 FWH3 FWH4 FWL1 FWL2 FWL3 FWL4"
    for region in $(jq -r '.regions[]' "$CONFIG_FILE"); do
        if ! echo "$VALID_REGIONS" | grep -qw "$region"; then
            echo "ERROR: invalid region '$region'. Valid: $VALID_REGIONS" >&2
            exit 1
        fi
    done
fi

# Mode-specific validation
if [[ "$MODE" == "design" ]]; then
    NUM_SEQ=$(jq -r '.num_sequences // 1' "$CONFIG_FILE")
    if ! [[ "$NUM_SEQ" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: num_sequences must be a positive integer, got '$NUM_SEQ'" >&2
        exit 1
    fi

    TEMP=$(jq -r '.temperature // 0.2' "$CONFIG_FILE")
    if ! echo "$TEMP > 0" | bc -l > /dev/null 2>&1; then
        echo "ERROR: temperature must be positive, got '$TEMP'" >&2
        exit 1
    fi
fi

echo "Config validation passed."
exit 0
