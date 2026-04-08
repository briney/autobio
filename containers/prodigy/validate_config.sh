#!/usr/bin/env bash
# validate_config.sh — Phase 1: verify config.json for PRODIGY
set -euo pipefail

CONFIG_FILE="$1"
ERRORS=0

error() {
    echo "ERROR: $1" >&2
    ERRORS=$((ERRORS + 1))
}

# Required field: structure_path
if ! jq -e '.structure_path' "$CONFIG_FILE" > /dev/null 2>&1; then
    error "config.json missing required field 'structure_path'"
fi

# Validate structure file exists
STRUCTURE_PATH=$(jq -r '.structure_path // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$STRUCTURE_PATH" ] && [ "$STRUCTURE_PATH" != "null" ]; then
    if [ ! -f "$STRUCTURE_PATH" ]; then
        error "Structure file not found: $STRUCTURE_PATH"
    fi
fi

# Validate selection is a non-empty string if present
SELECTION=$(jq -r '.selection // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$SELECTION" ] && [ "$SELECTION" = "null" ]; then
    SELECTION=""
fi
# selection can be empty/null (means all inter-chain), so no error needed

# Validate temperature is a number if present
TEMP=$(jq -r '.temperature // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$TEMP" ] && [ "$TEMP" != "null" ]; then
    if ! echo "$TEMP" | grep -qE '^-?[0-9]+\.?[0-9]*$'; then
        error "temperature must be a number, got: $TEMP"
    fi
fi

# Validate distance_cutoff is a positive number if present
CUTOFF=$(jq -r '.distance_cutoff // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$CUTOFF" ] && [ "$CUTOFF" != "null" ]; then
    if ! echo "$CUTOFF" | grep -qE '^[0-9]+\.?[0-9]*$'; then
        error "distance_cutoff must be a positive number, got: $CUTOFF"
    fi
fi

if [ "$ERRORS" -gt 0 ]; then
    echo "Config validation failed with $ERRORS error(s)." >&2
    exit 1
fi

echo "Config validation passed."
exit 0
