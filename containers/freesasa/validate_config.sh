#!/usr/bin/env bash
# validate_config.sh — Phase 1: verify config.json for FreeSASA
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

# Required field: mode
MODE=$(jq -r '.mode // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -z "$MODE" ] || [ "$MODE" = "null" ]; then
    error "config.json missing required field 'mode'"
elif [ "$MODE" != "bsa" ] && [ "$MODE" != "sasa" ]; then
    error "mode must be 'bsa' or 'sasa', got: $MODE"
fi

# BSA mode: require partner1 and partner2
if [ "$MODE" = "bsa" ]; then
    PARTNER1=$(jq -r '.partner1 // empty' "$CONFIG_FILE" 2>/dev/null || true)
    PARTNER2=$(jq -r '.partner2 // empty' "$CONFIG_FILE" 2>/dev/null || true)
    if [ -z "$PARTNER1" ] || [ "$PARTNER1" = "null" ]; then
        error "BSA mode requires 'partner1' (comma-separated chain IDs)"
    fi
    if [ -z "$PARTNER2" ] || [ "$PARTNER2" = "null" ]; then
        error "BSA mode requires 'partner2' (comma-separated chain IDs)"
    fi
fi

# Validate algorithm if present
ALGORITHM=$(jq -r '.algorithm // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$ALGORITHM" ] && [ "$ALGORITHM" != "null" ]; then
    if [ "$ALGORITHM" != "LeeRichards" ] && [ "$ALGORITHM" != "ShrakeRupley" ]; then
        error "algorithm must be 'LeeRichards' or 'ShrakeRupley', got: $ALGORITHM"
    fi
fi

# Validate probe_radius is a positive number if present
PROBE=$(jq -r '.probe_radius // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$PROBE" ] && [ "$PROBE" != "null" ]; then
    if ! echo "$PROBE" | grep -qE '^[0-9]+\.?[0-9]*$'; then
        error "probe_radius must be a positive number, got: $PROBE"
    elif [ "$(echo "$PROBE <= 0" | bc -l)" -eq 1 ]; then
        error "probe_radius must be positive, got: $PROBE"
    fi
fi

if [ "$ERRORS" -gt 0 ]; then
    echo "Config validation failed with $ERRORS error(s)." >&2
    exit 1
fi

echo "Config validation passed."
exit 0
