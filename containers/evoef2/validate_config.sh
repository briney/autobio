#!/usr/bin/env bash
# validate_config.sh — Phase 1: verify config.json for EvoEF2
set -euo pipefail

CONFIG_FILE="$1"
ERRORS=0

error() {
    echo "ERROR: $1" >&2
    ERRORS=$((ERRORS + 1))
}

# ---------------------------------------------------------------------------
# Required fields
# ---------------------------------------------------------------------------

for field in command structure_path evoef2_bin; do
    if ! jq -e ".$field" "$CONFIG_FILE" > /dev/null 2>&1; then
        error "config.json missing required field '$field'"
    fi
done

# ---------------------------------------------------------------------------
# Validate command
# ---------------------------------------------------------------------------

COMMAND=$(jq -r '.command // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$COMMAND" ] && [ "$COMMAND" != "null" ]; then
    case "$COMMAND" in
        RepairStructure|ComputeBinding|BuildMutant) ;;
        *) error "Unknown command: '$COMMAND'. Expected one of: RepairStructure, ComputeBinding, BuildMutant" ;;
    esac
fi

# ---------------------------------------------------------------------------
# Validate referenced files exist
# ---------------------------------------------------------------------------

STRUCTURE_PATH=$(jq -r '.structure_path // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$STRUCTURE_PATH" ] && [ "$STRUCTURE_PATH" != "null" ]; then
    if [ ! -f "$STRUCTURE_PATH" ]; then
        error "Structure file not found: $STRUCTURE_PATH"
    fi
fi

EVOEF2_BIN=$(jq -r '.evoef2_bin // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$EVOEF2_BIN" ] && [ "$EVOEF2_BIN" != "null" ]; then
    if [ ! -x "$EVOEF2_BIN" ]; then
        error "EvoEF2 binary not found or not executable: $EVOEF2_BIN"
    fi
fi

# ---------------------------------------------------------------------------
# Command-specific validation
# ---------------------------------------------------------------------------

if [ "$COMMAND" = "BuildMutant" ]; then
    if ! jq -e '.mutant_file' "$CONFIG_FILE" > /dev/null 2>&1; then
        error "BuildMutant requires 'mutant_file' in config.json"
    else
        MUTANT_FILE=$(jq -r '.mutant_file // empty' "$CONFIG_FILE" 2>/dev/null || true)
        if [ -n "$MUTANT_FILE" ] && [ "$MUTANT_FILE" != "null" ]; then
            if [ ! -f "$MUTANT_FILE" ]; then
                error "Mutant file not found: $MUTANT_FILE"
            fi
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

if [ "$ERRORS" -gt 0 ]; then
    echo "Config validation failed with $ERRORS error(s)." >&2
    exit 1
fi

echo "Config validation passed."
exit 0
