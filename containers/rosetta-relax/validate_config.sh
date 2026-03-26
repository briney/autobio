#!/usr/bin/env bash
# validate_config.sh — Phase 1: verify config.json for Rosetta relax
set -euo pipefail

CONFIG_FILE="$1"
ERRORS=0

error() {
    echo "ERROR: $1" >&2
    ERRORS=$((ERRORS + 1))
}

# Required fields
for field in structure_path database_path xml_path; do
    if ! jq -e ".$field" "$CONFIG_FILE" > /dev/null 2>&1; then
        error "config.json missing required field '$field'"
    fi
done

# Validate referenced files exist
STRUCTURE_PATH=$(jq -r '.structure_path // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$STRUCTURE_PATH" ] && [ "$STRUCTURE_PATH" != "null" ]; then
    if [ ! -f "$STRUCTURE_PATH" ]; then
        error "Structure file not found: $STRUCTURE_PATH"
    fi
fi

DATABASE_PATH=$(jq -r '.database_path // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$DATABASE_PATH" ] && [ "$DATABASE_PATH" != "null" ]; then
    if [ ! -d "$DATABASE_PATH" ]; then
        error "Rosetta database not found: $DATABASE_PATH"
    fi
fi

XML_PATH=$(jq -r '.xml_path // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$XML_PATH" ] && [ "$XML_PATH" != "null" ]; then
    if [ ! -f "$XML_PATH" ]; then
        error "XML protocol file not found: $XML_PATH"
    fi
fi

# Validate nstruct is a positive integer
NSTRUCT=$(jq -r '.nstruct // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$NSTRUCT" ] && [ "$NSTRUCT" != "null" ]; then
    if ! [[ "$NSTRUCT" =~ ^[1-9][0-9]*$ ]]; then
        error "nstruct must be a positive integer, got '$NSTRUCT'"
    fi
fi

if [ "$ERRORS" -gt 0 ]; then
    echo "Config validation failed with $ERRORS error(s)." >&2
    exit 1
fi

echo "Config validation passed."
exit 0
