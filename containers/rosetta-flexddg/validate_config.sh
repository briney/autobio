#!/usr/bin/env bash
# validate_config.sh — Phase 1: verify config.json for flex-ddG
set -euo pipefail

CONFIG_FILE="$1"
ERRORS=0

error() {
    echo "ERROR: $1" >&2
    ERRORS=$((ERRORS + 1))
}

# Required fields
for field in structure_path database_path mutations chains_to_move; do
    if ! jq -e ".$field" "$CONFIG_FILE" > /dev/null 2>&1; then
        error "config.json missing required field '$field'"
    fi
done

# Validate mutations is a non-empty array
MUTATIONS_COUNT=$(jq -r '.mutations | length' "$CONFIG_FILE" 2>/dev/null || echo "0")
if [ "$MUTATIONS_COUNT" = "0" ]; then
    error "mutations must be a non-empty list"
fi

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


if [ "$ERRORS" -gt 0 ]; then
    echo "Config validation failed with $ERRORS error(s)." >&2
    exit 1
fi

echo "Config validation passed."
exit 0
