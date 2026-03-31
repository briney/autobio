#!/usr/bin/env bash
# validate_config.sh — Phase 1: verify config.json for OpenMM amber minimize
set -euo pipefail

CONFIG_FILE="$1"
ERRORS=0

error() {
    echo "ERROR: $1" >&2
    ERRORS=$((ERRORS + 1))
}

# Required fields
for field in structure_path; do
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

# Validate force_field if present
FORCE_FIELD=$(jq -r '.force_field // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$FORCE_FIELD" ] && [ "$FORCE_FIELD" != "null" ]; then
    case "$FORCE_FIELD" in
        amber14-all.xml|amber99sb.xml|charmm36.xml) ;;
        *) error "Invalid force_field '$FORCE_FIELD'. Allowed: amber14-all.xml, amber99sb.xml, charmm36.xml" ;;
    esac
fi

# Validate restraint_set if present
RESTRAINT_SET=$(jq -r '.restraint_set // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$RESTRAINT_SET" ] && [ "$RESTRAINT_SET" != "null" ]; then
    case "$RESTRAINT_SET" in
        none|ca|heavy_atoms) ;;
        *) error "Invalid restraint_set '$RESTRAINT_SET'. Allowed: none, ca, heavy_atoms" ;;
    esac
fi

# Validate numeric fields if present
TOLERANCE=$(jq -r '.tolerance // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$TOLERANCE" ] && [ "$TOLERANCE" != "null" ]; then
    if ! echo "$TOLERANCE" | grep -qE '^[0-9]*\.?[0-9]+$'; then
        error "tolerance must be a positive number, got '$TOLERANCE'"
    fi
fi

MAX_ITERATIONS=$(jq -r '.max_iterations // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$MAX_ITERATIONS" ] && [ "$MAX_ITERATIONS" != "null" ]; then
    if ! echo "$MAX_ITERATIONS" | grep -qE '^[0-9]+$'; then
        error "max_iterations must be a non-negative integer, got '$MAX_ITERATIONS'"
    fi
fi

if [ "$ERRORS" -gt 0 ]; then
    echo "Config validation failed with $ERRORS error(s)." >&2
    exit 1
fi

echo "Config validation passed."
exit 0
