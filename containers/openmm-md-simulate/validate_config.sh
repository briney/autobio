#!/usr/bin/env bash
# validate_config.sh — Phase 1: verify config.json for OpenMM MD simulation
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

# Validate water_model if present
WATER_MODEL=$(jq -r '.water_model // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$WATER_MODEL" ] && [ "$WATER_MODEL" != "null" ]; then
    case "$WATER_MODEL" in
        tip3p|tip4pew|spce) ;;
        *) error "Invalid water_model '$WATER_MODEL'. Allowed: tip3p, tip4pew, spce" ;;
    esac
fi

# Validate box_shape if present
BOX_SHAPE=$(jq -r '.box_shape // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$BOX_SHAPE" ] && [ "$BOX_SHAPE" != "null" ]; then
    case "$BOX_SHAPE" in
        cubic|dodecahedron|truncated_octahedron) ;;
        *) error "Invalid box_shape '$BOX_SHAPE'. Allowed: cubic, dodecahedron, truncated_octahedron" ;;
    esac
fi

# Validate ion_type if present
ION_TYPE=$(jq -r '.ion_type // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$ION_TYPE" ] && [ "$ION_TYPE" != "null" ]; then
    case "$ION_TYPE" in
        NaCl|KCl) ;;
        *) error "Invalid ion_type '$ION_TYPE'. Allowed: NaCl, KCl" ;;
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

# Validate trajectory_format if present
TRAJECTORY_FORMAT=$(jq -r '.trajectory_format // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$TRAJECTORY_FORMAT" ] && [ "$TRAJECTORY_FORMAT" != "null" ]; then
    case "$TRAJECTORY_FORMAT" in
        dcd|xtc|pdb) ;;
        *) error "Invalid trajectory_format '$TRAJECTORY_FORMAT'. Allowed: dcd, xtc, pdb" ;;
    esac
fi

# Validate positive numeric fields
for field in temperature pressure box_padding; do
    VALUE=$(jq -r ".$field // empty" "$CONFIG_FILE" 2>/dev/null || true)
    if [ -n "$VALUE" ] && [ "$VALUE" != "null" ]; then
        if ! echo "$VALUE" | grep -qE '^[0-9]*\.?[0-9]+$'; then
            error "$field must be a positive number, got '$VALUE'"
        elif [ "$(echo "$VALUE <= 0" | bc -l)" -eq 1 ]; then
            error "$field must be greater than 0, got '$VALUE'"
        fi
    fi
done

# Validate non-negative numeric fields
ION_CONCENTRATION=$(jq -r '.ion_concentration // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$ION_CONCENTRATION" ] && [ "$ION_CONCENTRATION" != "null" ]; then
    if ! echo "$ION_CONCENTRATION" | grep -qE '^[0-9]*\.?[0-9]+$'; then
        error "ion_concentration must be a non-negative number, got '$ION_CONCENTRATION'"
    fi
fi

# Validate total_time_ns if present (must be > 0)
TOTAL_TIME_NS=$(jq -r '.total_time_ns // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$TOTAL_TIME_NS" ] && [ "$TOTAL_TIME_NS" != "null" ]; then
    if ! echo "$TOTAL_TIME_NS" | grep -qE '^[0-9]*\.?[0-9]+$'; then
        error "total_time_ns must be a positive number, got '$TOTAL_TIME_NS'"
    elif [ "$(echo "$TOTAL_TIME_NS <= 0" | bc -l)" -eq 1 ]; then
        error "total_time_ns must be greater than 0, got '$TOTAL_TIME_NS'"
    fi
fi

# Validate reporting_interval_steps if present (must be > 0)
REPORTING_INTERVAL=$(jq -r '.reporting_interval_steps // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$REPORTING_INTERVAL" ] && [ "$REPORTING_INTERVAL" != "null" ]; then
    if ! echo "$REPORTING_INTERVAL" | grep -qE '^[1-9][0-9]*$'; then
        error "reporting_interval_steps must be a positive integer, got '$REPORTING_INTERVAL'"
    fi
fi

# Validate non-negative integer fields
for field in equilibration_nvt_steps equilibration_npt_steps; do
    VALUE=$(jq -r ".$field // empty" "$CONFIG_FILE" 2>/dev/null || true)
    if [ -n "$VALUE" ] && [ "$VALUE" != "null" ]; then
        if ! echo "$VALUE" | grep -qE '^[0-9]+$'; then
            error "$field must be a non-negative integer, got '$VALUE'"
        fi
    fi
done

if [ "$ERRORS" -gt 0 ]; then
    echo "Config validation failed with $ERRORS error(s)." >&2
    exit 1
fi

echo "Config validation passed."
exit 0
