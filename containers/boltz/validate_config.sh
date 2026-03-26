#!/usr/bin/env bash
# validate_config.sh — Phase 1: verify config.json for Boltz
#
# Catches config errors before the slow model loads.
set -euo pipefail

CONFIG_FILE="$1"
ERRORS=0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

error() {
    echo "ERROR: $1" >&2
    ERRORS=$((ERRORS + 1))
}

# ---------------------------------------------------------------------------
# Required fields
# ---------------------------------------------------------------------------

for field in input_path model output_dir; do
    if ! jq -e ".$field" "$CONFIG_FILE" > /dev/null 2>&1; then
        error "config.json missing required field '$field'"
    fi
done

# ---------------------------------------------------------------------------
# Enum: model
# ---------------------------------------------------------------------------

MODEL=$(jq -r '.model // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$MODEL" ]; then
    if [ "$MODEL" != "boltz1" ] && [ "$MODEL" != "boltz2" ]; then
        error "model must be 'boltz1' or 'boltz2', got '$MODEL'"
    fi
fi

# ---------------------------------------------------------------------------
# File existence: input_path
# ---------------------------------------------------------------------------

INPUT_PATH=$(jq -r '.input_path // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$INPUT_PATH" ] && [ "$INPUT_PATH" != "null" ]; then
    if [ ! -f "$INPUT_PATH" ]; then
        error "input file not found: $INPUT_PATH"
    fi
fi

# ---------------------------------------------------------------------------
# Cache directory
# ---------------------------------------------------------------------------

CACHE_DIR=$(jq -r '.cache_dir // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$CACHE_DIR" ] && [ "$CACHE_DIR" != "null" ]; then
    if [ ! -d "$CACHE_DIR" ]; then
        error "cache directory not found: $CACHE_DIR"
    fi
fi

# ---------------------------------------------------------------------------
# Optional field validation
# ---------------------------------------------------------------------------

# Enum: output_format
OUTPUT_FORMAT=$(jq -r '.output_format // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$OUTPUT_FORMAT" ] && [ "$OUTPUT_FORMAT" != "null" ]; then
    if [ "$OUTPUT_FORMAT" != "pdb" ] && [ "$OUTPUT_FORMAT" != "mmcif" ]; then
        error "output_format must be 'pdb' or 'mmcif', got '$OUTPUT_FORMAT'"
    fi
fi

# Positive integers
for numfield in diffusion_samples sampling_steps recycling_steps \
                max_parallel_samples devices max_msa_seqs \
                sampling_steps_affinity diffusion_samples_affinity \
                num_subsampled_msa; do
    VAL=$(jq -r ".$numfield // empty" "$CONFIG_FILE" 2>/dev/null || true)
    if [ -n "$VAL" ] && [ "$VAL" != "null" ]; then
        if ! [[ "$VAL" =~ ^[0-9]+$ ]] || [ "$VAL" -lt 1 ]; then
            error "$numfield must be a positive integer, got '$VAL'"
        fi
    fi
done

# Positive floats
for floatfield in step_scale; do
    VAL=$(jq -r ".$floatfield // empty" "$CONFIG_FILE" 2>/dev/null || true)
    if [ -n "$VAL" ] && [ "$VAL" != "null" ]; then
        if ! echo "$VAL" | grep -qE '^[0-9]*\.?[0-9]+$'; then
            error "$floatfield must be a positive number, got '$VAL'"
        fi
    fi
done

# Booleans
for bool_field in use_msa_server use_potentials override \
                  write_full_pae write_full_pde write_embeddings \
                  subsample_msa; do
    VAL=$(jq -r ".$bool_field // empty" "$CONFIG_FILE" 2>/dev/null || true)
    if [ -n "$VAL" ] && [ "$VAL" != "null" ] && [ "$VAL" != "true" ] && [ "$VAL" != "false" ]; then
        error "$bool_field must be a boolean, got '$VAL'"
    fi
done

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

if [ "$ERRORS" -gt 0 ]; then
    echo "Config validation failed with $ERRORS error(s)." >&2
    exit 1
fi

echo "Config validation passed."
exit 0
