#!/usr/bin/env bash
# validate_config.sh — Phase 1: verify config.json for ESM embedding
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

for field in model_name input_fasta output_dir hf_cache; do
    if ! jq -e ".$field" "$CONFIG_FILE" > /dev/null 2>&1; then
        error "config.json missing required field '$field'"
    fi
done

# ---------------------------------------------------------------------------
# Validate model_name is a known ESM model
# ---------------------------------------------------------------------------

MODEL_NAME=$(jq -r '.model_name // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$MODEL_NAME" ] && [ "$MODEL_NAME" != "null" ]; then
    case "$MODEL_NAME" in
        facebook/esm1b_t33_650M_UR50S|\
        facebook/esm2_t6_8M_UR50D|\
        facebook/esm2_t12_35M_UR50D|\
        facebook/esm2_t30_150M_UR50D|\
        facebook/esm2_t33_650M_UR50D|\
        facebook/esm2_t36_3B_UR50D|\
        facebook/esm2_t48_15B_UR50D)
            ;;
        *)
            error "Unknown model_name: $MODEL_NAME"
            ;;
    esac

    # Verify the model exists in the HuggingFace cache
    HF_CACHE=$(jq -r '.hf_cache // "/app/esm/hf_cache"' "$CONFIG_FILE")
    # HuggingFace cache stores models under hub/models--<org>--<name>
    CACHE_DIR_NAME=$(echo "$MODEL_NAME" | sed 's|/|--|g')
    EXPECTED_DIR="$HF_CACHE/hub/models--$CACHE_DIR_NAME"
    if [ ! -d "$EXPECTED_DIR" ]; then
        error "Model not found in HF cache: $MODEL_NAME (expected $EXPECTED_DIR)"
    fi
fi

# ---------------------------------------------------------------------------
# File existence: input_fasta
# ---------------------------------------------------------------------------

INPUT_FASTA=$(jq -r '.input_fasta // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$INPUT_FASTA" ] && [ "$INPUT_FASTA" != "null" ]; then
    if [ ! -f "$INPUT_FASTA" ]; then
        error "FASTA file not found: $INPUT_FASTA"
    fi
fi

# ---------------------------------------------------------------------------
# Validate pooling
# ---------------------------------------------------------------------------

POOLING=$(jq -r '.pooling // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$POOLING" ] && [ "$POOLING" != "null" ]; then
    case "$POOLING" in
        mean|cls|per_residue) ;;
        *) error "pooling must be 'mean', 'cls', or 'per_residue', got '$POOLING'" ;;
    esac
fi

# ---------------------------------------------------------------------------
# Validate layer (positive integer or null)
# ---------------------------------------------------------------------------

LAYER=$(jq -r '.layer // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$LAYER" ] && [ "$LAYER" != "null" ]; then
    if ! [[ "$LAYER" =~ ^[0-9]+$ ]]; then
        error "layer must be a non-negative integer, got '$LAYER'"
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
