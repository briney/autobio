#!/usr/bin/env bash
# validate_config.sh — Phase 1: verify config.json for Chai-1
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

for field in fasta_path output_dir; do
    if ! jq -e ".$field" "$CONFIG_FILE" > /dev/null 2>&1; then
        error "config.json missing required field '$field'"
    fi
done

# ---------------------------------------------------------------------------
# File existence: fasta_path
# ---------------------------------------------------------------------------

FASTA_PATH=$(jq -r '.fasta_path // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$FASTA_PATH" ] && [ "$FASTA_PATH" != "null" ]; then
    if [ ! -f "$FASTA_PATH" ]; then
        error "FASTA file not found: $FASTA_PATH"
    fi
fi

# ---------------------------------------------------------------------------
# Downloads directory
# ---------------------------------------------------------------------------

DOWNLOADS_DIR=$(jq -r '.downloads_dir // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$DOWNLOADS_DIR" ] && [ "$DOWNLOADS_DIR" != "null" ]; then
    if [ ! -d "$DOWNLOADS_DIR" ]; then
        error "downloads directory not found: $DOWNLOADS_DIR"
    fi
fi

# ---------------------------------------------------------------------------
# Optional file/directory existence
# ---------------------------------------------------------------------------

CONSTRAINT_PATH=$(jq -r '.constraint_path // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$CONSTRAINT_PATH" ] && [ "$CONSTRAINT_PATH" != "null" ]; then
    if [ ! -f "$CONSTRAINT_PATH" ]; then
        error "constraint file not found: $CONSTRAINT_PATH"
    fi
fi

MSA_DIR=$(jq -r '.msa_directory // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$MSA_DIR" ] && [ "$MSA_DIR" != "null" ]; then
    if [ ! -d "$MSA_DIR" ]; then
        error "MSA directory not found: $MSA_DIR"
    fi
fi

# ---------------------------------------------------------------------------
# Positive integers
# ---------------------------------------------------------------------------

for numfield in num_diffn_samples num_trunk_recycles num_diffn_timesteps \
                num_trunk_samples seed; do
    VAL=$(jq -r ".$numfield // empty" "$CONFIG_FILE" 2>/dev/null || true)
    if [ -n "$VAL" ] && [ "$VAL" != "null" ]; then
        if ! [[ "$VAL" =~ ^[0-9]+$ ]] || [ "$VAL" -lt 1 ]; then
            error "$numfield must be a positive integer, got '$VAL'"
        fi
    fi
done

# ---------------------------------------------------------------------------
# Booleans
# ---------------------------------------------------------------------------

for bool_field in use_msa_server use_templates_server use_esm_embeddings low_memory; do
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
