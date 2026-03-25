#!/usr/bin/env bash
# validate_config.sh — Phase 1: verify config.json for Proteina-Complexa
#
# Catches config errors *before* the slow model loads.  Validates known keys,
# types, ranges, and structural coherence.
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

warn() {
    echo "WARNING: $1" >&2
}

# ---------------------------------------------------------------------------
# Valid key allowlists
# ---------------------------------------------------------------------------

# Per-spec keys (inside each design_specs entry)
VALID_SPEC_KEYS=(
    input target_input hotspot_residues binder_length
    binder_center pdb_id ligand_chain motif_residues
    ligand ligand_only smiles use_bonds_from_file contig_atoms
)

# Top-level config keys
VALID_TOP_KEYS=(
    variant pipeline_config ckpt_name ae_ckpt_name weights_dir
    design_specs n_batches out_dir
    batch_size n_samples_per_length binder_length_samples
    search_algorithm seed gen_njobs eval_njobs mode
)

# Valid variants
VALID_VARIANTS=(protein_binder ligand_binder ame)

# Valid search algorithms
VALID_ALGORITHMS=(single-pass best-of-n beam-search fk-steering mcts)

# ---------------------------------------------------------------------------
# Required field checks
# ---------------------------------------------------------------------------

# variant must exist and be valid
VARIANT=$(jq -r '.variant // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -z "$VARIANT" ] || [ "$VARIANT" = "null" ]; then
    error "config.json missing required field 'variant'"
else
    valid_variant=false
    for v in "${VALID_VARIANTS[@]}"; do
        if [ "$VARIANT" = "$v" ]; then valid_variant=true; break; fi
    done
    if [ "$valid_variant" = false ]; then
        error "variant must be one of: ${VALID_VARIANTS[*]}; got '$VARIANT'"
    fi
fi

# mode must be 'generate' or 'design' (defaults to 'generate' if absent)
MODE=$(jq -r '.mode // "generate"' "$CONFIG_FILE" 2>/dev/null || true)
if [ "$MODE" != "generate" ] && [ "$MODE" != "design" ]; then
    error "mode must be 'generate' or 'design'; got '$MODE'"
fi

# In design mode, verify community model weights are available
if [ "$MODE" = "design" ]; then
    if [ -z "$AF2_DIR" ] || [ ! -d "$AF2_DIR" ]; then
        error "mode=design requires AF2 weights. AF2_DIR='${AF2_DIR:-<unset>}' is empty or does not exist. Use the full image (not generate-only)."
    fi
    if [ -z "$RF3_CKPT_PATH" ] || [ ! -f "$RF3_CKPT_PATH" ]; then
        error "mode=design requires RF3 checkpoint. RF3_CKPT_PATH='${RF3_CKPT_PATH:-<unset>}' is empty or does not exist. Use the full image (not generate-only)."
    fi
fi

# design_specs must exist and be non-empty
if ! jq -e '.design_specs' "$CONFIG_FILE" > /dev/null 2>&1; then
    error "config.json missing required field 'design_specs'"
fi

SPEC_COUNT=$(jq '.design_specs | length' "$CONFIG_FILE" 2>/dev/null || echo 0)
if [ "$SPEC_COUNT" -eq 0 ]; then
    error "design_specs must contain at least one specification"
fi

# n_batches must be a positive integer
N_BATCHES=$(jq -r '.n_batches // 1' "$CONFIG_FILE")
if ! [[ "$N_BATCHES" =~ ^[0-9]+$ ]] || [ "$N_BATCHES" -lt 1 ]; then
    error "n_batches must be a positive integer, got '$N_BATCHES'"
fi

# out_dir must be set
if ! jq -e '.out_dir' "$CONFIG_FILE" > /dev/null 2>&1; then
    error "config.json missing required field 'out_dir'"
fi

# weights_dir must exist
WEIGHTS_DIR=$(jq -r '.weights_dir // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$WEIGHTS_DIR" ] && [ "$WEIGHTS_DIR" != "null" ] && [ ! -d "$WEIGHTS_DIR" ]; then
    error "weights_dir does not exist: $WEIGHTS_DIR"
fi

# ---------------------------------------------------------------------------
# Top-level key validation
# ---------------------------------------------------------------------------

TOP_KEYS=$(jq -r 'keys[]' "$CONFIG_FILE" 2>/dev/null)
for key in $TOP_KEYS; do
    found=false
    for valid in "${VALID_TOP_KEYS[@]}"; do
        if [ "$key" = "$valid" ]; then found=true; break; fi
    done
    if [ "$found" = false ]; then
        warn "Unknown top-level config key '$key'. This will be passed through as a generation parameter."
    fi
done

# ---------------------------------------------------------------------------
# Per-spec validation
# ---------------------------------------------------------------------------

SPEC_NAMES=$(jq -r '.design_specs | keys[]' "$CONFIG_FILE" 2>/dev/null)
for spec_name in $SPEC_NAMES; do
    # Check that spec value is an object
    SPEC_TYPE=$(jq -r ".design_specs[\"$spec_name\"] | type" "$CONFIG_FILE")
    if [ "$SPEC_TYPE" != "object" ]; then
        error "design_specs['$spec_name'] must be an object, got '$SPEC_TYPE'"
        continue
    fi

    # --- File existence: if input is set, verify file exists ---
    HAS_INPUT=$(jq -e ".design_specs[\"$spec_name\"].input" "$CONFIG_FILE" > /dev/null 2>&1 && echo true || echo false)
    if [ "$HAS_INPUT" = "true" ]; then
        INPUT_PATH=$(jq -r ".design_specs[\"$spec_name\"].input" "$CONFIG_FILE")
        if [ ! -f "$INPUT_PATH" ]; then
            error "Spec '$spec_name': input structure not found: $INPUT_PATH"
        fi
    fi

    # --- target_input format check ---
    TARGET_INPUT=$(jq -r ".design_specs[\"$spec_name\"].target_input // empty" "$CONFIG_FILE" 2>/dev/null || true)
    if [ -n "$TARGET_INPUT" ] && [ "$TARGET_INPUT" != "null" ]; then
        if ! echo "$TARGET_INPUT" | grep -qE '^[A-Z][0-9]+-[0-9]+$'; then
            warn "Spec '$spec_name': target_input '$TARGET_INPUT' does not match expected format '<Chain><Start>-<End>' (e.g., 'A1-115')"
        fi
    fi

    # --- binder_length must be a two-element array of positive integers ---
    HAS_LENGTH=$(jq -e ".design_specs[\"$spec_name\"].binder_length" "$CONFIG_FILE" > /dev/null 2>&1 && echo true || echo false)
    if [ "$HAS_LENGTH" = "true" ]; then
        LENGTH_COUNT=$(jq ".design_specs[\"$spec_name\"].binder_length | length" "$CONFIG_FILE" 2>/dev/null || echo 0)
        if [ "$LENGTH_COUNT" -ne 2 ]; then
            error "Spec '$spec_name': binder_length must be a [min, max] array with 2 elements, got $LENGTH_COUNT"
        else
            MIN_LEN=$(jq ".design_specs[\"$spec_name\"].binder_length[0]" "$CONFIG_FILE")
            MAX_LEN=$(jq ".design_specs[\"$spec_name\"].binder_length[1]" "$CONFIG_FILE")
            if [ "$MIN_LEN" -lt 1 ] 2>/dev/null; then
                error "Spec '$spec_name': binder_length min must be positive, got $MIN_LEN"
            fi
            if [ "$MAX_LEN" -lt "$MIN_LEN" ] 2>/dev/null; then
                error "Spec '$spec_name': binder_length max ($MAX_LEN) must be >= min ($MIN_LEN)"
            fi
        fi
    fi

    # --- hotspot_residues must be an array if present ---
    HAS_HOTSPOTS=$(jq -e ".design_specs[\"$spec_name\"].hotspot_residues" "$CONFIG_FILE" > /dev/null 2>&1 && echo true || echo false)
    if [ "$HAS_HOTSPOTS" = "true" ]; then
        HS_TYPE=$(jq -r ".design_specs[\"$spec_name\"].hotspot_residues | type" "$CONFIG_FILE")
        if [ "$HS_TYPE" != "array" ]; then
            error "Spec '$spec_name': hotspot_residues must be an array, got '$HS_TYPE'"
        fi
    fi
done

# ---------------------------------------------------------------------------
# CLI-level arg validation
# ---------------------------------------------------------------------------

# search_algorithm enum check
ALGO=$(jq -r '.search_algorithm // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$ALGO" ] && [ "$ALGO" != "null" ]; then
    valid_algo=false
    for a in "${VALID_ALGORITHMS[@]}"; do
        if [ "$ALGO" = "$a" ]; then valid_algo=true; break; fi
    done
    if [ "$valid_algo" = false ]; then
        error "search_algorithm must be one of: ${VALID_ALGORITHMS[*]}; got '$ALGO'"
    fi
fi

# batch_size must be a positive integer
BATCH_SIZE=$(jq -r '.batch_size // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$BATCH_SIZE" ] && [ "$BATCH_SIZE" != "null" ]; then
    if ! [[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] || [ "$BATCH_SIZE" -lt 1 ]; then
        error "batch_size must be a positive integer, got '$BATCH_SIZE'"
    fi
fi

# seed must be a non-negative integer
SEED=$(jq -r '.seed // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$SEED" ] && [ "$SEED" != "null" ]; then
    if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
        error "seed must be a non-negative integer, got '$SEED'"
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
