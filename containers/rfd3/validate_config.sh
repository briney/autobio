#!/usr/bin/env bash
# validate_config.sh — Phase 1: verify config.json for RFDiffusion3
#
# Catches config errors *before* the slow RFD3 model loads.  Validates
# known keys (with did-you-mean on typos), types, ranges, enums, and
# structural coherence.
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

# Levenshtein distance (pure bash, sufficient for short strings)
levenshtein() {
    local s1="$1" s2="$2"
    local len1=${#s1} len2=${#s2}
    local -a d
    local i j cost

    for ((i = 0; i <= len1; i++)); do d[$((i * (len2 + 1)))]=$i; done
    for ((j = 0; j <= len2; j++)); do d[$j]=$j; done

    for ((i = 1; i <= len1; i++)); do
        for ((j = 1; j <= len2; j++)); do
            if [ "${s1:$((i-1)):1}" = "${s2:$((j-1)):1}" ]; then
                cost=0
            else
                cost=1
            fi
            local del=$((d[$(( (i-1)*(len2+1) + j ))] + 1))
            local ins=$((d[$(( i*(len2+1) + j-1 ))] + 1))
            local sub=$((d[$(( (i-1)*(len2+1) + j-1 ))] + cost))
            local min=$del
            if [ $ins -lt $min ]; then min=$ins; fi
            if [ $sub -lt $min ]; then min=$sub; fi
            d[$((i * (len2 + 1) + j))]=$min
        done
    done
    echo "${d[$((len1 * (len2 + 1) + len2))]}"
}

suggest_key() {
    local unknown="$1"
    shift
    local best="" best_dist=999
    for known in "$@"; do
        local dist
        dist=$(levenshtein "$unknown" "$known")
        if [ "$dist" -lt "$best_dist" ]; then
            best_dist=$dist
            best=$known
        fi
    done
    if [ "$best_dist" -le 3 ]; then
        echo "$best"
    fi
}

# ---------------------------------------------------------------------------
# Valid key allowlists
# ---------------------------------------------------------------------------

# Per-spec keys (inside each design_specs entry)
VALID_SPEC_KEYS=(
    input contig length unindex ligand dialect
    select_fixed_atoms select_unfixed_sequence
    select_buried select_partially_buried select_exposed
    select_hbond_donor select_hbond_acceptor select_hotspots
    redesign_motif_sidechains is_non_loopy plddt_enhanced
    partial_t infer_ori_strategy ori_token
    symmetry cif_parser_args extra
    cleanup_guideposts cleanup_virtual_atoms
    output_full_json dump_prediction_metadata_json
    align_trajectory_structures
)

# Top-level config keys (CLI-level args + autobio protocol keys)
VALID_TOP_KEYS=(
    design_specs n_batches out_dir
    diffusion_batch_size num_timesteps step_scale kind
    low_memory_mode ckpt_path global_prefix
    skip_existing dump_trajectories prevalidate_inputs
    use_classifier_free_guidance cfg_scale cfg_features cfg_t_max
    center_option s_trans noise_scale p gamma_0 gamma_min
    s_jitter_origin allow_realignment
)

# ---------------------------------------------------------------------------
# Top-level structure checks
# ---------------------------------------------------------------------------

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
        suggestion=$(suggest_key "$key" "${VALID_TOP_KEYS[@]}")
        if [ -n "$suggestion" ]; then
            error "Unknown top-level config key '$key'. Did you mean '$suggestion'?"
        else
            warn "Unknown top-level config key '$key'. This will be ignored."
        fi
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

    # --- Known-key validation with did-you-mean ---
    SPEC_KEYS=$(jq -r ".design_specs[\"$spec_name\"] | keys[]" "$CONFIG_FILE" 2>/dev/null)
    for key in $SPEC_KEYS; do
        found=false
        for valid in "${VALID_SPEC_KEYS[@]}"; do
            if [ "$key" = "$valid" ]; then found=true; break; fi
        done
        if [ "$found" = false ]; then
            suggestion=$(suggest_key "$key" "${VALID_SPEC_KEYS[@]}")
            if [ -n "$suggestion" ]; then
                error "Spec '$spec_name': unknown key '$key'. Did you mean '$suggestion'?"
            else
                error "Spec '$spec_name': unknown key '$key'. Valid keys: ${VALID_SPEC_KEYS[*]}"
            fi
        fi
    done

    # --- Structural coherence: contig with chain refs requires input ---
    HAS_INPUT=$(jq -e ".design_specs[\"$spec_name\"].input" "$CONFIG_FILE" > /dev/null 2>&1 && echo true || echo false)
    CONTIG=$(jq -r ".design_specs[\"$spec_name\"].contig // empty" "$CONFIG_FILE" 2>/dev/null || true)
    if [ -n "$CONTIG" ] && echo "$CONTIG" | grep -qE '[A-Z][0-9]'; then
        if [ "$HAS_INPUT" = "false" ]; then
            error "Spec '$spec_name': contig '$CONTIG' references chain IDs but no 'input' structure is provided"
        fi
    fi

    # --- File existence: if input is set, verify file exists ---
    if [ "$HAS_INPUT" = "true" ]; then
        INPUT_PATH=$(jq -r ".design_specs[\"$spec_name\"].input" "$CONFIG_FILE")
        if [ ! -f "$INPUT_PATH" ]; then
            error "Spec '$spec_name': input structure not found: $INPUT_PATH"
        fi
    fi

    # --- Type validation: booleans ---
    for bool_field in redesign_motif_sidechains is_non_loopy plddt_enhanced; do
        VAL=$(jq -r ".design_specs[\"$spec_name\"].$bool_field // empty" "$CONFIG_FILE" 2>/dev/null || true)
        if [ -n "$VAL" ] && [ "$VAL" != "null" ] && [ "$VAL" != "true" ] && [ "$VAL" != "false" ]; then
            error "Spec '$spec_name': '$bool_field' must be a boolean, got '$VAL'"
        fi
    done

    # --- Range validation: partial_t ---
    PARTIAL_T=$(jq -r ".design_specs[\"$spec_name\"].partial_t // empty" "$CONFIG_FILE" 2>/dev/null || true)
    if [ -n "$PARTIAL_T" ] && [ "$PARTIAL_T" != "null" ]; then
        if ! echo "$PARTIAL_T" | grep -qE '^[0-9]*\.?[0-9]+$'; then
            error "Spec '$spec_name': partial_t must be a positive number, got '$PARTIAL_T'"
        elif [ "$(echo "$PARTIAL_T <= 0" | bc -l)" -eq 1 ]; then
            error "Spec '$spec_name': partial_t must be positive, got '$PARTIAL_T'"
        fi
    fi

    # --- Enum validation: infer_ori_strategy ---
    ORI_STRATEGY=$(jq -r ".design_specs[\"$spec_name\"].infer_ori_strategy // empty" "$CONFIG_FILE" 2>/dev/null || true)
    if [ -n "$ORI_STRATEGY" ] && [ "$ORI_STRATEGY" != "null" ]; then
        if [ "$ORI_STRATEGY" != "com" ] && [ "$ORI_STRATEGY" != "hotspots" ]; then
            error "Spec '$spec_name': infer_ori_strategy must be 'com' or 'hotspots', got '$ORI_STRATEGY'"
        fi
    fi

    # --- Enum validation: dialect ---
    DIALECT=$(jq -r ".design_specs[\"$spec_name\"].dialect // empty" "$CONFIG_FILE" 2>/dev/null || true)
    if [ -n "$DIALECT" ] && [ "$DIALECT" != "null" ]; then
        if [ "$DIALECT" != "1" ] && [ "$DIALECT" != "2" ]; then
            error "Spec '$spec_name': dialect must be 1 or 2, got '$DIALECT'"
        fi
    fi

    # --- Symmetry: if set, symmetry.id must be present ---
    HAS_SYMMETRY=$(jq -e ".design_specs[\"$spec_name\"].symmetry" "$CONFIG_FILE" > /dev/null 2>&1 && echo true || echo false)
    if [ "$HAS_SYMMETRY" = "true" ]; then
        SYM_ID=$(jq -r ".design_specs[\"$spec_name\"].symmetry.id // empty" "$CONFIG_FILE" 2>/dev/null || true)
        if [ -z "$SYM_ID" ] || [ "$SYM_ID" = "null" ]; then
            error "Spec '$spec_name': symmetry is set but 'symmetry.id' is missing (e.g., 'C3', 'D2')"
        elif ! echo "$SYM_ID" | grep -qE '^[CD][0-9]+$'; then
            error "Spec '$spec_name': symmetry.id must be a C or D group (e.g., 'C3', 'D2'), got '$SYM_ID'"
        fi
    fi
done

# ---------------------------------------------------------------------------
# CLI-level arg validation
# ---------------------------------------------------------------------------

# Range checks for numeric CLI args
for numfield in diffusion_batch_size num_timesteps; do
    VAL=$(jq -r ".$numfield // empty" "$CONFIG_FILE" 2>/dev/null || true)
    if [ -n "$VAL" ] && [ "$VAL" != "null" ]; then
        if ! [[ "$VAL" =~ ^[0-9]+$ ]] || [ "$VAL" -lt 1 ]; then
            error "$numfield must be a positive integer, got '$VAL'"
        fi
    fi
done

for floatfield in step_scale gamma_0; do
    VAL=$(jq -r ".$floatfield // empty" "$CONFIG_FILE" 2>/dev/null || true)
    if [ -n "$VAL" ] && [ "$VAL" != "null" ]; then
        if ! echo "$VAL" | grep -qE '^[0-9]*\.?[0-9]+$'; then
            error "$floatfield must be a positive number, got '$VAL'"
        elif [ "$(echo "$VAL <= 0" | bc -l)" -eq 1 ]; then
            error "$floatfield must be positive, got '$VAL'"
        fi
    fi
done

# Enum: kind (sampler type)
KIND=$(jq -r '.kind // empty' "$CONFIG_FILE" 2>/dev/null || true)
if [ -n "$KIND" ] && [ "$KIND" != "null" ]; then
    if [ "$KIND" != "default" ] && [ "$KIND" != "symmetry" ]; then
        error "kind must be 'default' or 'symmetry', got '$KIND'"
    fi
fi

# Boolean CLI-level args
for bool_cli in low_memory_mode skip_existing dump_trajectories prevalidate_inputs \
                use_classifier_free_guidance allow_realignment; do
    VAL=$(jq -r ".$bool_cli // empty" "$CONFIG_FILE" 2>/dev/null || true)
    if [ -n "$VAL" ] && [ "$VAL" != "null" ] && [ "$VAL" != "true" ] && [ "$VAL" != "false" ]; then
        error "$bool_cli must be a boolean, got '$VAL'"
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
