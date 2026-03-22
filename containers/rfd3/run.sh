#!/usr/bin/env bash
# run.sh — Phase 2: execute RFDiffusion3
set -euo pipefail

WORKSPACE="$1"
CONFIG="$WORKSPACE/config.json"

# --- Extract design_specs to a separate JSON file for rfd3 CLI ----------------
jq '.design_specs' "$CONFIG" > "$WORKSPACE/inputs/rfd3_inputs.json"

# --- Required fields ----------------------------------------------------------
OUT_DIR=$(jq -r '.out_dir' "$CONFIG")
N_BATCHES=$(jq -r '.n_batches // 1' "$CONFIG")

# --- Build base command -------------------------------------------------------
CMD=(
    rfd3 design
    "out_dir=$OUT_DIR"
    "inputs=$WORKSPACE/inputs/rfd3_inputs.json"
    "n_batches=$N_BATCHES"
)

# --- Helper: append CLI arg if present in config ------------------------------
append_if_set() {
    local config_key="$1"
    local cli_arg="$2"
    local val
    val=$(jq -r ".$config_key // empty" "$CONFIG" 2>/dev/null || true)
    if [ -n "$val" ] && [ "$val" != "null" ]; then
        CMD+=("$cli_arg=$val")
    fi
}

# --- Top-level CLI args -------------------------------------------------------
append_if_set diffusion_batch_size     diffusion_batch_size
append_if_set low_memory_mode          low_memory_mode
append_if_set ckpt_path                ckpt_path
append_if_set global_prefix            global_prefix
append_if_set skip_existing            skip_existing
append_if_set dump_trajectories        dump_trajectories

# --- Inference sampler args (use inference_sampler.X prefix) ------------------
append_if_set num_timesteps            inference_sampler.num_timesteps
append_if_set step_scale               inference_sampler.step_scale
append_if_set gamma_0                  inference_sampler.gamma_0
append_if_set gamma_min                inference_sampler.gamma_min
append_if_set kind                     inference_sampler.kind
append_if_set noise_scale              inference_sampler.noise_scale
append_if_set p                        inference_sampler.p
append_if_set s_trans                  inference_sampler.s_trans
append_if_set s_jitter_origin          inference_sampler.s_jitter_origin
append_if_set center_option            inference_sampler.center_option
append_if_set allow_realignment        inference_sampler.allow_realignment

# --- Classifier-free guidance -------------------------------------------------
append_if_set use_classifier_free_guidance  inference_sampler.use_classifier_free_guidance
append_if_set cfg_scale                     inference_sampler.cfg_scale
append_if_set cfg_t_max                     inference_sampler.cfg_t_max

echo "[rfd3] Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "$WORKSPACE/logs/tool.log"
