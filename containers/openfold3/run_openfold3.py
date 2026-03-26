#!/usr/bin/env python3
"""Run OpenFold3 inference from a workspace config.json.

This script bridges the autobio container protocol (config.json) and the
OpenFold3 Python API (``InferenceExperimentRunner``).  The ``run_openfold``
CLI does not expose many parameters (MSA server URL, PAE settings, etc.)
directly, so we call the Python API for full control.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def run(workspace: Path) -> None:
    """Read config.json and invoke OpenFold3 inference."""
    from openfold3.entry_points.experiment_runner import InferenceExperimentRunner
    from openfold3.entry_points.validator import InferenceExperimentConfig
    from openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )

    config = json.loads((workspace / "config.json").read_text())

    # --- Required ---
    query_json_path = Path(config["query_json_path"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Construct experiment config ---
    # InferenceExperimentConfig accepts nested Pydantic models for each
    # setting group.  We build a kwargs dict from our flat config.
    expt_kwargs: dict = {}

    # Checkpoint path
    ckpt_path = config.get("checkpoint_path")
    if ckpt_path:
        expt_kwargs["inference_ckpt_path"] = Path(ckpt_path)

    # Model presets (PAE, low memory)
    presets = ["predict"]
    if config.get("pae_enabled", True):
        presets.append("pae_enabled")
    if config.get("low_memory", False):
        presets.append("low_mem")
    expt_kwargs["model_update"] = {"presets": presets}

    # MSA computation settings
    msa_settings: dict = {}
    if "msa_server_url" in config:
        msa_settings["server_url"] = config["msa_server_url"]
    if "msa_output_directory" in config:
        msa_settings["msa_output_directory"] = config["msa_output_directory"]
        msa_settings["cleanup_msa_dir"] = False
    if msa_settings:
        expt_kwargs["msa_computation_settings"] = msa_settings

    # Output writer settings
    output_settings: dict = {}
    output_format = config.get("output_format", "cif")
    if output_format != "cif":
        output_settings["structure_format"] = output_format
    if config.get("write_latent_outputs", False):
        output_settings["write_latent_outputs"] = True
    if output_settings:
        expt_kwargs["output_writer_settings"] = output_settings

    # Custom seeds
    if "seed" in config:
        expt_kwargs["experiment_settings"] = {"seeds": [int(config["seed"])]}

    # Multi-GPU
    if "num_devices" in config:
        expt_kwargs["pl_trainer_args"] = {"devices": int(config["num_devices"])}

    expt_config = InferenceExperimentConfig(**expt_kwargs)

    # --- Create runner ---
    num_diffusion_samples = int(config.get("num_diffusion_samples", 5))
    num_model_seeds = int(config.get("num_model_seeds", 1))
    use_msa_server = config.get("use_msa_server", True)
    use_templates = config.get("use_templates", True)

    runner = InferenceExperimentRunner(
        expt_config,
        num_diffusion_samples=num_diffusion_samples,
        num_model_seeds=num_model_seeds,
        use_msa_server=use_msa_server,
        use_templates=use_templates,
        output_dir=output_dir,
    )

    # --- Load queries ---
    query_set = InferenceQuerySet.from_json(query_json_path)

    print(f"[openfold3] Query JSON: {query_json_path}")
    print(f"[openfold3] Output: {output_dir}")
    print(f"[openfold3] Diffusion samples: {num_diffusion_samples}")
    print(f"[openfold3] Model seeds: {num_model_seeds}")
    print(f"[openfold3] MSA server: {use_msa_server}")
    print(f"[openfold3] Templates: {use_templates}")
    print(f"[openfold3] Queries: {list(query_set.queries.keys())}")

    # --- Run inference ---
    runner.setup()
    try:
        runner.run(query_set)
    finally:
        runner.cleanup()

    # Count output files to report
    cif_files = list(output_dir.rglob("*_model.cif"))
    pdb_files = list(output_dir.rglob("*_model.pdb"))
    total = len(cif_files) + len(pdb_files)
    print(f"[openfold3] Completed. Generated {total} candidate(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenFold3 inference.")
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    run(args.workspace)
