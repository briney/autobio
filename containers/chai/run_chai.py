#!/usr/bin/env python3
"""Run Chai-1 inference from a workspace config.json.

This script bridges the autobio container protocol (config.json) and the
Chai-1 Python API (``chai_lab.chai1.run_inference``).  The ``chai-lab fold``
CLI does not expose many parameters (seed, num_diffn_samples, constraint_path,
etc.), so we call the Python API directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def run(workspace: Path) -> None:
    """Read config.json and invoke Chai-1 inference."""
    from chai_lab.chai1 import run_inference

    config = json.loads((workspace / "config.json").read_text())

    # --- Required ---
    fasta_file = Path(config["fasta_path"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Optional with defaults ---
    kwargs: dict = {
        "fasta_file": fasta_file,
        "output_dir": output_dir,
        "device": "cuda:0",
    }

    # Inference parameters
    if "num_diffn_samples" in config:
        kwargs["num_diffn_samples"] = int(config["num_diffn_samples"])
    if "num_trunk_recycles" in config:
        kwargs["num_trunk_recycles"] = int(config["num_trunk_recycles"])
    if "num_diffn_timesteps" in config:
        kwargs["num_diffn_timesteps"] = int(config["num_diffn_timesteps"])
    if "num_trunk_samples" in config:
        kwargs["num_trunk_samples"] = int(config["num_trunk_samples"])
    if "seed" in config:
        kwargs["seed"] = int(config["seed"])

    # Boolean options
    if "use_esm_embeddings" in config:
        kwargs["use_esm_embeddings"] = config["use_esm_embeddings"]
    if "low_memory" in config:
        kwargs["low_memory"] = config["low_memory"]

    # MSA options
    if "use_msa_server" in config:
        kwargs["use_msa_server"] = config["use_msa_server"]
    if "msa_server_url" in config and config["msa_server_url"]:
        kwargs["msa_server_url"] = config["msa_server_url"]
    if "msa_directory" in config and config["msa_directory"]:
        kwargs["msa_directory"] = Path(config["msa_directory"])

    # Template options
    if "use_templates_server" in config:
        kwargs["use_templates_server"] = config["use_templates_server"]

    # Constraints (restraints / covalent bonds)
    if config.get("constraint_path"):
        kwargs["constraint_path"] = Path(config["constraint_path"])

    # Recycle MSA subsampling
    if "recycle_msa_subsample" in config:
        kwargs["recycle_msa_subsample"] = int(config["recycle_msa_subsample"])

    print(f"[chai] FASTA: {fasta_file}")
    print(f"[chai] Output: {output_dir}")
    print(f"[chai] Parameters: { {k: v for k, v in kwargs.items() if k not in ('fasta_file', 'output_dir')} }")

    run_inference(**kwargs)

    # Count output files to report
    cif_files = list(output_dir.glob("pred.model_idx_*.cif"))
    print(f"[chai] Completed. Generated {len(cif_files)} candidate(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Chai-1 inference.")
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    run(args.workspace)
