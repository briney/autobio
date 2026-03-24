#!/usr/bin/env python3
"""Standardize Proteina-Complexa outputs into autobio schema format.

Walks the raw output directory for PDB files produced by ``complexa generate``,
copies them to ``outputs/standardized/``, and produces ``result_data.json``
conforming to the ``StructureDesignOutput`` schema.

Proteina-Complexa output structure::

    <out_dir>/inference/<config_name>_<task_name>[_<run_name>]/
        job_<job_id>_n_<length>_id_<idx>/
            job_<job_id>_n_<length>_id_<idx>.pdb
        rewards_<config_name>_<job_id>.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

# Matches: job_{job_id}_n_{length}_id_{idx}[_{metadata_tag}]
_FILENAME_RE = re.compile(
    r"^job_(?P<job>\d+)_n_(?P<length>\d+)_id_(?P<idx>\d+)(?:_(?P<tag>.+))?$"
)


def _load_rewards(raw_dir: Path) -> dict[str, dict]:
    """Load reward data from CSV files, keyed by PDB path."""
    rewards: dict[str, dict] = {}

    for csv_path in sorted(raw_dir.rglob("rewards_*.csv")):
        try:
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pdb_path = row.get("pdb_path", "")
                    if pdb_path:
                        rewards[pdb_path] = {
                            k: v
                            for k, v in row.items()
                            if k != "pdb_path"
                        }
        except (csv.Error, OSError):
            continue

    return rewards


def _find_designs(raw_dir: Path) -> list[dict]:
    """Find all design PDB outputs in the raw output directory.

    Proteina-Complexa outputs are nested under:
        inference/<run_dir>/job_<id>_n_<len>_id_<idx>/job_<id>_n_<len>_id_<idx>.pdb
    """
    designs: list[dict] = []

    for pdb_path in sorted(raw_dir.rglob("*.pdb")):
        # Skip intermediate binder-only PDBs
        if pdb_path.name.endswith("_binder.pdb"):
            continue

        stem = pdb_path.stem
        m = _FILENAME_RE.match(stem)

        if m:
            job_id = int(m.group("job"))
            design_index = int(m.group("idx"))
            binder_length = int(m.group("length"))
            metadata_tag = m.group("tag")
        else:
            # Fallback: treat entire stem as identifier
            job_id = 0
            design_index = 0
            binder_length = 0
            metadata_tag = None

        # Derive spec name from the run directory name.
        # Output dirs are: inference/<config_name>_<spec_name>/job_.../
        # The spec name is the run_name we passed as the last component.
        run_dir = pdb_path.parent.parent
        run_dir_name = run_dir.name
        # The run_name is the last underscore-separated segment
        # e.g., "search_binder_local_pipeline_pdl1_binder" → "pdl1_binder"
        # Since config names can have underscores, we look for the inference
        # parent to help parse.
        spec_name = run_dir_name

        designs.append({
            "spec_name": spec_name,
            "batch_index": job_id,
            "design_index": design_index,
            "raw_pdb_path": pdb_path,
            "binder_length": binder_length,
            "metadata_tag": metadata_tag,
        })

    return designs


def _build_dir_to_spec_map(workspace: Path) -> dict[str, str]:
    """Map output directory names back to original spec names.

    Output dirs follow the pattern:
        {pipeline_config}_{spec_name}_{spec_name}
    We read config.json to get the pipeline_config and spec names,
    then build the mapping.
    """
    config_path = workspace / "config.json"
    mapping: dict[str, str] = {}

    if config_path.exists():
        config = json.loads(config_path.read_text())
        pipeline_config = config.get("pipeline_config", "")
        for spec_name in config.get("design_specs", {}):
            # complexa generate outputs to: {pipeline}_{task_name}_{run_name}
            # We set task_name = run_name = spec_name
            dir_name = f"{pipeline_config}_{spec_name}_{spec_name}"
            mapping[dir_name] = spec_name

    return mapping


def standardize(workspace: Path) -> None:
    """Transform raw Proteina-Complexa outputs into standardized result_data.json."""
    raw_dir = workspace / "outputs" / "raw"
    std_dir = workspace / "outputs" / "standardized"
    std_dir.mkdir(parents=True, exist_ok=True)

    raw_designs = _find_designs(raw_dir)

    if not raw_designs:
        raise RuntimeError(
            f"No Proteina-Complexa design outputs (.pdb) found in {raw_dir}. "
            f"Check logs/tool.log for execution errors."
        )

    # Map output directory names back to original spec names
    dir_to_spec = _build_dir_to_spec_map(workspace)

    # Remap spec names from directory names to original spec names
    for d in raw_designs:
        d["spec_name"] = dir_to_spec.get(d["spec_name"], d["spec_name"])

    # Load reward data for metadata
    rewards = _load_rewards(raw_dir)

    designs: list[dict] = []
    spec_summary: dict[str, int] = {}

    for d in raw_designs:
        # Copy PDB to standardized directory
        dest_name = (
            f"{d['spec_name']}_b{d['batch_index']}_d{d['design_index']}.pdb"
        )
        dest_path = std_dir / dest_name
        shutil.copy2(d["raw_pdb_path"], dest_path)

        # Build diffusion metadata
        diffusion_metadata: dict = {
            "binder_length": d["binder_length"],
        }
        if d["metadata_tag"]:
            diffusion_metadata["sample_type"] = d["metadata_tag"]

        # Attach reward data if available
        raw_pdb_str = str(d["raw_pdb_path"])
        reward_data = rewards.get(raw_pdb_str)
        if reward_data:
            diffusion_metadata["rewards"] = reward_data

        designs.append({
            "spec_name": d["spec_name"],
            "batch_index": d["batch_index"],
            "design_index": d["design_index"],
            "structure_path": str(dest_path),
            "diffusion_metadata": diffusion_metadata,
        })

        spec_summary[d["spec_name"]] = spec_summary.get(d["spec_name"], 0) + 1

    result_data = {
        "designs": designs,
        "spec_summary": spec_summary,
    }
    (std_dir / "result_data.json").write_text(json.dumps(result_data, indent=2))

    print(
        f"[complexa-standardize] Standardized {len(designs)} designs "
        f"across {len(spec_summary)} spec(s)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standardize Proteina-Complexa outputs."
    )
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    standardize(args.workspace)
