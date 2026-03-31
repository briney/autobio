#!/usr/bin/env python3
"""Standardize OpenMM amber minimize outputs into autobio ScoringOutput format."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def standardize(workspace: Path) -> None:
    """Parse energy.json, copy PDB, and write result_data.json."""
    raw_dir = workspace / "outputs" / "raw"
    std_dir = workspace / "outputs" / "standardized"

    # Parse energy data
    energy_path = raw_dir / "energy.json"
    if not energy_path.exists():
        raise RuntimeError(
            f"No energy.json found in {raw_dir}. "
            f"Files present: {[f.name for f in raw_dir.iterdir()]}"
        )
    energy_data = json.loads(energy_path.read_text())

    # Find and copy minimized PDB
    pdb_files = sorted(raw_dir.glob("*.pdb"))
    if not pdb_files:
        raise RuntimeError(
            f"No PDB files found in {raw_dir}. "
            f"Files present: {[f.name for f in raw_dir.iterdir()]}"
        )

    structure_path = None
    for pdb in pdb_files:
        dest = std_dir / pdb.name
        shutil.copy2(pdb, dest)
        structure_path = f"/workspace/outputs/standardized/{pdb.name}"

    # Build score breakdown
    score_breakdown = dict(energy_data.get("energy_terms", {}))
    score_breakdown["initial_energy"] = energy_data["initial_energy_kj_mol"]
    score_breakdown["num_minimization_rounds"] = energy_data.get(
        "num_minimization_rounds", 1
    )
    score_breakdown["remaining_violations"] = len(
        energy_data.get("violations", [])
    )

    # Build result_data.json
    scores = [
        {
            "total_score": energy_data["final_energy_kj_mol"],
            "score_breakdown": score_breakdown,
            "units": "kJ/mol",
            "structure_path": structure_path,
            "per_residue_scores": None,
            "ddg": None,
            "mutations": None,
        }
    ]

    result_data = {"scores": scores}
    (std_dir / "result_data.json").write_text(json.dumps(result_data, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    standardize(args.workspace)
