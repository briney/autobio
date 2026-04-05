#!/usr/bin/env python3
"""Standardize LigandMPNN sidechain packing outputs into autobio ScoringOutput format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def standardize(workspace: Path) -> None:
    """Parse packing raw outputs and write result_data.json."""
    raw_dir = workspace / "outputs" / "raw"
    std_dir = workspace / "outputs" / "standardized"
    config = json.loads((workspace / "config.json").read_text())

    mutations = config.get("mutations", [])

    # Read packing scores
    scores_file = raw_dir / "packing_scores.json"
    if not scores_file.exists():
        print("ERROR: packing_scores.json not found in raw output directory.", file=sys.stderr)
        raise RuntimeError(
            f"packing_scores.json not found in {raw_dir}. "
            f"Files present: {[f.name for f in raw_dir.iterdir()]}"
        )

    packing_scores = json.loads(scores_file.read_text())

    # Find all packed PDB files
    packed_pdbs = sorted(raw_dir.glob("packed_*.pdb"))
    if not packed_pdbs:
        print("ERROR: No packed PDB files found in raw output directory.", file=sys.stderr)
        raise RuntimeError(
            f"No packed_*.pdb files found in {raw_dir}. "
            f"Files present: {[f.name for f in raw_dir.iterdir()]}"
        )

    # Build score lookup by filename
    score_lookup: dict[str, dict] = {}
    for entry in packing_scores:
        score_lookup[entry["structure_file"]] = entry

    scores: list[dict] = []
    for pdb in packed_pdbs:
        entry = score_lookup.get(pdb.name, {})

        scores.append({
            "total_score": entry.get("total_score", 0.0),
            "per_residue_scores": entry.get("per_residue_scores"),
            "score_breakdown": None,
            "units": "LigandMPNN_SC_logprob",
            "structure_path": f"/workspace/outputs/raw/{pdb.name}",
            "ddg": None,
            "mutations": mutations,
        })

    result_data = {"scores": scores}
    (std_dir / "result_data.json").write_text(json.dumps(result_data, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    standardize(args.workspace)
