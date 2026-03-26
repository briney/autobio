#!/usr/bin/env python3
"""Standardize Rosetta relax/minimize outputs into autobio ScoringOutput format.

Relax and minimize produce both score files and refined PDB structures.
Each scored structure in the output includes a reference to its PDB.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/opt/rosetta")
from parse_scorefile import extract_scored_structure, parse_score_file


def standardize(workspace: Path) -> None:
    """Parse score file, copy PDBs, and write result_data.json."""
    raw_dir = workspace / "outputs" / "raw"
    std_dir = workspace / "outputs" / "standardized"

    # Parse score file
    score_files = sorted(raw_dir.glob("*.sc"))
    if not score_files:
        raise RuntimeError(
            f"No score files (.sc) found in {raw_dir}. "
            f"Files present: {[f.name for f in raw_dir.iterdir()]}"
        )

    rows = []
    for sc_file in score_files:
        rows.extend(parse_score_file(sc_file))

    if not rows:
        raise RuntimeError("No score entries found in score files.")

    # Find PDB outputs and build a description->path mapping
    pdb_files = sorted(raw_dir.glob("*.pdb"))
    pdb_by_description: dict[str, Path] = {}
    for pdb in pdb_files:
        # Rosetta names output PDBs based on the description column
        pdb_by_description[pdb.stem] = pdb

    scores = []
    for row in rows:
        scored = extract_scored_structure(row)
        description = str(row.get("description", ""))

        # Find and copy the corresponding PDB
        structure_path = None
        if description in pdb_by_description:
            src_pdb = pdb_by_description[description]
            dest_pdb = std_dir / src_pdb.name
            shutil.copy2(src_pdb, dest_pdb)
            structure_path = f"/workspace/outputs/standardized/{src_pdb.name}"

        scored["structure_path"] = structure_path
        scored["per_residue_scores"] = None
        scored["ddg"] = None
        scored["mutations"] = None
        scores.append(scored)

    result_data = {"scores": scores}
    (std_dir / "result_data.json").write_text(json.dumps(result_data, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    standardize(args.workspace)
