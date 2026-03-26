#!/usr/bin/env python3
"""Standardize Rosetta score_jd2 outputs into autobio ScoringOutput format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Shared parser baked into the base image
sys.path.insert(0, "/opt/rosetta")
from parse_scorefile import extract_scored_structure, parse_score_file


def standardize(workspace: Path) -> None:
    """Parse the Rosetta score file and write result_data.json."""
    raw_dir = workspace / "outputs" / "raw"
    std_dir = workspace / "outputs" / "standardized"

    # Find the score file
    score_files = sorted(raw_dir.glob("*.sc"))
    if not score_files:
        raise RuntimeError(
            f"No score files (.sc) found in {raw_dir}. "
            f"Files present: {[f.name for f in raw_dir.iterdir()]}"
        )

    # Parse all score files (usually just one for scoring)
    scores = []
    for sc_file in score_files:
        rows = parse_score_file(sc_file)
        for row in rows:
            scored = extract_scored_structure(row)
            # No structure output for pure scoring
            scored["structure_path"] = None
            scored["per_residue_scores"] = None
            scored["ddg"] = None
            scored["mutations"] = None
            scores.append(scored)

    if not scores:
        raise RuntimeError(
            f"No score entries found in {[f.name for f in score_files]}. "
            "Score files may be empty or malformed."
        )

    result_data = {"scores": scores}
    (std_dir / "result_data.json").write_text(json.dumps(result_data, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    standardize(args.workspace)
