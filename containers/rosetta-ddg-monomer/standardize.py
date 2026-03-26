#!/usr/bin/env python3
"""Standardize Rosetta ddg_monomer outputs into autobio ScoringOutput format.

The ddg_monomer application produces a ``ddg_predictions.out`` file with
wild-type and mutant scores. The DDG is the difference: mut_score - wt_score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_ddg_predictions(path: Path) -> list[dict]:
    """Parse ddg_predictions.out produced by ddg_monomer.

    Format (whitespace-delimited):
        ddG: <description> <ddg_value> <wt_score> <mut_score> ...

    Returns:
        List of dicts with ddg, wt_score, mut_score, and description.
    """
    results = []
    for line in path.read_text().strip().splitlines():
        line = line.strip()
        if not line or not line.startswith("ddG:"):
            continue
        parts = line.split()
        # ddG: <desc> <ddg> <wt_total> <mut_total> ...
        if len(parts) < 5:
            continue
        results.append(
            {
                "description": parts[1],
                "ddg": float(parts[2]),
                "wt_score": float(parts[3]),
                "mut_score": float(parts[4]),
            }
        )
    return results


def standardize(workspace: Path) -> None:
    """Parse DDG predictions and write result_data.json."""
    raw_dir = workspace / "outputs" / "raw"
    std_dir = workspace / "outputs" / "standardized"
    config = json.loads((workspace / "config.json").read_text())

    # Find ddg_predictions.out
    ddg_file = raw_dir / "ddg_predictions.out"
    if not ddg_file.exists():
        # Fall back to any .out file
        out_files = sorted(raw_dir.glob("*.out"))
        if not out_files:
            raise RuntimeError(
                f"No DDG output files found in {raw_dir}. "
                f"Files present: {[f.name for f in raw_dir.iterdir()]}"
            )
        ddg_file = out_files[0]

    predictions = parse_ddg_predictions(ddg_file)
    if not predictions:
        raise RuntimeError(f"No DDG predictions found in {ddg_file}")

    mutations = config.get("mutations", [])

    scores = []
    for pred in predictions:
        scores.append(
            {
                "total_score": pred["ddg"],
                "score_breakdown": {
                    "wt_score": pred["wt_score"],
                    "mut_score": pred["mut_score"],
                },
                "units": "REU",
                "per_residue_scores": None,
                "structure_path": None,
                "ddg": pred["ddg"],
                "mutations": mutations,
            }
        )

    result_data = {"scores": scores}
    (std_dir / "result_data.json").write_text(json.dumps(result_data, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    standardize(args.workspace)
