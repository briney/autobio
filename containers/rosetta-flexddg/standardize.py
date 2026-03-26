#!/usr/bin/env python3
"""Standardize flex-ddG outputs into autobio ScoringOutput format.

Flex-ddG runs an ensemble of backrub trajectories. Each trajectory produces
a score file. The standardizer aggregates DDG values across the ensemble,
reporting per-sample DDG and ensemble statistics (mean, std).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "/opt/rosetta")
from parse_scorefile import extract_scored_structure, parse_score_file


def standardize(workspace: Path) -> None:
    """Parse ensemble score files and write result_data.json with DDG stats."""
    raw_dir = workspace / "outputs" / "raw"
    std_dir = workspace / "outputs" / "standardized"
    config = json.loads((workspace / "config.json").read_text())

    mutations = config.get("mutations", [])
    chains_to_move = config.get("chains_to_move", "")

    # Parse all score files from the ensemble
    score_files = sorted(raw_dir.glob("*.sc"))
    if not score_files:
        raise RuntimeError(
            f"No score files (.sc) found in {raw_dir}. "
            f"Files present: {[f.name for f in raw_dir.iterdir()]}"
        )

    all_rows = []
    for sc_file in score_files:
        all_rows.extend(parse_score_file(sc_file))

    if not all_rows:
        raise RuntimeError("No score entries found in score files.")

    # For flex-ddG, we need to compute DDG from WT and mutant scores.
    # The score file contains both WT and mutant entries, identified by
    # description containing "wt" or "mut".
    wt_scores = []
    mut_scores = []
    for row in all_rows:
        desc = str(row.get("description", "")).lower()
        total = float(row.get("total_score", 0.0))
        if "wt" in desc:
            wt_scores.append(total)
        elif "mut" in desc:
            mut_scores.append(total)

    # If we can't distinguish WT/mut from descriptions, treat all as
    # individual samples and report raw scores
    scores = []
    if wt_scores and mut_scores:
        # Compute per-pair DDG values
        n_pairs = min(len(wt_scores), len(mut_scores))
        ddg_values = [
            mut_scores[i] - wt_scores[i] for i in range(n_pairs)
        ]
        ddg_mean = sum(ddg_values) / len(ddg_values)
        ddg_std = math.sqrt(
            sum((v - ddg_mean) ** 2 for v in ddg_values) / len(ddg_values)
        ) if len(ddg_values) > 1 else 0.0

        # Report ensemble summary as the primary entry
        scores.append(
            {
                "total_score": ddg_mean,
                "score_breakdown": {
                    "ddg_mean": ddg_mean,
                    "ddg_std": ddg_std,
                    "n_samples": n_pairs,
                    "ddg_values": ddg_values,
                    "chains_to_move": chains_to_move,
                },
                "units": "REU",
                "per_residue_scores": None,
                "structure_path": None,
                "ddg": ddg_mean,
                "mutations": mutations,
            }
        )

        # Report individual samples
        for i, ddg_val in enumerate(ddg_values):
            scores.append(
                {
                    "total_score": ddg_val,
                    "score_breakdown": {
                        "wt_score": wt_scores[i],
                        "mut_score": mut_scores[i],
                        "sample_index": i,
                    },
                    "units": "REU",
                    "per_residue_scores": None,
                    "structure_path": None,
                    "ddg": ddg_val,
                    "mutations": mutations,
                }
            )
    else:
        # Fall back: report raw scores without DDG computation
        for row in all_rows:
            scored = extract_scored_structure(row)
            scored["structure_path"] = None
            scored["per_residue_scores"] = None
            scored["ddg"] = None
            scored["mutations"] = mutations
            scores.append(scored)

    result_data = {"scores": scores}
    (std_dir / "result_data.json").write_text(json.dumps(result_data, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    standardize(args.workspace)
