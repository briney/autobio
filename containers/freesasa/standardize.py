#!/usr/bin/env python3
"""Standardize FreeSASA outputs into autobio ScoringOutput format.

run_freesasa.py writes output.json with SASA/BSA values and breakdowns.
This script maps that JSON into the standard scoring schema format.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def standardize(workspace: Path) -> None:
    """Parse FreeSASA output JSON and write result_data.json."""
    raw_dir = workspace / "outputs" / "raw"
    std_dir = workspace / "outputs" / "standardized"

    output_file = raw_dir / "output.json"
    if not output_file.exists():
        raise RuntimeError(
            f"output.json not found in {raw_dir}. "
            f"Files present: {[f.name for f in raw_dir.iterdir() if f.is_file()]}"
        )

    data = json.loads(output_file.read_text())
    mode = data.get("mode", "sasa")

    if mode == "bsa":
        score_breakdown: dict[str, object] = {
            "polar_bsa": data["polar_bsa"],
            "apolar_bsa": data["apolar_bsa"],
            "complex_sasa": data["complex_sasa"],
            "partner1_sasa": data["partner1_sasa"],
            "partner2_sasa": data["partner2_sasa"],
            "partner1_chains": data["partner1_chains"],
            "partner2_chains": data["partner2_chains"],
            "algorithm": data["algorithm"],
            "probe_radius": data["probe_radius"],
            "per_chain_sasa": data.get("per_chain_sasa", {}),
        }

        per_residue_scores = None
        if "per_residue_bsa" in data:
            per_residue_scores = [r["bsa"] for r in data["per_residue_bsa"]]
            score_breakdown["per_residue_detail"] = data["per_residue_bsa"]

        scored = {
            "total_score": data["total_bsa"],
            "units": "angstrom^2",
            "score_breakdown": score_breakdown,
            "per_residue_scores": per_residue_scores,
        }
    else:
        score_breakdown = {
            "polar_sasa": data["polar_sasa"],
            "apolar_sasa": data["apolar_sasa"],
            "algorithm": data["algorithm"],
            "probe_radius": data["probe_radius"],
            "per_chain_sasa": data.get("per_chain_sasa", {}),
        }

        per_residue_scores = None
        if "per_residue_sasa" in data:
            per_residue_scores = [r["total"] for r in data["per_residue_sasa"]]
            score_breakdown["per_residue_detail"] = data["per_residue_sasa"]

        scored = {
            "total_score": data["total_sasa"],
            "units": "angstrom^2",
            "score_breakdown": score_breakdown,
            "per_residue_scores": per_residue_scores,
        }

    result_data = {"scores": [scored]}
    (std_dir / "result_data.json").write_text(json.dumps(result_data, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    standardize(args.workspace)
